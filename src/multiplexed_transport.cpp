// zlink/multiplexed_transport.cpp — Multi-channel TCP transport implementation

#include <zlink/multiplexed_transport.hpp>

#include <cstring>
#include <algorithm>

namespace zlink {

// ── 16KB threshold: above this, memory ops go to bulk channel ────────
static constexpr std::size_t bulk_threshold = 16 * 1024;

// ── Constructor / Destructor ─────────────────────────────────────────

multiplexed_transport::multiplexed_transport(config cfg)
    : cfg_(std::move(cfg))
{
    for (auto& ch : channels_) {
        ch = std::make_unique<tcp_transport>();
    }
}

multiplexed_transport::~multiplexed_transport() {
    close();
}

// ── transport interface ──────────────────────────────────────────────

std::error_code multiplexed_transport::connect(const std::string& host, std::uint16_t port) {
    // Try multi-port first
    auto ec = connect_all(host, port);
    if (!ec) {
        multi_port_mode_.store(true);
        connected_.store(true);
        return {};
    }

    // Fallback to single connection
    if (cfg_.use_single_fallback) {
        ec = connect_single(host, port);
        if (!ec) {
            multi_port_mode_.store(false);
            connected_.store(true);
            return {};
        }
        return ec;
    }

    return ec;
}

std::error_code multiplexed_transport::listen(const std::string& bind_addr, std::uint16_t port) {
    auto ec = listen_all(bind_addr, port);
    if (!ec) {
        multi_port_mode_.store(true);
        return {};
    }

    if (cfg_.use_single_fallback) {
        ec = listen_single(bind_addr, port);
        if (!ec) {
            multi_port_mode_.store(false);
            return {};
        }
        return ec;
    }

    return ec;
}

std::error_code multiplexed_transport::accept() {
    if (multi_port_mode_.load()) {
        auto ec = accept_all();
        connected_.store(!ec);
        return ec;
    }
    auto ec = accept_single();
    connected_.store(!ec);
    return ec;
}

std::error_code multiplexed_transport::send(const frame& f) {
    auto ch = route_frame(f);
    return send_on(ch, f);
}

std::error_code multiplexed_transport::receive(frame& out) {
    // Default receive: try RPC control channel first
    if (multi_port_mode_.load()) {
        return receive_on(channel_id::rpc_control, out);
    }
    // Single-connection mode: just receive from channel 0
    return channels_[0]->receive(out);
}

void multiplexed_transport::close() noexcept {
    connected_.store(false);
    for (auto& ch : channels_) {
        if (ch) ch->close();
    }
}

bool multiplexed_transport::is_connected() const noexcept {
    if (!connected_.load()) return false;
    if (multi_port_mode_.load()) {
        // All channels must be connected
        for (auto& ch : channels_) {
            if (!ch || !ch->is_connected()) return false;
        }
        return true;
    }
    return channels_[0] && channels_[0]->is_connected();
}

// ── Channel-aware send/receive ───────────────────────────────────────

std::error_code multiplexed_transport::send_on(channel_id ch, const frame& f) {
    auto idx = static_cast<std::size_t>(ch);

    if (!multi_port_mode_.load()) {
        // Single-connection mode: everything goes on channel 0
        return channels_[0]->send(f);
    }

    if (idx >= channels_.size() || !channels_[idx]) {
        return std::make_error_code(std::errc::not_connected);
    }

    return channels_[idx]->send(f);
}

std::error_code multiplexed_transport::receive_on(channel_id ch, frame& out) {
    auto idx = static_cast<std::size_t>(ch);

    if (!multi_port_mode_.load()) {
        return channels_[0]->receive(out);
    }

    if (idx >= channels_.size() || !channels_[idx]) {
        return std::make_error_code(std::errc::not_connected);
    }

    return channels_[idx]->receive(out);
}

// ── Channel routing logic ────────────────────────────────────────────

multiplexed_transport::channel_id
multiplexed_transport::route_frame(const frame& f) const {
    switch (f.type) {
        // RPC control channel: latency-sensitive, small frames
        case frame_type::request:
        case frame_type::response:
        case frame_type::error:
        case frame_type::heartbeat:
            return channel_id::rpc_control;

        // Bulk data channel: throughput-sensitive, large frames
        case frame_type::pipeline_request:
        case frame_type::pipeline_response:
        case frame_type::pipeline_mem:
            return channel_id::bulk_data;

        // Memory ops: route by size
        case frame_type::memory_op:
        case frame_type::memory_reply:
            if (f.payload.size() >= bulk_threshold) {
                return channel_id::bulk_data;
            }
            return channel_id::rpc_control;

        // Write-behind ACKs: small, go on RPC channel
        case frame_type::write_ack:
            return channel_id::rpc_control;

        // Prefetch: background channel
        case frame_type::prefetch_request:
        case frame_type::prefetch_response:
            return channel_id::prefetch;

        // Session management: RPC channel
        case frame_type::session_resume:
        case frame_type::session_resume_ack:
            return channel_id::rpc_control;

        default:
            return channel_id::rpc_control;
    }
}

// ── Access individual channels ───────────────────────────────────────

tcp_transport& multiplexed_transport::channel(channel_id ch) {
    return *channels_[static_cast<std::size_t>(ch)];
}

const tcp_transport& multiplexed_transport::channel(channel_id ch) const {
    return *channels_[static_cast<std::size_t>(ch)];
}

// ── Internal: multi-port connection management ───────────────────────

std::error_code multiplexed_transport::connect_all(
    const std::string& host, std::uint16_t base_port)
{
    // Connect to base_port (RPC), base_port+1 (bulk), base_port+2 (prefetch)
    for (std::size_t i = 0; i < 3; i++) {
        auto port = static_cast<std::uint16_t>(base_port + i);
        auto ec = channels_[i]->connect(host, port);
        if (ec) {
            // Clean up already-connected channels
            for (std::size_t j = 0; j < i; j++) {
                channels_[j]->close();
            }
            return ec;
        }
    }
    return {};
}

std::error_code multiplexed_transport::listen_all(
    const std::string& bind_addr, std::uint16_t base_port)
{
    for (std::size_t i = 0; i < 3; i++) {
        auto port = static_cast<std::uint16_t>(base_port + i);
        auto ec = channels_[i]->listen(bind_addr, port);
        if (ec) {
            for (std::size_t j = 0; j < i; j++) {
                channels_[j]->close();
            }
            return ec;
        }
    }
    return {};
}

std::error_code multiplexed_transport::accept_all() {
    for (std::size_t i = 0; i < 3; i++) {
        auto ec = channels_[i]->accept();
        if (ec) {
            return ec;
        }
    }
    return {};
}

// ── Internal: single-connection fallback ─────────────────────────────

std::error_code multiplexed_transport::connect_single(
    const std::string& host, std::uint16_t port)
{
    return channels_[0]->connect(host, port);
}

std::error_code multiplexed_transport::listen_single(
    const std::string& bind_addr, std::uint16_t port)
{
    return channels_[0]->listen(bind_addr, port);
}

std::error_code multiplexed_transport::accept_single() {
    return channels_[0]->accept();
}

} // namespace zlink
