// zlink/connection_manager.cpp — Connection lifecycle implementation

#include <zlink/connection_manager.hpp>

#include <algorithm>

namespace zlink {

connection_manager::connection_manager(transport& tp, connection_config cfg)
    : transport_(tp)
    , cfg_(std::move(cfg))
{}

connection_manager::~connection_manager() {
    stop_heartbeat();
}

// ── Lifecycle ────────────────────────────────────────────────────────

std::error_code connection_manager::connect(const std::string& host, std::uint16_t port) {
    host_ = host;
    port_ = port;

    auto ec = transport_.connect(host, port);
    if (ec) return ec;

    // Start heartbeat
    start_heartbeat();
    return {};
}

void connection_manager::start_heartbeat() {
    if (heartbeat_running_.load()) return;
    heartbeat_running_.store(true);
    heartbeat_thread_ = std::thread(&connection_manager::heartbeat_loop, this);
}

void connection_manager::stop_heartbeat() {
    heartbeat_running_.store(false);
    if (heartbeat_thread_.joinable()) {
        heartbeat_thread_.join();
    }
}

// ── Session recovery ─────────────────────────────────────────────────

void connection_manager::set_recovery_handler(recovery_fn fn) {
    recovery_fn_ = std::move(fn);
}

// ── Query ────────────────────────────────────────────────────────────

bool connection_manager::is_connected() const {
    return transport_.is_connected();
}

// ── Heartbeat loop ───────────────────────────────────────────────────

void connection_manager::heartbeat_loop() {
    while (heartbeat_running_.load()) {
        std::this_thread::sleep_for(cfg_.heartbeat_interval);

        if (!heartbeat_running_.load()) break;

        if (!transport_.is_connected()) {
            handle_disconnect();
            continue;
        }

        // Send heartbeat
        frame hb;
        hb.call_id = 0;
        hb.type = frame_type::heartbeat;
        auto ec = transport_.send(hb);
        if (ec) {
            handle_disconnect();
            continue;
        }

        // Wait for heartbeat response
        frame resp;
        ec = transport_.receive(resp);
        if (ec || resp.type != frame_type::heartbeat) {
            handle_disconnect();
        }
    }
}

// ── Disconnect handling ──────────────────────────────────────────────

void connection_manager::handle_disconnect() {
    if (!heartbeat_running_.load()) return;

    // Attempt reconnection with exponential backoff
    for (int attempt = 0; attempt < cfg_.max_reconnect_attempts; attempt++) {
        if (!heartbeat_running_.load()) return;

        auto delay = std::min(
            cfg_.reconnect_base * (1 << attempt),
            cfg_.reconnect_max);
        std::this_thread::sleep_for(delay);

        auto ec = attempt_reconnect();
        if (!ec) {
            reconnects_.fetch_add(1);

            // Run recovery handler (re-register virtual handles, etc.)
            if (recovery_fn_) {
                recovery_fn_();
            }
            return;
        }
    }

    // All reconnection attempts failed
    // In production, you'd want to notify the application layer
    heartbeat_running_.store(false);
}

std::error_code connection_manager::attempt_reconnect() {
    std::lock_guard lock(connect_mutex_);
    return transport_.connect(host_, port_);
}

} // namespace zlink
