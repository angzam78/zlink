#pragma once
// zlink/multiplexed_transport.hpp — Multi-channel TCP transport
//
// Replaces single tcp_transport with 3 parallel channels:
//   Channel 0: RPC Control  — small frames, TCP_NODELAY, latency-sensitive
//   Channel 1: Bulk Data    — large frames, TCP_CORK, throughput-sensitive
//   Channel 2: Prefetch     — background ops, best-effort
//
// Eliminates head-of-line blocking between RPC control and bulk data.
// A 64MB cuMemcpyHtoD on the bulk channel does NOT delay a
// cuCtxSynchronize on the RPC channel.
//
// DESIGN GOALS:
//   1. Drop-in replacement for tcp_transport (implements transport interface)
//   2. Automatic frame routing based on frame type + size
//   3. Fallback to single-connection mode if multi-port unavailable
//   4. Thread-safe: multiple threads can send on different channels concurrently

#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/config.hpp>

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <string>
#include <system_error>

namespace zlink {

class multiplexed_transport final : public transport {
public:
    // ── Channel identifiers ──────────────────────────────────────
    enum class channel_id : std::uint8_t {
        rpc_control = 0,   // Small frames, TCP_NODELAY
        bulk_data   = 1,   // Large frames, TCP_CORK
        prefetch    = 2,   // Background, best-effort
        count       = 3,
    };

    // ── Configuration ────────────────────────────────────────────
    struct config {
        std::uint16_t base_port           = default_port;
        bool          use_single_fallback = true;   // Fallback to 1 conn if multi-port fails
        bool          tcp_nodelay_bulk    = false;  // TCP_NODELAY on bulk channel? (usually off)
        bool          tcp_cork_bulk       = true;   // TCP_CORK on bulk channel? (batch small writes)
        int           listen_backlog      = 64;
    };

    multiplexed_transport();
    explicit multiplexed_transport(const config& cfg);
    ~multiplexed_transport() override;

    // ── transport interface ──────────────────────────────────────
    std::error_code connect(const std::string& host, std::uint16_t port) override;
    std::error_code listen(const std::string& bind_addr, std::uint16_t port) override;
    std::error_code accept() override;

    // Automatic routing: inspects frame type and routes to channel
    //   request/response/error → rpc_control
    //   pipeline_request/pipeline_response → bulk_data
    //   memory_op/memory_reply → bulk_data if payload > 16KB, else rpc_control
    //   heartbeat → rpc_control
    std::error_code send(const frame& f) override;

    // Receive from the RPC control channel (default)
    // For receiving on specific channels, use receive_on()
    std::error_code receive(frame& out) override;

    void           close() noexcept override;
    bool           is_connected() const noexcept override;

    // ── Channel-aware send/receive ───────────────────────────────
    std::error_code send_on(channel_id ch, const frame& f);
    std::error_code receive_on(channel_id ch, frame& out);

    // ── Channel routing logic ────────────────────────────────────
    channel_id route_frame(const frame& f) const;

    // ── Access individual channels ───────────────────────────────
    tcp_transport& channel(channel_id ch);
    const tcp_transport& channel(channel_id ch) const;

    // ── Mode query ───────────────────────────────────────────────
    bool is_multi_port() const noexcept { return multi_port_mode_.load(); }

private:
    config cfg_;
    std::array<std::unique_ptr<tcp_transport>, 3> channels_;
    std::atomic<bool> multi_port_mode_{false};
    std::atomic<bool> connected_{false};

    // For default receive(): mux between channels
    mutable std::mutex receive_mutex_;
    std::atomic<int> next_receive_channel_{0};

    // Internal: connect/listen/accept all 3 channels
    std::error_code connect_all(const std::string& host, std::uint16_t base_port);
    std::error_code listen_all(const std::string& bind_addr, std::uint16_t base_port);
    std::error_code accept_all();

    // Internal: fallback to single connection
    std::error_code connect_single(const std::string& host, std::uint16_t port);
    std::error_code listen_single(const std::string& bind_addr, std::uint16_t port);
    std::error_code accept_single();
};

} // namespace zlink
