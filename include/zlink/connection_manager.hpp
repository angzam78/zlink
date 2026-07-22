#pragma once
// zlink/connection_manager.hpp — Connection lifecycle management
//
// Current zlink has zero reconnection logic. If the TCP socket drops,
// the entire session dies. This manager adds:
//
//   1. Automatic reconnection with exponential backoff
//   2. Heartbeat health checks (using frame_type::heartbeat)
//   3. Session recovery — re-register virtual handles after reconnect

#include <zlink/transport.hpp>
#include <zlink/virtual_handle.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>

namespace zlink {

class multiplexed_transport;

// ── Connection manager configuration ─────────────────────────────────
struct connection_config {
    std::chrono::milliseconds heartbeat_interval  {5000};   // Health check interval
    std::chrono::milliseconds connect_timeout      {10000};  // Initial connect timeout
    std::chrono::milliseconds reconnect_base       {1000};   // First reconnect delay
    std::chrono::milliseconds reconnect_max        {30000};  // Max reconnect delay
    int                       max_reconnect_attempts = 10;
};

// ── Connection manager ───────────────────────────────────────────────
class connection_manager {
public:
    using recovery_fn = std::function<std::error_code()>;

    explicit connection_manager(transport& tp);
    explicit connection_manager(transport& tp, const connection_config& cfg);
    ~connection_manager();

    // ── Lifecycle ────────────────────────────────────────────────
    std::error_code connect(const std::string& host, std::uint16_t port);
    void start_heartbeat();
    void stop_heartbeat();

    // ── Session recovery ─────────────────────────────────────────
    // After reconnection, replay virtual handle registrations
    // The recovery function is called after each successful reconnect
    void set_recovery_handler(recovery_fn fn);

    // ── Query ────────────────────────────────────────────────────
    bool          is_connected()    const;
    std::size_t   reconnect_count() const { return reconnects_.load(); }
    std::string   remote_host()     const { return host_; }
    std::uint16_t remote_port()     const { return port_; }

private:
    void heartbeat_loop();
    void handle_disconnect();
    std::error_code attempt_reconnect();

    transport&         transport_;
    connection_config  cfg_;

    std::atomic<bool>  heartbeat_running_{false};
    std::thread        heartbeat_thread_;

    recovery_fn        recovery_fn_;
    std::string        host_;
    std::uint16_t      port_ = 0;

    std::atomic<std::size_t> reconnects_{0};
    std::mutex         connect_mutex_;
};

} // namespace zlink
