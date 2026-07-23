#pragma once
// zlink/tcp_transport.hpp — TCP transport with length-prefixed framing

#include <zlink/transport.hpp>
#include <string>
#include <array>
#include <mutex>

namespace zlink {

class tcp_transport final : public transport {
public:
    tcp_transport();
    ~tcp_transport() override;

    // ── transport interface ────────────────────────────────────────────
    std::error_code connect(const std::string& host, std::uint16_t port) override;
    std::error_code listen(const std::string& bind_addr, std::uint16_t port) override;
    std::error_code accept() override;
    std::error_code send(const frame& f) override;
    std::error_code receive(frame& out) override;
    void           close() noexcept override;
    bool           is_connected() const noexcept override;

    // Expose socket fd for poll()-based multiplexing
    int native_handle() const noexcept { return sock_fd_; }

private:
    // Platform-specific socket handle
    int sock_fd_ = -1;
    int listen_fd_ = -1;
    bool connected_ = false;
    std::mutex send_mutex_;  // Protect concurrent sends

    // Read exactly N bytes from socket
    std::error_code read_exact(std::span<std::byte> buf);
    // Write exactly N bytes to socket
    std::error_code write_exact(std::span<const std::byte> buf);
};

} // namespace zlink
