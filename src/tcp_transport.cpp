// zlink/tcp_transport.cpp — TCP transport implementation
//
// Implements length-prefixed framing over TCP:
//   [4 bytes: total_length][4 bytes: call_id][1 byte: frame_type][N bytes: payload]
//
// Supports both blocking and async modes.

#include <zlink/tcp_transport.hpp>
#include <zlink/config.hpp>

#include <cstring>
#include <system_error>
#include <stdexcept>
#include <algorithm>

#ifdef ZLINK_LINUX
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <errno.h>
#elif ZLINK_WINDOWS
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
#endif

namespace zlink {

// ── Frame serialization ────────────────────────────────────────────────
std::vector<std::byte> frame::serialize() const {
    std::vector<std::byte> buf(wire_size());
    auto* p = buf.data();

    // Length (4 bytes, network byte order)
    std::uint32_t len = static_cast<std::uint32_t>(payload.size() + 5); // +5 for call_id + type
    p[0] = static_cast<std::byte>((len >> 24) & 0xFF);
    p[1] = static_cast<std::byte>((len >> 16) & 0xFF);
    p[2] = static_cast<std::byte>((len >> 8) & 0xFF);
    p[3] = static_cast<std::byte>(len & 0xFF);
    p += 4;

    // Call ID (4 bytes)
    p[0] = static_cast<std::byte>((call_id >> 24) & 0xFF);
    p[1] = static_cast<std::byte>((call_id >> 16) & 0xFF);
    p[2] = static_cast<std::byte>((call_id >> 8) & 0xFF);
    p[3] = static_cast<std::byte>(call_id & 0xFF);
    p += 4;

    // Frame type (1 byte)
    p[0] = static_cast<std::byte>(type);
    p += 1;

    // Payload
    if (!payload.empty()) {
        std::memcpy(p, payload.data(), payload.size());
    }

    return buf;
}

std::size_t frame::deserialize(std::span<const std::byte> data, frame& out) {
    if (data.size() < frame_header_size) {
        return 0; // Not enough data for header
    }

    // Read length
    std::uint32_t len = (static_cast<std::uint32_t>(data[0]) << 24) |
                        (static_cast<std::uint32_t>(data[1]) << 16) |
                        (static_cast<std::uint32_t>(data[2]) << 8)  |
                        static_cast<std::uint32_t>(data[3]);

    std::size_t total_size = 4 + len; // 4 for the length field itself
    if (data.size() < total_size) {
        return 0; // Not enough data for full frame
    }

    // Read call ID
    out.call_id = (static_cast<std::uint32_t>(data[4]) << 24) |
                  (static_cast<std::uint32_t>(data[5]) << 16) |
                  (static_cast<std::uint32_t>(data[6]) << 8)  |
                  static_cast<std::uint32_t>(data[7]);

    // Read frame type
    out.type = static_cast<frame_type>(data[8]);

    // Read payload
    std::size_t payload_len = len - 5; // Subtract call_id + type
    out.payload.assign(data.begin() + frame_header_size,
                       data.begin() + frame_header_size + payload_len);

    return total_size;
}

// ── TCP transport ──────────────────────────────────────────────────────
tcp_transport::tcp_transport() = default;

tcp_transport::~tcp_transport() {
    close();
}

std::error_code tcp_transport::connect(const std::string& host, std::uint16_t port) {
#ifdef ZLINK_WINDOWS
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        return std::make_error_code(std::errc::connection_refused);
    }
#endif

    sock_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd_ < 0) {
        return std::make_error_code(std::errc::invalid_argument);
    }

    // Disable Nagle's algorithm for low latency
    int flag = 1;
    ::setsockopt(sock_fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        close();
        return std::make_error_code(std::errc::invalid_argument);
    }

    if (::connect(sock_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        close();
        return std::make_error_code(std::errc::connection_refused);
    }

    connected_ = true;
    return {};
}

std::error_code tcp_transport::listen(const std::string& bind_addr, std::uint16_t port) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        return std::make_error_code(std::errc::invalid_argument);
    }

    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (bind_addr.empty() || bind_addr == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        ::inet_pton(AF_INET, bind_addr.c_str(), &addr.sin_addr);
    }

    if (::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        return std::make_error_code(std::errc::address_in_use);
    }

    if (::listen(listen_fd_, 64) < 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        return std::make_error_code(std::errc::invalid_argument);
    }

    return {};
}

std::error_code tcp_transport::accept() {
    if (listen_fd_ < 0) {
        return std::make_error_code(std::errc::not_connected);
    }

    struct sockaddr_in client_addr{};
    socklen_t len = sizeof(client_addr);
    sock_fd_ = ::accept(listen_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &len);

    if (sock_fd_ < 0) {
        return std::make_error_code(std::errc::connection_refused);
    }

    // Disable Nagle for low latency
    int flag = 1;
    ::setsockopt(sock_fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    connected_ = true;
    return {};
}

std::error_code tcp_transport::send(const frame& f) {
    std::lock_guard lock(send_mutex_);

    auto wire = f.serialize();
    return write_exact(wire);
}

std::error_code tcp_transport::receive(frame& out) {
    // Read the 4-byte length prefix first
    std::array<std::byte, 4> len_buf{};
    if (auto ec = read_exact(len_buf); ec) return ec;

    std::uint32_t len = (static_cast<std::uint32_t>(len_buf[0]) << 24) |
                        (static_cast<std::uint32_t>(len_buf[1]) << 16) |
                        (static_cast<std::uint32_t>(len_buf[2]) << 8)  |
                        static_cast<std::uint32_t>(len_buf[3]);

    if (len < 5 || len > max_frame_size) {
        return std::make_error_code(std::errc::invalid_argument);
    }

    // Read the rest of the frame (call_id + type + payload)
    std::vector<std::byte> rest(len);
    if (auto ec = read_exact(rest); ec) return ec;

    // Reconstruct the full frame data for parsing
    std::vector<std::byte> full(len_buf.begin(), len_buf.end());
    full.insert(full.end(), rest.begin(), rest.end());

    auto consumed = frame::deserialize(full, out);
    if (consumed == 0) {
        return std::make_error_code(std::errc::invalid_argument);
    }

    return {};
}

void tcp_transport::close() noexcept {
    connected_ = false;
    if (sock_fd_ >= 0) {
        ::close(sock_fd_);
        sock_fd_ = -1;
    }
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
}

bool tcp_transport::is_connected() const noexcept {
    return connected_ && sock_fd_ >= 0;
}

std::error_code tcp_transport::read_exact(std::span<std::byte> buf) {
    std::size_t total = buf.size();
    std::size_t done = 0;

    while (done < total) {
        auto n = ::recv(sock_fd_, reinterpret_cast<char*>(buf.data() + done),
                        static_cast<int>(total - done), 0);
        if (n <= 0) {
            connected_ = false;
            return std::make_error_code(std::errc::connection_reset);
        }
        done += static_cast<std::size_t>(n);
    }
    return {};
}

std::error_code tcp_transport::write_exact(std::span<const std::byte> buf) {
    std::size_t total = buf.size();
    std::size_t done = 0;

    while (done < total) {
        auto n = ::send(sock_fd_, reinterpret_cast<const char*>(buf.data() + done),
                        static_cast<int>(total - done), 0);
        if (n <= 0) {
            connected_ = false;
            return std::make_error_code(std::errc::connection_reset);
        }
        done += static_cast<std::size_t>(n);
    }
    return {};
}

// ── Transport factory ──────────────────────────────────────────────────
std::unique_ptr<transport> make_transport(transport_kind kind) {
    switch (kind) {
        case transport_kind::tcp:
            return std::make_unique<tcp_transport>();
        default:
            return std::make_unique<tcp_transport>();
    }
}

} // namespace zlink
