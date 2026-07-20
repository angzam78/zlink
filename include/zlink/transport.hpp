#pragma once
// zlink/transport.hpp — Abstract transport interface for RPC frames

#include <zlink/config.hpp>
#include <cstdint>
#include <span>
#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <system_error>

namespace zlink {

// ── Frame: wire-level unit of communication ────────────────────────────
struct frame {
    std::uint32_t call_id = 0;
    frame_type    type    = frame_type::request;
    std::vector<std::byte> payload;

    // Serialize to wire format: [4 len][4 call_id][1 type][payload...]
    std::vector<std::byte> serialize() const;

    // Deserialize from wire format. Returns bytes consumed, or 0 if incomplete.
    static std::size_t deserialize(std::span<const std::byte> data, frame& out);

    // Total wire size
    std::size_t wire_size() const noexcept {
        return frame_header_size + payload.size();
    }
};

// ── Transport: abstract interface for sending/receiving frames ─────────
class transport {
public:
    using receive_handler = std::function<void(frame&&)>;

    virtual ~transport() = default;

    // Connect to a remote endpoint (client side)
    virtual std::error_code connect(const std::string& host, std::uint16_t port) = 0;

    // Start listening for connections (server side)
    virtual std::error_code listen(const std::string& bind_addr, std::uint16_t port) = 0;

    // Accept one connection (server side, blocking)
    virtual std::error_code accept() = 0;

    // Send a frame
    virtual std::error_code send(const frame& f) = 0;

    // Receive a frame (blocking)
    virtual std::error_code receive(frame& out) = 0;

    // Close the connection
    virtual void close() noexcept = 0;

    // Is connected?
    virtual bool is_connected() const noexcept = 0;

    // Set callback for async receive mode
    virtual void set_receive_handler(receive_handler h) { handler_ = std::move(h); }

protected:
    receive_handler handler_;
};

// ── Transport factory ──────────────────────────────────────────────────
enum class transport_kind { tcp, unix_socket };

std::unique_ptr<transport> make_transport(transport_kind kind = transport_kind::tcp);

} // namespace zlink
