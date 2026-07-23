#pragma once
// zlink/rpc.hpp — RPC engine wrapping zpp_bits for network RPC
//
// rpc_client_base adapts zpp_bits's in-memory RPC to work over a
// network transport. It provides:
//   - Synchronous send/receive of serialized RPC payloads
//   - Pointer translation via ptr_map
//   - Direct transport access for memory_op frames

#include <zlink/config.hpp>
#include <zlink/transport.hpp>
#include <zlink/ptr_map.hpp>
#include <zlink/memory.hpp>

// zpp_bits — header-only C++20 serialization + RPC
#include <zpp_bits.h>

#include <cstdint>
#include <span>
#include <vector>
#include <mutex>
#include <atomic>
#include <system_error>

namespace zlink {

// ── RPC Client ─────────────────────────────────────────────────────────
// Sends RPC requests over a transport and receives responses.

class rpc_client_base {
public:
    explicit rpc_client_base(transport& tp)
        : transport_(tp) {}

    virtual ~rpc_client_base() = default;

    // Send raw request bytes and receive raw response bytes.
    std::error_code send_request(std::span<const std::byte> request_data,
                                 std::vector<std::byte>& response_data);

    // Access the pointer map for handle translation
    ptr_map& pointers() noexcept { return ptrs_; }
    const ptr_map& pointers() const noexcept { return ptrs_; }

    // Access the underlying transport (for memory_op frames)
    transport& get_transport() noexcept { return transport_; }

protected:
    transport&              transport_;
    ptr_map                 ptrs_;
    std::mutex              send_mutex_;
    std::atomic<std::uint32_t> next_call_id_{1};
};

} // namespace zlink
