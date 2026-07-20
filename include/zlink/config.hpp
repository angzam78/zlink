#pragma once
// zlink/config.hpp — Build-time configuration and feature detection

#include <cstdint>
#include <cstddef>

namespace zlink {

// ── Protocol constants ─────────────────────────────────────────────────
inline constexpr std::uint16_t default_port = 14833;  // Same as Lupine for easy migration
inline constexpr std::size_t   max_frame_size = 64 * 1024 * 1024; // 64 MiB
inline constexpr std::size_t   frame_header_size = 9;   // 4 len + 4 call_id + 1 type
inline constexpr std::size_t   initial_buffer_size = 4096;
inline constexpr int           max_concurrent_calls = 256;

// ── Frame types ────────────────────────────────────────────────────────
enum class frame_type : std::uint8_t {
    request            = 0x01,
    response           = 0x02,
    error              = 0x03,
    pipeline_request   = 0x04,  // Batched RPC: func,func,func → one frame
    pipeline_response  = 0x05,  // Batched response: reply,reply,reply → one frame
    pipeline_mem       = 0x06,  // Pipeline with inline memory ops:
                                   // [sync_data...][rpc_calls...][read_reqs...]
                                   // → [sync_acks][rpc_replies][read_data...]
    memory_op          = 0x10,  // r3map-inspired remote memory ops
    memory_reply       = 0x11,
    heartbeat          = 0xFF,
};

// ── Error codes ────────────────────────────────────────────────────────
enum class error_code : std::int32_t {
    ok                = 0,
    unknown_function  = -1,
    serialization     = -2,
    transport         = -3,
    call_id_mismatch  = -4,
    server_error      = -5,
    pointer_not_found = -6,
    timeout           = -7,
    shutdown          = -8,
};

inline constexpr bool ok(error_code e) noexcept { return e == error_code::ok; }
inline constexpr bool failed(error_code e) noexcept { return e != error_code::ok; }

// ── Feature detection ──────────────────────────────────────────────────
#if __linux__
    #define ZLINK_LINUX 1
    #define ZLINK_HAS_USERFAULTFD 1
#elif _WIN32
    #define ZLINK_WINDOWS 1
    #define ZLINK_HAS_USERFAULTFD 0
#elif __APPLE__
    #define ZLINK_MACOS 1
    #define ZLINK_HAS_USERFAULTFD 0
#endif

} // namespace zlink
