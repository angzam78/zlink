#pragma once
// zlink/write_behind_buffer.hpp — Asynchronous write-behind for host→server data
//
// THE THESIS INSIGHT:
//   Writes don't need to be confirmed immediately. cuMemcpyHtoD can return
//   as soon as data is buffered locally; the actual network transfer happens
//   asynchronously on the bulk channel.
//
//   Thesis data: write-behind async push was the second biggest win
//   after prefetch. At 10ms RTT, a 64MB cuMemcpyHtoD that currently
//   blocks for ~130ms returns in <1ms with write-behind.
//
// FENCE MECHANISM:
//   Every write gets a monotonically increasing fence ID.
//   Barriers call wait_fence() or drain() to ensure all prior writes
//   are confirmed by the server before proceeding.
//
//   This ensures correctness:
//     cuMemcpyHtoD → buffer_write() → returns fence F1 immediately
//     cuLaunchKernel → enqueued (no wait)
//     cuCtxSynchronize → barrier → drain() → all writes confirmed
//
// WIRE PROTOCOL:
//   write_behind frame (client → server, on bulk channel):
//     [8B fence_id][mem_request][data...]
//
//   write_ack frame (server → client, on RPC channel):
//     [8B fence_id]

#include <zlink/config.hpp>
#include <zlink/transport.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <vector>

namespace zlink {

// Forward declaration — avoid including multiplexed_transport.hpp
class multiplexed_transport;

// ── Write-behind fence ───────────────────────────────────────────────
struct write_fence {
    std::uint64_t                          id;
    std::shared_ptr<std::atomic<bool>>     completed;

    bool is_completed() const noexcept {
        return completed && completed->load(std::memory_order_acquire);
    }

    void wait() const {
        while (!is_completed()) {
            std::this_thread::yield();
        }
    }
};

// ── Write-behind configuration ───────────────────────────────────────
struct write_behind_config {
    std::size_t            max_dirty_bytes  = 256 * 1024 * 1024; // 256 MB
    std::chrono::milliseconds flush_interval{50};               // Push every 50ms
    std::size_t            max_pending_ops  = 1024;
    bool                   verbose          = false;
};

// ── Write-behind buffer ──────────────────────────────────────────────
class write_behind_buffer {
public:
    explicit write_behind_buffer(transport& bulk_transport);
    explicit write_behind_buffer(transport& bulk_transport, const write_behind_config& cfg);
    ~write_behind_buffer();

    // ── Buffer a write (non-blocking) ────────────────────────────
    // Returns immediately. Data will be pushed asynchronously.
    // Returns a fence that can be waited on for synchronization.
    write_fence buffer_write(std::uintptr_t remote_addr,
                             std::span<const std::byte> data);

    // ── Synchronization ──────────────────────────────────────────
    // Wait until all writes up to the given fence are confirmed by server
    void wait_fence(const write_fence& f);

    // Wait until ALL pending writes are confirmed
    void drain();

    // ── Process a write_ack from the server ──────────────────────
    // Called by the RPC channel receive loop when it gets a write_ack frame
    void handle_ack(std::uint64_t fence_id);

    // ── Lifecycle ────────────────────────────────────────────────
    void start();
    void stop();

    // ── Query ────────────────────────────────────────────────────
    std::size_t pending_bytes() const;
    std::size_t pending_ops()   const;
    bool        is_running()    const { return running_.load(); }

private:
    struct pending_write {
        std::uintptr_t                      remote_addr;
        std::vector<std::byte>              data;
        std::uint64_t                       fence_id;
        std::shared_ptr<std::atomic<bool>>  completed;
    };

    void pusher_loop();
    void send_write(const pending_write& w);
    void mark_completed(std::uint64_t fence_id);

    transport&            transport_;
    write_behind_config   cfg_;

    std::atomic<bool>     running_{false};
    std::thread           pusher_thread_;

    // Queue of pending writes (ordered by fence_id)
    mutable std::mutex    queue_mutex_;
    std::deque<pending_write> queue_;
    std::condition_variable queue_cv_;

    // Fence tracking
    std::atomic<std::uint64_t> next_fence_id_{1};
    std::atomic<std::uint64_t> last_completed_fence_{0};

    // Bytes tracking
    mutable std::mutex    bytes_mutex_;
    std::atomic<std::size_t> pending_bytes_{0};
};

} // namespace zlink
