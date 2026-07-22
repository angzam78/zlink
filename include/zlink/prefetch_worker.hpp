#pragma once
// zlink/prefetch_worker.hpp — Background page prefetch with access pattern detection
//
// THE KEY INSIGHT FROM THE r3map THESIS:
//   The single biggest performance win is speculative page prefetching.
//   When the workload accesses GPU memory in a predictable pattern
//   (sequential, strided), pages can be fetched BEFORE the access fault,
//   eliminating the RTT penalty entirely.
//
//   Thesis data: managed mounts with pull workers sustained 500MB/s
//   at 25ms RTT, where direct mounts collapsed below 1MB/s.
//
// ARCHITECTURE:
//   1. Access pattern detector: records every memory access,
//      classifies pattern as sequential/strided/random
//   2. Priority queue: prefetch pages predicted by pattern detector
//   3. Background pull thread: fetches pages from remote via chunk_cache
//   4. Cancel mechanism: stop prefetch for pages about to be written

#include <zlink/chunk_cache.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>

namespace zlink {

// ── Access pattern classification ────────────────────────────────────
enum class access_pattern {
    sequential,   // Monotonically increasing page indices
    strided,      // Fixed stride between accesses
    random,       // No discernible pattern
};

// ── Prefetch worker configuration ────────────────────────────────────
struct prefetch_config {
    std::size_t             max_outstanding  = 32;     // Max prefetch in flight
    std::size_t             lookahead        = 8;      // Pages ahead to prefetch
    std::chrono::milliseconds idle_timeout   {1000};   // Pause when no hints
    std::size_t             min_confidence   = 3;      // Min consecutive matches before prefetch
    bool                    verbose          = false;
};

// ── Prefetch worker ──────────────────────────────────────────────────
//
// Integrates with chunk_cache: the prefetch worker calls
// chunk_cache::fetch_chunk() for pages not yet local. After the
// first fetch, pages stay local until explicitly invalidated,
// so subsequent accesses to the same memory region hit cache with
// ZERO network traffic.
//
// Integration with managed_pipeline:
//   - managed_pipeline calls record_access() on every memory op
//   - On barrier calls, prefetch is paused to avoid interference
//   - On write ops, prefetch for those pages is cancelled

class prefetch_worker {
public:
    prefetch_worker(chunk_cache& cache);
    prefetch_worker(chunk_cache& cache, const prefetch_config& cfg);
    ~prefetch_worker();

    // ── Lifecycle ────────────────────────────────────────────────
    void start();
    void stop();

    // ── Access pattern hints ─────────────────────────────────────
    // Called by managed_pipeline on each memory access.
    // The detector updates its internal state and may trigger
    // prefetch predictions for subsequent pages.
    void record_access(std::uintptr_t addr, std::size_t size, bool is_write);

    // ── Cancel pending prefetch for pages about to be written ────
    // Called before write-behind buffer processes dirty pages.
    void cancel_prefetch(std::span<const std::int64_t> page_offsets);

    // ── Pause/resume (called during barrier ops) ────────────────
    void pause();
    void resume();

    // ── Query ────────────────────────────────────────────────────
    std::size_t prefetch_hits()   const { return hits_.load(); }
    std::size_t prefetch_misses() const { return misses_.load(); }
    float       hit_rate() const {
        auto total = hits_.load() + misses_.load();
        return total > 0 ? static_cast<float>(hits_.load()) / total : 0.0f;
    }
    access_pattern current_pattern() const;

private:
    // ── Internal pattern state ───────────────────────────────────
    struct pattern_state {
        access_pattern type      = access_pattern::random;
        std::int64_t   stride    = 0;         // For strided: bytes between accesses
        std::int64_t   last_page = -1;        // Last accessed page index
        std::size_t    confidence = 0;         // Consecutive matches
    };

    // ── Worker thread ────────────────────────────────────────────
    void worker_loop();
    void update_pattern(std::int64_t page_index, bool is_write);
    std::vector<std::int64_t> predict_next_pages(std::int64_t current_page);

    // ── State ────────────────────────────────────────────────────
    chunk_cache&         cache_;
    prefetch_config      cfg_;

    std::atomic<bool>    running_{false};
    std::atomic<bool>    paused_{false};
    std::thread          worker_thread_;

    mutable std::mutex   pattern_mutex_;
    pattern_state        pattern_;

    // Pages currently being prefetched (avoid duplicate work)
    mutable std::mutex   in_flight_mutex_;
    std::unordered_set<std::int64_t> in_flight_;

    // Hint notification
    mutable std::mutex   hint_mutex_;
    std::condition_variable hint_cv_;
    std::vector<std::int64_t> recent_hints_;

    // Statistics
    std::atomic<std::size_t> hits_{0};
    std::atomic<std::size_t> misses_{0};
};

} // namespace zlink
