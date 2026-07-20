#pragma once
// zlink/chunk_cache.hpp — Page-level cache with coherence protocol
//
// Port of r3map's SyncedReadWriterAt + Puller + Pusher architecture
// to C++ for zlink's memory subsystem.
//
// The key insight from r3map: once a chunk has been fetched from remote,
// it is stored in a LOCAL backend. Subsequent reads of the same chunk
// are served from the local backend with ZERO network traffic.
//
// Architecture (mirrors r3map's managed mount):
//
//   ┌───────────────────────────────────────────────────────────┐
//   │                    chunk_cache                            │
//   │                                                           │
//   │  chunk_map: { page_offset → { local: bool, dirty: bool } }│
//   │                                                           │
//   │  read(offset):                                            │
//   │    if chunk_map[offset].local:                            │
//   │      → return from LOCAL store (no network!)              │
//   │    else:                                                  │
//   │      → fetch from remote via memory_client                │
//   │      → write to LOCAL store                               │
//   │      → set chunk.local = true                             │
//   │                                                           │
//   │  write(offset, data):                                     │
//   │    → write to LOCAL store                                 │
//   │    → set chunk.local = true, chunk.dirty = true           │
//   │                                                           │
//   │  invalidate(dirty_offsets):                               │
//   │    → set chunk.local = false (force re-fetch)             │
//   │                                                           │
//   │  Background pusher thread:                                │
//   │    → periodically syncs dirty chunks back to remote       │
//   │                                                           │
//   │  Background puller thread:                                │
//   │    → preemptively fetches chunks not yet local            │
//   └───────────────────────────────────────────────────────────┘
//
// This directly addresses the "pulling the same data over and over" concern:
// after the first fetch, data stays local until explicitly invalidated.

#include <zlink/config.hpp>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <system_error>
#include <cassert>

namespace zlink {

// ── Chunk metadata ────────────────────────────────────────────────────
struct chunk_state {
    bool local = false;   // Data is available in local store
    bool dirty = false;   // Local copy differs from remote
};

// ── Local storage backend interface ───────────────────────────────────
// Represents a local cache for chunks (e.g., a file, mmap'd region,
// or in-memory buffer). This is the r3map "local backend".
class local_store {
public:
    virtual ~local_store() = default;

    // Read a chunk from the local store
    virtual std::error_code read_at(std::uintptr_t offset,
                                     std::span<std::byte> buf) = 0;

    // Write a chunk to the local store
    virtual std::error_code write_at(std::uintptr_t offset,
                                      std::span<const std::byte> data) = 0;

    // Ensure data is persisted (for dirty writeback)
    virtual std::error_code sync() = 0;
};

// ── Remote backend interface ──────────────────────────────────────────
// Represents the remote memory that we're caching (the r3map "remote backend").
class remote_backend {
public:
    virtual ~remote_backend() = default;

    // Read a chunk from remote memory
    virtual std::error_code read_at(std::uintptr_t offset,
                                     std::span<std::byte> buf) = 0;

    // Write a chunk to remote memory (for pushback)
    virtual std::error_code write_at(std::uintptr_t offset,
                                      std::span<const std::byte> data) = 0;

    // Get the total size of the remote memory region
    virtual std::size_t size() const = 0;
};

// ── In-memory local store ─────────────────────────────────────────────
// Simple heap-backed local store for small-to-medium regions.
// For large regions, use an mmap-file-based store.
class memory_local_store : public local_store {
public:
    explicit memory_local_store(std::size_t capacity)
        : data_(capacity, std::byte{0}), capacity_(capacity) {}

    std::error_code read_at(std::uintptr_t offset,
                             std::span<std::byte> buf) override {
        if (offset + buf.size() > capacity_) {
            return std::make_error_code(std::errc::invalid_argument);
        }
        std::memcpy(buf.data(), data_.data() + offset, buf.size());
        // Note: <cstring> included above for std::memcpy
        return {};
    }

    std::error_code write_at(std::uintptr_t offset,
                              std::span<const std::byte> src) override {
        if (offset + src.size() > capacity_) {
            return std::make_error_code(std::errc::invalid_argument);
        }
        std::memcpy(data_.data() + offset, src.data(), src.size());
        return {};
    }

    std::error_code sync() override { return {}; } // In-memory, no sync needed

    std::byte* data() { return data_.data(); }
    std::size_t capacity() const { return capacity_; }

private:
    std::vector<std::byte> data_;
    std::size_t capacity_;
};

// ── Pull priority function ────────────────────────────────────────────
// Determines which chunks to pull first (r3map's pullPriority).
// Higher return value = higher priority.
// Default: all chunks have equal priority.
using pull_priority_fn = std::function<int64_t(std::int64_t offset)>;

// ── Chunk cache configuration ─────────────────────────────────────────
struct chunk_cache_config {
    std::size_t    chunk_size       = 4096;          // Page size (4 KiB default)
    std::size_t    pull_workers     = 4;             // Concurrent pull workers
    std::size_t    push_workers     = 4;             // Concurrent push workers
    std::chrono::milliseconds push_interval{5000};   // Dirty writeback interval
    bool           pull_first       = false;         // Wait for all chunks before serving?
    pull_priority_fn pull_priority  = nullptr;       // Priority heuristic
    bool           verbose          = false;
};

// ── Chunk cache ───────────────────────────────────────────────────────
// The core cache, porting r3map's SyncedReadWriterAt + Pusher + Puller.
//
// Key behavior:
//   - First read of a chunk → fetch from remote, store locally, mark as local
//   - Subsequent reads of same chunk → served from LOCAL store (no network!)
//   - Writes go to local store, chunk marked dirty
//   - Background pusher syncs dirty chunks back to remote periodically
//   - Background puller preemptively fetches not-yet-local chunks
//   - Invalidation marks chunks as non-local (force re-fetch)
class chunk_cache {
public:
    // Callback when a chunk becomes locally available
    using on_chunk_local_fn = std::function<void(std::int64_t offset)>;

    chunk_cache(std::shared_ptr<remote_backend> remote,
                std::shared_ptr<local_store> local,
                chunk_cache_config config = {});
    ~chunk_cache();

    // ── Core read/write (SyncedReadWriterAt equivalent) ───────────

    // Read from the cache. If the chunk is local, returns from local store.
    // If not, fetches from remote, stores locally, and returns.
    std::error_code read(std::uintptr_t offset, std::span<std::byte> buf);

    // Write to the local store and mark the chunk as dirty.
    // The background pusher will sync it to remote later.
    std::error_code write(std::uintptr_t offset, std::span<const std::byte> data);

    // ── Coherence ─────────────────────────────────────────────────

    // Invalidate specific chunks (mark as non-local, forcing re-fetch).
    // This is r3map's MarkAsRemote — used when the remote side has changed.
    void invalidate(std::span<const std::int64_t> dirty_offsets);

    // Invalidate all chunks
    void invalidate_all();

    // ── Background operations ─────────────────────────────────────

    // Start background puller and pusher threads
    void start_background();

    // Stop background threads and flush dirty data
    void stop_background();

    // Manually flush all dirty chunks to remote
    std::size_t sync_dirty();

    // ── Query ─────────────────────────────────────────────────────

    // Is a specific chunk available locally?
    bool is_local(std::int64_t chunk_index) const;

    // Get the number of local (cached) chunks
    std::size_t local_count() const;

    // Get total number of chunks
    std::size_t total_chunks() const;

    // Get chunk size
    std::size_t chunk_size() const { return config_.chunk_size; }

    // Set callback for when a chunk becomes local
    void set_on_chunk_local(on_chunk_local_fn fn) { on_chunk_local_ = std::move(fn); }

private:
    // Get chunk index from byte offset
    std::int64_t offset_to_chunk(std::uintptr_t offset) const {
        return static_cast<std::int64_t>(offset / config_.chunk_size);
    }

    // Get byte offset from chunk index
    std::uintptr_t chunk_to_offset(std::int64_t chunk_idx) const {
        return static_cast<std::uintptr_t>(chunk_idx) * config_.chunk_size;
    }

    // Get or create chunk state
    chunk_state& get_chunk(std::int64_t chunk_idx);

    // Fetch a chunk from remote and store locally
    std::error_code fetch_chunk(std::int64_t chunk_idx);

    // Push a single dirty chunk to remote
    std::error_code push_chunk(std::int64_t chunk_idx);

    // Background puller thread
    void puller_loop();

    // Background pusher thread
    void pusher_loop();

    // ── State ─────────────────────────────────────────────────────
    std::shared_ptr<remote_backend> remote_;
    std::shared_ptr<local_store>    local_;
    chunk_cache_config              config_;

    // Chunk state map (r3map's chunk map)
    mutable std::shared_mutex       chunks_mutex_;
    std::unordered_map<std::int64_t, chunk_state> chunks_;

    // Dirty chunk tracking (r3map's TrackingReadWriterAt)
    mutable std::mutex              dirty_mutex_;
    std::unordered_set<std::int64_t> dirty_chunks_;

    // Pushable chunks (chunks that have been written locally and can be pushed)
    mutable std::mutex              pushable_mutex_;
    std::unordered_set<std::int64_t> pushable_chunks_;

    // Background threads
    std::atomic<bool>               running_{false};
    std::thread                     puller_thread_;
    std::thread                     pusher_thread_;

    // Puller state
    mutable std::mutex              pull_queue_mutex_;
    std::vector<std::int64_t>       pull_queue_;
    std::size_t                     pull_next_ = 0;

    // Callback
    on_chunk_local_fn               on_chunk_local_;

    // Total size of remote region
    std::size_t                     total_size_ = 0;
};

} // namespace zlink
