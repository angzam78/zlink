#pragma once
// zlink/shared_mem.hpp — Shared Memory Plane
//
// The core insight from r3map: instead of copying pointer data through RPC
// payloads (which requires per-function marshalling logic), we make the
// client's memory directly accessible to the server via a shared memory plane.
//
// Architecture:
//
//   CLIENT                                   SERVER
//   ┌────────────────────────┐               ┌────────────────────────┐
//   │ Application             │               │ Real .so function       │
//   │    │                    │               │    │                    │
//   │    ▼                    │               │    ▼                    │
//   │ Host memory (malloc'd,  │   NBD/r3map   │ NBD block device        │
//   │ stack, mmap'd, etc.)    │══════════════►│ ┌────────────────────┐ │
//   │    │                    │  Backend IF   │ │ mmap'd into server  │ │
//   │    ▼                    │               │ │ address space       │ │
//   │ shared_mem_plane        │               │ └────────────────────┘ │
//   │   registers regions     │               │    │                    │
//   │   exposes via Backend   │               │    ▼                    │
//   │                          │               │ Function reads/writes   │
//   │ Region 0: [0x1000, 4K]  │───chunk 0────►│ → server addr [0x7000]  │
//   │ Region 1: [0x2000, 8K]  │───chunk 1────►│ → server addr [0x8000]  │
//   │ Region 2: [0x5000, 1M]  │───chunk 2────►│ → server addr [0xA000]  │
//   └────────────────────────┘               └────────────────────────┘
//
// When a server function receives a pointer like `const void* srcHost`
// pointing to client address 0x2000, the shared_mem_plane on the server
// has already mmap'd that data at a local server address. The server
// translates 0x2000 → 0x8000 (server-local) and the function just
// dereferences it normally. No per-function marshalling needed!
//
// This is exactly what r3map does: it provides a Backend interface
// (ReadAt/WriteAt/Size/Sync) and uses NBD to make remote chunks
// available on-demand. We adapt this to C++ and integrate it with
// the zlink RPC transport.

#include <zlink/config.hpp>
#include <zlink/ptr_map.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <functional>
#include <memory>
#include <system_error>
#include <string>

namespace zlink {

// ── Backend interface (mirrors r3map's Go Backend) ─────────────────────
// Any resource can implement this: local memory, file, S3, Redis, etc.
// This is the SAME interface as r3map's backend.Backend in Go:
//
//   type Backend interface {
//       ReadAt(p []byte, off int64) (n int, err error)
//       WriteAt(p []byte, off int64) (n int, err error)
//       Size() (int64, error)
//       Sync() error
//   }

class backend {
public:
    virtual ~backend() = default;

    virtual std::error_code read_at(std::span<std::byte> buf, std::int64_t offset) = 0;
    virtual std::error_code write_at(std::span<const std::byte> buf, std::int64_t offset) = 0;
    virtual std::int64_t size() const = 0;
    virtual std::error_code sync() = 0;
};

// ── Memory backend ─────────────────────────────────────────────────────
// Backed by a local memory region. Used on the client side to expose
// application memory (heap, stack, mmap'd) to the server.

class memory_backend : public backend {
public:
    // Wrap an existing memory region (non-owning)
    explicit memory_backend(std::span<std::byte> region)
        : region_(region) {}

    std::error_code read_at(std::span<std::byte> buf, std::int64_t offset) override {
        if (offset + static_cast<std::int64_t>(buf.size()) >
            static_cast<std::int64_t>(region_.size())) {
            return std::make_error_code(std::errc::invalid_argument);
        }
        std::memcpy(buf.data(), region_.data() + offset, buf.size());
        return {};
    }

    std::error_code write_at(std::span<const std::byte> buf, std::int64_t offset) override {
        if (offset + static_cast<std::int64_t>(buf.size()) >
            static_cast<std::int64_t>(region_.size())) {
            return std::make_error_code(std::errc::invalid_argument);
        }
        std::memcpy(region_.data() + offset, buf.data(), buf.size());
        return {};
    }

    std::int64_t size() const override {
        return static_cast<std::int64_t>(region_.size());
    }

    std::error_code sync() override {
        // No-op for memory backend; data is already in RAM
        return {};
    }

private:
    std::span<std::byte> region_;
};

// ── RPC backend ────────────────────────────────────────────────────────
// Proxies read_at/write_at over the zlink RPC transport.
// Used on the server side to access client memory remotely.
// This is the C++ equivalent of r3map's RPC backend.

class rpc_backend : public backend {
public:
    using read_fn  = std::function<std::error_code(std::int64_t offset,
                                                    std::span<std::byte> buf)>;
    using write_fn = std::function<std::error_code(std::int64_t offset,
                                                    std::span<const std::byte> buf)>;
    using size_fn  = std::function<std::int64_t()>;
    using sync_fn  = std::function<std::error_code()>;

    explicit rpc_backend(read_fn r, write_fn w, size_fn s, sync_fn sy)
        : read_fn_(std::move(r)), write_fn_(std::move(w)),
          size_fn_(std::move(s)), sync_fn_(std::move(sy)) {}

    std::error_code read_at(std::span<std::byte> buf, std::int64_t offset) override {
        return read_fn_(offset, buf);
    }

    std::error_code write_at(std::span<const std::byte> buf, std::int64_t offset) override {
        return write_fn_(offset, buf);
    }

    std::int64_t size() const override {
        return size_fn_();
    }

    std::error_code sync() override {
        return sync_fn_();
    }

private:
    read_fn  read_fn_;
    write_fn write_fn_;
    size_fn  size_fn_;
    sync_fn  sync_fn_;
};

// ── Chunked file backend ──────────────────────────────────────────────
// Stores chunks as individual files (like r3map's directory backend).
// Used for caching and persistence.

class chunked_backend : public backend {
public:
    explicit chunked_backend(const std::string& dir, std::size_t chunk_size = 4096);

    std::error_code read_at(std::span<std::byte> buf, std::int64_t offset) override;
    std::error_code write_at(std::span<const std::byte> buf, std::int64_t offset) override;
    std::int64_t size() const override;
    std::error_code sync() override;

private:
    std::string dir_;
    std::size_t chunk_size_;
    std::int64_t total_size_;
};

// ── Memory region descriptor ───────────────────────────────────────────
// Describes a region of memory registered with the shared memory plane.
struct mem_region {
    std::uintptr_t  client_addr;   // Address on the client
    std::size_t     size;          // Size in bytes
    std::uint64_t   region_id;     // Unique ID for this region
    bool            writable;      // Can the server write to this?
};

// ── Shared Memory Plane ────────────────────────────────────────────────
// The central coordinator that makes client memory accessible to the server.
//
// On the CLIENT:
//   1. Application calls a function with a host pointer
//   2. The shim registers the memory region containing that pointer
//   3. The shared_mem_plane exposes it via a Backend interface
//   4. The server accesses it on-demand through the Backend
//
// On the SERVER:
//   1. Receive an RPC call with a "client pointer"
//   2. Look up the pointer in the translation table
//   3. If the data is already local (mmap'd/cached), use it directly
//   4. If not, fetch it on-demand via the Backend (ReadAt)
//   5. After the function returns, write back any dirty regions (WriteAt)
//
// This approach means server functions just dereference pointers normally.
// No per-function marshalling code needed!

class shared_mem_plane {
public:
    // ── Client-side API ────────────────────────────────────────────────

    // Register a memory region for remote access.
    // Returns a region ID that the server can use to reference it.
    std::uint64_t register_region(std::uintptr_t client_addr,
                                   std::size_t size,
                                   bool writable = true);

    // Register a region using an existing backend (e.g., for S3/Redis)
    std::uint64_t register_region_with_backend(std::uintptr_t client_addr,
                                                std::size_t size,
                                                std::shared_ptr<backend> be,
                                                bool writable = true);

    // Unregister a region
    void unregister_region(std::uint64_t region_id);

    // Get the backend for a region (so the server can ReadAt/WriteAt)
    std::shared_ptr<backend> get_backend(std::uint64_t region_id) const;

    // Look up which region contains a given client address
    // Returns {region_id, offset_within_region}
    std::optional<std::pair<std::uint64_t, std::int64_t>>
    find_region(std::uintptr_t client_addr, std::size_t len = 1) const;

    // ── Server-side API ────────────────────────────────────────────────

    // Map a client pointer to a server-local address.
    // If the data isn't local yet, it's fetched on-demand from the
    // client's backend (via ReadAt over RPC).
    //
    // This is the key function: when the server function does
    // `memcpy(dst, srcHost, n)`, `srcHost` was a client pointer.
    // We translate it to a server-local pointer whose contents are
    // lazily fetched from the client.
    std::byte* translate_to_server(std::uintptr_t client_addr,
                                    std::size_t len,
                                    std::uint64_t region_id);

    // Mark a server-local region as dirty (needs write-back to client).
    // Called after a server function writes to a buffer that was
    // registered as writable.
    void mark_dirty(std::uintptr_t server_addr, std::size_t len);

    // Flush all dirty regions back to the client.
    // Called at the end of each RPC call if any output buffers exist.
    std::error_code flush_dirty();

    // ── Bulk data transfer ─────────────────────────────────────────────
    // For large transfers (cuMemcpyHtoD/DtoH), we still want explicit
    // bulk transfer rather than on-demand paging. The shared_mem_plane
    // supports both modes.

    // Pull a range from the client (explicit bulk read)
    std::error_code pull(std::uint64_t region_id,
                         std::int64_t offset,
                         std::span<std::byte> buf);

    // Push a range to the client (explicit bulk write)
    std::error_code push(std::uint64_t region_id,
                         std::int64_t offset,
                         std::span<const std::byte> buf);

    // ── Background pull/push (managed mount mode) ──────────────────────
    // Inspired by r3map's managed mount API. Pre-fetches chunks
    // before they're needed and writes back changes asynchronously.
    // Essential for WAN deployments where RTT is high.

    // Start background prefetching for a region
    void start_prefetch(std::uint64_t region_id,
                        std::int64_t priority_offset = 0);

    // Stop background prefetching
    void stop_prefetch(std::uint64_t region_id);

    // Start background writeback (periodic flush of dirty pages)
    void start_background_writeback(std::uint64_t region_id,
                                     int interval_ms = 100);

    void stop_background_writeback(std::uint64_t region_id);

    // ── Utilities ──────────────────────────────────────────────────────

    std::size_t region_count() const;
    void clear();

private:
    struct region_entry {
        mem_region                    descriptor;
        std::shared_ptr<backend>      be;
        std::vector<std::byte>        local_cache;  // Server-side cache
        std::vector<bool>             chunk_present; // Which chunks are cached
        std::vector<bool>             chunk_dirty;   // Which chunks need writeback
        std::size_t                   chunk_size = 4096;
    };

    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, region_entry> regions_;
    std::uint64_t next_region_id_ = 1;
    std::size_t chunk_size_ = 4096;
};

} // namespace zlink
