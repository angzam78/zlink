#pragma once
// zlink/memory.hpp — Remote memory subsystem (r3map-inspired)
//
// Provides transparent access to remote memory regions (e.g., GPU VRAM)
// via two modes:
//   1. Explicit copy API   — cuMemcpyHtoD/cuMemcpyDtoH style
//   2. Demand paging       — userfaultfd-based transparent access (Linux only)
//
// NEW: Bidirectional memory synchronization with page-level caching.
//
// The r3map architecture solves the "pulling the same data over and over"
// problem with a local backend that caches fetched chunks. After the first
// fetch, subsequent reads of the same page come from local storage —
// zero network overhead. Only invalidation or eviction forces a re-fetch.
//
// Architecture (r3map managed mount pattern):
//
//   Client process                         Server process
//   ┌─────────────────────┐               ┌──────────────────────┐
//   │ Application          │               │ Real .so              │
//   │    ↓                 │               │    ↓                  │
//   │ Shadow region (mmap) │               │ Real memory (GPU etc) │
//   │    ↓                 │               │    ↓                  │
//   │ ┌──────────────────┐ │               │                      │
//   │ │  chunk_cache     │ │  ── RPC ──→   │ host_memory_mirror        │
//   │ │  (local store +  │ │               │ (read/write/alloc/   │
//   │ │   chunk tracking)│ │               │  free handlers)      │
//   │ └──────────────────┘ │               │                      │
//   │    ↑                 │               │                      │
//   │  userfaultfd         │               │                      │
//   │  (demand paging)     │               │                      │
//   └─────────────────────┘               └──────────────────────┘
//
// Bidirectional sync (the zlink extension):
//
//   ┌──────────────────────────────────────────────────────────────┐
//   │  Client Host Memory                                         │
//   │  ┌─────────┐     ┌──────────────────────┐                   │
//   │  │ App buf  │ ←→  │ host_memory_mirror   │ ←── userfaultfd  │
//   │  └─────────┘     │ (chunk_cache + local │     on page fault  │
//   │                   │  store)              │     → auto-fetch   │
//   │                   └──────────┬───────────┘     from server    │
//   │                              │                                 │
//   │                   RPC: sync_host_page(offset, data)            │
//   │                              │                                 │
//   └──────────────────────────────┼─────────────────────────────────┘
//                                  ↓
//   ┌──────────────────────────────────────────────────────────────┐
//   │  Server                                                      │
//   │  host_memory_mirror_server receives client pages             │
//   │  → stores in server-local mmap region                        │
//   │  → server functions can dereference "client pointers"        │
//   │    because the memory is mirrored on the server!             │
//   └──────────────────────────────────────────────────────────────┘

#include <zlink/config.hpp>
#include <zlink/chunk_cache.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <memory>
#include <system_error>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace zlink {

// ── Memory operation types (sent as frame_type::memory_op) ────────────
enum class mem_op : std::uint8_t {
    read       = 0x01,  // Read from remote memory
    write      = 0x02,  // Write to remote memory
    alloc      = 0x03,  // Allocate remote memory
    free_op    = 0x04,  // Free remote memory
    sync       = 0x05,  // Synchronize (flush dirty / invalidate)
    invalidate = 0x06,  // Invalidate cached pages (server→client notification)
    host_sync  = 0x07,  // Bidirectional: sync client host page to server
    host_read  = 0x08,  // Server reads from mirrored client host memory
};

// ── Memory operation request ──────────────────────────────────────────
struct mem_request {
    mem_op          op;
    std::uintptr_t  remote_addr;  // Address in server's address space
    std::uint64_t   size;         // Number of bytes
    // For write ops, data follows after this header in the frame payload
};

// ── Memory operation response ─────────────────────────────────────────
struct mem_response {
    error_code      status;
    std::uint64_t   size;         // Bytes actually read/written
    std::uintptr_t  remote_addr;  // For alloc: the allocated address
    // For read ops, data follows after this header in the frame payload
};

// ── Cached memory client ─────────────────────────────────────────────
// Extends the basic memory_client with page-level caching.
//
// After the first read of a page, the data is stored in a local_store.
// Subsequent reads of the same page are served from local_store —
// ZERO network traffic. Only invalidation forces a re-fetch.
//
// This is the zlink equivalent of r3map's managed mount:
//   - chunk_cache  ↔ SyncedReadWriterAt
//   - local_store  ↔ local file backend
//   - remote_backend ↔ RPC backend (gRPC in r3map, zpp_bits in zlink)
class cached_memory_client {
public:
    cached_memory_client(class rpc_client_base& rpc,
                         std::size_t region_size);
    cached_memory_client(class rpc_client_base& rpc,
                         std::size_t region_size,
                         const chunk_cache_config& config);
    ~cached_memory_client();

    // ── Cached read/write ────────────────────────────────────────

    // Read from remote memory, using page cache.
    // If the page is already cached locally → no network round-trip!
    std::error_code read(std::uintptr_t remote_addr,
                         std::span<std::byte> local_buf);

    // Write to remote memory (writes to local cache + marks dirty).
    // Background pusher syncs to remote periodically.
    std::error_code write(std::uintptr_t remote_addr,
                          std::span<const std::byte> local_buf);

    // ── Remote memory management ─────────────────────────────────

    // Allocate remote memory, returns the remote address
    std::error_code alloc(std::size_t size, std::uintptr_t& out_addr);

    // Free remote memory
    std::error_code free(std::uintptr_t remote_addr);

    // ── Cache management ─────────────────────────────────────────

    // Flush all dirty pages to remote
    std::size_t flush_dirty();

    // Invalidate specific pages (e.g., after server notifies of changes)
    void invalidate(std::span<const std::int64_t> page_offsets);

    // Invalidate all cached pages
    void invalidate_all();

    // Start background puller (preemptive fetch) and pusher (dirty writeback)
    void start_background();

    // Stop background threads and flush
    void stop_background();

    // ── Demand paging + write tracking ─────────────────────────────
    // When enabled, page faults on the shadow region automatically
    // trigger reads through the chunk_cache. If the page is cached,
    // the fault is resolved from local store (no network!).
    // If not cached, it fetches from remote and caches for next time.
    // Write faults are tracked as dirty pages for flush_dirty().
    //
    // The memory_page_tracker factory auto-selects the best available tier:
    //   Tier 1: uffd WP_ASYNC (kernel 6.2+)
    //   Tier 2: uffd WP sync (kernel 4.11+)
    //   Tier 3: mprotect + SIGSEGV (any POSIX, including containers)

    std::error_code enable_demand_paging(std::uintptr_t base, std::size_t size);
    void disable_demand_paging();

    // ── Accessors ────────────────────────────────────────────────

    chunk_cache& cache() { return *cache_; }
    const chunk_cache& cache() const { return *cache_; }

private:
    struct impl;
    std::unique_ptr<impl> impl_;

    // The chunk cache (r3map's SyncedReadWriterAt equivalent)
    std::unique_ptr<chunk_cache> cache_;

    // RPC backend adapter (wraps memory_client RPC calls as a remote_backend)
    class rpc_remote_backend;
    std::shared_ptr<rpc_remote_backend> rpc_backend_;

    // Local store for caching
    std::shared_ptr<memory_local_store> local_store_;
};

// ── Host memory mirror (server side) ──────────────────────────────────
// When a client function takes a pointer to host memory (e.g., cuMemcpyHtoD
// with srcHost pointing to client buffer), the server needs to access that
// data. This class mirrors client host memory on the server side.
//
// Flow:
//   1. Client registers a host memory region with the shim
//   2. Shim syncs the region's pages to the server via host_sync RPC
//   3. Server stores the pages in a local mmap region
//   4. Server translates the client pointer to its local mirror address
//   5. Server function can now dereference the "client pointer"
//
// This is the bidirectional extension that r3map doesn't have but zlink needs.
class host_memory_mirror {
public:
    host_memory_mirror();
    ~host_memory_mirror();

    // Register a range of client host memory on the server.
    // The server allocates a local mirror region for it.
    // Returns the server-side base address that corresponds to the
    // client's base address.
    std::uintptr_t register_region(std::uintptr_t client_base,
                                    std::size_t size);

    // Unregister a previously registered region
    void unregister_region(std::uintptr_t client_base);

    // Sync a page of client memory to the server's mirror.
    // Called when the client sends a host_sync operation.
    std::error_code sync_page(std::uintptr_t client_addr,
                               std::span<const std::byte> data);

    // Read from the mirrored client memory (server-side access)
    std::error_code read(std::uintptr_t client_addr,
                          std::span<std::byte> buf) const;

    // Translate a client address to the server-side mirror address.
    // Returns nullopt if the client address is not in a registered region.
    std::optional<std::uintptr_t> translate(std::uintptr_t client_addr) const;

    // Check if a client address is in a registered region
    bool is_registered(std::uintptr_t client_addr) const;

private:
    struct region {
        std::uintptr_t client_base;
        std::size_t    size;
        std::uintptr_t server_base;  // mmap'd mirror region
        void*          server_mmap;  // For munmap
    };

    mutable std::mutex mutex_;
    std::vector<region> regions_;
};

} // namespace zlink
