# Memory Subsystem

This document describes zlink's remote memory subsystem as implemented in
`memory.hpp`, `chunk_cache.hpp`, `memory_page_tracker.hpp`, `ptr_map.hpp`,
and their corresponding `.cpp` files. The memory synchronization architecture
is based on [r3map](https://github.com/pojntfx/r3map)'s managed mount model
(SyncedReadWriterAt + Puller + Pusher). Research paper:
[Networked Linux Memory Synchronization](https://pojntfx.github.io/networked-linux-memsync/main.pdf).

## Overview

The memory subsystem solves two fundamental problems in GPU-over-IP:

1. **Host memory access**: When a CUDA function takes a pointer to host
   memory (e.g., `cuMemcpyHtoD(dst, srcHost, n)`), the server needs to
   access the client's data. The `host_memory_mirror` makes client host
   memory available on the server.

2. **Device memory access**: The client needs to read/write GPU memory
   remotely. The `cached_memory_client` provides page-level caching for
   efficient remote access.

## Host Memory Mirror

The `host_memory_mirror` class (in `memory.hpp`) mirrors client host memory
on the server side using mmap'd regions.

### How It Works

```
CLIENT                                            SERVER
┌────────────────────────┐                        ┌────────────────────────┐
│ host_data at 0x7fff0   │  host_sync RPC         │ host_memory_mirror     │
│ [float: 0.0, 1.5, 3.0]│ ──────────────────►    │                        │
│                        │                        │ register_region()      │
│                        │                        │   client_base=0x7fff0 │
│                        │                        │   server_base=0x80000  │
│                        │                        │                        │
│                        │  sync_page() data      │ sync_page()            │
│                        │ ──────────────────►    │   memcpy to 0x80000    │
│                        │                        │                        │
│                        │                        │ translate(0x7fff0)     │
│                        │                        │   → 0x80000            │
│                        │                        │                        │
│                        │                        │ CUDA reads 0x80000     │
│                        │                        │ = same data as client  │
└────────────────────────┘                        └────────────────────────┘
```

### Key Methods

- **`register_region(client_base, size)`** — Allocates a server-local mmap'd
  region to mirror the client's memory. Returns the server-side base address.
  Stores a `region` struct mapping `client_base → server_base`.

- **`sync_page(client_addr, data)`** — Copies client data into the server's
  mirror region. Called when the client sends a `host_sync` operation. The
  data is written to `server_base + (client_addr - client_base)`.

- **`translate(client_addr)`** — Translates a client address to the
  corresponding server-local address. Returns `nullopt` if the client address
  is not in a registered region.

- **`is_registered(client_addr)`** — Checks whether a client address falls
  within a registered mirror region.

### Implementation Details

- Mirror regions are created with `mmap(MAP_ANONYMOUS | MAP_PRIVATE)`.
- Region lookup is linear (scans `regions_` vector). For workloads with many
  regions, this could be optimized with an interval tree.
- All methods are mutex-protected for thread safety.

## Cached Memory Client

The `cached_memory_client` class (in `memory.hpp`) provides page-level caching
for remote memory access, inspired by r3map's managed mount architecture.

### Architecture

```
Application
    ↓
cached_memory_client
    ├─ memory_page_tracker (demand paging + dirty tracking)
    │   ├─ uffd WP_ASYNC (Tier 1, kernel 6.2+)
    │   ├─ uffd WP sync  (Tier 2, kernel 4.11+)
    │   └─ mprotect+SIGSEGV (Tier 3, any POSIX)
    ├─ chunk_cache (page-level coherence)
    │   ├─ chunk_map: { offset → { local, dirty } }
    │   └─ background threads (puller + pusher)
    ├─ local_store (in-memory cache)
    └─ rpc_remote_backend (network access to server)
```

### Key Methods

- **`read(remote_addr, local_buf)`** — Read from remote memory with caching.
  If the page is already cached locally, returns from local store (zero
  network traffic). Otherwise, fetches from remote and caches.

- **`write(remote_addr, local_buf)`** — Write to local cache, mark as dirty.
  Background pusher syncs to remote periodically.

- **`alloc(size, out_addr)` / `free(addr)`** — Remote memory allocation and
  deallocation.

- **`flush_dirty()`** — Sync all dirty pages to remote. Collects dirty pages
  from the `memory_page_tracker` (if demand paging is enabled), writes them into the
  `chunk_cache`'s local store, then pushes all dirty chunks to the server via
  `host_sync`. The ordering is critical: memory_page_tracker dirty pages must be
  written to the cache *before* `sync_dirty()` is called, otherwise they would
  not be pushed.

- **`invalidate(offsets)` / `invalidate_all()`** — Mark cached pages as
  non-local, forcing re-fetch on next access.

- **`enable_demand_paging(base, size)`** (Linux/POSIX) — Register a shadow
  memory region with the `memory_page_tracker`. The tracker handles both demand
  paging (read faults → fetch from server via `chunk_cache`) and write
  tracking (write faults → mark dirty for `flush_dirty()`). The factory
  auto-selects the best available tier: uffd WP_ASYNC → uffd WP sync →
  mprotect+SIGSEGV.

### Demand Paging + Dirty Page Tracking (Linux/userfaultfd, or mprotect fallback)

On Linux, `cached_memory_client` uses `memory_page_tracker` for transparent demand
paging and dirty page tracking. The `memory_page_tracker` is the single fault handler for
the shadow region and serves both roles:

**Read-side demand paging** — when the application first accesses a page:
1. A shadow memory region is `mmap`'d on the client (PROT_NONE or registered
   with userfaultfd)
2. The first access triggers a page fault
3. The fault handler calls `read_fault_cb`, which fetches the page from the
   remote server through `chunk_cache`
4. If the page is already cached, it's served from local store (no network!)
5. The tracker installs the page (UFFDIO_COPY for uffd, mprotect+memcpy for
   the fallback) and write-protects it for dirty tracking
6. The fault is resolved and the application continues

**Write-side dirty tracking** — when the application writes to an installed page:
1. The write triggers a WP fault (uffd) or SIGSEGV (mprotect)
2. The page is marked dirty in an internal bitmap
3. The page is made writable (auto-resolved in WP_ASYNC, or via ioctl /
   mprotect in sync modes)
4. On the next `flush_dirty()`, dirty pages are collected and pushed to the
   server via `host_sync` on the memory channel

**Three-tier runtime selection** (`memory_page_tracker::create()`):

| Tier | Mechanism | Requirement | Description |
|------|-----------|-------------|-------------|
| 1 | userfaultfd WP + WP_ASYNC | Linux 6.2+ | Write faults auto-resolved by kernel; zero ioctls per write fault |
| 2 | userfaultfd WP synchronous | Linux 4.11+ | One `UFFDIO_WRITEPROTECT` ioctl per first-write per page |
| 3 | mprotect + SIGSEGV | Any POSIX | Fallback for containers where seccomp blocks `userfaultfd(2)` |

The factory probes each tier at runtime and selects the best available. Tier 3
is always available on POSIX systems, ensuring demand paging and dirty page
tracking work even in restricted environments (e.g., Docker containers with
default seccomp profiles that return `EPERM` for `userfaultfd(2)`).

**Address translation**: The `memory_page_tracker` operates on absolute virtual
addresses from the shadow region. The `cached_memory_client` stores the shadow
region's base address and translates:
- Absolute → offset for `chunk_cache` and `local_store` (in `read()`,
  `write()`, `flush_dirty()`, and the `read_fault_cb`)
- Offset → absolute for `rpc_remote_backend` (in `read_at()` and
  `write_at()`, via `set_base()`)

This ensures the server's `host_memory_mirror` receives the same absolute
client addresses that CUDA RPC calls reference (e.g., the `srcHost` pointer
in `cuMemcpyHtoD`).

## Chunk Cache

The `chunk_cache` class (in `chunk_cache.hpp`) implements page-level caching
with coherence, porting r3map's `SyncedReadWriterAt` + Puller + Pusher
architecture to C++.

### Core Behavior

| Operation | Behavior |
|-----------|----------|
| First read of a chunk | Fetch from remote, store locally, mark as `local` |
| Subsequent reads | Serve from local store (no network!) |
| Write | Write to local store, mark as `local + dirty` |
| Invalidate | Mark as non-local (force re-fetch on next read) |
| Background pusher | Periodically sync dirty chunks back to remote |
| Background puller | Preemptively fetch chunks not yet local |

### Configuration

```cpp
struct chunk_cache_config {
    size_t chunk_size = 4096;          // Page size (4 KiB default)
    size_t pull_workers = 4;           // Concurrent pull workers
    size_t push_workers = 4;           // Concurrent push workers
    chrono::milliseconds push_interval{5000}; // Dirty writeback interval
    bool pull_first = false;           // Wait for all chunks before serving?
    pull_priority_fn pull_priority;    // Priority heuristic for pull order
    bool verbose = false;
};
```

### Backend Interfaces

- **`local_store`** — Abstract interface for local caching. Implemented by
  `memory_local_store` (heap-backed, in-memory). For large regions, a
  file-backed implementation would be more appropriate.

- **`remote_backend`** — Abstract interface for remote data access. Both
  `read_at()` and `write_at()` are virtual, allowing different transport
  mechanisms.

## Pointer Map

The `ptr_map` class (in `ptr_map.hpp`) manages bidirectional mapping between
client-side "fake" pointers and server-side real pointers.

### Shadow Regions

When a shadow memory region is configured (`set_shadow_region(base, size)`),
local pointers are allocated from this region. This makes them look like real
addresses to the client application — the application can do pointer
arithmetic, comparisons, and dereference them (via demand paging).

Without a shadow region, local pointers are sequential IDs (1, 2, 3, ...).

### Key Methods

- **`map(remote_ptr)`** — Register a remote↔local mapping. If already mapped,
  returns the existing local pointer. Otherwise allocates a new one.

- **`to_remote(local_ptr)`** — Look up the remote pointer for a local pointer.

- **`to_local(remote_ptr)`** — Look up the local pointer for a remote pointer.

- **`translate_pointers(span)`** — Batch translate: walk a span of local
  pointers and replace each with its remote equivalent. Used for function
  arguments that contain embedded pointers (e.g., `kernelParams` in
  `cuLaunchKernel`).

- **`unmap(local_ptr)`** — Remove a mapping (both directions).

## Memory Backends and Cache

The `chunk_cache` class (in `chunk_cache.hpp`) is the central coordinator
that makes remote memory locally accessible with caching. It adapts r3map's
SyncedReadWriterAt model for zlink's use case.

### Backend Interfaces

```cpp
class local_store {
    virtual error_code read_at(uintptr_t offset, span<byte> buf) = 0;
    virtual error_code write_at(uintptr_t offset, span<const byte> data) = 0;
    virtual error_code sync() = 0;
};

class remote_backend {
    virtual error_code read_at(uintptr_t offset, span<byte> buf) = 0;
    virtual error_code write_at(uintptr_t offset, span<const byte> data) = 0;
    virtual size_t size() const = 0;
};
```

### Backend Implementations

| Backend | Purpose | Location |
|---------|---------|----------|
| `memory_local_store` | In-memory buffer backing the local cache | Client side |
| `rpc_remote_backend` | Proxies read_at/write_at over RPC transport | Client side |

### Client-Server Memory Flow

For `cuMemcpyHtoD(dst, srcHost=0x7fff1234, 1024)`:

1. Client sends a `host_sync` memory_op frame with the host data to the
   server's `host_memory_mirror`
2. Client enqueues the RPC call `memcpy_hto_d(VH(dst), client_addr, 1024)`
3. Server receives the `host_sync`: stores data in mirror region
4. Server receives the RPC: translates `client_addr` → server-local mirror
   address via `host_memory_mirror::translate()`
5. Server calls the real `cuMemcpyHtoD(real_dst, local_ptr, 1024)`
   → CUDA reads from the server-local pointer, which has the same data as
     the client's original buffer

For `cuMemcpyDtoH(dstHost, src, 1024)` (output buffer):

1. Client calls `pipe.call_readback<memcpy_dto_h>(client_addr, VH(src), 1024)`
   which flushes any pending batch, then sends the RPC synchronously
2. Server processes the RPC: translates VH and client address, calls real
   `cuMemcpyDtoH(server_local_ptr, real_src, 1024)`
3. Client calls `pipe.host_read(dstHost, 1024)`: sends a `host_read` frame
4. Server reads from the mirror region and sends data back to client
5. Client copies data into `dstHost` buffer

## Memory Operation Protocol

Memory operations use `frame_type::memory_op` (0x10) and
`frame_type::memory_reply` (0x11) frames. See [wire-protocol.md](wire-protocol.md)
for the frame format details.

The `host_sync` (0x07) operation is the most important for CUDA pipelining.
It allows the client to push host data to the server's mirror region. This
operates on the memory channel, separate from RPC traffic.


## Memory Operations

Defined in `include/zlink/memory.hpp`:

| Operation | Code | Direction | Description |
|-----------|------|-----------|-------------|
| `read` | 0x01 | Server→Client | Read from remote memory |
| `write` | 0x02 | Client→Server | Write to remote memory |
| `alloc` | 0x03 | Client→Server | Allocate remote memory |
| `free_op` | 0x04 | Client→Server | Free remote memory |
| `sync` | 0x05 | Bidirectional | Flush dirty / invalidate |
| `invalidate` | 0x06 | Server→Client | Invalidate cached pages |
| `host_sync` | 0x07 | Client→Server | Sync client host page to server mirror |
| `host_read` | 0x08 | Server→Client | Read from mirrored client host memory |

## Separation of RPC and Memory Data

Memory data flows on a separate channel from RPC calls. This is a deliberate
architectural decision: RPC frames stay small (just call metadata + virtual
handles), while bulk memory transfers use the memory channel without blocking
RPC traffic.

### cuMemcpyHtoD flow (2 channels, 1 RPC round-trip):

```
1. App writes to shadow region → memory_page_tracker marks page dirty
2. flush_dirty() → collect dirty pages → host_sync → server mirrors data  [memory channel]
3. pipeline_request → server calls cuMemcpyHtoD using mirrored data       [RPC channel, 1 RT]
```

The host_sync is sent before the RPC batch. The memory_page_tracker transparently
detects which pages were written; only dirty pages are pushed.

### cuMemcpyDtoH flow (readback, 1 RPC round-trip + demand paging):

```
1. invalidate_all() → clear local cache for fresh data
2. pipeline_request (flushes batch) → cuMemcpyDtoH writes to server mirror  [RPC channel, 1 RT]
3. App reads shadow region → demand page fault → host_read → server sends    [memory channel]
   mirror data back → page installed + cached for next read
```

The readback buffer is accessed page-by-page via demand paging. The server's
`memcpy_dto_h` handler auto-registers the mirror region for the readback
address if it hasn't been synced by the client.

## Implementation: `src/memory_region.cpp`

The `sync_page()` method auto-registers mirror regions. When the server receives
a `host_sync` for an address that isn't in a registered region, it automatically
allocates a new mmap region to cover the address range. This allows the client
to sync any host memory without prior registration.

The server's `memcpy_dto_h` handler (in `examples/cuda/cuda_server.cpp`) also
auto-registers the mirror region for readback buffer addresses that haven't
been previously synced. This is necessary because the client's readback buffer
is only written to by the server (via `cuMemcpyDtoH`), never synced from the
client.

### Address Translation

The `cached_memory_client` stores the shadow region's base address
(`impl_->shadow_base`) and translates between absolute virtual addresses
(used by the shadow region, memory_page_tracker, and CUDA RPC calls) and 0-based
offsets (used by `chunk_cache` and `local_store`):

```
External (absolute addr)          Internal (offset)
─────────────────────────         ──────────────────
shadow_base + 0          ←→       0
shadow_base + 4096       ←→       4096
shadow_base + N          ←→       N

read(addr)        →  cache_->read(addr - shadow_base)
write(addr)       →  cache_->write(addr - shadow_base)
flush_dirty()     →  cache_->write(r.addr - shadow_base, ...)
read_fault_cb     →  cache_->read(fault_addr - shadow_base)
```

The `rpc_remote_backend` does the reverse translation: it adds `base_` to
the offset when sending `host_sync` / `host_read` requests, so the server's
`host_memory_mirror` receives the same absolute client addresses that CUDA
RPC calls reference.
