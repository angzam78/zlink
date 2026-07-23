# Memory Subsystem

This document describes zlink's remote memory subsystem as implemented in
`memory.hpp`, `chunk_cache.hpp`, `shared_mem.hpp`, `ptr_map.hpp`, and their
corresponding `.cpp` files. The memory synchronization architecture is based
on [r3map](https://github.com/pojntfx/r3map)'s managed mount model
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

- **`flush_dirty()`** — Sync all dirty pages to remote.

- **`invalidate(offsets)` / `invalidate_all()`** — Mark cached pages as
  non-local, forcing re-fetch on next access.

- **`enable_demand_paging(base, size)`** (Linux only) — Register a shadow
  memory region with userfaultfd. Page faults automatically trigger reads
  through the chunk_cache.

### Demand Paging (Linux/userfaultfd)

On Linux, `cached_memory_client` can use `userfaultfd` for transparent demand
paging:

1. A shadow memory region is `mmap`'d on the client
2. `userfaultfd` is registered on this region
3. When the application accesses a page in the shadow region, a page fault
   occurs
4. The fault handler reads the page from the remote server through
   `chunk_cache`
5. If the page is already cached, it's served from local store (no network!)
6. The fault is resolved and the application continues

This makes remote memory appear as local memory — the application doesn't
need to know that the data is on another machine.

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

## Shared Memory Plane

The `shared_mem_plane` class (in `shared_mem.hpp`) is the central coordinator
that makes client memory accessible to the server. It adapts r3map's Backend
interface for zlink's use case.

### Backend Interface

```cpp
class backend {
    virtual error_code read_at(span<byte> buf, int64_t offset) = 0;
    virtual error_code write_at(span<const byte> buf, int64_t offset) = 0;
    virtual int64_t size() const = 0;
    virtual error_code sync() = 0;
};
```

This mirrors r3map's Go Backend interface (`ReadAt/WriteAt/Size/Sync`).

### Backend Implementations

| Backend | Purpose | Location |
|---------|---------|----------|
| `memory_backend` | Wraps a local memory span (non-owning) | Client side |
| `rpc_backend` | Proxies ReadAt/WriteAt over RPC transport | Server side |
| `chunked_backend` | Stores chunks as individual files | Either side |

### Client-Server Memory Flow

For `cuMemcpyHtoD(dst, srcHost=0x7fff1234, 1024)`:

1. Client shim registers `srcHost` with `shared_mem_plane::register_region()`
   → creates a `memory_backend` wrapping the host data
2. Client sends RPC call with pointer + region ID
3. Server calls `translate_to_server(srcHost, 1024, region_id)`
   → checks if data is already in local cache
   → if not, fetches via `rpc_backend::read_at()` back to the client
   → returns a server-local pointer
4. Server calls the real `cuMemcpyHtoD(dst, local_ptr, 1024)`
   → CUDA reads from the server-local pointer, which has the same data as
     the client's original buffer

For `cuMemcpyDtoH(dstHost, src, 1024)` (output buffer):

1. Client registers the output buffer as writable
2. Server calls `translate_to_server()` to get a server-local pointer
3. Server calls the real `cuMemcpyDtoH(local_ptr, src, 1024)`
4. Server calls `mark_dirty()` and `flush_dirty()`
5. Dirty data is written back to the client via `rpc_backend::write_at()`
6. Client's buffer now contains the GPU data

## Memory Operation Protocol

Memory operations use `frame_type::memory_op` (0x10) and
`frame_type::memory_reply` (0x11) frames. See [wire-protocol.md](wire-protocol.md)
for the frame format details.

The `host_sync` (0x07) operation is the most important for CUDA pipelining.
It allows the client to push host data to the server's mirror region without
a separate round-trip — the data is packed inline in the `pipeline_mem`
frame.


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

## Inline Memory Ops in pipeline_mem

The `pipeline_mem` frame (frame_type 0x06) eliminates separate round-trips for
memory operations by inlining them into the pipeline batch:

### Without pipeline_mem (3 round-trips for cuMemcpyHtoD):

```
1. host_sync frame → server mirrors data    [1 RT]
2. request frame   → server calls cuMemcpyHtoD  [1 RT]
3. response frame  → client gets CUresult    [1 RT]
```

### With pipeline_mem (1 round-trip):

```
1. pipeline_mem frame →
   [sync: host data][rpc: cuMemcpyHtoD call]
   → server processes: apply sync, execute call
   → response: [rpc result]                 [1 RT]
```

### With pipeline_mem + readback for cuMemcpyDtoH (1 round-trip):

```
1. pipeline_mem frame →
   [sync: zero-fill dest buffer][rpc: cuMemcpyDtoH call][read: dest addr+size]
   → server processes: apply sync, execute DtoH, read mirror data
   → response: [rpc result][read data]      [1 RT]
```

## Implementation: `src/memory_region.cpp`

The `sync_page()` method auto-registers mirror regions. When the server receives
a `host_sync` for an address that isn't in a registered region, it automatically
allocates a new mmap region to cover the address range. This allows the client
to sync any host memory without prior registration.
