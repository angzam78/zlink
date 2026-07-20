# Memory System

zlink's memory system provides bidirectional memory access between client and
server, enabling transparent pointer dereferencing across the network.

## The Problem

CUDA functions often take pointers to host memory. For example, `cuMemcpyHtoD`
takes a `srcHost` pointer to client-side data that the server needs to read
from. The server can't directly dereference a client address — it needs the
data mirrored locally.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Client Host Memory                                              │
│  ┌─────────┐     ┌──────────────────────┐                       │
│  │ App buf  │ ←→  │ host_memory_mirror   │ ←── userfaultfd      │
│  └─────────┘     │ (chunk_cache + local │     on page fault     │
│                   │  store)              │     → auto-fetch      │
│                   └──────────┬───────────┘     from server       │
│                              │                                   │
│                   RPC: sync_host_page(offset, data)              │
└──────────────────────────────┼───────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│  Server                                                          │
│  host_memory_mirror receives client pages                        │
│  → stores in server-local mmap region                            │
│  → server functions can dereference "client pointers"            │
│    because the memory is mirrored on the server!                 │
└──────────────────────────────────────────────────────────────────┘
```

## Components

### `host_memory_mirror` (Server-Side)

Defined in `include/zlink/memory.hpp`. Mirrors client host memory on the server
so that server-side CUDA functions can dereference "client pointers" as if they
were local.

Key operations:
- **`register_region(client_base, size)`** — Allocates a server-local mmap
  region to mirror a range of client addresses.
- **`sync_page(client_addr, data)`** — Writes client data into the mirrored
  region. Called when the client sends a host_sync operation.
- **`translate(client_addr)`** — Maps a client address to the server's mirror
  address. Returns `nullopt` if the address is not in a registered region.
- **`read(client_addr, buf)`** — Reads from the mirrored memory into a buffer.

Server functions use `translate()` to convert client pointers:

```cpp
CopyHtoDRet memcpy_htod(std::uint64_t dst_dev_ptr,
                         std::uint64_t src_client_addr,
                         std::uint64_t byte_count) {
    auto real_dst = g_vhandles.translate(dst_dev_ptr);
    auto mirror_addr = g_host_mirror.translate(src_client_addr);
    const void* real_src = mirror_addr
        ? reinterpret_cast<const void*>(*mirror_addr)
        : reinterpret_cast<const void*>(src_client_addr);
    CUresult r = cuMemcpyHtoD(cu_dst, real_src, byte_count);
    ...
}
```

### `cached_memory_client` (Client-Side)

Extends basic memory access with page-level caching. After the first read of a
page, data is stored in a `local_store`. Subsequent reads of the same page
come from local storage with zero network overhead.

Key operations:
- **`read(remote_addr, local_buf)`** — Read from remote, using page cache
- **`write(remote_addr, local_buf)`** — Write to local cache + mark dirty
- **`alloc(size, out_addr)`** — Allocate remote memory
- **`flush_dirty()`** — Push dirty pages to remote
- **`invalidate(span)`** — Invalidate specific cached pages

On Linux, supports **demand paging** via `userfaultfd` — page faults on the
shadow region automatically trigger reads through the chunk_cache.

### `chunk_cache`

r3map-inspired caching layer. Provides the `SyncedReadWriterAt` abstraction:
- Local file-backed store for fetched chunks
- Chunk-level tracking (dirty, cached, invalid)
- Background puller (preemptive fetch) and pusher (dirty writeback)

### `memory_server`

Server-side handler for memory operations. Registers memory regions (e.g., GPU
device memory) and handles read/write/alloc/free requests from clients. Also
hosts the `host_memory_mirror` for client→server pointer translation.

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
