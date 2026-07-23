# zlink Architecture

This document describes zlink's architecture as it exists in code today. Every
component, class, and data flow referenced here corresponds to actual
implemented source files.

## High-Level Overview

zlink is a C++20 framework that wraps any shared library for remote execution
over TCP. Its primary use case is **GPU-over-IP**: running CUDA workloads on
CPU-only machines by forwarding GPU calls to a remote server with a real GPU.

The architecture separates concerns into two independent layers:

1. **RPC layer** — Batches CUDA calls (metadata only, no memory data) into
   pipeline frames. Virtual handles eliminate dependencies between calls.
2. **Memory layer** — Moves host buffer data independently via a backend
   interface (ReadAt/WriteAt/Sync), with demand paging and write tracking.

This separation is deliberate: RPC frames stay small and fast, while bulk
memory transfers flow on a separate channel without blocking RPC traffic.

```
  Client (CPU-only machine)                     Server (GPU machine)
  ┌──────────────────────────┐                  ┌──────────────────────────┐
  │ Application code          │                  │ zlink server              │
  │    ↓                      │                  │    ↓                      │
  │ cuda_pipeline<RpcDef>     │                  │ handle_pipeline_request() │
  │  (RPC batching, VH)       │                  │    ↓                      │
  │    ↓                      │  RPC channel     │ zpp_bits dispatch         │
  │ Transport (multiplexed)   │ ═════════════►   │ Transport (multiplexed)   │
  │    ↓                      │  ◄═════════════  │    ↓                      │
  │ write_tracker + chunk_cache│ Bulk channel    │ host_memory_mirror        │
  │  (demand paging + cache)  │ ═════════════►   │  (server-side mirror)     │
  │                           │  ◄═════════════  │                           │
  │ Virtual handle allocator  │                  │ Handle table (VH→real)    │
  └──────────────────────────┘                  └──────────────────────────┘
```

The client application calls CUDA functions exactly as it would with a local
GPU. Wrapper functions categorize each call by its dependency type, and the
pipeline automatically batches calls into the fewest possible network
round-trips. Memory data for `cuMemcpyHtoD` is synced separately and does not
block the RPC pipeline.

## Source Layout

```
zlink/
├── include/zlink/
│   ├── config.hpp           # Protocol constants, frame types, error codes
│   ├── transport.hpp        # Abstract transport interface + frame struct
│   ├── tcp_transport.hpp    # TCP transport with length-prefixed framing
│   ├── multiplexed_transport.hpp  # 3-channel TCP (RPC/bulk/prefetch)
│   ├── rpc.hpp              # RPC engine (typed client, pipeline caller, server)
│   ├── ptr_map.hpp          # Bidirectional pointer mapping
│   ├── memory.hpp           # Remote memory subsystem (host_memory_mirror,
│   │                        #   mem_request/mem_response, memory_server)
│   ├── chunk_cache.hpp      # Page-level cache with coherence (r3map-inspired)
│   ├── write_tracker.hpp    # Write tracking + demand paging: uffd WP / mprotect+SIGSEGV
│   ├── shared_mem.hpp       # Backend interface (ReadAt/WriteAt/Size/Sync)
│   ├── virtual_handle.hpp   # Virtual handle system (VH encoding, handle_table,
│   │                        #   handle manifest serialization)
│   ├── cuda_pipeline.hpp    # Dependency-aware CUDA RPC pipeline (batching only)
│   ├── cuda_dep_spec.hpp    # CUDA API dependency categorization
│   ├── compress.hpp         # LZ4 compression for memory transfers
│   ├── shim.hpp             # LD_PRELOAD shim + symbol interception
│   ├── client.hpp           # High-level client framework
│   └── server.hpp           # Server framework (dlopen, function registry)
├── src/
│   ├── tcp_transport.cpp    # TCP framing implementation
│   ├── multiplexed_transport.cpp  # 3-channel routing
│   ├── rpc.cpp              # RPC send/receive implementation
│   ├── memory_region.cpp    # Memory region management
│   ├── chunk_cache.cpp      # Chunk cache implementation
│   ├── write_tracker.cpp    # Write tracking (uffd WP + mprotect+SIGSEGV)
│   ├── ptr_map.cpp          # Pointer map implementation
│   ├── shared_mem.cpp       # Backend interface implementation
│   ├── server.cpp           # Server framework implementation
│   └── server_main.cpp      # Server entry point
├── examples/
│   ├── libmath/             # Remote libm example
│   └── cuda/                # CUDA RPC server + pipeline client
├── tests/
│   ├── test_rpc.cpp         # RPC unit tests
│   └── test_compress.cpp    # Compression unit tests
├── cmake/
│   └── FetchZppBits.cmake   # CMake fetch for zpp_bits dependency
└── docs/                    # Documentation (this directory)
```

## Component Details

### 1. Transport Layer (`transport.hpp`, `tcp_transport.hpp`)

The transport layer provides a blocking send/receive interface for `frame`
objects. Each frame has a 9-byte header:

```
[4 bytes: payload length (big-endian)] [4 bytes: call_id (big-endian)] [1 byte: frame_type]
```

**Implemented frame types** (defined in `config.hpp`):

| Type | Value | Purpose |
|------|-------|---------|
| `request` | `0x01` | Single synchronous RPC call |
| `response` | `0x02` | RPC response |
| `error` | `0x03` | Error response |
| `pipeline_request` | `0x04` | Batched RPC: multiple calls + handle manifest |
| `pipeline_response` | `0x05` | Batched response for pipeline |
| `memory_op` | `0x10` | Remote memory operation (host_sync, host_read) |
| `memory_reply` | `0x11` | Memory operation response |
| `heartbeat` | `0xFF` | Keep-alive |

`tcp_transport` implements the transport over TCP sockets with `TCP_NODELAY`
enabled for minimal latency. All sends are mutex-protected for thread safety.

### 2. Multiplexed Transport (`multiplexed_transport.hpp`)

Three logical channels over separate TCP connections, designed for future
QUIC migration (QUIC streams map naturally to this model):

| Channel | Purpose | Frame types routed here |
|---------|---------|------------------------|
| `rpc_control` | Low-latency RPC | `request`, `response`, `error`, `heartbeat` |
| `bulk_data` | Throughput-sensitive transfers | `pipeline_request`, `pipeline_response`, large `memory_op`/`memory_reply` |
| `prefetch` | Background page prefetch | (reserved for demand-paging prefetch traffic) |

Routing is automatic: `send()` inspects the frame type and picks the channel.
Small memory ops (< 16 KB) go on `rpc_control` to avoid connection overhead;
larger ones go on `bulk_data`. When only a single port is available, the
transport falls back to a single connection.

### 3. RPC Engine (`rpc.hpp`)

The RPC engine wraps [zpp_bits](https://github.com/eyalz800/zpp_bits) for
network RPC. Key classes:

- **`rpc_client<RpcDef>`** — Typed synchronous RPC client. Uses fresh
  `data_in_out()` contexts for each request/response to avoid stale archive
  state (a proven pattern learned from debugging zpp_bits).

- **`pipeline_caller<RpcDef>`** — Batches multiple RPC calls into a single
  `pipeline_request` frame. Turns N round-trips into 1.

- **`rpc_server<RpcDef>`** — Typed RPC server. Receives frames, dispatches
  to bound functions via zpp_bits `serve()`, returns responses.

- **`dynamic_rpc_server`** — Runtime function registration (no compile-time
  API declaration needed). Functions are registered by ID with generic
  handlers.

**API declaration pattern**: Both client and server share the same API type:

```cpp
// In a shared header (examples/cuda/cuda_api.hpp):
namespace cuda_gen {
    struct MemAllocRet { int32_t result; uint64_t dptr; };
    MemAllocRet mem_alloc(uint64_t bytesize);
}

using cuda_gen_rpc = zpp::bits::rpc<
    zpp::bits::bind<&mem_alloc, func_index::mem_alloc>
>;
```

The integer bind IDs (stored in `func_index::*` constants, 0-29) are the
wire-level function identifiers. Client and server must use the same IDs.

### 4. CUDA Pipeline (`cuda_pipeline.hpp`)

The dependency-aware pipeline that makes GPU-over-IP performant. It batches
RPC calls (metadata only — no memory data) and manages virtual handles.

**Key methods on `cuda_pipeline<RpcDef>`**:

| Method | Category | Behavior |
|--------|----------|----------|
| `call_barrier<N>(args...)` | barrier | Flush pending calls, send single sync RPC |
| `enqueue<N>(args...)` | enqueued | Batch into pipeline, no network traffic |
| `enqueue_produces_handle<N>(args...)` | enqueued | Batch + allocate virtual handle ID |
| `call_readback<N>(args...)` | readback | Flush + sync RPC (data via `host_read`) |
| `host_read(ptr, size)` | — | Pull data from server mirror via `memory_op` |
| `flush()` | — | Send all pending calls as a `pipeline_request` batch |

Memory data for `cuMemcpyHtoD` is synced separately via the memory layer
(`host_sync` / `chunk_cache`), not packed into RPC frames.

### 5. Virtual Handle System (`virtual_handle.hpp`)

Breaks the dependency chain by replacing real GPU handles with client-side
virtual IDs.

**Encoding**: Bit 63 set = virtual handle. Lower 63 bits = virtual ID.
Real CUDA pointers are always < 2^63, so this is safe and unambiguous.

```cpp
VH(0) = 0x8000000000000000
VH(1) = 0x8000000000000001
Real ptr: 0x7f18c6c00000 (bit 63 clear)
```

**Key types**:

- **`handle_table`** — Server-side mapping: virtual ID → real CUDA handle.
  Thread-safe with mutex protection.

- **`virtual_handle_allocator`** — Client-side sequential ID allocator.

- **`handle_manifest_entry`** — Describes which pipeline calls produce handles.
  Sent alongside `pipeline_request` frames so the server knows which return
  values to register in the handle table.

### 6. Write Tracker (`write_tracker.hpp`, `write_tracker.cpp`)

The `write_tracker` provides transparent demand paging and write tracking on
shadow memory regions. It is the single fault handler for a registered region
and serves two roles:

- **Read-side demand paging**: When a page is first accessed (MISSING fault /
  PROT_NONE), the tracker calls a `read_fault_cb` callback that fetches the
  page from the server via `chunk_cache`. The tracker then installs the page
  (UFFDIO_COPY for uffd, `mprotect` + `memcpy` for the fallback).
- **Write-side dirty tracking**: After a page is installed, it is
  write-protected. On the next write, a WP fault fires and the page is marked
  dirty. `collect_dirty()` returns the dirty ranges and re-protects them.

**Three-tier runtime selection** (probed at construction time via
`write_tracker::create()`):

| Tier | Mechanism | Kernel requirement | When to use |
|------|-----------|--------------------|-------------|
| 1 | userfaultfd WP + `WP_ASYNC` | Linux 6.2+ | Best performance — write faults auto-resolved by kernel, zero ioctls per write |
| 2 | userfaultfd WP synchronous | Linux 4.11+ | Standard — one `UFFDIO_WRITEPROTECT` ioctl per first-write per page |
| 3 | mprotect + SIGSEGV | Any POSIX | Fallback — works in containers where seccomp blocks `userfaultfd(2)` (e.g., default Docker profile) |

The factory tries Tier 1 → Tier 2 → Tier 3 and returns the first that succeeds.
Tier 3 is always available on POSIX systems, so `create()` only returns
`nullptr` on memory allocation failure.

**Address translation**: The `write_tracker` works with absolute virtual
addresses from the shadow region. The `cached_memory_client` translates these
to 0-based offsets for `chunk_cache` / `local_store`, and the
`rpc_remote_backend` translates offsets back to absolute addresses for server
communication (`host_sync` / `host_read`).

### 7. Memory Subsystem (`memory.hpp`, `chunk_cache.hpp`)

The memory layer moves host buffer data independently of RPC traffic.

**Key classes**:

- **`host_memory_mirror`** (server side) — Mirrors client host memory on the
  server via mmap'd regions. When a client sends a `host_sync` operation, the
  data is stored in the server's mirror region and the client address is
  translated to a server-local address that CUDA functions can dereference.

- **`chunk_cache`** — Page-level cache (r3map-inspired) with background
  puller/pusher threads. Provides `SyncedReadWriterAt` semantics: reads pull
  from remote on miss, writes mark pages dirty for background push.

- **`memory_server`** — Server-side handler for memory operations (read, write,
  alloc, free, host_sync).

### 8. Backend Interface (`shared_mem.hpp`)

Adapts the r3map Backend interface (`ReadAt/WriteAt/Size/Sync`) for zlink's
use case. This is the abstraction layer through which all memory data flows.
Provides:

- **`backend`** — Abstract interface matching r3map's Go Backend
- **`memory_backend`** — Wraps a local memory span as a backend
- **`rpc_backend`** — Proxies ReadAt/WriteAt over the zlink RPC transport
- **`chunked_backend`** — Stores chunks as individual files
- **`shared_mem_plane`** — Central coordinator for client↔server memory access

### 9. Pointer Map (`ptr_map.hpp`)

Bidirectional mapping between client-side "fake" pointers and server-side real
pointers. Supports shadow mmap regions so client pointers look like real
addresses.

Key operations:
- `map(remote_ptr)` → returns a local pointer (from shadow region or sequential ID)
- `to_remote(local_ptr)` → look up the real pointer
- `to_local(remote_ptr)` → look up the local pointer
- `translate_pointers(span)` → batch translate embedded pointers

### 10. Shim Layer (`shim.hpp`)

LD_PRELOAD-based interception. Key classes:

- **`symbol_interceptor`** — Manages mapping between intercepted symbols and
  their RPC wrappers. Supports both explicit registration and auto-generation
  of wrappers for common calling conventions.

- **`opaque_wrapper<Signature>`** — Generic wrapper that serializes arguments,
  sends RPC, and deserializes responses. Pre-generated for arities 0-16.

### 11. Server Framework (`server.hpp`)

- **`function_registry`** — Maps function names/IDs to real implementations
  loaded from a shared library via `dlopen`/`dlsym`.

- **`connection_handler`** — Per-connection handler that receives frames and
  dispatches to registered functions or memory operations.

- **`server`** — Main server class. Opens target library, listens for TCP
  connections, spawns handler threads.

## Wire Protocol Summary

### Single RPC Call

```
Client → Server:  [frame: type=request,  payload=zpp_bits(request)]
Server → Client:  [frame: type=response, payload=zpp_bits(response)]
```

### Pipeline Batch (RPC only, no memory data)

```
Client → Server:  [frame: type=pipeline_request,
                    payload=[4B count][4B len1][req1][4B len2][req2]...
                              [handle_manifest: 4B count + N × handle_manifest_entry]]
Server → Client:  [frame: type=pipeline_response,
                    payload=[4B count][4B len1][resp1][4B len2][resp2]...]
```

### Memory Operation (separate from RPC)

```
Client → Server:  [frame: type=memory_op,
                    payload=[mem_request: op, addr, size][data... for host_sync]]
Server → Client:  [frame: type=memory_reply,
                    payload=[mem_response: status, size, addr][data... for host_read]]
```

See [wire-protocol.md](wire-protocol.md) for the complete protocol specification.

## Data Flow: Typical Workload

A typical CUDA workload goes through these phases:

1. **Setup (barriers)** — cuInit, cuDeviceGetCount, cuDeviceGetAttribute.
   Each is a synchronous RPC round-trip because the result affects subsequent
   decisions (which GPU to use, how much memory, etc.).

2. **Resource Creation (enqueued with VH)** — cuCtxCreate, cuMemAlloc,
   cuModuleLoadData, cuModuleGetFunction, cuStreamCreate. All return virtual
   handles. No round-trips needed — they're batched.

3. **Data Transfer + Compute (enqueued)** — cuMemcpyHtoD, cuLaunchKernel,
   cuCtxSynchronize. All batched. Host data for HtoD is synced via the memory
   layer (host_sync), not packed into RPC frames. The write tracker
   transparently detects dirty pages in the shadow region; `flush_dirty()`
   pushes only modified pages to the server.

4. **Readback (flush)** — cuMemcpyDtoH. This triggers the pipeline flush,
   sending all accumulated calls in one round-trip. Data comes back via
   `host_read` on the memory channel. Touching the readback buffer triggers
   demand paging — each page is fetched from the server's mirror on first
   access and cached locally for subsequent reads.

5. **Cleanup (enqueued)** — cuMemFree, cuCtxDestroy. Batched and flushed
   with the next explicit flush or at connection close.

## Configuration Constants (`config.hpp`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `default_port` | 14833 | Default TCP port |
| `max_frame_size` | 64 MiB | Maximum frame payload size |
| `frame_header_size` | 9 bytes | 4 len + 4 call_id + 1 type |
| `initial_buffer_size` | 4096 | Initial serialization buffer |
| `max_concurrent_calls` | 256 | Maximum in-flight RPC calls |

## Dependencies

- **[zpp_bits](https://github.com/eyalz800/zpp_bits)** — Header-only C++20
  binary serialization + RPC framework. Fetched at build time via CMake
  `FetchContent`. Provides `zpp::bits::rpc<>`, `zpp::bits::bind<>`, and
  `data_in_out()` for zero-overhead serialization. zlink's entire RPC layer
  (request/response framing, pipeline batch serialization, server dispatch)
  is built on zpp_bits.

- **[r3map](https://github.com/pojntfx/r3map)** — Remote memory region
  mounting library. zlink's memory synchronization subsystem (chunk cache,
  host memory mirror, backend interface) is based on r3map's managed mount
  architecture (SyncedReadWriterAt + Puller + Pusher pattern). The research
  paper is at <https://pojntfx.github.io/networked-linux-memsync/main.pdf>.

- **[LZ4](https://github.com/lz4/lz4)** — Fast compression for memory
  transfers on the bulk channel.

- **C++20** — Required for concepts, `std::span`, designated initializers.

- **POSIX** — For `dlopen`/`dlsym`, `mmap`/`munmap`, TCP sockets.

- **Linux** (optional) — `userfaultfd` for demand paging on shadow regions.
  When `userfaultfd(2)` is blocked (e.g., by container seccomp profiles),
  the write tracker transparently falls back to `mprotect` + `SIGSEGV`,
  which works on any POSIX system.

## Design Principles

1. **Correctness over performance** — If in doubt, make it a barrier. That's
   always correct. Optimize to enqueued only when you're sure it's safe.

2. **Separation of RPC and memory** — RPC frames carry only call metadata
   and virtual handles. Memory data flows independently via the backend
   interface, so bulk transfers never block RPC traffic.

3. **On-demand flushing** — The pipeline flushes only when forced (barrier or
   readback). No timers, no eager sending. This naturally maximizes batch sizes.

4. **No code generation** — The API surface is declared using C++20 templates
   and zpp_bits bindings. No IDL, no protoc, no build step.

5. **Virtual handles eliminate barriers** — The biggest performance win comes
   from making handle-producing calls non-blocking. VH is the key innovation.

6. **QUIC-ready transport** — The 3-channel multiplexed transport maps
   naturally to QUIC streams, enabling future migration from TCP to QUIC
   without application-level changes.
