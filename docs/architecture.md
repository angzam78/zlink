# zlink Architecture

This document describes zlink's architecture as it exists in code today. Every
component, class, and data flow referenced here corresponds to actual
implemented source files.

## High-Level Overview

zlink is a C++20 framework that wraps any shared library for remote execution
over TCP. Its primary use case is **GPU-over-IP**: running CUDA workloads on
CPU-only machines by forwarding GPU calls to a remote server with a real GPU.

The architecture is designed around one core principle: **pipelining that
always works is better than complex pipelining that sometimes breaks.**

```
  Client (CPU-only machine)                     Server (GPU machine)
  ┌──────────────────────────┐                  ┌──────────────────────────┐
  │ Application code          │                  │ zlink server              │
  │    ↓                      │                  │    ↓                      │
  │ cuda_pipeline<RpcDef>     │                  │ handle_pipeline_mem()     │
  │    ↓                      │                  │    ↓                      │
  │ zpp_bits serialization    │   TCP frames     │ zpp_bits deserialization  │
  │    ↓                      │ ═════════════►   │    ↓                      │
  │ Transport (TCP)           │                  │ Transport (TCP)           │
  │                           │  ◄═════════════  │                           │
  │ Virtual handle allocator  │   TCP frames     │ Handle table (VH→real)    │
  │ Host memory mirror client │                  │ Host memory mirror server │
  └──────────────────────────┘                  └──────────────────────────┘
```

The client application calls CUDA functions exactly as it would with a local
GPU. Wrapper functions categorize each call by its dependency type, and the
pipeline automatically batches calls into the fewest possible network
round-trips.

## Source Layout

```
zlink/
├── include/zlink/
│   ├── config.hpp           # Protocol constants, frame types, error codes
│   ├── transport.hpp        # Abstract transport interface + frame struct
│   ├── tcp_transport.hpp    # TCP transport with length-prefixed framing
│   ├── rpc.hpp              # RPC engine (typed client, pipeline caller, server)
│   ├── ptr_map.hpp          # Bidirectional pointer mapping
│   ├── memory.hpp           # Remote memory subsystem (cached_memory_client,
│   │                        #   host_memory_mirror, memory_server)
│   ├── chunk_cache.hpp      # Page-level cache with coherence (r3map-inspired)
│   ├── shared_mem.hpp       # Shared memory plane (backend interface)
│   ├── virtual_handle.hpp   # Virtual handle system (VH encoding, handle_table,
│   │                        #   handle manifest serialization)
│   ├── cuda_pipeline.hpp    # Dependency-aware CUDA RPC pipeline
│   ├── cuda_dep_spec.hpp    # CUDA API dependency categorization
│   ├── compress.hpp         # LZ4 compression for pipeline frames
│   ├── shim.hpp             # LD_PRELOAD shim + symbol interception
│   ├── client.hpp           # High-level client framework
│   └── server.hpp           # Server framework (dlopen, function registry)
├── src/
│   ├── tcp_transport.cpp    # TCP framing implementation
│   ├── rpc.cpp              # RPC send/receive implementation
│   ├── memory_region.cpp    # Memory region management
│   ├── chunk_cache.cpp      # Chunk cache implementation
│   ├── ptr_map.cpp          # Pointer map implementation
│   ├── shared_mem.cpp       # Shared memory plane implementation
│   ├── connection_manager.cpp  # Auto-reconnect + heartbeat
│   ├── multiplexed_transport.cpp  # 3-channel TCP (RPC/bulk/prefetch)
│   ├── prefetch_worker.cpp  # Background page prefetch
│   ├── write_behind_buffer.cpp  # Async write-behind with fence sync
│   ├── server.cpp           # Server framework implementation
│   └── server_main.cpp      # Server entry point
├── examples/
│   ├── libmath/             # Remote libm example
│   └── cuda/                # CUDA RPC server + pipeline client
├── tests/
│   ├── test_rpc.cpp         # RPC unit tests
│   └── test_compress.cpp    # Compression unit tests
├── sim/                     # Performance simulation + charts
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
| `pipeline_request` | `0x04` | Batched RPC: multiple calls in one frame |
| `pipeline_response` | `0x05` | Batched response for pipeline |
| `pipeline_mem` | `0x06` | Pipeline with inline memory operations (sync+RPC+reads) |
| `memory_op` | `0x10` | Remote memory operation |
| `memory_reply` | `0x11` | Memory operation response |
| `heartbeat` | `0xFF` | Keep-alive |

`tcp_transport` implements the transport over TCP sockets with `TCP_NODELAY`
enabled for minimal latency. All sends are mutex-protected for thread safety.

### 2. RPC Engine (`rpc.hpp`)

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
// In a shared header:
namespace cuda_rpc_api {
    struct AllocRet { int32_t result; uint64_t dev_ptr; };
    AllocRet mem_alloc(uint64_t bytesize);
}

using cuda_rpc = zpp::bits::rpc<
    zpp::bits::bind<&cuda_rpc_api::mem_alloc, 7>
>;
```

The integer bind IDs (0, 1, 2, ...) are the wire-level function identifiers.
Client and server must use the same IDs.

### 3. CUDA Pipeline (`cuda_pipeline.hpp`)

The dependency-aware pipeline that makes GPU-over-IP performant. See
[cuda-pipeline.md](cuda-pipeline.md) for full details.

**Key methods on `cuda_pipeline<RpcDef>`**:

| Method | Category | Behavior |
|--------|----------|----------|
| `call_barrier<N>(args...)` | barrier | Flush pending calls, send single sync RPC |
| `enqueue<N>(args...)` | enqueued | Batch into pipeline, no network traffic |
| `enqueue_with_sync<N>(host_ptr, size, args...)` | enqueued | Batch + inline host_sync data |
| `enqueue_produces_handle<N>(args...)` | enqueued | Batch + allocate virtual handle ID |
| `enqueue_produces_handle_with_sync<N>(...)` | enqueued | Combine: sync + handle production |
| `call_readback<N>(args...)` | readback | Flush + sync RPC |
| `call_readback_with_sync_read<N>(ptr, size, args...)` | readback | Full DtoH: pre-sync + RPC + read |
| `flush()` | — | Send all pending calls/syncs as a batch |
| `flush_with_reads()` | — | Send batch + process read requests |

### 4. Virtual Handle System (`virtual_handle.hpp`)

Breaks the dependency chain by replacing real GPU handles with client-side
virtual IDs. See [cuda-pipeline.md](cuda-pipeline.md) for the full algorithm.

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
  Sent alongside `pipeline_mem` frames so the server knows which return values
  to register in the handle table.

### 5. Memory Subsystem (`memory.hpp`, `chunk_cache.hpp`)

The remote memory subsystem provides transparent access to memory on the other
side of the network. See [memory.md](memory.md) for full details.

**Key classes**:

- **`host_memory_mirror`** (server side) — Mirrors client host memory on the
  server via mmap'd regions. When a client sends a `host_sync` operation, the
  data is stored in the server's mirror region and the client address is
  translated to a server-local address that CUDA functions can dereference.

- **`cached_memory_client`** — Client-side cached access to remote memory.
  Uses `chunk_cache` for page-level caching: first fetch goes to remote,
  subsequent reads are served from local cache with zero network overhead.

- **`memory_server`** — Server-side handler for memory operations (read, write,
  alloc, free, host_sync).

### 6. Shared Memory Plane (`shared_mem.hpp`)

Adapts the r3map Backend interface (`ReadAt/WriteAt/Size/Sync`) for zlink's
use case. Provides:

- **`backend`** — Abstract interface matching r3map's Go Backend
- **`memory_backend`** — Wraps a local memory span as a backend
- **`rpc_backend`** — Proxies ReadAt/WriteAt over the zlink RPC transport
- **`chunked_backend`** — Stores chunks as individual files
- **`shared_mem_plane`** — Central coordinator for client↔server memory access

### 7. Pointer Map (`ptr_map.hpp`)

Bidirectional mapping between client-side "fake" pointers and server-side real
pointers. Supports shadow mmap regions so client pointers look like real
addresses.

Key operations:
- `map(remote_ptr)` → returns a local pointer (from shadow region or sequential ID)
- `to_remote(local_ptr)` → look up the real pointer
- `to_local(remote_ptr)` → look up the local pointer
- `translate_pointers(span)` → batch translate embedded pointers

### 8. Shim Layer (`shim.hpp`)

LD_PRELOAD-based interception. Key classes:

- **`symbol_interceptor`** — Manages mapping between intercepted symbols and
  their RPC wrappers. Supports both explicit registration and auto-generation
  of wrappers for common calling conventions.

- **`opaque_wrapper<Signature>`** — Generic wrapper that serializes arguments,
  sends RPC, and deserializes responses. Pre-generated for arities 0-16.

### 9. Server Framework (`server.hpp`)

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

### Pipeline Batch

```
Client → Server:  [frame: type=pipeline_request,
                    payload=[4B count][4B len1][req1][4B len2][req2]...]
Server → Client:  [frame: type=pipeline_response,
                    payload=[4B count][4B len1][resp1][4B len2][resp2]...]
```

### Pipeline with Memory Ops (pipeline_mem)

```
Client → Server:  [frame: type=pipeline_mem,
  [4B sync_count]
    for each sync: [8B addr][8B original_size][1B comp_flag][4B data_size][data...]
  [4B rpc_count]
    for each rpc: [4B len][rpc_bytes]
  [4B read_count]
    for each read: [8B addr][8B size]
  [handle_manifest: 4B count + N × handle_manifest_entry]
]
Server → Client:  [frame: type=pipeline_response,
  [4B rpc_count]
    for each rpc: [4B resp_len][resp_bytes]
  [4B read_count]
    for each read: [4B data_len][1B comp_flag][data...]
]
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
   cuCtxSynchronize. All batched. Host data is inline-synced in the pipeline_mem
   frame.

4. **Readback (flush)** — cuMemcpyDtoH. This triggers the pipeline flush,
   sending all accumulated calls in one round-trip.

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

- **zpp_bits** — Header-only C++20 binary serialization + RPC framework.
  Fetched at build time via CMake `FetchContent`. Provides `zpp::bits::rpc<>`,
  `zpp::bits::bind<>`, and `data_in_out()` for zero-overhead serialization.

- **C++20** — Required for concepts, `std::span`, designated initializers.

- **POSIX** — For `dlopen`/`dlsym`, `mmap`/`munmap`, TCP sockets.

- **Linux** (optional) — `userfaultfd` for demand paging on shadow regions.

## Design Principles

1. **Correctness over performance** — If in doubt, make it a barrier. That's
   always correct. Optimize to enqueued only when you're sure it's safe.

2. **On-demand flushing** — The pipeline flushes only when forced (barrier or
   readback). No timers, no eager sending. This naturally maximizes batch sizes.

3. **No code generation** — The API surface is declared using C++20 templates
   and zpp_bits bindings. No IDL, no protoc, no build step.

4. **Virtual handles eliminate barriers** — The biggest performance win comes
   from making handle-producing calls non-blocking. VH is the key innovation.

5. **Wire efficiency** — LZ4 compression for large transfers, inline memory
   ops to avoid extra round-trips, pipeline_mem frames that combine everything
   into a single network message.
