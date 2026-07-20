# zlink Architecture

zlink is an RPC wrapper for CUDA that enables GPU-over-IP — running CUDA workloads on
CPU-only machines by proxying all CUDA Driver API calls to a remote GPU server over a
TCP connection. The architecture is designed around one core principle: **pipelining that
always works is better than complex pipelining that sometimes breaks.**

## High-Level Architecture

```
┌──────────────────────────┐         TCP          ┌──────────────────────────┐
│  Client (CPU-only)       │ ───────────────────── │  Server (GPU machine)    │
│                          │                       │                          │
│  Application             │                       │  Real CUDA Driver        │
│      ↓                   │                       │      ↓                   │
│  CUDA wrapper functions  │                       │  CUDA API impl           │
│      ↓                   │                       │      ↓                   │
│  cuda_pipeline<RpcDef>   │ ── pipeline_mem ──→   │  handle_pipeline_mem()   │
│      ↓                   │                       │      ↓                   │
│  transport (TCP)         │ ── frames ─────────→  │  transport (TCP)         │
└──────────────────────────┘                       └──────────────────────────┘
```

The client application calls CUDA functions exactly as it would with a local GPU.
Wrapper functions categorize each call by its dependency type, and the pipeline
automatically batches calls into the fewest possible network round-trips.

## Core Components

### Transport Layer (`transport.hpp`, `tcp_transport.hpp`)

Abstract transport interface with TCP implementation. All communication uses a
framed protocol:

```
Frame wire format: [4B length][4B call_id][1B type][payload...]
```

Frame types (defined in `config.hpp`):
- `request (0x01)` / `response (0x02)` — single synchronous RPC
- `pipeline_request (0x04)` / `pipeline_response (0x05)` — batched RPC
- `pipeline_mem (0x06)` — batched RPC + inline memory ops + virtual handle manifest
- `memory_op (0x10)` / `memory_reply (0x11)` — standalone memory operations

### RPC Framework (`rpc.hpp`)

Built on zpp_bits (C++20 header-only binary serialization + RPC). The framework
provides:

- **`rpc_client<RpcDef>`** — typed synchronous calls via `call<FuncIndex>(args...)`
- **`rpc_server<RpcDef>`** — dispatches incoming calls to bound functions
- **`pipeline_caller<RpcDef>`** — queues calls and flushes as a batch
- **`dynamic_rpc_server`** — runtime function registration (no codegen needed)

The API surface is declared using `zpp::bits::bind<>` with explicit index numbers.
Client and server must use the same binding indices.

### Dependency-Aware Pipeline (`cuda_pipeline.hpp`)

The heart of zlink. See [pipeline.md](pipeline.md) for full details.

Three-category call classification:
- **barrier** — must have return value before next call (auto-flushes pipeline)
- **enqueued** — batch into pipeline, no immediate return needed
- **readback** — flush pipeline + get data back from server

### Virtual Handle System (`virtual_handle.hpp`, `cuda_dep_spec.hpp`)

Breaks dependency chains on the client side. See [virtual-handles.md](virtual-handles.md).

Key idea: handle-producing calls (cuMemAlloc, cuCtxCreate, etc.) return a virtual
handle ID immediately instead of waiting for the real GPU pointer. The server
translates VH → real handle when processing the batch.

### Memory System (`memory.hpp`, `chunk_cache.hpp`)

Bidirectional memory mirroring for zero-copy pointer dereferencing. See
[memory-system.md](memory-system.md).

- **`host_memory_mirror`** (server-side) — mirrors client host memory so server
  functions can dereference "client pointers"
- **`cached_memory_client`** (client-side) — page-level caching with demand paging
- **`chunk_cache`** — r3map-inspired local store for fetched pages

### LZ4 Compression (`compress.hpp`)

Per-entry compression for large data transfers in pipeline_mem frames. See
[compression.md](compression.md).

- Threshold: only compress entries ≥ 4 KB
- Ratio check: only use compressed version if ≥ 10% smaller
- Applies to sync data (client→server) and read data (server→client)
- RPC payloads are never compressed (too small)

### Pointer Map (`ptr_map.hpp`)

Bidirectional pointer mapping between local (client) and remote (server) address
spaces. Supports shadow regions (mmap'd address ranges where "pointers" are real
addresses) and sequential ID allocation. Used by the generic RPC layer when
translating opaque handles.

## Wire Protocol

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
