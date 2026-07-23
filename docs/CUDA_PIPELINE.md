# CUDA Pipeline and Virtual Handles

This document describes zlink's dependency-aware CUDA RPC pipeline as
implemented in `cuda_pipeline.hpp`, `virtual_handle.hpp`, and
`cuda_dep_spec.hpp`.

## The Core Problem

CUDA workloads are sequential: `cuMemAlloc` returns a device pointer that
`cuMemcpyHtoD` needs. In a naive RPC system, every call is a synchronous
round-trip:

```
cuInit()           → 1 RT (wait for result)
cuDeviceGetCount() → 1 RT (wait for result)
cuMemAlloc(256)    → 1 RT (MUST wait for dev_ptr!)
cuMemcpyHtoD(ptr)  → 1 RT (uses dev_ptr from above)
cuCtxSynchronize() → 1 RT (wait for completion)
cuMemcpyDtoH(out)  → 1 RT (wait for data)
Total: 6 round-trips for a trivial workload
```

On a LAN with 0.1ms RTT, that's 0.6ms of pure latency. On a WAN with 10ms
RTT, that's 60ms — completely unusable for real GPU workloads.

## The Solution: Dependency-Aware Pipelining

zlink categorizes each CUDA call by its dependency type:

### Barrier

The caller **must** have the return value before the next call. The pipeline
auto-flushes any pending batch, then sends this call as a single synchronous
RPC.

**When to use**: Client code branches on the return value in a way that
affects subsequent GPU operations.

**Examples**: `cuInit` (need to know if CUDA is available), `cuDeviceGetCount`
(need count to iterate), `cuDeviceGetAttribute` (need value for occupancy
calculations).

### Enqueued

The caller doesn't need the return value immediately. The call is batched
into the pipeline with no network traffic until the next flush.

**When to use**: The call's return value is just a status code that can be
checked later, or the call doesn't produce data needed by subsequent client
code.

**Examples**: `cuMemcpyHtoD`, `cuLaunchKernel`, `cuCtxSynchronize`,
`cuMemFree`, `cuStreamSynchronize`.

### Readback

The caller needs data back from the server after execution. The pipeline
auto-flushes (so all prior work executes first), then sends the RPC + host
read as one round-trip.

**When to use**: Copying GPU data back to host memory.

**Examples**: `cuMemcpyDtoH`, `cuMemcpyDtoHAsync`.

## Virtual Handles: Breaking the Barrier Chain

Even with the three-category system, many calls were barriers because they
produced handle values (device pointers, contexts, modules, etc.) needed by
subsequent calls. Virtual handles eliminate these barriers.

### How It Works

**Without virtual handles** (every alloc is a barrier):
```
cuMemAlloc(256)           → BARRIER: wait for dev_ptr    (1 RT)
cuMemcpyHtoD(dev_ptr,...) → ENQUEUED: batch it
cuCtxSynchronize()        → ENQUEUED: batch it
cuMemcpyDtoH(out,dev_ptr) → READBACK: flush + get data  (1 RT)
Total: 2 RTs, but only 1 call in the pipeline batch
```

**With virtual handles** (alloc returns a virtual ID, no barrier):
```
cuMemAlloc(256)           → ENQUEUED: returns VH(0)      (no wait!)
cuMemcpyHtoD(VH(0),...)  → ENQUEUED: references VH(0)   (no wait!)
cuCtxSynchronize()        → ENQUEUED: batch it           (no wait!)
cuMemcpyDtoH(out,VH(0))  → READBACK: flush ALL + read   (1 RT)
Total: 1 RT for the entire pipeline batch!
```

### Virtual Handle Encoding

Virtual handles use bit 63 as a flag:

```cpp
// From virtual_handle.hpp:
constexpr uint64_t vhandle_flag = (1ULL << 63);

VH(0) = 0x8000000000000000
VH(1) = 0x8000000000000001
VH(2) = 0x8000000000000002
...
Real CUDA ptr: 0x7f18c6c00000  (bit 63 is clear)
```

This encoding is safe because user-space addresses on Linux are always below
`0x7fffffffffff` (47-bit canonical addresses). Bit 63 never appears in real
pointers, so the encoding is unambiguous.

### Client-Side Flow

1. Client calls `enqueue_produces_handle<7>(256)` for `cuMemAlloc(256)`
2. The pipeline:
   - Allocates a virtual handle ID: `vh_id = vhandle_alloc_.allocate()` → 0
   - Enqueues the RPC call normally
   - Records a `handle_manifest_entry`: call_index=0, virtual_id=0,
     return_field=1 (the `dev_ptr` field in `AllocRet`)
   - Returns `vh_id = 0`
3. Client wraps it: `make_virtual_handle(0)` → `0x8000000000000000`
4. Subsequent calls pass this VH value as an argument:
   `enqueue<9>(VH(0), client_addr, byte_count)` for `cuMemcpyHtoD`
5. At flush time, the handle manifest is serialized into the `pipeline_mem`
   frame alongside the sync data and RPC calls

### Server-Side Flow

1. Server receives a `pipeline_mem` frame with:
   - Sync entries (host data to mirror)
   - RPC calls (serialized zpp_bits requests)
   - Read requests (addresses to read back)
   - Handle manifest (which calls produce handles)
2. Server processes calls sequentially:
   - For each call, check the handle manifest
   - After executing a handle-producing call, read `g_last_produced_handle`
     and register it: `g_vhandles.register_handle(vh_id, real_value)`
   - For subsequent calls that consume handles, translate:
     `g_vhandles.translate(vh_value)` → real CUDA handle
3. After all calls are processed, handle the read requests

### Handle Manifest Wire Format

The handle manifest is serialized at the end of the `pipeline_mem` frame:

```
[4 bytes: entry count]
[entry 0: call_index(4B) + virtual_id(4B) + return_field(4B) + reserved(4B)]
[entry 1: ...]
...
```

This tells the server exactly which pipeline calls produce handles and which
virtual IDs to assign, so it can register them in the `handle_table` after
executing each call.

## The Handle Manifest + g_last_produced_handle Pattern

The server needs to know the real handle value produced by a CUDA function
(e.g., the real `CUdeviceptr` from `cuMemAlloc`). However, the return value
is serialized inside a zpp_bits varint-encoded response, which is difficult
to parse reliably.

Instead, each CUDA wrapper function on the server side sets a global variable
`g_last_produced_handle` after calling the real CUDA function:

```cpp
// In cuda_server.cpp:
CUdeviceptr real_ptr;
CUresult res = cuMemAlloc(&real_ptr, bytesize);
g_last_produced_handle = static_cast<uint64_t>(real_ptr);
g_has_produced_handle = true;
```

The pipeline handler checks this global after each `serve()` call:

```cpp
if (g_has_produced_handle) {
    g_vhandles.register_handle(entry.virtual_id, g_last_produced_handle);
    g_has_produced_handle = false;
}
```

This avoids parsing the zpp_bits response while still getting the handle value.

## Pipeline Memory Frame Format

The `pipeline_mem` frame (type `0x06`) combines sync data, RPC calls, read
requests, and the handle manifest into a single network round-trip:

```
┌─────────────────────────────────────────────────────────────┐
│ Sync section:                                                │
│   [4B sync_count]                                            │
│   For each sync: [8B client_addr][8B size][data bytes...]   │
├─────────────────────────────────────────────────────────────┤
│ RPC section:                                                 │
│   [4B rpc_count]                                             │
│   For each rpc: [4B len][serialized zpp_bits request bytes]  │
├─────────────────────────────────────────────────────────────┤
│ Read section:                                                │
│   [4B read_count]                                            │
│   For each read: [8B client_addr][8B size]                   │
├─────────────────────────────────────────────────────────────┤
│ Handle manifest:                                             │
│   [4B entry_count]                                           │
│   For each entry: [4B call_index][4B virtual_id]             │
│                   [4B return_field][4B reserved]              │
└─────────────────────────────────────────────────────────────┘
```

The response is a `pipeline_response` frame containing RPC results and
read data:

```
┌─────────────────────────────────────────────────────────────┐
│ RPC results:                                                 │
│   [4B rpc_count]                                             │
│   For each rpc: [4B len][serialized zpp_bits response bytes] │
├─────────────────────────────────────────────────────────────┤
│ Read data:                                                   │
│   [4B read_count]                                            │
│   For each read: [4B len][data bytes...]                     │
└─────────────────────────────────────────────────────────────┘
```

Read data is automatically written to the client's host buffers (registered
via `pending_read` entries) during response parsing.

## Typical Workload Pipeline

A typical CUDA workload with virtual handles:

```
cuInit(0)                    → barrier (1 RT)
cuDeviceGetCount()           → barrier (1 RT)
cuDeviceGetAttribute(...)    → barrier (1 RT)
cuCtxCreate(...)             → enqueued: VH(0) = ctx
cuMemAlloc(N)                → enqueued: VH(1) = input_buf
cuMemAlloc(M)                → enqueued: VH(2) = output_buf
cuModuleLoadData(...)        → enqueued: VH(3) = module  (+ inline sync)
cuModuleGetFunction(VH(3))   → enqueued: VH(4) = kernel_func
cuMemcpyHtoD(VH(1), ...)     → enqueued (+ inline sync)
cuLaunchKernel(VH(4), ...)   → enqueued
cuMemcpyDtoH(out, VH(2),...) → READBACK: flush! (1 RT for ALL above)

Total: 3 barrier RTs + 1 pipeline batch RT = 4 round-trips
Without VH: every alloc/create is a barrier → 10+ round-trips
```

## Categorization Reference

The full categorization for the PyTorch-relevant CUDA Driver API subset is
documented in `cuda_dep_spec.hpp`. Key rules:

| Pattern | Category | Reason |
|---------|----------|--------|
| Returns a handle field | `enqueue_produces_handle` | VH replaces the real handle |
| Takes handles as input | `enqueue` | Server translates VH → real |
| Copies data to host | `readback` | Must flush + get data |
| Client branches on result | `barrier` | Must have value before next call |
| Otherwise | `enqueued` | Safe default for most functions |

**Rule of thumb**: If in doubt, make it a barrier. That's always correct.
Optimize to enqueued only when you're sure it's safe.

## Handle Producers (~30 out of ~2000 CUDA functions)

Only functions whose return structs contain a handle field need
`enqueue_produces_handle`. The complete list of producers:

- `cuMemAlloc`, `cuMemAllocManaged`, `cuMemHostAlloc` → device/host pointer
- `cuCtxCreate` → context handle
- `cuModuleLoadData`, `cuModuleLoad` → module handle
- `cuModuleGetFunction` → function handle
- `cuModuleGetGlobal` → global pointer
- `cuStreamCreate` → stream handle
- `cuEventCreate` → event handle

Everything else either consumes handles (pass VH, server translates) or has
no handle involvement at all.

## Tradeoffs

### What Virtual Handles Sacrifice

1. **Deferred error reporting** — If `cuLaunchKernel` fails, you won't know
   until the next flush/readback. In practice, CUDA errors are checked after
   synchronization anyway (`cuCtxSynchronize` or `cuStreamSynchronize`), so
   this matches real usage patterns.

2. **Not maximum theoretical pipelining** — `cuDeviceGetAttribute` is a
   barrier, but its value could theoretically be cached. However, the
   complexity of a caching system far outweighs the benefit of saving 1-2
   round-trips during initialization (which happens once).

3. **Session-scoped virtual handles** — VH IDs are only valid within a single
   pipeline batch or session. You cannot pass a VH from one connection to
   another. CUDA handles are already context-scoped, so this isn't a real
   limitation in practice.

### Why This Is the Right Approach

The design prioritizes **correctness over maximum pipelining**:

- The safe default is always `barrier` — if you can't categorize a function,
  it still works correctly
- Categorization is deterministic from function signatures — no dataflow
  analysis, no runtime dependency tracking, no speculation
- The performance win is real — 3-4 round-trips vs 10+ for typical workloads
- Complexity is bounded — ~30 handle producers, ~5 true barriers,
  everything else is trivially enqueued
