# CUDA Pipeline and Virtual Handles

This document describes zlink's dependency-aware CUDA RPC pipeline as
implemented in `cuda_pipeline.hpp`, `virtual_handle.hpp`, and
`cuda_dep_spec.hpp`. It covers the three-category call classification, the
virtual handle system that eliminates barriers on handle-producing calls, the
`pipeline_mem` wire format, and how to add new CUDA functions to the pipeline.

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

## The Solution: Three-Category Classification

Every CUDA call is classified by its dependency type:

| Category | Semantics | Pipeline Behavior | Examples |
|----------|-----------|-------------------|----------|
| **barrier** | Client MUST have return value before next call | Auto-flushes pipeline, then sends as sync RPC | cuInit, cuDeviceGetCount, cuDeviceGetAttribute |
| **enqueued** | Client doesn't need return value immediately | Batches into pipeline, processed in order | cuMemcpyHtoD, cuLaunchKernel, cuCtxSynchronize, cuMemFree |
| **readback** | Client needs data back from server after execution | Auto-flushes pipeline (ensures all prior work runs), then sends DtoH + read | cuMemcpyDtoH, cuMemcpyDtoHAsync |

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

## Implementation: `cuda_pipeline<RpcDef>`

Defined in `include/zlink/cuda_pipeline.hpp`. Template parameter `RpcDef` is the
zpp_bits RPC binding type (e.g., `cuda_gen_rpc`).

Key methods:

```cpp
// Barrier call — flushes pipeline, then sync RPC
template<auto FuncIndex, typename... Args>
auto call_barrier(Args&&... args);

// Enqueued call — batches, no immediate result
template<auto FuncIndex, typename... Args>
void enqueue(Args&&... args);

// Enqueued call with inline host_sync
template<auto FuncIndex, typename... Args>
void enqueue_with_sync(const void* host_ptr, std::size_t sync_size, Args&&... args);

// Enqueued call that produces a virtual handle
template<auto FuncIndex, typename... Args>
std::uint32_t enqueue_produces_handle(Args&&... args);

// Readback call — flushes + gets data back
template<auto FuncIndex, typename... Args>
auto call_readback(Args&&... args);

// Readback with inline sync + read (full cuMemcpyDtoH pattern)
template<auto FuncIndex, typename... Args>
auto call_readback_with_sync_read(void* host_ptr, std::size_t size, Args&&... args);

// Explicit flush — sends all pending calls as a batch
std::vector<pipe_result> flush();
```

### Auto-Flush Behavior

The pipeline tracks three queues:
- `calls_` — pending RPC calls (serialized zpp_bits requests)
- `syncs_` — pending host memory syncs (client data to mirror on server)
- `reads_` — pending read requests (server mirror data to read back)

When a `call_barrier` or `call_readback` is invoked, the pipeline automatically
flushes all pending items before processing the current call. This guarantees that
all prior work is executed on the server before the barrier/readback.

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

The client assigns sequential virtual handle IDs (0, 1, 2, ...) immediately
when enqueuing a handle-producing call. Subsequent calls reference these
virtual IDs. The server translates VH → real CUDA handle when processing the
pipeline batch.

### Encoding

Defined in `include/zlink/virtual_handle.hpp`:

```
Bit 63 set = virtual handle. Lower 63 bits = virtual ID.
Real CUDA pointers are always < 2^63 (user-space addresses), so this is unambiguous.

Example: VH(0) = 0x8000000000000000
         VH(1) = 0x8000000000000001
         Real ptr: 0x7f18c6c00000 (bit 63 clear)
```

Core functions:

```cpp
bool is_virtual_handle(std::uint64_t val);         // Check bit 63
std::uint64_t make_virtual_handle(std::uint32_t id); // Set bit 63
std::uint32_t virtual_handle_id(std::uint64_t val);  // Extract lower 63 bits
```

This encoding is safe because user-space addresses on Linux are always below
`0x7fffffffffff` (47-bit canonical addresses). Bit 63 never appears in real
pointers, so the encoding is unambiguous.

### Client-Side Flow

1. Client calls `enqueue_produces_handle<7>(256)` for `cuMemAlloc(256)`
2. The pipeline:
   - Allocates a virtual handle ID: `vh_id = vhandle_alloc_.allocate()` → 0
   - Enqueues the RPC call normally
   - Records a `handle_manifest_entry`: call_index, virtual_id, return_field
   - Returns `vh_id = 0`
3. Client wraps it: `make_virtual_handle(0)` → `0x8000000000000000`
4. Subsequent calls pass this VH value as an argument:
   `enqueue<9>(VH(0), client_addr, byte_count)` for `cuMemcpyHtoD`
5. At flush time, the handle manifest is serialized into the `pipeline_mem`
   frame alongside the sync data and RPC calls

```cpp
// In cuda_pipeline.hpp:
template<auto FuncIndex, typename... Args>
std::uint32_t enqueue_produces_handle(Args&&... args) {
    std::uint32_t vh_id = vhandle_alloc_.allocate();  // 0, 1, 2, ...
    enqueue<FuncIndex>(std::forward<Args>(args)...);

    handle_manifest_entry entry;
    entry.call_index = static_cast<std::uint32_t>(calls_.size() - 1);
    entry.virtual_id = vh_id;
    entry.return_field = 1;  // Convention: handle is field 1 (after result field 0)
    handle_manifest_.push_back(entry);

    return vh_id;
}
```

### Server-Side Flow

1. Server receives a `pipeline_mem` frame with:
   - Sync entries (host data to mirror)
   - RPC calls (serialized zpp_bits requests)
   - Read requests (addresses to read back)
   - Handle manifest (which calls produce handles)
2. Server parses the handle manifest **before** processing RPCs, building a
   `call_index → virtual_id` map.
3. Server processes calls sequentially:
   - For each call, check the handle manifest
   - After executing a handle-producing call, read `g_last_produced_handle`
     and register it under the call's `virtual_id` (from the manifest map)
   - For subsequent calls that consume handles, translate:
     `g_vhandles.translate(vh_value)` → real CUDA handle
4. After all calls are processed, handle the read requests

The manifest is parsed before RPCs (not after) so that handles are registered
under their correct `virtual_id` immediately during processing. This ensures
intra-batch dependencies resolve correctly even when `virtual_id != call_index`.

### Handle Manifest Wire Format

The handle manifest is serialized at the end of the `pipeline_mem` frame:

```cpp
struct handle_manifest_entry {
    std::uint32_t call_index;   // 0-indexed position in the pipeline
    std::uint32_t virtual_id;   // The virtual handle ID to assign
    std::uint32_t return_field; // Which field in the return struct is the handle
    std::uint32_t _reserved;    // Padding
};
```

```
[4 bytes: entry count]
[entry 0: call_index(4B) + virtual_id(4B) + return_field(4B) + reserved(4B)]
[entry 1: ...]
...
```

This tells the server exactly which pipeline calls produce handles and which
virtual IDs to assign, so it can register them in the `handle_table` after
executing each call.

### Handle Production Hook

Server-side functions that produce handles use `handle_producer_guard`:

```cpp
AllocRet mem_alloc(std::uint64_t bytesize) {
    handle_producer_guard guard;
    CUdeviceptr ptr = 0;
    CUresult r = cuMemAlloc(&ptr, bytesize);
    if (r == CUDA_SUCCESS) guard.set(ptr);  // Sets g_last_produced_handle
    return {static_cast<int32_t>(r), ptr};
}
```

The guard sets `g_has_produced_handle = true` on destruction, signaling to
the pipeline handler that this call produced a handle. The handler then reads
`g_last_produced_handle` and registers it in the handle table under the
correct `virtual_id`.

### Server-Side Handle Table

`handle_table` (defined in `virtual_handle.hpp`) maps virtual IDs to real CUDA
handle values on the server. Thread-safe with mutex protection.

```cpp
class handle_table {
    void register_handle(std::uint32_t virtual_id, std::uint64_t real_value);
    std::optional<std::uint64_t> lookup(std::uint32_t virtual_id) const;
    std::uint64_t translate(std::uint64_t val) const;  // VH → real, pass-through if not VH
};
```

The server's RPC wrapper functions call `g_vhandles.translate()` on every
handle-type parameter:

```cpp
// cuMemcpyHtoD — dst_dev_ptr is a virtual handle
CopyHtoDRet memcpy_htod(std::uint64_t dst_dev_ptr, ...) {
    auto real_dst = g_vhandles.translate(dst_dev_ptr);  // VH → real pointer
    CUresult r = cuMemcpyHtoD(static_cast<CUdeviceptr>(real_dst), ...);
    ...
}
```

## Pipeline Frame Types

### Simple Pipeline (`pipeline_request` / `pipeline_response`)

Used when there are no syncs, no reads, and no virtual handle manifest. Just
a batch of RPC calls.

```
Request:  [4B count][4B len1][req1_bytes][4B len2][req2_bytes]...
Response: [4B count][4B len1][resp1_bytes][4B len2][resp2_bytes]...
```

### Pipeline with Memory Ops (`pipeline_mem`)

The full-featured frame that combines everything into one round-trip. Used
automatically when there are pending syncs, reads, or handle manifests.

```
┌─────────────────────────────────────────────────────────────┐
│ Sync section:                                                │
│   [4B sync_count]                                            │
│   For each sync: [8B addr][8B original_size][1B comp_flag]   │
│                  [4B data_size][data bytes...]              │
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

The response is a `pipeline_response` frame:

```
┌─────────────────────────────────────────────────────────────┐
│ RPC results:                                                 │
│   [4B rpc_count]                                             │
│   For each rpc: [4B resp_len][resp_bytes]                    │
├─────────────────────────────────────────────────────────────┤
│ Read data:                                                   │
│   [4B read_count]                                            │
│   For each read: [4B data_len][1B comp_flag][data bytes...]  │
└─────────────────────────────────────────────────────────────┘
```

The server processes this frame in order:
1. Parse handle manifest (build `call_index → virtual_id` map)
2. Apply all syncs (decompress if LZ4, write to host_memory_mirror)
3. Execute all RPC calls in sequence (register VH → real handle under virtual_id)
4. Process read requests (read from mirror, compress if beneficial)
5. Send combined response

Read data is automatically written to the client's host buffers (registered
via `pending_read` entries) during response parsing.

## Typical Workload Pipeline

A typical CUDA workload with virtual handles:

```
cuInit(0)                     → barrier: flush (empty), sync RPC [1 RT]
cuDeviceGetCount()            → barrier: flush (empty), sync RPC [1 RT]
cuDeviceGetAttribute(...)     → barrier: flush (empty), sync RPC [1 RT]
cuCtxCreate(...)              → enqueued: VH(0) = ctx
cuMemAlloc(N)                 → enqueued: VH(1) = input_buf
cuMemAlloc(M)                 → enqueued: VH(2) = output_buf
cuModuleLoadData(...)         → enqueued: VH(3) = module (+ inline sync)
cuModuleGetFunction(VH(3),..) → enqueued: VH(4) = kernel_func
cuMemcpyHtoD(VH(1), ...)     → enqueued (+ inline sync)
cuLaunchKernel(VH(4), ...)   → enqueued
cuMemcpyDtoH(out, VH(2),...) → READBACK: flush! [1 RT for ALL above]

Total: 3 barrier RTs + 1 pipeline batch RT = 4 round-trips
Without virtual handles: every alloc/create is a barrier → 10+ round-trips
```

## How to Add a New CUDA Function

Adding a new CUDA function to the pipeline requires three steps:

### 1. Declare the RPC function signature

In the shared API header (both client and server include it):

```cpp
struct MyFuncRet { int32_t result; std::uint64_t some_handle; };
MyFuncRet my_func(std::uint64_t arg1, int arg2);
```

### 2. Add zpp_bits binding

Append to the `zpp::bits::rpc<>` type with the next index number:

```cpp
using cuda_gen_rpc = zpp::bits::rpc<
    // ... existing bindings ...
    zpp::bits::bind<&cuda_rpc_api::my_func, 20>,  // Next available index
>;
```

### 3. Write a wrapper function that declares the dependency type

```cpp
// If it produces a handle → enqueue_produces_handle
static uint64_t cu_my_func_vh(cuda_pipeline<cuda_gen_rpc>& pipe, ...) {
    uint32_t vh = pipe.enqueue_produces_handle<20>(...);
    return zlink::make_virtual_handle(vh);
}

// If it only consumes handles → enqueue
static int32_t cu_my_func(cuda_pipeline<cuda_gen_rpc>& pipe, uint64_t some_vh, ...) {
    pipe.enqueue<20>(some_vh, ...);
    return 0;
}

// If client needs the result immediately → call_barrier
static int32_t cu_my_func_barrier(cuda_pipeline<cuda_gen_rpc>& pipe, ...) {
    auto r = pipe.call_barrier<20>(...);
    return r.result;
}
```

### On the server side

Implement the actual function with `g_vhandles.translate()` for any handle
parameters, and use `handle_producer_guard` if the function produces a handle.

## Pipeline Result Handling

Enqueued calls produce deferred results accessible after `flush()`:

```cpp
auto results = pipe.flush();
for (auto& r : results) {
    if (r.valid) {
        auto typed = pipeline_result_get<cuda_gen_rpc, 5, CtxSyncRet>(r);
        // Use typed.result
    }
}
```

For readback calls, `call_readback_with_sync_read` automatically copies data
to the host buffer during response parsing — no separate extraction needed.

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

### Detailed Categorization

| Function | Category | Handle Role | Notes |
|----------|----------|-------------|-------|
| cuInit | barrier | — | First call, check availability |
| cuDriverGetVersion | barrier | — | Need version string |
| cuDeviceGet | barrier | — | Need device ordinal |
| cuDeviceGetCount | barrier | — | Need count for iteration |
| cuDeviceGetName | barrier | — | Need string back |
| cuDeviceTotalMem | barrier | — | Need value for alloc planning |
| cuDeviceGetAttribute | barrier | — | Need value for occupancy calc |
| cuMemGetInfo | barrier | — | Need free/total for alloc strategy |
| cuCtxCreate | enqueued | PRODUCES | Returns VH for context |
| cuCtxDestroy | enqueued | consumes | Takes VH context |
| cuCtxSetCurrent | enqueued | consumes | Takes VH context |
| cuCtxGetCurrent | barrier | PRODUCES | Need current ctx |
| cuCtxSynchronize | enqueued | — | Just a sync point |
| cuMemAlloc | enqueued | PRODUCES | Returns VH for dev_ptr |
| cuMemAllocManaged | enqueued | PRODUCES | Returns VH for unified mem ptr |
| cuMemFree | enqueued | consumes | Takes VH dev_ptr |
| cuMemcpyHtoD | enqueued | consumes | + inline host_sync |
| cuMemcpyDtoH | readback | consumes | + inline host_read |
| cuMemcpyDtoD | enqueued | consumes | Both dev_ptrs are VHs |
| cuModuleLoadData | enqueued | PRODUCES | + inline host_sync for image |
| cuModuleGetFunction | enqueued | PROD+CON | Consumes VH module, produces VH func |
| cuLaunchKernel | enqueued | consumes | VH func, VH stream, VH args |
| cuStreamCreate | enqueued | PRODUCES | Returns VH for stream |
| cuStreamDestroy | enqueued | consumes | Takes VH stream |
| cuStreamSynchronize | enqueued | consumes | Takes VH stream |
| cuEventCreate | enqueued | PRODUCES | Returns VH for event |
| cuEventDestroy | enqueued | consumes | Takes VH event |
| cuEventRecord | enqueued | consumes | VH event + VH stream |
| cuEventSynchronize | enqueued | consumes | Takes VH event |
| cuEventElapsedTime | barrier | consumes | Need float value back |

### Key Insight

With virtual handles, the only barriers are functions where the client
needs a return value to make a decision:

- `cuInit` — need to check if CUDA is available
- `cuDriverGetVersion` — need version string
- `cuDeviceGetCount` — need count to decide which GPU to use
- `cuDeviceGetName` — need name string
- `cuDeviceTotalMem` — need value for alloc planning
- `cuDeviceGetAttribute` — need value for occupancy calculations
- `cuMemGetInfo` — need free/total for alloc strategy
- `cuEventElapsedTime` — need the float value for profiling

Everything else pipelines. Even error codes are deferred. In a typical
workload, the barriers are all front-loaded during setup; the hot loop
is entirely enqueued + one readback.

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
