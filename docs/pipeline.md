# Dependency-Aware Pipeline

The pipeline is zlink's core mechanism for reducing network round-trips. It batches
CUDA API calls and sends them as a single network frame, only flushing when
correctness requires it.

## The Problem

CUDA workloads are sequential — `cuMemAlloc` returns a device pointer that
`cuMemcpyHtoD` needs, which feeds `cuLaunchKernel`, etc. A naive RPC would require
one round-trip per call, making GPU-over-IP unusably slow.

## The Solution: Three-Category Classification

Every CUDA call is classified by its dependency type:

| Category | Semantics | Pipeline Behavior | Examples |
|----------|-----------|-------------------|----------|
| **barrier** | Client MUST have return value before next call | Auto-flushes pipeline, then sends as sync RPC | cuInit, cuDeviceGetCount, cuDeviceGetAttribute |
| **enqueued** | Client doesn't need return value immediately | Batches into pipeline, processed in order | cuMemcpyHtoD, cuLaunchKernel, cuCtxSynchronize, cuMemFree |
| **readback** | Client needs data back from server after execution | Auto-flushes pipeline (ensures all prior work runs), then sends DtoH + read | cuMemcpyDtoH, cuMemcpyDtoHAsync |

### Implementation: `cuda_pipeline<RpcDef>`

Defined in `include/zlink/cuda_pipeline.hpp`. Template parameter `RpcDef` is the
zpp_bits RPC binding type (e.g., `cuda_test_rpc`).

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

## Auto-Flush Behavior

The pipeline tracks three queues:
- `calls_` — pending RPC calls (serialized zpp_bits requests)
- `syncs_` — pending host memory syncs (client data to mirror on server)
- `reads_` — pending read requests (server mirror data to read back)

When a `call_barrier` or `call_readback` is invoked, the pipeline automatically
flushes all pending items before processing the current call. This guarantees that
all prior work is executed on the server before the barrier/readback.

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
Request:
  [4B sync_count]
    for each sync: [8B addr][8B original_size][1B comp_flag][4B data_size][data...]
  [4B rpc_count]
    for each rpc: [4B len][rpc_bytes]
  [4B read_count]
    for each read: [8B addr][8B size]
  [handle_manifest: 4B count + N × handle_manifest_entry]

Response:
  [4B rpc_count]
    for each rpc: [4B resp_len][resp_bytes]
  [4B read_count]
    for each read: [4B data_len][1B comp_flag][data...]
```

The server processes this frame in order:
1. Apply all syncs (decompress if LZ4, write to host_memory_mirror)
2. Execute all RPC calls in sequence (register VH → real handle mappings)
3. Process read requests (read from mirror, compress if beneficial)
4. Process handle manifest (re-register VH IDs under their correct virtual IDs)
5. Send combined response

## Typical Workload Timeline

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
```

Without virtual handles, every handle-producing call would be a barrier:
10+ round-trips instead of 4.

## How to Add a New CUDA Function

Adding a new CUDA function to the pipeline requires three steps:

### 1. Declare the RPC function signature

In the shared API namespace (both client and server):

```cpp
struct MyFuncRet { int32_t result; std::uint64_t some_handle; };
MyFuncRet my_func(std::uint64_t arg1, int arg2);
```

### 2. Add zpp_bits binding

Append to the `zpp::bits::rpc<>` type with the next index number:

```cpp
using cuda_test_rpc = zpp::bits::rpc<
    // ... existing bindings ...
    zpp::bits::bind<&cuda_rpc_api::my_func, 20>,  // Next available index
>;
```

### 3. Write a wrapper function that declares the dependency type

```cpp
// If it produces a handle → enqueue_produces_handle
static uint64_t cu_my_func_vh(zlink::cuda_pipeline<cuda_test_rpc>& pipe, ...) {
    uint32_t vh = pipe.enqueue_produces_handle<20>(...);
    return zlink::make_virtual_handle(vh);
}

// If it only consumes handles → enqueue
static int32_t cu_my_func(zlink::cuda_pipeline<cuda_test_rpc>& pipe, uint64_t some_vh, ...) {
    pipe.enqueue<20>(some_vh, ...);
    return 0;
}

// If client needs the result immediately → call_barrier
static int32_t cu_my_func_barrier(zlink::cuda_pipeline<cuda_test_rpc>& pipe, ...) {
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
        auto typed = pipeline_result_get<cuda_test_rpc, 5, CtxSyncRet>(r);
        // Use typed.result
    }
}
```

For readback calls, `call_readback_with_sync_read` automatically copies data
to the host buffer during response parsing — no separate extraction needed.
