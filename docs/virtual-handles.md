# Virtual Handle System

Virtual handles are zlink's key innovation for eliminating pipeline barriers.
They allow handle-producing CUDA calls (cuMemAlloc, cuCtxCreate, etc.) to be
enqueued instead of requiring synchronous round-trips.

## The Problem

CUDA workloads have sequential handle dependencies:

```
cuMemAlloc → dev_ptr → cuMemcpyHtoD(dev_ptr, ...) → cuLaunchKernel(dev_ptr, ...)
```

Without virtual handles, `cuMemAlloc` must be a **barrier** — the client waits
for the real GPU pointer before it can issue `cuMemcpyHtoD`. This defeats
pipelining because every handle-producing call costs a full round-trip.

## The Solution

Virtual handles break the dependency chain. The client assigns sequential virtual
handle IDs (0, 1, 2, ...) immediately when enqueuing a handle-producing call.
Subsequent calls reference these virtual IDs. The server translates VH → real
CUDA handle when processing the pipeline batch.

```
cuMemAlloc       → VH(0)     // enqueued, returns immediately!
cuMemcpyHtoD(VH(0), ...)    // enqueued, references VH(0)
cuLaunchKernel(VH(0), ...)  // enqueued, references VH(0)
cuMemcpyDtoH(..., VH(0), ...) // readback: flush + get data
→ 1 round-trip instead of N
```

## Encoding

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

## Client Side: Virtual Handle Allocator

`virtual_handle_allocator` (defined in `virtual_handle.hpp`) assigns sequential
IDs starting from 0. Used by `cuda_pipeline::enqueue_produces_handle()`:

```cpp
template<auto FuncIndex, typename... Args>
std::uint32_t enqueue_produces_handle(Args&&... args) {
    std::uint32_t vh_id = vhandle_alloc_.allocate();  // 0, 1, 2, ...
    enqueue<FuncIndex>(std::forward<Args>(args)...);

    // Record that this call produces a handle
    handle_manifest_entry entry;
    entry.call_index = static_cast<std::uint32_t>(calls_.size() - 1);
    entry.virtual_id = vh_id;
    entry.return_field = 1;  // Convention: handle is field 1 (after result field 0)
    handle_manifest_.push_back(entry);

    return vh_id;
}
```

The client wrapper then wraps the VH ID with `make_virtual_handle()`:

```cpp
static uint64_t cu_mem_alloc_vh(cuda_pipeline<cuda_test_rpc>& pipe, uint64_t size) {
    uint32_t vh = pipe.enqueue_produces_handle<7>(size);
    return zlink::make_virtual_handle(vh);
}
```

## Server Side: Handle Table

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

## Handle Manifest

The handle manifest tells the server which pipeline calls produce handles and
which virtual IDs to assign. It's serialized into the pipeline_mem frame.

```cpp
struct handle_manifest_entry {
    std::uint32_t call_index;   // 0-indexed position in the pipeline
    std::uint32_t virtual_id;   // The virtual handle ID to assign
    std::uint32_t return_field; // Which field in the return struct is the handle
    std::uint32_t _reserved;    // Padding
};
```

### Server Processing Flow

1. **During RPC processing**: The server uses `handle_producer_guard` to detect
   when a call produces a handle. It temporarily registers the handle under the
   call's index.

2. **After all RPCs**: The server processes the handle manifest. For each entry,
   it looks up the handle registered under `call_index` and re-registers it
   under `virtual_id`.

```cpp
for (const auto& entry : manifest) {
    auto real_handle = g_vhandles.lookup(entry.call_index);
    if (real_handle) {
        g_vhandles.register_handle(entry.virtual_id, *real_handle);
    }
}
```

This two-step process avoids needing to parse zpp_bits varint-encoded responses
to extract handle values.

## Handle Production Hook

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
the pipeline handler that this call produced a handle.

## CUDA API Categorization

Defined in `include/zlink/cuda_dep_spec.hpp`. The full categorization for
PyTorch-relevant CUDA Driver API functions:

| Function | Category | Handle Role | Notes |
|----------|----------|-------------|-------|
| cuInit | barrier | — | First call, check availability |
| cuDeviceGet | barrier | — | Need device ordinal |
| cuDeviceGetCount | barrier | — | Need count for iteration |
| cuDeviceGetName | barrier | — | Need string back |
| cuDeviceTotalMem | barrier | — | Need value for alloc planning |
| cuDeviceGetAttribute | barrier | — | Need value for occupancy calc |
| cuCtxCreate | enqueued | PRODUCES | Returns VH for context |
| cuCtxDestroy | enqueued | consumes | Takes VH context |
| cuCtxSynchronize | enqueued | — | Just a sync point |
| cuMemAlloc | enqueued | PRODUCES | Returns VH for dev_ptr ★ |
| cuMemFree | enqueued | consumes | Takes VH dev_ptr |
| cuMemcpyHtoD | enqueued | consumes | + inline host_sync |
| cuMemcpyDtoH | readback | consumes | + inline host_read |
| cuModuleLoadData | enqueued | PRODUCES | + inline host_sync for image |
| cuModuleGetFunction | enqueued | PROD+CON | Consumes VH module, produces VH func |
| cuLaunchKernel | enqueued | consumes | VH func, VH stream, VH args |
| cuStreamCreate | enqueued | PRODUCES | Returns VH for stream |
| cuStreamSynchronize | enqueued | consumes | Takes VH stream |
| cuEventCreate | enqueued | PRODUCES | Returns VH for event |
| cuEventRecord | enqueued | consumes | VH event + VH stream |
| cuEventElapsedTime | barrier | consumes | Need float value back |

### Key Insight

With virtual handles, the ONLY true barriers are:
- `cuInit` — need to check if CUDA is available
- `cuDeviceGetCount` — need count to decide which GPU to use
- `cuDeviceGetAttribute` — need value for occupancy calculations
- `cuEventElapsedTime` — need the float value for profiling

Everything else pipelines. Even error codes are deferred.

### Adding a New Handle-Producing Function

Only ~30 out of ~2000 CUDA API functions produce handles. The pattern is
obvious from the function signature: any function whose return struct has a
`uint64_t` handle field produces a handle. For these, use
`enqueue_produces_handle` on the client and `handle_producer_guard` on the server.

For the vast majority of the API (handle consumers), just use `enqueue` on the
client and `g_vhandles.translate()` on the server for each handle parameter.
