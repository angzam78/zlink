# Running PyTorch Over zlink

> **Status: Not yet tested.** This document describes the planned approach
> for running PyTorch workloads (like Stable Diffusion 1.5) over zlink.
> It will be updated after actual testing.

## Goal

Run a PyTorch script on a CPU-only machine, transparently using a remote GPU
via zlink's CUDA RPC. The script should run unmodified — zlink intercepts
CUDA calls and proxies them to the server.

## Approach: LD_PRELOAD Shim

The plan is to use an LD_PRELOAD shim that intercepts CUDA Driver API calls
from PyTorch and redirects them through zlink's pipeline.

```
┌───────────────────────────────────────┐
│  Python / PyTorch                     │
│      ↓                                │
│  libcuda.so (intercepted by shim)     │
│      ↓                                │
│  zlink shim (LD_PRELOAD)              │
│      ↓                                │
│  cuda_pipeline → transport → TCP      │
└───────────────────────────────────────┘
                ↕ TCP
┌───────────────────────────────────────┐
│  zlink CUDA server                    │
│      ↓                                │
│  Real libcuda.so                      │
│      ↓                                │
│  GPU                                  │
└───────────────────────────────────────┘
```

### Why LD_PRELOAD instead of PyTorch extension?

1. **No PyTorch modification needed** — The shim works with any PyTorch version.
2. **Full API coverage** — Intercept at the CUDA Driver API level (cuMemAlloc,
   cuLaunchKernel, etc.) which is lower than PyTorch's Python/C++ API.
3. **Transparent** — The Python script doesn't know it's using a remote GPU.

## Current API Coverage

The zlink CUDA server currently implements these functions (bound indices 0-19):

| Index | Function | Category | Status |
|-------|----------|----------|--------|
| 0 | cuInit | barrier | ✅ Working |
| 1 | cuDeviceGetCount | barrier | ✅ Working |
| 2 | cuDeviceGetName | barrier | ✅ Working |
| 3 | cuDeviceTotalMem | barrier | ✅ Working |
| 4 | cuCtxCreate | enqueued (VH produces) | ✅ Working |
| 5 | cuCtxSynchronize | enqueued | ✅ Working |
| 6 | cuCtxDestroy | enqueued (VH consumes) | ✅ Working |
| 7 | cuMemAlloc | enqueued (VH produces) | ✅ Working |
| 8 | cuMemFree | enqueued (VH consumes) | ✅ Working |
| 9 | cuMemcpyHtoD | enqueued (VH consumes + sync) | ✅ Working |
| 10 | cuMemcpyDtoH | readback (VH consumes + read) | ✅ Working |
| 11 | cuModuleLoadData | enqueued (VH produces + sync) | ✅ Working |
| 12 | cuModuleGetFunction | enqueued (VH prod+con) | ✅ Working |
| 13 | cuLaunchKernel | enqueued (VH consumes) | ✅ Working |
| 14 | cuStreamCreate | enqueued (VH produces) | ✅ Working |
| 15 | cuStreamSynchronize | enqueued (VH consumes) | ✅ Working |
| 16 | cuStreamDestroy | enqueued (VH consumes) | ✅ Working |
| 17 | cuEventCreate | enqueued (VH produces) | ✅ Working |
| 18 | cuEventRecord | enqueued (VH consumes) | ✅ Working |
| 19 | cuEventSynchronize | enqueued (VH consumes) | ✅ Working |

## Functions Needed for PyTorch

PyTorch's CUDA backend uses additional functions not yet in zlink:

### High Priority (needed for basic inference)

| Function | Category | Notes |
|----------|----------|-------|
| cuDeviceGet | barrier | Get device by ordinal |
| cuDeviceGetAttribute | barrier | Query device capabilities |
| cuDevicePrimaryCtxRetain | enqueued (VH prod) | PyTorch uses primary context |
| cuDevicePrimaryCtxRelease | enqueued (VH con) | Release primary context |
| cuMemAllocManaged | enqueued (VH prod) | Managed memory |
| cuMemHostAlloc | enqueued (VH prod) | Pinned host memory |
| cuMemHostFree | enqueued (VH con) | Free pinned memory |
| cuMemHostRegister | enqueued (VH con) | Register host memory |
| cuMemHostUnregister | enqueued (VH con) | Unregister host memory |
| cuMemcpyHtoDAsync | enqueued (VH con + sync) | Async HtoD with stream |
| cuMemcpyDtoHAsync | readback (VH con + read) | Async DtoH with stream |
| cuMemcpyDtoD | enqueued (VH con) | Device-to-device copy |
| cuModuleLoad | enqueued (VH prod) | Load from filename |
| cuModuleUnload | enqueued (VH con) | Unload module |
| cuMemGetInfo | barrier | Free/total memory query |
| cuGetProcAddress | enqueued (VH prod) | Get kernel by name (newer API) |

### Medium Priority (needed for training / advanced features)

| Function | Category | Notes |
|----------|----------|-------|
| cuOccupancyMaxPotentialBlockSize | barrier | Launch configuration |
| cuEventDestroy | enqueued (VH con) | Destroy event |
| cuEventElapsedTime | barrier | Need float value |
| cuStreamCreateWithPriority | enqueued (VH prod) | Priority streams |
| cuCtxPushCurrent | enqueued (VH con) | Context stack |
| cuCtxPopCurrent | enqueued (VH prod) | Context stack |
| cuCtxSetCurrent | enqueued (VH con) | Set active context |
| cuCtxGetCurrent | barrier (VH prod) | Get current context |
| cuArrayCreate | enqueued (VH prod) | Surface/texture memory |
| cuArrayDestroy | enqueued (VH con) | Destroy array |
| cuTexObjectCreate | enqueued (VH prod) | Texture objects |
| cuTexObjectDestroy | enqueued (VH con) | Destroy texture |
| cuSurfObjectCreate | enqueued (VH prod) | Surface objects |
| cuSurfObjectDestroy | enqueued (VH con) | Destroy surface |

### Lower Priority (optimization / rare use)

| Function | Category | Notes |
|----------|----------|-------|
| cuMemPrefetchAsync | enqueued (VH con) | Managed memory prefetch |
| cuMemAdvise | enqueued (VH con) | Managed memory advice |
| cuStreamWaitEvent | enqueued (VH con) | Stream-event synchronization |
| cuStreamAddCallback | barrier | Host callback from GPU |
| cuStreamQuery | barrier | Check stream status |
| cuEventQuery | barrier | Check event status |
| cuGraphCreate / cuGraphLaunch | enqueued (VH prod/con) | CUDA graphs |

## Implementation Plan

### Phase 1: Shim Skeleton

1. Create `zlink_shim.so` — LD_PRELOAD library that intercepts `cuInit`,
   `cuMemAlloc`, `cuMemcpyHtoD`, etc.
2. Each intercepted function creates a `cuda_pipeline` call instead of calling
   the real CUDA driver.
3. Test with a simple PyTorch script that just does tensor allocation and copy.

### Phase 2: Expand API Coverage

4. Add the high-priority functions listed above.
5. Handle `cuDevicePrimaryCtxRetain` — PyTorch uses this instead of `cuCtxCreate`.
6. Handle `cuGetProcAddress` — newer PyTorch versions use this for kernel lookup.

### Phase 3: Stable Diffusion Test

7. Install diffusers + transformers on the CPU-only client.
8. Run a Stable Diffusion 1.5 inference pipeline.
9. Measure round-trip count, latency, and compare with local GPU execution.

### Phase 4: Optimization

10. Profile the workload — identify which calls are barriers and whether they
    can be converted to enqueued with VH.
11. Tune compression thresholds based on actual tensor sizes.
12. Consider batched kernel launches for multi-step diffusion.

## Expected Performance

For Stable Diffusion 1.5 inference (~50 UNet steps):

- **Model loading** (one-time): ~2 GB of FP16 weights via cuMemcpyHtoD.
  With LZ4 compression at ~10x ratio, ~200 MB over the wire.
  On 1 Gbps: ~1.6 seconds. On 10 Gbps: ~160 ms.

- **Each UNet step**: ~4 attention operations + ~4 conv operations.
  Each involves cuMemcpyHtoD (latent), cuLaunchKernel, cuMemcpyDtoH (output).
  With pipelining: 1-2 round-trips per step.
  At ~0.5 ms per RT on LAN: ~1 ms overhead per step.

- **Total inference**: ~50 steps × ~1 ms overhead + model loading.
  The dominant cost should be the actual GPU compute, not the RPC overhead.

## Testing Checklist

When testing PyTorch over zlink, verify:

- [ ] `import torch; torch.cuda.is_available()` returns True
- [ ] `torch.cuda.device_count()` returns the correct count
- [ ] `torch.cuda.current_device()` works
- [ ] Tensor allocation works: `torch.zeros(1024, device='cuda')`
- [ ] Host-to-device copy works: `torch.tensor([1,2,3]).cuda()`
- [ ] Device-to-host copy works: `torch.zeros(1024, device='cuda').cpu()`
- [ ] Kernel launch works: `torch.nn.functional.linear(...)`
- [ ] Stream creation and synchronization works
- [ ] Event creation and elapsed time works
- [ ] Stable Diffusion 1.5 inference produces correct images
- [ ] Memory is properly freed (no leaks over multiple inferences)
