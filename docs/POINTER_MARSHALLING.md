# zlink Pointer Marshalling

This document describes how zlink handles pointer arguments across the
network boundary — the hardest problem in GPU-over-IP.

## The Core Problem

When an RPC server executes a function on behalf of a remote client, any
pointer arguments in that function refer to the **client's** address space.
The server cannot dereference them. This is the fundamental challenge in
any "GPU over IP" or "shared library over network" system.

## Previous Approaches

### Lupine (per-function codegen)

Lupine's approach:
1. Parse CUDA header files with a Python script
2. For each function, generate code that knows which params are pointers
3. For input pointers: serialize the pointed-to data into the RPC message
4. For output pointers: deserialize data from the response into the buffer
5. For inout pointers: serialize before call, deserialize after

**Problems:**
- Requires codegen for every new library/version
- Can't handle arbitrary shared libraries
- Deep pointer chasing (pointer-to-pointer) is hard to codegen
- Each function needs custom marshalling logic

### RCUDA / SCUDA (similar codegen)

Same fundamental approach as Lupine with the same limitations.

## zlink's Current Approach: Inline Host Sync in Pipeline Frames

zlink currently handles pointer marshalling by packing host data inline into
`pipeline_mem` frames. This is a simpler and more performant approach than
on-demand fetching for the CUDA use case.

### How It Works Today

Instead of making the server fetch client data on demand (which requires
extra round-trips), zlink packs the host data directly into the pipeline
frame alongside the RPC calls:

```
CLIENT                                            SERVER
┌────────────────────────┐                        ┌────────────────────────┐
│ cuMemcpyHtoD(VH(1),    │                        │ Receives pipeline_mem   │
│   host_data, 256)      │                        │ frame:                 │
│                        │                        │                        │
│ enqueue_with_sync<9>(  │                        │ 1. Apply sync entries  │
│   host_ptr, 256,       │                        │    → write to mirror   │
│   VH(1), addr, 256)    │                        │    → client addr now   │
│                        │                        │      has server data   │
│ Pipeline batches:      │                        │                        │
│  sync: [addr=0x7fff,   │  pipeline_mem frame    │ 2. Process RPC calls   │
│         size=256,      │ ══════════════════►    │    translate VH→real   │
│         data=...]      │                        │    translate client    │
│  rpc:  [cuMemcpyHtoD  │                        │      ptr→server ptr    │
│         (VH(1),       │                        │    call real cuMemcpy-  │
│          0x7fff, 256)] │                        │      HtoD()            │
│  manifest: [VH(1)←call0]│                       │                        │
└────────────────────────┘                        └────────────────────────┘
```

### Step-by-Step: cuMemcpyHtoD with Inline Sync

1. **Client** calls `cu_memcpy_htod(pipe, dev_vh, host_data, byte_count)`
2. This calls `pipe.enqueue_with_sync<9>(host_data, byte_count, dev_vh, client_addr, byte_count)`
3. The pipeline:
   - Queues a `pending_sync` with the host data copied from `host_data`
   - Queues the RPC call `memcpy_htod(dev_vh, client_addr, byte_count)`
4. When the pipeline flushes (at the next readback or explicit flush):
   - Builds a `pipeline_mem` frame with sync entries, RPC calls, and handle
     manifest
   - Sends it in a single network round-trip
5. **Server** receives the frame:
   - Processes sync entries: `g_host_mirror.sync_page(client_addr, data)`
     → stores data in server's mmap'd mirror region
   - Processes RPC calls:
     - Translates `dev_vh` → real device pointer via `g_vhandles.translate()`
     - Translates `client_addr` → server-local address via `g_host_mirror.translate()`
     - Calls real `cuMemcpyHtoD(real_dev_ptr, server_local_ptr, byte_count)`
6. CUDA reads from the server-local pointer, which has the same data as
   the client's original buffer

### Step-by-Step: cuMemcpyDtoH with Inline Read

1. **Client** calls `cu_memcpy_dtoh(pipe, host_data, dev_vh, byte_count)`
2. This calls `pipe.call_readback_with_sync_read<10>(host_data, byte_count, client_addr, dev_vh, byte_count)`
3. The pipeline:
   - Queues a `pending_sync` (zero-fill, to create the mirror region)
   - Queues a `pending_read` (requesting the data back after execution)
   - Queues the RPC call
   - Immediately flushes (readback forces flush)
4. **Server** receives the frame:
   - Processes sync entries (creates mirror region)
   - Processes RPC call:
     - Translates VH and client address
     - Calls real `cuMemcpyDtoH(server_local_ptr, real_dev_ptr, byte_count)`
     - GPU data is now in the server's mirror region
   - Processes read requests: reads data from mirror region
5. **Server** sends response with RPC results + read data
6. **Client** parses response: read data is automatically copied to
   `host_data` buffer

### The host_memory_mirror Class

The `host_memory_mirror` (in `memory.hpp`) is the server-side component that
stores synced client data:

- `register_region(client_base, size)` — Allocates a server-local mmap'd
  region to mirror the client's address range
- `sync_page(client_addr, data)` — Writes client data into the mirror region
- `translate(client_addr)` — Returns the server-local address corresponding
  to a client address, or `nullopt` if not registered

The mirror region uses `mmap(MAP_ANONYMOUS | MAP_PRIVATE)`, so it's
allocated from virtual memory without needing a file backing.

### Auto-Registering on Sync

The `sync_page()` method in `memory_region.cpp` automatically registers a
new region if the client address isn't already in a known region:

```cpp
std::error_code host_memory_mirror::sync_page(std::uintptr_t client_addr,
                                               std::span<const std::byte> data) {
    auto it = std::find_if(regions_.begin(), regions_.end(),
        [&](const region& r) {
            return client_addr >= r.client_base &&
                   client_addr < r.client_base + r.size;
        });

    if (it == regions_.end()) {
        // Auto-register: allocate a new mirror region for this address
        register_region(client_addr & ~0xFFFULL, 65536);  // 64K aligned
    }

    // Now write the data into the mirror
    // ...
}
```

This means the client doesn't need to explicitly register regions before
syncing — the first `host_sync` for an address range automatically creates
the mirror.

## Virtual Handles for Device Pointers

Device pointers are handled differently from host pointers. They use the
virtual handle system (see [CUDA_PIPELINE.md](CUDA_PIPELINE.md) for full
details):

- `cuMemAlloc` returns a virtual handle (VH) instead of a real device
  pointer
- Subsequent calls pass VH values as arguments
- The server translates VH → real handle via `handle_table::translate()`
- The encoding is safe: bit 63 is set for virtual handles, which never
  appears in real user-space pointers

## Kernel Launch: Deep Pointer Handling

`cuLaunchKernel` is the hardest case because `kernelParams` is an array of
`void*` pointers, each of which may point to device memory. In the current
implementation (as seen in `cuda_test_server.cpp`):

```cpp
// Server-side kernel launch handler:
case 13: {  // launch_kernel
    // ...
    // Translate the function handle
    real_func = (CUfunction)g_vhandles.translate(func_vh);

    // The args were sent via host_sync, already in the mirror region
    // Translate the args client address to server-local
    auto args_server = g_host_mirror.translate(args_client_addr);

    // Each arg in kernelParams may be a device pointer (VH)
    // We need to translate them
    void* kernel_params[MAX_KERNEL_PARAMS];
    for (int i = 0; i < n_args; i++) {
        uint64_t* arg_ptr = (uint64_t*)(args_server_local + i * 8);
        uint64_t arg_val = *arg_ptr;
        if (zlink::is_virtual_handle(arg_val)) {
            *arg_ptr = g_vhandles.translate(arg_val);
        }
    }

    cuLaunchKernel(real_func, ...);
}
```

This approach walks the kernel args array on the server side, translating
any virtual handles found in the argument slots. It works because:

1. The client sends the args data via `host_sync` (inline in pipeline_mem)
2. The server mirrors the data at a known server-local address
3. The server walks the args, translating VH values to real device pointers
4. CUDA receives real pointers and the kernel runs correctly

## Comparison: Current vs Planned

| Aspect | Current (inline sync) | Planned (on-demand) |
|--------|----------------------|---------------------|
| Input pointers | Inline in pipeline_mem frame | Auto-fetch via ReadAt |
| Output pointers | Inline read in pipeline_mem response | Auto-writeback via WriteAt |
| Extra round-trips | None (data is inline) | 1 RTT per on-demand fetch |
| Code complexity | Simple (pack/unpack) | Complex (caching, coherence) |
| Best for | LAN, moderate data sizes | WAN, sparse access patterns |
| Implemented | Yes | Partially (framework exists) |

The inline approach is currently used because it gives the best performance
for the common CUDA workload pattern (batch HtoD → compute → DtoH) with
zero extra round-trips. The on-demand approach via `shared_mem_plane` and
`chunk_cache` is implemented as a framework but not yet the default path
for CUDA operations.

## Future: On-Demand Access via Shared Memory Plane

The `shared_mem_plane` (in `shared_mem.hpp`) and `chunk_cache` (in
`chunk_cache.hpp`) provide the infrastructure for on-demand access:

1. **`backend` interface** — `ReadAt/WriteAt/Size/Sync`, matching r3map's Go
   Backend
2. **`memory_backend`** — Wraps local memory as a backend
3. **`rpc_backend`** — Proxies ReadAt/WriteAt over the RPC transport
4. **`shared_mem_plane`** — Coordinates client↔server memory access with
   lazy fetching and background push/pull

When enabled, the server would fetch client data on-demand rather than
requiring it to be pre-synced. This is beneficial for:

- WAN deployments where pre-syncing large buffers is expensive
- Sparse access patterns where only a few pages are actually needed
- Workloads with repeated access to the same pages (caching benefit)

The framework is in place; what remains is integrating it as an alternative
to the inline sync path for CUDA operations.
