# zlink Pointer Marshalling Design

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

## zlink's Approach: Shared Memory Plane (r3map-inspired)

Instead of per-function marshalling, we make the **client's entire address
space accessible** to the server through a shared memory plane, inspired by
r3map's architecture.

### How r3map Works

r3map provides:

1. **Backend interface**: `ReadAt(offset, buf) / WriteAt(offset, buf) / Size() / Sync()`
   - Any resource can implement this: memory, file, S3, Redis, etc.

2. **NBD (Network Block Device)**: The backend is exposed as a Linux block
   device (/dev/nbd0). The kernel handles page faults on mmap'd regions
   of this device by calling ReadAt/WriteAt on the backend.

3. **Slice frontend**: `mmap` the NBD device to get a `[]byte` that
   transparently fetches chunks from the backend on-demand.

4. **Managed mounts**: Background push/pull for WAN optimization.
   Pre-fetches chunks before they're needed; writes back dirty pages
   periodically instead of synchronously.

The result: **remote memory appears as local memory**. Accessing
`remote_slice[1000:2000]` transparently fetches that range from the
backend, which may be on another machine.

### zlink's Adaptation

We adapt r3map's architecture for the RPC use case:

```
CLIENT                                  SERVER
┌─────────────────────────┐             ┌─────────────────────────┐
│ Application              │             │ Real .so function        │
│   │                      │             │   │                      │
│   ▼                      │             │   ▼                      │
│ Shim intercepts call     │             │ translate_to_server()    │
│   │                      │             │   │                      │
│   ▼                      │             │   ▼                      │
│ register_host_memory()   │             │ Server-local cache       │
│   │                      │             │ (fetched on-demand       │
│   ▼                      │             │  from client via         │
│ shared_mem_plane         │  Backend    │  ReadAt/WriteAt)         │
│   ├─ memory_backend ─────┼═══════════► │   ├─ rpc_backend         │
│   │  (wraps real mem)    │  over RPC   │   │  (proxies to client)  │
│   │                      │             │   │                      │
│   ▼                      │             │   ▼                      │
│ RPC call with ptr+region │──────────► │ Real cuMemcpyHtoD()      │
│                          │             │   reads from local cache │
│                          │             │   = same data as client  │
└─────────────────────────┘             └─────────────────────────┘
```

### Step-by-Step: cuMemcpyHtoD(dst, srcHost=0x7fff1234, 1024)

1. **Client shim** intercepts `cuMemcpyHtoD(dst, 0x7fff1234, 1024)`
2. **Client shim** calls `register_host_memory(0x7fff1234, 1024, read_only)`
   - Creates a `memory_backend` wrapping bytes at 0x7fff1234
   - Registers it with the shared_mem_plane
   - Returns region_id = 42
3. **Client shim** sends RPC call with the pointer and region_id
4. **Server** receives the call, sees srcHost=0x7fff1234, region_id=42
5. **Server** calls `translate_to_server(0x7fff1234, 1024, 42)`
   - Checks if chunk is already in local cache
   - If not, calls `backend->read_at(offset=0, buf[1024])`
   - The backend is an `rpc_backend` that sends a ReadAt request
     back to the client over the existing transport
   - Client's `memory_backend::read_at()` reads from real memory
   - Data is copied into the server's local cache
6. **Server** gets back a server-local pointer, e.g., 0x80004000
7. **Server** calls `::cuMemcpyHtoD(remote_dst, 0x80004000, 1024)`
   - The real CUDA function reads from the server-local cache
   - The data is identical to what the client had at 0x7fff1234
8. Function succeeds! No per-function marshalling code needed.

### Step-by-Step: cuMemcpyDtoH(dstHost=0x7fff5678, src, 1024)

1. **Client shim** intercepts `cuMemcpyDtoH(0x7fff5678, src, 1024)`
2. **Client shim** registers the output buffer: `register_host_memory(0x7fff5678, 1024, writable=true)`
3. **Client shim** sends RPC call
4. **Server** calls `translate_to_server(0x7fff5678, 1024, 43)`
   - Fetches the current contents (probably uninitialized/zero)
   - Returns a server-local pointer
5. **Server** calls `::cuMemcpyDtoH(local_ptr, remote_src, 1024)`
   - CUDA writes into the server-local cache
6. **Server** calls `mark_dirty(local_ptr, 1024)`
7. **Server** calls `flush_dirty()`
   - For each dirty chunk, calls `backend->write_at(offset, data)`
   - The `rpc_backend` sends the data back to the client
   - Client's `memory_backend::write_at()` copies into the real buffer
8. Client's memory at 0x7fff5678 now contains the GPU data!

### Step-by-Step: cuLaunchKernel(f, ..., kernelParams, ...)

This is the hardest case: `kernelParams` is an `array of void* pointers`,
each of which may point to host or device memory. This requires **deep
pointer chasing**:

1. **Client shim** intercepts `cuLaunchKernel(f, ..., params, ...)`
2. **Client shim** registers the params array itself: `register_host_memory(params, N*sizeof(void*))`
3. **Client shim** iterates over each param pointer:
   - If it's a device pointer (found in ptr_map): no action needed
   - If it's a host pointer: register the pointed-to data as a region
4. **Client shim** sends RPC call with all region IDs
5. **Server** translates the params array:
   - `translate_to_server(params_addr, ...)` gets server-local params array
   - For each pointer in the array:
     - If device: translate via ptr_map
     - If host: translate via shared_mem_plane
6. **Server** calls `::cuLaunchKernel(f, ..., translated_params, ...)`
7. After the kernel completes, flush dirty regions back to client

## Comparison of Approaches

| Aspect | Lupine (codegen) | zlink V1 (no marshalling) | zlink V2 (shared mem plane) |
|--------|-------------------|---------------------------|-----------------------------|
| Input pointers | Serialize in RPC msg | Broken (raw ptr) | Auto-fetch via ReadAt |
| Output pointers | Deserialize from RPC msg | Broken (raw ptr) | Auto-writeback via WriteAt |
| Inout pointers | Serialize + deserialize | Broken | Fetch + mark_dirty + flush |
| String args | Serialize as bytes | Broken | Register as read-only region |
| Deep pointers | Custom per-function | Broken | Recursive registration |
| Arbitrary .so support | No (CUDA only) | No (broken ptrs) | Yes |
| Codegen required | Yes | No | No |
| Latency (small) | ~0.2ms | N/A | ~0.2ms (1 RTT for fetch) |
| Latency (first access) | N/A | N/A | ~0.3ms (fetch on demand) |
| Latency (cached) | N/A | N/A | ~0us (already local) |
| WAN support | Poor (sync) | N/A | Good (bg push/pull) |

## Optimization: Bulk Transfer vs On-Demand

For large transfers (cuMemcpyHtoD/DtoH with megabytes of data),
on-demand chunk fetching would be slow (1 RTT per 4K chunk).
Instead, we use **explicit bulk transfer**:

```cpp
// Client side: for large HtoD copies, push all data immediately
g_mem_plane.push(region_id, 0, span_of_all_data);

// Server side: data is already in cache, translate returns immediately
auto* local = g_mem_plane.translate_to_server(addr, size, region_id);
// local points to already-cached data, no on-demand fetch needed
```

This gives the same performance as explicit serialization for large
transfers, while keeping the on-demand model for small/infrequent accesses.

## Optimization: Background Push/Pull (r3map Managed Mounts)

For WAN deployments where RTT is high, r3map's managed mount API
pre-fetches chunks before they're needed:

```cpp
// Start prefetching from the beginning of the region
g_mem_plane.start_prefetch(region_id, priority_offset=0);

// Start periodic writeback of dirty pages
g_mem_plane.start_background_writeback(region_id, interval_ms=100);
```

This hides latency by overlapping data transfer with computation.

## Future: NBD Kernel Module Integration

The current implementation uses user-space caching. For even better
performance, we could use r3map's NBD approach directly:

1. Client exposes its memory as an NBD server (via go-nbd or a C++ NBD impl)
2. Server connects via /dev/nbdX
3. Server mmaps the NBD device
4. Kernel handles page faults automatically
5. Zero user-space overhead for cached pages

This would eliminate the user-space caching overhead entirely,
at the cost of requiring the nbd kernel module and root access.

## Protocol Extension: Region Metadata in RPC Frames

The current RPC protocol needs a small extension to carry region
metadata alongside pointer arguments:

```
┌─────────────┬──────────────────────┬──────────────────────────┐
│ RPC payload │ Region table (new!)  │ Region data (optional)   │
│ (zpp_bits)  │ [id, addr, size, rw] │ [bulk push data]         │
└─────────────┴──────────────────────┴──────────────────────────┘
```

The region table tells the server which memory regions the client
has registered, so it can set up the rpc_backend for on-demand
access. For small regions, the data can be included inline (bulk
push). For large regions, only the metadata is sent, and the
server fetches on demand.

This hybrid approach gives the best of both worlds:
- Small data: inline in RPC message (low latency)
- Large data: on-demand fetch (avoids blocking the RPC call)
- Cached data: zero-copy (already in server's local cache)
