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

## zlink's Approach: Separated Memory Layer

zlink separates pointer marshalling into two independent mechanisms:

1. **Device pointers** → Virtual handles (encoded in the pointer value itself)
2. **Host pointers** → Memory layer (host_sync / host_read via the backend interface)

RPC frames carry only call metadata and virtual handle IDs. Host buffer data
flows on a separate channel, never blocking RPC traffic.

### Device Pointers: Virtual Handles

Device pointers are replaced with client-side virtual IDs:

- `cuMemAlloc` returns a virtual handle (VH) instead of a real device pointer
- Subsequent calls pass VH values as arguments
- The server translates VH → real handle via `handle_table::translate()`
- The encoding is safe: bit 63 is set for virtual handles, which never
  appears in real user-space pointers

```
VH(0) = 0x8000000000000000
VH(1) = 0x8000000000000001
Real ptr: 0x7f18c6c00000 (bit 63 clear)
```

See [architecture.md](architecture.md) § Virtual Handle System for details.

### Host Pointers: Memory Layer

Host buffer data is synced via the memory layer, not packed into RPC frames.

```
CLIENT                                            SERVER
┌────────────────────────┐                        ┌────────────────────────┐
│ cuMemcpyHtoD(VH(1),    │                        │ Receives:               │
│   host_data, 256)      │                        │  1. memory_op (sync)   │
│                        │                        │  2. pipeline_request   │
│ Step 1: host_sync      │                        │                        │
│   Sends host_data to   │  memory_op frame       │  sync_page() stores    │
│   server mirror        │ ══════════════════►    │  data in mirror region │
│                        │                        │                        │
│ Step 2: enqueue RPC    │                        │  translate VH→real     │
│   memcpy_hto_d(VH(1),  │  pipeline_request      │  translate client addr │
│   client_addr, 256)    │ ══════════════════►    │    → server-local addr │
│                        │                        │  call cuMemcpyHtoD()   │
└────────────────────────┘                        └────────────────────────┘
```

### Step-by-Step: cuMemcpyHtoD

1. **Client** calls `cu_memcpy_htod(pipe, dev_vh, host_data, byte_count)`
2. This enqueues the RPC call: `pipe.enqueue<memcpy_hto_d>(dev_vh, client_addr, byte_count)`
3. Before the RPC batch is sent, the client syncs host data via the memory
   layer (`host_sync` frame on the memory channel)
4. **Server** receives the `host_sync`:
   - `g_host_mirror.sync_page(client_addr, data)` stores data in server's
     mmap'd mirror region
5. **Server** receives the `pipeline_request`:
   - Translates `dev_vh` → real device pointer via `g_vhandles.translate()`
   - Translates `client_addr` → server-local address via `g_host_mirror.translate()`
   - Calls real `cuMemcpyHtoD(real_dev_ptr, server_local_ptr, byte_count)`
6. CUDA reads from the server-local pointer, which has the same data as
   the client's original buffer

### Step-by-Step: cuMemcpyDtoH (Readback)

1. **Client** calls `cu_memcpy_dtoh(pipe, host_data, dev_vh, byte_count)`
2. This calls `pipe.call_readback<memcpy_dto_h>(client_addr, dev_vh, byte_count)`
   which flushes any pending batch, then sends the RPC synchronously
3. **Server** processes the RPC:
   - Translates VH and client address
   - Calls real `cuMemcpyDtoH(server_local_ptr, real_dev_ptr, byte_count)`
   - GPU data is now in the server's mirror region
4. **Client** calls `pipe.host_read(host_data, byte_count)`:
   - Sends a `host_read` memory_op frame
   - Server reads from the mirror region and sends data back
   - Client copies data into `host_data` buffer

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

## Kernel Launch: Deep Pointer Handling

`cuLaunchKernel` is the hardest case because `kernelParams` is an array of
`void*` pointers, each of which may point to device memory. The server-side
handler walks the args array, translating any virtual handles:

```cpp
// Server-side kernel launch handler:
case 13: {  // launch_kernel
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

This approach works because:
1. The client syncs the args data via `host_sync` (memory channel)
2. The server mirrors the data at a known server-local address
3. The server walks the args, translating VH values to real device pointers
4. CUDA receives real pointers and the kernel runs correctly

## Backend Interface

The `shared_mem_plane` (in `shared_mem.hpp`) and `chunk_cache` (in
`chunk_cache.hpp`) provide the infrastructure for on-demand memory access
via the r3map Backend interface:

1. **`backend` interface** — `ReadAt/WriteAt/Size/Sync`, matching r3map's Go
   Backend
2. **`memory_backend`** — Wraps local memory as a backend
3. **`rpc_backend`** — Proxies ReadAt/WriteAt over the transport
4. **`shared_mem_plane`** — Coordinates client↔server memory access with
   lazy fetching and background push/pull

This enables future enhancements:
- Demand paging via `userfaultfd` (read-side)
- Write tracking via `mprotect` + `SIGSEGV` (write-side)
- Page-level caching with coherence (chunk_cache)

## Summary

| Pointer type | Mechanism | Channel |
|-------------|-----------|---------|
| Device pointers | Virtual handles (bit 63 encoding) | RPC (in call args) |
| Host input pointers | `host_sync` to server mirror | Memory channel |
| Host output pointers | `host_read` from server mirror | Memory channel |
| Kernel args (deep pointers) | `host_sync` + server-side VH walk | Memory + RPC |
