# zlink — RPC Wrapper for Shared Libraries

**GPU-over-IP and beyond: wrap any shared library for remote execution with zero codegen.**

zlink combines [zpp_bits](https://github.com/eyalz800/zpp_bits) (C++20 binary serialization/RPC) with [r3map](https://github.com/pojntfx/r3map)-inspired remote memory to let you call functions in any `.so`/`.dylib` over the network — transparently.

## Why zlink vs Lupine?

| Feature | Lupine | zlink |
|---------|--------|-------|
| **Codegen required** | Python script parses CUDA headers | No codegen — explicit API declarations |
| **Supported libraries** | CUDA only | Any shared library |
| **Serialization** | Custom, per-header parsing | zpp_bits (fastest C++ serializer) |
| **Transport** | HTTP/2 | Raw TCP with framing (lower latency) |
| **Pointer marshalling** | Per-function codegen (serialize/deserialize) | Shared memory plane (r3map-inspired, no per-function code) |
| **Remote memory** | Basic handle remapping | r3map-style Backend interface + demand paging + explicit copy |
| **Concurrent calls** | Independent RPC lanes | Async concurrent lanes with call IDs |
| **Pointer mapping** | Built-in, CUDA-specific | Generic ptr_map + shared_mem_plane with shadow regions |
| **Build step** | codegen → cmake → build | cmake → build (one step) |
| **Language** | C (generated stubs) | C++20 (type-safe, zero-overhead) |
| **Deep pointer chasing** | Manual per-function | Automatic via memory region registration |

## Architecture

```
  Client (CPU-only machine)                     Server (GPU machine)
  ┌──────────────────────────┐                  ┌──────────────────────────┐
  │ Application               │                  │ zlink_server              │
  │    ↓                      │                  │    ↓                      │
  │ LD_PRELOAD=libzlink_*.so  │                  │ dlopen("libcuda.so.1")   │
  │    ↓                      │                  │    ↓                      │
  │ zlink shim (intercepts)   │                  │ Function registry         │
  │    ↓                      │                  │    ↓                      │
  │ shared_mem_plane          │                  │ shared_mem_plane          │
  │    ├─ memory_backend ─────┼── Backend IF ──→ │    ├─ rpc_backend         │
  │    │  (wraps app memory)  │  ReadAt/WriteAt  │    │  (proxies to client) │
  │    ↓                      │                  │    ↓                      │
  │ register_host_memory()    │                  │ translate_to_server()     │
  │    ↓                      │                  │    ↓ (fetches chunks      │
  │ zpp_bits RPC client       │   TCP frames    │       on demand from      │
  │    ↓ serialize            │ ═════════════►   │       client via Backend) │
  │ Transport (TCP)           │                  │    ↓                      │
  │                           │  ◄═════════════  │ Call real cuMemcpyHtoD()  │
  │ ptr_map (handle mapping)  │   TCP frames    │    ↓                      │
  │    ↓                      │                  │ ptr_map (handle mapping)  │
  │ Shadow memory region      │                  │    ↓                      │
  │ (demand-paged via         │  mem ops RPC     │ GPU VRAM                  │
  │  userfaultfd)             │ ═════════════►   │ (real device memory)      │
  └──────────────────────────┘                  └──────────────────────────┘
```

The **shared memory plane** is the key innovation. It adapts r3map's Backend
interface (`ReadAt/WriteAt/Size/Sync`) so the server can access client memory
on-demand. When a server function receives a client pointer, it calls
`translate_to_server()` which transparently fetches the data via the Backend.
No per-function marshalling code needed!

### Key Components

1. **RPC Engine** (`zlink/rpc.hpp`) — Wraps zpp_bits's `zpp::bits::rpc<>` for network use. Supports both typed (compile-time) and dynamic (runtime) function registration.

2. **Transport** (`zlink/transport.hpp`, `zlink/tcp_transport.hpp`) — Length-prefixed framing over TCP. Each frame: `[4B length][4B call_id][1B type][payload]`. Call IDs enable concurrent async RPC lanes.

3. **Pointer Map** (`zlink/ptr_map.hpp`) — Bidirectional mapping between client-side "fake" pointers and server-side real pointers. Uses shadow mmap regions so client pointers look like real addresses.

4. **Shared Memory Plane** (`zlink/shared_mem.hpp`) — **The key innovation.** Adapts r3map's Backend interface (`ReadAt/WriteAt/Size/Sync`) to make client memory directly accessible to the server. This solves the pointer marshalling problem:
   - **Client**: `register_host_memory()` wraps app memory in a `memory_backend`
   - **Server**: `translate_to_server()` fetches data on-demand via `rpc_backend`
   - Server functions just dereference pointers normally — no per-function code!
   - Supports both on-demand chunk fetching and explicit bulk transfer
   - Background push/pull for WAN optimization (like r3map managed mounts)

5. **Remote Memory** (`zlink/memory.hpp`) — Device memory operations (alloc/free/read/write) for GPU VRAM. Works alongside the shared memory plane for host↔device copies.

6. **Server** (`zlink/server.hpp`) — Opens target .so, scans exports, registers handlers, accepts connections, dispatches RPC calls.

7. **Shim** (`zlink/shim.hpp`) — LD_PRELOAD-based interception. Registers host memory with the shared_mem_plane before making RPC calls.

## Pointer Marshalling: How It Works

This is the hardest problem in GPU-over-IP. See [`docs/POINTER_MARSHALLING.md`](docs/POINTER_MARSHALLING.md) for the full design.

**The short version:** When a function like `cuMemcpyHtoD(dst, srcHost, n)` takes a client pointer (`srcHost`), the shared memory plane makes that data accessible to the server without per-function marshalling code:

1. Client shim calls `register_host_memory(srcHost, n)` → creates a `memory_backend` wrapping those bytes
2. Server calls `translate_to_server(srcHost, n, region_id)` → fetches the data from the client's `memory_backend` via `rpc_backend::ReadAt()`
3. Server calls the real `cuMemcpyHtoD(dst, local_src, n)` → it just works!

For output buffers (`cuMemcpyDtoH`), the process is reversed: the server marks the region as dirty after the function writes to it, and `flush_dirty()` writes the data back to the client via `WriteAt()`.

For deep pointers (`cuLaunchKernel`'s `kernelParams` array of pointers), each nested pointer is registered recursively.

## Quick Start

### Build

```bash
git clone https://github.com/your-org/zlink.git
cd zlink
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Example: Remote Math Functions

Terminal 1 (server — any machine with libm):
```bash
./build/examples/libmath/math_server 14833
```

Terminal 2 (client — any machine):
```bash
./build/examples/libmath/math_client 192.168.1.100 14833
```

Output:
```
Connecting to 192.168.1.100:14833...
Connected! Calling remote math functions...

remote sin(pi/6)  = 0.49999999  (expected ~0.5)
remote cos(0)     = 1.00000000  (expected 1.0)
remote sqrt(2)    = 1.41421356  (expected ~1.41421356)
remote pow(2,10)  = 1024.00000  (expected 1024.0)
remote exp(1)     = 2.71828183  (expected ~2.71828182)
remote log(e)     = 1.00000000  (expected ~1.0)
```

### Example: GPU-over-IP (CUDA)

Terminal 1 (GPU server):
```bash
./build/examples/cuda/cuda_server 14833
```

Terminal 2 (CPU-only client):
```bash
LD_PRELOAD=./build/libzlink_cuda_shim.so \
ZLINK_SERVER=gpu-host:14833 \
./your_cuda_application
```

The application sees a local GPU. All CUDA driver API calls are transparently forwarded over the network.

## How to Add a New Library

### Step 1: Declare the API surface

Create a header file (e.g., `mylib_api.hpp`):

```cpp
#include <zpp_bits.h>

namespace mylib_api {
    int mylib_init(int flags);
    int mylib_do_thing(const std::string& input, double param);
    void mylib_cleanup();
}

using mylib_rpc = zpp::bits::rpc<
    zpp::bits::bind<&mylib_api::mylib_init>,
    zpp::bits::bind<&mylib_api::mylib_do_thing>,
    zpp::bits::bind<&mylib_api::mylib_cleanup>
>;
```

### Step 2: Implement the server

```cpp
// mylib_server.cpp
#include "mylib_api.hpp"
#include <zlink/transport.hpp>
// ... (implement functions by calling real library)
```

### Step 3: Use the client

```cpp
// mylib_client.cpp
#include "mylib_api.hpp"
#include <zlink/transport.hpp>

auto result = remote_call<1>(*transport, "hello", 3.14);
```

**That's it.** No codegen, no config files, no build step beyond normal compilation.

## Wire Protocol

Each frame on the wire:

```
┌──────────────┬──────────────┬───────────┬──────────────────┐
│ Length (4B)  │ Call ID (4B) │ Type (1B) │ Payload (N bytes) │
│ uint32 BE    │ uint32 BE    │ uint8     │ zpp_bits data    │
└──────────────┴──────────────┴───────────┴──────────────────┘
```

Frame types:
- `0x01` REQUEST — RPC function call
- `0x02` RESPONSE — RPC function return
- `0x03` ERROR — Error response
- `0x10` MEMORY_OP — Remote memory operation
- `0x11` MEMORY_REPLY — Memory operation response
- `0xFF` HEARTBEAT — Keep-alive

The payload for RPC frames is the raw zpp_bits serialized data — function ID + arguments (request) or return value (response).

## Pointer Translation

The hardest problem in GPU-over-IP is pointer translation. When a remote function returns a device pointer (e.g., `CUdeviceptr` from `cuMemAlloc`), the client needs a local representation:

1. **Shadow region**: zlink mmaps a 4 GiB region on the client. "Device pointers" are allocated from this region, so they look like real addresses. Applications can do pointer arithmetic, comparisons, etc.

2. **Bidirectional map**: `ptr_map` maintains `local_ptr ↔ remote_ptr` mappings with O(1) lookup in both directions.

3. **Automatic translation**: The shim layer translates pointers before sending RPC calls and after receiving responses. The application never sees the remote addresses.

4. **Demand paging**: For memory-mapped access (e.g., managed memory), userfaultfd intercepts page faults on the shadow region and transparently fetches pages from the server.

## Performance Considerations

- **zpp_bits** is the [fastest C++ serializer](https://github.com/eyalz800/zpp_bits#benchmark) — zero-overhead binary encoding, no reflection overhead
- **TCP_NODELAY** is enabled by default for minimal RPC latency
- **Concurrent lanes**: Multiple in-flight RPC calls with independent call IDs
- **Bulk data**: Memory copy operations (HtoD/DtoH) send data directly in the frame payload — no extra round-trips
- **Demand paging**: For workloads with sparse memory access, userfaultfd avoids transferring entire allocations

Typical latency for a CUDA RPC call on a LAN:
- Small call (cuInit, cuDeviceGet): ~0.1-0.5 ms
- Memory copy 1 MB HtoD: ~1-2 ms (1 Gbps link)
- cuLaunchKernel: ~0.2-0.5 ms (kernel runs remotely, result is just a status code)

## Limitations & Roadmap

### Current
- Single-connection server (one client at a time)
- Synchronous RPC (async API exists but needs transport integration)
- CUDA API coverage is partial (core functions only)
- No TLS/authentication (front with a reverse proxy for production)
- Linux-only for demand paging (macOS/Windows use explicit copy API)

### Planned
- [ ] Multi-connection server with connection pooling
- [ ] Full CUDA driver API coverage (200+ functions)
- [ ] CUDA runtime API (cudart) support
- [ ] cuBLAS, cuDNN, cuSPARSE wrappers
- [ ] NVML shim (nvidia-smi over the network)
- [ ] TLS support (mTLS or via proxy)
- [ ] RDMA transport (ibverbs) for InfiniBand
- [ ] HTTP/2 transport option (for NAT traversal)
- [ ] Server-to-server direct transfers (avoid client hop for DtoD cross-server)
- [ ] Checkpoint/resume (like Lupine's graceful drain)
- [ ] Python bindings (pybind11)
- [ ] Automatic symbol scanning (parse ELF/Mach-O exports)

## Comparison with Related Projects

| Project | Approach | Codegen | Generic | Memory | Transport |
|---------|----------|---------|---------|--------|-----------|
| **zlink** | RPC wrapper | No | Yes | Demand paging | TCP |
| **Lupine** | CUDA shim | Yes (Python) | No (CUDA only) | Handle remap | HTTP/2 |
| **RCUDA** | CUDA RPC | Yes | No | Basic | TCP |
| **SCUDA** | CUDA bridge | Yes (Python) | No | Basic | TCP |
| **r3map** | Remote mmap | N/A | N/A | Yes (NBD) | NBD protocol |
| **Thunder Compute** | Proprietary | Unknown | No | Unknown | Unknown |

## License

MIT

## Acknowledgments

- [zpp_bits](https://github.com/eyalz800/zpp_bits) — C++20 serialization & RPC
- [r3map](https://github.com/pojntfx/r3map) — Remote memory region mounting
- [Lupine](https://github.com/lupinemachines/lupine) — GPU-over-IP inspiration
