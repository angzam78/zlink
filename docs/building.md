# Building and Running zlink

## Prerequisites

- **C++20 compiler** — GCC 10+, Clang 12+, or MSVC 2019 16.10+
- **CMake 3.20+**
- **zpp_bits** — Fetched automatically by CMake via `FetchContent`
- **CUDA Toolkit** (for CUDA examples only) — Installed on the server machine

## Building

```bash
git clone https://github.com/angzam78/zlink.git
cd zlink
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

This produces:

- `build/libzlink_core.a` — Static library with core zlink functionality
- `build/zlink_server` — Generic server binary
- `build/examples/libmath/math_server` — Math example server
- `build/examples/libmath/math_client` — Math example client

### CUDA Examples

The CUDA examples are integrated into the CMake build and enabled with
`-DZLINK_CUDA_EXAMPLES=ON`. The server target requires the CUDA toolkit
(`find_package(CUDAToolkit)`); the client target builds on any machine.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DZLINK_CUDA_EXAMPLES=ON
cmake --build build -j$(nproc)
```

This produces:

- `build/examples/cuda/cuda_server` — GPU server (needs CUDA toolkit)
- `build/examples/cuda/cuda_client` — Pipeline client (CPU-only, no CUDA needed)

The server and client share `cuda_api.hpp`, a hand-written set of CUDA Driver
API RPC declarations (30 functions) with zpp_bits `bind<>` entries. Adding a
new function requires three lines: declare the return struct + signature, add
a `bind<>` entry, and categorize it (barrier / enqueued / readback).

## Running the Examples

### Example 1: Remote Math Functions

This example demonstrates basic RPC: the client has **zero** dependency on
`libm` — all computation happens on the server.

**Terminal 1 (server — any machine with libm):**
```bash
./build/examples/libmath/math_server 14833
```

**Terminal 2 (client — any machine):**
```bash
./build/examples/libmath/math_client 127.0.0.1 14833
```

Expected output (client):
```
Connecting to 127.0.0.1:14833...
Connected! Calling remote math functions...

remote sin(pi/6)  = 0.49999999  (expected ~0.5)
remote cos(0)     = 1.00000000  (expected 1.0)
remote sqrt(2)    = 1.4421356  (expected ~1.41421356)
remote pow(2,10)  = 1024.00000  (expected 1024.0)
remote exp(1)     = 2.71828183  (expected ~2.71828182)
remote log(e)     = 1.00000000  (expected ~1.0)
```

### Example 2: CUDA Pipeline with Virtual Handles

This example demonstrates the dependency-aware CUDA pipeline. It runs on a
single machine for testing (client and server on localhost).

**Terminal 1 (GPU server):**
```bash
./build/examples/cuda/cuda_server
```

**Terminal 2 (client — can be CPU-only in production):**
```bash
./build/examples/cuda/cuda_client 127.0.0.1 14833
```

Expected output (client):
```
Connecting to 127.0.0.1:14833...
Connected (multi-port: 3 channels)

[client] demand paging + write tracking enabled
=== Phase 1: Setup (barriers) ===
cuInit: 0 OK
Driver version: 13000
Device count: 2
GPU 0: NVIDIA GeForce RTX 3090
Total memory: 24122 MB
Warp size: 64

=== Phase 2: Virtual Handle Pipeline ===
ALL calls enqueued — NO barriers between alloc/HtoD/sync!

cuCtxCreate: VH(0) — enqueued
cuMemAlloc:  VH(1) — enqueued
cuMemAlloc(2): VH(2) — enqueued
cuStreamCreate: VH(3) — enqueued
[client] flushed 2 dirty pages
cuMemcpyHtoD(VH(1)): enqueued (dirty pages flushed)
cuMemcpyDtoD: enqueued
cuCtxSynchronize: enqueued

=== cuMemcpyDtoH: READBACK — flushing pipeline ===
cuMemcpyDtoH result: 0 (SUCCESS)

=== Data Verification (DtoD round-trip) ===
  All 64 values match!
  VH pipeline verified: alloc+alloc+stream+HtoD+DtoD+sync+DtoH in 1 round-trip!
  Demand paging + write tracking: WORKING

=== Phase 3: Event Pipeline Test ===
cuEventCreate x2: VH(4), VH(5) — enqueued
cuEventRecord(start): enqueued
cuEventRecord(end): enqueued
cuEventSynchronize(end): enqueued
Event batch flushed: 5 results
cuEventElapsedTime: 0 OK, 0.002048 ms

=== Phase 4: Managed Memory + Cleanup ===
cuMemAllocManaged: VH(6) — enqueued
cuMemcpyHtoD(managed) + cuCtxSynchronize: enqueued
cuMemcpyDtoH(managed) result: 0 (SUCCESS)
  Managed memory round-trip: VERIFIED
Cleanup (3×mem_free + stream_destroy): enqueued
Cleanup batch flushed: 4 results

=== All tests complete ===
```

The entire Phase 2 batch (ctx create + 2× alloc + stream + HtoD + DtoD + sync)
is enqueued with virtual handles and flushed in a single round-trip by the
`cuMemcpyDtoH` readback call. The `[client] demand paging + write tracking
enabled` line confirms the memory_page_tracker was initialized; `[client] flushed N
dirty pages` confirms the page tracker detected dirty pages in the shadow
region and pushed them to the server.

### Running in Containers (Docker, Kubernetes)

When `userfaultfd(2)` is blocked by seccomp (the default Docker profile returns
`EPERM`), the memory_page_tracker transparently falls back to `mprotect` + `SIGSEGV`
(Tier 3). No code changes or configuration are needed — the factory auto-selects
the best available tier at runtime. The `[client] demand paging + write
tracking enabled` message confirms the fallback is active.

To enable `userfaultfd` in Docker, add `--security-opt seccomp=unconfined` or
use a custom seccomp profile that allows syscall 323 (`userfaultfd`).

### Cross-Host Testing

To test between two machines:

1. On the GPU server, start the server listening on all interfaces:
   ```bash
   ./build/examples/cuda/cuda_server
   ```

2. On the CPU-only client, connect to the server:
   ```bash
   ./build/examples/cuda/cuda_client 192.168.1.100 14833
   ```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Debug` | Build type (`Release` recommended for benchmarks) |
| `CMAKE_CXX_STANDARD` | `20` | C++ standard (must be 20 or later) |
| `ZLINK_CUDA_EXAMPLES` | `OFF` | Enable CUDA server + client targets (requires CUDAToolkit for server) |

The build uses `-Wall -Wextra -Wpedantic` and `-O2` in Release mode.


## Configuration Constants

Port and frame size constants are in `include/zlink/config.hpp`:

```cpp
inline constexpr std::uint16_t default_port = 14833;
inline constexpr std::size_t   max_frame_size = 64 * 1024 * 1024;  // 64 MiB
```

Compression thresholds are in `include/zlink/compress.hpp`:

```cpp
inline constexpr std::size_t compress_threshold = 4096;     // 4 KB minimum
inline constexpr double compress_ratio_threshold = 0.9;      // Must save >= 10%
```

## Running Tests

```bash
cd build
ctest --output-on-failure
```
