# Build and Run

## Prerequisites

- C++20 compiler (GCC 12+, Clang 15+)
- CMake 3.20+
- LZ4 development library (`liblz4-dev` on Debian/Ubuntu)
- CUDA Toolkit (for CUDA examples — server only)
- Python 3 (for build/test scripts)

## Building

### Core Library + Math Example

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

This builds:
- `libzlink_core.a` — static library with transport, RPC, memory system
- `zlink_server` — generic server binary
- `math_server` / `math_client` — math service example

### CUDA Examples

The CUDA examples are integrated into the CMake build and enabled with
`-DZLINK_CUDA_EXAMPLES=ON`. The server target requires the CUDA toolkit
(`find_package(CUDAToolkit)`); the client target builds on any machine.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DZLINK_CUDA_EXAMPLES=ON
cmake --build build -j$(nproc)
```

This builds:
- `examples/cuda/cuda_server` — GPU server (needs CUDA toolkit)
- `examples/cuda/cuda_client` — Pipeline client (CPU-only, no CUDA needed)

## Running the CUDA Example

### Same Machine

Terminal 1 (server — needs GPU):
```bash
./build/examples/cuda/cuda_server
```

Terminal 2 (client — can be CPU-only):
```bash
./build/examples/cuda/cuda_client 127.0.0.1 14833
```

### Cross-Machine

On the GPU server:
```bash
./build/examples/cuda/cuda_server
```

On the CPU-only client:
```bash
./build/examples/cuda/cuda_client 192.168.1.100 14833
```

### Expected Output

Client:
```
Connecting to 127.0.0.1:14833...
Connected!

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
cuMemcpyHtoD(VH(1)): enqueued (inline sync)
cuMemcpyDtoD: enqueued
cuCtxSynchronize: enqueued

=== cuMemcpyDtoH: READBACK — flushing pipeline ===
cuMemcpyDtoH result: 0 (SUCCESS)

=== Data Verification (DtoD round-trip) ===
  All 64 values match!
  VH pipeline verified: alloc+alloc+stream+HtoD+DtoD+sync+DtoH in 1 round-trip!

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

## Running Tests

```bash
cd build
ctest --output-on-failure
```

## Project Structure

```
zlink/
├── include/zlink/           # Header-only core library
│   ├── config.hpp           # Protocol constants, frame types
│   ├── transport.hpp        # Abstract transport interface
│   ├── tcp_transport.hpp    # TCP transport implementation
│   ├── rpc.hpp              # RPC framework (zpp_bits wrappers)
│   ├── cuda_pipeline.hpp    # Dependency-aware CUDA pipeline
│   ├── cuda_dep_spec.hpp    # CUDA API dependency categorization
│   ├── virtual_handle.hpp   # Virtual handle system
│   ├── compress.hpp         # LZ4 compression for pipeline data
│   ├── memory.hpp           # Memory mirror + cached client
│   ├── chunk_cache.hpp      # Page-level caching
│   ├── ptr_map.hpp          # Bidirectional pointer mapping
│   ├── shared_mem.hpp       # Shared memory regions
│   ├── client.hpp           # Client-side interface
│   ├── server.hpp           # Server-side interface
│   └── shim.hpp             # LD_PRELOAD shim interface
├── src/                     # Compiled source files
│   ├── tcp_transport.cpp
│   ├── rpc.cpp
│   ├── memory_region.cpp
│   ├── chunk_cache.cpp
│   ├── ptr_map.cpp
│   ├── shared_mem.cpp
│   └── server.cpp
├── examples/
│   ├── cuda/                # CUDA RPC example (server + client)
│   └── libmath/             # Math service example
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── cmake/                   # CMake modules
│   └── FetchZppBits.cmake
├── CMakeLists.txt
└── zlink.toml
```

## Configuration

Port and frame size constants are in `include/zlink/config.hpp`:

```cpp
inline constexpr std::uint16_t default_port = 14833;
inline constexpr std::size_t   max_frame_size = 64 * 1024 * 1024;  // 64 MiB
```

Compression thresholds are in `include/zlink/compress.hpp`:

```cpp
inline constexpr std::size_t compress_threshold = 4096;     // 4 KB minimum
inline constexpr double compress_ratio_threshold = 0.9;      // Must save ≥ 10%
```
