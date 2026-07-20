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

The CUDA examples are not in the CMake build system (they require CUDA on the
build machine). Use the build scripts:

```bash
# Build and test pipeline on the same machine
python3 scripts/build_test_pipeline.py

# Build and test across two machines
python3 scripts/cross_host_test.py <server_host> [port]
```

Or compile manually:

```bash
# Server (needs CUDA)
g++ -std=c++20 -O2 \
    -I include -I build/_deps/zpp_bits-src \
    examples/cuda/cuda_test_server.cpp \
    -L build -lzlink_core -llz4 -lpthread \
    -lcuda \
    -o cuda_test_server

# Client (CPU-only, no CUDA needed)
g++ -std=c++20 -O2 \
    -I include -I build/_deps/zpp_bits-src \
    examples/cuda/cuda_test_client.cpp \
    -L build -lzlink_core -llz4 -lpthread \
    -o cuda_test_client
```

## Running the CUDA Example

### Same Machine

Terminal 1 (server — needs GPU):
```bash
./cuda_test_server 14833
```

Terminal 2 (client — can be CPU-only):
```bash
./cuda_test_client 127.0.0.1 14833
```

### Cross-Machine

On the GPU server:
```bash
./cuda_test_server 14833
```

On the CPU-only client:
```bash
./cuda_test_client 192.168.1.100 14833
```

### Expected Output

Client:
```
Connecting to 127.0.0.1:14833...
Connected!

=== Phase 1: Setup (barriers) ===
cuInit: 0 OK
Device count: 1
GPU 0: NVIDIA GeForce RTX 4090
Total memory: 24564 MB

=== Phase 2: Virtual Handle Pipeline ===
ALL calls enqueued — NO barriers between alloc/HtoD/sync!

cuCtxCreate: VH(0) — enqueued
cuMemAlloc: VH(1) — enqueued
cuMemcpyHtoD(VH(1)): enqueued + inline sync
cuCtxSynchronize: enqueued

=== cuMemcpyDtoH: READBACK — flushing pipeline ===
...

=== Data Verification ===
  All 64 values match!
  Virtual handle pipeline verified: alloc→HtoD→sync→DtoH in 1 round-trip!

=== Pipeline Benchmark: Virtual Handles vs Barriers ===
  Virtual handles (1-2 round-trips):  XXX us
  Barrier style (3+ round-trips):     XXX us
  Speedup: X.XXx
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
│   ├── cuda/                # CUDA RPC example (client + server)
│   └── libmath/             # Math service example
├── scripts/                 # Build and test scripts
│   ├── build_test_pipeline.py
│   └── cross_host_test.py
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
