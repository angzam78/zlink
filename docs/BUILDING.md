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

The CUDA examples require `libcuda.so` on the server side. They are built
manually (not integrated into the main CMakeLists.txt yet) because they
need the CUDA toolkit. See `examples/cuda/` for source files.

Build the CUDA test client and server with:

```bash
cd zlink
g++ -std=c++20 -O2 \
    -I include -I build/_deps/zpp_bits-src \
    examples/cuda/cuda_test_client.cpp \
    -L build -lzlink_core -lpthread \
    -o build/cuda_test_client

g++ -std=c++20 -O2 \
    -I include -I build/_deps/zpp_bits-src \
    examples/cuda/cuda_test_server.cpp \
    -L build -lzlink_core -lpthread -lcuda \
    -o build/cuda_test_server
```

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
./build/cuda_test_server 14833
```

**Terminal 2 (client — can be CPU-only in production):**
```bash
./build/cuda_test_client 127.0.0.1 14833
```

Expected output (client):
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
Pipeline sends: [sync_data][ctx_create][mem_alloc][memcpy_htod]
[ctx_sync][memcpy_dtoh][read_req][handle_manifest]
Server processes in order, translates VH → real handles

cuMemcpyDtoH result: 0 (SUCCESS)

=== Data Verification ===
  All 64 values match!
  Virtual handle pipeline verified: alloc→HtoD→sync→DtoH in 1 round-trip!

Cleanup: cuMemFree + cuCtxDestroy enqueued
Final flush: 2 deferred calls

=== Pipeline Benchmark: Virtual Handles vs Barriers ===
  Virtual handles (1-2 round-trips):  XXX us
  Barrier style (3+ round-trips):     XXX us
  Speedup: X.XXx
```

### Cross-Host Testing

To test between two machines:

1. On the GPU server, start the server listening on all interfaces:
   ```bash
   ./build/cuda_test_server 0.0.0.0 14833
   ```

2. On the CPU-only client, connect to the server:
   ```bash
   ./build/cuda_test_client 192.168.1.100 14833
   ```

## Project Configuration

The `zlink.toml` file in the project root is the configuration file for the
shim layer (LD_PRELOAD mode). It specifies:

- Target library name (e.g., `libcuda.so.1`)
- Server host and port
- Symbols to intercept or pass through

Environment variables (alternative to config file):

- `ZLINK_SERVER` — Server host:port (e.g., `gpu-host:14833`)
- `ZLINK_LIBRARY` — Target library name
- `ZLINK_CONFIG` — Path to config file

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Debug` | Build type (`Release` recommended for benchmarks) |
| `CMAKE_CXX_STANDARD` | `20` | C++ standard (must be 20 or later) |

The build uses `-Wall -Wextra -Wpedantic` and `-O2` in Release mode.
