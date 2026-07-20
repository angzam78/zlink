# zlink — GPU-over-IP via CUDA RPC Pipeline

**Run CUDA workloads on CPU-only machines by forwarding GPU calls to a remote server.**

zlink combines [zpp_bits](https://github.com/eyalz800/zpp_bits) (C++20 binary serialization/RPC) with [r3map](https://github.com/pojntfx/r3map)-inspired remote memory to let you call functions in any shared library over the network — transparently.

## Key Features

- **Dependency-aware CUDA pipeline** — Batches multiple GPU calls into single network round-trips using virtual handles
- **Virtual handles** — Break the barrier chain: `cuMemAlloc` returns a virtual ID instead of blocking for the real pointer
- **Inline memory operations** — Host data sync and device data readback are packed into pipeline frames, zero extra round-trips
- **Zero codegen** — Explicit API declarations with zpp_bits `bind<>`, no Python scripts or generated stubs
- **Type-safe** — C++20 templates ensure compile-time type safety for all RPC calls

## Architecture

```
  Client (CPU-only machine)                     Server (GPU machine)
  ┌──────────────────────────┐                  ┌──────────────────────────┐
  │ Application code          │                  │ zlink server              │
  │    ↓                      │                  │    ↓                      │
  │ cuda_pipeline<RpcDef>     │                  │ handle_pipeline_mem()     │
  │    ↓                      │                  │    ↓                      │
  │ Virtual handle allocator  │                  │ Handle table (VH→real)    │
  │    ↓                      │                  │    ↓                      │
  │ zpp_bits serialization    │   TCP frames     │ zpp_bits deserialization  │
  │    ↓                      │ ═════════════►   │    ↓                      │
  │ Transport (TCP)           │                  │ Real CUDA calls           │
  │                           │  ◄═════════════  │    ↓                      │
  │ Host mirror client        │   TCP frames     │ Host memory mirror        │
  └──────────────────────────┘                  └──────────────────────────┘
```

## Quick Start

### Build

```bash
git clone https://github.com/angzam78/zlink.git
cd zlink
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Example: Remote Math Functions

Terminal 1 (server):
```bash
./build/examples/libmath/math_server 14833
```

Terminal 2 (client):
```bash
./build/examples/libmath/math_client 127.0.0.1 14833
```

### Example: CUDA Pipeline with Virtual Handles

Terminal 1 (GPU server):
```bash
./build/cuda_test_server 14833
```

Terminal 2 (client):
```bash
./build/cuda_test_client 127.0.0.1 14833
```

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and component overview |
| [CUDA_PIPELINE.md](docs/CUDA_PIPELINE.md) | Virtual handles and dependency-aware pipelining |
| [WIRE_PROTOCOL.md](docs/WIRE_PROTOCOL.md) | Frame format and network protocol |
| [MEMORY_SUBSYSTEM.md](docs/MEMORY_SUBSYSTEM.md) | Remote memory, host mirror, chunk cache |
| [POINTER_MARSHALLING.md](docs/POINTER_MARSHALLING.md) | How pointers cross the network boundary |
| [BUILDING.md](docs/BUILDING.md) | Build instructions and running examples |

## Pipeline Performance

With virtual handles, a typical CUDA workload goes from 10+ round-trips to 3-4:

```
cuInit()                    → barrier (1 RT)
cuDeviceGetCount()          → barrier (1 RT)
cuDeviceGetAttribute(...)   → barrier (1 RT)
cuCtxCreate(...)            → enqueued: VH(0)     ┐
cuMemAlloc(N)               → enqueued: VH(1)     │
cuMemAlloc(M)               → enqueued: VH(2)     │
cuModuleLoadData(...)       → enqueued: VH(3)     ├─ 1 batch RT!
cuModuleGetFunction(VH(3))  → enqueued: VH(4)     │
cuMemcpyHtoD(VH(1), ...)    → enqueued             │
cuLaunchKernel(VH(4), ...)  → enqueued             │
cuMemcpyDtoH(out, VH(2))   → readback: flush!    ┘

Total: 3 barrier RTs + 1 pipeline batch RT = 4 round-trips
```

## Wire Protocol

Each frame on the wire:

```
┌──────────────┬──────────────┬───────────┬──────────────────┐
│ Length (4B)  │ Call ID (4B) │ Type (1B) │ Payload (N bytes) │
│ uint32 BE    │ uint32 BE    │ uint8     │ zpp_bits data    │
└──────────────┴──────────────┴───────────┴──────────────────┘
```

The `pipeline_mem` frame type (`0x06`) combines host sync data, RPC calls,
read requests, and the virtual handle manifest into a single round-trip.

See [WIRE_PROTOCOL.md](docs/WIRE_PROTOCOL.md) for full protocol details.

## Project Structure

```
zlink/
├── include/zlink/          # Public headers (mostly header-only)
│   ├── config.hpp          # Protocol constants, frame types
│   ├── transport.hpp       # Abstract transport + frame struct
│   ├── tcp_transport.hpp   # TCP transport implementation
│   ├── rpc.hpp             # RPC engine (client, pipeline, server)
│   ├── cuda_pipeline.hpp   # Dependency-aware CUDA pipeline
│   ├── virtual_handle.hpp  # Virtual handle system
│   ├── cuda_dep_spec.hpp   # CUDA API dependency categorization
│   ├── memory.hpp          # Remote memory subsystem
│   ├── chunk_cache.hpp     # Page-level cache (r3map-inspired)
│   ├── shared_mem.hpp      # Shared memory plane
│   ├── ptr_map.hpp         # Bidirectional pointer mapping
│   ├── shim.hpp            # LD_PRELOAD shim
│   ├── client.hpp          # Client framework
│   └── server.hpp          # Server framework
├── src/                    # Implementation files
├── examples/
│   ├── libmath/            # Remote math example
│   └── cuda/               # CUDA examples (test client/server)
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── cmake/                  # CMake modules (zpp_bits fetch)
```

## Comparison with Related Projects

| Project | Approach | Codegen | Generic | Pipelining | Memory | Transport |
|---------|----------|---------|---------|------------|--------|-----------|
| **zlink** | RPC pipeline | No | Yes | Virtual handles | Inline sync + demand paging | TCP |
| **Lupine** | CUDA shim | Yes (Python) | No (CUDA only) | Basic | Handle remap | HTTP/2 |
| **RCUDA** | CUDA RPC | Yes | No | Basic | Basic | TCP |
| **SCUDA** | CUDA bridge | Yes (Python) | No | Basic | Basic | TCP |
| **r3map** | Remote mmap | N/A | N/A | N/A | Yes (NBD) | NBD |

## License

MIT

## Acknowledgments

- [zpp_bits](https://github.com/eyalz800/zpp_bits) — C++20 serialization & RPC
- [r3map](https://github.com/pojntfx/r3map) — Remote memory region mounting
- [Lupine](https://github.com/lupinemachines/lupine) — GPU-over-IP inspiration
