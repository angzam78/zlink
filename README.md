# zlink — GPU-over-IP via CUDA RPC Pipeline

**Run CUDA workloads on CPU-only machines by forwarding GPU calls to a remote server.**

zlink combines [zpp_bits](https://github.com/eyalz800/zpp_bits) (C++20 binary serialization/RPC) with [r3map](https://github.com/pojntfx/r3map)-inspired remote memory to let you call functions in any shared library over the network — transparently.

## Key Features

- **Dependency-aware CUDA pipeline** — Batches multiple GPU calls into single network round-trips using virtual handles
- **Managed pipeline** — Prefetch, write-behind, and multiplexed transport sustain throughput at WAN RTTs (3.2x speedup at 10ms RTT)
- **Virtual handles** — Break the barrier chain: `cuMemAlloc` returns a virtual ID instead of blocking for the real pointer
- **Inline memory operations** — Host data sync and device data readback are packed into pipeline frames, zero extra round-trips
- **Zero codegen** — Explicit API declarations with zpp_bits `bind<>`, no Python scripts or generated stubs
- **Type-safe** — C++20 templates ensure compile-time type safety for all RPC calls
- **Workload-agnostic** — Managed pipeline benefits any iterative CUDA workload (ML training, compute, transfer, mixed)

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

Build with CUDA examples (server requires the CUDA toolkit):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DZLINK_CUDA_EXAMPLES=ON
cmake --build build -j$(nproc)
```

Terminal 1 (GPU server):
```bash
./build/examples/cuda/cuda_server
```

Terminal 2 (client):
```bash
./build/examples/cuda/cuda_client 127.0.0.1 14833
```

## Documentation

| Document | Description |
|----------|-------------|
| [architecture.md](docs/architecture.md) | System architecture and component overview |
| [building.md](docs/building.md) | Build instructions and running examples |
| [cuda-pipeline.md](docs/cuda-pipeline.md) | Virtual handles and dependency-aware pipelining |
| [wire-protocol.md](docs/wire-protocol.md) | Frame format and network protocol |
| [memory.md](docs/memory.md) | Remote memory, host mirror, chunk cache |
| [pointer-marshalling.md](docs/pointer-marshalling.md) | How pointers cross the network boundary |
| [compression.md](docs/compression.md) | LZ4 compression in pipeline frames |
| [rpc-framework.md](docs/rpc-framework.md) | zpp_bits RPC engine and API declaration |
| [managed-pipeline.md](docs/managed-pipeline.md) | Managed pipeline architecture (prefetch, write-behind, multiplexed transport) |
| [managed-pipeline-integration.md](docs/managed-pipeline-integration.md) | Step-by-step integration guide for managed pipeline |
| [pytorch-over-zlink.md](docs/pytorch-over-zlink.md) | Planned PyTorch integration via LD_PRELOAD shim |

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

See [wire-protocol.md](docs/wire-protocol.md) for full protocol details.

## Project Structure

```
zlink/
├── include/zlink/          # Public headers (mostly header-only)
│   ├── config.hpp          # Protocol constants, frame types (incl. managed pipeline)
│   ├── transport.hpp       # Abstract transport + frame struct
│   ├── tcp_transport.hpp   # TCP transport implementation
│   ├── multiplexed_transport.hpp # 3-channel TCP (RPC/bulk/prefetch) — managed pipeline
│   ├── rpc.hpp             # RPC engine (client, pipeline, server)
│   ├── cuda_pipeline.hpp   # Dependency-aware CUDA pipeline
│   ├── managed_pipeline.hpp # Managed pipeline wrapper (prefetch + write-behind)
│   ├── virtual_handle.hpp  # Virtual handle system
│   ├── cuda_dep_spec.hpp   # CUDA API dependency categorization
│   ├── memory.hpp          # Remote memory subsystem
│   ├── chunk_cache.hpp     # Page-level cache (r3map-inspired)
│   ├── prefetch_worker.hpp # Background page prefetch with pattern detection
│   ├── write_behind_buffer.hpp # Async write-behind with fence sync
│   ├── connection_manager.hpp # Auto-reconnect + heartbeat + session recovery
│   ├── shared_mem.hpp      # Shared memory plane
│   ├── ptr_map.hpp         # Bidirectional pointer mapping
│   ├── compress.hpp        # LZ4 compression for large payloads
│   ├── shim.hpp            # LD_PRELOAD shim
│   ├── client.hpp          # Client framework
│   └── server.hpp          # Server framework
├── src/                    # Implementation files
├── examples/
│   ├── libmath/            # Remote math example
│   └── cuda/               # CUDA RPC server + pipeline client
│       ├── cuda_api.hpp    # Hand-written CUDA Driver API RPC declarations
│       ├── cuda_server.cpp # GPU server: real CUDA calls + pipeline_mem handler
│       ├── cuda_client.cpp # Pipeline client: virtual handles + readback test
│       └── CMakeLists.txt  # Builds cuda_server (needs CUDA) + cuda_client
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── sim/                    # Performance simulation + charts
└── cmake/                  # CMake modules (zpp_bits fetch)
```

## Comparison with Related Projects

| Project | Approach | Codegen | Generic | Pipelining | Memory | Transport |
|---------|----------|---------|---------|------------|--------|-----------|
| **zlink** | RPC pipeline + managed mount | No | Yes | Virtual handles + prefetch + write-behind | Inline sync + demand paging + cache | TCP (3-channel) |
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
