# zlink — GPU-over-IP via CUDA RPC Pipeline

**Run CUDA workloads on CPU-only machines by forwarding GPU calls to a remote server.**

zlink combines [zpp_bits](https://github.com/eyalz800/zpp_bits) (C++20 binary serialization/RPC) with [r3map](https://github.com/pojntfx/r3map)-inspired remote memory to let you call functions in any shared library over the network — transparently.

## Key Features

- **Dependency-aware CUDA pipeline** — Batches multiple GPU calls into single network round-trips using virtual handles
- **Virtual handles** — Break the barrier chain: `cuMemAlloc` returns a virtual ID instead of blocking for the real pointer
- **Separated memory layer** — Host data syncs via the backend interface (ReadAt/WriteAt/Sync), not packed into RPC frames
- **Demand paging + write tracking** — Transparent read-side demand paging and write-side dirty tracking on shadow regions, with a three-tier runtime fallback: userfaultfd WP_ASYNC (kernel 6.2+) → userfaultfd WP sync (kernel 4.11+) → mprotect + SIGSEGV (any POSIX system, including containers where seccomp blocks userfaultfd)
- **Multiplexed transport** — 3 logical channels (RPC/bulk/prefetch), QUIC-ready
- **Zero codegen** — Explicit API declarations with zpp_bits `bind<>`, no Python scripts or generated stubs
- **Type-safe** — C++20 templates ensure compile-time type safety for all RPC calls
- **LZ4 compression** — Large memory transfers compressed automatically

## Architecture

```
  Client (CPU-only machine)                     Server (GPU machine)
  ┌──────────────────────────┐                  ┌──────────────────────────┐
  │ Application code          │                  │ zlink server              │
  │    ↓                      │                  │    ↓                      │
  │ cuda_pipeline<RpcDef>     │                  │ handle_pipeline_request() │
  │  (RPC batching + VH)      │                  │    ↓                      │
  │    ↓                      │  RPC channel     │ zpp_bits dispatch         │
  │ Transport (multiplexed)   │ ═════════════►   │ Transport (multiplexed)   │
  │    ↓                      │  ◄═════════════  │    ↓                      │
  │ memory_page_tracker + chunk_cache│ Bulk channel    │ host_memory_mirror        │
  │  (demand paging + cache)  │ ═════════════►   │  (server-side mirror)     │
  │                           │  ◄═════════════  │                           │
  │ Virtual handle allocator  │                  │ Handle table (VH→real)    │
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
| [wire-protocol.md](docs/wire-protocol.md) | Frame format and network protocol |
| [memory.md](docs/memory.md) | Remote memory, host mirror, chunk cache |
| [pointer-marshalling.md](docs/pointer-marshalling.md) | How pointers cross the network boundary |
| [compression.md](docs/compression.md) | LZ4 compression for memory transfers |
| [rpc-framework.md](docs/rpc-framework.md) | zpp_bits RPC engine and API declaration |
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

Memory data for `cuMemcpyHtoD` flows on the bulk channel, independent of RPC traffic.

## Wire Protocol

Each frame on the wire:

```
┌──────────────┬──────────────┬───────────┬──────────────────┐
│ Length (4B)  │ Call ID (4B) │ Type (1B) │ Payload (N bytes) │
│ uint32 BE    │ uint32 BE    │ uint8     │ zpp_bits data    │
└──────────────┴──────────────┴───────────┴──────────────────┘
```

The `pipeline_request` frame type (`0x04`) batches multiple RPC calls and
includes the virtual handle manifest. Memory data flows via separate
`memory_op` frames on the bulk channel.

See [wire-protocol.md](docs/wire-protocol.md) for full protocol details.

## Project Structure

```
zlink/
├── include/zlink/          # Public headers (mostly header-only)
│   ├── config.hpp          # Protocol constants, frame types
│   ├── transport.hpp       # Abstract transport + frame struct
│   ├── tcp_transport.hpp   # TCP transport implementation
│   ├── multiplexed_transport.hpp # 3-channel TCP (RPC/bulk/prefetch)
│   ├── rpc.hpp             # RPC engine (client, pipeline, server)
│   ├── cuda_pipeline.hpp   # Dependency-aware CUDA RPC pipeline
│   ├── virtual_handle.hpp  # Virtual handle system
│   ├── cuda_dep_spec.hpp   # CUDA API dependency categorization
│   ├── memory.hpp          # Remote memory subsystem
│   ├── chunk_cache.hpp     # Page-level cache (r3map-inspired)
│   ├── memory_page_tracker.hpp # Page tracking + demand paging: uffd WP / mprotect+SIGSEGV
│   ├── ptr_map.hpp         # Bidirectional pointer mapping
│   ├── compress.hpp        # LZ4 compression for memory transfers
├── src/                    # Implementation files
├── examples/
│   ├── libmath/            # Remote math example
│   └── cuda/               # CUDA RPC server + pipeline client
│       ├── cuda_api.hpp    # Hand-written CUDA Driver API RPC declarations
│       ├── cuda_server.cpp # GPU server: real CUDA calls + pipeline handler
│       ├── cuda_client.cpp # Pipeline client: virtual handles + readback test
│       └── CMakeLists.txt  # Builds cuda_server (needs CUDA) + cuda_client
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── cmake/                  # CMake modules (zpp_bits fetch)
```

## Comparison with Related Projects

| Project | Approach | Codegen | Generic | Pipelining | Memory | Transport |
|---------|----------|---------|---------|------------|--------|-----------|
| **zlink** | RPC pipeline + backend interface | No | Yes | Virtual handles + batched RPC | Separated memory layer + cache | TCP (3-channel, QUIC-ready) |
| **Lupine** | CUDA shim | Yes (Python) | No (CUDA only) | Basic | Handle remap | HTTP/2 |
| **RCUDA** | CUDA RPC | Yes | No | Basic | Basic | TCP |
| **SCUDA** | CUDA bridge | Yes (Python) | No | Basic | Basic | TCP |
| **r3map** | Remote mmap | N/A | N/A | N/A | Yes (NBD) | NBD |

## License

MIT

## Acknowledgments

- [zpp_bits](https://github.com/eyalz800/zpp_bits) — C++20 serialization & RPC
- [r3map](https://github.com/pojntfx/r3map) — Remote memory region mounting; research paper: [Networked Linux Memory Synchronization](https://pojntfx.github.io/networked-linux-memsync/main.pdf)
- [Lupine](https://github.com/lupinemachines/lupine) — GPU-over-IP inspiration
