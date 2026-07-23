# zlink — Repository Knowledge

## Project Overview
zlink is a C++20 GPU-over-IP system. A client sends CUDA Driver API calls over TCP
to a server with real GPU hardware, using a batched pipeline protocol with virtual
handles to minimize round-trips. Memory data flows on a separate channel from RPC
traffic.

## Build
- GPU machine: `cmake -DZLINK_CUDA_EXAMPLES=ON .. && make -j$(nproc)`
- CPU-only (client only): same cmake, server target is skipped without CUDAToolkit
- Dependencies: zpp_bits (RPC serialization, fetched by CMake), LZ4 (compression), r3map concepts

## GPU Machine Access
- SSH: `ssh -p 2222 -i ~/.ssh/zlink_gpu root@192.168.1.130`
- 2 GPUs: RTX 3090 + RTX 3080, CUDA 12.8, driver 580.x
- Repo at `/root/zlink` on branch `reimplement-cuda-pipeline`
- No rsync — use git push/pull to sync code

## Testing (same host)
```bash
killall -9 cuda_server cuda_client; sleep 3
nohup /root/zlink/build/examples/cuda/cuda_server > /tmp/zlink_server.log 2>&1 &
sleep 3
/root/zlink/build/examples/cuda/cuda_client 127.0.0.1 14833
```

## Architecture: Separated RPC + Memory Layers
- RPC frames carry only call metadata + virtual handles (no memory data)
- Memory data flows via the backend interface (ReadAt/WriteAt/Sync) on the bulk channel
- `cuda_pipeline<RpcDef>` handles RPC batching and virtual handle manifest
- `host_sync` / `host_read` memory ops handle data transfer via `host_memory_mirror`
- `multiplexed_transport` provides 3 TCP channels (RPC/bulk/prefetch), QUIC-ready
- `multiplexed_transport` auto-falls back to single-connection if multi-port unavailable

## Key Files
- `examples/cuda/cuda_client.cpp` — pipeline client with virtual handles
- `examples/cuda/cuda_server.cpp` — CUDA RPC server with pipeline_request handler
- `include/zlink/cuda_pipeline.hpp` — RPC batching + virtual handles (no memory data)
- `include/zlink/multiplexed_transport.hpp` — 3-channel TCP transport
- `include/zlink/virtual_handle.hpp` — virtual handle system + manifest serialization
- `include/zlink/memory.hpp` — host_memory_mirror, mem_request/mem_response
- `include/zlink/chunk_cache.hpp` — page-level cache (r3map-inspired)
- `include/zlink/shared_mem.hpp` — backend interface (ReadAt/WriteAt/Size/Sync)
- `include/zlink/config.hpp` — frame types, `default_port = 14833`
- `src/multiplexed_transport.cpp` — channel routing logic

## Frame Types (config.hpp)
- `request` (0x01), `response` (0x02), `error` (0x03) — single RPC
- `pipeline_request` (0x04), `pipeline_response` (0x05) — batched RPC + handle manifest
- `memory_op` (0x10), `memory_reply` (0x11) — host_sync / host_read
- `heartbeat` (0xFF)

## Planned Enhancements
1. **Write tracking** — mprotect + SIGSEGV handler (lupine-style dirty page tracking)
2. **Demand paging** — userfaultfd for read-side page faults
3. **QUIC transport** — migrate from TCP to QUIC streams (3 channels map naturally)
