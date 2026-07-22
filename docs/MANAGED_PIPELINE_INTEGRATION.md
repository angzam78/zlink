# zlink Managed Pipeline — Integration Guide

## How to Add Managed Pipeline to Your zlink Project

This guide shows how to incrementally add the managed pipeline components
to the existing zlink codebase, one phase at a time.

---

## Prerequisites

Copy these files into your zlink repository:

```
zlink/
├── include/zlink/
│   ├── config.hpp                          ← REPLACE (adds new frame types)
│   ├── multiplexed_transport.hpp           ← NEW
│   ├── prefetch_worker.hpp                 ← NEW
│   ├── write_behind_buffer.hpp             ← NEW
│   ├── connection_manager.hpp              ← NEW
│   └── managed_pipeline.hpp                ← NEW
├── src/
│   ├── multiplexed_transport.cpp           ← NEW
│   ├── prefetch_worker.cpp                 ← NEW
│   ├── write_behind_buffer.cpp             ← NEW
│   ├── connection_manager.cpp              ← NEW
│   └── managed_client_example.cpp          ← NEW (example)
```

---

## Phase 1: Multiplexed Transport (Lowest Risk)

### What changes

| File | Change |
|------|--------|
| `include/zlink/config.hpp` | Add new `frame_type` entries |
| `include/zlink/multiplexed_transport.hpp` | New file |
| `src/multiplexed_transport.cpp` | New file |
| `src/tcp_transport.cpp` | No change |
| `CMakeLists.txt` | Add new source files |

### CMakeLists.txt additions

```cmake
# Add to zlink_core static library sources
target_sources(zlink_core PRIVATE
    src/multiplexed_transport.cpp
)

# Add to include paths (no change needed if using existing include/zlink)
target_include_directories(zlink_core PUBLIC include)
```

### Server changes

The server needs to accept 3 connections instead of 1. In `server_main.cpp`:

```cpp
// Before:
auto tp = zlink::make_transport(zlink::transport_kind::tcp);
tp->listen("0.0.0.0", port);
tp->accept();

// After:
auto mtp = std::make_unique<zlink::multiplexed_transport>();
mtp->listen("0.0.0.0", port);
mtp->accept();
```

The multiplexed_transport handles the details internally. Your RPC
serve loop remains unchanged because `multiplexed_transport` implements
the same `transport` interface.

### Client changes

In your client or shim:

```cpp
// Before:
auto tp = zlink::make_transport(zlink::transport_kind::tcp);
tp->connect(host, port);

// After:
auto mtp = std::make_unique<zlink::multiplexed_transport>();
mtp->connect(host, port);
```

### Testing Phase 1

1. Run the existing `libmath` example over multiplexed_transport
2. Verify the server accepts 3 connections
3. Check that small RPC frames go on channel 0, bulk frames on channel 1
4. Test fallback: connect to a server that only has 1 port open

---

## Phase 2: Write-Behind Buffer

### What changes

| File | Change |
|------|--------|
| `include/zlink/write_behind_buffer.hpp` | New file |
| `src/write_behind_buffer.cpp` | New file |
| `include/zlink/managed_pipeline.hpp` | New file |
| Server | Add write_ack frame handling |

### Server changes

Add write-behind ACK handling in your serve loop:

```cpp
// In your server's memory_op handler:
if (frame.type == frame_type::memory_op) {
    // Parse fence_id (first 8 bytes of payload)
    std::uint64_t fence_id;
    std::memcpy(&fence_id, frame.payload.data(), 8);

    // Process the memory op (existing logic)
    handle_memory_op(frame.payload.data() + 8, frame.payload.size() - 8);

    // Send ACK on RPC channel
    frame ack;
    ack.call_id = 0;
    ack.type = frame_type::write_ack;
    std::uint8_t ack_data[8];
    std::memcpy(ack_data, &fence_id, 8);
    ack.payload.assign(
        reinterpret_cast<std::byte*>(ack_data),
        reinterpret_cast<std::byte*>(ack_data) + 8);

    // Send on RPC channel (not bulk!)
    rpc_transport.send(ack);
}
```

### Client changes

Use `managed_pipeline` instead of `cuda_pipeline`:

```cpp
// Before:
using pipe_t = zlink::cuda_pipeline<cuda_v2_rpc>;
pipe_t pipe(*transport);

// After:
using pipe_t = zlink::managed_pipeline<cuda_v2_rpc>;
pipe_t pipe(*mtransport);
pipe.start();  // Start write-behind worker
```

And replace sync calls with managed variants:

```cpp
// Before:
pipe.enqueue_with_sync<18>(host_data, size, dev_vh, addr, byte_count);

// After:
pipe.enqueue_with_sync_managed<18>(host_data, size, dev_vh, addr, byte_count);
```

### Testing Phase 2

1. Benchmark `cuMemcpyHtoD` latency with and without write-behind
2. At 10ms RTT, a 64MB write should return in <1ms (vs ~130ms blocking)
3. Verify barrier calls correctly drain the write buffer
4. Test error recovery: kill server during write-behind push

---

## Phase 3: Prefetch Worker

### What changes

| File | Change |
|------|--------|
| `include/zlink/prefetch_worker.hpp` | New file |
| `src/prefetch_worker.cpp` | New file |
| Server | Add prefetch_request/response handling |

### Server changes

Add prefetch request handling:

```cpp
// In a separate thread reading from the prefetch channel:
while (true) {
    frame req;
    auto ec = prefetch_transport.receive(req);
    if (ec) break;

    if (req.type == frame_type::prefetch_request) {
        // Parse: [8B start_addr][4B page_count][4B page_size]
        std::uintptr_t start_addr;
        std::uint32_t page_count, page_size;
        std::memcpy(&start_addr, req.payload.data(), 8);
        std::memcpy(&page_count, req.payload.data() + 8, 4);
        std::memcpy(&page_size, req.payload.data() + 12, 4);

        // Read pages from host mirror
        std::size_t total = page_count * page_size;
        std::vector<std::byte> data(total);
        for (std::uint32_t i = 0; i < page_count; i++) {
            host_mirror.read(start_addr + i * page_size,
                std::span<std::byte>(data.data() + i * page_size, page_size));
        }

        // Compress and send response
        auto compressed = zlink::compress(std::span<const std::byte>(data));
        // ... pack into prefetch_response frame ...
        prefetch_transport.send(resp);
    }
}
```

### Client changes

After creating `cached_memory_client`, pass its cache to the pipeline:

```cpp
// Create cached memory client (existing)
zlink::cached_memory_client mem_client(rpc_client, region_size);

// Connect the pipeline's prefetch worker to the cache
pipe.set_cache(&mem_client.cache());
```

### Testing Phase 3

1. Run an iterative CUDA workload that reads the same memory regions each iteration
2. First iteration: pages fetched from remote (cache miss)
3. Subsequent iterations: pages hit local cache (0ms, no network)
4. Verify prefetch pattern detection: sequential access should trigger lookahead

---

## Phase 4: Connection Manager

### Server changes

Minimal — just handle `session_resume` frames:

```cpp
if (frame.type == frame_type::session_resume) {
    // Re-register any state that was lost during disconnect
    // (virtual handle table, memory regions, etc.)

    frame ack;
    ack.type = frame_type::session_resume_ack;
    transport.send(ack);
}
```

### Client changes

```cpp
zlink::connection_manager conn_mgr(*mtp);
conn_mgr.set_recovery_handler([&]() -> std::error_code {
    // Re-register virtual handles, re-sync memory regions
    pipe.set_cache(nullptr);  // Cache is invalid after reconnect
    return {};
});
conn_mgr.connect(host, port);
conn_mgr.start_heartbeat();
```

---

## Performance Validation Checklist

After each phase, benchmark these scenarios:

| Scenario | Metric | Target |
|----------|--------|--------|
| `libmath` example | Latency per call | Same as before (no regression) |
| Small RPC (cuInit) | Latency at 10ms RTT | Same as before |
| Bulk transfer (64MB cuMemcpyHtoD) | Throughput at 0ms RTT | ~450 MB/s |
| Bulk transfer with write-behind | Return latency at 10ms RTT | <1ms |
| Iterative CUDA workload (20 iterations) | Total time at 10ms RTT | 2x faster than baseline |
| Iterative CUDA workload (20 iterations) | Total time at 25ms RTT | 2.3x faster than baseline |
| Server kill + restart | Client reconnect time | <5 seconds |

---

## Quick Reference: API Changes

### New Types

| Type | Header | Purpose |
|------|--------|---------|
| `multiplexed_transport` | `multiplexed_transport.hpp` | 3-channel TCP transport |
| `prefetch_worker` | `prefetch_worker.hpp` | Background page prefetch |
| `write_behind_buffer` | `write_behind_buffer.hpp` | Async write-behind |
| `write_fence` | `write_behind_buffer.hpp` | Fence for write completion |
| `connection_manager` | `connection_manager.hpp` | Auto-reconnect + heartbeat |
| `managed_pipeline<RpcDef>` | `managed_pipeline.hpp` | Full managed pipeline |

### New Frame Types

| Type | Value | Direction | Description |
|------|-------|-----------|-------------|
| `write_ack` | `0x20` | S→C | ACK for write-behind data |
| `prefetch_request` | `0x21` | C→S | Request prefetch of pages |
| `prefetch_response` | `0x22` | S→C | Prefetched page data |
| `session_resume` | `0x30` | C→S | Resume session after reconnect |
| `session_resume_ack` | `0x31` | S→C | Session resume confirmed |

### Method Mapping

| Old (cuda_pipeline) | New (managed_pipeline) | Difference |
|---------------------|----------------------|------------|
| `call_barrier<F>(...)` | `call_barrier_managed<F>(...)` | Drains write-behind first |
| `enqueue_with_sync<F>(ptr, sz, ...)` | `enqueue_with_sync_managed<F>(ptr, sz, ...)` | Uses write-behind |
| `call_readback_with_sync_read<F>(ptr, sz, ...)` | `call_readback_managed<F>(ptr, sz, ...)` | Records access pattern |
| `flush()` | `flush_managed()` | Drains write-behind first |
| N/A | `start()` | Start prefetch + write-behind |
| N/A | `stop()` | Stop workers |
| N/A | `handle_write_ack(fence_id)` | Process server ACK |

### Backward Compatibility

All existing `cuda_pipeline` methods are inherited by `managed_pipeline`.
You can use `managed_pipeline` as a drop-in replacement:

```cpp
// This still works — just doesn't use write-behind or prefetch:
pipe.enqueue_with_sync<18>(host_data, size, dev_vh, addr, byte_count);
pipe.call_barrier<0>(flags);
```

Mix managed and unmanaged calls as needed during migration.
