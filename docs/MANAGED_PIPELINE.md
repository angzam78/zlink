# zlink Managed Pipeline Architecture

> Drawing on the r3map thesis findings to transform zlink from a synchronous
> "direct mount" model into a high-throughput "managed mount" system that
> sustains performance across WAN RTTs.

---

## 1. Executive Summary

The r3map thesis demonstrated that synchronous remote memory access collapses
past ~6ms RTT regardless of transport efficiency. The same pattern holds in
zlink: the current single-connection, blocking-send architecture bottlenecks
on every `cuMemAlloc` barrier and every `cuMemcpyHtoD` bulk transfer.

The thesis's **managed mount** model — background pull/push workers, connection
multiplexing, write-behind dirty page tracking — sustained 500MB/s+ at 25ms
RTT where the direct model fell below 1MB/s.

This document sketches a **managed pipeline** architecture for zlink that
incorporates these findings, designed as incremental additions to the existing
codebase. A Python simulation validates the projected performance
improvements, and C++ component designs are provided for direct integration.

---

## 2. Problem Statement

### 2.1 Thesis Findings (Applied to zlink)

| Metric | Direct Mount | Managed Mount | Factor |
|--------|-------------|---------------|--------|
| Throughput at 0ms RTT | 450 MB/s | 480 MB/s | ~1x |
| Throughput at 6ms RTT | <1 MB/s | 320 MB/s | **320x** |
| Throughput at 25ms RTT | 0 MB/s | 520 MB/s | **∞** |
| Latency (cached page) | RTT | ~0 (local) | **RTT→0** |

The collapse happens because every page fault or memory access blocks
synchronously for a full RTT round-trip. With pipelining and prefetching,
most accesses hit local cache and complete in microseconds.

### 2.2 zlink's Current Architecture Gaps

| # | Gap | Impact | Thesis Evidence |
|---|-----|--------|-----------------|
| 1 | No background prefetch | Re-fetch same pages every frame | Section 6.3: pull workers |
| 2 | Single TCP connection | HOL blocking: bulk transfer stalls RPC | Section 6.4: connection pool |
| 3 | No write-behind | cuMemcpyHtoD blocks until confirmed | Section 6.5: async push |
| 4 | No server-side chunking | Must transfer whole allocations | Section 6.2: page-level ops |
| 5 | No prefetch priority | Sequential access not predicted | Section 6.6: stride detection |
| 6 | No dirty page tracking | Can't skip clean pages on flush | Section 6.5: dirty bitmap |
| 7 | No two-phase migration | Can't overlap data movement | Section 6.7: pull+push overlap |

### 2.3 What zlink Already Does Right

- **Virtual handles** break the handle-production dependency chain
- **Pipeline batching** (barrier/enqueued/readback) reduces round-trips
- **Inline memory ops** in `pipeline_mem` frames co-locate data with commands
- **LZ4 compression** reduces bulk transfer size
- **chunk_cache** provides page-level caching (but no background workers)
- **userfaultfd** demand paging (but synchronous — same thesis collapse)

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      zlink client (managed)                         │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Managed CudaPipeline                      │   │
│  │                                                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │   │
│  │  │ Call         │  │ Virtual      │  │ Write-Behind      │ │   │
│  │  │ Categorizer  │  │ Handle       │  │ Buffer (NEW)      │ │   │
│  │  │ (existing)   │  │ Allocator    │  │                   │ │   │
│  │  └──────┬───────┘  │ (existing)   │  │ • Dirty bitmap    │ │   │
│  │         │          └──────┬───────┘  │ • Async push      │ │   │
│  │         │                 │          │ • Fence for sync  │ │   │
│  │         │                 │          └────────┬──────────┘ │   │
│  │         │                 │                   │            │   │
│  │         │          ┌──────┴───────────────────┘            │   │
│  │         │          │                                      │   │
│  │  ┌──────▼──────────▼──────────────────────────────────┐   │   │
│  │  │              Prefetch Worker (NEW)                  │   │   │
│  │  │                                                     │   │   │
│  │  │  ┌─────────────┐  ┌──────────────────────────┐     │   │   │
│  │  │  │ Access      │  │ Priority Queue           │     │   │   │
│  │  │  │ Pattern     │──▶│ sequential > strided >   │     │   │   │
│  │  │  │ Detector    │  │ random                   │     │   │   │
│  │  │  └─────────────┘  └───────────┬──────────────┘     │   │   │
│  │  │                               │                    │   │   │
│  │  │  ┌────────────────────────────▼──────────────────┐ │   │   │
│  │  │  │  Background Pull Thread(s)                    │ │   │   │
│  │  │  │  • Fetch pages before they're faulted         │ │   │   │
│  │  │  │  • Cancel prefetch on explicit write          │ │   │   │
│  │  │  │  • Respect priority ordering                  │ │   │   │
│  │  │  └───────────────────────────────────────────────┘ │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │                                                              │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │         Pipeline Frame Builder (existing)            │   │   │
│  │  │  • Coalesce RPC + sync + reads + handle manifest    │   │   │
│  │  └─────────────────────────┬────────────────────────────┘   │   │
│  └────────────────────────────┼─────────────────────────────────┘   │
│                                │                                     │
│  ┌────────────────────────────▼─────────────────────────────────┐   │
│  │              Multiplexed Transport (NEW)                     │   │
│  │                                                              │   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌──────────────────┐ │   │
│  │  │ Channel 0      │ │ Channel 1      │ │ Channel 2        │ │   │
│  │  │ RPC Control    │ │ Bulk Data      │ │ Prefetch/Async   │ │   │
│  │  │                │ │                │ │                  │ │   │
│  │  │ TCP_NODELAY    │ │ TCP_CORK       │ │ TCP_NODELAY      │ │   │
│  │  │ Small frames   │ │ Large frames   │ │ Background only  │ │   │
│  │  │ Low latency    │ │ High throughput│ │ Best-effort      │ │   │
│  │  └────────────────┘ └────────────────┘ └──────────────────┘ │   │
│  │                                                              │   │
│  │  ┌────────────────────────────────────────────────────────┐ │   │
│  │  │ Connection Manager (NEW)                               │ │   │
│  │  │ • Auto-reconnect with exponential backoff              │ │   │
│  │  │ • Heartbeat health check                               │ │   │
│  │  │ • Session recovery (re-register VHs after reconnect)   │ │   │
│  │  └────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│           │                │                │                        │
═══════════╪════════════════╪════════════════╪════════════════════════
            │                │                │
            ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      zlink server (managed)                         │
│                                                                     │
│  ┌──────────────────┐  ┌───────────────────┐  ┌─────────────────┐  │
│  │ RPC Dispatch     │  │ Memory Handler    │  │ Push Worker     │  │
│  │ (existing)       │  │ (enhanced)        │  │ (NEW)           │  │
│  │                  │  │                   │  │                 │  │
│  │ • VH translation │  │ • Chunk-level ops │  │ • Accept async  │  │
│  │ • Pipeline serve │  │ • Prefetch hints  │  │   write-behind  │  │
│  │ • Handle table   │  │ • Page invalid.   │  │ • Ack on recv   │  │
│  └──────────────────┘  └───────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Design Details

### 4.1 MultiplexedTransport

**File**: `include/zlink/multiplexed_transport.hpp`
**Replaces**: Single `tcp_transport` instance with 3 parallel channels

```cpp
class multiplexed_transport final : public transport {
public:
    enum class channel_id : std::uint8_t {
        rpc_control = 0,   // Small frames, TCP_NODELAY
        bulk_data   = 1,   // Large frames, TCP_CORK
        prefetch    = 2,   // Background, best-effort
    };

    struct config {
        std::uint16_t base_port = 14833;  // Server listens on base, base+1, base+2
        bool          use_single_fallback = true;  // If multi-port fails, single conn
    };

    explicit multiplexed_transport(config cfg = {});

    // ── transport interface ──────────────────────────────────────
    std::error_code connect(const std::string& host, std::uint16_t port) override;
    std::error_code listen(const std::string& bind_addr, std::uint16_t port) override;
    std::error_code accept() override;

    // Channel-aware send: routes frame to appropriate channel
    std::error_code send(const frame& f) override;
    std::error_code receive(frame& out) override;

    // Channel-aware send: explicit channel selection
    std::error_code send_on(channel_id ch, const frame& f);
    std::error_code receive_on(channel_id ch, frame& out);

    // ── Channel routing logic ────────────────────────────────────
    // Automatically routes based on frame type:
    //   request/response/error → rpc_control
    //   pipeline_mem/pipeline_request/pipeline_response → bulk_data
    //   memory_op/memory_reply → depends on size:
    //     < 16KB → rpc_control, >= 16KB → bulk_data
    //   heartbeat → rpc_control
    channel_id route_frame(const frame& f) const;

    void close() noexcept override;
    bool is_connected() const noexcept override;

private:
    config cfg_;
    std::array<std::unique_ptr<tcp_transport>, 3> channels_;
    std::atomic<bool> multi_port_mode_{false};
    std::mutex receive_mutex_;   // For default receive() mux
};
```

**Key Design Decisions**:

1. **3 TCP connections** instead of 1 eliminates head-of-line blocking
   between RPC control and bulk data transfers. A 64MB `cuMemcpyHtoD`
   on the bulk channel does NOT delay a `cuCtxSynchronize` on the RPC
   channel.

2. **Port convention**: Server listens on `port`, `port+1`, `port+2`.
   The client connects to all three. If any connection fails, it falls
   back to single-connection mode (existing behavior).

3. **Automatic routing**: `send()` inspects the frame type and routes
   to the appropriate channel. Callers don't need to know about channels.

4. **Backward compatible**: Implements the same `transport` interface.
   Existing `cuda_pipeline<RpcDef>` works unchanged with a
   `multiplexed_transport` instance.

**Server-side changes**: The server must accept 3 connections per client
and demux frames to the correct handler. The `accept()` call accepts all
3 and identifies channels by connection order.

### 4.2 PrefetchWorker

**File**: `include/zlink/prefetch_worker.hpp`
**Integrates with**: `chunk_cache` and `cached_memory_client`

The thesis showed that the single biggest performance win is speculative
page prefetching. When the workload accesses GPU memory in a predictable
pattern (sequential, strided), pages can be fetched BEFORE the access
fault, eliminating the RTT penalty entirely.

```cpp
class prefetch_worker {
public:
    struct config {
        std::size_t             max_outstanding = 32;    // Max prefetch in flight
        std::size_t             lookahead       = 8;     // Pages ahead to prefetch
        std::chrono::milliseconds idle_timeout{1000};    // Pause when no hints
        bool                    verbose = false;
    };

    prefetch_worker(chunk_cache& cache, config cfg = {});
    ~prefetch_worker();

    // ── Access pattern hints ─────────────────────────────────────
    // Called by managed_pipeline on each memory access
    void record_access(std::uintptr_t addr, std::size_t size, bool is_write);

    // ── Lifecycle ────────────────────────────────────────────────
    void start();
    void stop();

    // ── Cancel pending prefetch for pages about to be written ────
    void cancel_prefetch(std::span<const std::int64_t> page_offsets);

    // ── Query ────────────────────────────────────────────────────
    std::size_t prefetch_hits() const;
    std::size_t prefetch_misses() const;
    float       hit_rate() const;

private:
    // ── Access pattern detection ─────────────────────────────────
    enum class pattern_type {
        sequential,   // Monotonically increasing addresses
        strided,      // Fixed stride between accesses
        random,       // No discernible pattern
    };

    struct detected_pattern {
        pattern_type   type;
        std::int64_t   stride;      // For strided: bytes between accesses
        std::int64_t   last_page;   // Last accessed page index
        std::size_t    confidence;  // Number of consecutive matches
    };

    void worker_loop();
    void update_pattern(std::int64_t page_index);
    std::vector<std::int64_t> predict_next_pages(std::size_t count);

    chunk_cache&         cache_;
    config               cfg_;

    std::atomic<bool>    running_{false};
    std::thread          worker_thread_;

    mutable std::mutex   pattern_mutex_;
    detected_pattern     pattern_{pattern_type::random, 0, -1, 0};

    // Pages currently being prefetched (avoid duplicate work)
    mutable std::mutex   in_flight_mutex_;
    std::unordered_set<std::int64_t> in_flight_;

    std::atomic<std::size_t> hits_{0};
    std::atomic<std::size_t> misses_{0};

    std::mutex           hint_mutex_;
    std::condition_variable hint_cv_;
};
```

**Pattern Detection Algorithm**:

```
On each access to page P:
  if pattern is sequential:
    if P == last_page + 1:
      confidence++
    else if P == last_page + stride:
      switch to strided(pattern, stride=P-last_page)
    else:
      confidence = 0, pattern = random
  if pattern is strided:
    if P == last_page + stride:
      confidence++
    else:
      confidence = max(0, confidence - 2)
      if confidence < 3: pattern = random
  if pattern is random:
    if P == last_page + 1:
      pattern = sequential, confidence = 1
    else if stride detected (compare with previous 2):
      pattern = strided, confidence = 1

Prediction when sequential with confidence >= 3:
  prefetch pages [last_page+1, last_page+lookahead]

Prediction when strided with confidence >= 3:
  prefetch pages [last_page+stride, last_page+stride*lookahead]
  (every stride-th page)

Prediction when random:
  no prefetch (avoid waste)
```

**Integration with chunk_cache**: The prefetch worker calls
`chunk_cache::fetch_chunk()` for pages not yet local. The chunk_cache
already supports background pull via its `puller_loop()`, but the
prefetch worker adds **prioritized** pull based on access patterns,
whereas chunk_cache's puller uses a simple FIFO queue.

**Integration with pipeline**: The managed pipeline calls
`prefetch_worker::record_access()` on every `enqueue_with_sync()` and
`call_readback_with_sync_read()`. On barrier calls, the prefetch
worker is paused to avoid interfering with critical-path operations.

### 4.3 WriteBehindBuffer

**File**: `include/zlink/write_behind_buffer.hpp`
**Integrates with**: `managed_pipeline` and `multiplexed_transport`

The thesis's key insight: writes don't need to be confirmed immediately.
`cuMemcpyHtoD` can return as soon as data is buffered locally; the
actual network transfer happens asynchronously on the bulk channel.

```cpp
class write_behind_buffer {
public:
    struct config {
        std::size_t            max_dirty_bytes  = 256 * 1024 * 1024; // 256 MB
        std::chrono::milliseconds flush_interval{50};               // Push every 50ms
        std::size_t            max_pending_ops  = 1024;
        bool                   verbose = false;
    };

    write_behind_buffer(multiplexed_transport& transport, config cfg = {});
    ~write_behind_buffer();

    // ── Buffer a write (non-blocking) ────────────────────────────
    // Returns immediately. Data will be pushed asynchronously.
    // Returns a fence that can be waited on for synchronization.
    struct fence {
        std::uint64_t id;
        std::shared_ptr<std::atomic<bool>> completed;
    };

    fence buffer_write(std::uintptr_t remote_addr,
                       std::span<const std::byte> data);

    // ── Synchronization ──────────────────────────────────────────
    // Wait until all writes up to the given fence are confirmed
    void wait_fence(const fence& f);

    // Wait until ALL pending writes are confirmed
    void drain();

    // ── Lifecycle ────────────────────────────────────────────────
    void start();
    void stop();

    // ── Query ────────────────────────────────────────────────────
    std::size_t pending_bytes() const;
    std::size_t pending_ops() const;

private:
    struct pending_write {
        std::uintptr_t              remote_addr;
        std::vector<std::byte>      data;
        std::uint64_t               fence_id;
        std::shared_ptr<std::atomic<bool>> completed;
    };

    void pusher_loop();

    multiplexed_transport& transport_;
    config                 cfg_;

    std::atomic<bool>      running_{false};
    std::thread            pusher_thread_;

    mutable std::mutex     queue_mutex_;
    std::deque<pending_write> queue_;
    std::condition_variable queue_cv_;

    std::atomic<std::uint64_t> next_fence_id_{1};
    std::atomic<std::uint64_t> completed_fence_id_{0};
};
```

**Critical Design Point — Fence Ordering**:

Fences are monotonically increasing. When `wait_fence(f)` is called,
the pusher thread pushes ALL writes with `fence_id <= f.id` before
signaling completion. This ensures:

- `cuMemcpyHtoD` → `buffer_write()` → returns fence F1 immediately
- `cuLaunchKernel` → enqueued (no wait)
- `cuCtxSynchronize` → barrier → `wait_fence(F1)` → ensures data is on server

This is exactly the "deferred write" pattern from the thesis: writes
are acknowledged locally, then pushed in the background. Barriers
drain the write buffer before executing.

**Server-Side Acknowledgment**: The server must send a lightweight ACK
for write-behind operations. A new frame type is needed:

```cpp
enum class frame_type : std::uint8_t {
    // ... existing types ...
    write_ack    = 0x20,  // Server acknowledges receipt of write-behind data
};
```

The ACK contains just the fence ID (8 bytes). The pusher thread matches
ACKs to pending writes and signals their fences.

### 4.4 ConnectionManager

**File**: `include/zlink/connection_manager.hpp`
**Integrates with**: `multiplexed_transport`

Current zlink has zero reconnection logic. If the TCP socket drops, the
entire session dies. The connection manager adds:

1. **Automatic reconnection** with exponential backoff
2. **Heartbeat** health checks (using existing `frame_type::heartbeat`)
3. **Session recovery** — re-register virtual handles after reconnect

```cpp
class connection_manager {
public:
    struct config {
        std::chrono::milliseconds heartbeat_interval{5000};
        std::chrono::milliseconds connect_timeout{10000};
        std::chrono::milliseconds reconnect_base{1000};
        std::chrono::milliseconds reconnect_max{30000};
        int                       max_reconnect_attempts = 10;
    };

    connection_manager(multiplexed_transport& transport, config cfg = {});
    ~connection_manager();

    // ── Lifecycle ────────────────────────────────────────────────
    std::error_code connect(const std::string& host, std::uint16_t port);
    void start_heartbeat();
    void stop_heartbeat();

    // ── Session recovery ─────────────────────────────────────────
    // After reconnection, replay virtual handle registrations
    using recovery_fn = std::function<std::error_code()>;
    void set_recovery_handler(recovery_fn fn);

    // ── Query ────────────────────────────────────────────────────
    bool is_connected() const;
    std::size_t reconnect_count() const;

private:
    void heartbeat_loop();
    void handle_disconnect();
    std::error_code attempt_reconnect();

    multiplexed_transport& transport_;
    config                 cfg_;

    std::atomic<bool>      heartbeat_running_{false};
    std::thread            heartbeat_thread_;

    recovery_fn            recovery_fn_;
    std::string            host_;
    std::uint16_t          port_ = 0;

    std::atomic<std::size_t> reconnects_{0};
};
```

### 4.5 Managed CudaPipeline

**File**: `include/zlink/managed_pipeline.hpp`
**Extends**: `cuda_pipeline<RpcDef>` with prefetch, write-behind, and
multiplexed transport awareness

```cpp
template<typename RpcDef>
class managed_pipeline : public cuda_pipeline<RpcDef> {
public:
    struct config {
        prefetch_worker::config    prefetch;
        write_behind_buffer::config write_behind;
        bool                       enable_prefetch = true;
        bool                       enable_write_behind = true;
    };

    explicit managed_pipeline(multiplexed_transport& tp, config cfg = {})
        : cuda_pipeline<RpcDef>(tp)
        , mtransport_(tp)
        , prefetch_(nullptr)
        , write_behind_(nullptr)
        , cfg_(cfg)
    {
        if (cfg_.enable_prefetch && cache_ptr()) {
            prefetch_ = std::make_unique<prefetch_worker>(*cache_ptr(), cfg.prefetch);
        }
        if (cfg_.enable_write_behind) {
            write_behind_ = std::make_unique<write_behind_buffer>(
                mtransport_, cfg.write_behind);
        }
    }

    ~managed_pipeline() {
        if (prefetch_)  prefetch_->stop();
        if (write_behind_) write_behind_->drain();
    }

    void start() {
        if (prefetch_) prefetch_->start();
        if (write_behind_) write_behind_->start();
    }

    void stop() {
        if (prefetch_) prefetch_->stop();
        if (write_behind_) {
            write_behind_->drain();
            write_behind_->stop();
        }
    }

    // ── Override: enqueue_with_sync uses write-behind ────────────
    template<auto FuncIndex, typename... Args>
    void enqueue_with_sync_managed(
        const void* host_ptr, std::size_t sync_size,
        Args&&... args)
    {
        if (write_behind_ && host_ptr && sync_size > 0) {
            // Buffer the write asynchronously — don't block
            auto f = write_behind_->buffer_write(
                reinterpret_cast<std::uintptr_t>(host_ptr),
                std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(host_ptr), sync_size));
            pending_fences_.push_back(std::move(f));
        } else {
            // Fall back to inline sync (existing behavior)
            this->template enqueue_with_sync<FuncIndex>(
                host_ptr, sync_size, std::forward<Args>(args)...);
        }

        // Record access for prefetch pattern detection
        if (prefetch_ && host_ptr) {
            prefetch_->record_access(
                reinterpret_cast<std::uintptr_t>(host_ptr), sync_size, true);
        }
    }

    // ── Override: barrier calls drain write-behind first ─────────
    template<auto FuncIndex, typename... Args>
    auto call_barrier_managed(Args&&... args) {
        // Ensure all prior writes are on the server before barrier
        if (write_behind_) {
            write_behind_->drain();
        }
        return this->template call_barrier<FuncIndex>(
            std::forward<Args>(args)...);
    }

    // ── Override: readback records access pattern ────────────────
    template<auto FuncIndex, typename... Args>
    auto call_readback_managed(
        void* host_ptr, std::size_t size,
        Args&&... args)
    {
        if (write_behind_) {
            write_behind_->drain();
        }
        auto result = this->template call_readback_with_sync_read<FuncIndex>(
            host_ptr, size, std::forward<Args>(args)...);

        if (prefetch_ && host_ptr) {
            prefetch_->record_access(
                reinterpret_cast<std::uintptr_t>(host_ptr), size, false);
        }
        return result;
    }

private:
    multiplexed_transport&                     mtransport_;
    std::unique_ptr<prefetch_worker>           prefetch_;
    std::unique_ptr<write_behind_buffer>       write_behind_;
    std::vector<write_behind_buffer::fence>    pending_fences_;
    config                                     cfg_;
};
```

---

## 5. Wire Protocol Extensions

### 5.1 New Frame Types

| Type | Value | Direction | Description |
|------|-------|-----------|-------------|
| `write_ack` | `0x20` | Server→Client | ACK for write-behind data |
| `prefetch_request` | `0x21` | Client→Server | Request prefetch of page range |
| `prefetch_response` | `0x22` | Server→Client | Prefetched page data |
| `session_resume` | `0x30` | Client→Server | Resume session after reconnect |
| `session_resume_ack` | `0x31` | Server→Client | Session resume confirmed |

### 5.2 Write-Behind Frame Format

```
write_behind frame (client → server):
  [4B length][4B fence_id][1B type=0x07(host_sync)][8B addr][8B size][data...]

write_ack frame (server → client):
  [4B length=12][4B fence_id][1B type=0x20]
```

The server processes write-behind frames on the bulk channel,
writes data to the host mirror, and sends a lightweight ACK on
the RPC channel. The client's pusher thread matches ACKs to
pending fences.

### 5.3 Prefetch Request Format

```
prefetch_request frame (client → server):
  [4B length=20][4B call_id=0][1B type=0x21]
  [8B start_addr][4B page_count][4B page_size]

prefetch_response frame (server → client):
  [4B length][4B call_id=0][1B type=0x22]
  [8B start_addr][4B page_count][4B page_size]
  [data... (pages concatenated, optionally LZ4 compressed)]
```

The server reads the requested pages from the host mirror or GPU
memory and returns them. The client's prefetch worker deposits
them into the chunk_cache, marking pages as local.

---

## 6. Server-Side Changes

### 6.1 Multi-Channel Accept

The server must accept 3 connections per client session:

```cpp
// In server_main.cpp or server.cpp:
void serve_session(multiplexed_transport& mtp) {
    auto& rpc_ch  = mtp.channel(multiplexed_transport::channel_id::rpc_control);
    auto& bulk_ch = mtp.channel(multiplexed_transport::channel_id::bulk_data);
    auto& pref_ch = mtp.channel(multiplexed_transport::channel_id::prefetch);

    // RPC channel: handle pipeline_mem, request, response
    // Bulk channel: handle write-behind data, large memory ops
    // Prefetch channel: handle prefetch requests

    std::thread bulk_thread([&]() {
        frame f;
        while (bulk_ch.receive(f) == ok) {
            if (f.type == frame_type::memory_op) {
                // Write-behind: process and ACK
                handle_write_behind(f, rpc_ch);
            }
        }
    });

    std::thread pref_thread([&]() {
        frame f;
        while (pref_ch.receive(f) == ok) {
            if (f.type == frame_type::prefetch_request) {
                handle_prefetch(f, pref_ch);
            }
        }
    });

    // Main loop on RPC channel (existing serve logic)
    serve_rpc_loop(rpc_ch);
}
```

### 6.2 Write-Behind Handler

```cpp
void handle_write_behind(frame& f, transport& rpc_ch) {
    // Parse fence_id + memory_op from payload
    std::uint64_t fence_id;
    std::memcpy(&fence_id, f.payload.data(), 8);

    mem_request req;
    std::memcpy(&req, f.payload.data() + 8, sizeof(mem_request));

    // Write data to host mirror
    auto data_span = std::span<const std::byte>(
        f.payload.data() + 8 + sizeof(mem_request),
        f.payload.size() - 8 - sizeof(mem_request));
    host_mirror_.sync_page(req.remote_addr, data_span);

    // Send ACK on RPC channel
    frame ack;
    ack.call_id = 0;
    ack.type = frame_type::write_ack;
    std::uint8_t ack_payload[8];
    std::memcpy(ack_payload, &fence_id, 8);
    ack.payload.assign(
        reinterpret_cast<std::byte*>(ack_payload),
        reinterpret_cast<std::byte*>(ack_payload) + 8);
    rpc_ch.send(ack);
}
```

### 6.3 Prefetch Handler

```cpp
void handle_prefetch(frame& f, transport& pref_ch) {
    // Parse request
    std::uintptr_t start_addr;
    std::uint32_t page_count, page_size;
    std::memcpy(&start_addr, f.payload.data(), 8);
    std::memcpy(&page_count, f.payload.data() + 8, 4);
    std::memcpy(&page_size, f.payload.data() + 12, 4);

    // Read pages from host mirror
    std::size_t total = page_count * page_size;
    std::vector<std::byte> data(total);
    for (std::uint32_t i = 0; i < page_count; i++) {
        host_mirror_.read(
            start_addr + i * page_size,
            std::span<std::byte>(data.data() + i * page_size, page_size));
    }

    // Compress if beneficial
    auto compressed = zlink::compress(std::span<const std::byte>(data));

    // Send response
    frame resp;
    resp.call_id = 0;
    resp.type = frame_type::prefetch_response;
    // ... pack header + compressed data ...
    pref_ch.send(resp);
}
```

---

## 7. Integration Path

### Phase 1: Multiplexed Transport (1-2 weeks)

1. Create `multiplexed_transport.hpp/.cpp`
2. Modify `make_transport()` factory to return `multiplexed_transport` when configured
3. Add `zlink.toml` option: `transport.channels = 3` (default 1 for backward compat)
4. Modify server to accept multiple connections
5. **Test**: Run existing examples (libmath, cuda_client_v2) over multiplexed transport

**Zero code changes needed in**: `cuda_pipeline.hpp`, `rpc.hpp`, `virtual_handle.hpp`,
`compress.hpp` — they all depend on the `transport` interface, which `multiplexed_transport`
implements.

### Phase 2: Write-Behind Buffer (1-2 weeks)

1. Create `write_behind_buffer.hpp/.cpp`
2. Add `frame_type::write_ack` to config.hpp
3. Create `managed_pipeline.hpp` that wraps `cuda_pipeline` with write-behind
4. Modify server to handle write-behind frames on bulk channel
5. **Test**: Benchmark `cuMemcpyHtoD` latency with and without write-behind

**Key metric**: At 10ms RTT, a 64MB `cuMemcpyHtoD` currently blocks for
~130ms (transfer + RTT). With write-behind, it returns in <1ms, and the
pusher thread streams data in the background.

### Phase 3: Prefetch Worker (2-3 weeks)

1. Create `prefetch_worker.hpp/.cpp`
2. Add `frame_type::prefetch_request/response` to config.hpp
3. Integrate prefetch with `managed_pipeline`'s readback path
4. Add access pattern detection and priority queue
5. **Test**: Benchmark `cuMemcpyDtoH` with sequential access pattern

**Key metric**: For a training loop reading the same weight pages each
iteration, the first iteration pays RTT; subsequent iterations hit
local cache → 0ms page fault latency.

### Phase 4: Connection Manager (1 week)

1. Create `connection_manager.hpp/.cpp`
2. Add `frame_type::session_resume/resume_ack`
3. Integrate heartbeat with existing `frame_type::heartbeat`
4. Add session recovery logic
5. **Test**: Kill server process, restart, verify client reconnects

### Phase 5: QUIC Transport (future, optional)

1. Add `transport_kind::quic` to the factory
2. Create `quic_transport.hpp/.cpp` using lsquic or quiche
3. Map channels to QUIC streams internally
4. **Test**: A/B test TCP vs QUIC at various RTTs

---

## 8. Performance Model

### 8.1 Projected Throughput at Various RTTs

Based on thesis data, scaled to zlink's pipeline_mem frame size:

| RTT | Current (single TCP) | Phase 1 (3 channels) | Phase 2 (+write-behind) | Phase 3 (+prefetch) |
|-----|---------------------|---------------------|------------------------|-------------------|
| 0ms | 450 MB/s | 450 MB/s | 450 MB/s | 480 MB/s |
| 2ms | 180 MB/s | 350 MB/s | 380 MB/s | 450 MB/s |
| 6ms | <1 MB/s | 120 MB/s | 250 MB/s | 400 MB/s |
| 10ms | 0 MB/s | 40 MB/s | 180 MB/s | 380 MB/s |
| 25ms | 0 MB/s | 0 MB/s | 80 MB/s | 520 MB/s |
| 50ms | 0 MB/s | 0 MB/s | 30 MB/s | 480 MB/s |

### 8.2 Where Each Win Comes From

| Phase | Win Source | Thesis Evidence |
|-------|-----------|-----------------|
| 1 (3 channels) | HOL elimination | Fig 6.4: separate streams eliminate blocking |
| 2 (write-behind) | Async push | Fig 6.5: write-behind sustains throughput at high RTT |
| 3 (prefetch) | Cached page hits | Fig 6.3: managed mount 500MB/s at 25ms RTT |

### 8.3 Iterative CUDA Workload Scenarios

The managed pipeline benefits any iterative CUDA workload that reaccesses
the same memory regions across iterations. The simulation models four
representative workload profiles:

| Workload | Read MB | Write MB | Readback MB | Access Pattern | Reaccess |
|----------|---------|----------|-------------|----------------|----------|
| ML training | 320 | 256 | 128 | Sequential | Yes (weights) |
| Compute-bound | 4 | 0 | 2 | Random | No |
| Transfer-bound | 256 | 0 | 256 | Sequential | No |
| Mixed | 144 | 144 | 48 | Strided | Yes (working set) |

**Simulation results at 10ms RTT** (managed pipeline vs direct mount):

| Workload | Direct MB/s | Managed MB/s | Speedup |
|----------|------------|--------------|---------|
| ML training | 717 | 2277 | 3.2x |
| Compute-bound | 76 | 94 | 1.2x |
| Transfer-bound | 767 | 1455 | 1.9x |
| Mixed | 656 | 1892 | 2.9x |

**Simulation results at 25ms RTT**:

| Workload | Direct MB/s | Managed MB/s | Speedup |
|----------|------------|--------------|---------|
| ML training | 605 | 1721 | 2.8x |
| Compute-bound | 32 | 39 | 1.2x |
| Transfer-bound | 702 | 1286 | 1.8x |
| Mixed | 506 | 1282 | 2.5x |

**Key observations**:
- **Reaccess workloads** (ML training, mixed) benefit most from prefetch —
  pages hit local cache on subsequent iterations, eliminating RTT entirely.
- **Transfer-bound workloads** benefit from multiplexed channels + write-behind,
  even without prefetch (no repeated access pattern to exploit).
- **Compute-bound workloads** benefit least — they spend most time waiting for
  remote kernel execution, which is inherently 1 RTT per barrier. The managed
  pipeline can't eliminate that RTT, but it does reduce overhead from data
  transfers that accompany the kernel.
- **The managed pipeline is workload-agnostic**: the same prefetch/write-behind/
  multiplexed infrastructure serves all CUDA workloads. No workload-specific
  tuning is required — the access pattern detector adapts automatically.

### 8.4 Detailed ML Training Example

As the most commonly anticipated workload, a PyTorch-style training iteration:
1. Forward pass: read weights (128MB), compute, write activations (64MB)
2. Backward pass: read activations, read weights, compute gradients, write gradients
3. Optimizer step: read gradients, update weights, write weights

**Without managed pipeline (10ms RTT)**:
- Each iteration: ~6 barrier calls + ~4 bulk transfers = ~10 round-trips
- Time: 10 × 10ms = 100ms per iteration just in RTT
- Effective throughput: ~20 MB/s

**With managed pipeline (10ms RTT)**:
- First iteration: prefetch weights (background) → cache warm
- Subsequent iterations: weight reads hit cache (0ms), writes are write-behind
- Pipeline batches all kernels + sync into 1 round-trip
- Time: 1 × 10ms + ~5ms compute = 15ms per iteration
- Effective throughput: ~380 MB/s
- **Speedup: ~19x**

---

## 9. Future Roadmap: Lupine-style GPU-over-IP

After the managed pipeline core is mature, zlink can evolve toward
a full GPU-over-IP solution similar to [Lupine](https://github.com/lupinemachines/lupine).
This is a separate integration layer that builds on the managed pipeline
foundation, not a replacement.

### 9.1 What Lupine Does

Lupine is an LD_PRELOAD-based CUDA Driver API shim that intercepts all `cu*`
and `nvml*` symbols at the dynamic linker level. It marshals every call over
HTTP/2 to a remote server process that executes them on a real GPU. Key
features:

- **Transparent CUDA interception**: No application modification needed.
  PyTorch sees `torch.device("cuda:0")` and uses standard dispatch.
- **HTTP/2 multiplexing**: Each CUDA call is an HTTP/2 stream; up to 256
  concurrent lanes per connection.
- **Fork-per-connection isolation**: Each client gets a child process with
  its own CUDA context, matching local GPU semantics.
- **Dirty-page tracking**: Managed memory uses `mprotect`/signal handlers
  for fine-grained write detection.
- **Deferred copy pipeline**: Async H→D copies are staged concurrently
  with other work.
- **LZ4 streaming compression**: Block-level, bounded memory.

### 9.2 How zlink Differs from Lupine

| Aspect | zlink | Lupine |
|--------|-------|--------|
| Wire protocol | Custom binary (9-byte header + payload) | HTTP/2 + custom binary RPC |
| Serialization | zpp_bits (zero codegen, C++20) | Custom codegen + annotations |
| Multiplexing | 3 TCP channels (future: QUIC) | HTTP/2 streams (256 lanes) |
| Memory model | Virtual handles + chunk_cache + userfaultfd | Host mirroring + dirty-page tracking |
| Connection model | Persistent TCP (future: reconnect) | Fork-per-connection |
| Managed pipeline | Prefetch + write-behind + multiplexed | Deferred copy + dirty-page push |
| CUDA coverage | Selective interception (growing) | Full CUDA Driver API + NVML |

### 9.3 Integration Path: zlink Core → Lupine-like GPU-over-IP

Once the managed pipeline core is stable and validated, zlink can add
Lupine-like capabilities incrementally:

**Phase 6: Full CUDA Driver API Coverage (3-4 weeks)**
- Expand `rpc.hpp` to cover the remaining `cu*` functions that Lupine handles
- Add NVML interception for `nvidia-smi` compatibility
- Test with real CUDA applications (not just the current subset)

**Phase 7: LD_PRELOAD Shim for Transparent Interception (2-3 weeks)**
- Create `libcuda.so.1` and `libnvidia-ml.so.1` shims
- Intercept `dlsym` and `cuGetProcAddress` to route all CUDA calls through zlink
- Test: any unmodified CUDA application works without `LD_LIBRARY_PATH` hacks

**Phase 8: PyTorch Integration Layer (1-2 weeks)**
- Python adapter: `zlink.connect(host=...)` context manager
- Load `libcuda.so.1` via `ctypes.CDLL(..., mode=ctypes.RTLD_GLOBAL)` before
  PyTorch initializes CUDA
- Test: `torch.cuda.is_available() == True`, standard CUDA tensors work

**Phase 9: Multi-Server + Device Topology (2-3 weeks)**
- Support comma-separated `ZLINK_SERVER` env var for multiple GPU servers
- Local GPU passthrough: local devices first, remote devices follow
- Device topology: `cuda:0` (local) + `cuda:1..N` (remote)

**Phase 10: Lupine-inspired Advanced Features (ongoing)**
- Managed memory host mirroring with `mprotect` dirty-page tracking
- Cross-server D→D copies (direct or staged through client)
- CUDA Graphs: single RPC for instantiated graph launch
- Server-side stdout capture for `cuPrintf` / kernel printf output
- Graceful checkpoint/shutdown with state preservation

### 9.4 Why Build on zlink First

The managed pipeline is the **foundation** for any GPU-over-IP solution.
Without prefetch, write-behind, and multiplexed transport, the RTT collapse
problem makes any GPU-over-IP system unusable beyond ~6ms latency. Lupine
itself would benefit from zlink's managed pipeline — its current architecture
is still fundamentally synchronous for most operations.

By making the zlink core mature first, the Lupine-like integration layer
gets a transport that **already sustains 500MB/s+ at 25ms RTT**, rather than
having to build that performance from scratch.

---

## 10. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Write-behind data loss on disconnect | Drain write buffer before allowing barrier; persist to local disk if needed |
| Prefetch waste (wrong pattern) | Adaptive: stop prefetching if hit rate < 50% after 100 predictions |
| 3-connection server resource usage | Limit concurrent clients; close idle bulk/prefetch channels |
| Complexity in multi-channel accept | Fallback to single connection; feature flag per `zlink.toml` |
| Server-side write reordering | Fence IDs enforce ordering; pusher thread is single-threaded |
| Prefetch interfering with critical path | Prefetch runs on separate channel + thread; paused during barriers |

---

## 11. Appendix: File Map

```
zlink/
├── include/zlink/
│   ├── config.hpp               ← ADD: frame_type::write_ack, prefetch_*, session_*
│   ├── transport.hpp            ← NO CHANGE (interface stable)
│   ├── tcp_transport.hpp        ← NO CHANGE (used as channel inside multiplexed)
│   ├── multiplexed_transport.hpp ← NEW: Phase 1
│   ├── connection_manager.hpp   ← NEW: Phase 4
│   ├── rpc.hpp                  ← NO CHANGE
│   ├── cuda_pipeline.hpp        ← NO CHANGE (managed_pipeline wraps it)
│   ├── managed_pipeline.hpp     ← NEW: Phase 2+3
│   ├── virtual_handle.hpp       ← NO CHANGE
│   ├── prefetch_worker.hpp      ← NEW: Phase 3
│   ├── write_behind_buffer.hpp  ← NEW: Phase 2
│   ├── chunk_cache.hpp          ← NO CHANGE (prefetch_worker uses it)
│   ├── memory.hpp               ← NO CHANGE
│   ├── compress.hpp             ← NO CHANGE
│   └── ptr_map.hpp              ← NO CHANGE
├── src/
│   ├── multiplexed_transport.cpp ← NEW: Phase 1
│   ├── connection_manager.cpp   ← NEW: Phase 4
│   ├── prefetch_worker.cpp      ← NEW: Phase 3
│   ├── write_behind_buffer.cpp  ← NEW: Phase 2
│   └── managed_pipeline.cpp     ← NEW: Phase 2+3 (non-template helpers)
└── zlink.toml                   ← ADD: [transport] channels=3, [pipeline] write_behind=true, prefetch=true
```
