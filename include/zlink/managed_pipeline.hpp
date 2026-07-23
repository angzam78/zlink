#pragma once
// zlink/managed_pipeline.hpp — Dependency-aware CUDA pipeline with managed memory
//
// Extends cuda_pipeline<RpcDef> with three thesis-backed optimizations:
//
//   1. PREFETCH WORKER: Background pull of pages before they're faulted.
//      The single biggest win from the r3map thesis. For sequential/strided
//      access patterns, pages are fetched speculatively, eliminating RTT.
//
//   2. WRITE-BEHIND BUFFER: cuMemcpyHtoD returns immediately, actual
//      network transfer happens asynchronously on the bulk channel.
//      Barrier calls drain the buffer before executing.
//
//   3. MULTIPLEXED TRANSPORT: 3 TCP channels (RPC/bulk/prefetch)
//      eliminate head-of-line blocking between control and data.
//
// USAGE (drop-in replacement for cuda_pipeline):
//
//   // Before:
//   cuda_pipeline<cuda_v2_rpc> pipe(transport);
//
//   // After:
//   multiplexed_transport mtp;
//   mtp.connect(host, port);
//   managed_pipeline<cuda_v2_rpc> pipe(mtp);
//   pipe.start();  // Start prefetch + write-behind workers
//
//   // Then use exactly like cuda_pipeline:
//   pipe.call_barrier_managed<0>(flags);                    // instead of call_barrier
//   pipe.enqueue_with_sync_managed<18>(ptr, sz, ...);      // instead of enqueue_with_sync
//   pipe.call_readback_managed<19>(ptr, sz, ...);          // instead of call_readback_with_sync_read
//
// BACKWARD COMPATIBILITY:
//   All existing cuda_pipeline methods are inherited and work unchanged.
//   The *_managed variants add prefetch/write-behind integration.
//   You can mix managed and unmanaged calls in the same pipeline.

#include <zlink/cuda_pipeline.hpp>
#include <zlink/multiplexed_transport.hpp>
#include <zlink/prefetch_worker.hpp>
#include <zlink/write_behind_buffer.hpp>
#include <zlink/chunk_cache.hpp>

#include <memory>
#include <vector>

namespace zlink {

// ── Managed pipeline configuration ───────────────────────────────────
struct managed_pipeline_config {
    prefetch_config       prefetch;
    write_behind_config   write_behind;
    bool                  enable_prefetch     = true;
    bool                  enable_write_behind = true;
};

// ── Managed pipeline ─────────────────────────────────────────────────
template<typename RpcDef>
class managed_pipeline : public cuda_pipeline<RpcDef> {
public:
    using base_type = cuda_pipeline<RpcDef>;
    using rpc_type  = RpcDef;

    explicit managed_pipeline(multiplexed_transport& mtp)
        : base_type(mtp)
        , mtransport_(mtp)
        , cfg_{}
    {
        if (cfg_.enable_write_behind) {
            write_behind_ = std::make_unique<write_behind_buffer>(
                mtransport_.channel(multiplexed_transport::channel_id::bulk_data));
        }
    }

    explicit managed_pipeline(multiplexed_transport& mtp,
                              const managed_pipeline_config& cfg)
        : base_type(mtp)
        , mtransport_(mtp)
        , cfg_(cfg)
    {
        if (cfg_.enable_write_behind) {
            write_behind_ = std::make_unique<write_behind_buffer>(
                mtransport_.channel(multiplexed_transport::channel_id::bulk_data),
                cfg_.write_behind);
        }
    }

    ~managed_pipeline() {
        stop();
    }

    // ── Set the chunk cache for prefetch integration ─────────────
    void set_cache(chunk_cache* cache) {
        if (cfg_.enable_prefetch && cache) {
            prefetch_ = std::make_unique<prefetch_worker>(*cache, cfg_.prefetch);
        }
    }

    // ── Start/stop background workers ────────────────────────────
    void start() {
        if (prefetch_)  prefetch_->start();
        if (write_behind_) write_behind_->start();
    }

    void stop() {
        if (prefetch_)  prefetch_->stop();
        if (write_behind_) {
            write_behind_->drain();
            write_behind_->stop();
        }
    }

    // ── Managed barrier: drains write-behind first ───────────────
    // Ensures all prior writes are on the server before the barrier
    // call is sent. This maintains the correctness invariant:
    // "barrier means all prior work is complete."
    template<auto FuncIndex, typename... Args>
    auto call_barrier_managed(Args&&... args) {
        // Drain write-behind before barrier
        if (write_behind_) {
            write_behind_->drain();
        }

        // Pause prefetch during barrier (avoid interference)
        if (prefetch_) {
            prefetch_->pause();
        }

        // Execute barrier via base class
        auto result = this->template call_barrier<FuncIndex>(
            std::forward<Args>(args)...);

        // Resume prefetch after barrier
        if (prefetch_) {
            prefetch_->resume();
        }

        return result;
    }

    // ── Managed enqueue_with_sync: uses write-behind ─────────────
    // Currently falls back to inline sync because the server's
    // pipeline_mem handler expects sync data inline — it can't match
    // separately-received write-behind data to pending RPC calls yet.
    // Write-behind requires server-side protocol changes (TODO).
    template<auto FuncIndex, typename... Args>
    void enqueue_with_sync_managed(
        const void* host_ptr, std::size_t sync_size,
        Args&&... args)
    {
        // Fall back to inline sync (existing behavior)
        this->template enqueue_with_sync<FuncIndex>(
            host_ptr, sync_size, std::forward<Args>(args)...);

        // Record access for prefetch pattern detection
        if (prefetch_ && host_ptr) {
            prefetch_->record_access(
                reinterpret_cast<std::uintptr_t>(host_ptr), sync_size, true);
        }
    }

    // ── Managed enqueue_produces_handle_with_sync ────────────────
    // Same as above but returns a virtual handle. Also falls back to
    // inline sync until server-side write-behind support is added.
    template<auto FuncIndex, typename... Args>
    std::uint32_t enqueue_produces_handle_with_sync_managed(
        const void* host_ptr, std::size_t sync_size,
        Args&&... args)
    {
        // Fall back to inline sync (existing behavior)
        return this->template enqueue_produces_handle_with_sync<FuncIndex>(
            host_ptr, sync_size, std::forward<Args>(args)...);
    }

    // ── Managed readback: records access pattern ─────────────────
    // Drains write-behind first, then performs readback, then
    // records the access pattern for prefetch prediction.
    template<auto FuncIndex, typename... Args>
    auto call_readback_managed(
        void* host_ptr, std::size_t size,
        Args&&... args)
    {
        // Drain write-behind before readback
        if (write_behind_) {
            write_behind_->drain();
        }

        // Execute readback via base class
        auto result = this->template call_readback_with_sync_read<FuncIndex>(
            host_ptr, size, std::forward<Args>(args)...);

        // Record access for prefetch (read)
        if (prefetch_ && host_ptr) {
            prefetch_->record_access(
                reinterpret_cast<std::uintptr_t>(host_ptr), size, false);
        }

        return result;
    }

    // ── Override flush: drain write-behind first ─────────────────
    std::vector<pipe_result> flush_managed() {
        if (write_behind_) {
            write_behind_->drain();
        }
        return this->flush();
    }

    // ── Handle write_ack frames from server ──────────────────────
    // Should be called when a write_ack frame is received on the
    // RPC channel. Typically done in the main receive loop.
    void handle_write_ack(std::uint64_t fence_id) {
        if (write_behind_) {
            write_behind_->handle_ack(fence_id);
        }
    }

    // ── Access workers (for configuration/query) ─────────────────
    prefetch_worker*       prefetch()       { return prefetch_.get(); }
    const prefetch_worker* prefetch() const { return prefetch_.get(); }

    write_behind_buffer*       write_behind()       { return write_behind_.get(); }
    const write_behind_buffer* write_behind() const { return write_behind_.get(); }

private:
    multiplexed_transport&                     mtransport_;
    std::unique_ptr<prefetch_worker>           prefetch_;
    std::unique_ptr<write_behind_buffer>       write_behind_;
    std::vector<write_fence>                   pending_fences_;
    managed_pipeline_config                    cfg_;
};

} // namespace zlink
