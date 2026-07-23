#pragma once
// zlink/cuda_pipeline.hpp — Dependency-aware CUDA RPC pipeline
//
// RPC BATCHING ONLY — no inline memory data.
//
// Memory data moves over a separate layer (cached_memory_client with
// memory_page_tracker + chunk_cache + userfaultfd demand paging). This pipeline
// carries ONLY serialized RPC calls + the virtual handle manifest.
//
// CALL CATEGORIES:
//
//   barrier   — caller MUST have the return value before the next call
//               (cuMemAlloc → dev_ptr, cuDeviceGetCount → count)
//               → Auto-flushes the pipeline BEFORE this call,
//                 then sends this call as a single synchronous RPC.
//
//   enqueued  — caller doesn't need the return value immediately
//               (cuMemcpyHtoD, cuLaunchKernel, cuCtxSynchronize, cuMemFree)
//               → Batches into the pipeline. Server processes in order.
//                 Return value (usually just CUresult) is collected at
//                 the next flush.
//
//   readback  — caller needs data back from the server AFTER execution
//               (cuMemcpyDtoH — needs GPU results)
//               → Auto-flushes the pipeline (so all prior work executes),
//                 then sends this call as a single synchronous RPC.
//                 Data comes back through the memory layer (host_read).
//
// TYPICAL WORKLOAD:
//
//   cuInit(0)                  → barrier: sync RPC
//   cuDeviceGetCount()         → barrier: sync RPC
//   cuMemAlloc(256)            → barrier: sync RPC → dev_ptr (virtual handle)
//   cuMemcpyHtoD(dev_ptr,...)  → enqueued: batch it (data synced separately)
//   cuLaunchKernel(...)        → enqueued: batch it
//   cuLaunchKernel(...)        → enqueued: batch it
//   cuCtxSynchronize()         → enqueued: batch it
//   cuMemcpyDtoH(out,dev_ptr)  → readback: flush batch, sync RPC, host_read
//
//   Net result: 3 sync RPCs + 1 pipelined batch + 1 readback RPC
//   Memory data for cuMemcpyHtoD flows on the bulk channel in the background.

#include <zlink/config.hpp>
#include <zlink/transport.hpp>
#include <zlink/memory.hpp>
#include <zlink/virtual_handle.hpp>

#include <zpp_bits.h>

#include <cstdint>
#include <cstring>
#include <vector>
#include <span>
#include <stdexcept>
#include <system_error>

namespace zlink {

// ── Pipeline call dependency category ──────────────────────────────────
enum class call_dep : std::uint8_t {
    barrier,    // Must flush pipeline, then send synchronously
    enqueued,   // Batch into pipeline — don't need return value now
    readback,   // Flush pipeline + this call + get data back
};

// ── Pipeline result holder ─────────────────────────────────────────────
struct pipe_result {
    std::vector<std::byte> data;  // Raw serialized response
    bool valid = false;
};

// ── Dependency-aware CUDA pipeline ─────────────────────────────────────
//
// Batches multiple RPC calls into a single pipeline_request frame.
// Memory data is handled by a separate layer — this class never touches
// host buffer contents.
template<typename RpcDef>
class cuda_pipeline {
public:
    using rpc_type = RpcDef;

    explicit cuda_pipeline(transport& tp) : transport_(tp) {}
    ~cuda_pipeline() = default;

    // ── Barrier call ──────────────────────────────────────────────
    // Flushes any pending batch, then sends this call as a single
    // synchronous RPC. The caller MUST have the result before the
    // next call (e.g., cuMemAlloc returns a device pointer).
    template<auto FuncIndex, typename... Args>
    auto call_barrier(Args&&... args) {
        flush_if_needed();
        return sync_call<FuncIndex>(std::forward<Args>(args)...);
    }

    // ── Enqueued call ─────────────────────────────────────────────
    // Batches this call into the pipeline. No network traffic yet.
    // The return value will be available after the next flush.
    // Use this for calls where you don't need the return value
    // immediately (e.g., cuMemcpyHtoD, cuLaunchKernel).
    template<auto FuncIndex, typename... Args>
    void enqueue(Args&&... args) {
        using namespace zpp::bits;
        auto [data, in, out] = data_in_out();
        typename rpc_type::client client{in, out};
        client.template request<FuncIndex>(std::forward<Args>(args)...).or_throw();

        queued_call qc;
        qc.func_index = static_cast<int>(FuncIndex);
        qc.rpc_data = std::move(data);
        calls_.push_back(std::move(qc));
    }

    // ── Enqueued call that produces a virtual handle ──────────────
    // Like enqueue(), but also allocates a virtual handle ID for the
    // return value. The manifest entry tells the server which call
    // produces a handle so it can register VH→real mapping.
    template<auto FuncIndex, typename... Args>
    std::uint64_t enqueue_produces_handle(Args&&... args) {
        std::uint32_t vh = vhandle_alloc_.allocate();
        handle_manifest_.push_back({
            static_cast<std::uint32_t>(calls_.size()), vh
        });
        enqueue<FuncIndex>(std::forward<Args>(args)...);
        return zlink::make_virtual_handle(vh);
    }

    // ── Readback call ─────────────────────────────────────────────
    // Flushes any pending batch (so all prior work executes on the
    // server), then sends this RPC call synchronously.
    // Memory data flows over the separate memory layer (demand paging
    // + write tracking via cached_memory_client).
    template<auto FuncIndex, typename... Args>
    auto call_readback(Args&&... args) {
        flush_if_needed();
        return sync_call<FuncIndex>(std::forward<Args>(args)...);
    }

    // ── Explicit flush ────────────────────────────────────────────
    // Sends all pending enqueued calls as a pipeline_request batch.
    // Returns results for each enqueued call (in order).
    std::vector<pipe_result> flush() {
        if (calls_.empty()) return {};
        return flush_pipeline();
    }

    // ── Flush if there are pending calls ──────────────────────────
    void flush_if_needed() {
        if (has_pending()) {
            flush();
        }
    }

    // ── Query pipeline state ──────────────────────────────────────
    std::size_t pending_count() const {
        return calls_.size();
    }

    bool has_pending() const {
        return !calls_.empty();
    }

private:
    // ── Queued RPC call ───────────────────────────────────────────
    struct queued_call {
        int func_index;
        std::vector<std::byte> rpc_data;  // Serialized zpp_bits request
    };

    // ── Synchronous single RPC call ──────────────────────────────
    template<auto FuncIndex, typename... Args>
    auto sync_call(Args&&... args) {
        using namespace zpp::bits;

        auto [req_data, req_in, req_out] = data_in_out();
        typename rpc_type::client req_client{req_in, req_out};
        req_client.template request<FuncIndex>(std::forward<Args>(args)...).or_throw();

        frame req_frame;
        req_frame.call_id = 1;
        req_frame.type = frame_type::request;
        req_frame.payload.assign(req_data.begin(), req_data.end());

        auto ec = transport_.send(req_frame);
        if (ec) throw std::system_error(ec);

        frame resp_frame;
        ec = transport_.receive(resp_frame);
        if (ec) throw std::system_error(ec);

        auto [resp_data, resp_in, resp_out] = data_in_out();
        resp_data.assign(resp_frame.payload.begin(), resp_frame.payload.end());
        typename rpc_type::client resp_client{resp_in, resp_out};
        return resp_client.template response<FuncIndex>().or_throw();
    }

    // ── Flush pending calls as a pipeline_request batch ──────────
    // Wire format (pipeline_request):
    //   [4B rpc_count]
    //   for each rpc: [4B rpc_len][rpc_bytes]
    //   [handle_manifest]
    std::vector<pipe_result> flush_pipeline() {
        if (calls_.empty()) return {};

        std::vector<std::byte> payload;
        payload.resize(4);
        std::uint32_t count = static_cast<std::uint32_t>(calls_.size());
        std::memcpy(payload.data(), &count, 4);

        for (auto& c : calls_) {
            std::uint32_t len = static_cast<std::uint32_t>(c.rpc_data.size());
            std::size_t prev = payload.size();
            payload.resize(prev + 4 + len);
            std::memcpy(payload.data() + prev, &len, 4);
            std::memcpy(payload.data() + prev + 4, c.rpc_data.data(), len);
        }

        // Append handle manifest so server can register VH→real mappings
        zlink::serialize_handle_manifest(handle_manifest_, payload);

        frame req_frame;
        req_frame.call_id = 1;
        req_frame.type = frame_type::pipeline_request;
        req_frame.payload = payload;

        auto ec = transport_.send(req_frame);
        if (ec) throw std::system_error(ec);

        frame resp_frame;
        ec = transport_.receive(resp_frame);
        if (ec) throw std::system_error(ec);

        if (resp_frame.type != frame_type::pipeline_response) {
            throw std::runtime_error("Expected pipeline_response frame");
        }

        auto results = parse_pipeline_response(resp_frame.payload);
        calls_.clear();
        handle_manifest_.clear();
        return results;
    }

    // ── Parse pipeline_response payload ──────────────────────────
    // Format: [4B count][4B len1][resp1][4B len2][resp2]...
    std::vector<pipe_result> parse_pipeline_response(
        std::span<const std::byte> payload)
    {
        if (payload.size() < 4) return {};

        std::uint32_t count = 0;
        std::memcpy(&count, payload.data(), 4);

        std::vector<pipe_result> results(count);
        std::size_t offset = 4;

        for (std::uint32_t i = 0; i < count && offset + 4 <= payload.size(); i++) {
            std::uint32_t len = 0;
            std::memcpy(&len, payload.data() + offset, 4);
            offset += 4;

            if (offset + len > payload.size()) break;

            results[i].data.resize(len);
            std::memcpy(results[i].data.data(), payload.data() + offset, len);
            results[i].valid = true;
            offset += len;
        }

        return results;
    }

    transport& transport_;
    std::vector<queued_call> calls_;
    std::vector<zlink::handle_manifest_entry> handle_manifest_;
    zlink::virtual_handle_allocator vhandle_alloc_;
};

// ── Helper: deserialize a pipe_result into a typed return value ──────
template<typename RpcDef, auto FuncIndex, typename RetType>
RetType pipeline_result_get(const pipe_result& result) {
    using namespace zpp::bits;
    auto [data, in, out] = data_in_out();
    data.assign(result.data.begin(), result.data.end());
    typename RpcDef::client client{in, out};
    return client.template response<FuncIndex>().or_throw();
}

} // namespace zlink
