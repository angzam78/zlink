#pragma once
// zlink/cuda_pipeline.hpp — Dependency-aware CUDA RPC pipeline
//
// THE CORE PROBLEM:
//   CUDA workloads are sequential — cuMemAlloc returns a device pointer
//   that cuMemcpyHtoD needs. You can't blindly pipeline all calls
//   because some depend on return values of previous calls.
//
// THE SOLUTION:
//   Categorize each CUDA call by its dependency type:
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
//                 then sends the DtoH RPC + host_read as one round-trip.
//
// TYPICAL WORKLOAD WITH AUTO-FLUSH:
//
//   cuInit(0)                  → barrier: flush (empty), sync RPC
//   cuDeviceGetCount()         → barrier: flush (empty), sync RPC
//   cuMemAlloc(256)            → barrier: flush (empty), sync RPC → dev_ptr
//   cuMemcpyHtoD(dev_ptr,...)  → enqueued: batch it!
//   cuLaunchKernel(...)        → enqueued: batch it!
//   cuLaunchKernel(...)        → enqueued: batch it!
//   cuCtxSynchronize()         → enqueued: batch it!
//   cuMemcpyDtoH(out,dev_ptr)  → readback: flush batch (1 round-trip for
//                                 all enqueued calls), then sync DtoH
//
//   Net result: 3 sync RPCs (init, get_count, alloc) +
//               1 pipelined batch (memcpy+2 kernels+sync) +
//               1 readback RPC = 5 round-trips instead of 8
//               With larger workloads the savings grow dramatically.
//
// PIPELINE+MEM FRAME (frame_type::pipeline_mem):
//   Further optimization: inline memory ops into the pipeline frame.
//   Instead of separate host_sync + RPC round-trips for cuMemcpyHtoD,
//   the sync data is packed into the pipeline frame itself:
//
//   Wire format (pipeline_mem request):
//     [4B sync_count]
//     for each sync: [8B addr][8B original_size][1B comp_flag][4B data_size][data...]
//       comp_flag: 0=raw (data=original bytes), 1=LZ4 (data=compressed)
//       data_size: number of bytes in data field (wire bytes, not original)
//     [4B rpc_count]
//     for each rpc: [4B rpc_len][rpc_bytes]
//     [4B read_count]
//     for each read: [8B addr][8B size]
//     [handle_manifest]
//
//   Wire format (pipeline_mem response):
//     [4B rpc_count]
//     for each rpc: [4B resp_len][resp_bytes]
//     [4B read_count]
//     for each read: [4B data_len][1B comp_flag][data...]
//       comp_flag: 0=raw, 1=LZ4
//       When LZ4: data = [8B original_size][compressed_bytes]
//
//   This eliminates 2 extra round-trips per cuMemcpyHtoD and cuMemcpyDtoH!

#include <zlink/config.hpp>
#include <zlink/transport.hpp>
#include <zlink/memory.hpp>
#include <zlink/virtual_handle.hpp>
#include <zlink/compress.hpp>

#include <zpp_bits.h>

#include <cstdint>
#include <cstring>
#include <vector>
#include <span>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <functional>
#include <cassert>

namespace zlink {

// ── Pipeline call dependency category ──────────────────────────────────
enum class call_dep : std::uint8_t {
    barrier,    // Must flush pipeline, then send synchronously
    enqueued,   // Batch into pipeline — don't need return value now
    readback,   // Flush pipeline + this call + get data back
};

// ── Pending memory sync entry (client → server data) ──────────────────
struct pending_sync {
    std::uintptr_t  client_addr;
    std::vector<std::byte> data;
};

// ── Pending memory read request (server → client data) ────────────────
struct pending_read {
    std::uintptr_t  client_addr;  // Client's host buffer address
    std::size_t     size;         // Bytes to read back
};

// ── Pipeline result holder ─────────────────────────────────────────────
struct pipe_result {
    std::vector<std::byte> data;  // Raw serialized response
    bool valid = false;
};

// ── Dependency-aware CUDA pipeline ─────────────────────────────────────
//
// This is the main class that CUDA wrapper functions use.
// It tracks pending enqueued calls, pending memory syncs, and pending
// memory reads. When a barrier or readback call is made, it auto-flushes.
//
// Usage from wrapper functions:
//
//   // barrier call — returns typed result
//   auto r = pipe.call_barrier<7>(bytesize);  // cuMemAlloc
//   uint64_t dev_ptr = r.dev_ptr;
//
//   // enqueued call — batches, returns void (deferred result)
//   pipe.enqueue<9>(dev_ptr, client_addr, byte_count);  // cuMemcpyHtoD
//
//   // enqueued with inline host_sync — no separate round-trip!
//   pipe.enqueue_with_sync<9>(host_ptr, size, dev_ptr, client_addr, byte_count);
//
//   // readback call — flushes + does DtoH
//   auto r = pipe.call_readback<10>(client_addr, dev_ptr, byte_count);
//   // then pipe.host_read(host_ptr, size) — or inline in readback
//
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
        flush_if_needed();  // Execute any pending enqueued calls first

        // Now send this call as a single synchronous RPC
        return sync_call<FuncIndex>(std::forward<Args>(args)...);
    }

    // ── Enqueued call (no inline memory sync) ────────────────────
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

    // ── Enqueued call with inline host_sync ──────────────────────
    // Like enqueue(), but also queues a host_sync for the given
    // host memory region. When the pipeline flushes, the sync data
    // is packed into the pipeline_mem frame BEFORE the RPC calls,
    // so the server has the data ready when it processes the calls.
    // This eliminates the separate host_sync round-trip!
    template<auto FuncIndex, typename... Args>
    void enqueue_with_sync(const void* host_ptr, std::size_t sync_size,
                            Args&&... args) {
        // Queue the memory sync
        if (host_ptr && sync_size > 0) {
            pending_sync ps;
            ps.client_addr = reinterpret_cast<std::uintptr_t>(host_ptr);
            ps.data.resize(sync_size);
            std::memcpy(ps.data.data(), host_ptr, sync_size);
            syncs_.push_back(std::move(ps));
        }

        // Queue the RPC call
        enqueue<FuncIndex>(std::forward<Args>(args)...);
    }

    // ── Enqueued call that produces a virtual handle ──────────────
    // Like enqueue(), but this call produces a handle (e.g., cuMemAlloc
    // returns a device pointer, cuCtxCreate returns a context handle).
    // Returns a virtual handle ID that subsequent calls can reference.
    // The server translates VH → real handle when processing the batch.
    //
    // This is the KEY method: it makes cuMemAlloc non-blocking!
    // Previously: cuMemAlloc → BARRIER (wait for dev_ptr)
    // Now:        cuMemAlloc → enqueue_produces_handle → VH(0) → ENQUEUED!
    template<auto FuncIndex, typename... Args>
    std::uint32_t enqueue_produces_handle(Args&&... args) {
        // Allocate a virtual handle ID
        std::uint32_t vh_id = vhandle_alloc_.allocate();

        // Enqueue the call
        enqueue<FuncIndex>(std::forward<Args>(args)...);

        // Record that this call produces a handle
        // The call index is calls_.size() - 1 (just pushed)
        handle_manifest_entry entry;
        entry.call_index = static_cast<std::uint32_t>(calls_.size() - 1);
        entry.virtual_id = vh_id;
        entry.return_field = 1;  // Convention: handle is field 1 (after result field 0)
        entry._reserved = 0;
        handle_manifest_.push_back(entry);

        return vh_id;
    }

    // ── Enqueued call that produces a virtual handle + inline host_sync ──
    // Combines: queue host_sync + enqueue RPC + record handle manifest entry.
    // Used for cuModuleLoadData (sync the image + enqueue load + get VH back).
    template<auto FuncIndex, typename... Args>
    std::uint32_t enqueue_produces_handle_with_sync(
        const void* host_ptr, std::size_t sync_size,
        Args&&... args)
    {
        // Queue the memory sync FIRST (server needs data before RPC)
        if (host_ptr && sync_size > 0) {
            pending_sync ps;
            ps.client_addr = reinterpret_cast<std::uintptr_t>(host_ptr);
            ps.data.resize(sync_size);
            std::memcpy(ps.data.data(), host_ptr, sync_size);
            syncs_.push_back(std::move(ps));
        }

        // Allocate a virtual handle ID
        std::uint32_t vh_id = vhandle_alloc_.allocate();

        // Enqueue the RPC call
        enqueue<FuncIndex>(std::forward<Args>(args)...);

        // Record that this call produces a handle
        handle_manifest_entry entry;
        entry.call_index = static_cast<std::uint32_t>(calls_.size() - 1);
        entry.virtual_id = vh_id;
        entry.return_field = 1;  // Convention: handle is field 1
        entry._reserved = 0;
        handle_manifest_.push_back(entry);

        return vh_id;
    }

    // ── Queue only a host_sync (no RPC call) ────────────────────────
    // Useful when you need to sync data that will be referenced by
    // a subsequent enqueue_produces_handle call (e.g., module image
    // data for cuModuleLoadData).
    void queue_host_sync(const void* host_ptr, std::size_t sync_size) {
        if (!host_ptr || sync_size == 0) return;
        pending_sync ps;
        ps.client_addr = reinterpret_cast<std::uintptr_t>(host_ptr);
        ps.data.resize(sync_size);
        std::memcpy(ps.data.data(), host_ptr, sync_size);
        syncs_.push_back(std::move(ps));
    }

    // ── Readback call ─────────────────────────────────────────────
    // Flushes any pending batch (so all prior work executes on the
    // server), then sends this RPC call synchronously. After the
    // RPC reply, performs a host_read to get the data back.
    //
    // This is for cuMemcpyDtoH: the server needs to execute all
    // prior GPU work, then copy from GPU to mirror, then we read
    // the mirror data back.
    template<auto FuncIndex, typename... Args>
    auto call_readback(Args&&... args) {
        flush_if_needed();  // Execute all prior enqueued calls

        // Send this call synchronously
        return sync_call<FuncIndex>(std::forward<Args>(args)...);
    }

    // ── Readback with inline host_sync + host_read ───────────────
    // The full cuMemcpyDtoH pattern:
    //   1. host_sync the destination buffer (ensures mirror exists)
    //   2. RPC call (server copies GPU → mirror)
    //   3. host_read (pull mirror data back to client)
    //
    // Steps 1-3 can be combined into a single pipeline_mem frame
    // if there are also pending enqueued calls to flush.
    // Otherwise they go as individual frames.
    template<auto FuncIndex, typename... Args>
    auto call_readback_with_sync_read(
        void* host_ptr, std::size_t size,
        Args&&... args)
    {
        if (host_ptr && size > 0) {
            // Queue the pre-sync (ensures mirror region exists)
            pending_sync ps;
            ps.client_addr = reinterpret_cast<std::uintptr_t>(host_ptr);
            ps.data.resize(size, std::byte{0});  // Zero-fill — just to create mirror
            syncs_.push_back(std::move(ps));
        }

        // Queue the read request (will be fulfilled after RPC)
        pending_read pr;
        pr.client_addr = reinterpret_cast<std::uintptr_t>(host_ptr);
        pr.size = size;
        reads_.push_back(pr);

        // Queue the RPC call itself
        enqueue<FuncIndex>(std::forward<Args>(args)...);

        // Now flush everything: syncs + RPC + reads
        return flush_with_reads();
    }

    // ── Host read (pull data from server mirror) ─────────────────
    // Sends a host_read frame and receives the data back.
    // Used after a readback RPC when not using the combined frame.
    void host_read(void* host_ptr, std::size_t size) {
        if (!host_ptr || size == 0) return;

        mem_request req;
        req.op = mem_op::host_read;
        req.remote_addr = reinterpret_cast<std::uintptr_t>(host_ptr);
        req.size = size;

        std::vector<std::byte> req_data(sizeof(mem_request));
        std::memcpy(req_data.data(), &req, sizeof(mem_request));

        frame read_frame;
        read_frame.call_id = 0;
        read_frame.type = frame_type::memory_op;
        read_frame.payload = req_data;

        auto ec = transport_.send(read_frame);
        if (ec) throw std::system_error(ec);

        frame resp_frame;
        ec = transport_.receive(resp_frame);
        if (ec) throw std::system_error(ec);

        if (resp_frame.type == frame_type::memory_reply &&
            resp_frame.payload.size() >= sizeof(mem_response)) {
            mem_response resp;
            std::memcpy(&resp, resp_frame.payload.data(), sizeof(resp));

            if (ok(resp.status) && resp.size > 0) {
                const std::byte* data_start = resp_frame.payload.data() + sizeof(mem_response);
                std::size_t data_size = resp_frame.payload.size() - sizeof(mem_response);
                std::size_t copy_size = std::min(data_size, size);
                std::memcpy(host_ptr, data_start, copy_size);
            }
        }
    }

    // ── Explicit flush ────────────────────────────────────────────
    // Sends all pending enqueued calls + syncs as a batch.
    // Returns results for each enqueued call (in order).
    std::vector<pipe_result> flush() {
        if (calls_.empty() && syncs_.empty() && reads_.empty()) return {};

        // IMPORTANT: If we have ANY handle manifest entries, we MUST use
        // pipeline_mem frame so the server receives the VH→call mapping.
        // Without the manifest, the server can't register VH IDs for
        // handle-producing calls (cuEventCreate, cuStreamCreate, etc.)
        if (!handle_manifest_.empty() || !syncs_.empty()) {
            return flush_pipeline_mem();
        }

        // No handle manifests and no syncs: use simple pipeline_request
        std::vector<pipe_result> results;
        if (!calls_.empty()) {
            results = flush_pipeline();
        }

        reads_.clear();
        return results;
    }

    // ── Flush with pending reads ──────────────────────────────────
    // Like flush(), but also processes pending host_read requests
    // using inline data from the pipeline_mem response.
    std::vector<pipe_result> flush_with_reads() {
        if (calls_.empty() && syncs_.empty() && reads_.empty()) return {};

        // Use pipeline_mem frame to combine everything
        return flush_pipeline_mem_with_reads();
    }

    // ── Query pipeline state ──────────────────────────────────────
    std::size_t pending_count() const {
        return calls_.size() + syncs_.size() + reads_.size();
    }

    bool has_pending() const {
        return !calls_.empty() || !syncs_.empty() || !reads_.empty();
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

    // ── Send a host_sync frame ────────────────────────────────────
    void send_host_sync(std::uintptr_t client_addr,
                         const std::vector<std::byte>& data) {
        mem_request req;
        req.op = mem_op::host_sync;
        req.remote_addr = client_addr;
        req.size = data.size();

        std::vector<std::byte> req_data(sizeof(mem_request) + data.size());
        std::memcpy(req_data.data(), &req, sizeof(mem_request));
        std::memcpy(req_data.data() + sizeof(mem_request), data.data(), data.size());

        frame sync_frame;
        sync_frame.call_id = 0;
        sync_frame.type = frame_type::memory_op;
        sync_frame.payload = req_data;

        auto ec = transport_.send(sync_frame);
        if (ec) throw std::system_error(ec);

        // Wait for ack
        frame resp_frame;
        ec = transport_.receive(resp_frame);
        if (ec) throw std::system_error(ec);
    }

    // ── Flush pending calls as a standard pipeline ───────────────
    std::vector<pipe_result> flush_pipeline() {
        if (calls_.empty()) return {};

        // Build pipeline_request payload
        std::uint32_t count = static_cast<std::uint32_t>(calls_.size());
        std::size_t total_size = 4; // count prefix
        for (auto& c : calls_) total_size += 4 + c.rpc_data.size();

        std::vector<std::byte> payload(total_size);
        std::memcpy(payload.data(), &count, 4);
        std::size_t offset = 4;
        for (auto& c : calls_) {
            std::uint32_t len = static_cast<std::uint32_t>(c.rpc_data.size());
            std::memcpy(payload.data() + offset, &len, 4);
            std::memcpy(payload.data() + offset + 4, c.rpc_data.data(), len);
            offset += 4 + len;
        }

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
        return results;
    }

    // ── Flush using pipeline_mem frame (syncs + RPCs, no reads) ──
    // Combines host_sync data + RPC calls into ONE network round-trip.
    // Sync data is LZ4-compressed if >= 4 KB and compressible.
    std::vector<pipe_result> flush_pipeline_mem() {
        // Build pipeline_mem payload:
        //   [4B sync_count]
        //   for each sync: [8B addr][8B original_size][1B comp_flag][data...]
        //   [4B rpc_count]
        //   for each rpc: [4B len][rpc_bytes]

        std::vector<std::byte> payload;

        // ── Sync entries (with optional LZ4 compression) ──
        // Compress first, then build payload with known sizes
        std::vector<compress_result> compressed_syncs;
        compressed_syncs.reserve(syncs_.size());
        for (auto& s : syncs_) {
            compressed_syncs.push_back(
                zlink::compress(std::span<const std::byte>(s.data.data(), s.data.size())));
        }

        std::uint32_t sync_count = static_cast<std::uint32_t>(syncs_.size());
        std::size_t sync_section_size = 4;
        for (auto& cs : compressed_syncs) {
            sync_section_size += 8 + 8 + 1 + 4 + cs.data.size(); // addr + orig_size + comp_flag + data_size + data
        }
        payload.resize(sync_section_size);
        std::memcpy(payload.data(), &sync_count, 4);
        std::size_t off = 4;
        for (std::size_t i = 0; i < syncs_.size(); i++) {
            auto& s = syncs_[i];
            auto& cs = compressed_syncs[i];
            std::uint64_t addr = s.client_addr;
            std::uint64_t orig_sz = cs.original_size;
            std::uint8_t cflag = cs.comp_flag;
            std::uint32_t data_sz = static_cast<std::uint32_t>(cs.data.size());
            std::memcpy(payload.data() + off, &addr, 8); off += 8;
            std::memcpy(payload.data() + off, &orig_sz, 8); off += 8;
            std::memcpy(payload.data() + off, &cflag, 1); off += 1;
            std::memcpy(payload.data() + off, &data_sz, 4); off += 4;
            std::memcpy(payload.data() + off, cs.data.data(), cs.data.size()); off += cs.data.size();
        }

        // ── RPC entries ──
        std::uint32_t rpc_count = static_cast<std::uint32_t>(calls_.size());
        std::size_t rpc_section_offset = payload.size();
        std::size_t rpc_section_size = 4;
        for (auto& c : calls_) rpc_section_size += 4 + c.rpc_data.size();
        payload.resize(payload.size() + rpc_section_size);
        std::memcpy(payload.data() + rpc_section_offset, &rpc_count, 4);
        off = rpc_section_offset + 4;
        for (auto& c : calls_) {
            std::uint32_t len = static_cast<std::uint32_t>(c.rpc_data.size());
            std::memcpy(payload.data() + off, &len, 4); off += 4;
            std::memcpy(payload.data() + off, c.rpc_data.data(), len); off += len;
        }

        // ── Read entries (empty for this variant) ──
        std::uint32_t read_count = 0;
        std::size_t read_section_offset = payload.size();
        payload.resize(payload.size() + 4);
        std::memcpy(payload.data() + read_section_offset, &read_count, 4);

        // ── Handle manifest ──
        zlink::serialize_handle_manifest(handle_manifest_, payload);

        frame req_frame;
        req_frame.call_id = 1;
        req_frame.type = frame_type::pipeline_mem;
        req_frame.payload = payload;

        auto ec = transport_.send(req_frame);
        if (ec) throw std::system_error(ec);

        frame resp_frame;
        ec = transport_.receive(resp_frame);
        if (ec) throw std::system_error(ec);

        if (resp_frame.type != frame_type::pipeline_response) {
            throw std::runtime_error("Expected pipeline_response for pipeline_mem");
        }

        auto results = parse_pipeline_response(resp_frame.payload);
        syncs_.clear();
        calls_.clear();
        reads_.clear();
        handle_manifest_.clear();
        return results;
    }

    // ── Flush using pipeline_mem frame with reads ────────────────
    // Combines: syncs + RPCs + read requests → one round-trip
    // Server processes: apply syncs (with decompression), run RPCs, then
    // read mirror data (compress it) and pack into the response.
    std::vector<pipe_result> flush_pipeline_mem_with_reads() {
        std::vector<std::byte> payload;

        // ── Sync entries (with optional LZ4 compression) ──
        std::vector<compress_result> compressed_syncs;
        compressed_syncs.reserve(syncs_.size());
        for (auto& s : syncs_) {
            compressed_syncs.push_back(
                zlink::compress(std::span<const std::byte>(s.data.data(), s.data.size())));
        }

        std::uint32_t sync_count = static_cast<std::uint32_t>(syncs_.size());
        std::size_t sync_section_size = 4;
        for (auto& cs : compressed_syncs) {
            sync_section_size += 8 + 8 + 1 + 4 + cs.data.size(); // addr + orig_size + comp_flag + data_size + data
        }
        payload.resize(sync_section_size);
        std::memcpy(payload.data(), &sync_count, 4);
        std::size_t off = 4;
        for (std::size_t i = 0; i < syncs_.size(); i++) {
            auto& s = syncs_[i];
            auto& cs = compressed_syncs[i];
            std::uint64_t addr = s.client_addr;
            std::uint64_t orig_sz = cs.original_size;
            std::uint8_t cflag = cs.comp_flag;
            std::uint32_t data_sz = static_cast<std::uint32_t>(cs.data.size());
            std::memcpy(payload.data() + off, &addr, 8); off += 8;
            std::memcpy(payload.data() + off, &orig_sz, 8); off += 8;
            std::memcpy(payload.data() + off, &cflag, 1); off += 1;
            std::memcpy(payload.data() + off, &data_sz, 4); off += 4;
            std::memcpy(payload.data() + off, cs.data.data(), cs.data.size()); off += cs.data.size();
        }

        // ── RPC entries ──
        std::uint32_t rpc_count = static_cast<std::uint32_t>(calls_.size());
        std::size_t rpc_section_offset = payload.size();
        std::size_t rpc_section_size = 4;
        for (auto& c : calls_) rpc_section_size += 4 + c.rpc_data.size();
        payload.resize(payload.size() + rpc_section_size);
        std::memcpy(payload.data() + rpc_section_offset, &rpc_count, 4);
        off = rpc_section_offset + 4;
        for (auto& c : calls_) {
            std::uint32_t len = static_cast<std::uint32_t>(c.rpc_data.size());
            std::memcpy(payload.data() + off, &len, 4); off += 4;
            std::memcpy(payload.data() + off, c.rpc_data.data(), len); off += len;
        }

        // ── Read entries ──
        std::uint32_t read_count = static_cast<std::uint32_t>(reads_.size());
        std::size_t read_section_offset = payload.size();
        std::size_t read_section_size = 4 + reads_.size() * (8 + 8); // addr + size per read
        payload.resize(payload.size() + read_section_size);
        std::memcpy(payload.data() + read_section_offset, &read_count, 4);
        off = read_section_offset + 4;
        for (auto& r : reads_) {
            std::uint64_t addr = r.client_addr;
            std::uint64_t sz = r.size;
            std::memcpy(payload.data() + off, &addr, 8); off += 8;
            std::memcpy(payload.data() + off, &sz, 8); off += 8;
        }

        // ── Handle manifest ──
        zlink::serialize_handle_manifest(handle_manifest_, payload);

        frame req_frame;
        req_frame.call_id = 1;
        req_frame.type = frame_type::pipeline_mem;
        req_frame.payload = payload;

        auto ec = transport_.send(req_frame);
        if (ec) throw std::system_error(ec);

        frame resp_frame;
        ec = transport_.receive(resp_frame);
        if (ec) throw std::system_error(ec);

        if (resp_frame.type != frame_type::pipeline_response) {
            throw std::runtime_error("Expected pipeline_response for pipeline_mem_with_reads");
        }

        // Parse the combined response: rpc results + read data
        auto results = parse_pipeline_mem_response(resp_frame.payload);

        syncs_.clear();
        calls_.clear();
        reads_.clear();
        handle_manifest_.clear();
        return results;
    }

    // ── Flush if there are pending calls ──────────────────────────
    void flush_if_needed() {
        if (has_pending()) {
            flush();
        }
    }

    // ── Parse pipeline_response payload ──────────────────────────
    // Standard format: [4B count][4B len1][resp1][4B len2][resp2]...
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

    // ── Parse pipeline_mem response ──────────────────────────────
    // Format: [4B rpc_count][4B len1][resp1]...[4B read_count]
    //         for each read: [4B data_len][1B comp_flag][data...]
    //         When comp_flag=LZ4: data = [8B original_size][compressed_bytes]
    // Also writes read data to the host buffers registered in reads_.
    std::vector<pipe_result> parse_pipeline_mem_response(
        std::span<const std::byte> payload)
    {
        if (payload.size() < 4) return {};

        std::size_t off = 0;

        // ── RPC results ──
        std::uint32_t rpc_count = 0;
        std::memcpy(&rpc_count, payload.data() + off, 4); off += 4;

        std::vector<pipe_result> results(rpc_count);
        for (std::uint32_t i = 0; i < rpc_count && off + 4 <= payload.size(); i++) {
            std::uint32_t len = 0;
            std::memcpy(&len, payload.data() + off, 4); off += 4;

            if (off + len > payload.size()) break;

            results[i].data.resize(len);
            std::memcpy(results[i].data.data(), payload.data() + off, len);
            results[i].valid = true;
            off += len;
        }

        // ── Read data (with optional LZ4 decompression) ──
        if (off + 4 <= payload.size()) {
            std::uint32_t read_count = 0;
            std::memcpy(&read_count, payload.data() + off, 4); off += 4;

            for (std::uint32_t i = 0; i < read_count && i < reads_.size(); i++) {
                if (off + 4 > payload.size()) break;

                std::uint32_t data_len = 0;
                std::memcpy(&data_len, payload.data() + off, 4); off += 4;

                if (off + data_len > payload.size()) break;

                // Read comp_flag
                std::uint8_t comp_flag = zlink::comp_flag_raw;
                if (data_len >= 1) {
                    std::memcpy(&comp_flag, payload.data() + off, 1);
                    off += 1;
                    data_len -= 1;
                }

                auto& rd = reads_[i];
                void* host_ptr = reinterpret_cast<void*>(rd.client_addr);
                std::size_t copy_size = rd.size; // Expected size

                if (comp_flag == zlink::comp_flag_lz4 && data_len >= 8) {
                    // LZ4 compressed: [8B original_size][compressed_bytes]
                    std::uint64_t original_size = 0;
                    std::memcpy(&original_size, payload.data() + off, 8);
                    off += 8;
                    data_len -= 8;

                    // Decompress
                    auto decompressed = zlink::decompress(
                        std::span<const std::byte>(payload.data() + off, data_len),
                        comp_flag, static_cast<std::size_t>(original_size));

                    std::size_t copy_len = std::min(decompressed.size(), copy_size);
                    std::memcpy(host_ptr, decompressed.data(), copy_len);
                    off += data_len;
                } else {
                    // Raw data
                    std::size_t copy_len = std::min(static_cast<std::size_t>(data_len), copy_size);
                    std::memcpy(host_ptr, payload.data() + off, copy_len);
                    off += data_len;
                }
            }
        }

        return results;
    }

    transport& transport_;
    std::vector<queued_call> calls_;
    std::vector<pending_sync> syncs_;
    std::vector<pending_read> reads_;
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
