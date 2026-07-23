// zlink/examples/cuda/cuda_server.cpp — CUDA RPC server over the zlink pipeline
//
// Runs on the GPU machine. Receives RPC calls (plain + batched) from the
// client, executes them on real CUDA hardware, and sends responses back.
//
// Key components:
//   - cuda_gen::* handlers: real CUDA calls; handle producers set
//     g_last_produced_handle via an RAII guard; handle consumers translate
//     virtual handles via g_vhandles.
//   - handle_pipeline_request(): parses the handle manifest BEFORE running
//     RPCs (so handles register under their virtual_id immediately), then
//     runs the RPCs in order.
//   - handle_memory_op(): host_sync (client→server data) and host_read
//     (server→client data) via the host_memory_mirror.

#include <zlink/transport.hpp>
#include <zlink/multiplexed_transport.hpp>
#include <zlink/config.hpp>
#include <zlink/virtual_handle.hpp>
#include <zlink/memory.hpp>

#include "cuda_api.hpp"

#include <zpp_bits.h>

#include <cuda.h>

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <span>
#include <mutex>
#include <unordered_map>

// ── Server global state ────────────────────────────────────────────────
static zlink::handle_table g_vhandles;          // virtual → real handle map
static zlink::host_memory_mirror g_host_mirror;  // client host memory on server

// Handle production capture: producing handlers set g_last_produced_handle.
// After serve(), the pipeline handler reads it to register VH→real.
static std::uint64_t g_last_produced_handle = 0;
static bool g_has_produced_handle = false;

// RAII guard: set g_last_produced_handle + g_has_produced_handle on success.
struct handle_producer_guard {
    bool produced = false;
    ~handle_producer_guard() { if (produced) g_has_produced_handle = true; }
    void set(std::uint64_t real_handle) {
        g_last_produced_handle = real_handle;
        produced = true;
    }
};

namespace cuda_gen {

// ── barrier handlers ───────────────────────────────────────────────────
InitRet init(uint32_t flags) {
    InitRet ret;
    ret.result = static_cast<int32_t>(cuInit(flags));
    return ret;
}
DriverGetVersionRet driver_get_version() {
    int v = 0;
    CUresult r = cuDriverGetVersion(&v);
    return {static_cast<int32_t>(r), v};
}
DeviceGetCountRet device_get_count() {
    int c = 0;
    CUresult r = cuDeviceGetCount(&c);
    return {static_cast<int32_t>(r), c};
}
DeviceGetRet device_get(int32_t ordinal) {
    CUdevice d = 0;
    CUresult r = cuDeviceGet(&d, ordinal);
    return {static_cast<int32_t>(r), static_cast<int32_t>(d)};
}
DeviceGetNameRet device_get_name(int32_t len, int32_t dev) {
    char name[256] = {};
    CUresult r = cuDeviceGetName(name, len, static_cast<CUdevice>(dev));
    return {static_cast<int32_t>(r), std::string(name)};
}
DeviceTotalMemRet device_total_mem(int32_t dev) {
    size_t b = 0;
    CUresult r = cuDeviceTotalMem(&b, static_cast<CUdevice>(dev));
    return {static_cast<int32_t>(r), static_cast<uint64_t>(b)};
}
DeviceGetAttributeRet device_get_attribute(uint32_t attrib, int32_t dev) {
    int v = 0;
    CUresult r = cuDeviceGetAttribute(&v,
        static_cast<CUdevice_attribute>(attrib), static_cast<CUdevice>(dev));
    return {static_cast<int32_t>(r), v};
}
CtxGetCurrentRet ctx_get_current() {
    CUcontext ctx = nullptr;
    CUresult r = cuCtxGetCurrent(&ctx);
    return {static_cast<int32_t>(r), reinterpret_cast<uint64_t>(ctx)};
}
MemGetInfoRet mem_get_info() {
    size_t free_b = 0, total_b = 0;
    CUresult r = cuMemGetInfo(&free_b, &total_b);
    return {static_cast<int32_t>(r), static_cast<uint64_t>(free_b),
            static_cast<uint64_t>(total_b)};
}
EventElapsedTimeRet event_elapsed_time(uint64_t start_handle, uint64_t end_handle) {
    CUevent s = reinterpret_cast<CUevent>(g_vhandles.translate(start_handle));
    CUevent e = reinterpret_cast<CUevent>(g_vhandles.translate(end_handle));
    float ms = 0.0f;
    CUresult r = cuEventElapsedTime(&ms, s, e);
    return {static_cast<int32_t>(r), ms};
}

// ── enqueued handle producers ──────────────────────────────────────────
CtxCreateRet ctx_create(uint32_t flags, int32_t dev) {
    handle_producer_guard guard;
    CUcontext ctx = nullptr;
    CUresult r = cuCtxCreate(&ctx, flags, static_cast<CUdevice>(dev));
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<uint64_t>(ctx));
    return {static_cast<int32_t>(r), reinterpret_cast<uint64_t>(ctx)};
}
MemAllocRet mem_alloc(uint64_t bytesize) {
    handle_producer_guard guard;
    CUdeviceptr p = 0;
    CUresult r = cuMemAlloc(&p, static_cast<size_t>(bytesize));
    if (r == CUDA_SUCCESS) guard.set(static_cast<uint64_t>(p));
    return {static_cast<int32_t>(r), static_cast<uint64_t>(p)};
}
MemAllocManagedRet mem_alloc_managed(uint64_t bytesize, uint32_t flags) {
    handle_producer_guard guard;
    CUdeviceptr p = 0;
    CUresult r = cuMemAllocManaged(&p, static_cast<size_t>(bytesize), flags);
    if (r == CUDA_SUCCESS) guard.set(static_cast<uint64_t>(p));
    return {static_cast<int32_t>(r), static_cast<uint64_t>(p)};
}
StreamCreateRet stream_create(uint32_t flags) {
    handle_producer_guard guard;
    CUstream s = nullptr;
    CUresult r = cuStreamCreate(&s, flags);
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<uint64_t>(s));
    return {static_cast<int32_t>(r), reinterpret_cast<uint64_t>(s)};
}
EventCreateRet event_create(uint32_t flags) {
    handle_producer_guard guard;
    CUevent e = nullptr;
    CUresult r = cuEventCreate(&e, flags);
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<uint64_t>(e));
    return {static_cast<int32_t>(r), reinterpret_cast<uint64_t>(e)};
}
ModuleLoadDataRet module_load_data(uint64_t image_addr, uint64_t byte_count) {
    handle_producer_guard guard;
    CUmodule mod = nullptr;
    void* image = nullptr;
    auto mirror = g_host_mirror.translate(static_cast<std::uintptr_t>(image_addr));
    if (mirror) image = reinterpret_cast<void*>(*mirror);
    CUresult r = (image) ? cuModuleLoadData(&mod, image)
                         : static_cast<CUresult>(CUDA_ERROR_INVALID_VALUE);
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<uint64_t>(mod));
    return {static_cast<int32_t>(r), reinterpret_cast<uint64_t>(mod)};
}
ModuleGetFunctionRet module_get_function(uint64_t module_handle, std::string name) {
    handle_producer_guard guard;
    CUmodule mod = reinterpret_cast<CUmodule>(g_vhandles.translate(module_handle));
    CUfunction fn = nullptr;
    CUresult r = cuModuleGetFunction(&fn, mod, name.c_str());
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<uint64_t>(fn));
    return {static_cast<int32_t>(r), reinterpret_cast<uint64_t>(fn)};
}

// ── enqueued handle consumers ──────────────────────────────────────────
CtxDestroyRet ctx_destroy(uint64_t ctx_handle) {
    CUcontext ctx = reinterpret_cast<CUcontext>(g_vhandles.translate(ctx_handle));
    return {static_cast<int32_t>(cuCtxDestroy(ctx))};
}
CtxSetCurrentRet ctx_set_current(uint64_t ctx_handle) {
    CUcontext ctx = reinterpret_cast<CUcontext>(g_vhandles.translate(ctx_handle));
    return {static_cast<int32_t>(cuCtxSetCurrent(ctx))};
}
CtxSynchronizeRet ctx_synchronize() {
    return {static_cast<int32_t>(cuCtxSynchronize())};
}
MemFreeRet mem_free(uint64_t dptr) {
    CUdeviceptr p = static_cast<CUdeviceptr>(g_vhandles.translate(dptr));
    return {static_cast<int32_t>(cuMemFree(p))};
}
MemcpyHtoDRet memcpy_hto_d(uint64_t dst_device, uint64_t src_host_addr, uint64_t byte_count) {
    CUdeviceptr dst = static_cast<CUdeviceptr>(g_vhandles.translate(dst_device));
    void* src = nullptr;
    auto mirror = g_host_mirror.translate(static_cast<std::uintptr_t>(src_host_addr));
    if (mirror) src = reinterpret_cast<void*>(*mirror);
    CUresult r = src ? cuMemcpyHtoD(dst, src, static_cast<size_t>(byte_count))
                     : static_cast<CUresult>(CUDA_ERROR_INVALID_VALUE);
    return {static_cast<int32_t>(r)};
}
MemcpyDtoDRet memcpy_dto_d(uint64_t dst_device, uint64_t src_device, uint64_t byte_count) {
    CUdeviceptr dst = static_cast<CUdeviceptr>(g_vhandles.translate(dst_device));
    CUdeviceptr src = static_cast<CUdeviceptr>(g_vhandles.translate(src_device));
    return {static_cast<int32_t>(cuMemcpyDtoD(dst, src, static_cast<size_t>(byte_count)))};
}
// readback: cuMemcpyDtoH — copy from GPU device to the mirrored client host
// memory. The client later pulls the bytes back via host_read (or demand paging).
MemcpyDtoHRet memcpy_dto_h(uint64_t dst_host_addr, uint64_t src_device, uint64_t byte_count) {
    CUdeviceptr src = static_cast<CUdeviceptr>(g_vhandles.translate(src_device));

    // Auto-register the mirror region for this address if needed.
    // The client may not have synced this buffer before (it's a readback target).
    std::uintptr_t dst_addr = static_cast<std::uintptr_t>(dst_host_addr);
    auto mirror = g_host_mirror.translate(dst_addr);
    if (!mirror) {
        // Create a mirror region covering the readback buffer
        std::vector<std::byte> dummy(byte_count);
        g_host_mirror.sync_page(dst_addr, dummy);
        mirror = g_host_mirror.translate(dst_addr);
    }

    void* dst = mirror ? reinterpret_cast<void*>(*mirror) : nullptr;
    CUresult r = dst ? cuMemcpyDtoH(dst, src, static_cast<size_t>(byte_count))
                     : static_cast<CUresult>(CUDA_ERROR_INVALID_VALUE);
    return {static_cast<int32_t>(r)};
}
StreamDestroyRet stream_destroy(uint64_t stream_handle) {
    CUstream s = reinterpret_cast<CUstream>(g_vhandles.translate(stream_handle));
    return {static_cast<int32_t>(cuStreamDestroy(s))};
}
StreamSynchronizeRet stream_synchronize(uint64_t stream_handle) {
    CUstream s = reinterpret_cast<CUstream>(g_vhandles.translate(stream_handle));
    return {static_cast<int32_t>(cuStreamSynchronize(s))};
}
EventDestroyRet event_destroy(uint64_t event_handle) {
    CUevent e = reinterpret_cast<CUevent>(g_vhandles.translate(event_handle));
    return {static_cast<int32_t>(cuEventDestroy(e))};
}
EventRecordRet event_record(uint64_t event_handle, uint64_t stream_handle) {
    CUevent e = reinterpret_cast<CUevent>(g_vhandles.translate(event_handle));
    CUstream s = reinterpret_cast<CUstream>(g_vhandles.translate(stream_handle));
    return {static_cast<int32_t>(cuEventRecord(e, s))};
}
EventSynchronizeRet event_synchronize(uint64_t event_handle) {
    CUevent e = reinterpret_cast<CUevent>(g_vhandles.translate(event_handle));
    return {static_cast<int32_t>(cuEventSynchronize(e))};
}

// cuLaunchKernel: translate the function + stream handles, read kernel args
// from the mirrored client memory, and translate any device pointers in args.
LaunchKernelRet launch_kernel(
    uint64_t func_handle,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    uint32_t shared_mem,
    uint64_t stream_handle,
    uint64_t args_addr, uint64_t args_byte_count) {

    CUfunction f = reinterpret_cast<CUfunction>(g_vhandles.translate(func_handle));
    CUstream s = (stream_handle != 0)
        ? reinterpret_cast<CUstream>(g_vhandles.translate(stream_handle))
        : nullptr;

    int n_args = static_cast<int>(args_byte_count / sizeof(std::uint64_t));
    std::vector<void*> arg_ptrs(n_args);
    std::vector<std::uint64_t> arg_vals(n_args);

    if (n_args > 0) {
        auto mirror = g_host_mirror.translate(static_cast<std::uintptr_t>(args_addr));
        if (mirror) {
            std::memcpy(arg_vals.data(), reinterpret_cast<void*>(*mirror),
                        static_cast<std::size_t>(args_byte_count));
        }
        // Translate any virtual device pointers among the args.
        for (int i = 0; i < n_args; i++) {
            if (zlink::is_virtual_handle(arg_vals[i])) {
                arg_vals[i] = g_vhandles.translate(arg_vals[i]);
            }
            arg_ptrs[i] = &arg_vals[i];
        }
    }

    CUresult r = cuLaunchKernel(f,
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        shared_mem, s,
        arg_ptrs.data(), nullptr);
    return {static_cast<int32_t>(r)};
}

} // namespace cuda_gen

// ══════════════════════════════════════════════════════════════════════════
// RPC server: dispatch + frame handling
// ══════════════════════════════════════════════════════════════════════════
class cuda_rpc_server {
public:
    void serve(zlink::transport& tp) {
        using namespace zlink;

        auto ec = tp.listen("0.0.0.0", default_port);
        if (ec) { std::cerr << "[server] listen failed: " << ec.message() << "\n"; return; }
        std::cerr << "[server] listening on port " << default_port << "...\n";

        ec = tp.accept();
        if (ec) { std::cerr << "[server] accept failed: " << ec.message() << "\n"; return; }
        std::cerr << "[server] client connected\n";

        while (tp.is_connected()) {
            frame req;
            ec = tp.receive(req);
            if (ec) { std::cerr << "[server] recv error: " << ec.message() << "\n"; break; }

            if (req.type == frame_type::memory_op) {
                handle_memory_op(tp, req);
            } else if (req.type == frame_type::request) {
                handle_request(tp, req);
            } else if (req.type == frame_type::pipeline_request) {
                handle_pipeline_request(tp, req);
            } else {
                std::cerr << "[server] unknown frame type 0x"
                          << std::hex << static_cast<int>(req.type) << std::dec << "\n";
            }
        }
        std::cerr << "[server] client disconnected\n";
    }

private:
    // ── Plain single-RPC request ──────────────────────────────────────
    void handle_request(zlink::transport& tp, const zlink::frame& req) {
        using namespace zpp::bits;
        auto [data, in, out] = data_in_out();
        data.assign(req.payload.begin(), req.payload.end());
        g_has_produced_handle = false;

        cuda_gen::cuda_gen_rpc::server server{in, out};
        auto result = server.serve();
        if (failure(result)) {
            std::cerr << "[server] dispatch failed\n";
            zlink::frame resp;
            resp.call_id = req.call_id;
            resp.type = zlink::frame_type::response;
            int32_t err = static_cast<int32_t>(CUDA_ERROR_INVALID_VALUE);
            resp.payload.resize(sizeof(err));
            std::memcpy(resp.payload.data(), &err, sizeof(err));
            tp.send(resp);
            return;
        }
        send_response(tp, req.call_id, zlink::frame_type::response, data);
    }

    // ── Pipeline request (batched RPC + virtual handle manifest) ──────
    // Wire format (pipeline_request):
    //   [4B rpc_count]
    //   per rpc: [4B len][rpc_bytes]
    //   [handle_manifest: 4B count + N × handle_manifest_entry]
    //
    // The manifest maps call_index → virtual_id. We parse it BEFORE running
    // RPCs so that produced handles are registered under their virtual_id
    // immediately, letting subsequent RPCs in the same batch translate
    // virtual handles correctly.
    void handle_pipeline_request(zlink::transport& tp, const zlink::frame& req) {
        using namespace zpp::bits;
        if (req.payload.size() < 4) return;

        auto* p = req.payload.data();
        auto total = req.payload.size();
        std::size_t off = 0;

        std::uint32_t count = 0;
        std::memcpy(&count, p + off, 4);
        off += 4;

        // Skip past RPCs to find the manifest at the end
        std::size_t rpc_start = off;
        for (std::uint32_t i = 0; i < count && off + 4 <= total; ++i) {
            std::uint32_t len = 0;
            std::memcpy(&len, p + off, 4); off += 4;
            off += len;
        }

        // Parse manifest
        std::vector<zlink::handle_manifest_entry> manifest;
        if (off + 4 <= total) {
            std::size_t manifest_bytes = 0;
            manifest = zlink::parse_handle_manifest(
                std::span<const std::byte>(p + off, total - off), manifest_bytes);
        }

        // Build call_index → virtual_id lookup
        std::unordered_map<std::uint32_t, std::uint32_t> call_to_vhid;
        for (const auto& entry : manifest) {
            call_to_vhid[entry.call_index] = entry.virtual_id;
        }

        // Process RPCs in order, registering produced handles
        std::vector<std::byte> resp_payload;
        resp_payload.resize(4);
        std::uint32_t resp_count = 0;

        off = rpc_start;
        for (std::uint32_t i = 0; i < count && off + 4 <= total; ++i) {
            std::uint32_t len = 0;
            std::memcpy(&len, p + off, 4); off += 4;
            if (off + len > total) break;

            auto [data, in, out] = data_in_out();
            data.assign(p + off, p + off + len);
            off += len;

            g_has_produced_handle = false;
            cuda_gen::cuda_gen_rpc::server server{in, out};
            (void)server.serve();

            if (g_has_produced_handle) {
                std::uint32_t reg_id = i;
                auto it = call_to_vhid.find(i);
                if (it != call_to_vhid.end()) reg_id = it->second;
                g_vhandles.register_handle(reg_id, g_last_produced_handle);
            }

            std::uint32_t resp_len = static_cast<std::uint32_t>(data.size());
            std::size_t prev = resp_payload.size();
            resp_payload.resize(prev + 4 + resp_len);
            std::memcpy(resp_payload.data() + prev, &resp_len, 4);
            std::memcpy(resp_payload.data() + prev + 4, data.data(), resp_len);
            ++resp_count;
        }
        std::memcpy(resp_payload.data(), &resp_count, 4);
        send_response(tp, req.call_id, zlink::frame_type::pipeline_response, resp_payload);
    }

    // ── Memory op (host_sync / host_read outside the pipeline) ─────────
    void handle_memory_op(zlink::transport& tp, const zlink::frame& req) {
        if (req.payload.size() < sizeof(zlink::mem_request)) return;
        zlink::mem_request mr;
        std::memcpy(&mr, req.payload.data(), sizeof(mr));

        zlink::frame resp;
        resp.call_id = req.call_id;
        resp.type = zlink::frame_type::memory_reply;

        if (mr.op == zlink::mem_op::host_sync) {
            auto data_span = std::span<const std::byte>(
                req.payload.data() + sizeof(mr),
                req.payload.size() - sizeof(mr));
            g_host_mirror.sync_page(mr.remote_addr, data_span);
            zlink::mem_response mresp{zlink::error_code::ok, data_span.size(), mr.remote_addr};
            resp.payload.resize(sizeof(mresp));
            std::memcpy(resp.payload.data(), &mresp, sizeof(mresp));
        } else if (mr.op == zlink::mem_op::host_read) {
            auto mirror = g_host_mirror.translate(mr.remote_addr);
            zlink::mem_response mresp{zlink::error_code::ok, 0, mr.remote_addr};
            if (mirror) {
                mresp.size = mr.size;
                resp.payload.resize(sizeof(mresp) + mr.size);
                std::memcpy(resp.payload.data(), &mresp, sizeof(mresp));
                std::memcpy(resp.payload.data() + sizeof(mresp),
                            reinterpret_cast<void*>(*mirror), mr.size);
            } else {
                mresp.status = zlink::error_code::pointer_not_found;
                resp.payload.resize(sizeof(mresp));
                std::memcpy(resp.payload.data(), &mresp, sizeof(mresp));
            }
        } else {
            return;
        }
        tp.send(resp);
    }

    void send_response(zlink::transport& tp, std::uint32_t call_id,
                       zlink::frame_type type, std::span<const std::byte> payload) {
        zlink::frame resp;
        resp.call_id = call_id;
        resp.type = type;
        resp.payload.assign(payload.begin(), payload.end());
        tp.send(resp);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════════
int main() {
    std::cerr << "zlink CUDA RPC Server\n";

    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) { std::cerr << "[server] cuInit failed: " << r << "\n"; return 1; }

    int dev_count = 0;
    cuDeviceGetCount(&dev_count);
    std::cerr << "[server] " << dev_count << " CUDA device(s)\n";
    if (dev_count == 0) { std::cerr << "[server] no CUDA devices\n"; return 1; }

    cuda_rpc_server server;
    auto tp = std::make_unique<zlink::multiplexed_transport>();
    server.serve(*tp);
    return 0;
}
