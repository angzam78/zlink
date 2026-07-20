// zlink/examples/cuda/cuda_test_server.cpp
//
// CUDA RPC server — virtual handles + kernel launch + r3map memory transport.
//
// Architecture:
//   Virtual handles: cuMemAlloc returns VH(0), cuMemcpyHtoD(VH(0), ...)
//   The server translates VH → real CUDA handles when processing pipeline batches.
//   This eliminates ALL barrier calls except the final data readback.
//
//   Handle translation: Each CUDA function checks its handle-type parameters
//   with g_vhandles.translate(). Virtual handles (bit 63 set) get resolved
//   to real CUDA values. Non-virtual values pass through unchanged.

#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/ptr_map.hpp>
#include <zlink/memory.hpp>
#include <zlink/chunk_cache.hpp>
#include <zlink/config.hpp>
#include <zlink/virtual_handle.hpp>
#include <zlink/compress.hpp>

#include <zpp_bits.h>

#include <cuda.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>

// ── Global state ──────────────────────────────────────────────────────
static zlink::ptr_map g_ptr_map;
static zlink::host_memory_mirror g_host_mirror;
static zlink::handle_table g_vhandles;  // Virtual → real handle mapping

// ── Handle production hook ─────────────────────────────────────────────
// When a CUDA wrapper function produces a handle, it stores the real value
// here. The pipeline handler reads this after each serve() and registers
// the virtual→real mapping. This avoids parsing zpp_bits varint-encoded
// responses.
static std::uint64_t g_last_produced_handle = 0;
static bool g_has_produced_handle = false;

// RAII guard: set g_last_produced_handle and clear on next call
struct handle_producer_guard {
    handle_producer_guard() : produced(false) {}
    ~handle_producer_guard() { if (produced) g_has_produced_handle = true; }
    void set(std::uint64_t real_handle) {
        g_last_produced_handle = real_handle;
        produced = true;
    }
    bool produced;
};

// ── RPC-friendly CUDA API ─────────────────────────────────────────────
// All handle-type parameters go through g_vhandles.translate().
// This is safe for both virtual handles (bit 63 set → translate) and
// real handles (bit 63 clear → pass through).

namespace cuda_rpc_api {

// cuInit
struct InitRet { int32_t result; };
InitRet cuda_init(unsigned int flags) {
    std::cerr << "  [server] cuInit(" << flags << ")\n";
    return {static_cast<int32_t>(cuInit(flags))};
}

// cuDeviceGetCount
struct DevCountRet { int32_t result; int32_t count; };
DevCountRet get_device_count() {
    int count = 0;
    CUresult r = cuDeviceGetCount(&count);
    std::cerr << "  [server] cuDeviceGetCount() -> " << count << "\n";
    return {static_cast<int32_t>(r), count};
}

// cuDeviceGetName
struct DevNameRet { int32_t result; std::string name; };
DevNameRet get_device_name(int ordinal) {
    char name[256] = {};
    CUdevice dev;
    CUresult r1 = cuDeviceGet(&dev, ordinal);
    if (r1 != CUDA_SUCCESS) return {static_cast<int32_t>(r1), ""};
    CUresult r2 = cuDeviceGetName(name, sizeof(name), dev);
    std::cerr << "  [server] cuDeviceGetName(" << ordinal << ") -> \"" << name << "\"\n";
    return {static_cast<int32_t>(r2), std::string(name)};
}

// cuDeviceTotalMem
struct DevMemRet { int32_t result; std::uint64_t bytes; };
DevMemRet get_device_total_mem(int ordinal) {
    CUdevice dev;
    CUresult r1 = cuDeviceGet(&dev, ordinal);
    if (r1 != CUDA_SUCCESS) return {static_cast<int32_t>(r1), 0};
    size_t bytes = 0;
    CUresult r2 = cuDeviceTotalMem(&bytes, dev);
    std::cerr << "  [server] cuDeviceTotalMem(" << ordinal << ") -> " << bytes << "\n";
    return {static_cast<int32_t>(r2), static_cast<std::uint64_t>(bytes)};
}

// cuCtxCreate — handle-producing: ctx_handle field is virtual
struct CtxCreateRet { int32_t result; std::uint64_t ctx_handle; };
CtxCreateRet ctx_create(unsigned int flags, int device_ordinal) {
    handle_producer_guard guard;
    CUdevice dev;
    CUresult r1 = cuDeviceGet(&dev, device_ordinal);
    if (r1 != CUDA_SUCCESS) return {static_cast<int32_t>(r1), 0};
    CUcontext ctx;
    CUresult r2 = cuCtxCreate(&ctx, nullptr, flags, dev);
    if (r2 == CUDA_SUCCESS) guard.set(reinterpret_cast<std::uint64_t>(ctx));
    std::cerr << "  [server] cuCtxCreate() -> ctx=" << ctx << "\n";
    return {static_cast<int32_t>(r2), reinterpret_cast<std::uint64_t>(ctx)};
}

// cuCtxSynchronize
struct CtxSyncRet { int32_t result; };
CtxSyncRet ctx_synchronize() {
    CUresult r = cuCtxSynchronize();
    std::cerr << "  [server] cuCtxSynchronize()\n";
    return {static_cast<int32_t>(r)};
}

// cuCtxDestroy — handle-consuming: ctx_handle is translated
struct CtxDestroyRet { int32_t result; };
CtxDestroyRet ctx_destroy(std::uint64_t ctx_handle) {
    auto real_ctx = g_vhandles.translate(ctx_handle);
    CUresult r = cuCtxDestroy(reinterpret_cast<CUcontext>(real_ctx));
    std::cerr << "  [server] cuCtxDestroy(vh=0x" << std::hex << ctx_handle
              << " -> real=0x" << real_ctx << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

// cuMemAlloc — handle-producing: dev_ptr field is virtual
struct AllocRet { int32_t result; std::uint64_t dev_ptr; };
AllocRet mem_alloc(std::uint64_t bytesize) {
    handle_producer_guard guard;
    CUdeviceptr ptr = 0;
    CUresult r = cuMemAlloc(&ptr, bytesize);
    if (r == CUDA_SUCCESS) guard.set(ptr);
    std::cerr << "  [server] cuMemAlloc(" << bytesize << ") -> ptr=0x"
              << std::hex << ptr << std::dec << "\n";
    return {static_cast<int32_t>(r), ptr};
}

// cuMemFree — handle-consuming: dev_ptr is translated
struct FreeRet { int32_t result; };
FreeRet mem_free(std::uint64_t dev_ptr) {
    auto real_ptr = g_vhandles.translate(dev_ptr);
    CUresult r = cuMemFree(static_cast<CUdeviceptr>(real_ptr));
    std::cerr << "  [server] cuMemFree(vh=0x" << std::hex << dev_ptr
              << " -> real=0x" << real_ptr << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

// cuMemcpyHtoD — handle-consuming: dst_dev_ptr is translated
struct CopyHtoDRet { int32_t result; };
CopyHtoDRet memcpy_htod(std::uint64_t dst_dev_ptr,
                         std::uint64_t src_client_addr,
                         std::uint64_t byte_count) {
    auto real_dst = g_vhandles.translate(dst_dev_ptr);
    CUdeviceptr cu_dst = static_cast<CUdeviceptr>(real_dst);

    auto mirror_addr = g_host_mirror.translate(src_client_addr);
    const void* real_src = mirror_addr
        ? reinterpret_cast<const void*>(*mirror_addr)
        : reinterpret_cast<const void*>(src_client_addr);

    CUresult r = cuMemcpyHtoD(cu_dst, real_src, byte_count);
    std::cerr << "  [server] cuMemcpyHtoD(vh=0x" << std::hex << dst_dev_ptr
              << " -> dst=0x" << real_dst
              << ", src_client=0x" << src_client_addr
              << ", mirror=0x" << (mirror_addr ? *mirror_addr : 0)
              << ", n=" << std::dec << byte_count << ")\n";
    return {static_cast<int32_t>(r)};
}

// cuMemcpyDtoH — handle-consuming: src_dev_ptr is translated
struct CopyDtoHRet { int32_t result; };
CopyDtoHRet memcpy_dtoh(std::uint64_t dst_client_addr,
                          std::uint64_t src_dev_ptr,
                          std::uint64_t byte_count) {
    auto real_src = g_vhandles.translate(src_dev_ptr);
    CUdeviceptr cu_src = static_cast<CUdeviceptr>(real_src);

    auto mirror_addr = g_host_mirror.translate(dst_client_addr);
    void* real_dst = mirror_addr
        ? reinterpret_cast<void*>(*mirror_addr)
        : reinterpret_cast<void*>(dst_client_addr);

    CUresult r = cuMemcpyDtoH(real_dst, cu_src, byte_count);
    std::cerr << "  [server] cuMemcpyDtoH(dst_client=0x" << std::hex << dst_client_addr
              << ", mirror=0x" << (mirror_addr ? *mirror_addr : 0)
              << ", vh=0x" << src_dev_ptr
              << " -> src=0x" << real_src
              << ", n=" << std::dec << byte_count << ")\n";
    return {static_cast<int32_t>(r)};
}

// ── New: cuModuleLoadData — handle-producing: module_handle is virtual ──
struct ModuleLoadRet { int32_t result; std::uint64_t module_handle; };
ModuleLoadRet module_load_data(std::uint64_t src_client_addr, std::uint64_t byte_count) {
    handle_producer_guard guard;
    auto mirror_addr = g_host_mirror.translate(src_client_addr);
    const void* image = mirror_addr
        ? reinterpret_cast<const void*>(*mirror_addr)
        : reinterpret_cast<const void*>(src_client_addr);

    CUmodule mod = nullptr;
    CUresult r = cuModuleLoadData(&mod, image);
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<std::uint64_t>(mod));
    std::cerr << "  [server] cuModuleLoadData(image_client=0x" << std::hex << src_client_addr
              << ", mirror=0x" << (mirror_addr ? *mirror_addr : 0)
              << ", n=" << std::dec << byte_count
              << ") -> mod=" << mod << "\n";
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(mod)};
}

// ── New: cuModuleGetFunction — handle-consuming (module) + handle-producing (func) ──
struct ModuleGetFuncRet { int32_t result; std::uint64_t func_handle; };
ModuleGetFuncRet module_get_function(std::uint64_t module_handle, std::string func_name) {
    handle_producer_guard guard;
    auto real_mod = g_vhandles.translate(module_handle);
    CUmodule cu_mod = reinterpret_cast<CUmodule>(real_mod);

    CUfunction func = nullptr;
    CUresult r = cuModuleGetFunction(&func, cu_mod, func_name.c_str());
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<std::uint64_t>(func));
    std::cerr << "  [server] cuModuleGetFunction(vh_mod=0x" << std::hex << module_handle
              << " -> mod=" << cu_mod
              << ", name=\"" << func_name << "\""
              << ") -> func=" << func << std::dec << "\n";
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(func)};
}

// ── New: cuLaunchKernel — handle-consuming (func, stream) ──
// Kernel arguments are passed as a packed byte buffer that the server
// unpacks. Device pointer arguments in the buffer are virtual handles
// that get translated.
struct LaunchKernelRet { int32_t result; };
LaunchKernelRet launch_kernel(
    std::uint64_t func_handle,
    std::uint32_t grid_dim_x, std::uint32_t grid_dim_y, std::uint32_t grid_dim_z,
    std::uint32_t block_dim_x, std::uint32_t block_dim_y, std::uint32_t block_dim_z,
    std::uint32_t shared_mem_bytes,
    std::uint64_t stream_handle,
    std::uint64_t args_client_addr,
    std::uint64_t args_byte_count)
{
    auto real_func = g_vhandles.translate(func_handle);
    CUfunction cu_func = reinterpret_cast<CUfunction>(real_func);

    auto real_stream_val = g_vhandles.translate(stream_handle);
    CUstream cu_stream = (stream_handle == 0) ? nullptr
        : reinterpret_cast<CUstream>(real_stream_val);

    // The kernel args are in the host mirror — they contain virtual device
    // pointers that need translation. We read them from the mirror and
    // fix up any virtual handles before passing to cuLaunchKernel.
    void* kernel_args[64] = {};  // Max 64 kernel args
    std::vector<std::byte> args_buffer;

    if (args_byte_count > 0) {
        auto mirror_addr = g_host_mirror.translate(args_client_addr);
        const void* args_src = mirror_addr
            ? reinterpret_cast<const void*>(*mirror_addr)
            : reinterpret_cast<const void*>(args_client_addr);

        args_buffer.resize(args_byte_count);
        std::memcpy(args_buffer.data(), args_src, args_byte_count);

        // The args buffer contains an array of pointers (void**).
        // Each pointer may be a virtual device pointer.
        // For cuLaunchKernel, we pass void** where each void* points to
        // the actual argument. Our convention: the client sends the
        // args as a flat array of uint64_t values, with virtual handles
        // for device pointers.
        std::size_t n_args = args_byte_count / sizeof(std::uint64_t);
        std::uint64_t* args_u64 = reinterpret_cast<std::uint64_t*>(args_buffer.data());

        // Build the void** array for cuLaunchKernel
        for (std::size_t i = 0; i < n_args && i < 64; i++) {
            // Translate virtual handles in args
            args_u64[i] = g_vhandles.translate(args_u64[i]);
            kernel_args[i] = &args_u64[i];
        }
    }

    CUresult r = cuLaunchKernel(
        cu_func,
        grid_dim_x, grid_dim_y, grid_dim_z,
        block_dim_x, block_dim_y, block_dim_z,
        shared_mem_bytes, cu_stream,
        kernel_args, nullptr);

    std::cerr << "  [server] cuLaunchKernel(vh_func=0x" << std::hex << func_handle
              << " -> func=" << cu_func
              << ", grid=(" << std::dec << grid_dim_x << "," << grid_dim_y << "," << grid_dim_z << ")"
              << ", block=(" << block_dim_x << "," << block_dim_y << "," << block_dim_z << ")"
              << ", vh_stream=0x" << std::hex << stream_handle
              << ", args_n=" << std::dec << args_byte_count / 8
              << ")\n";
    return {static_cast<int32_t>(r)};
}

// ── New: cuStreamCreate — handle-producing ──
struct StreamCreateRet { int32_t result; std::uint64_t stream_handle; };
StreamCreateRet stream_create(unsigned int flags) {
    handle_producer_guard guard;
    CUstream stream = nullptr;
    CUresult r = cuStreamCreate(&stream, flags);
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<std::uint64_t>(stream));
    std::cerr << "  [server] cuStreamCreate(flags=" << flags << ") -> stream=" << stream << "\n";
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(stream)};
}

// ── New: cuStreamSynchronize — handle-consuming (stream) ──
struct StreamSyncRet { int32_t result; };
StreamSyncRet stream_synchronize(std::uint64_t stream_handle) {
    auto real_stream_val = g_vhandles.translate(stream_handle);
    CUstream cu_stream = (stream_handle == 0) ? nullptr
        : reinterpret_cast<CUstream>(real_stream_val);
    CUresult r = cuStreamSynchronize(cu_stream);
    std::cerr << "  [server] cuStreamSynchronize(vh=0x" << std::hex << stream_handle
              << " -> stream=" << cu_stream << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

// ── New: cuStreamDestroy — handle-consuming ──
struct StreamDestroyRet { int32_t result; };
StreamDestroyRet stream_destroy(std::uint64_t stream_handle) {
    auto real_stream_val = g_vhandles.translate(stream_handle);
    CUstream cu_stream = reinterpret_cast<CUstream>(real_stream_val);
    CUresult r = cuStreamDestroy(cu_stream);
    std::cerr << "  [server] cuStreamDestroy(vh=0x" << std::hex << stream_handle << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

// ── New: cuEventCreate — handle-producing ──
struct EventCreateRet { int32_t result; std::uint64_t event_handle; };
EventCreateRet event_create(unsigned int flags) {
    handle_producer_guard guard;
    CUevent event = nullptr;
    CUresult r = cuEventCreate(&event, flags);
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<std::uint64_t>(event));
    std::cerr << "  [server] cuEventCreate() -> event=" << event << "\n";
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(event)};
}

struct EventRecordRet { int32_t result; };
EventRecordRet event_record(std::uint64_t event_handle, std::uint64_t stream_handle) {
    auto real_event_val = g_vhandles.translate(event_handle);
    auto real_stream_val = g_vhandles.translate(stream_handle);
    CUevent cu_event = reinterpret_cast<CUevent>(real_event_val);
    CUstream cu_stream = (stream_handle == 0) ? nullptr
        : reinterpret_cast<CUstream>(real_stream_val);
    CUresult r = cuEventRecord(cu_event, cu_stream);
    std::cerr << "  [server] cuEventRecord(vh_event=0x" << std::hex << event_handle
              << ", vh_stream=0x" << stream_handle << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

struct EventSyncRet { int32_t result; };
EventSyncRet event_synchronize(std::uint64_t event_handle) {
    auto real_event_val = g_vhandles.translate(event_handle);
    CUevent cu_event = reinterpret_cast<CUevent>(real_event_val);
    CUresult r = cuEventSynchronize(cu_event);
    std::cerr << "  [server] cuEventSynchronize(vh=0x" << std::hex << event_handle << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

} // namespace cuda_rpc_api

// ── RPC binding ────────────────────────────────────────────────────────
using cuda_test_rpc = zpp::bits::rpc<
    zpp::bits::bind<&cuda_rpc_api::cuda_init,           0>,
    zpp::bits::bind<&cuda_rpc_api::get_device_count,    1>,
    zpp::bits::bind<&cuda_rpc_api::get_device_name,     2>,
    zpp::bits::bind<&cuda_rpc_api::get_device_total_mem,3>,
    zpp::bits::bind<&cuda_rpc_api::ctx_create,          4>,
    zpp::bits::bind<&cuda_rpc_api::ctx_synchronize,     5>,
    zpp::bits::bind<&cuda_rpc_api::ctx_destroy,         6>,
    zpp::bits::bind<&cuda_rpc_api::mem_alloc,           7>,
    zpp::bits::bind<&cuda_rpc_api::mem_free,            8>,
    zpp::bits::bind<&cuda_rpc_api::memcpy_htod,         9>,
    zpp::bits::bind<&cuda_rpc_api::memcpy_dtoh,         10>,
    zpp::bits::bind<&cuda_rpc_api::module_load_data,    11>,
    zpp::bits::bind<&cuda_rpc_api::module_get_function, 12>,
    zpp::bits::bind<&cuda_rpc_api::launch_kernel,       13>,
    zpp::bits::bind<&cuda_rpc_api::stream_create,       14>,
    zpp::bits::bind<&cuda_rpc_api::stream_synchronize,  15>,
    zpp::bits::bind<&cuda_rpc_api::stream_destroy,      16>,
    zpp::bits::bind<&cuda_rpc_api::event_create,        17>,
    zpp::bits::bind<&cuda_rpc_api::event_record,        18>,
    zpp::bits::bind<&cuda_rpc_api::event_synchronize,   19>
>;

// ── Handle manifest: which functions produce handles, and which field ──
// This tells the pipeline processor which return values to register
// in the virtual handle table.
struct handle_producer_info {
    int func_index;        // RPC function index
    int handle_field;      // 0-indexed field in the return struct that's a handle
};

// Functions that produce handles (return structs with handle fields)
static const handle_producer_info handle_producers[] = {
    {4,  1},   // ctx_create:        CtxCreateRet { result(0), ctx_handle(1) }
    {7,  1},   // mem_alloc:         AllocRet { result(0), dev_ptr(1) }
    {11, 1},   // module_load_data:  ModuleLoadRet { result(0), module_handle(1) }
    {12, 1},   // module_get_function: ModuleGetFuncRet { result(0), func_handle(1) }
    {14, 1},   // stream_create:     StreamCreateRet { result(0), stream_handle(1) }
    {17, 1},   // event_create:      EventCreateRet { result(0), event_handle(1) }
};
static constexpr int handle_producer_count = sizeof(handle_producers) / sizeof(handle_producers[0]);

// Check if a function produces a handle, return its field index or -1
static int get_handle_producer_field(int func_index) {
    for (int i = 0; i < handle_producer_count; i++) {
        if (handle_producers[i].func_index == func_index)
            return handle_producers[i].handle_field;
    }
    return -1;
}

// ── Handle a memory_op frame ──────────────────────────────────────────
static bool handle_memory_op(zlink::transport& tp, const zlink::frame& req_frame) {
    if (req_frame.payload.size() < sizeof(zlink::mem_request)) {
        std::cerr << "  [server] memory_op frame too small\n";
        return false;
    }

    zlink::mem_request mem_req;
    std::memcpy(&mem_req, req_frame.payload.data(), sizeof(mem_req));

    zlink::frame resp_frame;
    resp_frame.call_id = req_frame.call_id;
    resp_frame.type = zlink::frame_type::memory_reply;

    switch (mem_req.op) {
    case zlink::mem_op::host_sync: {
        std::span<const std::byte> sync_data(
            req_frame.payload.data() + sizeof(mem_req),
            req_frame.payload.size() - sizeof(mem_req)
        );
        g_host_mirror.sync_page(mem_req.remote_addr, sync_data);
        std::cerr << "  [server] host_sync: addr=0x" << std::hex
                  << mem_req.remote_addr << std::dec
                  << " size=" << sync_data.size() << "\n";

        zlink::mem_response resp{zlink::error_code::ok, sync_data.size(), 0};
        std::vector<std::byte> resp_data(sizeof(resp));
        std::memcpy(resp_data.data(), &resp, sizeof(resp));
        resp_frame.payload = resp_data;
        break;
    }

    case zlink::mem_op::host_read: {
        std::vector<std::byte> mirror_data(mem_req.size, std::byte{0});
        std::size_t bytes_read = 0;

        auto mirror_addr = g_host_mirror.translate(mem_req.remote_addr);
        if (mirror_addr) {
            std::memcpy(mirror_data.data(),
                        reinterpret_cast<const void*>(*mirror_addr),
                        static_cast<std::size_t>(mem_req.size));
            bytes_read = static_cast<std::size_t>(mem_req.size);
        }

        zlink::mem_response resp{zlink::error_code::ok, bytes_read, 0};
        std::vector<std::byte> resp_data(sizeof(resp) + bytes_read);
        std::memcpy(resp_data.data(), &resp, sizeof(resp));
        if (bytes_read > 0) {
            std::memcpy(resp_data.data() + sizeof(resp), mirror_data.data(), bytes_read);
        }
        resp_frame.payload = resp_data;
        break;
    }

    default:
        std::cerr << "  [server] unknown memory_op: " << static_cast<int>(mem_req.op) << "\n";
        zlink::mem_response resp{zlink::error_code::server_error, 0, 0};
        std::vector<std::byte> resp_data(sizeof(resp));
        std::memcpy(resp_data.data(), &resp, sizeof(resp));
        resp_frame.payload = resp_data;
        break;
    }

    auto ec = tp.send(resp_frame);
    return !ec;
}

// ── Handle a pipeline_request frame ──────────────────────────────────
static bool handle_pipeline(zlink::transport& tp, const zlink::frame& req_frame) {
    if (req_frame.payload.size() < 4) return false;

    std::uint32_t count = 0;
    std::memcpy(&count, req_frame.payload.data(), 4);

    std::cerr << "  [server] pipeline: " << count << " calls\n";

    std::vector<std::byte> resp_payload;
    std::uint32_t resp_count = 0;
    resp_payload.resize(4);
    std::memcpy(resp_payload.data(), &resp_count, 4);

    std::size_t offset = 4;
    for (std::uint32_t i = 0; i < count && offset + 4 <= req_frame.payload.size(); i++) {
        std::uint32_t req_len = 0;
        std::memcpy(&req_len, req_frame.payload.data() + offset, 4);
        offset += 4;

        if (offset + req_len > req_frame.payload.size()) break;

        auto [data, in, out] = zpp::bits::data_in_out();
        data.assign(req_frame.payload.data() + offset,
                    req_frame.payload.data() + offset + req_len);
        offset += req_len;

        cuda_test_rpc::server server{in, out};
        auto result = server.serve();
        if (zpp::bits::failure(result)) {
            std::cerr << "  [server] pipeline call " << i << " serve error\n";
            break;
        }

        std::uint32_t resp_len = static_cast<std::uint32_t>(data.size());
        std::size_t prev_size = resp_payload.size();
        resp_payload.resize(prev_size + 4 + resp_len);
        std::memcpy(resp_payload.data() + prev_size, &resp_len, 4);
        std::memcpy(resp_payload.data() + prev_size + 4, data.data(), resp_len);
        resp_count++;
    }

    std::memcpy(resp_payload.data(), &resp_count, 4);
    std::cerr << "  [server] pipeline done: " << resp_count << " responses\n";

    zlink::frame resp_frame;
    resp_frame.call_id = req_frame.call_id;
    resp_frame.type = zlink::frame_type::pipeline_response;
    resp_frame.payload = resp_payload;

    auto ec = tp.send(resp_frame);
    return !ec;
}

// ── Handle a pipeline_mem frame (v2: with handle manifest + LZ4 compression) ──
// Wire format (request):
//   [4B sync_count]
//   for each sync: [8B addr][8B original_size][1B comp_flag][4B data_size][data...]
//     comp_flag: 0=raw, 1=LZ4 compressed
//     data_size: wire bytes (compressed or raw)
//   [4B rpc_count]
//   for each rpc: [4B len][rpc_bytes]
//   [4B read_count]
//   for each read: [8B addr][8B size]
//   [handle_manifest: 4B count + N × handle_manifest_entry]
//
// Wire format (response):
//   [4B rpc_count] [4B len1][resp1]...
//   [4B read_count]
//   for each read: [4B data_len][1B comp_flag][data...]
//     When comp_flag=LZ4: data = [8B original_size][compressed_bytes]
//
// Handle registration works via the handle_producer_guard mechanism:
// Each handle-producing wrapper function sets g_last_produced_handle.
// After serve(), we check the manifest for this call index and
// register the virtual→real mapping.
static bool handle_pipeline_mem(zlink::transport& tp, const zlink::frame& req_frame) {
    if (req_frame.payload.size() < 4) return false;

    std::size_t off = 0;

    // ── 0. Parse handle manifest FIRST ──────────────────────────────
    // We need the manifest before processing RPCs, so we know which
    // virtual IDs to register when handles are produced.
    // Skip to the end of the frame to find the manifest.
    // Actually, the manifest is at the end. We'll parse it lazily.
    // For now, store the full payload for later manifest parsing.
    auto payload_span = std::span<const std::byte>(req_frame.payload.data(), req_frame.payload.size());

    // ── 1. Process sync entries ─────────────────────────────────────
    std::uint32_t sync_count = 0;
    std::memcpy(&sync_count, req_frame.payload.data() + off, 4); off += 4;

    std::cerr << "  [server] pipeline_mem: " << sync_count << " syncs";

    for (std::uint32_t i = 0; i < sync_count; i++) {
        if (off + 17 > req_frame.payload.size()) break;  // 8 addr + 8 orig_size + 1 comp_flag

        std::uint64_t addr = 0, orig_sz = 0;
        std::uint8_t comp_flag = 0;
        std::memcpy(&addr, req_frame.payload.data() + off, 8); off += 8;
        std::memcpy(&orig_sz, req_frame.payload.data() + off, 8); off += 8;
        std::memcpy(&comp_flag, req_frame.payload.data() + off, 1); off += 1;

        // Determine data size: if compressed, we need to read compressed bytes
        // until the next entry. For raw data, data size = orig_sz.
        // With the new format, the data follows directly after comp_flag.
        // The remaining data for this sync entry is orig_sz bytes for raw,
        // or a variable amount for compressed data.
        // We read orig_sz bytes for raw; for compressed, we don't know the
        // compressed size from the header alone. But since entries are
        // sequential, we need a different approach: the wire format includes
        // the original_size so the decompressor knows the expected output,
        // but we need to know how many compressed bytes to read.
        //
        // Solution: for compressed entries, we read the remaining payload
        // up to what makes sense. The client writes the compressed data
        // contiguously, so we need to know the compressed size.
        // We DON'T have it in the header. Two options:
        //   1. Add a 4B compressed_size field before the data
        //   2. Use orig_sz for raw, and for compressed, store [4B comp_size][data]
        //
        // Actually, looking at the client code: it writes comp_flag then data
        // contiguously. We need a compressed_size field. Let me re-check...
        //
        // The current client writes: [8B addr][8B orig_sz][1B comp_flag][data...]
        // For raw: data is orig_sz bytes.
        // For LZ4: data is compressed_size bytes (unknown to reader!).
        //
        // Fix: add [4B data_size] after comp_flag so the reader knows
        // exactly how many bytes to consume.
        //
        // Updated format: [8B addr][8B orig_sz][1B comp_flag][4B data_size][data...]
        // This is a BREAKING CHANGE to the wire format.

        std::uint32_t data_size = 0;
        if (off + 4 > req_frame.payload.size()) break;
        std::memcpy(&data_size, req_frame.payload.data() + off, 4); off += 4;

        if (off + data_size > req_frame.payload.size()) break;

        // Decompress if needed
        std::vector<std::byte> sync_data;
        if (comp_flag == zlink::comp_flag_lz4) {
            sync_data = zlink::decompress(
                std::span<const std::byte>(req_frame.payload.data() + off, data_size),
                comp_flag, static_cast<std::size_t>(orig_sz));
            if (sync_data.empty()) {
                std::cerr << "  [server] WARNING: LZ4 decompression failed for sync entry " << i << "\n";
            }
        } else {
            sync_data.assign(
                req_frame.payload.data() + off,
                req_frame.payload.data() + off + data_size);
        }
        off += data_size;

        g_host_mirror.sync_page(static_cast<std::uintptr_t>(addr),
                                std::span<const std::byte>(sync_data.data(), sync_data.size()));
        std::cerr << ", sync(addr=0x" << std::hex << addr
                  << ",orig=" << std::dec << orig_sz
                  << ",comp=" << (comp_flag ? "LZ4" : "raw")
                  << ",wire=" << data_size << ")";
    }
    std::cerr << "\n";

    // ── 2. Process RPC calls ────────────────────────────────────────
    std::uint32_t rpc_count = 0;
    std::memcpy(&rpc_count, req_frame.payload.data() + off, 4); off += 4;

    std::cerr << "  [server] pipeline_mem: " << rpc_count << " RPCs\n";

    std::vector<std::byte> rpc_responses;
    std::uint32_t rpc_resp_count = 0;
    rpc_responses.resize(4);
    std::memcpy(rpc_responses.data(), &rpc_resp_count, 4);

    for (std::uint32_t i = 0; i < rpc_count && off + 4 <= req_frame.payload.size(); i++) {
        std::uint32_t req_len = 0;
        std::memcpy(&req_len, req_frame.payload.data() + off, 4); off += 4;

        if (off + req_len > req_frame.payload.size()) break;

        // Reset handle production flag before serving
        g_has_produced_handle = false;

        auto [data, in, out] = zpp::bits::data_in_out();
        data.assign(req_frame.payload.data() + off,
                    req_frame.payload.data() + off + req_len);
        off += req_len;

        cuda_test_rpc::server server{in, out};
        auto result = server.serve();
        if (zpp::bits::failure(result)) {
            std::cerr << "  [server] pipeline_mem RPC " << i << " serve error\n";
            break;
        }

        // If this call produced a handle, register it with a temporary ID.
        // We'll update with the correct virtual ID from the manifest later.
        // For now, register under the call index as a placeholder.
        if (g_has_produced_handle) {
            // Register under call index temporarily
            g_vhandles.register_handle(i, g_last_produced_handle);
            std::cerr << "  [server]   handle produced: call " << i
                      << " -> real=0x" << std::hex << g_last_produced_handle
                      << std::dec << "\n";
        }

        std::uint32_t resp_len = static_cast<std::uint32_t>(data.size());
        std::size_t prev_size = rpc_responses.size();
        rpc_responses.resize(prev_size + 4 + resp_len);
        std::memcpy(rpc_responses.data() + prev_size, &resp_len, 4);
        std::memcpy(rpc_responses.data() + prev_size + 4, data.data(), resp_len);
        rpc_resp_count++;
    }

    std::memcpy(rpc_responses.data(), &rpc_resp_count, 4);

    // ── 3. Process read requests ────────────────────────────────────
    std::uint32_t read_count = 0;
    if (off + 4 <= req_frame.payload.size()) {
        std::memcpy(&read_count, req_frame.payload.data() + off, 4); off += 4;
    }

    std::cerr << "  [server] pipeline_mem: " << read_count << " reads\n";

    std::vector<std::byte> read_responses;
    std::uint32_t read_resp_count = 0;
    read_responses.resize(4);
    std::memcpy(read_responses.data(), &read_resp_count, 4);

    for (std::uint32_t i = 0; i < read_count; i++) {
        if (off + 16 > req_frame.payload.size()) break;

        std::uint64_t addr = 0, sz = 0;
        std::memcpy(&addr, req_frame.payload.data() + off, 8); off += 8;
        std::memcpy(&sz, req_frame.payload.data() + off, 8); off += 8;

        std::vector<std::byte> mirror_data(static_cast<std::size_t>(sz), std::byte{0});
        std::size_t bytes_read = 0;

        auto mirror_addr = g_host_mirror.translate(static_cast<std::uintptr_t>(addr));
        if (mirror_addr) {
            std::memcpy(mirror_data.data(),
                        reinterpret_cast<const void*>(*mirror_addr),
                        static_cast<std::size_t>(sz));
            bytes_read = static_cast<std::size_t>(sz);
        }

        // Compress read data with LZ4 (large reads benefit significantly)
        auto compressed = zlink::compress(
            std::span<const std::byte>(mirror_data.data(), bytes_read));

        // Response format: [4B data_len][1B comp_flag][data...]
        // When LZ4: data = [8B original_size][compressed_bytes]
        // When raw: data = raw_bytes
        std::size_t data_field_size = 0;
        if (compressed.comp_flag == zlink::comp_flag_lz4) {
            // [8B original_size][compressed_bytes]
            data_field_size = 8 + compressed.data.size();
        } else {
            // raw bytes
            data_field_size = compressed.data.size();
        }

        std::uint32_t total_len = static_cast<std::uint32_t>(1 + data_field_size); // 1 for comp_flag
        std::size_t prev_size = read_responses.size();
        read_responses.resize(prev_size + 4 + total_len);
        std::memcpy(read_responses.data() + prev_size, &total_len, 4);
        std::size_t write_off = prev_size + 4;

        // Write comp_flag
        std::uint8_t cflag = compressed.comp_flag;
        std::memcpy(read_responses.data() + write_off, &cflag, 1); write_off += 1;

        if (compressed.comp_flag == zlink::comp_flag_lz4) {
            // Write original_size then compressed data
            std::uint64_t orig_sz = compressed.original_size;
            std::memcpy(read_responses.data() + write_off, &orig_sz, 8); write_off += 8;
            std::memcpy(read_responses.data() + write_off, compressed.data.data(),
                        compressed.data.size());
        } else {
            // Write raw data
            std::memcpy(read_responses.data() + write_off, compressed.data.data(),
                        compressed.data.size());
        }

        read_resp_count++;
        std::cerr << "  [server]   read " << i << ": " << bytes_read << " bytes, "
                  << (compressed.comp_flag == zlink::comp_flag_lz4 ? "LZ4" : "raw")
                  << " (" << compressed.data.size() << " on wire)\n";
    }

    std::memcpy(read_responses.data(), &read_resp_count, 4);

    // ── 4. Process handle manifest ─────────────────────────────────
    // The manifest maps: call_index → virtual_id
    // We registered handles under call_index during processing.
    // Now add the virtual_id → same_real_handle mapping.
    if (off + 4 <= req_frame.payload.size()) {
        std::size_t manifest_bytes = 0;
        auto manifest = zlink::parse_handle_manifest(
            std::span<const std::byte>(req_frame.payload.data() + off,
                                       req_frame.payload.size() - off),
            manifest_bytes);
        off += manifest_bytes;

        if (!manifest.empty()) {
            std::cerr << "  [server] pipeline_mem: " << manifest.size() << " handle registrations\n";

            for (const auto& entry : manifest) {
                // The handle was registered under call_index during processing.
                // Look it up and re-register under the virtual_id.
                auto real_handle = g_vhandles.lookup(entry.call_index);
                if (real_handle) {
                    g_vhandles.register_handle(entry.virtual_id, *real_handle);
                    std::cerr << "  [server]   VH(" << entry.virtual_id
                              << ") -> real=0x" << std::hex << *real_handle
                              << std::dec << "\n";
                }
            }
        }
    }

    // ── 5. Combine and send response ────────────────────────────────
    std::vector<std::byte> resp_payload;
    resp_payload.reserve(rpc_responses.size() + read_responses.size());
    resp_payload.insert(resp_payload.end(), rpc_responses.begin(), rpc_responses.end());
    resp_payload.insert(resp_payload.end(), read_responses.begin(), read_responses.end());

    std::cerr << "  [server] pipeline_mem done: " << rpc_resp_count
              << " RPC responses, " << read_resp_count << " reads"
              << ", handles=" << g_vhandles.size() << "\n";

    zlink::frame resp_frame;
    resp_frame.call_id = req_frame.call_id;
    resp_frame.type = zlink::frame_type::pipeline_response;
    resp_frame.payload = resp_payload;

    auto ec = tp.send(resp_frame);
    return !ec;
}

// ── Main ───────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::uint16_t port = zlink::default_port;
    if (argc > 1) port = static_cast<std::uint16_t>(std::stoi(argv[1]));

    std::cout << "zlink CUDA test server — virtual handles + GPU-over-IP\n"
              << "Listening on port " << port << "...\n" << std::endl;

    auto tp = zlink::make_transport(zlink::transport_kind::tcp);
    auto ec = tp->listen("0.0.0.0", port);
    if (ec) {
        std::cerr << "Failed to listen: " << ec.message() << "\n";
        return 1;
    }

    ec = tp->accept();
    if (ec) {
        std::cerr << "Failed to accept: " << ec.message() << "\n";
        return 1;
    }

    std::cout << "Client connected. Serving CUDA RPC calls...\n";

    while (tp->is_connected()) {
        try {
            zlink::frame req_frame;
            ec = tp->receive(req_frame);
            if (ec) {
                std::cerr << "Receive error: " << ec.message() << "\n";
                break;
            }

            if (req_frame.type == zlink::frame_type::memory_op) {
                if (!handle_memory_op(*tp, req_frame)) break;
                continue;
            }

            if (req_frame.type == zlink::frame_type::pipeline_request) {
                if (!handle_pipeline(*tp, req_frame)) break;
                continue;
            }

            if (req_frame.type == zlink::frame_type::pipeline_mem) {
                if (!handle_pipeline_mem(*tp, req_frame)) break;
                continue;
            }

            // Single RPC call
            auto [data, in, out] = zpp::bits::data_in_out();
            data.assign(req_frame.payload.begin(), req_frame.payload.end());

            cuda_test_rpc::server server{in, out};
            auto result = server.serve();
            if (zpp::bits::failure(result)) {
                std::cerr << "Serve error\n";
                break;
            }

            zlink::frame resp_frame;
            resp_frame.call_id = req_frame.call_id;
            resp_frame.type = zlink::frame_type::response;
            resp_frame.payload.assign(data.begin(), data.end());

            ec = tp->send(resp_frame);
            if (ec) break;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            break;
        }
    }

    std::cout << "CUDA test server shutting down.\n";
    return 0;
}
