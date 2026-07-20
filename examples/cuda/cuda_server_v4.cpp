// zlink/examples/cuda/cuda_server_v4.cpp
//
// CUDA RPC server v4 — Full PyTorch API with virtual handles.
//
// Handles ALL 48 CUDA Driver API functions from cuda_api_v2.hpp,
// including the 10 PyTorch-critical additions (indices 38-47).
//
// Key additions over v3:
//   - cuDeviceGet (38)
//   - cuDevicePrimaryCtxRetain (39) ★★★ CRITICAL for PyTorch
//   - cuDevicePrimaryCtxRelease (40)
//   - cuDevicePrimaryCtxSetFlags (41)
//   - cuDevicePrimaryCtxGetState (42)
//   - cuGetProcAddress (43) ★★★ CRITICAL for PyTorch 2.0+
//   - cuCtxPushCurrent (44)
//   - cuCtxPopCurrent (45)
//   - cuStreamCreateWithPriority (46)
//   - cuMemHostGetDevicePointer (47)

#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/ptr_map.hpp>
#include <zlink/memory.hpp>
#include <zlink/chunk_cache.hpp>
#include <zlink/config.hpp>
#include <zlink/virtual_handle.hpp>

#include "cuda_api_v2.hpp"

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
static zlink::handle_table g_vhandles;

// ── Handle production hook ─────────────────────────────────────────────
static std::uint64_t g_last_produced_handle = 0;
static bool g_has_produced_handle = false;

struct handle_producer_guard {
    handle_producer_guard() : produced(false) {}
    ~handle_producer_guard() { if (produced) g_has_produced_handle = true; }
    void set(std::uint64_t real_handle) {
        g_last_produced_handle = real_handle;
        produced = true;
    }
    bool produced;
};

// ══════════════════════════════════════════════════════════════════════
// SERVER-SIDE CUDA API IMPLEMENTATIONS
// ══════════════════════════════════════════════════════════════════════

namespace cuda_api_v2 {

// ── Initialization & Device Management (indices 0-4) ──────────────────

InitRet cu_init(unsigned int flags) {
    std::cerr << "  [server] cuInit(" << flags << ")\n";
    return {static_cast<int32_t>(cuInit(flags))};
}

DevCountRet device_get_count() {
    int count = 0;
    CUresult r = cuDeviceGetCount(&count);
    std::cerr << "  [server] cuDeviceGetCount() -> " << count << "\n";
    return {static_cast<int32_t>(r), count};
}

DevNameRet device_get_name(int ordinal) {
    char name[256] = {};
    CUdevice dev;
    CUresult r1 = cuDeviceGet(&dev, ordinal);
    if (r1 != CUDA_SUCCESS) return {static_cast<int32_t>(r1), ""};
    CUresult r2 = cuDeviceGetName(name, sizeof(name), dev);
    std::cerr << "  [server] cuDeviceGetName(" << ordinal << ") -> \"" << name << "\"\n";
    return {static_cast<int32_t>(r2), std::string(name)};
}

DevMemRet device_total_mem(int ordinal) {
    CUdevice dev;
    CUresult r1 = cuDeviceGet(&dev, ordinal);
    if (r1 != CUDA_SUCCESS) return {static_cast<int32_t>(r1), 0};
    size_t bytes = 0;
    CUresult r2 = cuDeviceTotalMem(&bytes, dev);
    std::cerr << "  [server] cuDeviceTotalMem(" << ordinal << ") -> " << bytes << "\n";
    return {static_cast<int32_t>(r2), static_cast<std::uint64_t>(bytes)};
}

DevAttrRet device_get_attribute(int attrib, int device_ordinal) {
    CUdevice dev;
    CUresult r1 = cuDeviceGet(&dev, device_ordinal);
    if (r1 != CUDA_SUCCESS) return {static_cast<int32_t>(r1), 0};
    int value = 0;
    CUresult r2 = cuDeviceGetAttribute(&value, static_cast<CUdevice_attribute>(attrib), dev);
    std::cerr << "  [server] cuDeviceGetAttribute(" << attrib << ", dev=" << device_ordinal << ") -> " << value << "\n";
    return {static_cast<int32_t>(r2), value};
}

// ── Context Management (indices 5-9) ──────────────────────────────────

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

CtxDestroyRet ctx_destroy(std::uint64_t ctx_handle) {
    auto real_ctx = g_vhandles.translate(ctx_handle);
    CUresult r = cuCtxDestroy(reinterpret_cast<CUcontext>(real_ctx));
    std::cerr << "  [server] cuCtxDestroy(vh=0x" << std::hex << ctx_handle
              << " -> real=0x" << real_ctx << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

CtxSetCurrentRet ctx_set_current(std::uint64_t ctx_handle) {
    auto real_ctx = g_vhandles.translate(ctx_handle);
    CUresult r = cuCtxSetCurrent(reinterpret_cast<CUcontext>(real_ctx));
    std::cerr << "  [server] cuCtxSetCurrent(vh=0x" << std::hex << ctx_handle << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

CtxGetCurrentRet ctx_get_current() {
    handle_producer_guard guard;
    CUcontext ctx = nullptr;
    CUresult r = cuCtxGetCurrent(&ctx);
    if (r == CUDA_SUCCESS && ctx) guard.set(reinterpret_cast<std::uint64_t>(ctx));
    std::cerr << "  [server] cuCtxGetCurrent() -> ctx=" << ctx << "\n";
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(ctx)};
}

CtxSyncRet ctx_synchronize() {
    CUresult r = cuCtxSynchronize();
    std::cerr << "  [server] cuCtxSynchronize()\n";
    return {static_cast<int32_t>(r)};
}

// ── Memory Management (indices 10-17) ─────────────────────────────────

AllocRet mem_alloc(std::uint64_t bytesize) {
    handle_producer_guard guard;
    CUdeviceptr ptr = 0;
    CUresult r = cuMemAlloc(&ptr, bytesize);
    if (r == CUDA_SUCCESS) guard.set(ptr);
    std::cerr << "  [server] cuMemAlloc(" << bytesize << ") -> ptr=0x"
              << std::hex << ptr << std::dec << "\n";
    return {static_cast<int32_t>(r), ptr};
}

AllocManagedRet mem_alloc_managed(std::uint64_t bytesize, unsigned int flags) {
    handle_producer_guard guard;
    CUdeviceptr ptr = 0;
    CUresult r = cuMemAllocManaged(&ptr, bytesize, flags);
    if (r == CUDA_SUCCESS) guard.set(ptr);
    std::cerr << "  [server] cuMemAllocManaged(" << bytesize << ", flags=" << flags
              << ") -> ptr=0x" << std::hex << ptr << std::dec << "\n";
    return {static_cast<int32_t>(r), ptr};
}

FreeRet mem_free(std::uint64_t dev_ptr) {
    auto real_ptr = g_vhandles.translate(dev_ptr);
    CUresult r = cuMemFree(static_cast<CUdeviceptr>(real_ptr));
    std::cerr << "  [server] cuMemFree(vh=0x" << std::hex << dev_ptr
              << " -> real=0x" << real_ptr << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

FreeHostRet mem_free_host(std::uint64_t host_ptr) {
    auto real_ptr = g_vhandles.translate(host_ptr);
    CUresult r = cuMemFreeHost(reinterpret_cast<void*>(real_ptr));
    std::cerr << "  [server] cuMemFreeHost(vh=0x" << std::hex << host_ptr << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

HostAllocRet mem_host_alloc(std::uint64_t bytesize, unsigned int flags) {
    handle_producer_guard guard;
    void* ptr = nullptr;
    CUresult r = cuMemHostAlloc(&ptr, bytesize, flags);
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<std::uint64_t>(ptr));
    std::cerr << "  [server] cuMemHostAlloc(" << bytesize << ") -> ptr=" << ptr << "\n";
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(ptr)};
}

HostRegisterRet mem_host_register(std::uint64_t host_ptr, std::uint64_t bytesize, unsigned int flags) {
    auto real_ptr = g_vhandles.translate(host_ptr);
    CUresult r = cuMemHostRegister(reinterpret_cast<void*>(real_ptr), bytesize, flags);
    std::cerr << "  [server] cuMemHostRegister(vh=0x" << std::hex << host_ptr << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

HostUnregisterRet mem_host_unregister(std::uint64_t host_ptr) {
    auto real_ptr = g_vhandles.translate(host_ptr);
    CUresult r = cuMemHostUnregister(reinterpret_cast<void*>(real_ptr));
    std::cerr << "  [server] cuMemHostUnregister(vh=0x" << std::hex << host_ptr << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

MemInfoRet mem_get_info() {
    size_t free_bytes = 0, total_bytes = 0;
    CUresult r = cuMemGetInfo(&free_bytes, &total_bytes);
    std::cerr << "  [server] cuMemGetInfo() -> free=" << (free_bytes/(1024*1024))
              << " MB, total=" << (total_bytes/(1024*1024)) << " MB\n";
    return {static_cast<int32_t>(r), static_cast<std::uint64_t>(free_bytes),
            static_cast<std::uint64_t>(total_bytes)};
}

// ── Memory Copy (indices 18-22) ───────────────────────────────────────

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
              << " -> dst=0x" << real_dst << ", n=" << std::dec << byte_count << ")\n";
    return {static_cast<int32_t>(r)};
}

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
    std::cerr << "  [server] cuMemcpyDtoH(vh=0x" << std::hex << src_dev_ptr
              << " -> src=0x" << real_src << ", n=" << std::dec << byte_count << ")\n";
    return {static_cast<int32_t>(r)};
}

CopyDtoDRet memcpy_dtod(std::uint64_t dst_dev_ptr,
                         std::uint64_t src_dev_ptr,
                         std::uint64_t byte_count) {
    auto real_dst = g_vhandles.translate(dst_dev_ptr);
    auto real_src = g_vhandles.translate(src_dev_ptr);
    CUresult r = cuMemcpyDtoD(static_cast<CUdeviceptr>(real_dst),
                               static_cast<CUdeviceptr>(real_src),
                               byte_count);
    std::cerr << "  [server] cuMemcpyDtoD(vh_dst=0x" << std::hex << dst_dev_ptr
              << " -> 0x" << real_dst << ", vh_src=0x" << src_dev_ptr
              << " -> 0x" << real_src << ", n=" << std::dec << byte_count << ")\n";
    return {static_cast<int32_t>(r)};
}

CopyHtoDAsyncRet memcpy_htod_async(std::uint64_t dst_dev_ptr,
                                     std::uint64_t src_client_addr,
                                     std::uint64_t byte_count,
                                     std::uint64_t stream_handle) {
    auto real_dst = g_vhandles.translate(dst_dev_ptr);
    auto real_stream_val = g_vhandles.translate(stream_handle);
    CUstream cu_stream = (stream_handle == 0) ? nullptr
        : reinterpret_cast<CUstream>(real_stream_val);

    auto mirror_addr = g_host_mirror.translate(src_client_addr);
    const void* real_src = mirror_addr
        ? reinterpret_cast<const void*>(*mirror_addr)
        : reinterpret_cast<const void*>(src_client_addr);

    CUresult r = cuMemcpyHtoDAsync(static_cast<CUdeviceptr>(real_dst),
                                    real_src, byte_count, cu_stream);
    std::cerr << "  [server] cuMemcpyHtoDAsync(vh=0x" << std::hex << dst_dev_ptr
              << ", stream=0x" << stream_handle << ", n=" << std::dec << byte_count << ")\n";
    return {static_cast<int32_t>(r)};
}

CopyDtoHAsyncRet memcpy_dtoh_async(std::uint64_t dst_client_addr,
                                     std::uint64_t src_dev_ptr,
                                     std::uint64_t byte_count,
                                     std::uint64_t stream_handle) {
    auto real_src = g_vhandles.translate(src_dev_ptr);
    auto real_stream_val = g_vhandles.translate(stream_handle);
    CUstream cu_stream = (stream_handle == 0) ? nullptr
        : reinterpret_cast<CUstream>(real_stream_val);

    auto mirror_addr = g_host_mirror.translate(dst_client_addr);
    void* real_dst = mirror_addr
        ? reinterpret_cast<void*>(*mirror_addr)
        : reinterpret_cast<void*>(dst_client_addr);

    CUresult r = cuMemcpyDtoHAsync(real_dst, static_cast<CUdeviceptr>(real_src),
                                    byte_count, cu_stream);
    std::cerr << "  [server] cuMemcpyDtoHAsync(vh=0x" << std::hex << src_dev_ptr
              << ", stream=0x" << stream_handle << ", n=" << std::dec << byte_count << ")\n";
    return {static_cast<int32_t>(r)};
}

// ── Module & Kernel Management (indices 23-28) ────────────────────────

ModuleLoadRet module_load_data(std::uint64_t src_client_addr, std::uint64_t byte_count) {
    handle_producer_guard guard;
    auto mirror_addr = g_host_mirror.translate(src_client_addr);
    const void* image = mirror_addr
        ? reinterpret_cast<const void*>(*mirror_addr)
        : reinterpret_cast<const void*>(src_client_addr);

    CUmodule mod = nullptr;
    CUresult r = cuModuleLoadData(&mod, image);
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<std::uint64_t>(mod));
    std::cerr << "  [server] cuModuleLoadData() -> mod=" << mod << "\n";
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(mod)};
}

ModuleLoadFileRet module_load(std::string filename) {
    handle_producer_guard guard;
    CUmodule mod = nullptr;
    CUresult r = cuModuleLoad(&mod, filename.c_str());
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<std::uint64_t>(mod));
    std::cerr << "  [server] cuModuleLoad(\"" << filename << "\") -> mod=" << mod << "\n";
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(mod)};
}

ModuleUnloadRet module_unload(std::uint64_t module_handle) {
    auto real_mod = g_vhandles.translate(module_handle);
    CUresult r = cuModuleUnload(reinterpret_cast<CUmodule>(real_mod));
    std::cerr << "  [server] cuModuleUnload(vh=0x" << std::hex << module_handle << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

ModuleGetFuncRet module_get_function(std::uint64_t module_handle, std::string func_name) {
    handle_producer_guard guard;
    auto real_mod = g_vhandles.translate(module_handle);
    CUmodule cu_mod = reinterpret_cast<CUmodule>(real_mod);

    CUfunction func = nullptr;
    CUresult r = cuModuleGetFunction(&func, cu_mod, func_name.c_str());
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<std::uint64_t>(func));
    std::cerr << "  [server] cuModuleGetFunction(vh_mod=0x" << std::hex << module_handle
              << ", name=\"" << func_name << "\") -> func=" << func << std::dec << "\n";
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(func)};
}

ModuleGetGlobalRet module_get_global(std::uint64_t module_handle, std::string global_name) {
    handle_producer_guard guard;
    auto real_mod = g_vhandles.translate(module_handle);
    CUmodule cu_mod = reinterpret_cast<CUmodule>(real_mod);

    CUdeviceptr ptr = 0;
    size_t size = 0;
    CUresult r = cuModuleGetGlobal(&ptr, &size, cu_mod, global_name.c_str());
    if (r == CUDA_SUCCESS) guard.set(ptr);
    std::cerr << "  [server] cuModuleGetGlobal(vh_mod=0x" << std::hex << module_handle
              << ", name=\"" << global_name << "\") -> ptr=0x" << ptr
              << ", size=" << std::dec << size << "\n";
    return {static_cast<int32_t>(r), ptr, static_cast<std::uint64_t>(size)};
}

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

    void* kernel_args[64] = {};
    std::vector<std::byte> args_buffer;

    if (args_byte_count > 0) {
        auto mirror_addr = g_host_mirror.translate(args_client_addr);
        const void* args_src = mirror_addr
            ? reinterpret_cast<const void*>(*mirror_addr)
            : reinterpret_cast<const void*>(args_client_addr);

        args_buffer.resize(args_byte_count);
        std::memcpy(args_buffer.data(), args_src, args_byte_count);

        std::size_t n_args = args_byte_count / sizeof(std::uint64_t);
        std::uint64_t* args_u64 = reinterpret_cast<std::uint64_t*>(args_buffer.data());

        for (std::size_t i = 0; i < n_args && i < 64; i++) {
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
              << ", grid=(" << std::dec << grid_dim_x << "," << grid_dim_y << "," << grid_dim_z << ")"
              << ", block=(" << block_dim_x << "," << block_dim_y << "," << block_dim_z << ")"
              << ", args_n=" << args_byte_count / 8 << ")\n";
    return {static_cast<int32_t>(r)};
}

// ── Stream Management (indices 29-31) ─────────────────────────────────

StreamCreateRet stream_create(unsigned int flags) {
    handle_producer_guard guard;
    CUstream stream = nullptr;
    CUresult r = cuStreamCreate(&stream, flags);
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<std::uint64_t>(stream));
    std::cerr << "  [server] cuStreamCreate(flags=" << flags << ") -> stream=" << stream << "\n";
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(stream)};
}

StreamDestroyRet stream_destroy(std::uint64_t stream_handle) {
    auto real_stream_val = g_vhandles.translate(stream_handle);
    CUstream cu_stream = reinterpret_cast<CUstream>(real_stream_val);
    CUresult r = cuStreamDestroy(cu_stream);
    std::cerr << "  [server] cuStreamDestroy(vh=0x" << std::hex << stream_handle << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

StreamSyncRet stream_synchronize(std::uint64_t stream_handle) {
    auto real_stream_val = g_vhandles.translate(stream_handle);
    CUstream cu_stream = (stream_handle == 0) ? nullptr
        : reinterpret_cast<CUstream>(real_stream_val);
    CUresult r = cuStreamSynchronize(cu_stream);
    std::cerr << "  [server] cuStreamSynchronize(vh=0x" << std::hex << stream_handle << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

// ── Event Management (indices 32-36) ──────────────────────────────────

EventCreateRet event_create(unsigned int flags) {
    handle_producer_guard guard;
    CUevent event = nullptr;
    CUresult r = cuEventCreate(&event, flags);
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<std::uint64_t>(event));
    std::cerr << "  [server] cuEventCreate() -> event=" << event << "\n";
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(event)};
}

EventDestroyRet event_destroy(std::uint64_t event_handle) {
    auto real_event_val = g_vhandles.translate(event_handle);
    CUevent cu_event = reinterpret_cast<CUevent>(real_event_val);
    CUresult r = cuEventDestroy(cu_event);
    std::cerr << "  [server] cuEventDestroy(vh=0x" << std::hex << event_handle << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

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

EventSyncRet event_synchronize(std::uint64_t event_handle) {
    auto real_event_val = g_vhandles.translate(event_handle);
    CUevent cu_event = reinterpret_cast<CUevent>(real_event_val);
    CUresult r = cuEventSynchronize(cu_event);
    std::cerr << "  [server] cuEventSynchronize(vh=0x" << std::hex << event_handle << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

EventElapsedRet event_elapsed_time(std::uint64_t start_event, std::uint64_t end_event) {
    auto real_start = g_vhandles.translate(start_event);
    auto real_end = g_vhandles.translate(end_event);
    CUevent cu_start = reinterpret_cast<CUevent>(real_start);
    CUevent cu_end = reinterpret_cast<CUevent>(real_end);
    float ms = 0.0f;
    CUresult r = cuEventElapsedTime(&ms, cu_start, cu_end);
    std::cerr << "  [server] cuEventElapsedTime() -> " << ms << " ms\n";
    return {static_cast<int32_t>(r), ms};
}

// ── Occupancy (index 37) ──────────────────────────────────────────────

OccupancyRet occupancy_max_potential_block_size(
    std::uint64_t func_handle,
    std::uint32_t shared_mem_bytes,
    int32_t block_size_limit)
{
    auto real_func = g_vhandles.translate(func_handle);
    CUfunction cu_func = reinterpret_cast<CUfunction>(real_func);

    int min_grid_size = 0, block_size = 0;
    CUresult r = cuOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size, cu_func, nullptr,
        shared_mem_bytes, block_size_limit);
    std::cerr << "  [server] cuOccupancyMaxPotentialBlockSize() -> min_grid="
              << min_grid_size << ", block_size=" << block_size << "\n";
    return {static_cast<int32_t>(r), min_grid_size, block_size};
}

// ══════════════════════════════════════════════════════════════════════
// PYTORCH-CRITICAL ADDITIONS (indices 38-47)
// ══════════════════════════════════════════════════════════════════════

// ── cuDeviceGet (index 38) ─────────────────────────────────────────────
// Simple: get device ordinal by index. Just returns the ordinal since
// CUdevice is an integer on the real CUDA API.

DevGetRet device_get(int ordinal) {
    CUdevice dev;
    CUresult r = cuDeviceGet(&dev, ordinal);
    std::cerr << "  [server] cuDeviceGet(" << ordinal << ") -> dev=" << dev << "\n";
    return {static_cast<int32_t>(r), static_cast<int32_t>(dev)};
}

// ── cuDevicePrimaryCtxRetain (index 39) ★★★ CRITICAL ──────────────────
// PyTorch uses this exclusively to get CUDA contexts.
// It never calls cuCtxCreate — instead it retains the primary context
// for a device, which is a shared, lazily-initialized context.
//
// The server calls the real cuDevicePrimaryCtxRetain and returns
// the context handle for the virtual handle system.

PrimaryCtxRetainRet device_primary_ctx_retain(int device_ordinal) {
    handle_producer_guard guard;
    CUdevice dev;
    CUresult r1 = cuDeviceGet(&dev, device_ordinal);
    if (r1 != CUDA_SUCCESS) {
        std::cerr << "  [server] cuDevicePrimaryCtxRetain: cuDeviceGet failed for ordinal "
                  << device_ordinal << "\n";
        return {static_cast<int32_t>(r1), 0};
    }

    CUcontext ctx = nullptr;
    CUresult r2 = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (r2 == CUDA_SUCCESS) {
        guard.set(reinterpret_cast<std::uint64_t>(ctx));
        std::cerr << "  [server] cuDevicePrimaryCtxRetain(dev=" << dev
                  << ") -> ctx=" << ctx << "\n";
    } else {
        std::cerr << "  [server] cuDevicePrimaryCtxRetain FAILED: error="
                  << static_cast<int>(r2) << "\n";
    }
    return {static_cast<int32_t>(r2), reinterpret_cast<std::uint64_t>(ctx)};
}

// ── cuDevicePrimaryCtxRelease (index 40) ──────────────────────────────

PrimaryCtxReleaseRet device_primary_ctx_release(int device_ordinal) {
    CUdevice dev;
    CUresult r1 = cuDeviceGet(&dev, device_ordinal);
    if (r1 != CUDA_SUCCESS) return {static_cast<int32_t>(r1)};

    CUresult r2 = cuDevicePrimaryCtxRelease(dev);
    std::cerr << "  [server] cuDevicePrimaryCtxRelease(dev=" << dev << ")\n";
    return {static_cast<int32_t>(r2)};
}

// ── cuDevicePrimaryCtxSetFlags (index 41) ─────────────────────────────

PrimaryCtxSetFlagsRet device_primary_ctx_set_flags(int device_ordinal, unsigned int flags) {
    CUdevice dev;
    CUresult r1 = cuDeviceGet(&dev, device_ordinal);
    if (r1 != CUDA_SUCCESS) return {static_cast<int32_t>(r1)};

    CUresult r2 = cuDevicePrimaryCtxSetFlags(dev, flags);
    std::cerr << "  [server] cuDevicePrimaryCtxSetFlags(dev=" << dev
              << ", flags=" << flags << ")\n";
    return {static_cast<int32_t>(r2)};
}

// ── cuDevicePrimaryCtxGetState (index 42) ─────────────────────────────

PrimaryCtxGetStateRet device_primary_ctx_get_state(int device_ordinal) {
    CUdevice dev;
    CUresult r1 = cuDeviceGet(&dev, device_ordinal);
    if (r1 != CUDA_SUCCESS) return {static_cast<int32_t>(r1), 0, 0};

    unsigned int flags = 0;
    int active = 0;
    CUresult r2 = cuDevicePrimaryCtxGetState(dev, &flags, &active);
    std::cerr << "  [server] cuDevicePrimaryCtxGetState(dev=" << dev
              << ") -> flags=" << flags << ", active=" << active << "\n";
    return {static_cast<int32_t>(r2), static_cast<int32_t>(flags), active};
}

// ── cuGetProcAddress (index 43) ★★★ CRITICAL ─────────────────────────
// PyTorch 2.0+ uses this to dynamically resolve kernel function pointers.
// The symbol name is typically a mangled CUDA kernel name.
//
// Note: cuGetProcAddress can work with or without a module handle.
// When hmod is null, it searches across all loaded modules.
// When hmod is provided, it searches within that module.

GetProcAddressRet get_proc_address(std::string symbol_name, int32_t cuda_version, std::uint64_t flags) {
    handle_producer_guard guard;

    void* func_ptr = nullptr;
    CUdriverProcAddressQueryResult symbol_status = CU_GET_PROC_ADDRESS_SUCCESS;

    // cuGetProcAddress signature in CUDA 13.2:
    // CUresult cuGetProcAddress(const char *symbol, void **pfn,
    //     int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus)
    CUresult r = cuGetProcAddress(symbol_name.c_str(), &func_ptr,
                                   cuda_version, flags, &symbol_status);

    if (r == CUDA_SUCCESS && func_ptr) {
        guard.set(reinterpret_cast<std::uint64_t>(func_ptr));
        std::cerr << "  [server] cuGetProcAddress(\"" << symbol_name
                  << "\", version=" << cuda_version
                  << ", flags=" << flags
                  << ") -> func=0x" << std::hex
                  << reinterpret_cast<std::uint64_t>(func_ptr)
                  << std::dec << " status=" << static_cast<int>(symbol_status) << "\n";
    } else {
        std::cerr << "  [server] cuGetProcAddress(\"" << symbol_name
                  << "\") FAILED: error=" << static_cast<int>(r)
                  << " status=" << static_cast<int>(symbol_status) << "\n";
    }

    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(func_ptr)};
}

// ── cuCtxPushCurrent (index 44) ────────────────────────────────────────

CtxPushCurrentRet ctx_push_current(std::uint64_t ctx_handle) {
    auto real_ctx = g_vhandles.translate(ctx_handle);
    CUresult r = cuCtxPushCurrent(reinterpret_cast<CUcontext>(real_ctx));
    std::cerr << "  [server] cuCtxPushCurrent(vh=0x" << std::hex << ctx_handle << std::dec << ")\n";
    return {static_cast<int32_t>(r)};
}

// ── cuCtxPopCurrent (index 45) ─────────────────────────────────────────

CtxPopCurrentRet ctx_pop_current() {
    handle_producer_guard guard;
    CUcontext ctx = nullptr;
    CUresult r = cuCtxPopCurrent(&ctx);
    if (r == CUDA_SUCCESS && ctx) {
        guard.set(reinterpret_cast<std::uint64_t>(ctx));
        std::cerr << "  [server] cuCtxPopCurrent() -> ctx=" << ctx << "\n";
    }
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(ctx)};
}

// ── cuStreamCreateWithPriority (index 46) ──────────────────────────────

StreamCreateWithPriorityRet stream_create_with_priority(unsigned int flags, int priority) {
    handle_producer_guard guard;
    CUstream stream = nullptr;
    CUresult r = cuStreamCreateWithPriority(&stream, flags, priority);
    if (r == CUDA_SUCCESS) guard.set(reinterpret_cast<std::uint64_t>(stream));
    std::cerr << "  [server] cuStreamCreateWithPriority(flags=" << flags
              << ", priority=" << priority << ") -> stream=" << stream << "\n";
    return {static_cast<int32_t>(r), reinterpret_cast<std::uint64_t>(stream)};
}

// ── cuMemHostGetDevicePointer (index 47) ──────────────────────────────
// Maps a pinned host memory pointer to a device-accessible pointer.
// This is used by PyTorch for pinned memory async transfers.

HostGetDevPtrRet mem_host_get_device_pointer(std::uint64_t host_ptr) {
    handle_producer_guard guard;
    CUdeviceptr dev_ptr = 0;
    // Note: the real API takes (CUdeviceptr*, void*, unsigned int)
    // The third parameter is always 0 (reserved)
    CUresult r = cuMemHostGetDevicePointer(&dev_ptr,
                                            reinterpret_cast<void*>(host_ptr), 0);
    if (r == CUDA_SUCCESS) guard.set(dev_ptr);
    std::cerr << "  [server] cuMemHostGetDevicePointer(host=0x" << std::hex << host_ptr
              << ") -> dev_ptr=0x" << dev_ptr << std::dec << "\n";
    return {static_cast<int32_t>(r), dev_ptr};
}

} // namespace cuda_api_v2

// ── PyTorch RPC namespace (10 functions, indices 0-9) ────────────────
// The client-side shim uses this smaller RPC type to work around
// zpp_bits template depth issues with 48+ bindings. The server
// must support both the full 48-function RPC and this 10-function RPC.
namespace pytorch_rpc {
    using rpc = zpp::bits::rpc<
        zpp::bits::bind<&cuda_api_v2::device_get,                    0>,
        zpp::bits::bind<&cuda_api_v2::device_primary_ctx_retain,     1>,
        zpp::bits::bind<&cuda_api_v2::device_primary_ctx_release,    2>,
        zpp::bits::bind<&cuda_api_v2::device_primary_ctx_set_flags,  3>,
        zpp::bits::bind<&cuda_api_v2::device_primary_ctx_get_state,  4>,
        zpp::bits::bind<&cuda_api_v2::get_proc_address,              5>,
        zpp::bits::bind<&cuda_api_v2::ctx_push_current,              6>,
        zpp::bits::bind<&cuda_api_v2::ctx_pop_current,               7>,
        zpp::bits::bind<&cuda_api_v2::stream_create_with_priority,   8>,
        zpp::bits::bind<&cuda_api_v2::mem_host_get_device_pointer,   9>
    >;
}

// ── Handle producer info (maps func index → handle field) ────────────
struct handle_producer_info {
    int func_index;
    int handle_field;
};

static const handle_producer_info handle_producers[] = {
    // Original functions (indices 0-37)
    {5,  1},   // ctx_create:        CtxCreateRet { result(0), ctx_handle(1) }
    {8,  1},   // ctx_get_current:   CtxGetCurrentRet { result(0), ctx_handle(1) }
    {10, 1},   // mem_alloc:         AllocRet { result(0), dev_ptr(1) }
    {11, 1},   // mem_alloc_managed: AllocManagedRet { result(0), dev_ptr(1) }
    {14, 1},   // mem_host_alloc:    HostAllocRet { result(0), host_ptr(1) }
    {23, 1},   // module_load_data:  ModuleLoadRet { result(0), module_handle(1) }
    {24, 1},   // module_load:       ModuleLoadFileRet { result(0), module_handle(1) }
    {26, 1},   // module_get_function: ModuleGetFuncRet { result(0), func_handle(1) }
    {27, 1},   // module_get_global: ModuleGetGlobalRet { result(0), global_ptr(1) }
    {29, 1},   // stream_create:     StreamCreateRet { result(0), stream_handle(1) }
    {32, 1},   // event_create:      EventCreateRet { result(0), event_handle(1) }
    // PyTorch-critical additions (indices 38-47)
    {39, 1},   // device_primary_ctx_retain: PrimaryCtxRetainRet { result(0), ctx_handle(1) }
    {43, 1},   // get_proc_address:  GetProcAddressRet { result(0), func_ptr(1) }
    {45, 1},   // ctx_pop_current:   CtxPopCurrentRet { result(0), ctx_handle(1) }
    {46, 1},   // stream_create_with_priority: StreamCreateWithPriorityRet { result(0), stream_handle(1) }
    {47, 1},   // mem_host_get_device_pointer: HostGetDevPtrRet { result(0), dev_ptr(1) }
};
static constexpr int handle_producer_count = sizeof(handle_producers) / sizeof(handle_producers[0]);

static int get_handle_producer_field(int func_index) {
    for (int i = 0; i < handle_producer_count; i++) {
        if (handle_producers[i].func_index == func_index)
            return handle_producers[i].handle_field;
    }
    return -1;
}

// ── Handle a memory_op frame ──────────────────────────────────────────
static bool handle_memory_op(zlink::transport& tp, const zlink::frame& req_frame) {
    if (req_frame.payload.size() < sizeof(zlink::mem_request)) return false;

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

    default: {
        zlink::mem_response resp{zlink::error_code::server_error, 0, 0};
        std::vector<std::byte> resp_data(sizeof(resp));
        std::memcpy(resp_data.data(), &resp, sizeof(resp));
        resp_frame.payload = resp_data;
        break;
    }
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

    std::size_t offset = 4;
    for (std::uint32_t i = 0; i < count && offset + 4 <= req_frame.payload.size(); i++) {
        std::uint32_t req_len = 0;
        std::memcpy(&req_len, req_frame.payload.data() + offset, 4);
        offset += 4;
        if (offset + req_len > req_frame.payload.size()) break;

        g_has_produced_handle = false;

        auto [data, in, out] = zpp::bits::data_in_out();
        data.assign(req_frame.payload.data() + offset,
                    req_frame.payload.data() + offset + req_len);
        offset += req_len;

        cuda_v2_rpc_full::server server{in, out};
        auto result = server.serve();
        if (zpp::bits::failure(result)) {
            std::cerr << "  [server] pipeline call " << i << " serve error\n";
            break;
        }

        if (g_has_produced_handle) {
            static std::uint32_t vh_base = 1000;
            std::uint32_t vh_id = vh_base++;
            g_vhandles.register_handle(vh_id, g_last_produced_handle);
            std::cerr << "  [server]   pipeline handle: call " << i
                      << " -> VH(" << vh_id << ") = real=0x"
                      << std::hex << g_last_produced_handle << std::dec << "\n";
        }

        std::uint32_t resp_len = static_cast<std::uint32_t>(data.size());
        std::size_t prev_size = resp_payload.size();
        resp_payload.resize(prev_size + 4 + resp_len);
        std::memcpy(resp_payload.data() + prev_size, &resp_len, 4);
        std::memcpy(resp_payload.data() + prev_size + 4, data.data(), resp_len);
        resp_count++;
    }

    std::memcpy(resp_payload.data(), &resp_count, 4);

    zlink::frame resp_frame;
    resp_frame.call_id = req_frame.call_id;
    resp_frame.type = zlink::frame_type::pipeline_response;
    resp_frame.payload = resp_payload;

    auto ec = tp.send(resp_frame);
    return !ec;
}

// ── Handle a pipeline_mem frame ─────────────────────────────────────
static bool handle_pipeline_mem(zlink::transport& tp, const zlink::frame& req_frame) {
    if (req_frame.payload.size() < 4) return false;

    std::size_t off = 0;

    // ── 1. Process sync entries ─────────────────────────────────────
    std::uint32_t sync_count = 0;
    std::memcpy(&sync_count, req_frame.payload.data() + off, 4); off += 4;

    std::cerr << "  [server] pipeline_mem: " << sync_count << " syncs";

    for (std::uint32_t i = 0; i < sync_count; i++) {
        if (off + 16 > req_frame.payload.size()) break;
        std::uint64_t addr = 0, sz = 0;
        std::memcpy(&addr, req_frame.payload.data() + off, 8); off += 8;
        std::memcpy(&sz, req_frame.payload.data() + off, 8); off += 8;
        if (off + sz > req_frame.payload.size()) break;

        std::span<const std::byte> sync_data(
            req_frame.payload.data() + off, static_cast<std::size_t>(sz));
        off += sz;

        g_host_mirror.sync_page(static_cast<std::uintptr_t>(addr), sync_data);
        std::cerr << ", sync(addr=0x" << std::hex << addr << ",size=" << std::dec << sz << ")";
    }
    std::cerr << "\n";

    // ── 2. Pre-parse manifest ───────────────────────────────────────
    std::size_t rpc_section_off = off;

    std::uint32_t rpc_count = 0;
    std::memcpy(&rpc_count, req_frame.payload.data() + off, 4); off += 4;

    for (std::uint32_t i = 0; i < rpc_count && off + 4 <= req_frame.payload.size(); i++) {
        std::uint32_t req_len = 0;
        std::memcpy(&req_len, req_frame.payload.data() + off, 4); off += 4;
        off += req_len;
    }

    std::uint32_t read_count = 0;
    if (off + 4 <= req_frame.payload.size()) {
        std::memcpy(&read_count, req_frame.payload.data() + off, 4); off += 4;
    }
    for (std::uint32_t i = 0; i < read_count; i++) {
        off += 16;
    }

    std::vector<zlink::handle_manifest_entry> manifest;
    if (off + 4 <= req_frame.payload.size()) {
        std::size_t manifest_bytes = 0;
        manifest = zlink::parse_handle_manifest(
            std::span<const std::byte>(req_frame.payload.data() + off,
                                       req_frame.payload.size() - off),
            manifest_bytes);
    }

    std::unordered_map<std::uint32_t, std::uint32_t> call_to_vh;
    for (const auto& entry : manifest) {
        call_to_vh[entry.call_index] = entry.virtual_id;
    }

    if (!manifest.empty()) {
        std::cerr << "  [server] pipeline_mem: pre-parsed " << manifest.size() << " manifest entries\n";
    }

    // ── 3. Process RPC calls ────────────────────────────────────────
    off = rpc_section_off + 4;

    std::cerr << "  [server] pipeline_mem: " << rpc_count << " RPCs\n";

    std::vector<std::byte> rpc_responses;
    std::uint32_t rpc_resp_count = 0;
    rpc_responses.resize(4);
    std::memcpy(rpc_responses.data(), &rpc_resp_count, 4);

    for (std::uint32_t i = 0; i < rpc_count && off + 4 <= req_frame.payload.size(); i++) {
        std::uint32_t req_len = 0;
        std::memcpy(&req_len, req_frame.payload.data() + off, 4); off += 4;
        if (off + req_len > req_frame.payload.size()) break;

        g_has_produced_handle = false;

        auto [data, in, out] = zpp::bits::data_in_out();
        data.assign(req_frame.payload.data() + off,
                    req_frame.payload.data() + off + req_len);
        off += req_len;

        cuda_v2_rpc_full::server server{in, out};
        auto result = server.serve();
        if (zpp::bits::failure(result)) {
            std::cerr << "  [server] pipeline_mem RPC " << i << " serve error\n";
            break;
        }

        if (g_has_produced_handle) {
            auto it = call_to_vh.find(i);
            if (it != call_to_vh.end()) {
                g_vhandles.register_handle(it->second, g_last_produced_handle);
                std::cerr << "  [server]   handle produced: call " << i
                          << " -> VH(" << it->second << ") = real=0x"
                          << std::hex << g_last_produced_handle << std::dec << "\n";
            } else {
                g_vhandles.register_handle(i, g_last_produced_handle);
                std::cerr << "  [server]   handle produced: call " << i
                          << " -> real=0x" << std::hex << g_last_produced_handle
                          << std::dec << " (no manifest entry)\n";
            }
        }

        std::uint32_t resp_len = static_cast<std::uint32_t>(data.size());
        std::size_t prev_size = rpc_responses.size();
        rpc_responses.resize(prev_size + 4 + resp_len);
        std::memcpy(rpc_responses.data() + prev_size, &resp_len, 4);
        std::memcpy(rpc_responses.data() + prev_size + 4, data.data(), resp_len);
        rpc_resp_count++;
    }

    std::memcpy(rpc_responses.data(), &rpc_resp_count, 4);

    // ── 4. Process read requests ────────────────────────────────────
    if (off + 4 <= req_frame.payload.size()) {
        std::memcpy(&read_count, req_frame.payload.data() + off, 4); off += 4;
    } else {
        read_count = 0;
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

        std::uint32_t data_len = static_cast<std::uint32_t>(bytes_read);
        std::size_t prev_size = read_responses.size();
        read_responses.resize(prev_size + 4 + bytes_read);
        std::memcpy(read_responses.data() + prev_size, &data_len, 4);
        if (bytes_read > 0) {
            std::memcpy(read_responses.data() + prev_size + 4, mirror_data.data(), bytes_read);
        }
        read_resp_count++;
    }

    std::memcpy(read_responses.data(), &read_resp_count, 4);

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

    std::cout << "zlink CUDA server v4 — Full PyTorch API (48 functions)\n"
              << "Listening on port " << port << "...\n" << std::endl;

    auto tp = zlink::make_transport(zlink::transport_kind::tcp);
    auto ec = tp->listen("0.0.0.0", port);
    if (ec) {
        std::cerr << "Failed to listen: " << ec.message() << "\n";
        return 1;
    }

    // Outer loop: accept connections one at a time.
    // When a client disconnects, wait for the next one.
    // This allows the server to handle multiple sequential clients
    // (e.g., PyTorch creates multiple short-lived connections during init).
    while (true) {
        std::cout << "Waiting for client connection...\n";
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

                // Single RPC call — try both full and pytorch RPC types
                auto [data, in, out] = zpp::bits::data_in_out();
                data.assign(req_frame.payload.begin(), req_frame.payload.end());

                // Try the full 48-function RPC first
                cuda_v2_rpc_full::server server{in, out};
                auto result = server.serve();
                if (zpp::bits::failure(result)) {
                    // Fall back to pytorch 10-function RPC
                    pytorch_rpc::rpc::server pytorch_server{in, out};
                    result = pytorch_server.serve();
                    if (zpp::bits::failure(result)) {
                        std::cerr << "Serve error (both RPC types failed)\n";
                        break;
                    }
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

        std::cout << "Client disconnected. Waiting for next connection...\n";
        // Reset the listening socket so we can accept a new connection
        tp->close();
        ec = tp->listen("0.0.0.0", port);
        if (ec) {
            std::cerr << "Failed to re-listen: " << ec.message() << "\n";
            return 1;
        }
    }

    return 0;
}
