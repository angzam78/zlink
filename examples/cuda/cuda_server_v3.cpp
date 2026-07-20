// zlink/examples/cuda/cuda_server_v3.cpp
//
// CUDA RPC server v3 — Full PyTorch-relevant API with virtual handles.
//
// Handles ALL 38 CUDA Driver API functions from cuda_api_v2.hpp.
// Every handle-type parameter goes through g_vhandles.translate().
// Every handle-producing function uses handle_producer_guard.

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

// ── Initialization & Device Management ─────────────────────────────────

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

// ── Context Management ─────────────────────────────────────────────────

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

// ── Memory Management ──────────────────────────────────────────────────

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

// ── Memory Copy ────────────────────────────────────────────────────────

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

// ── Module & Kernel Management ─────────────────────────────────────────

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

// ── Stream Management ──────────────────────────────────────────────────

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

// ── Event Management ───────────────────────────────────────────────────

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

// ── Occupancy ──────────────────────────────────────────────────────────

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

} // namespace cuda_api_v2

// ── Handle producer info (maps func index → handle field index) ────────
struct handle_producer_info {
    int func_index;
    int handle_field;
};

static const handle_producer_info handle_producers[] = {
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
// NOTE: Unlike pipeline_mem, pipeline_request doesn't carry a handle
// manifest. But we still register produced handles using the
// g_has_produced_handle mechanism, keyed by call index.
// The client must use pipeline_mem (with manifest) for correct VH
// translation WITHIN a batch. pipeline_request is used for calls
// that DON'T reference each other's VHs within the batch.
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

        // Reset handle production flag before serving
        g_has_produced_handle = false;

        auto [data, in, out] = zpp::bits::data_in_out();
        data.assign(req_frame.payload.data() + offset,
                    req_frame.payload.data() + offset + req_len);
        offset += req_len;

        cuda_v2_rpc::server server{in, out};
        auto result = server.serve();
        if (zpp::bits::failure(result)) {
            std::cerr << "  [server] pipeline call " << i << " serve error\n";
            break;
        }

        // Register produced handles under call index
        // (For pipeline_request, VHs are only used by subsequent flush calls,
        //  not within the same batch — so call-index registration is fine)
        if (g_has_produced_handle) {
            // Use a stable ID: call_index + base offset to avoid collisions
            // with previous batches. We use a monotonically increasing counter.
            static std::uint32_t vh_base = 1000; // Start above pipeline_mem VHs
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
// Wire format (request):
//   [4B sync_count]
//   for each sync: [8B addr][8B size][data...]
//   [4B rpc_count]
//   for each rpc: [4B len][rpc_bytes]
//   [4B read_count]
//   for each read: [8B addr][8B size]
//   [handle_manifest: 4B count + N × handle_manifest_entry]
//
// CRITICAL: The handle manifest is parsed BEFORE RPC processing so
// we know which VH IDs to assign to each handle-producing call.
// As each call produces a handle, we immediately register it under
// the correct VH ID from the manifest, enabling subsequent calls
// in the SAME batch to reference it via g_vhandles.translate().
static bool handle_pipeline_mem(zlink::transport& tp, const zlink::frame& req_frame) {
    if (req_frame.payload.size() < 4) return false;

    // ── 0. Pre-parse handle manifest from the END of the frame ───────
    // The manifest is at the end of the payload. We need to find it
    // before processing RPCs so we know which call_index → virtual_id
    // mappings exist. We'll scan backwards: find the manifest by
    // parsing forward through syncs/RPCs/reads to skip them, then
    // parse the manifest.
    //
    // Actually, we can just parse the entire frame linearly, collecting
    // the manifest last. But we need the manifest DURING RPC processing.
    // Solution: two-pass — first pass finds manifest offset, second
    // pass processes everything. Or: pre-scan to find manifest.
    //
    // Simplest approach: parse the manifest FIRST by scanning through
    // syncs/RPCs/reads just to find offsets, then process normally.

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

    // ── 2. Pre-parse manifest to build call_index → virtual_id map ───
    // We need to skip RPCs and reads to find the manifest at the end.
    // Save the RPC start offset for later processing.
    std::size_t rpc_section_off = off;

    std::uint32_t rpc_count = 0;
    std::memcpy(&rpc_count, req_frame.payload.data() + off, 4); off += 4;

    // Skip over RPC entries to find the reads section
    for (std::uint32_t i = 0; i < rpc_count && off + 4 <= req_frame.payload.size(); i++) {
        std::uint32_t req_len = 0;
        std::memcpy(&req_len, req_frame.payload.data() + off, 4); off += 4;
        off += req_len; // skip RPC data
    }

    // Skip over read entries
    std::uint32_t read_count = 0;
    if (off + 4 <= req_frame.payload.size()) {
        std::memcpy(&read_count, req_frame.payload.data() + off, 4); off += 4;
    }
    for (std::uint32_t i = 0; i < read_count; i++) {
        off += 16; // skip addr(8) + size(8)
    }

    // Now off points to the manifest
    std::vector<zlink::handle_manifest_entry> manifest;
    if (off + 4 <= req_frame.payload.size()) {
        std::size_t manifest_bytes = 0;
        manifest = zlink::parse_handle_manifest(
            std::span<const std::byte>(req_frame.payload.data() + off,
                                       req_frame.payload.size() - off),
            manifest_bytes);
    }

    // Build call_index → virtual_id lookup
    std::unordered_map<std::uint32_t, std::uint32_t> call_to_vh;
    for (const auto& entry : manifest) {
        call_to_vh[entry.call_index] = entry.virtual_id;
    }

    if (!manifest.empty()) {
        std::cerr << "  [server] pipeline_mem: pre-parsed " << manifest.size() << " manifest entries\n";
    }

    // ── 3. Process RPC calls (with immediate VH registration) ────────
    off = rpc_section_off + 4; // skip the rpc_count we already read

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

        cuda_v2_rpc::server server{in, out};
        auto result = server.serve();
        if (zpp::bits::failure(result)) {
            std::cerr << "  [server] pipeline_mem RPC " << i << " serve error\n";
            break;
        }

        // If this call produced a handle, register it IMMEDIATELY
        // under the virtual ID from the manifest (not call index).
        // This allows subsequent calls in the SAME batch to translate VHs.
        if (g_has_produced_handle) {
            auto it = call_to_vh.find(i);
            if (it != call_to_vh.end()) {
                g_vhandles.register_handle(it->second, g_last_produced_handle);
                std::cerr << "  [server]   handle produced: call " << i
                          << " -> VH(" << it->second << ") = real=0x"
                          << std::hex << g_last_produced_handle << std::dec << "\n";
            } else {
                // No manifest entry — register under call index as fallback
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

    std::cout << "zlink CUDA server v3 — Full API + virtual handles\n"
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

            cuda_v2_rpc::server server{in, out};
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

    std::cout << "CUDA server v3 shutting down.\n";
    return 0;
}
