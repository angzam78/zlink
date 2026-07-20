#pragma once
// zlink/examples/cuda/cuda_api_v2.hpp — Expanded CUDA API for PyTorch workloads
//
// This is the unified API surface for GPU-over-IP with virtual handles.
// It covers all CUDA Driver API functions needed for PyTorch workloads
// like Stable Diffusion 1.5.
//
// DEPENDENCY CATEGORIZATION (see cuda_dep_spec.hpp for full details):
//   barrier   — Must have return value before next call (cuInit, cuDeviceGetAttribute)
//   enqueued  — Batch into pipeline (most functions)
//   readback  — Flush + get data back (cuMemcpyDtoH)
//
// VIRTUAL HANDLE ANNOTATION:
//   PRODUCES — This call creates a handle; use enqueue_produces_handle
//   consumes — This call takes handle params; server auto-translates VH
//   -        — No handle involvement

#include <zpp_bits.h>
#include <cstdint>
#include <string>

namespace cuda_api_v2 {

// ── Initialization & Device Management ─────────────────────────────────
// cuInit | barrier | - | First call, check CUDA availability
struct InitRet { int32_t result; };
InitRet cu_init(unsigned int flags);

// cuDeviceGetCount | barrier | - | Need count for device selection
struct DevCountRet { int32_t result; int32_t count; };
DevCountRet device_get_count();

// cuDeviceGetName | barrier | - | Need string back
struct DevNameRet { int32_t result; std::string name; };
DevNameRet device_get_name(int ordinal);

// cuDeviceTotalMem | barrier | - | Need value for alloc planning
struct DevMemRet { int32_t result; std::uint64_t bytes; };
DevMemRet device_total_mem(int ordinal);

// cuDeviceGetAttribute | barrier | - | Need value for occupancy calculations
struct DevAttrRet { int32_t result; int32_t value; };
DevAttrRet device_get_attribute(int attrib, int device_ordinal);

// ── Context Management ─────────────────────────────────────────────────
// cuCtxCreate | enqueued | PRODUCES (ctx_handle) | Returns VH for context
struct CtxCreateRet { int32_t result; std::uint64_t ctx_handle; };
CtxCreateRet ctx_create(unsigned int flags, int device_ordinal);

// cuCtxDestroy | enqueued | consumes (ctx) | Takes VH context
struct CtxDestroyRet { int32_t result; };
CtxDestroyRet ctx_destroy(std::uint64_t ctx_handle);

// cuCtxSetCurrent | enqueued | consumes (ctx) | Takes VH context
struct CtxSetCurrentRet { int32_t result; };
CtxSetCurrentRet ctx_set_current(std::uint64_t ctx_handle);

// cuCtxGetCurrent | barrier | PRODUCES (ctx_handle) | Need current ctx to use
struct CtxGetCurrentRet { int32_t result; std::uint64_t ctx_handle; };
CtxGetCurrentRet ctx_get_current();

// cuCtxSynchronize | enqueued | - | Just a sync point
struct CtxSyncRet { int32_t result; };
CtxSyncRet ctx_synchronize();

// ── Memory Management ──────────────────────────────────────────────────
// cuMemAlloc | enqueued | PRODUCES (dev_ptr) | ★ KEY: Returns VH for dev_ptr
struct AllocRet { int32_t result; std::uint64_t dev_ptr; };
AllocRet mem_alloc(std::uint64_t bytesize);

// cuMemAllocManaged | enqueued | PRODUCES (dev_ptr) | Returns VH for unified mem
struct AllocManagedRet { int32_t result; std::uint64_t dev_ptr; };
AllocManagedRet mem_alloc_managed(std::uint64_t bytesize, unsigned int flags);

// cuMemFree | enqueued | consumes (dev_ptr) | Takes VH dev_ptr
struct FreeRet { int32_t result; };
FreeRet mem_free(std::uint64_t dev_ptr);

// cuMemFreeHost | enqueued | consumes (host_ptr) | Takes VH host_ptr
struct FreeHostRet { int32_t result; };
FreeHostRet mem_free_host(std::uint64_t host_ptr);

// cuMemHostAlloc | enqueued | PRODUCES (host_ptr) | Returns VH for host_ptr
struct HostAllocRet { int32_t result; std::uint64_t host_ptr; };
HostAllocRet mem_host_alloc(std::uint64_t bytesize, unsigned int flags);

// cuMemHostRegister | enqueued | consumes (host_ptr) | Registers existing host memory
struct HostRegisterRet { int32_t result; };
HostRegisterRet mem_host_register(std::uint64_t host_ptr, std::uint64_t bytesize, unsigned int flags);

// cuMemHostUnregister | enqueued | consumes (host_ptr)
struct HostUnregisterRet { int32_t result; };
HostUnregisterRet mem_host_unregister(std::uint64_t host_ptr);

// cuMemGetInfo | barrier | - | Need free/total values for alloc strategy
struct MemInfoRet { int32_t result; std::uint64_t free_bytes; std::uint64_t total_bytes; };
MemInfoRet mem_get_info();

// ── Memory Copy ────────────────────────────────────────────────────────
// cuMemcpyHtoD | enqueued | consumes (dst) | + inline host_sync
struct CopyHtoDRet { int32_t result; };
CopyHtoDRet memcpy_htod(std::uint64_t dst_dev_ptr,
                         std::uint64_t src_client_addr,
                         std::uint64_t byte_count);

// cuMemcpyDtoH | readback | consumes (src) | + inline host_read
struct CopyDtoHRet { int32_t result; };
CopyDtoHRet memcpy_dtoh(std::uint64_t dst_client_addr,
                          std::uint64_t src_dev_ptr,
                          std::uint64_t byte_count);

// cuMemcpyDtoD | enqueued | consumes (dst, src) | Both VH dev_ptrs
struct CopyDtoDRet { int32_t result; };
CopyDtoDRet memcpy_dtod(std::uint64_t dst_dev_ptr,
                         std::uint64_t src_dev_ptr,
                         std::uint64_t byte_count);

// cuMemcpyHtoDAsync | enqueued | consumes (dst, stream) | + inline host_sync
struct CopyHtoDAsyncRet { int32_t result; };
CopyHtoDAsyncRet memcpy_htod_async(std::uint64_t dst_dev_ptr,
                                     std::uint64_t src_client_addr,
                                     std::uint64_t byte_count,
                                     std::uint64_t stream_handle);

// cuMemcpyDtoHAsync | readback | consumes (src, stream) | + inline host_read
struct CopyDtoHAsyncRet { int32_t result; };
CopyDtoHAsyncRet memcpy_dtoh_async(std::uint64_t dst_client_addr,
                                     std::uint64_t src_dev_ptr,
                                     std::uint64_t byte_count,
                                     std::uint64_t stream_handle);

// ── Module & Kernel Management ─────────────────────────────────────────
// cuModuleLoadData | enqueued | PRODUCES (module) | + inline host_sync for image
struct ModuleLoadRet { int32_t result; std::uint64_t module_handle; };
ModuleLoadRet module_load_data(std::uint64_t src_client_addr, std::uint64_t byte_count);

// cuModuleLoad | enqueued | PRODUCES (module) | Filename as string
struct ModuleLoadFileRet { int32_t result; std::uint64_t module_handle; };
ModuleLoadFileRet module_load(std::string filename);

// cuModuleUnload | enqueued | consumes (module)
struct ModuleUnloadRet { int32_t result; };
ModuleUnloadRet module_unload(std::uint64_t module_handle);

// cuModuleGetFunction | enqueued | PROD+CON (module→func) | Gets kernel from module
struct ModuleGetFuncRet { int32_t result; std::uint64_t func_handle; };
ModuleGetFuncRet module_get_function(std::uint64_t module_handle, std::string func_name);

// cuModuleGetGlobal | enqueued | PROD+CON (module→global) | Gets global variable
struct ModuleGetGlobalRet { int32_t result; std::uint64_t global_ptr; std::uint64_t global_size; };
ModuleGetGlobalRet module_get_global(std::uint64_t module_handle, std::string global_name);

// cuLaunchKernel | enqueued | consumes (func, stream, args) | ★ THE hot path
struct LaunchKernelRet { int32_t result; };
LaunchKernelRet launch_kernel(
    std::uint64_t func_handle,
    std::uint32_t grid_dim_x, std::uint32_t grid_dim_y, std::uint32_t grid_dim_z,
    std::uint32_t block_dim_x, std::uint32_t block_dim_y, std::uint32_t block_dim_z,
    std::uint32_t shared_mem_bytes,
    std::uint64_t stream_handle,
    std::uint64_t args_client_addr,
    std::uint64_t args_byte_count);

// ── Stream Management ──────────────────────────────────────────────────
// cuStreamCreate | enqueued | PRODUCES (stream)
struct StreamCreateRet { int32_t result; std::uint64_t stream_handle; };
StreamCreateRet stream_create(unsigned int flags);

// cuStreamDestroy | enqueued | consumes (stream)
struct StreamDestroyRet { int32_t result; };
StreamDestroyRet stream_destroy(std::uint64_t stream_handle);

// cuStreamSynchronize | enqueued | consumes (stream)
struct StreamSyncRet { int32_t result; };
StreamSyncRet stream_synchronize(std::uint64_t stream_handle);

// ── Event Management ───────────────────────────────────────────────────
// cuEventCreate | enqueued | PRODUCES (event)
struct EventCreateRet { int32_t result; std::uint64_t event_handle; };
EventCreateRet event_create(unsigned int flags);

// cuEventDestroy | enqueued | consumes (event)
struct EventDestroyRet { int32_t result; };
EventDestroyRet event_destroy(std::uint64_t event_handle);

// cuEventRecord | enqueued | consumes (event, stream)
struct EventRecordRet { int32_t result; };
EventRecordRet event_record(std::uint64_t event_handle, std::uint64_t stream_handle);

// cuEventSynchronize | enqueued | consumes (event)
struct EventSyncRet { int32_t result; };
EventSyncRet event_synchronize(std::uint64_t event_handle);

// cuEventElapsedTime | barrier | consumes (start, end) | Need float back
struct EventElapsedRet { int32_t result; float milliseconds; };
EventElapsedRet event_elapsed_time(std::uint64_t start_event, std::uint64_t end_event);

// ── Occupancy ──────────────────────────────────────────────────────────
// cuOccupancyMaxPotentialBlockSize | barrier | - | Need values for launch config
struct OccupancyRet { int32_t result; int32_t min_grid_size; int32_t block_size; };
OccupancyRet occupancy_max_potential_block_size(
    std::uint64_t func_handle,
    std::uint32_t shared_mem_bytes,
    int32_t block_size_limit);

} // namespace cuda_api_v2

// ── RPC binding (all functions, numbered 0-34) ────────────────────────
using cuda_v2_rpc = zpp::bits::rpc<
    // Initialization & Device (indices 0-4)
    zpp::bits::bind<&cuda_api_v2::cu_init,                  0>,
    zpp::bits::bind<&cuda_api_v2::device_get_count,         1>,
    zpp::bits::bind<&cuda_api_v2::device_get_name,          2>,
    zpp::bits::bind<&cuda_api_v2::device_total_mem,         3>,
    zpp::bits::bind<&cuda_api_v2::device_get_attribute,     4>,
    // Context (indices 5-9)
    zpp::bits::bind<&cuda_api_v2::ctx_create,               5>,
    zpp::bits::bind<&cuda_api_v2::ctx_destroy,              6>,
    zpp::bits::bind<&cuda_api_v2::ctx_set_current,          7>,
    zpp::bits::bind<&cuda_api_v2::ctx_get_current,          8>,
    zpp::bits::bind<&cuda_api_v2::ctx_synchronize,          9>,
    // Memory (indices 10-17)
    zpp::bits::bind<&cuda_api_v2::mem_alloc,                10>,
    zpp::bits::bind<&cuda_api_v2::mem_alloc_managed,        11>,
    zpp::bits::bind<&cuda_api_v2::mem_free,                 12>,
    zpp::bits::bind<&cuda_api_v2::mem_free_host,            13>,
    zpp::bits::bind<&cuda_api_v2::mem_host_alloc,           14>,
    zpp::bits::bind<&cuda_api_v2::mem_host_register,        15>,
    zpp::bits::bind<&cuda_api_v2::mem_host_unregister,      16>,
    zpp::bits::bind<&cuda_api_v2::mem_get_info,             17>,
    // Copy (indices 18-22)
    zpp::bits::bind<&cuda_api_v2::memcpy_htod,              18>,
    zpp::bits::bind<&cuda_api_v2::memcpy_dtoh,              19>,
    zpp::bits::bind<&cuda_api_v2::memcpy_dtod,              20>,
    zpp::bits::bind<&cuda_api_v2::memcpy_htod_async,        21>,
    zpp::bits::bind<&cuda_api_v2::memcpy_dtoh_async,        22>,
    // Module & Kernel (indices 23-28)
    zpp::bits::bind<&cuda_api_v2::module_load_data,         23>,
    zpp::bits::bind<&cuda_api_v2::module_load,              24>,
    zpp::bits::bind<&cuda_api_v2::module_unload,            25>,
    zpp::bits::bind<&cuda_api_v2::module_get_function,      26>,
    zpp::bits::bind<&cuda_api_v2::module_get_global,        27>,
    zpp::bits::bind<&cuda_api_v2::launch_kernel,            28>,
    // Stream (indices 29-31)
    zpp::bits::bind<&cuda_api_v2::stream_create,            29>,
    zpp::bits::bind<&cuda_api_v2::stream_destroy,           30>,
    zpp::bits::bind<&cuda_api_v2::stream_synchronize,       31>,
    // Event (indices 32-36)
    zpp::bits::bind<&cuda_api_v2::event_create,             32>,
    zpp::bits::bind<&cuda_api_v2::event_destroy,            33>,
    zpp::bits::bind<&cuda_api_v2::event_record,             34>,
    zpp::bits::bind<&cuda_api_v2::event_synchronize,        35>,
    zpp::bits::bind<&cuda_api_v2::event_elapsed_time,       36>,
    // Occupancy (index 37)
    zpp::bits::bind<&cuda_api_v2::occupancy_max_potential_block_size, 37>
>;

// ── Named indices ─────────────────────────────────────────────────────
namespace v2_func {
    constexpr std::size_t cu_init                    = 0;
    constexpr std::size_t device_get_count           = 1;
    constexpr std::size_t device_get_name            = 2;
    constexpr std::size_t device_total_mem           = 3;
    constexpr std::size_t device_get_attribute       = 4;
    constexpr std::size_t ctx_create                 = 5;
    constexpr std::size_t ctx_destroy                = 6;
    constexpr std::size_t ctx_set_current            = 7;
    constexpr std::size_t ctx_get_current            = 8;
    constexpr std::size_t ctx_synchronize            = 9;
    constexpr std::size_t mem_alloc                  = 10;
    constexpr std::size_t mem_alloc_managed          = 11;
    constexpr std::size_t mem_free                   = 12;
    constexpr std::size_t mem_free_host              = 13;
    constexpr std::size_t mem_host_alloc             = 14;
    constexpr std::size_t mem_host_register          = 15;
    constexpr std::size_t mem_host_unregister        = 16;
    constexpr std::size_t mem_get_info               = 17;
    constexpr std::size_t memcpy_htod                = 18;
    constexpr std::size_t memcpy_dtoh                = 19;
    constexpr std::size_t memcpy_dtod                = 20;
    constexpr std::size_t memcpy_htod_async          = 21;
    constexpr std::size_t memcpy_dtoh_async          = 22;
    constexpr std::size_t module_load_data           = 23;
    constexpr std::size_t module_load                = 24;
    constexpr std::size_t module_unload              = 25;
    constexpr std::size_t module_get_function        = 26;
    constexpr std::size_t module_get_global          = 27;
    constexpr std::size_t launch_kernel              = 28;
    constexpr std::size_t stream_create              = 29;
    constexpr std::size_t stream_destroy             = 30;
    constexpr std::size_t stream_synchronize         = 31;
    constexpr std::size_t event_create               = 32;
    constexpr std::size_t event_destroy              = 33;
    constexpr std::size_t event_record               = 34;
    constexpr std::size_t event_synchronize          = 35;
    constexpr std::size_t event_elapsed_time         = 36;
    constexpr std::size_t occupancy_max_potential    = 37;
}
