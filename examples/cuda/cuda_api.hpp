// zlink/examples/cuda/cuda_api.hpp — Hand-written CUDA Driver API RPC declarations
//
// A clean, minimal API surface for the CUDA-over-IP pipeline test. Each function
// is categorized (barrier / enqueued / readback) and annotated for handle
// production/consumption. The client and server both include this header so the
// zpp_bits::rpc<> binding is shared.
//
// Design: opaque CUDA handles travel as uint64_t. Device pointers (CUdeviceptr)
// also travel as uint64_t and use virtual handles (bit 63 set) on the wire.
// Host pointers that must be dereferenced server-side travel as uint64_t address
// values and rely on the host_memory_mirror for the actual bytes.
//
// Adding a new function requires exactly 3 things (see cuda_dep_spec.hpp):
//   1. Declare the return struct + function signature here
//   2. Add a zpp::bits::bind entry with the next index
//   3. Categorize it: barrier / enqueued(+produces/consumes) / readback

#pragma once

#include <zpp_bits.h>
#include <cstdint>
#include <string>

namespace cuda_gen {

// ── Return structs ─────────────────────────────────────────────────────
// Convention: field 0 is always the CUresult `result`. Handle fields follow.
// The virtual-handle manifest uses `return_field = 1` to mean "the handle is
// the field after result" (the standard convention).

struct InitRet { int32_t result; };
struct DriverGetVersionRet { int32_t result; int32_t driverVersion; };
struct DeviceGetCountRet { int32_t result; int32_t count; };
struct DeviceGetRet { int32_t result; int32_t device; };
struct DeviceGetNameRet { int32_t result; std::string name; };
struct DeviceTotalMemRet { int32_t result; uint64_t bytes; };
struct DeviceGetAttributeRet { int32_t result; int32_t value; };
struct CtxCreateRet { int32_t result; uint64_t ctx_handle; };
struct CtxDestroyRet { int32_t result; };
struct CtxSetCurrentRet { int32_t result; };
struct CtxGetCurrentRet { int32_t result; uint64_t ctx_handle; };
struct CtxSynchronizeRet { int32_t result; };
struct MemAllocRet { int32_t result; uint64_t dptr; };
struct MemAllocManagedRet { int32_t result; uint64_t dptr; };
struct MemFreeRet { int32_t result; };
struct MemGetInfoRet { int32_t result; uint64_t free_bytes; uint64_t total_bytes; };
struct MemcpyHtoDRet { int32_t result; };
struct MemcpyDtoHRet { int32_t result; };
struct MemcpyDtoDRet { int32_t result; };
struct StreamCreateRet { int32_t result; uint64_t stream_handle; };
struct StreamDestroyRet { int32_t result; };
struct StreamSynchronizeRet { int32_t result; };
struct EventCreateRet { int32_t result; uint64_t event_handle; };
struct EventDestroyRet { int32_t result; };
struct EventRecordRet { int32_t result; };
struct EventSynchronizeRet { int32_t result; };
struct EventElapsedTimeRet { int32_t result; float milliseconds; };
struct ModuleLoadDataRet { int32_t result; uint64_t module_handle; };
struct ModuleGetFunctionRet { int32_t result; uint64_t func_handle; };
struct LaunchKernelRet { int32_t result; };

// ── Function declarations (shared by client + server) ──────────────────
// Server implements these with real CUDA calls; client's zpp_bits client
// only needs the declarations (not definitions) to serialize requests.

// barrier — cuInit
InitRet init(uint32_t flags);
// barrier — cuDriverGetVersion
DriverGetVersionRet driver_get_version();
// barrier — cuDeviceGetCount
DeviceGetCountRet device_get_count();
// barrier — cuDeviceGet
DeviceGetRet device_get(int32_t ordinal);
// barrier — cuDeviceGetName
DeviceGetNameRet device_get_name(int32_t len, int32_t dev);
// barrier — cuDeviceTotalMem
DeviceTotalMemRet device_total_mem(int32_t dev);
// barrier — cuDeviceGetAttribute
DeviceGetAttributeRet device_get_attribute(uint32_t attrib, int32_t dev);
// enqueued, PRODUCES — cuCtxCreate
CtxCreateRet ctx_create(uint32_t flags, int32_t dev);
// enqueued, consumes — cuCtxDestroy
CtxDestroyRet ctx_destroy(uint64_t ctx_handle);
// enqueued, consumes — cuCtxSetCurrent
CtxSetCurrentRet ctx_set_current(uint64_t ctx_handle);
// barrier, PRODUCES — cuCtxGetCurrent
CtxGetCurrentRet ctx_get_current();
// enqueued — cuCtxSynchronize
CtxSynchronizeRet ctx_synchronize();
// enqueued, PRODUCES — cuMemAlloc
MemAllocRet mem_alloc(uint64_t bytesize);
// enqueued, PRODUCES — cuMemAllocManaged
MemAllocManagedRet mem_alloc_managed(uint64_t bytesize, uint32_t flags);
// enqueued, consumes — cuMemFree
MemFreeRet mem_free(uint64_t dptr);
// barrier — cuMemGetInfo
MemGetInfoRet mem_get_info();
// enqueued, consumes + host_sync — cuMemcpyHtoD
MemcpyHtoDRet memcpy_hto_d(uint64_t dst_device, uint64_t src_host_addr, uint64_t byte_count);
// readback, consumes + host_read — cuMemcpyDtoH
MemcpyDtoHRet memcpy_dto_h(uint64_t dst_host_addr, uint64_t src_device, uint64_t byte_count);
// enqueued, consumes — cuMemcpyDtoD
MemcpyDtoDRet memcpy_dto_d(uint64_t dst_device, uint64_t src_device, uint64_t byte_count);
// enqueued, PRODUCES — cuStreamCreate
StreamCreateRet stream_create(uint32_t flags);
// enqueued, consumes — cuStreamDestroy
StreamDestroyRet stream_destroy(uint64_t stream_handle);
// enqueued, consumes — cuStreamSynchronize
StreamSynchronizeRet stream_synchronize(uint64_t stream_handle);
// enqueued, PRODUCES — cuEventCreate
EventCreateRet event_create(uint32_t flags);
// enqueued, consumes — cuEventDestroy
EventDestroyRet event_destroy(uint64_t event_handle);
// enqueued, consumes — cuEventRecord
EventRecordRet event_record(uint64_t event_handle, uint64_t stream_handle);
// enqueued, consumes — cuEventSynchronize
EventSynchronizeRet event_synchronize(uint64_t event_handle);
// barrier, consumes — cuEventElapsedTime
EventElapsedTimeRet event_elapsed_time(uint64_t start_handle, uint64_t end_handle);
// enqueued, PRODUCES + host_sync — cuModuleLoadData
ModuleLoadDataRet module_load_data(uint64_t image_addr, uint64_t byte_count);
// enqueued, PROD+CON — cuModuleGetFunction
ModuleGetFunctionRet module_get_function(uint64_t module_handle, std::string name);
// enqueued, consumes + host_sync — cuLaunchKernel
LaunchKernelRet launch_kernel(
    uint64_t func_handle,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    uint32_t shared_mem,
    uint64_t stream_handle,
    uint64_t args_addr, uint64_t args_byte_count);

// ── Stable function indices for the RPC binding ───────────────────────
namespace func_index {
inline constexpr int init                  = 0;
inline constexpr int driver_get_version   = 1;
inline constexpr int device_get_count      = 2;
inline constexpr int device_get            = 3;
inline constexpr int device_get_name        = 4;
inline constexpr int device_total_mem       = 5;
inline constexpr int device_get_attribute   = 6;
inline constexpr int ctx_create             = 7;
inline constexpr int ctx_destroy            = 8;
inline constexpr int ctx_set_current        = 9;
inline constexpr int ctx_get_current        = 10;
inline constexpr int ctx_synchronize        = 11;
inline constexpr int mem_alloc              = 12;
inline constexpr int mem_alloc_managed      = 13;
inline constexpr int mem_free               = 14;
inline constexpr int mem_get_info           = 15;
inline constexpr int memcpy_hto_d           = 16;
inline constexpr int memcpy_dto_h           = 17;
inline constexpr int memcpy_dto_d           = 18;
inline constexpr int stream_create          = 19;
inline constexpr int stream_destroy         = 20;
inline constexpr int stream_synchronize     = 21;
inline constexpr int event_create           = 22;
inline constexpr int event_destroy          = 23;
inline constexpr int event_record           = 24;
inline constexpr int event_synchronize      = 25;
inline constexpr int event_elapsed_time     = 26;
inline constexpr int module_load_data       = 27;
inline constexpr int module_get_function    = 28;
inline constexpr int launch_kernel          = 29;
} // namespace func_index

// ── The RPC type ────────────────────────────────────────────────────────
// The bind indices match func_index::* above.
using cuda_gen_rpc = zpp::bits::rpc<
    zpp::bits::bind<&init,                  func_index::init>,
    zpp::bits::bind<&driver_get_version,    func_index::driver_get_version>,
    zpp::bits::bind<&device_get_count,      func_index::device_get_count>,
    zpp::bits::bind<&device_get,            func_index::device_get>,
    zpp::bits::bind<&device_get_name,       func_index::device_get_name>,
    zpp::bits::bind<&device_total_mem,      func_index::device_total_mem>,
    zpp::bits::bind<&device_get_attribute,  func_index::device_get_attribute>,
    zpp::bits::bind<&ctx_create,            func_index::ctx_create>,
    zpp::bits::bind<&ctx_destroy,           func_index::ctx_destroy>,
    zpp::bits::bind<&ctx_set_current,       func_index::ctx_set_current>,
    zpp::bits::bind<&ctx_get_current,       func_index::ctx_get_current>,
    zpp::bits::bind<&ctx_synchronize,       func_index::ctx_synchronize>,
    zpp::bits::bind<&mem_alloc,             func_index::mem_alloc>,
    zpp::bits::bind<&mem_alloc_managed,     func_index::mem_alloc_managed>,
    zpp::bits::bind<&mem_free,              func_index::mem_free>,
    zpp::bits::bind<&mem_get_info,          func_index::mem_get_info>,
    zpp::bits::bind<&memcpy_hto_d,          func_index::memcpy_hto_d>,
    zpp::bits::bind<&memcpy_dto_h,          func_index::memcpy_dto_h>,
    zpp::bits::bind<&memcpy_dto_d,          func_index::memcpy_dto_d>,
    zpp::bits::bind<&stream_create,         func_index::stream_create>,
    zpp::bits::bind<&stream_destroy,        func_index::stream_destroy>,
    zpp::bits::bind<&stream_synchronize,    func_index::stream_synchronize>,
    zpp::bits::bind<&event_create,          func_index::event_create>,
    zpp::bits::bind<&event_destroy,         func_index::event_destroy>,
    zpp::bits::bind<&event_record,          func_index::event_record>,
    zpp::bits::bind<&event_synchronize,     func_index::event_synchronize>,
    zpp::bits::bind<&event_elapsed_time,    func_index::event_elapsed_time>,
    zpp::bits::bind<&module_load_data,      func_index::module_load_data>,
    zpp::bits::bind<&module_get_function,   func_index::module_get_function>,
    zpp::bits::bind<&launch_kernel,         func_index::launch_kernel>
>;

} // namespace cuda_gen
