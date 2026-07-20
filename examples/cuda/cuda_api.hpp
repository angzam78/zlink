#pragma once
// zlink/examples/cuda/cuda_api.hpp — CUDA API declarations for GPU-over-IP
//
// This file demonstrates how to declare a CUDA API surface for
// remote execution via zlink. Unlike Lupine, which requires a Python
// codegen script to parse CUDA headers, zlink uses a simple, explicit
// declaration list that you write once.
//
// Benefits over Lupine's approach:
//   1. No codegen step — just write the function signatures
//   2. Works with any CUDA version — just add/remove functions
//   3. Compile-time type safety — wrong signatures won't compile
//   4. Zero overhead — zpp_bits generates optimal serialization
//
// To add a new CUDA function:
//   ZLINK_CUDA_FUNC(return_type, name, param_types...)
//
// Pointer translation is handled automatically by the shim layer:
//   - CUdeviceptr values are mapped via ptr_map
//   - Host pointers in copy operations are sent as data
//   - Opaque handles (CUcontext, CUstream, etc.) are mapped as IDs

#include <zpp_bits.h>
#include <cstdint>

// ── CUDA type stubs ────────────────────────────────────────────────────
// These define the CUDA handle types as integers so we can serialize them.
// On the server, these map to real CUDA handles via ptr_map.

using CUdeviceptr  = std::uint64_t;
using CUcontext    = std::uint64_t;
using CUmodule     = std::uint64_t;
using CUfunction   = std::uint64_t;
using CUstream     = std::uint64_t;
using CUevent      = std::uint64_t;
using CUresult     = std::int32_t;
using CUdevice     = std::int32_t;

// ── CUDA API namespace ─────────────────────────────────────────────────
// Free function declarations matching CUDA driver API signatures.
// These are the "contract" between client and server.
// The server implements them by calling the real cuXxx functions.

namespace cuda_api {

// ── Initialization & Device Management ─────────────────────────────────
CUresult cuInit(unsigned int Flags);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuDeviceTotalMem(std::size_t* bytes, CUdevice dev);
CUresult cuDeviceGetAttribute(int* value, int attrib, CUdevice dev);

// ── Context Management ─────────────────────────────────────────────────
CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext* pctx);
CUresult cuCtxSynchronize();

// ── Memory Management ──────────────────────────────────────────────────
// These are the most performance-critical functions for GPU-over-IP.
// The shim layer translates device pointers and handles bulk data transfer.
CUresult cuMemAlloc(CUdeviceptr* dptr, std::size_t bytesize);
CUresult cuMemAllocManaged(CUdeviceptr* dptr, std::size_t bytesize, unsigned int flags);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemFreeHost(void* p);
CUresult cuMemHostAlloc(void** pp, std::size_t bytesize, unsigned int Flags);
CUresult cuMemHostRegister(void* p, std::size_t bytesize, unsigned int Flags);
CUresult cuMemHostUnregister(void* p);

// ── Memory Copy ────────────────────────────────────────────────────────
// These are where r3map-style remote memory makes a huge difference.
// For large transfers, we pipeline the data over the transport.
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, std::size_t ByteCount);
CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, std::size_t ByteCount);
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, std::size_t ByteCount);
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost,
                            std::size_t ByteCount, CUstream hStream);
CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice,
                            std::size_t ByteCount, CUstream hStream);

// ── Module & Kernel Management ─────────────────────────────────────────
CUresult cuModuleLoad(CUmodule* module, const char* fname);
CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);

// ── Kernel Launch ──────────────────────────────────────────────────────
// cuLaunchKernel has variable-length parameters, so we use opaque mode.
// The shim serializes the kernel params as a flat byte buffer.
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra);

// ── Stream & Event Management ──────────────────────────────────────────
CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags);
CUresult cuStreamDestroy(CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags);
CUresult cuEventDestroy(CUevent hEvent);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuEventSynchronize(CUevent hEvent);
CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd);

} // namespace cuda_api

// ── CUDA RPC type ──────────────────────────────────────────────────────
// This creates the compile-time RPC protocol. The function index
// corresponds to the order listed here.
using cuda_rpc = zpp::bits::rpc<
    // Initialization & Device
    zpp::bits::bind<&cuda_api::cuInit, 0>,
    zpp::bits::bind<&cuda_api::cuDeviceGet, 1>,
    zpp::bits::bind<&cuda_api::cuDeviceGetCount, 2>,
    zpp::bits::bind<&cuda_api::cuDeviceGetName, 3>,
    zpp::bits::bind<&cuda_api::cuDeviceTotalMem, 4>,
    zpp::bits::bind<&cuda_api::cuDeviceGetAttribute, 5>,
    // Context
    zpp::bits::bind<&cuda_api::cuCtxCreate, 6>,
    zpp::bits::bind<&cuda_api::cuCtxDestroy, 7>,
    zpp::bits::bind<&cuda_api::cuCtxSetCurrent, 8>,
    zpp::bits::bind<&cuda_api::cuCtxGetCurrent, 9>,
    zpp::bits::bind<&cuda_api::cuCtxSynchronize, 10>,
    // Memory
    zpp::bits::bind<&cuda_api::cuMemAlloc, 11>,
    zpp::bits::bind<&cuda_api::cuMemAllocManaged, 12>,
    zpp::bits::bind<&cuda_api::cuMemFree, 13>,
    zpp::bits::bind<&cuda_api::cuMemFreeHost, 14>,
    zpp::bits::bind<&cuda_api::cuMemHostAlloc, 15>,
    zpp::bits::bind<&cuda_api::cuMemHostRegister, 16>,
    zpp::bits::bind<&cuda_api::cuMemHostUnregister, 17>,
    // Copy
    zpp::bits::bind<&cuda_api::cuMemcpyHtoD, 18>,
    zpp::bits::bind<&cuda_api::cuMemcpyDtoH, 19>,
    zpp::bits::bind<&cuda_api::cuMemcpyDtoD, 20>,
    zpp::bits::bind<&cuda_api::cuMemcpyHtoDAsync, 21>,
    zpp::bits::bind<&cuda_api::cuMemcpyDtoHAsync, 22>,
    // Module & Kernel
    zpp::bits::bind<&cuda_api::cuModuleLoad, 23>,
    zpp::bits::bind<&cuda_api::cuModuleLoadData, 24>,
    zpp::bits::bind<&cuda_api::cuModuleUnload, 25>,
    zpp::bits::bind<&cuda_api::cuModuleGetFunction, 26>,
    zpp::bits::bind<&cuda_api::cuLaunchKernel, 27>,
    // Stream & Event
    zpp::bits::bind<&cuda_api::cuStreamCreate, 28>,
    zpp::bits::bind<&cuda_api::cuStreamDestroy, 29>,
    zpp::bits::bind<&cuda_api::cuStreamSynchronize, 30>,
    zpp::bits::bind<&cuda_api::cuEventCreate, 31>,
    zpp::bits::bind<&cuda_api::cuEventDestroy, 32>,
    zpp::bits::bind<&cuda_api::cuEventRecord, 33>,
    zpp::bits::bind<&cuda_api::cuEventSynchronize, 34>,
    zpp::bits::bind<&cuda_api::cuEventElapsedTime, 35>
>;

// ── Convenience indices ────────────────────────────────────────────────
// Named constants for function indices, so you don't have to count.
namespace cuda_func {
    constexpr std::size_t cuInit              = 0;
    constexpr std::size_t cuDeviceGet         = 1;
    constexpr std::size_t cuDeviceGetCount    = 2;
    constexpr std::size_t cuDeviceGetName     = 3;
    constexpr std::size_t cuDeviceTotalMem    = 4;
    constexpr std::size_t cuDeviceGetAttribute= 5;
    constexpr std::size_t cuCtxCreate         = 6;
    constexpr std::size_t cuCtxDestroy        = 7;
    constexpr std::size_t cuCtxSetCurrent     = 8;
    constexpr std::size_t cuCtxGetCurrent     = 9;
    constexpr std::size_t cuCtxSynchronize    = 10;
    constexpr std::size_t cuMemAlloc          = 11;
    constexpr std::size_t cuMemAllocManaged   = 12;
    constexpr std::size_t cuMemFree           = 13;
    constexpr std::size_t cuMemFreeHost       = 14;
    constexpr std::size_t cuMemHostAlloc      = 15;
    constexpr std::size_t cuMemHostRegister   = 16;
    constexpr std::size_t cuMemHostUnregister = 17;
    constexpr std::size_t cuMemcpyHtoD        = 18;
    constexpr std::size_t cuMemcpyDtoH        = 19;
    constexpr std::size_t cuMemcpyDtoD        = 20;
    constexpr std::size_t cuMemcpyHtoDAsync   = 21;
    constexpr std::size_t cuMemcpyDtoHAsync   = 22;
    constexpr std::size_t cuModuleLoad        = 23;
    constexpr std::size_t cuModuleLoadData    = 24;
    constexpr std::size_t cuModuleUnload      = 25;
    constexpr std::size_t cuModuleGetFunction = 26;
    constexpr std::size_t cuLaunchKernel      = 27;
    constexpr std::size_t cuStreamCreate      = 28;
    constexpr std::size_t cuStreamDestroy     = 29;
    constexpr std::size_t cuStreamSynchronize = 30;
    constexpr std::size_t cuEventCreate       = 31;
    constexpr std::size_t cuEventDestroy      = 32;
    constexpr std::size_t cuEventRecord       = 33;
    constexpr std::size_t cuEventSynchronize  = 34;
    constexpr std::size_t cuEventElapsedTime  = 35;
}
