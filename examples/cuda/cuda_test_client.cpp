// zlink/examples/cuda/cuda_test_client.cpp
//
// CUDA RPC test client — virtual handles + dependency-aware pipeline.
//
// VIRTUAL HANDLES: The key innovation that makes complex workloads pipelineable.
//
//   cuMemAlloc() no longer needs to be a barrier! It enqueues and returns
//   a virtual handle (VH). Subsequent calls reference VH values.
//   The server translates VH → real CUDA handles when processing the batch.
//
//   Without VH:  cuMemAlloc → BARRIER (wait for dev_ptr) → cuMemcpyHtoD → ...
//   With VH:     cuMemAlloc → ENQUEUED (returns VH(0)) → cuMemcpyHtoD(VH(0)) → ...
//
//   Result: the entire workload pipelines into ONE batch:
//   alloc(VH0) + HtoD(VH0) + kernel(VH0) + sync + DtoH(VH0) → 1 round-trip

#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/ptr_map.hpp>
#include <zlink/memory.hpp>
#include <zlink/config.hpp>
#include <zlink/rpc.hpp>
#include <zlink/cuda_pipeline.hpp>
#include <zlink/virtual_handle.hpp>

#include <zpp_bits.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdint>

// ── RPC binding (same as server) ──────────────────────────────────────
namespace cuda_rpc_api {

struct InitRet { int32_t result; };
InitRet cuda_init(unsigned int flags);

struct DevCountRet { int32_t result; int32_t count; };
DevCountRet get_device_count();

struct DevNameRet { int32_t result; std::string name; };
DevNameRet get_device_name(int ordinal);

struct DevMemRet { int32_t result; std::uint64_t bytes; };
DevMemRet get_device_total_mem(int ordinal);

struct CtxCreateRet { int32_t result; std::uint64_t ctx_handle; };
CtxCreateRet ctx_create(unsigned int flags, int device_ordinal);

struct CtxSyncRet { int32_t result; };
CtxSyncRet ctx_synchronize();

struct CtxDestroyRet { int32_t result; };
CtxDestroyRet ctx_destroy(std::uint64_t ctx_handle);

struct AllocRet { int32_t result; std::uint64_t dev_ptr; };
AllocRet mem_alloc(std::uint64_t bytesize);

struct FreeRet { int32_t result; };
FreeRet mem_free(std::uint64_t dev_ptr);

struct CopyHtoDRet { int32_t result; };
CopyHtoDRet memcpy_htod(std::uint64_t dst_dev_ptr,
                         std::uint64_t src_client_addr,
                         std::uint64_t byte_count);

struct CopyDtoHRet { int32_t result; };
CopyDtoHRet memcpy_dtoh(std::uint64_t dst_client_addr,
                          std::uint64_t src_dev_ptr,
                          std::uint64_t byte_count);

struct ModuleLoadRet { int32_t result; std::uint64_t module_handle; };
ModuleLoadRet module_load_data(std::uint64_t src_client_addr, std::uint64_t byte_count);

struct ModuleGetFuncRet { int32_t result; std::uint64_t func_handle; };
ModuleGetFuncRet module_get_function(std::uint64_t module_handle, std::string func_name);

struct LaunchKernelRet { int32_t result; };
LaunchKernelRet launch_kernel(
    std::uint64_t func_handle,
    std::uint32_t grid_dim_x, std::uint32_t grid_dim_y, std::uint32_t grid_dim_z,
    std::uint32_t block_dim_x, std::uint32_t block_dim_y, std::uint32_t block_dim_z,
    std::uint32_t shared_mem_bytes,
    std::uint64_t stream_handle,
    std::uint64_t args_client_addr,
    std::uint64_t args_byte_count);

struct StreamCreateRet { int32_t result; std::uint64_t stream_handle; };
StreamCreateRet stream_create(unsigned int flags);

struct StreamSyncRet { int32_t result; };
StreamSyncRet stream_synchronize(std::uint64_t stream_handle);

struct StreamDestroyRet { int32_t result; };
StreamDestroyRet stream_destroy(std::uint64_t stream_handle);

struct EventCreateRet { int32_t result; std::uint64_t event_handle; };
EventCreateRet event_create(unsigned int flags);

struct EventRecordRet { int32_t result; };
EventRecordRet event_record(std::uint64_t event_handle, std::uint64_t stream_handle);

struct EventSyncRet { int32_t result; };
EventSyncRet event_synchronize(std::uint64_t event_handle);

} // namespace cuda_rpc_api

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

// ── Virtual handle CUDA wrappers ──────────────────────────────────────
// Key change: cuMemAlloc and cuCtxCreate are now ENQUEUED, not barriers!
// They return virtual handles that subsequent calls reference.
// Only cuInit and data readbacks are barriers.

// cuInit — BARRIER (first call, need to check result)
static int32_t cu_init(zlink::cuda_pipeline<cuda_test_rpc>& pipe, unsigned int flags = 0) {
    auto r = pipe.call_barrier<0>(flags);
    return r.result;
}

// cuDeviceGetCount — BARRIER (need count)
static int32_t cu_device_get_count(zlink::cuda_pipeline<cuda_test_rpc>& pipe, int& out_count) {
    auto r = pipe.call_barrier<1>();
    out_count = r.count;
    return r.result;
}

// cuDeviceGetName — BARRIER
static int32_t cu_device_get_name(zlink::cuda_pipeline<cuda_test_rpc>& pipe, int ordinal, std::string& out_name) {
    auto r = pipe.call_barrier<2>(ordinal);
    out_name = r.name;
    return r.result;
}

// cuDeviceTotalMem — BARRIER
static int32_t cu_device_total_mem(zlink::cuda_pipeline<cuda_test_rpc>& pipe, int ordinal, uint64_t& out_bytes) {
    auto r = pipe.call_barrier<3>(ordinal);
    out_bytes = r.bytes;
    return r.result;
}

// cuCtxCreate — NOW ENQUEUED with virtual handle! No longer a barrier!
// Returns a virtual handle. The server will translate VH → real CUcontext.
static uint64_t cu_ctx_create_vh(zlink::cuda_pipeline<cuda_test_rpc>& pipe, unsigned int flags, int dev) {
    uint32_t vh = pipe.enqueue_produces_handle<4>(flags, dev);
    return zlink::make_virtual_handle(vh);
}

// cuCtxSynchronize — ENQUEUED
static int32_t cu_ctx_synchronize(zlink::cuda_pipeline<cuda_test_rpc>& pipe) {
    pipe.enqueue<5>();
    return 0;
}

// cuCtxDestroy — ENQUEUED (consumes virtual handle)
static int32_t cu_ctx_destroy(zlink::cuda_pipeline<cuda_test_rpc>& pipe, uint64_t ctx_vh) {
    pipe.enqueue<6>(ctx_vh);
    return 0;
}

// cuMemAlloc — NOW ENQUEUED with virtual handle! The big win!
// No longer a barrier — returns VH instead of waiting for real dev_ptr.
// Subsequent cuMemcpyHtoD, cuLaunchKernel, etc. reference this VH.
static uint64_t cu_mem_alloc_vh(zlink::cuda_pipeline<cuda_test_rpc>& pipe, uint64_t size) {
    uint32_t vh = pipe.enqueue_produces_handle<7>(size);
    return zlink::make_virtual_handle(vh);
}

// cuMemFree — ENQUEUED (consumes virtual handle)
static int32_t cu_mem_free(zlink::cuda_pipeline<cuda_test_rpc>& pipe, uint64_t dev_vh) {
    pipe.enqueue<8>(dev_vh);
    return 0;
}

// cuMemcpyHtoD — ENQUEUED + inline host_sync (consumes virtual dev_ptr handle)
static int32_t cu_memcpy_htod(zlink::cuda_pipeline<cuda_test_rpc>& pipe,
                               uint64_t dev_vh,
                               const void* host_data,
                               uint64_t byte_count) {
    pipe.enqueue_with_sync<9>(
        host_data, static_cast<std::size_t>(byte_count),
        dev_vh,
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host_data)),
        byte_count);
    return 0;
}

// cuMemcpyDtoH — READBACK (consumes virtual dev_ptr handle)
static int32_t cu_memcpy_dtoh(zlink::cuda_pipeline<cuda_test_rpc>& pipe,
                               void* host_data,
                               uint64_t dev_vh,
                               uint64_t byte_count) {
    auto results = pipe.call_readback_with_sync_read<10>(
        host_data, static_cast<std::size_t>(byte_count),
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host_data)),
        dev_vh,
        byte_count);

    if (!results.empty() && results.back().valid) {
        using namespace zpp::bits;
        auto [data, in, out] = data_in_out();
        data.assign(results.back().data.begin(), results.back().data.end());
        cuda_test_rpc::client client{in, out};
        auto r = client.template response<10>().or_throw();
        return r.result;
    }
    return -1;
}

// ── Test — everything pipelines with virtual handles! ─────────────────
int main(int argc, char* argv[]) {
    std::string host = "127.0.0.1";
    std::uint16_t port = zlink::default_port;
    if (argc > 1) host = argv[1];
    if (argc > 2) port = static_cast<std::uint16_t>(std::stoi(argv[2]));

    auto tp = zlink::make_transport(zlink::transport_kind::tcp);
    std::cout << "Connecting to " << host << ":" << port << "...\n";

    auto ec = tp->connect(host, port);
    if (ec) {
        std::cerr << "Failed to connect: " << ec.message() << "\n";
        return 1;
    }
    std::cout << "Connected!\n\n";

    zlink::cuda_pipeline<cuda_test_rpc> pipe(*tp);

    // ── Phase 1: Setup (barrier calls — need return values) ──────────
    std::cout << "=== Phase 1: Setup (barriers) ===" << std::endl;

    int32_t res = cu_init(pipe, 0);
    std::cout << "cuInit: " << res << (res == 0 ? " OK" : " FAIL") << "\n";
    if (res != 0) return 1;

    int dev_count = 0;
    cu_device_get_count(pipe, dev_count);
    std::cout << "Device count: " << dev_count << "\n";

    std::string gpu_name;
    cu_device_get_name(pipe, 0, gpu_name);
    std::cout << "GPU 0: " << gpu_name << "\n";

    uint64_t total_mem = 0;
    cu_device_total_mem(pipe, 0, total_mem);
    std::cout << "Total memory: " << (total_mem / (1024*1024)) << " MB\n";

    // ── Phase 2: Virtual handle pipeline — THE KEY TEST ──────────────
    // ALL of these calls are enqueued! No barriers between them!
    // cuCtxCreate, cuMemAlloc, cuMemcpyHtoD, cuCtxSynchronize — all batched.
    // Only cuMemcpyDtoH forces a flush (readback).
    std::cout << "\n=== Phase 2: Virtual Handle Pipeline ===" << std::endl;
    std::cout << "ALL calls enqueued — NO barriers between alloc/HtoD/sync!\n\n";

    // cuCtxCreate → VH(0) — ENQUEUED, not a barrier!
    uint64_t ctx_vh = cu_ctx_create_vh(pipe, 0u, 0);
    std::cout << "cuCtxCreate: VH(" << zlink::virtual_handle_id(ctx_vh) << ") — enqueued\n";

    // cuMemAlloc → VH(1) — ENQUEUED, not a barrier!
    const std::size_t buf_size = 256;
    uint64_t dev_ptr_vh = cu_mem_alloc_vh(pipe, static_cast<uint64_t>(buf_size));
    std::cout << "cuMemAlloc: VH(" << zlink::virtual_handle_id(dev_ptr_vh) << ") — enqueued\n";

    // cuMemcpyHtoD(VH(1), ...) — ENQUEUED + inline host_sync
    std::vector<float> host_data(buf_size / sizeof(float));
    for (size_t i = 0; i < host_data.size(); i++) {
        host_data[i] = static_cast<float>(i) * 1.5f;
    }

    res = cu_memcpy_htod(pipe, dev_ptr_vh, host_data.data(),
                          static_cast<uint64_t>(buf_size));
    std::cout << "cuMemcpyHtoD(VH(" << zlink::virtual_handle_id(dev_ptr_vh)
              << ")): enqueued + inline sync\n";

    // cuCtxSynchronize — ENQUEUED
    res = cu_ctx_synchronize(pipe);
    std::cout << "cuCtxSynchronize: enqueued\n";

    // cuMemcpyDtoH — READBACK: flush pipeline + get data back
    // This is where ALL the enqueued calls go out in ONE frame!
    std::cout << "\n=== cuMemcpyDtoH: READBACK — flushing pipeline ===" << std::endl;
    std::cout << "Pipeline sends: [sync_data][ctx_create][mem_alloc][memcpy_htod]"
              << "[ctx_sync][memcpy_dtoh][read_req][handle_manifest]\n";
    std::cout << "Server processes in order, translates VH → real handles\n\n";

    std::vector<float> readback(buf_size / sizeof(float), 0.0f);
    res = cu_memcpy_dtoh(pipe, readback.data(), dev_ptr_vh,
                          static_cast<uint64_t>(buf_size));
    std::cout << "cuMemcpyDtoH result: " << res
              << (res == 0 ? " (SUCCESS)" : " (FAILED)") << "\n";

    // Verify data round-trip
    if (res == 0) {
        std::cout << "\n=== Data Verification ===" << std::endl;
        bool match = true;
        int mismatches = 0;
        for (size_t i = 0; i < host_data.size(); i++) {
            if (readback[i] != host_data[i]) {
                if (mismatches < 5) {
                    std::cout << "  MISMATCH at [" << i << "]: expected "
                              << host_data[i] << ", got " << readback[i] << "\n";
                }
                match = false;
                mismatches++;
            }
        }
        if (match) {
            std::cout << "  All " << host_data.size() << " values match!\n";
            std::cout << "  Virtual handle pipeline verified: alloc→HtoD→sync→DtoH in 1 round-trip!\n";
        } else {
            std::cout << "  " << mismatches << " mismatches out of "
                      << host_data.size() << " values\n";
        }
    }

    // Cleanup — also enqueued with virtual handles
    cu_mem_free(pipe, dev_ptr_vh);
    cu_ctx_destroy(pipe, ctx_vh);
    std::cout << "\nCleanup: cuMemFree + cuCtxDestroy enqueued\n";

    // Final flush
    auto final_results = pipe.flush();
    std::cout << "Final flush: " << final_results.size() << " deferred calls\n";

    // ── Pipeline benchmark: virtual handles vs barriers ──────────────
    std::cout << "\n=== Pipeline Benchmark: Virtual Handles vs Barriers ===" << std::endl;

    // Test 1: Virtual handles — entire alloc+HtoD+sync+DtoH pipeline
    auto t_vh_start = std::chrono::high_resolution_clock::now();
    {
        zlink::cuda_pipeline<cuda_test_rpc> vh_pipe(*tp);

        // All enqueued with virtual handles — 1 round-trip at readback
        uint64_t vh_ctx = cu_ctx_create_vh(vh_pipe, 0u, 0);
        uint64_t vh_ptr = cu_mem_alloc_vh(vh_pipe, buf_size);

        std::vector<float> bench_data(buf_size / sizeof(float), 3.14f);
        cu_memcpy_htod(vh_pipe, vh_ptr, bench_data.data(), buf_size);
        cu_ctx_synchronize(vh_pipe);

        std::vector<float> bench_read(buf_size / sizeof(float), 0.0f);
        cu_memcpy_dtoh(vh_pipe, bench_read.data(), vh_ptr, buf_size);

        cu_mem_free(vh_pipe, vh_ptr);
        cu_ctx_destroy(vh_pipe, vh_ctx);
        vh_pipe.flush();
    }
    auto t_vh_end = std::chrono::high_resolution_clock::now();
    auto vh_us = std::chrono::duration_cast<std::chrono::microseconds>(t_vh_end - t_vh_start).count();

    // Test 2: Barrier-style — same calls, but alloc + ctx_create are barriers
    auto t_barrier_start = std::chrono::high_resolution_clock::now();
    {
        zlink::cuda_pipeline<cuda_test_rpc> bar_pipe(*tp);

        // Barrier calls (the old way)
        auto r1 = bar_pipe.call_barrier<4>(0u, 0);  // ctx_create
        uint64_t real_ctx = r1.ctx_handle;
        auto r2 = bar_pipe.call_barrier<7>(static_cast<uint64_t>(buf_size));  // mem_alloc
        uint64_t real_ptr = r2.dev_ptr;

        // Now these can be enqueued since we have real values
        std::vector<float> bench_data(buf_size / sizeof(float), 3.14f);
        bar_pipe.enqueue_with_sync<9>(
            bench_data.data(), buf_size,
            real_ptr,
            reinterpret_cast<std::uint64_t>(bench_data.data()),
            buf_size);
        bar_pipe.enqueue<5>();  // ctx_synchronize

        // Readback
        std::vector<float> bench_read(buf_size / sizeof(float), 0.0f);
        bar_pipe.call_readback_with_sync_read<10>(
            bench_read.data(), buf_size,
            reinterpret_cast<std::uint64_t>(bench_read.data()),
            real_ptr,
            buf_size);

        bar_pipe.enqueue<8>(real_ptr);
        bar_pipe.enqueue<6>(real_ctx);
        bar_pipe.flush();
    }
    auto t_barrier_end = std::chrono::high_resolution_clock::now();
    auto barrier_us = std::chrono::duration_cast<std::chrono::microseconds>(t_barrier_end - t_barrier_start).count();

    std::cout << "  Virtual handles (1-2 round-trips):  " << vh_us << " us\n"
              << "  Barrier style (3+ round-trips):     " << barrier_us << " us\n"
              << "  Speedup: " << (barrier_us > 0 ? static_cast<double>(barrier_us) / vh_us : 0.0) << "x\n";

    tp->close();
    return 0;
}
