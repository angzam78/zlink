// zlink/examples/cuda/cuda_client.cpp — CUDA pipeline client with virtual handles
//
// Connects to cuda_server and runs a test workload that exercises:
//   - barrier calls (cuInit, cuDeviceGetCount, cuDeviceGetAttribute, ...)
//   - enqueued handle producers (cuCtxCreate, cuMemAlloc, cuStreamCreate)
//   - enqueued with inline host_sync (cuMemcpyHtoD)
//   - enqueued handle consumers (cuMemcpyDtoD, cuCtxSynchronize)
//   - readback (cuMemcpyDtoH — flushes the pipeline + gets data back)
//
// The entire Phase-2 batch (alloc + alloc + stream + HtoD + DtoD + sync)
// is enqueued with virtual handles and flushed in ONE round-trip by the
// readback call. This is the core zlink performance win.
//
// Run: cuda_client [host [port]]
//
// Environment variables:
//   ZLINK_MANAGED=1  Enable managed pipeline (prefetch + write-behind + multiplexed transport)
//   ZLINK_MANAGED unset or 0  Use plain pipeline behavior (inline sync, no background workers)

#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/config.hpp>
#include <zlink/cuda_pipeline.hpp>
#include <zlink/managed_pipeline.hpp>
#include <zlink/multiplexed_transport.hpp>
#include <zlink/virtual_handle.hpp>

#include "cuda_api.hpp"

#include <zpp_bits.h>

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cmath>

using pipe_t = zlink::managed_pipeline<cuda_gen::cuda_gen_rpc>;

// ── BARRIER WRAPPERS ───────────────────────────────────────────────────

static int32_t cu_init(pipe_t& pipe, unsigned int flags = 0) {
    auto r = pipe.call_barrier_managed<cuda_gen::func_index::init>(flags);
    return r.result;
}
static int32_t cu_driver_get_version(pipe_t& pipe, int& out_ver) {
    auto r = pipe.call_barrier_managed<cuda_gen::func_index::driver_get_version>();
    out_ver = r.driverVersion;
    return r.result;
}
static int32_t cu_device_get_count(pipe_t& pipe, int& out_count) {
    auto r = pipe.call_barrier_managed<cuda_gen::func_index::device_get_count>();
    out_count = r.count;
    return r.result;
}
static int32_t cu_device_get_name(pipe_t& pipe, int ordinal, std::string& out_name) {
    auto r = pipe.call_barrier_managed<cuda_gen::func_index::device_get_name>(128, ordinal);
    out_name = r.name;
    return r.result;
}
static int32_t cu_device_total_mem(pipe_t& pipe, int ordinal, uint64_t& out_bytes) {
    auto r = pipe.call_barrier_managed<cuda_gen::func_index::device_total_mem>(ordinal);
    out_bytes = r.bytes;
    return r.result;
}
static int32_t cu_device_get_attribute(pipe_t& pipe, int attrib, int dev, int32_t& out_value) {
    auto r = pipe.call_barrier_managed<cuda_gen::func_index::device_get_attribute>(
        static_cast<uint32_t>(attrib), dev);
    out_value = r.value;
    return r.result;
}
static int32_t cu_mem_get_info(pipe_t& pipe, uint64_t& free_bytes, uint64_t& total_bytes) {
    auto r = pipe.call_barrier_managed<cuda_gen::func_index::mem_get_info>();
    free_bytes = r.free_bytes;
    total_bytes = r.total_bytes;
    return r.result;
}

// ── ENQUEUED: handle producers (return virtual handles, no barrier!) ───

static uint64_t cu_ctx_create(pipe_t& pipe, unsigned int flags, int dev) {
    uint32_t vh = pipe.enqueue_produces_handle<cuda_gen::func_index::ctx_create>(flags, dev);
    return zlink::make_virtual_handle(vh);
}
static uint64_t cu_mem_alloc(pipe_t& pipe, uint64_t size) {
    uint32_t vh = pipe.enqueue_produces_handle<cuda_gen::func_index::mem_alloc>(size);
    return zlink::make_virtual_handle(vh);
}
static uint64_t cu_mem_alloc_managed(pipe_t& pipe, uint64_t size, unsigned int flags) {
    uint32_t vh = pipe.enqueue_produces_handle<cuda_gen::func_index::mem_alloc_managed>(size, flags);
    return zlink::make_virtual_handle(vh);
}
static uint64_t cu_stream_create(pipe_t& pipe, unsigned int flags = 0) {
    uint32_t vh = pipe.enqueue_produces_handle<cuda_gen::func_index::stream_create>(flags);
    return zlink::make_virtual_handle(vh);
}
static uint64_t cu_event_create(pipe_t& pipe, unsigned int flags = 0) {
    uint32_t vh = pipe.enqueue_produces_handle<cuda_gen::func_index::event_create>(flags);
    return zlink::make_virtual_handle(vh);
}

// ── ENQUEUED: handle consumers ─────────────────────────────────────────

static void cu_ctx_set_current(pipe_t& pipe, uint64_t ctx_vh) {
    pipe.enqueue<cuda_gen::func_index::ctx_set_current>(ctx_vh);
}
static void cu_ctx_synchronize(pipe_t& pipe) {
    pipe.enqueue<cuda_gen::func_index::ctx_synchronize>();
}
static void cu_mem_free(pipe_t& pipe, uint64_t dev_vh) {
    pipe.enqueue<cuda_gen::func_index::mem_free>(dev_vh);
}
static void cu_stream_destroy(pipe_t& pipe, uint64_t stream_vh) {
    pipe.enqueue<cuda_gen::func_index::stream_destroy>(stream_vh);
}
static void cu_event_record(pipe_t& pipe, uint64_t event_vh, uint64_t stream_vh) {
    pipe.enqueue<cuda_gen::func_index::event_record>(event_vh, stream_vh);
}
static void cu_event_synchronize(pipe_t& pipe, uint64_t event_vh) {
    pipe.enqueue<cuda_gen::func_index::event_synchronize>(event_vh);
}

// cuMemcpyHtoD: consumes VH(dev_ptr) + inline host_sync (data packed into frame)
static void cu_memcpy_htod(pipe_t& pipe, uint64_t dev_vh,
                           const void* host_data, uint64_t byte_count) {
    pipe.enqueue_with_sync_managed<cuda_gen::func_index::memcpy_hto_d>(
        host_data, static_cast<std::size_t>(byte_count),
        dev_vh,
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host_data)),
        byte_count);
}

// cuMemcpyDtoD: consumes VH(dst) + VH(src)
static void cu_memcpy_dtod(pipe_t& pipe, uint64_t dst_vh, uint64_t src_vh, uint64_t byte_count) {
    pipe.enqueue<cuda_gen::func_index::memcpy_dto_d>(dst_vh, src_vh, byte_count);
}

// ── READBACK: cuMemcpyDtoH (flushes pipeline + gets data back) ──────────
// The host_sync creates the mirror region; the RPC runs the DtoH on the
// server (copies GPU→mirror); the host_read pulls mirror data back to the
// client. All packed into one pipeline_mem round-trip.
static int32_t cu_memcpy_dtoh(pipe_t& pipe, void* host_data,
                              uint64_t dev_vh, uint64_t byte_count) {
    auto results = pipe.call_readback_managed<cuda_gen::func_index::memcpy_dto_h>(
        host_data, static_cast<std::size_t>(byte_count),
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host_data)),
        dev_vh,
        byte_count);

    if (!results.empty() && results.back().valid) {
        using namespace zpp::bits;
        auto [data, in, out] = data_in_out();
        data.assign(results.back().data.begin(), results.back().data.end());
        cuda_gen::cuda_gen_rpc::client client{in, out};
        auto r = client.template response<cuda_gen::func_index::memcpy_dto_h>().or_throw();
        return r.result;
    }
    return -1;
}

// ── BARRIER: cuEventElapsedTime (needs the float value) ─────────────────
static int32_t cu_event_elapsed_time(pipe_t& pipe, uint64_t start_evh, uint64_t end_evh, float& out_ms) {
    auto r = pipe.call_barrier_managed<cuda_gen::func_index::event_elapsed_time>(start_evh, end_evh);
    out_ms = r.milliseconds;
    return r.result;
}

// ══════════════════════════════════════════════════════════════════════════
// TEST WORKLOAD
// ══════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    std::string host = "127.0.0.1";
    std::uint16_t port = zlink::default_port;
    if (argc > 1) host = argv[1];
    if (argc > 2) port = static_cast<std::uint16_t>(std::stoi(argv[2]));

    // ── Check ZLINK_MANAGED env var ──────────────────────────────
    const char* managed_env = std::getenv("ZLINK_MANAGED");
    bool use_managed = managed_env && managed_env[0] == '1';

    zlink::multiplexed_transport mtp;
    std::cout << "Connecting to " << host << ":" << port << "...\n";
    if (use_managed) {
        std::cout << "Managed pipeline ENABLED (prefetch + write-behind)\n";
    } else {
        std::cout << "Managed pipeline DISABLED (inline sync, no background workers)\n";
    }
    auto ec = mtp.connect(host, port);
    if (ec) { std::cerr << "connect failed: " << ec.message() << "\n"; return 1; }
    std::cout << "Connected"
              << (mtp.is_multi_port() ? " (multi-port mode)" : " (single-connection mode)")
              << "\n\n";

    // ── Create managed pipeline (workers toggled by config) ──────
    zlink::managed_pipeline_config mcfg;
    mcfg.enable_prefetch     = use_managed;
    mcfg.enable_write_behind = use_managed;

    pipe_t pipe(mtp, mcfg);

    // Prefetch needs a chunk_cache wired via set_cache().
    // For now, prefetch is enabled but has no cache to pull from —
    // it will simply not prefetch anything (no-op). Write-behind
    // works independently of the cache.
    pipe.start();

    // ── Phase 1: Setup (barrier calls) ────────────────────────────────
    std::cout << "=== Phase 1: Setup (barriers) ===\n";
    int32_t res = cu_init(pipe, 0);
    std::cout << "cuInit: " << res << (res == 0 ? " OK" : " FAIL") << "\n";
    if (res != 0) return 1;

    int drv_ver = 0;
    cu_driver_get_version(pipe, drv_ver);
    std::cout << "Driver version: " << drv_ver << "\n";

    int dev_count = 0;
    cu_device_get_count(pipe, dev_count);
    std::cout << "Device count: " << dev_count << "\n";

    std::string gpu_name;
    cu_device_get_name(pipe, 0, gpu_name);
    std::cout << "GPU 0: " << gpu_name << "\n";

    uint64_t total_mem = 0;
    cu_device_total_mem(pipe, 0, total_mem);
    std::cout << "Total memory: " << (total_mem / (1024 * 1024)) << " MB\n";

    int32_t warp_size = 0;
    cu_device_get_attribute(pipe, 4 /*CU_DEVICE_ATTRIBUTE_WARP_SIZE*/, 0, warp_size);
    std::cout << "Warp size: " << warp_size << "\n";

    uint64_t free_mem = 0, total_avail = 0;
    cu_mem_get_info(pipe, free_mem, total_avail);
    std::cout << "Free memory: " << (free_mem / (1024 * 1024)) << " MB / "
              << (total_avail / (1024 * 1024)) << " MB\n";

    // ── Phase 2: Virtual-handle pipeline (the key win!) ──────────────
    std::cout << "\n=== Phase 2: Virtual Handle Pipeline ===\n";
    std::cout << "ALL calls enqueued — NO barriers between alloc/HtoD/sync!\n\n";

    uint64_t ctx_vh = cu_ctx_create(pipe, 0u, 0);
    std::cout << "cuCtxCreate: VH(" << zlink::virtual_handle_id(ctx_vh) << ") — enqueued\n";

    const std::size_t buf_size = 256;
    uint64_t dev_ptr_vh = cu_mem_alloc(pipe, buf_size);
    std::cout << "cuMemAlloc:  VH(" << zlink::virtual_handle_id(dev_ptr_vh) << ") — enqueued\n";

    uint64_t dev_ptr2_vh = cu_mem_alloc(pipe, buf_size);
    std::cout << "cuMemAlloc(2): VH(" << zlink::virtual_handle_id(dev_ptr2_vh) << ") — enqueued\n";

    uint64_t stream_vh = cu_stream_create(pipe, 0);
    std::cout << "cuStreamCreate: VH(" << zlink::virtual_handle_id(stream_vh) << ") — enqueued\n";

    std::vector<float> host_data(buf_size / sizeof(float));
    for (size_t i = 0; i < host_data.size(); i++) {
        host_data[i] = static_cast<float>(i) * 1.5f;
    }

    cu_memcpy_htod(pipe, dev_ptr_vh, host_data.data(), buf_size);
    std::cout << "cuMemcpyHtoD(VH(" << zlink::virtual_handle_id(dev_ptr_vh)
              << ")): enqueued (inline sync)\n";

    cu_memcpy_dtod(pipe, dev_ptr2_vh, dev_ptr_vh, buf_size);
    std::cout << "cuMemcpyDtoD: enqueued\n";

    cu_ctx_synchronize(pipe);
    std::cout << "cuCtxSynchronize: enqueued\n";

    // cuMemcpyDtoH — READBACK: this flushes the ENTIRE batch above
    // (ctx_create + 2×alloc + stream + HtoD + DtoD + sync) in ONE round-trip,
    // then reads the data back.
    std::cout << "\n=== cuMemcpyDtoH: READBACK — flushing pipeline ===\n";
    std::vector<float> readback(buf_size / sizeof(float), 0.0f);
    res = cu_memcpy_dtoh(pipe, readback.data(), dev_ptr2_vh, buf_size);
    std::cout << "cuMemcpyDtoH result: " << res
              << (res == 0 ? " (SUCCESS)" : " (FAILED)") << "\n";

    if (res == 0) {
        std::cout << "\n=== Data Verification (DtoD round-trip) ===\n";
        bool match = true;
        int mismatches = 0;
        for (size_t i = 0; i < host_data.size(); i++) {
            if (std::abs(readback[i] - host_data[i]) > 1e-6f) {
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
            std::cout << "  VH pipeline verified: alloc+alloc+stream+HtoD+DtoD+sync+DtoH"
                      << " in 1 round-trip!\n";
        } else {
            std::cout << "  " << mismatches << " mismatches out of "
                      << host_data.size() << " values\n";
        }
    }

    // ── Phase 3: Event pipeline ──────────────────────────────────────
    std::cout << "\n=== Phase 3: Event Pipeline Test ===\n";

    uint64_t start_event_vh = cu_event_create(pipe, 0);
    uint64_t end_event_vh = cu_event_create(pipe, 0);
    std::cout << "cuEventCreate x2: VH(" << zlink::virtual_handle_id(start_event_vh)
              << "), VH(" << zlink::virtual_handle_id(end_event_vh) << ") — enqueued\n";

    cu_event_record(pipe, start_event_vh, stream_vh);
    std::cout << "cuEventRecord(start): enqueued\n";

    cu_event_record(pipe, end_event_vh, stream_vh);
    std::cout << "cuEventRecord(end): enqueued\n";

    cu_event_synchronize(pipe, end_event_vh);
    std::cout << "cuEventSynchronize(end): enqueued\n";

    // Flush the event batch
    auto event_results = pipe.flush_managed();
    std::cout << "Event batch flushed: " << event_results.size() << " results\n";

    // cuEventElapsedTime is a barrier (needs the float value)
    float elapsed_ms = 0.0f;
    res = cu_event_elapsed_time(pipe, start_event_vh, end_event_vh, elapsed_ms);
    std::cout << "cuEventElapsedTime: " << res
              << (res == 0 ? " OK" : " FAIL")
              << ", " << elapsed_ms << " ms\n";

    // ── Phase 4: Managed memory + cleanup ────────────────────────────
    std::cout << "\n=== Phase 4: Managed Memory + Cleanup ===\n";

    uint64_t managed_vh = cu_mem_alloc_managed(pipe, buf_size, 1 /*CU_MEM_ATTACH_GLOBAL*/);
    std::cout << "cuMemAllocManaged: VH(" << zlink::virtual_handle_id(managed_vh)
              << ") — enqueued\n";

    // Copy into managed memory, read back to verify
    cu_memcpy_htod(pipe, managed_vh, host_data.data(), buf_size);
    cu_ctx_synchronize(pipe);
    std::cout << "cuMemcpyHtoD(managed) + cuCtxSynchronize: enqueued\n";

    std::vector<float> managed_readback(buf_size / sizeof(float), 0.0f);
    res = cu_memcpy_dtoh(pipe, managed_readback.data(), managed_vh, buf_size);
    std::cout << "cuMemcpyDtoH(managed) result: " << res
              << (res == 0 ? " (SUCCESS)" : " (FAILED)") << "\n";

    if (res == 0) {
        bool match = true;
        for (size_t i = 0; i < host_data.size(); i++) {
            if (std::abs(managed_readback[i] - host_data[i]) > 1e-6f) { match = false; break; }
        }
        std::cout << "  Managed memory round-trip: "
                  << (match ? "VERIFIED" : "MISMATCH") << "\n";
    }

    // Cleanup: enqueue frees, then flush
    cu_mem_free(pipe, dev_ptr_vh);
    cu_mem_free(pipe, dev_ptr2_vh);
    cu_mem_free(pipe, managed_vh);
    cu_stream_destroy(pipe, stream_vh);
    std::cout << "Cleanup (3×mem_free + stream_destroy): enqueued\n";

    auto final_results = pipe.flush_managed();
    std::cout << "Cleanup batch flushed: " << final_results.size() << " results\n";

    std::cout << "\n=== All tests complete ===\n";
    pipe.stop();
    mtp.close();
    return 0;
}
