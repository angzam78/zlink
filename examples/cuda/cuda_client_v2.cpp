// zlink/examples/cuda/cuda_client_v2.cpp
//
// CUDA RPC client v2 — Full PyTorch-relevant API with virtual handles.
//
// TEMPLATE INDEX CONVENTION:
//   We use plain integer literals as template args (0, 1, 2...) matching
//   the zpp_bits binding IDs in cuda_api_v2.hpp. This avoids C++20
//   identifier parsing issues with names like "10_managed".

#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/ptr_map.hpp>
#include <zlink/memory.hpp>
#include <zlink/config.hpp>
#include <zlink/rpc.hpp>
#include <zlink/cuda_pipeline.hpp>
#include <zlink/virtual_handle.hpp>

#include "cuda_api_v2.hpp"

#include <zpp_bits.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cmath>

using pipe_t = zlink::cuda_pipeline<cuda_v2_rpc>;

// ── BARRIER WRAPPERS (indices 0-4, 8, 17, 36, 37) ──────────────────

static int32_t cu_init(pipe_t& pipe, unsigned int flags = 0) {
    auto r = pipe.call_barrier<0>(flags);
    return r.result;
}
static int32_t cu_device_get_count(pipe_t& pipe, int& out_count) {
    auto r = pipe.call_barrier<1>();
    out_count = r.count;
    return r.result;
}
static int32_t cu_device_get_name(pipe_t& pipe, int ordinal, std::string& out_name) {
    auto r = pipe.call_barrier<2>(ordinal);
    out_name = r.name;
    return r.result;
}
static int32_t cu_device_total_mem(pipe_t& pipe, int ordinal, uint64_t& out_bytes) {
    auto r = pipe.call_barrier<3>(ordinal);
    out_bytes = r.bytes;
    return r.result;
}
static int32_t cu_device_get_attribute(pipe_t& pipe, int attrib, int dev, int32_t& out_value) {
    auto r = pipe.call_barrier<4>(attrib, dev);
    out_value = r.value;
    return r.result;
}
static int32_t cu_ctx_get_current(pipe_t& pipe, uint64_t& out_ctx) {
    auto r = pipe.call_barrier<8>();
    out_ctx = r.ctx_handle;
    return r.result;
}
static int32_t cu_mem_get_info(pipe_t& pipe, uint64_t& free_bytes, uint64_t& total_bytes) {
    auto r = pipe.call_barrier<17>();
    free_bytes = r.free_bytes;
    total_bytes = r.total_bytes;
    return r.result;
}
static int32_t cu_event_elapsed_time(pipe_t& pipe, uint64_t start_evh, uint64_t end_evh, float& out_ms) {
    auto r = pipe.call_barrier<36>(start_evh, end_evh);
    out_ms = r.milliseconds;
    return r.result;
}

// ── ENQUEUED WRAPPERS — handle producers ────────────────────────────

// index 5: ctx_create → PRODUCES VH(ctx)
static uint64_t cu_ctx_create(pipe_t& pipe, unsigned int flags, int dev) {
    uint32_t vh = pipe.enqueue_produces_handle<5>(flags, dev);
    return zlink::make_virtual_handle(vh);
}
// index 10: mem_alloc → PRODUCES VH(dev_ptr) ★ THE KEY WIN
static uint64_t cu_mem_alloc(pipe_t& pipe, uint64_t size) {
    uint32_t vh = pipe.enqueue_produces_handle<10>(size);
    return zlink::make_virtual_handle(vh);
}
// index 11: mem_alloc_managed → PRODUCES VH(dev_ptr)
static uint64_t cu_mem_alloc_managed(pipe_t& pipe, uint64_t size, unsigned int flags) {
    uint32_t vh = pipe.enqueue_produces_handle<11>(size, flags);
    return zlink::make_virtual_handle(vh);
}
// index 14: mem_host_alloc → PRODUCES VH(host_ptr)
static uint64_t cu_mem_host_alloc(pipe_t& pipe, uint64_t size, unsigned int flags) {
    uint32_t vh = pipe.enqueue_produces_handle<14>(size, flags);
    return zlink::make_virtual_handle(vh);
}
// index 23: module_load_data → PRODUCES VH(module) + inline host_sync
static uint64_t cu_module_load_data(pipe_t& pipe, const void* image, uint64_t byte_count) {
    uint32_t vh = pipe.enqueue_produces_handle_with_sync<23>(
        image, static_cast<std::size_t>(byte_count),
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(image)),
        byte_count);
    return zlink::make_virtual_handle(vh);
}
// index 24: module_load → PRODUCES VH(module) via filename
static uint64_t cu_module_load(pipe_t& pipe, const std::string& filename) {
    uint32_t vh = pipe.enqueue_produces_handle<24>(filename);
    return zlink::make_virtual_handle(vh);
}
// index 26: module_get_function → PRODUCES VH(func) + consumes VH(module)
static uint64_t cu_module_get_function(pipe_t& pipe, uint64_t module_vh, const std::string& name) {
    uint32_t vh = pipe.enqueue_produces_handle<26>(module_vh, name);
    return zlink::make_virtual_handle(vh);
}
// index 27: module_get_global → PRODUCES VH(global_ptr) + consumes VH(module)
static uint64_t cu_module_get_global(pipe_t& pipe, uint64_t module_vh, const std::string& name) {
    uint32_t vh = pipe.enqueue_produces_handle<27>(module_vh, name);
    return zlink::make_virtual_handle(vh);
}
// index 29: stream_create → PRODUCES VH(stream)
static uint64_t cu_stream_create(pipe_t& pipe, unsigned int flags = 0) {
    uint32_t vh = pipe.enqueue_produces_handle<29>(flags);
    return zlink::make_virtual_handle(vh);
}
// index 32: event_create → PRODUCES VH(event)
static uint64_t cu_event_create(pipe_t& pipe, unsigned int flags = 0) {
    uint32_t vh = pipe.enqueue_produces_handle<32>(flags);
    return zlink::make_virtual_handle(vh);
}

// ── ENQUEUED WRAPPERS — handle consumers (no VH produced) ───────────

// index 6: ctx_destroy — consumes VH(ctx)
static int32_t cu_ctx_destroy(pipe_t& pipe, uint64_t ctx_vh) {
    pipe.enqueue<6>(ctx_vh);
    return 0;
}
// index 7: ctx_set_current — consumes VH(ctx)
static int32_t cu_ctx_set_current(pipe_t& pipe, uint64_t ctx_vh) {
    pipe.enqueue<7>(ctx_vh);
    return 0;
}
// index 9: ctx_synchronize
static int32_t cu_ctx_synchronize(pipe_t& pipe) {
    pipe.enqueue<9>();
    return 0;
}
// index 12: mem_free — consumes VH(dev_ptr)
static int32_t cu_mem_free(pipe_t& pipe, uint64_t dev_vh) {
    pipe.enqueue<12>(dev_vh);
    return 0;
}
// index 13: mem_free_host — consumes VH(host_ptr)
static int32_t cu_mem_free_host(pipe_t& pipe, uint64_t host_vh) {
    pipe.enqueue<13>(host_vh);
    return 0;
}
// index 15: mem_host_register — consumes host ptr
static int32_t cu_mem_host_register(pipe_t& pipe, uint64_t host_ptr, uint64_t size, unsigned int flags) {
    pipe.enqueue<15>(host_ptr, size, flags);
    return 0;
}
// index 16: mem_host_unregister — consumes host ptr
static int32_t cu_mem_host_unregister(pipe_t& pipe, uint64_t host_ptr) {
    pipe.enqueue<16>(host_ptr);
    return 0;
}
// index 18: memcpy_htod — consumes VH(dev_ptr) + inline host_sync
static int32_t cu_memcpy_htod(pipe_t& pipe, uint64_t dev_vh,
                               const void* host_data, uint64_t byte_count) {
    pipe.enqueue_with_sync<18>(
        host_data, static_cast<std::size_t>(byte_count),
        dev_vh,
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host_data)),
        byte_count);
    return 0;
}
// index 20: memcpy_dtod — consumes VH(dst) + VH(src)
static int32_t cu_memcpy_dtod(pipe_t& pipe, uint64_t dst_vh, uint64_t src_vh, uint64_t byte_count) {
    pipe.enqueue<20>(dst_vh, src_vh, byte_count);
    return 0;
}
// index 21: memcpy_htod_async — consumes VH(dev_ptr, stream) + inline host_sync
static int32_t cu_memcpy_htod_async(pipe_t& pipe, uint64_t dev_vh,
                                     const void* host_data, uint64_t byte_count,
                                     uint64_t stream_vh) {
    pipe.enqueue_with_sync<21>(
        host_data, static_cast<std::size_t>(byte_count),
        dev_vh,
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host_data)),
        byte_count,
        stream_vh);
    return 0;
}
// index 25: module_unload — consumes VH(module)
static int32_t cu_module_unload(pipe_t& pipe, uint64_t module_vh) {
    pipe.enqueue<25>(module_vh);
    return 0;
}
// index 28: launch_kernel — consumes VH(func, stream, args) + inline host_sync
static int32_t cu_launch_kernel(pipe_t& pipe,
    uint64_t func_vh,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    uint32_t shared_mem, uint64_t stream_vh,
    const void* args, uint64_t args_size) {
    pipe.enqueue_with_sync<28>(
        args, static_cast<std::size_t>(args_size),
        func_vh,
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        shared_mem, stream_vh,
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(args)),
        args_size);
    return 0;
}
// index 30: stream_destroy — consumes VH(stream)
static int32_t cu_stream_destroy(pipe_t& pipe, uint64_t stream_vh) {
    pipe.enqueue<30>(stream_vh);
    return 0;
}
// index 31: stream_synchronize — consumes VH(stream)
static int32_t cu_stream_synchronize(pipe_t& pipe, uint64_t stream_vh) {
    pipe.enqueue<31>(stream_vh);
    return 0;
}
// index 33: event_destroy — consumes VH(event)
static int32_t cu_event_destroy(pipe_t& pipe, uint64_t event_vh) {
    pipe.enqueue<33>(event_vh);
    return 0;
}
// index 34: event_record — consumes VH(event, stream)
static int32_t cu_event_record(pipe_t& pipe, uint64_t event_vh, uint64_t stream_vh) {
    pipe.enqueue<34>(event_vh, stream_vh);
    return 0;
}
// index 35: event_synchronize — consumes VH(event)
static int32_t cu_event_synchronize(pipe_t& pipe, uint64_t event_vh) {
    pipe.enqueue<35>(event_vh);
    return 0;
}

// ── READBACK WRAPPERS (indices 19, 22) ──────────────────────────────

// index 19: memcpy_dtoh — READBACK + consumes VH(src_dev_ptr)
static int32_t cu_memcpy_dtoh(pipe_t& pipe, void* host_data,
                               uint64_t dev_vh, uint64_t byte_count) {
    auto results = pipe.call_readback_with_sync_read<19>(
        host_data, static_cast<std::size_t>(byte_count),
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host_data)),
        dev_vh,
        byte_count);

    if (!results.empty() && results.back().valid) {
        using namespace zpp::bits;
        auto [data, in, out] = data_in_out();
        data.assign(results.back().data.begin(), results.back().data.end());
        cuda_v2_rpc::client client{in, out};
        auto r = client.template response<19>().or_throw();
        return r.result;
    }
    return -1;
}

// index 22: memcpy_dtoh_async — READBACK + consumes VH(src_dev_ptr, stream)
static int32_t cu_memcpy_dtoh_async(pipe_t& pipe, void* host_data,
                                      uint64_t dev_vh, uint64_t byte_count,
                                      uint64_t stream_vh) {
    auto results = pipe.call_readback_with_sync_read<22>(
        host_data, static_cast<std::size_t>(byte_count),
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host_data)),
        dev_vh,
        byte_count,
        stream_vh);

    if (!results.empty() && results.back().valid) {
        using namespace zpp::bits;
        auto [data, in, out] = data_in_out();
        data.assign(results.back().data.begin(), results.back().data.end());
        cuda_v2_rpc::client client{in, out};
        auto r = client.template response<22>().or_throw();
        return r.result;
    }
    return -1;
}

// ══════════════════════════════════════════════════════════════════════
// TEST
// ══════════════════════════════════════════════════════════════════════
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

    pipe_t pipe(*tp);

    // ── Phase 1: Setup (barrier calls) ───────────────────────────────
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

    int32_t warp_size = 0;
    cu_device_get_attribute(pipe, 4 /*CU_DEVICE_ATTRIBUTE_WARP_SIZE*/, 0, warp_size);
    std::cout << "Warp size: " << warp_size << "\n";

    uint64_t free_mem = 0, total_available = 0;
    cu_mem_get_info(pipe, free_mem, total_available);
    std::cout << "Free memory: " << (free_mem / (1024*1024)) << " MB / "
              << (total_available / (1024*1024)) << " MB\n";

    // ── Phase 2: Virtual handle pipeline ─────────────────────────────
    std::cout << "\n=== Phase 2: Virtual Handle Pipeline ===" << std::endl;
    std::cout << "ALL calls enqueued — NO barriers between alloc/HtoD/sync!\n\n";

    uint64_t ctx_vh = cu_ctx_create(pipe, 0u, 0);
    std::cout << "cuCtxCreate: VH(" << zlink::virtual_handle_id(ctx_vh) << ") — enqueued\n";

    const std::size_t buf_size = 256;
    uint64_t dev_ptr_vh = cu_mem_alloc(pipe, static_cast<uint64_t>(buf_size));
    std::cout << "cuMemAlloc: VH(" << zlink::virtual_handle_id(dev_ptr_vh) << ") — enqueued\n";

    uint64_t dev_ptr2_vh = cu_mem_alloc(pipe, static_cast<uint64_t>(buf_size));
    std::cout << "cuMemAlloc(2): VH(" << zlink::virtual_handle_id(dev_ptr2_vh) << ") — enqueued\n";

    uint64_t stream_vh = cu_stream_create(pipe, 0);
    std::cout << "cuStreamCreate: VH(" << zlink::virtual_handle_id(stream_vh) << ") — enqueued\n";

    std::vector<float> host_data(buf_size / sizeof(float));
    for (size_t i = 0; i < host_data.size(); i++) {
        host_data[i] = static_cast<float>(i) * 1.5f;
    }

    cu_memcpy_htod(pipe, dev_ptr_vh, host_data.data(), static_cast<uint64_t>(buf_size));
    std::cout << "cuMemcpyHtoD(VH(" << zlink::virtual_handle_id(dev_ptr_vh) << ")): enqueued\n";

    cu_memcpy_dtod(pipe, dev_ptr2_vh, dev_ptr_vh, static_cast<uint64_t>(buf_size));
    std::cout << "cuMemcpyDtoD: enqueued\n";

    cu_ctx_synchronize(pipe);
    std::cout << "cuCtxSynchronize: enqueued\n";

    // cuMemcpyDtoH — READBACK: flush pipeline + get data back
    std::cout << "\n=== cuMemcpyDtoH: READBACK — flushing pipeline ===" << std::endl;

    std::vector<float> readback(buf_size / sizeof(float), 0.0f);
    res = cu_memcpy_dtoh(pipe, readback.data(), dev_ptr2_vh, static_cast<uint64_t>(buf_size));
    std::cout << "cuMemcpyDtoH result: " << res
              << (res == 0 ? " (SUCCESS)" : " (FAILED)") << "\n";

    if (res == 0) {
        std::cout << "\n=== Data Verification (DtoD round-trip) ===" << std::endl;
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
            std::cout << "  VH pipeline verified: alloc+alloc+stream+HtoD+DtoD+sync+DtoH in 1 round-trip!\n";
        } else {
            std::cout << "  " << mismatches << " mismatches out of " << host_data.size() << " values\n";
        }
    }

    // ── Phase 3: Event pipeline ──────────────────────────────────────
    std::cout << "\n=== Phase 3: Event Pipeline Test ===" << std::endl;

    uint64_t start_event_vh = cu_event_create(pipe, 0);
    uint64_t end_event_vh = cu_event_create(pipe, 0);
    std::cout << "Events created: start=VH(" << zlink::virtual_handle_id(start_event_vh)
              << "), end=VH(" << zlink::virtual_handle_id(end_event_vh) << ")\n";

    cu_event_record(pipe, start_event_vh, stream_vh);
    cu_event_record(pipe, end_event_vh, stream_vh);
    cu_stream_synchronize(pipe, stream_vh);
    cu_event_synchronize(pipe, end_event_vh);

    auto event_results = pipe.flush();
    std::cout << "Event pipeline flush: " << event_results.size() << " results\n";

    float elapsed_ms = 0.0f;
    res = cu_event_elapsed_time(pipe, start_event_vh, end_event_vh, elapsed_ms);
    std::cout << "EventElapsedTime: " << elapsed_ms << " ms (result=" << res << ")\n";

    cu_event_destroy(pipe, start_event_vh);
    cu_event_destroy(pipe, end_event_vh);

    // ── Cleanup ──────────────────────────────────────────────────────
    cu_mem_free(pipe, dev_ptr_vh);
    cu_mem_free(pipe, dev_ptr2_vh);
    cu_stream_destroy(pipe, stream_vh);
    cu_ctx_destroy(pipe, ctx_vh);
    std::cout << "\nCleanup enqueued\n";

    auto final_results = pipe.flush();
    std::cout << "Final flush: " << final_results.size() << " deferred calls\n";

    // ── Benchmark ────────────────────────────────────────────────────
    std::cout << "\n=== Benchmark: Virtual Handles vs Barriers ===" << std::endl;

    auto t_vh_start = std::chrono::high_resolution_clock::now();
    {
        pipe_t vh_pipe(*tp);
        uint64_t vh_ctx = cu_ctx_create(vh_pipe, 0u, 0);
        uint64_t vh_ptr = cu_mem_alloc(vh_pipe, buf_size);
        uint64_t vh_ptr2 = cu_mem_alloc(vh_pipe, buf_size);
        uint64_t vh_stream = cu_stream_create(vh_pipe, 0);

        std::vector<float> bench_data(buf_size / sizeof(float), 3.14f);
        cu_memcpy_htod(vh_pipe, vh_ptr, bench_data.data(), buf_size);
        cu_memcpy_dtod(vh_pipe, vh_ptr2, vh_ptr, buf_size);
        cu_ctx_synchronize(vh_pipe);

        std::vector<float> bench_read(buf_size / sizeof(float), 0.0f);
        cu_memcpy_dtoh(vh_pipe, bench_read.data(), vh_ptr2, buf_size);

        cu_mem_free(vh_pipe, vh_ptr);
        cu_mem_free(vh_pipe, vh_ptr2);
        cu_stream_destroy(vh_pipe, vh_stream);
        cu_ctx_destroy(vh_pipe, vh_ctx);
        vh_pipe.flush();
    }
    auto t_vh_end = std::chrono::high_resolution_clock::now();
    auto vh_us = std::chrono::duration_cast<std::chrono::microseconds>(t_vh_end - t_vh_start).count();

    auto t_barrier_start = std::chrono::high_resolution_clock::now();
    {
        pipe_t bar_pipe(*tp);
        auto r1 = bar_pipe.call_barrier<5>(0u, 0);
        uint64_t real_ctx = r1.ctx_handle;
        auto r2 = bar_pipe.call_barrier<10>(static_cast<uint64_t>(buf_size));
        uint64_t real_ptr = r2.dev_ptr;
        auto r3 = bar_pipe.call_barrier<10>(static_cast<uint64_t>(buf_size));
        uint64_t real_ptr2 = r3.dev_ptr;

        std::vector<float> bench_data(buf_size / sizeof(float), 3.14f);
        bar_pipe.enqueue_with_sync<18>(
            bench_data.data(), buf_size,
            real_ptr,
            reinterpret_cast<std::uint64_t>(bench_data.data()),
            buf_size);
        bar_pipe.enqueue<20>(real_ptr2, real_ptr, buf_size);
        bar_pipe.enqueue<9>();

        std::vector<float> bench_read(buf_size / sizeof(float), 0.0f);
        bar_pipe.call_readback_with_sync_read<19>(
            bench_read.data(), buf_size,
            reinterpret_cast<std::uint64_t>(bench_read.data()),
            real_ptr2,
            buf_size);

        bar_pipe.enqueue<12>(real_ptr);
        bar_pipe.enqueue<12>(real_ptr2);
        bar_pipe.enqueue<6>(real_ctx);
        bar_pipe.flush();
    }
    auto t_barrier_end = std::chrono::high_resolution_clock::now();
    auto barrier_us = std::chrono::duration_cast<std::chrono::microseconds>(t_barrier_end - t_barrier_start).count();

    std::cout << "  Virtual handles (1-2 round-trips):  " << vh_us << " us\n"
              << "  Barrier style (4+ round-trips):     " << barrier_us << " us\n"
              << "  Speedup: " << (barrier_us > 0 ? static_cast<double>(barrier_us) / vh_us : 0.0) << "x\n";

    tp->close();
    return 0;
}
