// zlink/examples/cuda/cuda_client.cpp — CUDA pipeline client with virtual handles
//
// Uses the SEPARATE-LAYER MODEL:
//   - RPC batching over the RPC control channel (multiplexed_transport)
//   - Memory data over the bulk channel via cached_memory_client
//   - Demand paging + write tracking via write_tracker
//     (uffd WP_ASYNC / uffd WP sync / mprotect+SIGSEGV — auto-selected)
//
// Host buffers live in a shadow mmap region. The write_tracker transparently
// monitors reads (demand-paged fetch from server) and writes (dirty tracking).
// Before enqueuing cuMemcpyHtoD, flush_dirty() pushes only dirty pages.
// After cuMemcpyDtoH, reading the shadow region triggers demand paging.
//
// Run: cuda_client [host [port]]

#include <zlink/transport.hpp>
#include <zlink/multiplexed_transport.hpp>
#include <zlink/config.hpp>
#include <zlink/rpc.hpp>
#include <zlink/memory.hpp>
#include <zlink/cuda_pipeline.hpp>
#include <zlink/virtual_handle.hpp>

#include "cuda_api.hpp"

#include <zpp_bits.h>

#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <cstdint>
#include <cmath>
#include <stdexcept>

#include <sys/mman.h>

using pipe_t = zlink::cuda_pipeline<cuda_gen::cuda_gen_rpc>;

// ── Shadow memory region for demand-paged host buffers ──────────────────
static constexpr std::size_t SHADOW_SIZE = 16 * 1024 * 1024;  // 16 MB

struct memory_layer {
    zlink::rpc_client_base           rpc;
    zlink::cached_memory_client      mem;
    void*                             shadow_base = nullptr;
    std::size_t                       shadow_size = 0;
    std::size_t                       alloc_offset = 0;

    explicit memory_layer(zlink::transport& tp, std::size_t region_size)
        : rpc(tp)
        , mem(rpc, region_size)
        , shadow_size(region_size)
    {
        shadow_base = ::mmap(nullptr, region_size,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS,
                             -1, 0);
        if (shadow_base == MAP_FAILED) {
            throw std::runtime_error("mmap shadow region failed");
        }

#if ZLINK_HAS_USERFAULTFD
        auto ec = mem.enable_demand_paging(
            reinterpret_cast<std::uintptr_t>(shadow_base), region_size);
        if (ec) {
            std::cerr << "[client] demand paging failed: " << ec.message() << "\n";
        } else {
            std::cerr << "[client] demand paging + write tracking enabled\n";
        }
#else
        std::cerr << "[client] userfaultfd not compiled — "
                  << "demand paging disabled\n";
#endif
    }

    ~memory_layer() {
#if ZLINK_HAS_USERFAULTFD
        mem.disable_demand_paging();
#endif
        if (shadow_base && shadow_base != MAP_FAILED) {
            ::munmap(shadow_base, shadow_size);
        }
    }

    void* alloc(std::size_t size) {
        if (alloc_offset + size > shadow_size) return nullptr;
        void* ptr = static_cast<std::byte*>(shadow_base) + alloc_offset;
        alloc_offset += (size + 4095) & ~static_cast<std::size_t>(4095);
        return ptr;
    }

    // Flush dirty pages to the server (pushes via chunk_cache → host_sync)
    void flush() {
        std::size_t n = mem.flush_dirty();
        if (n > 0) {
            std::cerr << "[client] flushed " << n << " dirty pages\n";
        }
    }

    // Invalidate cached pages so the next read fetches fresh data from server
    void invalidate_all() {
        mem.invalidate_all();
    }
};

// ── BARRIER WRAPPERS ───────────────────────────────────────────────────

static int32_t cu_init(pipe_t& pipe, unsigned int flags = 0) {
    auto r = pipe.call_barrier<cuda_gen::func_index::init>(flags);
    return r.result;
}
static int32_t cu_driver_get_version(pipe_t& pipe, int& out_ver) {
    auto r = pipe.call_barrier<cuda_gen::func_index::driver_get_version>();
    out_ver = r.driverVersion;
    return r.result;
}
static int32_t cu_device_get_count(pipe_t& pipe, int& out_count) {
    auto r = pipe.call_barrier<cuda_gen::func_index::device_get_count>();
    out_count = r.count;
    return r.result;
}
static int32_t cu_device_get_name(pipe_t& pipe, int ordinal, std::string& out_name) {
    auto r = pipe.call_barrier<cuda_gen::func_index::device_get_name>(128, ordinal);
    out_name = r.name;
    return r.result;
}
static int32_t cu_device_total_mem(pipe_t& pipe, int ordinal, uint64_t& out_bytes) {
    auto r = pipe.call_barrier<cuda_gen::func_index::device_total_mem>(ordinal);
    out_bytes = r.bytes;
    return r.result;
}
static int32_t cu_device_get_attribute(pipe_t& pipe, int attrib, int dev, int32_t& out_value) {
    auto r = pipe.call_barrier<cuda_gen::func_index::device_get_attribute>(
        static_cast<uint32_t>(attrib), dev);
    out_value = r.value;
    return r.result;
}
static int32_t cu_mem_get_info(pipe_t& pipe, uint64_t& free_bytes, uint64_t& total_bytes) {
    auto r = pipe.call_barrier<cuda_gen::func_index::mem_get_info>();
    free_bytes = r.free_bytes;
    total_bytes = r.total_bytes;
    return r.result;
}

// ── ENQUEUED: handle producers (return virtual handles, no barrier!) ───

static uint64_t cu_ctx_create(pipe_t& pipe, unsigned int flags, int dev) {
    return pipe.enqueue_produces_handle<cuda_gen::func_index::ctx_create>(flags, dev);
}
static uint64_t cu_mem_alloc(pipe_t& pipe, uint64_t size) {
    return pipe.enqueue_produces_handle<cuda_gen::func_index::mem_alloc>(size);
}
static uint64_t cu_mem_alloc_managed(pipe_t& pipe, uint64_t size, unsigned int flags) {
    return pipe.enqueue_produces_handle<cuda_gen::func_index::mem_alloc_managed>(size, flags);
}
static uint64_t cu_stream_create(pipe_t& pipe, unsigned int flags = 0) {
    return pipe.enqueue_produces_handle<cuda_gen::func_index::stream_create>(flags);
}
static uint64_t cu_event_create(pipe_t& pipe, unsigned int flags = 0) {
    return pipe.enqueue_produces_handle<cuda_gen::func_index::event_create>(flags);
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

// cuMemcpyHtoD: flush dirty pages to server mirror, then enqueue RPC.
static void cu_memcpy_htod(pipe_t& pipe, memory_layer& mem,
                            uint64_t dev_vh,
                            const void* host_data, uint64_t byte_count) {
    mem.flush();

    pipe.enqueue<cuda_gen::func_index::memcpy_hto_d>(
        dev_vh,
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host_data)),
        byte_count);
}

// cuMemcpyDtoD: consumes VH(dst) + VH(src)
static void cu_memcpy_dtod(pipe_t& pipe, uint64_t dst_vh, uint64_t src_vh, uint64_t byte_count) {
    pipe.enqueue<cuda_gen::func_index::memcpy_dto_d>(dst_vh, src_vh, byte_count);
}

// ── READBACK: cuMemcpyDtoH (flushes pipeline + demand-paged readback) ───
static int32_t cu_memcpy_dtoh(pipe_t& pipe, memory_layer& mem,
                                void* host_data,
                                uint64_t dev_vh, uint64_t byte_count) {
    mem.invalidate_all();

    auto r = pipe.call_readback<cuda_gen::func_index::memcpy_dto_h>(
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(host_data)),
        dev_vh,
        byte_count);

    // Touch each page to trigger demand paging.
    // The write_tracker's fault handler fetches the page from the server.
    if (r.result == 0) {
        volatile std::byte* p = static_cast<volatile std::byte*>(host_data);
        constexpr std::size_t page_size = 4096;
        for (std::size_t i = 0; i < byte_count; i += page_size) {
            (void)p[i];
        }
        (void)p[byte_count - 1];
    }
    return r.result;
}

// ── BARRIER: cuEventElapsedTime (needs the float value) ─────────────────
static int32_t cu_event_elapsed_time(pipe_t& pipe, uint64_t start_evh, uint64_t end_evh, float& out_ms) {
    auto r = pipe.call_barrier<cuda_gen::func_index::event_elapsed_time>(start_evh, end_evh);
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

    zlink::multiplexed_transport tp;
    std::cout << "Connecting to " << host << ":" << port << "...\n";
    auto ec = tp.connect(host, port);
    if (ec) { std::cerr << "connect failed: " << ec.message() << "\n"; return 1; }
    std::cout << "Connected"
              << (tp.is_multi_port() ? " (multi-port: 3 channels)" : " (single-port fallback)")
              << "\n\n";

    memory_layer mem(tp, SHADOW_SIZE);
    pipe_t pipe(tp);

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

    float* host_data = static_cast<float*>(mem.alloc(buf_size));
    for (size_t i = 0; i < buf_size / sizeof(float); i++) {
        host_data[i] = static_cast<float>(i) * 1.5f;
    }

    cu_memcpy_htod(pipe, mem, dev_ptr_vh, host_data, buf_size);
    std::cout << "cuMemcpyHtoD(VH(" << zlink::virtual_handle_id(dev_ptr_vh)
              << ")): enqueued (dirty pages flushed)\n";

    cu_memcpy_dtod(pipe, dev_ptr2_vh, dev_ptr_vh, buf_size);
    std::cout << "cuMemcpyDtoD: enqueued\n";

    cu_ctx_synchronize(pipe);
    std::cout << "cuCtxSynchronize: enqueued\n";

    std::cout << "\n=== cuMemcpyDtoH: READBACK — flushing pipeline ===\n";
    float* readback = static_cast<float*>(mem.alloc(buf_size));
    res = cu_memcpy_dtoh(pipe, mem, readback, dev_ptr2_vh, buf_size);
    std::cout << "cuMemcpyDtoH result: " << res
              << (res == 0 ? " (SUCCESS)" : " (FAILED)") << "\n";

    if (res == 0) {
        std::cout << "\n=== Data Verification (DtoD round-trip) ===\n";
        bool match = true;
        int mismatches = 0;
        for (size_t i = 0; i < buf_size / sizeof(float); i++) {
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
            std::cout << "  All " << buf_size / sizeof(float) << " values match!\n";
            std::cout << "  VH pipeline verified: alloc+alloc+stream+HtoD+DtoD+sync+DtoH"
                      << " in 1 RPC round-trip!\n";
            std::cout << "  Demand paging + write tracking: WORKING\n";
        } else {
            std::cout << "  " << mismatches << " mismatches out of "
                      << buf_size / sizeof(float) << " values\n";
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

    auto event_results = pipe.flush();
    std::cout << "Event batch flushed: " << event_results.size() << " results\n";

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

    cu_memcpy_htod(pipe, mem, managed_vh, host_data, buf_size);
    cu_ctx_synchronize(pipe);
    std::cout << "cuMemcpyHtoD(managed) + cuCtxSynchronize: enqueued\n";

    float* managed_readback = static_cast<float*>(mem.alloc(buf_size));
    res = cu_memcpy_dtoh(pipe, mem, managed_readback, managed_vh, buf_size);
    std::cout << "cuMemcpyDtoH(managed) result: " << res
              << (res == 0 ? " (SUCCESS)" : " (FAILED)") << "\n";

    if (res == 0) {
        bool match = true;
        for (size_t i = 0; i < buf_size / sizeof(float); i++) {
            if (std::abs(managed_readback[i] - host_data[i]) > 1e-6f) { match = false; break; }
        }
        std::cout << "  Managed memory round-trip: "
                  << (match ? "VERIFIED" : "MISMATCH") << "\n";
    }

    cu_mem_free(pipe, dev_ptr_vh);
    cu_mem_free(pipe, dev_ptr2_vh);
    cu_mem_free(pipe, managed_vh);
    cu_stream_destroy(pipe, stream_vh);
    std::cout << "Cleanup (3×mem_free + stream_destroy): enqueued\n";

    auto final_results = pipe.flush();
    std::cout << "Cleanup batch flushed: " << final_results.size() << " results\n";

    std::cout << "\n=== All tests complete ===\n";
    tp.close();
    return 0;
}
