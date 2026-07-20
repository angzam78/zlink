// zlink/examples/cuda/cuda_shim_v2.cpp
//
// LD_PRELOAD shim V2 — with shared memory plane integration.
//
// KEY DIFFERENCE from V1:
//   V1: Tries to pass host pointers as RPC arguments (broken)
//   V2: Registers host memory with shared_mem_plane, sends region IDs
//
// The flow for a function like cuMemcpyHtoD(dst, src, n):
//
//   1. Application calls cuMemcpyHtoD(dst, src=0x7fff1234, 1024)
//   2. Shim intercepts the call
//   3. Shim registers the memory region [0x7fff1234, 1024] with mem_plane
//   4. Shim sends RPC call: cuMemcpyHtoD(dst, src, 1024, region_id=42)
//   5. Server receives the call
//   6. Server calls translate_to_server(0x7fff1234, 1024, 42)
//      → mem_plane fetches the data from client on-demand
//      → returns a server-local pointer
//   7. Server calls real cuMemcpyHtoD(dst, local_src, 1024)
//   8. Function works normally!
//
// For output buffers like cuMemcpyDtoH(dst=0x7fff5678, src, 1024):
//   1. Application calls cuMemcpyDtoH(dst, src, 1024)
//   2. Shim registers [0x7fff5678, 1024] as writable
//   3. Server translates the pointer, real function writes to local cache
//   4. Server marks region dirty, flushes back to client
//   5. Client's memory now contains the result!

#include "cuda_api.hpp"
#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/ptr_map.hpp>
#include <zlink/shared_mem.hpp>
#include <zlink/config.hpp>

#include <zpp_bits.h>

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <mutex>

// ── Global state ───────────────────────────────────────────────────────
static std::unique_ptr<zlink::transport> g_transport;
static zlink::ptr_map            g_ptr_map;
static zlink::shared_mem_plane   g_mem_plane;     // NEW: shared memory plane
static std::mutex                g_rpc_mutex;
static bool                      g_initialized = false;

// ── Extended RPC call that includes region information ─────────────────
// In the full implementation, we'd extend the RPC protocol to carry
// region IDs alongside pointer arguments. For now, we handle it
// in the shim layer.

// Register a host memory region with the shared memory plane
// and return its region ID. The server will use this ID to
// access the memory on-demand.
static zlink::mem_region register_host_memory(void* host_ptr, std::size_t size,
                                               bool writable = true) {
    auto addr = reinterpret_cast<std::uintptr_t>(host_ptr);

    // Check if we already have this region registered
    auto existing = g_mem_plane.find_region(addr, size);
    if (existing) {
        auto [region_id, offset] = *existing;
        return {
            .client_addr = addr,
            .size        = size,
            .region_id   = region_id,
            .writable    = writable,
        };
    }

    // Register a new region
    auto id = g_mem_plane.register_region(addr, size, writable);
    return {
        .client_addr = addr,
        .size        = size,
        .region_id   = id,
        .writable    = writable,
    };
}

// ── Initialize the shim ────────────────────────────────────────────────
static void init_shim() {
    if (g_initialized) return;

    const char* server_env = std::getenv("ZLINK_SERVER");
    if (!server_env) {
        std::cerr << "zlink shim v2: ZLINK_SERVER not set\n";
        return;
    }

    std::string host = server_env;
    std::uint16_t port = zlink::default_port;
    auto colon = host.find(':');
    if (colon != std::string::npos) {
        port = static_cast<std::uint16_t>(std::stoi(host.substr(colon + 1)));
        host = host.substr(0, colon);
    }

    g_transport = zlink::make_transport(zlink::transport_kind::tcp);
    auto ec = g_transport->connect(host, port);
    if (ec) {
        std::cerr << "zlink shim v2: Failed to connect: " << ec.message() << "\n";
        return;
    }

    g_ptr_map.set_shadow_region(0x7f0000000000ULL, 0x100000000ULL);
    g_initialized = true;
    std::cerr << "zlink shim v2: Connected to " << host << ":" << port
              << " (shared memory plane enabled)\n";
}

// ── RPC helper ─────────────────────────────────────────────────────────
template<std::size_t FuncIndex, typename... Args>
static auto cuda_rpc_call(Args&&... args) {
    std::lock_guard lock(g_rpc_mutex);

    using namespace zpp::bits;
    auto [data, in, out] = data_in_out();
    cuda_rpc::client client{in, out};

    client.template request<FuncIndex>(std::forward<Args>(args)...).or_throw();

    zlink::frame req_frame;
    req_frame.call_id = 1;
    req_frame.type = zlink::frame_type::request;
    req_frame.payload.assign(data.begin(), data.end());

    auto ec = g_transport->send(req_frame);
    if (ec) throw std::system_error(ec);

    zlink::frame resp_frame;
    ec = g_transport->receive(resp_frame);
    if (ec) throw std::system_error(ec);

    data.clear();
    data.assign(resp_frame.payload.begin(), resp_frame.payload.end());

    return client.template response<FuncIndex>().or_throw();
}

// ── Intercepted CUDA driver API functions ──────────────────────────────
// V2 versions that register host memory with the shared memory plane
// before making RPC calls.

extern "C" {

CUresult cuInit(unsigned int Flags) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return cuda_rpc_call<cuda_func::cuInit>(Flags);
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return cuda_rpc_call<cuda_func::cuDeviceGet>(device, ordinal);
}

CUresult cuDeviceGetCount(int* count) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return cuda_rpc_call<cuda_func::cuDeviceGetCount>(count);
}

// ── cuMemAlloc: server returns a remote device pointer ─────────────────
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    CUdeviceptr remote_ptr = 0;
    CUresult r = cuda_rpc_call<cuda_func::cuMemAlloc>(&remote_ptr, bytesize);
    if (r == CUDA_SUCCESS) {
        *dptr = g_ptr_map.map(remote_ptr);
    }
    return r;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto remote = g_ptr_map.to_remote(dptr);
    if (!remote) return CUDA_ERROR_INVALID_DEVICE_POINTER;
    g_ptr_map.unmap(dptr);
    return cuda_rpc_call<cuda_func::cuMemFree>(*remote);
}

// ── cuMemcpyHtoD: HOST → DEVICE ───────────────────────────────────────
// This is the critical function that demonstrates the shared memory plane.
//
// The application passes srcHost pointing to its own memory.
// We register that memory with the shared_mem_plane so the server
// can access it on-demand, then make the RPC call.
//
// The server will call translate_to_server(srcHost) which triggers
// a ReadAt from the client's backend → the data is fetched transparently.

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    // Translate device pointer
    auto remote_dst = g_ptr_map.to_remote(dstDevice);
    if (!remote_dst) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    // Register the host memory region with the shared memory plane.
    // This makes the data at srcHost accessible to the server via ReadAt.
    auto region = register_host_memory(
        const_cast<void*>(srcHost), ByteCount, false /* read-only */);

    // Now make the RPC call. The server will use the region_id to
    // access our memory on-demand via the shared memory plane.
    //
    // In the full implementation, we'd extend the RPC protocol to
    // carry region metadata. For now, the server looks up the region
    // by the client address.
    return cuda_rpc_call<cuda_func::cuMemcpyHtoD>(*remote_dst, srcHost, ByteCount);
}

// ── cuMemcpyDtoH: DEVICE → HOST ───────────────────────────────────────
// The application passes a dstHost buffer where the result should go.
// We register it as WRITABLE so the server can write into it.

CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto remote_src = g_ptr_map.to_remote(srcDevice);
    if (!remote_src) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    // Register the output buffer as WRITABLE.
    // The server will write into it and mark it dirty.
    auto region = register_host_memory(dstHost, ByteCount, true /* writable */);

    // Make the RPC call
    CUresult r = cuda_rpc_call<cuda_func::cuMemcpyDtoH>(dstHost, *remote_src, ByteCount);

    // After the call, the server has written to the shared memory plane.
    // The data is now in our local cache. But we need to make sure
    // it's written back to the application's actual memory.
    //
    // In the full implementation, the server would flush dirty regions
    // back to the client, and the client-side backend (memory_backend)
    // would write directly to the application's memory.
    //
    // For now, we do an explicit pull:
    if (r == CUDA_SUCCESS) {
        g_mem_plane.pull(region.region_id, 0,
            std::span<std::byte>(reinterpret_cast<std::byte*>(dstHost), ByteCount));
    }

    return r;
}

// ── cuModuleLoad: passes a filename string ─────────────────────────────
// Strings are just small buffers — register them with the mem plane too.

CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    // Register the string as a read-only region
    size_t fname_len = std::strlen(fname) + 1; // include null terminator
    auto region = register_host_memory(
        const_cast<char*>(fname), fname_len, false);

    CUmodule remote_mod = 0;
    CUresult r = cuda_rpc_call<cuda_func::cuModuleLoad>(&remote_mod,
                    reinterpret_cast<std::uintptr_t>(fname));
    if (r == CUDA_SUCCESS) {
        *module = g_ptr_map.map(remote_mod);
    }
    return r;
}

// ── cuLaunchKernel: array of pointers (deep chase needed) ──────────────
// kernelParams is an array of void* pointers, each of which may point
// to host or device memory. This requires "deep pointer chasing":
//   1. Register the kernelParams array itself as a region
//   2. For each pointer in the array, register the data it points to
//   3. Send all region IDs to the server
//   4. Server translates the array and each pointed-to buffer

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto func_remote = g_ptr_map.to_remote(f);
    auto stream_remote = g_ptr_map.to_remote(hStream);
    if (!func_remote) return CUDA_ERROR_INVALID_VALUE;

    // Register the kernelParams array itself
    if (kernelParams) {
        // Count the number of parameters (we need to know the kernel signature)
        // In practice, we'd get this from the CUDA function attributes
        // For now, we just register the raw pointer array
        auto region = register_host_memory(kernelParams,
            sizeof(void*) * gridDimX, // approximate
            false);
    }

    return cuda_rpc_call<cuda_func::cuLaunchKernel>(
        *func_remote,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, stream_remote.value_or(0),
        kernelParams, extra);
}

// ── Context and stream management ──────────────────────────────────────
CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    CUcontext remote_ctx = 0;
    CUresult r = cuda_rpc_call<cuda_func::cuCtxCreate>(&remote_ctx, flags, dev);
    if (r == CUDA_SUCCESS) {
        *pctx = g_ptr_map.map(remote_ctx);
    }
    return r;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    auto remote = g_ptr_map.to_remote(ctx);
    if (!remote) return CUDA_ERROR_INVALID_CONTEXT;
    return cuda_rpc_call<cuda_func::cuCtxSetCurrent>(*remote);
}

CUresult cuCtxSynchronize() {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    return cuda_rpc_call<cuda_func::cuCtxSynchronize>();
}

CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    CUstream remote_stream = 0;
    CUresult r = cuda_rpc_call<cuda_func::cuStreamCreate>(&remote_stream, Flags);
    if (r == CUDA_SUCCESS) {
        *phStream = g_ptr_map.map(remote_stream);
    }
    return r;
}

} // extern "C"

// ── Library constructor/destructor ─────────────────────────────────────
__attribute__((constructor))
static void shim_init() {
    std::cerr << "zlink CUDA shim v2 loaded (shared memory plane enabled)\n"
              << "  ZLINK_SERVER=" << (std::getenv("ZLINK_SERVER") ?: "not set") << "\n";
}

__attribute__((destructor))
static void shim_fini() {
    if (g_transport && g_transport->is_connected()) {
        g_transport->close();
        std::cerr << "zlink CUDA shim v2: disconnected\n";
    }
    g_mem_plane.clear();
}
