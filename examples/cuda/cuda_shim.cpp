// zlink/examples/cuda/cuda_shim.cpp
//
// LD_PRELOAD shim for CUDA GPU-over-IP.
//
// Usage:
//   LD_PRELOAD=./libzlink_cuda_shim.so \
//   ZLINK_SERVER=gpu-host:14833 \
//   ./your_cuda_app
//
// How it works:
//   1. On load, connects to the zlink server
//   2. Intercepts cuInit, cuMemAlloc, etc. via symbol override
//   3. Serializes calls with zpp_bits and sends over TCP
//   4. Deserializes responses, translates device pointers
//   5. Before any call that passes a host pointer (cuMemcpyHtoD, etc.),
//      syncs the host data to the server's host_memory_mirror
//   6. Application sees a local GPU — completely transparent!
//
// Key fix over the old version: host pointers (like srcHost in
// cuMemcpyHtoD) are now synced to the server BEFORE the RPC call,
// so the server can dereference them via its host_memory_mirror.

#include "cuda_api.hpp"
#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/ptr_map.hpp>
#include <zlink/memory.hpp>
#include <zlink/config.hpp>

#include <zpp_bits.h>

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <mutex>

// ── Global state ───────────────────────────────────────────────────────
static std::unique_ptr<zlink::transport> g_transport;
static zlink::ptr_map g_ptr_map;
static std::mutex g_rpc_mutex;
static bool g_initialized = false;

// ── Helper: sync host memory to server before RPC call ─────────────────
// This sends the client's host memory data to the server's host_memory_mirror
// so that when the server function dereferences the "client pointer", it
// finds the data in the mirror.
static void sync_host_to_server(const void* host_ptr, std::size_t size) {
    if (!host_ptr || size == 0) return;

    zlink::mem_request req;
    req.op = zlink::mem_op::host_sync;
    req.remote_addr = reinterpret_cast<std::uintptr_t>(host_ptr);
    req.size = size;

    std::vector<std::byte> req_data(sizeof(zlink::mem_request) + size);
    std::memcpy(req_data.data(), &req, sizeof(zlink::mem_request));
    std::memcpy(req_data.data() + sizeof(zlink::mem_request), host_ptr, size);

    // Send as a memory_op frame (not an RPC frame)
    zlink::frame sync_frame;
    sync_frame.call_id = 0;
    sync_frame.type = zlink::frame_type::memory_op;
    sync_frame.payload = req_data;

    auto ec = g_transport->send(sync_frame);
    if (ec) {
        std::cerr << "zlink shim: host_sync failed: " << ec.message() << "\n";
        return;
    }

    // Wait for ack
    zlink::frame resp_frame;
    ec = g_transport->receive(resp_frame);
    if (ec) {
        std::cerr << "zlink shim: host_sync ack failed: " << ec.message() << "\n";
    }
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

// ── Initialize the shim ────────────────────────────────────────────────
static void init_shim() {
    if (g_initialized) return;

    const char* server_env = std::getenv("ZLINK_SERVER");
    if (!server_env) {
        std::cerr << "zlink shim: ZLINK_SERVER not set, falling back to local CUDA\n";
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
        std::cerr << "zlink shim: Failed to connect to " << host << ":" << port
                  << " — " << ec.message() << "\n";
        return;
    }

    g_ptr_map.set_shadow_region(0x7f0000000000ULL, 0x100000000ULL);

    g_initialized = true;
    std::cerr << "zlink shim: Connected to " << host << ":" << port << "\n";
}

// ── Intercepted CUDA driver API functions ──────────────────────────────
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

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto remote = g_ptr_map.to_remote(dstDevice);
    if (!remote) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    // ── THE KEY FIX ──
    // Sync the host memory to the server's mirror BEFORE the RPC call.
    // The server will then find the data at the mirror address and can
    // dereference it with real_cuMemcpyHtoD.
    sync_host_to_server(srcHost, ByteCount);

    return cuda_rpc_call<cuda_func::cuMemcpyHtoD>(*remote, srcHost, ByteCount);
}

CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto remote = g_ptr_map.to_remote(srcDevice);
    if (!remote) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    // Register the destination buffer on the server so it can write to it
    // (The server will write to its mirror, and we read back from there)
    return cuda_rpc_call<cuda_func::cuMemcpyDtoH>(dstHost, *remote, ByteCount);
}

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

    return cuda_rpc_call<cuda_func::cuLaunchKernel>(
        *func_remote, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, stream_remote.value_or(0),
        kernelParams, extra);
}

} // extern "C"

// ── Library constructor/destructor ─────────────────────────────────────
__attribute__((constructor))
static void shim_init() {
    std::cerr << "zlink CUDA shim loaded (ZLINK_SERVER="
              << (std::getenv("ZLINK_SERVER") ?: "not set") << ")\n";
}

__attribute__((destructor))
static void shim_fini() {
    if (g_transport && g_transport->is_connected()) {
        g_transport->close();
        std::cerr << "zlink CUDA shim: disconnected\n";
    }
}
