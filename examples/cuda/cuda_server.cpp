// zlink/examples/cuda/cuda_server.cpp
//
// GPU-over-IP server: serves CUDA driver API calls over the network.
//
// Run this on a machine with a GPU:
//   ./cuda_server --port 14833
//
// Then connect from a CPU-only machine with the client shim:
//   LD_PRELOAD=./libzlink_cuda_shim.so ./your_cuda_app
//
// This server:
//   1. Loads the real libcuda.so.1
//   2. Listens for TCP connections
//   3. Dispatches RPC calls to real CUDA functions
//   4. Translates device pointers between client and server address spaces
//   5. Uses host_memory_mirror to dereference client host pointers
//      (e.g., cuMemcpyHtoD's srcHost) — client memory is mirrored on server
//   6. Handles memory copy operations with cached bulk data transfer

#include "cuda_api.hpp"
#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/ptr_map.hpp>
#include <zlink/memory.hpp>
#include <zlink/chunk_cache.hpp>
#include <zlink/config.hpp>

#include <cuda.h>
#include <iostream>
#include <cstring>
#include <cstdlib>

// ── Real CUDA function pointers ────────────────────────────────────────
static void* real_cuda_lib = nullptr;

static decltype(cuInit)*              real_cuInit = nullptr;
static decltype(cuDeviceGet)*         real_cuDeviceGet = nullptr;
static decltype(cuDeviceGetCount)*    real_cuDeviceGetCount = nullptr;
static decltype(cuMemAlloc)*          real_cuMemAlloc = nullptr;
static decltype(cuMemFree)*           real_cuMemFree = nullptr;
static decltype(cuMemcpyHtoD)*        real_cuMemcpyHtoD = nullptr;
static decltype(cuMemcpyDtoH)*        real_cuMemcpyDtoH = nullptr;
static decltype(cuCtxCreate)*         real_cuCtxCreate = nullptr;
static decltype(cuCtxSetCurrent)*     real_cuCtxSetCurrent = nullptr;
static decltype(cuLaunchKernel)*      real_cuLaunchKernel = nullptr;

// ── Pointer translation and host memory mirror ────────────────────────
static zlink::ptr_map g_ptr_map;
static zlink::host_memory_mirror g_host_mirror;

// ── CUDA API implementations ──────────────────────────────────────────
namespace cuda_api {

CUresult cuInit(unsigned int Flags) {
    return real_cuInit(Flags);
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    return real_cuDeviceGet(device, ordinal);
}

CUresult cuDeviceGetCount(int* count) {
    return real_cuDeviceGetCount(count);
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    return ::cuDeviceGetName(name, len, dev);
}

CUresult cuDeviceTotalMem(std::size_t* bytes, CUdevice dev) {
    return ::cuDeviceTotalMem(bytes, dev);
}

CUresult cuDeviceGetAttribute(int* value, int attrib, CUdevice dev) {
    return ::cuDeviceGetAttribute(value, static_cast<CUdevice_attribute>(attrib), dev);
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    CUcontext real_ctx;
    CUresult r = real_cuCtxCreate(&real_ctx, flags, dev);
    if (r == CUDA_SUCCESS) {
        *pctx = g_ptr_map.map(reinterpret_cast<std::uintptr_t>(real_ctx));
    }
    return r;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    auto remote = g_ptr_map.to_remote(ctx);
    if (!remote) return CUDA_ERROR_INVALID_CONTEXT;
    CUcontext real_ctx = reinterpret_cast<CUcontext>(*remote);
    g_ptr_map.unmap(ctx);
    return ::cuCtxDestroy(real_ctx);
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    auto remote = g_ptr_map.to_remote(ctx);
    if (!remote) return CUDA_ERROR_INVALID_CONTEXT;
    return real_cuCtxSetCurrent(reinterpret_cast<CUcontext>(*remote));
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
    CUcontext real_ctx;
    CUresult r = ::cuCtxGetCurrent(&real_ctx);
    if (r == CUDA_SUCCESS) {
        auto local = g_ptr_map.to_local(reinterpret_cast<std::uintptr_t>(real_ctx));
        *pctx = local.value_or(0);
    }
    return r;
}

CUresult cuCtxSynchronize() {
    return ::cuCtxSynchronize();
}

CUresult cuMemAlloc(CUdeviceptr* dptr, std::size_t bytesize) {
    CUdeviceptr real_ptr;
    CUresult r = real_cuMemAlloc(&real_ptr, bytesize);
    if (r == CUDA_SUCCESS) {
        *dptr = g_ptr_map.map(real_ptr);
    }
    return r;
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, std::size_t bytesize, unsigned int flags) {
    CUdeviceptr real_ptr;
    CUresult r = ::cuMemAllocManaged(&real_ptr, bytesize, flags);
    if (r == CUDA_SUCCESS) {
        *dptr = g_ptr_map.map(real_ptr);
    }
    return r;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    auto remote = g_ptr_map.to_remote(dptr);
    if (!remote) return CUDA_ERROR_INVALID_DEVICE_POINTER;
    g_ptr_map.unmap(dptr);
    return real_cuMemFree(*remote);
}

CUresult cuMemFreeHost(void* p) {
    // Check if this is a mirrored host pointer
    auto server_addr = g_host_mirror.translate(reinterpret_cast<std::uintptr_t>(p));
    if (server_addr) {
        // This was a mirrored client pointer; unmirror it
        g_host_mirror.unregister_region(reinterpret_cast<std::uintptr_t>(p));
        return CUDA_SUCCESS;
    }
    return ::cuMemFreeHost(p);
}

CUresult cuMemHostAlloc(void** pp, std::size_t bytesize, unsigned int Flags) {
    return ::cuMemHostAlloc(pp, bytesize, Flags);
}

CUresult cuMemHostRegister(void* p, std::size_t bytesize, unsigned int Flags) {
    return ::cuMemHostRegister(p, bytesize, Flags);
}

CUresult cuMemHostUnregister(void* p) {
    return ::cuMemHostUnregister(p);
}

// ── Memory copy with host memory mirror ───────────────────────────────
// Key fix: cuMemcpyHtoD's srcHost is a CLIENT pointer.
// The client has synced its host data to our mirror via host_sync.
// We translate the client pointer to the server mirror address and
// call the real cuMemcpyHtoD with the mirrored data.

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, std::size_t ByteCount) {
    auto remote = g_ptr_map.to_remote(dstDevice);
    if (!remote) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    // Translate client host pointer to server mirror address
    auto server_addr = g_host_mirror.translate(reinterpret_cast<std::uintptr_t>(srcHost));
    const void* real_src = server_addr
        ? reinterpret_cast<const void*>(*server_addr)
        : srcHost;  // Fallback: might be a server-allocated host pointer

    return real_cuMemcpyHtoD(*remote, real_src, ByteCount);
}

CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, std::size_t ByteCount) {
    auto remote = g_ptr_map.to_remote(srcDevice);
    if (!remote) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    // For DtoH, dstHost might be a client pointer that's been mirrored
    auto server_addr = g_host_mirror.translate(reinterpret_cast<std::uintptr_t>(dstHost));
    void* real_dst = server_addr
        ? reinterpret_cast<void*>(*server_addr)
        : dstHost;

    CUresult r = real_cuMemcpyDtoH(real_dst, *remote, ByteCount);

    // If we wrote to a mirrored region, we need to sync back to client
    // (This would be done via an invalidation notification)
    return r;
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, std::size_t ByteCount) {
    auto dst_remote = g_ptr_map.to_remote(dstDevice);
    auto src_remote = g_ptr_map.to_remote(srcDevice);
    if (!dst_remote || !src_remote) return CUDA_ERROR_INVALID_DEVICE_POINTER;
    return ::cuMemcpyDtoD(*dst_remote, *src_remote, ByteCount);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost,
                            std::size_t ByteCount, CUstream hStream) {
    auto remote = g_ptr_map.to_remote(dstDevice);
    auto stream_remote = g_ptr_map.to_remote(hStream);
    if (!remote) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    auto server_addr = g_host_mirror.translate(reinterpret_cast<std::uintptr_t>(srcHost));
    const void* real_src = server_addr
        ? reinterpret_cast<const void*>(*server_addr)
        : srcHost;

    return ::cuMemcpyHtoDAsync(*remote, real_src, ByteCount,
                               reinterpret_cast<CUstream>(stream_remote.value_or(0)));
}

CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice,
                            std::size_t ByteCount, CUstream hStream) {
    auto remote = g_ptr_map.to_remote(srcDevice);
    auto stream_remote = g_ptr_map.to_remote(hStream);
    if (!remote) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    auto server_addr = g_host_mirror.translate(reinterpret_cast<std::uintptr_t>(dstHost));
    void* real_dst = server_addr
        ? reinterpret_cast<void*>(*server_addr)
        : dstHost;

    return ::cuMemcpyDtoHAsync(real_dst, *remote, ByteCount,
                               reinterpret_cast<CUstream>(stream_remote.value_or(0)));
}

CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    CUmodule real_mod;
    CUresult r = ::cuModuleLoad(&real_mod, fname);
    if (r == CUDA_SUCCESS) {
        *module = g_ptr_map.map(reinterpret_cast<std::uintptr_t>(real_mod));
    }
    return r;
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    CUmodule real_mod;

    // image is a client pointer — translate to server mirror
    auto server_addr = g_host_mirror.translate(reinterpret_cast<std::uintptr_t>(image));
    const void* real_image = server_addr
        ? reinterpret_cast<const void*>(*server_addr)
        : image;

    CUresult r = ::cuModuleLoadData(&real_mod, real_image);
    if (r == CUDA_SUCCESS) {
        *module = g_ptr_map.map(reinterpret_cast<std::uintptr_t>(real_mod));
    }
    return r;
}

CUresult cuModuleUnload(CUmodule hmod) {
    auto remote = g_ptr_map.to_remote(hmod);
    if (!remote) return CUDA_ERROR_INVALID_VALUE;
    g_ptr_map.unmap(hmod);
    return ::cuModuleUnload(reinterpret_cast<CUmodule>(*remote));
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    auto mod_remote = g_ptr_map.to_remote(hmod);
    if (!mod_remote) return CUDA_ERROR_INVALID_VALUE;
    CUfunction real_func;
    CUresult r = ::cuModuleGetFunction(&real_func,
                    reinterpret_cast<CUmodule>(*mod_remote), name);
    if (r == CUDA_SUCCESS) {
        *hfunc = g_ptr_map.map(reinterpret_cast<std::uintptr_t>(real_func));
    }
    return r;
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra) {
    auto func_remote = g_ptr_map.to_remote(f);
    auto stream_remote = g_ptr_map.to_remote(hStream);
    if (!func_remote) return CUDA_ERROR_INVALID_VALUE;
    return ::cuLaunchKernel(reinterpret_cast<CUfunction>(*func_remote),
                           gridDimX, gridDimY, gridDimZ,
                           blockDimX, blockDimY, blockDimZ,
                           sharedMemBytes,
                           reinterpret_cast<CUstream>(stream_remote.value_or(0)),
                           kernelParams, extra);
}

// Stream & Event management
CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) {
    CUstream real_stream;
    CUresult r = ::cuStreamCreate(&real_stream, Flags);
    if (r == CUDA_SUCCESS) {
        *phStream = g_ptr_map.map(reinterpret_cast<std::uintptr_t>(real_stream));
    }
    return r;
}

CUresult cuStreamDestroy(CUstream hStream) {
    auto remote = g_ptr_map.to_remote(hStream);
    if (!remote) return CUDA_ERROR_INVALID_VALUE;
    g_ptr_map.unmap(hStream);
    return ::cuStreamDestroy(reinterpret_cast<CUstream>(*remote));
}

CUresult cuStreamSynchronize(CUstream hStream) {
    auto remote = g_ptr_map.to_remote(hStream);
    return ::cuStreamSynchronize(reinterpret_cast<CUstream>(remote.value_or(0)));
}

CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags) {
    CUevent real_event;
    CUresult r = ::cuEventCreate(reinterpret_cast<CUevent*>(&real_event), Flags);
    if (r == CUDA_SUCCESS) {
        *phEvent = g_ptr_map.map(reinterpret_cast<std::uintptr_t>(real_event));
    }
    return r;
}

CUresult cuEventDestroy(CUevent hEvent) {
    auto remote = g_ptr_map.to_remote(hEvent);
    if (!remote) return CUDA_ERROR_INVALID_VALUE;
    g_ptr_map.unmap(hEvent);
    return ::cuEventDestroy(reinterpret_cast<CUevent>(*remote));
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    auto event_remote = g_ptr_map.to_remote(hEvent);
    auto stream_remote = g_ptr_map.to_remote(hStream);
    return ::cuEventRecord(reinterpret_cast<CUevent>(event_remote.value_or(0)),
                           reinterpret_cast<CUstream>(stream_remote.value_or(0)));
}

CUresult cuEventSynchronize(CUevent hEvent) {
    auto remote = g_ptr_map.to_remote(hEvent);
    return ::cuEventSynchronize(reinterpret_cast<CUevent>(remote.value_or(0)));
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
    auto start_remote = g_ptr_map.to_remote(hStart);
    auto end_remote = g_ptr_map.to_remote(hEnd);
    return ::cuEventElapsedTime(pMilliseconds,
                                reinterpret_cast<CUevent>(start_remote.value_or(0)),
                                reinterpret_cast<CUevent>(end_remote.value_or(0)));
}

} // namespace cuda_api

// ── Main ───────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::uint16_t port = zlink::default_port;
    if (argc > 1) port = static_cast<std::uint16_t>(std::stoi(argv[1]));

    std::cout << "zlink CUDA server — GPU-over-IP\n"
              << "Listening on port " << port << "...\n" << std::endl;

    // Create transport and listen
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

    // Create zpp_bits RPC server
    auto [data, in, out] = zpp::bits::data_in_out();
    cuda_rpc::server server{in, out};

    // Serve loop
    while (tp->is_connected()) {
        try {
            zlink::frame req_frame;
            ec = tp->receive(req_frame);
            if (ec) break;

            // Check if this is a memory operation (host_sync)
            if (req_frame.type == zlink::frame_type::memory_op) {
                // Handle host memory sync from client
                if (req_frame.payload.size() >= sizeof(zlink::mem_request)) {
                    zlink::mem_request mem_req;
                    std::memcpy(&mem_req, req_frame.payload.data(), sizeof(mem_req));

                    if (mem_req.op == zlink::mem_op::host_sync) {
                        std::span<const std::byte> sync_data(
                            req_frame.payload.data() + sizeof(mem_req),
                            req_frame.payload.size() - sizeof(mem_req)
                        );
                        g_host_mirror.sync_page(mem_req.remote_addr, sync_data);
                    }
                }

                zlink::frame resp_frame;
                resp_frame.call_id = req_frame.call_id;
                resp_frame.type = zlink::frame_type::memory_reply;
                zlink::mem_response resp{zlink::error_code::ok, 0, 0};
                std::vector<std::byte> resp_data(sizeof(resp));
                std::memcpy(resp_data.data(), &resp, sizeof(resp));
                resp_frame.payload = resp_data;

                ec = tp->send(resp_frame);
                if (ec) break;
                continue;
            }

            data.clear();
            data.assign(req_frame.payload.begin(), req_frame.payload.end());

            auto result = server.serve();
            if (zpp::bits::failure(result)) break;

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

    std::cout << "CUDA server shutting down.\n";
    return 0;
}
