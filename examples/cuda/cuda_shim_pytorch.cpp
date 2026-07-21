// zlink/examples/cuda/cuda_shim_pytorch.cpp
//
// CUDA Driver API shim for PyTorch GPU workloads over zlink.
//
// This library is built as libcuda.so.1 and placed in LD_LIBRARY_PATH
// on the CPU client machine. When PyTorch calls dlopen("libcuda.so.1"),
// the dynamic linker finds THIS library instead of the real NVIDIA driver.
// All cu* function calls are intercepted, virtual handles are translated,
// and the calls are forwarded via zlink RPC to a remote GPU server
// running cuda_server_v4.
//
// The LD_LIBRARY_PATH approach is superior to LD_PRELOAD because:
//   - Works on machines with NO CUDA installed at all
//   - PyTorch naturally finds "libcuda.so.1" via the dynamic linker
//   - No need for fake library symlinks or dlopen tricks
//   - torch.cuda.is_available() returns True automatically
//
// USAGE (on the CPU client machine):
//   export ZLINK_SERVER=hostname:port
//   export LD_LIBRARY_PATH=/path/to/zlink-cuda-runtime:$LD_LIBRARY_PATH
//   python my_script.py
//
// Or use the runner script:
//   ./run_pytorch_zlink.sh python my_script.py
//
// KEY DESIGN DECISIONS:
//   - PyTorch uses cuDevicePrimaryCtxRetain exclusively (never cuCtxCreate)
//   - PyTorch 2.0+ uses cuGetProcAddress to resolve kernel function pointers
//   - All handle-producing calls register their return values in the handle map
//   - All handle-consuming calls translate via the handle map before RPC
//   - Host memory for memcpy operations is synced to server via host_sync frames

#include "cuda_api_v2.hpp"

#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/ptr_map.hpp>
#include <zlink/memory.hpp>
#include <zlink/config.hpp>
#include <zlink/virtual_handle.hpp>

#include <zpp_bits.h>

// CUDA Driver API type definitions (we define our own to avoid depending
// on CUDA headers — this shim runs on CPU-only machines)
#include <cstdint>
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUstream;
typedef void* CUevent;
typedef void* CUmodule;
typedef void* CUfunction;
typedef std::uint64_t CUdeviceptr;

// CUDA error codes
#define CUDA_SUCCESS                    0
#define CUDA_ERROR_NOT_INITIALIZED      3
#define CUDA_ERROR_INVALID_VALUE        2
#define CUDA_ERROR_INVALID_CONTEXT      7
#define CUDA_ERROR_INVALID_HANDLE       4
#define CUDA_ERROR_INVALID_DEVICE_POINTER 6
#define CUDA_ERROR_OUT_OF_MEMORY        2

// CUDA device attribute type
typedef unsigned int CUdevice_attribute;

// CUDA occupancy callback type (we never call it, just need the type)
typedef void (*cuOccupancyB2DSize)(int);

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <dlfcn.h>

// ── Global state ───────────────────────────────────────────────────────
static std::unique_ptr<zlink::transport> g_transport;
static zlink::ptr_map              g_ptr_map;        // virtual ↔ remote ptr mapping
static zlink::host_memory_mirror   g_host_mirror;     // client-side mirror for readback
static std::mutex                  g_rpc_mutex;
static bool                        g_initialized = false;

// ── Handle mapping ─────────────────────────────────────────────────────
// Maps client-side handles (CUcontext, CUstream, CUevent, CUfunction, CUmodule)
// to server-side real handles. We use a simple unordered_map since the client
// side receives opaque pointers from PyTorch that we must translate.

struct handle_entry {
    std::uint64_t server_handle;  // Real handle on the server
    std::string   type;           // "ctx", "stream", "event", "func", "module", "devptr"
};

static std::unordered_map<std::uint64_t, handle_entry> g_handle_map;
static std::mutex g_handle_mutex;

static void register_handle(std::uint64_t client_handle, std::uint64_t server_handle,
                            const std::string& type) {
    std::lock_guard lock(g_handle_mutex);
    g_handle_map[client_handle] = {server_handle, type};
    std::cerr << "  [shim] register_handle: client=0x" << std::hex << client_handle
              << " -> server=0x" << server_handle << " type=" << type << std::dec << "\n";
}

static std::optional<std::uint64_t> translate_handle(std::uint64_t client_handle) {
    std::lock_guard lock(g_handle_mutex);
    auto it = g_handle_map.find(client_handle);
    if (it != g_handle_map.end()) return it->second.server_handle;
    return std::nullopt;
}

static void unregister_handle(std::uint64_t client_handle) {
    std::lock_guard lock(g_handle_mutex);
    g_handle_map.erase(client_handle);
}

// ── Device pointer mapping ─────────────────────────────────────────────
// Device pointers (CUdeviceptr) need bidirectional mapping:
//   client_ptr (shadow region) ↔ server_ptr (real GPU address)
// We use ptr_map.map() which allocates a shadow-region pointer for a
// given remote pointer, and ptr_map.to_remote() for reverse lookup.

static CUdeviceptr register_devptr(CUdeviceptr server_ptr) {
    // map() returns a local shadow-region pointer for the remote pointer
    CUdeviceptr client_ptr = static_cast<CUdeviceptr>(g_ptr_map.map(server_ptr));
    std::cerr << "  [shim] register_devptr: server=0x" << std::hex << server_ptr
              << " -> client=0x" << client_ptr << std::dec << "\n";
    return client_ptr;
}

static std::optional<CUdeviceptr> translate_devptr(CUdeviceptr client_ptr) {
    auto remote = g_ptr_map.to_remote(client_ptr);
    if (remote) return static_cast<CUdeviceptr>(*remote);
    return std::nullopt;
}

static void unregister_devptr(CUdeviceptr client_ptr) {
    g_ptr_map.unmap(client_ptr);
}

// ── Host memory sync helper ────────────────────────────────────────────
// Before cuMemcpyHtoD, we must sync the host buffer to the server so
// it can access the data. We send a host_sync frame with the data.

static void sync_host_to_server(const void* host_ptr, std::size_t size) {
    if (!g_transport || !g_transport->is_connected()) return;

    zlink::frame sync_frame;
    sync_frame.call_id = 0;
    sync_frame.type = zlink::frame_type::memory_op;

    zlink::mem_request req;
    req.op = zlink::mem_op::host_sync;
    req.remote_addr = reinterpret_cast<std::uintptr_t>(host_ptr);
    req.size = size;

    sync_frame.payload.resize(sizeof(req) + size);
    std::memcpy(sync_frame.payload.data(), &req, sizeof(req));
    std::memcpy(sync_frame.payload.data() + sizeof(req), host_ptr, size);

    auto ec = g_transport->send(sync_frame);
    if (ec) {
        std::cerr << "  [shim] host_sync send error: " << ec.message() << "\n";
        return;
    }

    // Wait for acknowledgement
    zlink::frame resp_frame;
    ec = g_transport->receive(resp_frame);
    if (ec) {
        std::cerr << "  [shim] host_sync recv error: " << ec.message() << "\n";
    }
}

// ── Host memory readback helper ────────────────────────────────────────
// After cuMemcpyDtoH, we read back data from the server's mirror.

static void readback_from_server(void* host_ptr, std::size_t size) {
    if (!g_transport || !g_transport->is_connected()) return;

    zlink::frame read_frame;
    read_frame.call_id = 0;
    read_frame.type = zlink::frame_type::memory_op;

    zlink::mem_request req;
    req.op = zlink::mem_op::host_read;
    req.remote_addr = reinterpret_cast<std::uintptr_t>(host_ptr);
    req.size = size;

    read_frame.payload.resize(sizeof(req));
    std::memcpy(read_frame.payload.data(), &req, sizeof(req));

    auto ec = g_transport->send(read_frame);
    if (ec) {
        std::cerr << "  [shim] host_read send error: " << ec.message() << "\n";
        return;
    }

    zlink::frame resp_frame;
    ec = g_transport->receive(resp_frame);
    if (ec) {
        std::cerr << "  [shim] host_read recv error: " << ec.message() << "\n";
        return;
    }

    if (resp_frame.payload.size() >= sizeof(zlink::mem_response)) {
        zlink::mem_response resp;
        std::memcpy(&resp, resp_frame.payload.data(), sizeof(resp));
        if (resp.size > 0 && resp_frame.payload.size() >= sizeof(resp) + resp.size) {
            std::memcpy(host_ptr, resp_frame.payload.data() + sizeof(resp),
                       static_cast<std::size_t>(resp.size));
        }
    }
}

// ── Initialize the shim ────────────────────────────────────────────────
static void init_shim() {
    if (g_initialized) return;

    const char* server_env = std::getenv("ZLINK_SERVER");
    if (!server_env) {
        std::cerr << "zlink pytorch shim: ZLINK_SERVER not set\n";
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
        std::cerr << "zlink pytorch shim: Failed to connect: " << ec.message() << "\n";
        return;
    }

    g_ptr_map.set_shadow_region(0x7f0000000000ULL, 0x100000000ULL);
    g_initialized = true;
    std::cerr << "zlink pytorch shim: Connected to " << host << ":" << port << "\n";
}

// ── RPC helper ─────────────────────────────────────────────────────────
// We use two separate RPC clients because zpp_bits has a template
// depth issue with 48+ function bindings in a single rpc<> type
// on some compilers. Splitting into two groups works around this.

// Group 1: original 38 functions (indices 0-37)
template<int FuncIndex, typename... Args>
static auto cuda_rpc_call_base(Args&&... args) {
    std::lock_guard lock(g_rpc_mutex);

    using namespace zpp::bits;
    auto [data, in, out] = data_in_out();
    cuda_v2_rpc::client client{in, out};

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

// Group 2: PyTorch-critical additions (indices 38-47)
// These are in cuda_v2_rpc_full but we use the full type only
// for the server side. For the client, we use a separate rpc type
// with just the 10 new functions, renumbered 0-9.
namespace pytorch_rpc {
    using rpc = zpp::bits::rpc<
        zpp::bits::bind<&cuda_api_v2::device_get,                    0>,
        zpp::bits::bind<&cuda_api_v2::device_primary_ctx_retain,     1>,
        zpp::bits::bind<&cuda_api_v2::device_primary_ctx_release,    2>,
        zpp::bits::bind<&cuda_api_v2::device_primary_ctx_set_flags,  3>,
        zpp::bits::bind<&cuda_api_v2::device_primary_ctx_get_state,  4>,
        zpp::bits::bind<&cuda_api_v2::get_proc_address,              5>,
        zpp::bits::bind<&cuda_api_v2::ctx_push_current,              6>,
        zpp::bits::bind<&cuda_api_v2::ctx_pop_current,               7>,
        zpp::bits::bind<&cuda_api_v2::stream_create_with_priority,   8>,
        zpp::bits::bind<&cuda_api_v2::mem_host_get_device_pointer,   9>
    >;
}

template<int FuncIndex, typename... Args>
static auto cuda_rpc_call_pytorch(Args&&... args) {
    std::lock_guard lock(g_rpc_mutex);

    using namespace zpp::bits;
    auto [data, in, out] = data_in_out();
    pytorch_rpc::rpc::client client{in, out};

    client.template request<FuncIndex>(std::forward<Args>(args)...).or_throw();

    zlink::frame req_frame;
    req_frame.call_id = 2;  // 2 = pytorch RPC (to distinguish from base RPC on server)
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

// ══════════════════════════════════════════════════════════════════════
// INTERCEPTED CUDA DRIVER API FUNCTIONS
// ══════════════════════════════════════════════════════════════════════
// Each function follows the same pattern:
//   1. init_shim() — ensure connection
//   2. Translate handle parameters (virtual → server)
//   3. Sync host memory if needed
//   4. Make RPC call
//   5. Register produced handles
//   6. Readback results if needed

extern "C" {

// ── Initialization ─────────────────────────────────────────────────────

CUresult cuInit(unsigned int Flags) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    auto ret = cuda_rpc_call_base<v2_func::cu_init>(Flags);
    return static_cast<CUresult>(ret.result);
}

// ── Device Management ──────────────────────────────────────────────────

CUresult cuDeviceGetCount(int* count) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    auto ret = cuda_rpc_call_base<v2_func::device_get_count>();
    *count = ret.count;
    return static_cast<CUresult>(ret.result);
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    auto ret = cuda_rpc_call_pytorch<0>(ordinal);
    *device = ret.device;
    return static_cast<CUresult>(ret.result);
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    auto ret = cuda_rpc_call_base<v2_func::device_get_name>(static_cast<int>(dev));
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        std::strncpy(name, ret.name.c_str(), len - 1);
        name[len - 1] = '\0';
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    auto ret = cuda_rpc_call_base<v2_func::device_total_mem>(static_cast<int>(dev));
    *bytes = static_cast<size_t>(ret.bytes);
    return static_cast<CUresult>(ret.result);
}

CUresult cuDeviceGetAttribute(int* value, CUdevice_attribute attrib, CUdevice dev) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    auto ret = cuda_rpc_call_base<v2_func::device_get_attribute>(
        static_cast<int>(attrib), static_cast<int>(dev));
    *value = ret.value;
    return static_cast<CUresult>(ret.result);
}

// ── Primary Context Management ★★★ CRITICAL FOR PYTORCH ──────────────
// PyTorch uses primary contexts exclusively. It NEVER calls cuCtxCreate.
// Instead, PyTorch calls cuDevicePrimaryCtxRetain to get the primary
// context for a device, then uses cuCtxSetCurrent or cuCtxPushCurrent
// to make it active.

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_pytorch<1>(static_cast<int>(dev));
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        // Allocate a client-side handle to represent this context
        // We use a monotonically increasing address in the shadow region
        static std::uint64_t next_ctx_id = 0x7f0100000000ULL;
        std::uint64_t client_handle = next_ctx_id++;
        register_handle(client_handle, ret.ctx_handle, "ctx");
        *pctx = reinterpret_cast<CUcontext>(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    auto ret = cuda_rpc_call_pytorch<2>(static_cast<int>(dev));
    return static_cast<CUresult>(ret.result);
}

CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    auto ret = cuda_rpc_call_pytorch<3>(
        static_cast<int>(dev), flags);
    return static_cast<CUresult>(ret.result);
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    auto ret = cuda_rpc_call_pytorch<4>(static_cast<int>(dev));
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        if (flags) *flags = static_cast<unsigned int>(ret.flags);
        if (active) *active = ret.active;
    }
    return static_cast<CUresult>(ret.result);
}

// ── Context Management ─────────────────────────────────────────────────

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_base<v2_func::ctx_create>(flags, static_cast<int>(dev));
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        static std::uint64_t next_ctx_id = 0x7f0200000000ULL;
        std::uint64_t client_handle = next_ctx_id++;
        register_handle(client_handle, ret.ctx_handle, "ctx");
        *pctx = reinterpret_cast<CUcontext>(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuCtxDestroy(CUcontext ctx) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto client_handle = reinterpret_cast<std::uint64_t>(ctx);
    auto server_handle = translate_handle(client_handle);
    if (!server_handle) return CUDA_ERROR_INVALID_CONTEXT;

    auto ret = cuda_rpc_call_base<v2_func::ctx_destroy>(*server_handle);
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        unregister_handle(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto client_handle = reinterpret_cast<std::uint64_t>(ctx);
    auto server_handle = translate_handle(client_handle);
    if (!server_handle) return CUDA_ERROR_INVALID_CONTEXT;

    auto ret = cuda_rpc_call_base<v2_func::ctx_set_current>(*server_handle);
    return static_cast<CUresult>(ret.result);
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_base<v2_func::ctx_get_current>();
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS) && ret.ctx_handle != 0) {
        // Find existing client handle for this server handle, or create one
        std::uint64_t client_handle = 0;
        {
            std::lock_guard lock(g_handle_mutex);
            for (const auto& [ch, entry] : g_handle_map) {
                if (entry.server_handle == ret.ctx_handle && entry.type == "ctx") {
                    client_handle = ch;
                    break;
                }
            }
        }
        if (client_handle == 0) {
            static std::uint64_t next_ctx_id = 0x7f0300000000ULL;
            client_handle = next_ctx_id++;
            register_handle(client_handle, ret.ctx_handle, "ctx");
        }
        *pctx = reinterpret_cast<CUcontext>(client_handle);
    } else {
        *pctx = nullptr;
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuCtxSynchronize() {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    auto ret = cuda_rpc_call_base<v2_func::ctx_synchronize>();
    return static_cast<CUresult>(ret.result);
}

CUresult cuCtxPushCurrent(CUcontext ctx) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto client_handle = reinterpret_cast<std::uint64_t>(ctx);
    auto server_handle = translate_handle(client_handle);
    if (!server_handle) return CUDA_ERROR_INVALID_CONTEXT;

    auto ret = cuda_rpc_call_pytorch<6>(*server_handle);
    return static_cast<CUresult>(ret.result);
}

CUresult cuCtxPopCurrent(CUcontext* pctx) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_pytorch<7>();
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS) && ret.ctx_handle != 0) {
        static std::uint64_t next_ctx_id = 0x7f0400000000ULL;
        std::uint64_t client_handle = next_ctx_id++;
        register_handle(client_handle, ret.ctx_handle, "ctx");
        *pctx = reinterpret_cast<CUcontext>(client_handle);
    } else {
        *pctx = nullptr;
    }
    return static_cast<CUresult>(ret.result);
}

// ── Memory Management ──────────────────────────────────────────────────

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_base<v2_func::mem_alloc>(static_cast<std::uint64_t>(bytesize));
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        // Map the server's device pointer to a client-side shadow pointer
        CUdeviceptr client_ptr = g_ptr_map.map(ret.dev_ptr);
        *dptr = client_ptr;
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_base<v2_func::mem_alloc_managed>(
        static_cast<std::uint64_t>(bytesize), flags);
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        CUdeviceptr client_ptr = g_ptr_map.map(ret.dev_ptr);
        *dptr = client_ptr;
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuMemFree(CUdeviceptr dptr) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto remote = g_ptr_map.to_remote(dptr);
    if (!remote) return CUDA_ERROR_INVALID_DEVICE_POINTER;
    g_ptr_map.unmap(dptr);

    auto ret = cuda_rpc_call_base<v2_func::mem_free>(*remote);
    return static_cast<CUresult>(ret.result);
}

CUresult cuMemFreeHost(void* p) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto client_handle = reinterpret_cast<std::uint64_t>(p);
    auto server_handle = translate_handle(client_handle);
    if (!server_handle) return CUDA_ERROR_INVALID_VALUE;

    auto ret = cuda_rpc_call_base<v2_func::mem_free_host>(*server_handle);
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        unregister_handle(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int flags) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_base<v2_func::mem_host_alloc>(
        static_cast<std::uint64_t>(bytesize), flags);
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        // The server allocated host memory. We need a local representation.
        // For now, allocate local memory too and track the mapping.
        void* local_ptr = std::malloc(bytesize);
        if (!local_ptr) return CUDA_ERROR_OUT_OF_MEMORY;
        register_handle(reinterpret_cast<std::uint64_t>(local_ptr),
                       ret.host_ptr, "hostptr");
        *pp = local_ptr;
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuMemHostRegister(void* p, size_t bytesize, unsigned int flags) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    // Sync the host memory to the server first
    sync_host_to_server(p, bytesize);

    auto ret = cuda_rpc_call_base<v2_func::mem_host_register>(
        reinterpret_cast<std::uint64_t>(p),
        static_cast<std::uint64_t>(bytesize), flags);
    return static_cast<CUresult>(ret.result);
}

CUresult cuMemHostUnregister(void* p) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_base<v2_func::mem_host_unregister>(
        reinterpret_cast<std::uint64_t>(p));
    return static_cast<CUresult>(ret.result);
}

CUresult cuMemGetInfo(size_t* free, size_t* total) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_base<v2_func::mem_get_info>();
    *free = static_cast<size_t>(ret.free_bytes);
    *total = static_cast<size_t>(ret.total_bytes);
    return static_cast<CUresult>(ret.result);
}

CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int flags) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_pytorch<9>(
        reinterpret_cast<std::uint64_t>(p));
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        CUdeviceptr client_ptr = g_ptr_map.map(ret.dev_ptr);
        *pdptr = client_ptr;
    }
    return static_cast<CUresult>(ret.result);
}

// ── Memory Copy ────────────────────────────────────────────────────────

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto remote_dst = g_ptr_map.to_remote(dstDevice);
    if (!remote_dst) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    // Sync host memory to server before the copy
    sync_host_to_server(srcHost, ByteCount);

    auto ret = cuda_rpc_call_base<v2_func::memcpy_htod>(
        *remote_dst,
        reinterpret_cast<std::uint64_t>(srcHost),
        static_cast<std::uint64_t>(ByteCount));
    return static_cast<CUresult>(ret.result);
}

CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto remote_src = g_ptr_map.to_remote(srcDevice);
    if (!remote_src) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    // Sync the destination buffer address so server knows where to write
    sync_host_to_server(dstHost, ByteCount);

    auto ret = cuda_rpc_call_base<v2_func::memcpy_dtoh>(
        reinterpret_cast<std::uint64_t>(dstHost),
        *remote_src,
        static_cast<std::uint64_t>(ByteCount));

    // Read back the result from the server's mirror
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        readback_from_server(dstHost, ByteCount);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto remote_dst = g_ptr_map.to_remote(dstDevice);
    auto remote_src = g_ptr_map.to_remote(srcDevice);
    if (!remote_dst || !remote_src) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    auto ret = cuda_rpc_call_base<v2_func::memcpy_dtod>(
        *remote_dst, *remote_src, static_cast<std::uint64_t>(ByteCount));
    return static_cast<CUresult>(ret.result);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost,
                           size_t ByteCount, CUstream hStream) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto remote_dst = g_ptr_map.to_remote(dstDevice);
    if (!remote_dst) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    auto stream_val = hStream ? translate_handle(reinterpret_cast<std::uint64_t>(hStream))
                              : std::optional<std::uint64_t>(0);

    // Sync host memory to server
    sync_host_to_server(srcHost, ByteCount);

    auto ret = cuda_rpc_call_base<v2_func::memcpy_htod_async>(
        *remote_dst,
        reinterpret_cast<std::uint64_t>(srcHost),
        static_cast<std::uint64_t>(ByteCount),
        stream_val.value_or(0));
    return static_cast<CUresult>(ret.result);
}

CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto remote_src = g_ptr_map.to_remote(srcDevice);
    if (!remote_src) return CUDA_ERROR_INVALID_DEVICE_POINTER;

    auto stream_val = hStream ? translate_handle(reinterpret_cast<std::uint64_t>(hStream))
                              : std::optional<std::uint64_t>(0);

    sync_host_to_server(dstHost, ByteCount);

    auto ret = cuda_rpc_call_base<v2_func::memcpy_dtoh_async>(
        reinterpret_cast<std::uint64_t>(dstHost),
        *remote_src,
        static_cast<std::uint64_t>(ByteCount),
        stream_val.value_or(0));

    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        readback_from_server(dstHost, ByteCount);
    }
    return static_cast<CUresult>(ret.result);
}

// ── Module & Kernel Management ─────────────────────────────────────────

CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_base<v2_func::module_load>(std::string(fname));
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        static std::uint64_t next_mod_id = 0x7f0500000000ULL;
        std::uint64_t client_handle = next_mod_id++;
        register_handle(client_handle, ret.module_handle, "module");
        *module = reinterpret_cast<CUmodule>(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    // Determine the size of the CUDA fatbin image
    // CUDA images start with a magic number; we'll read the size from the header
    // For safety, use a reasonable maximum. In practice, PyTorch sends
    // the complete PTX/cubin which we need to sync to the server.
    //
    // The fatbin format starts with: 0x466243b1 (magic) followed by:
    //   version (2B), header_size (2B), ... total size encoded
    // For now, we sync a generous chunk and let the server parse it.
    const char* img = static_cast<const char*>(image);
    size_t image_size = 0;

    // Try to read fatbin size from header
    if (img[0] == 0x4e && img[1] == 0x45 && img[2] == 0x56 && img[3] == 0x49) {
        // This is an "NEVI" (NVIDIA) fatbin header
        // The size is at offset 8-12 in some versions
        // Fallback: scan for the end marker or use a large buffer
        image_size = 1024 * 1024; // 1MB max for now
    } else {
        // Could be PTX text or raw cubin — find the null terminator for PTX
        image_size = std::strlen(img) + 1;
        if (image_size < 16) image_size = 16 * 1024; // Minimum reasonable size
    }

    sync_host_to_server(image, image_size);

    auto ret = cuda_rpc_call_base<v2_func::module_load_data>(
        reinterpret_cast<std::uint64_t>(image),
        static_cast<std::uint64_t>(image_size));
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        static std::uint64_t next_mod_id = 0x7f0600000000ULL;
        std::uint64_t client_handle = next_mod_id++;
        register_handle(client_handle, ret.module_handle, "module");
        *module = reinterpret_cast<CUmodule>(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuModuleUnload(CUmodule hmod) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto client_handle = reinterpret_cast<std::uint64_t>(hmod);
    auto server_handle = translate_handle(client_handle);
    if (!server_handle) return CUDA_ERROR_INVALID_VALUE;

    auto ret = cuda_rpc_call_base<v2_func::module_unload>(*server_handle);
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        unregister_handle(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto client_mod = reinterpret_cast<std::uint64_t>(hmod);
    auto server_mod = translate_handle(client_mod);
    if (!server_mod) return CUDA_ERROR_INVALID_VALUE;

    auto ret = cuda_rpc_call_base<v2_func::module_get_function>(*server_mod, std::string(name));
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        static std::uint64_t next_func_id = 0x7f0700000000ULL;
        std::uint64_t client_handle = next_func_id++;
        register_handle(client_handle, ret.func_handle, "func");
        *hfunc = reinterpret_cast<CUfunction>(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod,
                           const char* name) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto client_mod = reinterpret_cast<std::uint64_t>(hmod);
    auto server_mod = translate_handle(client_mod);
    if (!server_mod) return CUDA_ERROR_INVALID_VALUE;

    auto ret = cuda_rpc_call_base<v2_func::module_get_global>(*server_mod, std::string(name));
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        CUdeviceptr client_ptr = g_ptr_map.map(ret.global_ptr);
        *dptr = client_ptr;
        if (bytes) *bytes = static_cast<size_t>(ret.global_size);
    }
    return static_cast<CUresult>(ret.result);
}

// ── cuGetProcAddress ★★★ CRITICAL FOR PYTORCH 2.0+ ───────────────────
// PyTorch 2.0+ and libcudart.so.12 use cuGetProcAddress to dynamically
// resolve function pointers. There are TWO types of symbols:
//
//   1. Driver API functions (cuInit, cuMemAlloc, cuLaunchKernel, etc.)
//      These are regular C functions that the caller expects to invoke
//      directly via the returned pointer. We MUST return pointers to
//      our own shim implementations, because our shim intercepts these
//      calls and forwards them over RPC.
//
//   2. GPU kernel entry points (e.g., kernel names from loaded modules)
//      These are GPU code addresses that can only be used with
//      cuLaunchKernel. For these, we forward the symbol to the server
//      and return a virtual handle.
//
// If we returned a virtual handle for a driver function (like
// cuDriverGetVersion), the caller would try to call it as a function
// pointer, which would crash because the virtual handle is not a valid
// code address.

// Map of driver API symbol names to our shim's function pointers.
// Built lazily on first call to avoid forward-reference issues.
static std::unordered_map<std::string, void*>& get_local_symbol_map() {
    static std::unordered_map<std::string, void*> symbols;
    static bool initialized = false;
    if (!initialized) {
        // Open our own shared library (libcuda.so.1) to find exported symbols.
        // We cannot use dlopen(NULL, ...) because that returns the main
        // executable handle, which does not include symbols from shared
        // libraries loaded via LD_LIBRARY_PATH even with RTLD_GLOBAL.
        // Using RTLD_NOLOAD avoids re-initializing the library.
        void* self = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (!self) {
            // Fallback: try dlopen(NULL) in case the library was linked
            self = dlopen(NULL, RTLD_NOW | RTLD_GLOBAL);
        }
        if (self) {
            // Core functions that libcudart resolves via cuGetProcAddress
            for (const char* name : {
                "cuInit", "cuDriverGetVersion", "cuDeviceGetCount",
                "cuDeviceGet", "cuDeviceGetName", "cuDeviceTotalMem",
                "cuDeviceGetAttribute", "cuDevicePrimaryCtxRetain",
                "cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxSetFlags",
                "cuDevicePrimaryCtxGetState", "cuCtxCreate", "cuCtxDestroy",
                "cuCtxSetCurrent", "cuCtxGetCurrent", "cuCtxSynchronize",
                "cuCtxPushCurrent", "cuCtxPopCurrent", "cuMemAlloc",
                "cuMemAllocManaged", "cuMemFree", "cuMemFreeHost",
                "cuMemHostAlloc", "cuMemHostRegister", "cuMemHostUnregister",
                "cuMemGetInfo", "cuMemHostGetDevicePointer",
                "cuMemcpyHtoD", "cuMemcpyDtoH", "cuMemcpyDtoD",
                "cuMemcpyHtoDAsync", "cuMemcpyDtoHAsync",
                "cuModuleLoadData", "cuModuleLoad", "cuModuleUnload",
                "cuModuleGetFunction", "cuModuleGetGlobal", "cuLaunchKernel",
                "cuStreamCreate", "cuStreamCreateWithPriority",
                "cuStreamDestroy", "cuStreamSynchronize",
                "cuEventCreate", "cuEventDestroy", "cuEventRecord",
                "cuEventSynchronize", "cuEventElapsedTime",
                "cuOccupancyMaxPotentialBlockSize",
                "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
                "cuGetProcAddress", "cuGetProcAddress_v2",
                "cuMemHostRegister_v2"
            }) {
                void* sym = dlsym(self, name);
                if (sym) symbols[name] = sym;
            }
        }
        initialized = true;
    }
    return symbols;
}

CUresult cuGetProcAddress(const char* symbol, void** pfn,
                          int cudaVersion, unsigned long long flags,
                          void* symbolStatus) {
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;

    // First: check if this is a driver API function we implement locally
    auto& symbols = get_local_symbol_map();
    auto it = symbols.find(symbol);
    if (it != symbols.end()) {
        *pfn = it->second;
        if (symbolStatus) {
            *static_cast<int*>(symbolStatus) = 0; // CU_GET_PROC_ADDRESS_SUCCESS
        }
        return CUDA_SUCCESS;
    }

    // Not a locally-known driver function — check if this is a driver API
    // function (starts with "cu") that the server knows about. For these,
    // we return the universal stub since PyTorch may call them directly
    // through the function pointer. The actual RPC happens when PyTorch
    // calls the same function through libcuda.so (our shim).
    //
    // For non-driver symbols (e.g., GPU kernel entry points from
    // cuModuleGetFunction), we DO need virtual handles because those
    // are used with cuLaunchKernel.
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_pytorch<5>(
        std::string(symbol), static_cast<int32_t>(cudaVersion),
        static_cast<std::uint64_t>(flags));
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS) && ret.func_ptr != 0) {
        // Check if this is a driver API function (starts with "cu" but not
        // a kernel entry point). Driver API functions start with "cu" and
        // the third character is uppercase (e.g., "cuInit", "cuMemAlloc").
        // Kernel entry points have mangled names or lowercase third chars.
        bool is_driver_api = (symbol[0] == 'c' && symbol[1] == 'u' &&
                              symbol[2] >= 'A' && symbol[2] <= 'Z');

        if (is_driver_api) {
            // Driver API function: return the stub (PyTorch may call via fp).
            // The actual work happens when PyTorch calls through libcuda.so.
            static CUresult (*stub_fn)() = []() -> CUresult { return CUDA_SUCCESS; };
            *pfn = reinterpret_cast<void*>(stub_fn);
        } else {
            // Kernel entry point or other symbol: use virtual handle.
            // These are used with cuLaunchKernel, never called directly.
            static std::uint64_t next_func_id = 0x7f0800000000ULL;
            std::uint64_t client_handle = next_func_id++;
            register_handle(client_handle, ret.func_ptr, "func");
            *pfn = reinterpret_cast<void*>(client_handle);
        }

        if (symbolStatus) {
            *static_cast<int*>(symbolStatus) = 0; // CU_GET_PROC_ADDRESS_SUCCESS
        }
        return CUDA_SUCCESS;
    }
    
    // Server didn't find the symbol either. Return a stub function pointer
    // that returns CUDA_SUCCESS (0). This prevents libcudart from reporting
    // "API call is not supported in the installed CUDA driver" (error 36).
    // Many CUDA driver functions are optional and libcudart checks for their
    // availability via cuGetProcAddress — if we return an error, libcudart
    // thinks the driver is too old.
    static CUresult (*stub_fn)() = []() -> CUresult { return CUDA_SUCCESS; };
    *pfn = reinterpret_cast<void*>(stub_fn);
    if (symbolStatus) {
        // CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 2
        *static_cast<int*>(symbolStatus) = 2;
    }
    return CUDA_SUCCESS;
}

// ── Kernel Launch ──────────────────────────────────────────────────────

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto client_func = reinterpret_cast<std::uint64_t>(f);
    auto server_func = translate_handle(client_func);
    if (!server_func) return CUDA_ERROR_INVALID_VALUE;

    auto stream_val = hStream
        ? translate_handle(reinterpret_cast<std::uint64_t>(hStream))
        : std::optional<std::uint64_t>(0);

    // Translate kernel arguments
    // kernelParams is an array of void* pointers to arguments.
    // Each argument may be a device pointer that needs translation,
    // or a scalar value that should be passed as-is.
    //
    // Strategy: serialize the argument buffer, translating device pointers.
    // We need to figure out how many args there are. PyTorch typically
    // passes all args via kernelParams (extra is usually nullptr).
    //
    // We'll count args by scanning until we hit a null pointer or
    // a reasonable maximum. Then we translate each arg that looks like
    // a device pointer (i.e., is in our ptr_map).

    std::vector<std::uint64_t> translated_args;
    int n_args = 0;

    if (kernelParams) {
        for (int i = 0; i < 64; i++) {  // Max 64 args
            if (!kernelParams[i]) break;
            n_args++;

            // The value pointed to by kernelParams[i] could be:
            //   - A device pointer (8 bytes, in our shadow region)
            //   - A scalar value (1-8 bytes)
            // We heuristically translate: if it looks like a mapped
            // device pointer, translate it.
            std::uint64_t arg_val = *reinterpret_cast<std::uint64_t*>(kernelParams[i]);

            auto remote_ptr = g_ptr_map.to_remote(static_cast<CUdeviceptr>(arg_val));
            if (remote_ptr) {
                // This is a device pointer — use the server-side value
                translated_args.push_back(*remote_ptr);
            } else {
                // This is a scalar — pass as-is
                translated_args.push_back(arg_val);
            }
        }
    }

    // Sync the translated args to server as host memory
    if (!translated_args.empty()) {
        sync_host_to_server(translated_args.data(),
                           translated_args.size() * sizeof(std::uint64_t));
    }

    auto ret = cuda_rpc_call_base<v2_func::launch_kernel>(
        *server_func,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        stream_val.value_or(0),
        reinterpret_cast<std::uint64_t>(translated_args.data()),
        translated_args.size() * sizeof(std::uint64_t));
    return static_cast<CUresult>(ret.result);
}

// ── Stream Management ──────────────────────────────────────────────────

CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_base<v2_func::stream_create>(Flags);
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        static std::uint64_t next_stream_id = 0x7f0900000000ULL;
        std::uint64_t client_handle = next_stream_id++;
        register_handle(client_handle, ret.stream_handle, "stream");
        *phStream = reinterpret_cast<CUstream>(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags,
                                    int priority) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_pytorch<8>(flags, priority);
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        static std::uint64_t next_stream_id = 0x7f0a00000000ULL;
        std::uint64_t client_handle = next_stream_id++;
        register_handle(client_handle, ret.stream_handle, "stream");
        *phStream = reinterpret_cast<CUstream>(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuStreamDestroy(CUstream hStream) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto client_handle = reinterpret_cast<std::uint64_t>(hStream);
    auto server_handle = translate_handle(client_handle);
    if (!server_handle) return CUDA_ERROR_INVALID_HANDLE;

    auto ret = cuda_rpc_call_base<v2_func::stream_destroy>(*server_handle);
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        unregister_handle(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuStreamSynchronize(CUstream hStream) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto stream_val = hStream
        ? translate_handle(reinterpret_cast<std::uint64_t>(hStream))
        : std::optional<std::uint64_t>(0);

    auto ret = cuda_rpc_call_base<v2_func::stream_synchronize>(stream_val.value_or(0));
    return static_cast<CUresult>(ret.result);
}

// ── Event Management ───────────────────────────────────────────────────

CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call_base<v2_func::event_create>(Flags);
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        static std::uint64_t next_event_id = 0x7f0b00000000ULL;
        std::uint64_t client_handle = next_event_id++;
        register_handle(client_handle, ret.event_handle, "event");
        *phEvent = reinterpret_cast<CUevent>(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuEventDestroy(CUevent hEvent) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto client_handle = reinterpret_cast<std::uint64_t>(hEvent);
    auto server_handle = translate_handle(client_handle);
    if (!server_handle) return CUDA_ERROR_INVALID_HANDLE;

    auto ret = cuda_rpc_call_base<v2_func::event_destroy>(*server_handle);
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        unregister_handle(client_handle);
    }
    return static_cast<CUresult>(ret.result);
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto event_val = translate_handle(reinterpret_cast<std::uint64_t>(hEvent));
    auto stream_val = hStream
        ? translate_handle(reinterpret_cast<std::uint64_t>(hStream))
        : std::optional<std::uint64_t>(0);

    if (!event_val) return CUDA_ERROR_INVALID_HANDLE;

    auto ret = cuda_rpc_call_base<v2_func::event_record>(
        *event_val, stream_val.value_or(0));
    return static_cast<CUresult>(ret.result);
}

CUresult cuEventSynchronize(CUevent hEvent) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto event_val = translate_handle(reinterpret_cast<std::uint64_t>(hEvent));
    if (!event_val) return CUDA_ERROR_INVALID_HANDLE;

    auto ret = cuda_rpc_call_base<v2_func::event_synchronize>(*event_val);
    return static_cast<CUresult>(ret.result);
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto start_val = translate_handle(reinterpret_cast<std::uint64_t>(hStart));
    auto end_val = translate_handle(reinterpret_cast<std::uint64_t>(hEnd));
    if (!start_val || !end_val) return CUDA_ERROR_INVALID_HANDLE;

    auto ret = cuda_rpc_call_base<v2_func::event_elapsed_time>(*start_val, *end_val);
    if (ret.result == static_cast<int32_t>(CUDA_SUCCESS)) {
        *pMilliseconds = ret.milliseconds;
    }
    return static_cast<CUresult>(ret.result);
}

// ── Occupancy ──────────────────────────────────────────────────────────

CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize,
                                           CUfunction func,
                                           cuOccupancyB2DSize block2Shmem,
                                           int blockSizeLimit) {
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto client_func = reinterpret_cast<std::uint64_t>(func);
    auto server_func = translate_handle(client_func);
    if (!server_func) return CUDA_ERROR_INVALID_VALUE;

    // Use hardcoded defaults for occupancy query over RPC.
    // PyTorch uses this for launch configuration but the result
    // is advisory; default values work for most kernels.
    if (minGridSize) *minGridSize = 1;
    if (blockSize) *blockSize = 256; // Common default block size
    return CUDA_SUCCESS;
}

// ── cuDriverGetVersion ★★★ CRITICAL FOR RUNTIME DETECTION ──────────────
// libcudart.so.12 calls dlopen("libcuda.so.1") and then tries to resolve
// cuDriverGetVersion FIRST to check the driver version. If this function
// is missing, libcudart reports "Found no NVIDIA driver on your system"
// and torch.cuda.is_available() returns False.
//
// The CUDA driver version encodes as: major * 1000 + minor * 10
// CUDA 12.4 = 12040, CUDA 12.1 = 12010, etc.
CUresult cuDriverGetVersion(int* driverVersion) {
    if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;
    // Report CUDA 12.4 driver (matches CUDA 12.4 runtime bundled with PyTorch)
    *driverVersion = 12040;
    return CUDA_SUCCESS;
}

// ── cuGetProcAddress_v2 ─────────────────────────────────────────────────
// Newer CUDA runtime versions (12.x) may call cuGetProcAddress_v2
// instead of cuGetProcAddress. The v2 version has an additional parameter
// (cuuint64_t flags). We delegate to our existing cuGetProcAddress.
CUresult cuGetProcAddress_v2(const char* symbol, void** pfn,
                              int cudaVersion, unsigned long long flags,
                              void* symbolStatus) {
    return cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
}

// ── cuOccupancyMaxActiveBlocksPerMultiprocessor ─────────────────────────
// Required by libcudart.so.12 for occupancy queries.
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                      CUfunction func,
                                                      int blockSize,
                                                      size_t dynamicSMemSize) {
    if (!numBlocks) return CUDA_ERROR_INVALID_VALUE;
    // Reasonable default for RTX 3070 (46 SMs, 1536 max threads/SM)
    // numBlocks = 1536 / blockSize (approximate)
    *numBlocks = (blockSize > 0) ? (1536 / blockSize) : 1;
    return CUDA_SUCCESS;
}

// ── cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ────────────────
// Required by libcudart.so.12 for occupancy queries.
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                               CUfunction func,
                                                               int blockSize,
                                                               size_t dynamicSMemSize,
                                                               unsigned int flags) {
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

// ── cuMemHostRegister_v2 ───────────────────────────────────────────────
// Some CUDA runtime versions call cuMemHostRegister_v2 instead of cuMemHostRegister.
// The v2 version has the same signature but may have different alignment requirements.
CUresult cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int flags) {
    return cuMemHostRegister(p, bytesize, flags);
}

} // extern "C"

// ── Library constructor/destructor ─────────────────────────────────────
__attribute__((constructor))
static void shim_init() {
    std::cerr << "zlink CUDA PyTorch shim loaded\n"
              << "  ZLINK_SERVER=" << (std::getenv("ZLINK_SERVER") ?: "not set") << "\n";
}

__attribute__((destructor))
static void shim_fini() {
    if (g_transport && g_transport->is_connected()) {
        g_transport->close();
        std::cerr << "zlink pytorch shim: disconnected\n";
    }
}
