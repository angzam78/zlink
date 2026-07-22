// zlink/examples/cuda/cuda_shim.cpp
//
// CUDA Driver API shim — comprehensive coverage via codegen.
//
// This file provides the infrastructure that the generated shim functions
// (gen_shim.inc) rely on, and includes the generated code.
//
// Built as libcuda.so.1, placed in LD_LIBRARY_PATH on the CPU client.
// All cu* function calls are intercepted and forwarded via zlink RPC
// to a remote GPU server running cuda_server.
//
// KEY DESIGN:
//   - gen_api.hpp defines the RPC structs and function bindings
//   - gen_shim.inc contains the extern "C" cu* function bodies
//   - This file provides: init_shim(), handle_map, ptr_map, host_sync, etc.
//   - Special functions (cuLaunchKernel, cuGetProcAddress) are hand-written
//     here because they need complex local logic
//   - Client and server always use the same zlink build (no protocol versioning)
//   - Works with CUDA 12.4+ including 13.x (no version negotiation)

#include "codegen/gen_api.hpp"

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
typedef void* CUgraph;
typedef void* CUgraphNode;
typedef void* CUgraphExec;
typedef void* CUmipmappedArray;
typedef void* CUarray;
typedef void* CUsurfref;
typedef void* CUtexref;
typedef void* CUextMemory;
typedef void* CUextSemaphore;
typedef void* CUkernel;
typedef void* CUlibrary;
typedef void* CUgraphicsResource;
typedef std::uint64_t CUdeviceptr;
typedef void* CUmemGenericAllocationHandle;
typedef void* CUexternalMemory;
typedef void* CUexternalSemaphore;
typedef void (*cuOccupancyB2DSize)(int);

// CUDA error codes
#define CUDA_SUCCESS                    0
#define CUDA_ERROR_NOT_INITIALIZED      3
#define CUDA_ERROR_INVALID_VALUE        2
#define CUDA_ERROR_INVALID_CONTEXT      7
#define CUDA_ERROR_INVALID_HANDLE       4
#define CUDA_ERROR_INVALID_DEVICE_POINTER 6
#define CUDA_ERROR_OUT_OF_MEMORY        2
#define CUDA_ERROR_DEVICE_UNAVAILABLE   46

// CUDA device attribute type
typedef unsigned int CUdevice_attribute;

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <dlfcn.h>

// ══════════════════════════════════════════════════════════════════════════
// Global state — same infrastructure as the hand-written shim
// ══════════════════════════════════════════════════════════════════════════
static std::unique_ptr<zlink::transport> g_transport;
static zlink::ptr_map              g_ptr_map;
static zlink::host_memory_mirror   g_host_mirror;
static std::mutex                  g_rpc_mutex;
static bool                        g_initialized = false;

// ── Handle mapping ─────────────────────────────────────────────────────
struct handle_entry {
    std::uint64_t server_handle;
    std::string   type;
};

static std::unordered_map<std::uint64_t, handle_entry> g_handle_map;
static std::mutex g_handle_mutex;

// Monotonically increasing client handle allocator
static std::uint64_t g_next_client_handle = 0x7f0100000000ULL;

static std::uint64_t allocate_client_handle(const std::string& type) {
    std::lock_guard lock(g_handle_mutex);
    std::uint64_t h = g_next_client_handle++;
    // Small gap between handle ranges per type for debugging
    return h;
}

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
static CUdeviceptr register_devptr(CUdeviceptr server_ptr) {
    CUdeviceptr client_ptr = static_cast<CUdeviceptr>(g_ptr_map.map(server_ptr));
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

    zlink::frame resp_frame;
    ec = g_transport->receive(resp_frame);
    if (ec) {
        std::cerr << "  [shim] host_sync recv error: " << ec.message() << "\n";
    }
}

// ── Host memory readback helper ────────────────────────────────────────
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
        std::cerr << "zlink codegen shim: ZLINK_SERVER not set\n";
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
        std::cerr << "zlink codegen shim: Failed to connect: " << ec.message() << "\n";
        return;
    }

    g_ptr_map.set_shadow_region(0x7f0000000000ULL, 0x100000000ULL);
    g_initialized = true;
    std::cerr << "zlink codegen shim: Connected to " << host << ":" << port << "\n";
}

// ── RPC helper ─────────────────────────────────────────────────────────
// Uses separate fresh zpp::bits contexts for request and response (same
// pattern as zlink::rpc_client::call()). This avoids any stale-archive
// issues when the data vector contents change between request and response.
// Over an internet connection, the extra construction cost is negligible
// compared to the RTT.
template<int FuncIndex, typename... Args>
static auto cuda_rpc_call(Args&&... args) {
    std::lock_guard lock(g_rpc_mutex);

    using namespace zpp::bits;

    // Serialize request (fresh context)
    auto [req_data, req_in, req_out] = data_in_out();
    cuda_gen::cuda_gen_rpc::client req_client{req_in, req_out};
    req_client.template request<FuncIndex>(std::forward<Args>(args)...).or_throw();

    zlink::frame req_frame;
    req_frame.call_id = 1;
    req_frame.type = zlink::frame_type::request;
    req_frame.payload.assign(req_data.begin(), req_data.end());

    auto ec = g_transport->send(req_frame);
    if (ec) throw std::system_error(ec);

    zlink::frame resp_frame;
    ec = g_transport->receive(resp_frame);
    if (ec) throw std::system_error(ec);

    // Deserialize response (fresh context)
    auto [resp_data, resp_in, resp_out] = data_in_out();
    resp_data.assign(resp_frame.payload.begin(), resp_frame.payload.end());
    cuda_gen::cuda_gen_rpc::client resp_client{resp_in, resp_out};

    return resp_client.template response<FuncIndex>().or_throw();
}

// ══════════════════════════════════════════════════════════════════════════
// HAND-WRITTEN SPECIAL FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════
// These functions need complex local logic that the codegen can't handle

// ── cuGetProcAddress ★★★ CRITICAL FOR PYTORCH 2.0+ ────────────────────
// Must resolve driver API symbols to our local shim implementations,
// and kernel symbols to virtual handles.
static std::unordered_map<std::string, void*>& get_local_symbol_map() {
    static std::unordered_map<std::string, void*> symbols;
    static bool initialized = false;
    if (!initialized) {
        void* self = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (!self) {
            self = dlopen(NULL, RTLD_NOW | RTLD_GLOBAL);
        }
        if (self) {
            // Use the generated symbol map to find all our shim functions
            #include "codegen/gen_symbol_map.inc"

            for (int i = 0; g_cuda_symbol_names[i] != nullptr; i++) {
                void* sym = dlsym(self, g_cuda_symbol_names[i]);
                if (sym) symbols[g_cuda_symbol_names[i]] = sym;
            }

            // Extra symbols that may have different names
            for (const char* name : {
                "cuGetProcAddress", "cuGetProcAddress_v2",
                "cuMemHostRegister_v2", "cuDriverGetVersion"
            }) {
                void* sym = dlsym(self, name);
                if (sym) symbols[name] = sym;
            }
        }
        initialized = true;
    }
    return symbols;
}

static CUresult get_proc_address_impl(const char* symbol, void** pfn,
                                       int cudaVersion, unsigned long long flags) {
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;

    // First: check if this is a driver API function we implement locally
    auto& symbols = get_local_symbol_map();
    auto it = symbols.find(symbol);
    if (it != symbols.end()) {
        *pfn = it->second;
        return CUDA_SUCCESS;
    }

    // Not a locally-known driver function — check if it's a kernel symbol
    // via the server
    init_shim();
    if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;

    auto ret = cuda_rpc_call<cuda_gen::func_index::get_proc_address>(
        std::string(symbol), static_cast<int32_t>(cudaVersion),
        static_cast<std::uint64_t>(flags));

    if (ret.result == CUDA_SUCCESS && ret.func_ptr != 0) {
        bool is_driver_api = (symbol[0] == 'c' && symbol[1] == 'u' &&
                              symbol[2] >= 'A' && symbol[2] <= 'Z');

        if (is_driver_api) {
            // Return a stub that returns CUDA_SUCCESS
            static CUresult (*stub_fn)() = []() -> CUresult { return CUDA_SUCCESS; };
            *pfn = reinterpret_cast<void*>(stub_fn);
        } else {
            // Kernel entry point — use virtual handle
            static std::uint64_t next_func_id = 0x7f0800000000ULL;
            std::uint64_t client_handle = next_func_id++;
            register_handle(client_handle, ret.func_ptr, "func");
            *pfn = reinterpret_cast<void*>(client_handle);
        }
        return CUDA_SUCCESS;
    }

    // Server didn't find the symbol either — return a stub
    static CUresult (*stub_fn)() = []() -> CUresult { return CUDA_SUCCESS; };
    *pfn = reinterpret_cast<void*>(stub_fn);
    return CUDA_SUCCESS;
}

// ── cuLaunchKernel ★★★ THE HOT PATH ────────────────────────────────────
static CUresult launch_kernel_impl(CUfunction f,
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
    std::vector<std::uint64_t> translated_args;
    int n_args = 0;

    if (kernelParams) {
        for (int i = 0; i < 64; i++) {
            if (!kernelParams[i]) break;
            n_args++;
            std::uint64_t arg_val = *reinterpret_cast<std::uint64_t*>(kernelParams[i]);
            auto remote_ptr = g_ptr_map.to_remote(static_cast<CUdeviceptr>(arg_val));
            if (remote_ptr) {
                translated_args.push_back(*remote_ptr);
            } else {
                translated_args.push_back(arg_val);
            }
        }
    }

    if (!translated_args.empty()) {
        sync_host_to_server(translated_args.data(),
                           translated_args.size() * sizeof(std::uint64_t));
    }

    auto ret = cuda_rpc_call<cuda_gen::func_index::launch_kernel>(
        *server_func,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        stream_val.value_or(0),
        reinterpret_cast<std::uint64_t>(translated_args.data()),
        translated_args.size() * sizeof(std::uint64_t));
    return static_cast<CUresult>(ret.result);
}

// ══════════════════════════════════════════════════════════════════════════
// GENERATED SHIM FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════
extern "C" {

#include "codegen/gen_shim.inc"

} // extern "C"

// ── dlsym override ──────────────────────────────────────────────────────
// Intercept dlsym() calls so that CUDA symbol lookups find our shim
// implementations. This is critical for PyTorch and other frameworks that
// call dlsym(RTLD_NEXT, "cuMemcpyHtoD_v2") or dlsym(RTLD_DEFAULT, "cuMemAlloc").
// Without this override, RTLD_NEXT would skip our shim and try to find the
// real CUDA driver (which doesn't exist on the CPU client machine).
//
// DESIGN (based on Lupine's approach):
//   1. Check if the symbol is a CUDA Driver API function (cu* prefix)
//   2. If yes, look it up in our local symbol map (built from gen_symbol_map.inc)
//   3. If not found in map, try dlsym on our own library
//   4. If not a CUDA symbol, delegate to the real dlsym
//
// We use __libc_dlsym() to call the real dlsym without recursion.
// This is a GNU extension available on glibc systems.

extern "C" {

// Declare the real dlsym from libc (bypasses our override)
extern void* __libc_dlsym(void*, const char*) __attribute__((weak));

void* dlsym(void* handle, const char* symbol) {
    // First: check if this is a CUDA Driver API symbol we implement
    // cu* functions where the third character is uppercase
    if (symbol && symbol[0] == 'c' && symbol[1] == 'u' &&
        symbol[2] >= 'A' && symbol[2] <= 'Z') {
        // Look up in our shim's function map
        auto& symbols = get_local_symbol_map();
        auto it = symbols.find(symbol);
        if (it != symbols.end()) {
            return it->second;
        }
        // Not in explicit map — try dlsym on ourselves (our exported symbols)
        void* self = __libc_dlsym ? __libc_dlsym(RTLD_DEFAULT, "libcuda_so_1_handle") : nullptr;
        if (!self) {
            // Try loading ourselves with RTLD_NOLOAD to get a handle
            self = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
            if (self) {
                void* sym = __libc_dlsym ? __libc_dlsym(self, symbol) : nullptr;
                if (sym) return sym;
            }
        }
        // Not found — return nullptr (CUDA symbol not supported yet)
        return nullptr;
    }

    // Not a CUDA symbol — delegate to real dlsym
    if (__libc_dlsym) {
        return __libc_dlsym(handle, symbol);
    }

    // Fallback: dlopen libdl and get the real dlsym
    // This should never happen on glibc systems
    static void* (*real_dlsym_fn)(void*, const char*) = nullptr;
    if (!real_dlsym_fn) {
        void* libdl = dlopen("libdl.so.2", RTLD_NOW | RTLD_LOCAL);
        if (libdl) {
            // Use dlvsym to get the original dlsym (GLIBC_2.34 removed libdl)
            real_dlsym_fn = reinterpret_cast<void*(*)(void*, const char*)>(
                dlvsym(libdl, "dlsym", "GLIBC_2.0"));
            if (!real_dlsym_fn) {
                real_dlsym_fn = reinterpret_cast<void*(*)(void*, const char*)>(
                    dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34"));
            }
        }
    }
    if (real_dlsym_fn) {
        return real_dlsym_fn(handle, symbol);
    }

    // Last resort: this shouldn't happen
    return nullptr;
}

} // extern "C"

// ── Library constructor/destructor ─────────────────────────────────────
__attribute__((constructor))
static void shim_init() {
    std::cerr << "zlink CUDA shim loaded\n"
              << "  ZLINK_SERVER=" << (std::getenv("ZLINK_SERVER") ?: "not set") << "\n";
}

__attribute__((destructor))
static void shim_fini() {
    if (g_transport && g_transport->is_connected()) {
        g_transport->close();
        std::cerr << "zlink shim: disconnected\n";
    }
}
