// zlink/examples/cuda/cuda_server.cpp
//
// CUDA RPC server — comprehensive coverage via codegen.
//
// Handles all CUDA Driver API functions from gen_api.hpp,
// including the generated handlers from gen_server.inc.
//
// Runs on the GPU machine. Receives RPC calls from the client shim
// and executes them on real CUDA hardware.
//
// KEY DESIGN (functionally equivalent to Lupine's server):
//   - gen_api.hpp defines the RPC structs and function bindings
//   - gen_server.inc contains the handler function bodies (cuda_gen namespace)
//   - zpp::bits server.serve() dispatches to bound handlers automatically
//   - Special functions (cuLaunchKernel, cuGetProcAddress) override generated
//     handlers with hand-written implementations
//   - Client and server always use the same zlink build (no protocol versioning)
//   - Single client connection (fork-per-client like Lupine can be added later)
//   - Memory ops (host_sync, host_read) handled alongside RPC requests

#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/ptr_map.hpp>
#include <zlink/memory.hpp>
#include <zlink/chunk_cache.hpp>
#include <zlink/config.hpp>
#include <zlink/virtual_handle.hpp>

#include "codegen/gen_api.hpp"

#include <zpp_bits.h>

#include <cuda.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>

// ── Global state ──────────────────────────────────────────────────────
static zlink::ptr_map g_ptr_map;
static zlink::host_memory_mirror g_host_mirror;
static zlink::handle_table g_vhandles;

// ── Handle production hook ─────────────────────────────────────────────
static std::uint64_t g_last_produced_handle = 0;
static bool g_has_produced_handle = false;

// ══════════════════════════════════════════════════════════════════════════
// HAND-WRITTEN SPECIAL HANDLERS
// ══════════════════════════════════════════════════════════════════════════
// These override the generated handlers for functions that need complex
// server-side logic (argument translation, pointer dereferencing, etc.)

// ── cuGetProcAddress server-side override ────────────────────────────────
// This overrides the generated cuda_gen::get_proc_address handler because
// we need to resolve symbols in the real CUDA driver on this server.
namespace cuda_gen {

GetProcAddressRet get_proc_address(std::string symbol_name, int32_t cuda_version, std::uint64_t flags) {
    GetProcAddressRet ret;
    ret.result = CUDA_ERROR_NOT_FOUND;
    ret.func_ptr = 0;

    void* real_fn = nullptr;
    CUresult res = cuGetProcAddress(symbol_name.c_str(), &real_fn,
                                     cuda_version, flags, nullptr);
    if (res == CUDA_SUCCESS && real_fn != nullptr) {
        ret.result = CUDA_SUCCESS;
        ret.func_ptr = reinterpret_cast<std::uint64_t>(real_fn);
        g_has_produced_handle = true;
    }

    return ret;
}

// ── cuLaunchKernel server-side override ──────────────────────────────────
// This overrides the generated cuda_gen::launch_kernel handler because
// we need to translate handles, dereference kernel args from host mirror,
// and pass them as void** to the real cuLaunchKernel.
LaunchKernelRet launch_kernel(std::uint64_t func_handle,
                               std::uint32_t grid_dim_x, std::uint32_t grid_dim_y, std::uint32_t grid_dim_z,
                               std::uint32_t block_dim_x, std::uint32_t block_dim_y, std::uint32_t block_dim_z,
                               std::uint32_t shared_mem_bytes,
                               std::uint64_t stream_handle,
                               std::uint64_t args_client_addr,
                               std::uint64_t args_byte_count) {

    LaunchKernelRet ret;

    // Translate function handle
    CUfunction real_func = reinterpret_cast<CUfunction>(g_vhandles.translate(func_handle));
    if (!real_func) {
        ret.result = static_cast<int32_t>(CUDA_ERROR_INVALID_VALUE);
        return ret;
    }

    // Translate stream handle
    CUstream real_stream = (stream_handle != 0)
        ? reinterpret_cast<CUstream>(g_vhandles.translate(stream_handle))
        : nullptr;

    // Read kernel arguments from the host mirror
    int n_args = args_byte_count / sizeof(std::uint64_t);
    std::vector<void*> arg_ptrs(n_args);
    std::vector<std::uint64_t> arg_vals(n_args);

    if (n_args > 0) {
        auto mirror_addr = g_host_mirror.translate(
            static_cast<std::uintptr_t>(args_client_addr));
        if (mirror_addr) {
            std::memcpy(arg_vals.data(),
                       reinterpret_cast<void*>(*mirror_addr),
                       args_byte_count);
        }

        // Translate device pointers in args
        for (int i = 0; i < n_args; i++) {
            auto remote = g_ptr_map.to_remote(
                static_cast<std::uintptr_t>(arg_vals[i]));
            if (remote) {
                arg_vals[i] = *remote;
            }
            arg_ptrs[i] = &arg_vals[i];
        }
    }

    CUresult res = cuLaunchKernel(real_func,
                                   grid_dim_x, grid_dim_y, grid_dim_z,
                                   block_dim_x, block_dim_y, block_dim_z,
                                   shared_mem_bytes, real_stream,
                                   arg_ptrs.data(), nullptr);

    ret.result = static_cast<int32_t>(res);
    return ret;
}

} // namespace cuda_gen

// ══════════════════════════════════════════════════════════════════════════
// GENERATED SERVER HANDLERS
// ══════════════════════════════════════════════════════════════════════════
// These provide implementations for all the function declarations in
// gen_api.hpp. The hand-written overrides above are defined BEFORE this
// include so they take precedence (same function names, same namespace).
#include "codegen/gen_server.inc"

// ══════════════════════════════════════════════════════════════════════════
// RPC DISPATCH SERVER
// ══════════════════════════════════════════════════════════════════════════
// Uses zpp::bits server.serve() for automatic dispatch.
// The gen_api.hpp binding maps function indices to cuda_gen::* handlers.
// The server receives a request frame, creates a zpp::bits server context,
// calls serve() which dispatches to the appropriate bound handler,
// and sends the serialized response back.

class cuda_rpc_server {
public:
    cuda_rpc_server() = default;

    void serve(zlink::transport& transport) {
        std::cerr << "[server] CUDA RPC server (codegen, "
                  << cuda_gen::func_index::mem_pool_destroy + 1
                  << " functions) listening...\n";

        auto ec = transport.listen("0.0.0.0", zlink::default_port);
        if (ec) {
            std::cerr << "[server] Listen failed: " << ec.message() << "\n";
            return;
        }

        ec = transport.accept();
        if (ec) {
            std::cerr << "[server] Accept failed: " << ec.message() << "\n";
            return;
        }

        std::cerr << "[server] Client connected\n";

        while (transport.is_connected()) {
            zlink::frame req_frame;
            ec = transport.receive(req_frame);
            if (ec) {
                std::cerr << "[server] Receive error: " << ec.message() << "\n";
                break;
            }

            // Handle memory operations (host_sync, host_read)
            if (req_frame.type == zlink::frame_type::memory_op) {
                handle_memory_op(transport, req_frame);
                continue;
            }

            // Handle RPC request via zpp::bits dispatch
            if (req_frame.type == zlink::frame_type::request) {
                handle_rpc_request(transport, req_frame);
            }

            // Handle pipeline request (batched calls)
            if (req_frame.type == zlink::frame_type::pipeline_request) {
                handle_pipeline_request(transport, req_frame);
            }
        }

        std::cerr << "[server] Client disconnected\n";
    }

private:
    void handle_memory_op(zlink::transport& transport, const zlink::frame& req_frame) {
        if (req_frame.payload.size() < sizeof(zlink::mem_request)) return;

        zlink::mem_request req;
        std::memcpy(&req, req_frame.payload.data(), sizeof(req));

        zlink::frame resp_frame;
        resp_frame.call_id = 0;
        resp_frame.type = zlink::frame_type::memory_op;

        if (req.op == zlink::mem_op::host_sync) {
            // Client is syncing host memory to us
            auto data_span = std::span<const std::byte>(
                req_frame.payload.data() + sizeof(req),
                req_frame.payload.size() - sizeof(req));
            g_host_mirror.sync_page(req.remote_addr, data_span);

            zlink::mem_response resp;
            resp.status = zlink::error_code::ok;
            resp.size = data_span.size();
            resp.remote_addr = req.remote_addr;
            resp_frame.payload.resize(sizeof(resp));
            std::memcpy(resp_frame.payload.data(), &resp, sizeof(resp));
            transport.send(resp_frame);
        }
        else if (req.op == zlink::mem_op::host_read) {
            // Client wants to read back from mirrored memory
            auto mirror_addr = g_host_mirror.translate(req.remote_addr);

            zlink::mem_response resp;
            resp.status = zlink::error_code::ok;
            resp.remote_addr = req.remote_addr;

            if (mirror_addr) {
                resp.size = req.size;
                resp_frame.payload.resize(sizeof(resp) + req.size);
                std::memcpy(resp_frame.payload.data(), &resp, sizeof(resp));
                std::memcpy(resp_frame.payload.data() + sizeof(resp),
                           reinterpret_cast<void*>(*mirror_addr), req.size);
            } else {
                resp.size = 0;
                resp.status = zlink::error_code::not_found;
                resp_frame.payload.resize(sizeof(resp));
                std::memcpy(resp_frame.payload.data(), &resp, sizeof(resp));
            }

            transport.send(resp_frame);
        }
    }

    void handle_rpc_request(zlink::transport& transport, const zlink::frame& req_frame) {
        using namespace zpp::bits;

        // Load request data into a fresh context
        auto [data, in, out] = data_in_out();
        data.assign(req_frame.payload.begin(), req_frame.payload.end());

        // Create server context and dispatch
        cuda_gen::cuda_gen_rpc::server server{in, out};

        // Reset handle production flag before dispatch
        g_has_produced_handle = false;

        // serve() deserializes the function index, calls the bound handler,
        // and serializes the response into the data buffer
        auto result = server.serve();
        if (failure(result)) {
            std::cerr << "[server] RPC dispatch failed\n";
            // Send error response
            zlink::frame resp_frame;
            resp_frame.call_id = req_frame.call_id;
            resp_frame.type = zlink::frame_type::request;
            // Return a generic error
            int32_t error_result = static_cast<int32_t>(CUDA_ERROR_INVALID_VALUE);
            resp_frame.payload.resize(sizeof(error_result));
            std::memcpy(resp_frame.payload.data(), &error_result, sizeof(error_result));
            transport.send(resp_frame);
            return;
        }

        // Send the serialized response back
        zlink::frame resp_frame;
        resp_frame.call_id = req_frame.call_id;
        resp_frame.type = zlink::frame_type::request;
        resp_frame.payload.assign(data.begin(), data.end());
        transport.send(resp_frame);
    }

    void handle_pipeline_request(zlink::transport& transport, const zlink::frame& req_frame) {
        // Pipeline request: batch of serialized RPC calls
        // Format: [4B count] [4B len1][req1_data] [4B len2][req2_data] ...
        // Process each call in order, serialize responses similarly

        if (req_frame.payload.size() < 4) return;

        std::uint32_t count = 0;
        std::memcpy(&count, req_frame.payload.data(), 4);

        std::size_t offset = 4;

        // Response buffer
        std::vector<std::byte> resp_payload;
        std::uint32_t resp_count = 0;
        resp_payload.resize(4); // Space for count

        for (std::uint32_t i = 0; i < count && offset + 4 <= req_frame.payload.size(); i++) {
            std::uint32_t len = 0;
            std::memcpy(&len, req_frame.payload.data() + offset, 4);
            offset += 4;

            if (offset + len > req_frame.payload.size()) break;

            // Dispatch this individual request
            using namespace zpp::bits;
            auto [data, in, out] = data_in_out();
            data.assign(req_frame.payload.begin() + offset,
                        req_frame.payload.begin() + offset + len);

            g_has_produced_handle = false;

            cuda_gen::cuda_gen_rpc::server server{in, out};
            auto result = server.serve();

            offset += len;

            // Serialize response
            std::uint32_t resp_len = static_cast<std::uint32_t>(data.size());
            std::size_t old_size = resp_payload.size();
            resp_payload.resize(old_size + 4 + resp_len);
            std::memcpy(resp_payload.data() + old_size, &resp_len, 4);
            std::memcpy(resp_payload.data() + old_size + 4, data.data(), resp_len);
            resp_count++;
        }

        // Write response count
        std::memcpy(resp_payload.data(), &resp_count, 4);

        zlink::frame resp_frame;
        resp_frame.call_id = req_frame.call_id;
        resp_frame.type = zlink::frame_type::pipeline_response;
        resp_frame.payload = resp_payload;
        transport.send(resp_frame);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    std::cerr << "zlink CUDA RPC Server\n";
    std::cerr << "  Covers all CUDA Driver API functions (CUDA 12.4+)\n";
    std::cerr << "  Client/server from same zlink build — no protocol versioning\n";

    // Initialize CUDA
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "[server] cuInit failed: " << res << "\n";
        return 1;
    }

    int device_count = 0;
    cuDeviceGetCount(&device_count);
    std::cerr << "[server] Found " << device_count << " CUDA device(s)\n";

    if (device_count == 0) {
        std::cerr << "[server] No CUDA devices found\n";
        return 1;
    }

    cuda_rpc_server server;
    auto transport = zlink::make_transport(zlink::transport_kind::tcp);
    server.serve(*transport);

    return 0;
}
