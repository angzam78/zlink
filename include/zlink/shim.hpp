#pragma once
// zlink/shim.hpp — Dynamic library shim generation and LD_PRELOAD utilities
//
// This is the key component that makes zlink work WITHOUT codegen.
// Instead of generating stub source files (like Lupine does), we:
//
//   1. Use dlopen/dlsym to discover symbols in the target library
//   2. Generate shim function pointers at runtime
//   3. Each shim serializes its arguments and forwards to the RPC client
//
// The shim .so is created by:
//   a) A generic shim library (libzlink_shim.so) that intercepts dlsym
//   b) A config file (zlink.toml) describing which library to remote
//   c) LD_PRELOAD to inject the shim before the real library
//
// This approach means ZERO code generation — just config and go!

#include <zlink/config.hpp>
#include <zlink/rpc.hpp>
#include <zlink/ptr_map.hpp>

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <functional>
#include <memory>
#include <dlfcn.h>

namespace zlink {

// ── Shim configuration ────────────────────────────────────────────────
struct shim_config {
    std::string target_library;     // e.g., "libcuda.so.1"
    std::string server_host;        // e.g., "gpu-server"
    std::uint16_t server_port = default_port;

    // Symbols to intercept (empty = intercept ALL found symbols)
    std::unordered_set<std::string> intercept_symbols;

    // Symbols to NOT intercept (pass through to real library)
    std::unordered_set<std::string> passthrough_symbols;

    // Path to real library for passthrough symbols
    std::string real_library_path;

    // Load config from environment variables:
    //   ZLINK_SERVER    = host:port
    //   ZLINK_LIBRARY   = target library name
    //   ZLINK_CONFIG    = path to config file
    static shim_config from_env();

    // Load from TOML config file
    static shim_config from_file(const std::string& path);
};

// ── Symbol interceptor ─────────────────────────────────────────────────
// Manages the mapping between intercepted symbols and their RPC wrappers.
class symbol_interceptor {
public:
    explicit symbol_interceptor(const shim_config& cfg);

    // Initialize: open the real library and set up RPC client
    std::error_code init();

    // Intercept a dlsym call: if the symbol should be remoted,
    // return a wrapper function pointer; otherwise return the real one.
    void* intercept(const char* symbol);

    // Check if a symbol should be intercepted
    bool should_intercept(std::string_view symbol) const;

    // Get the real function pointer from the target library
    void* get_real(const char* symbol);

    // Register a typed RPC wrapper for a specific symbol
    template<typename Ret, typename... Args>
    void register_wrapper(const std::string& symbol,
                          std::function<Ret(Args...)> wrapper);

    // Generate wrappers for ALL exported symbols in the target library
    // that match common function patterns (return + args)
    // This is the "no codegen" magic — we use generic calling conventions
    std::error_code auto_generate_wrappers();

private:
    shim_config cfg_;
    void* real_lib_handle_ = nullptr;

    // Symbol -> wrapper function pointer
    std::unordered_map<std::string, void*> wrappers_;

    // Symbol -> real function pointer (from dlsym on real library)
    std::unordered_map<std::string, void*> real_symbols_;

    mutable std::mutex mutex_;

    // The RPC client used by all wrappers
    std::unique_ptr<transport> transport_;
    // We store a generic RPC client base for dynamic dispatch
    rpc_client_base* rpc_client_ = nullptr;
    ptr_map ptrs_;
};

// ── LD_PRELOAD shim entry points ──────────────────────────────────────
// These are the actual intercepted dlopen/dlsym functions that get
// LD_PRELOAD'd into the target application.

// Install the shim (called automatically when libzlink_shim.so is loaded)
void install_shim();

// Get the global interceptor instance
symbol_interceptor& global_interceptor();

// ── Opaque function wrapper ───────────────────────────────────────────
// A generic wrapper that can call any function through RPC using
// zpp_bits's opaque mode. Each wrapper:
//   1. Serializes all arguments
//   2. Sends an RPC request
//   3. Deserializes the response
//   4. Translates pointers using ptr_map

// Template for N-argument function wrappers
// We pre-generate wrappers for common arities (0-16 args)
// This avoids needing to parse headers at build time

template<typename Signature>
struct opaque_wrapper;

// Specialization for function pointers
template<typename Ret, typename... Args>
struct opaque_wrapper<Ret(Args...)> {
    using rpc_handler = std::function<std::vector<std::byte>(std::span<const std::byte>)>;

    static Ret call(const std::string& func_name,
                    rpc_client_base& client,
                    ptr_map& ptrs,
                    Args... args) {
        using namespace zpp::bits;

        // Serialize arguments
        auto [data, in, out] = data_in_out();
        out(args...).or_throw();

        // Send and receive
        std::vector<std::byte> response;
        auto ec = client.send_request(data, response);
        if (ec) {
            if constexpr (std::is_default_constructible_v<Ret>) {
                return Ret{};
            } else {
                throw std::system_error(ec);
            }
        }

        // Deserialize response
        data = std::move(response);
        if constexpr (!std::is_void_v<Ret>) {
            Ret result;
            in(result).or_throw();
            return result;
        }
    }
};

// ── CUDA-specific pointer translation helpers ──────────────────────────
// These handle the special pointer semantics in CUDA:
//   - CUdeviceptr values are actually uint64 handles on Linux
//   - cuMemAlloc returns a CUdeviceptr that must be mapped
//   - Host pointers passed to cuMemcpyHtoD must be sent as data

inline std::uintptr_t translate_device_ptr(std::uintptr_t local_ptr, ptr_map& ptrs) {
    if (auto remote = ptrs.to_remote(local_ptr); remote) {
        return *remote;
    }
    // Not a mapped pointer — might be a host pointer, pass through
    return local_ptr;
}

inline std::uintptr_t register_device_ptr(std::uintptr_t remote_ptr, ptr_map& ptrs) {
    return ptrs.map(remote_ptr);
}

} // namespace zlink
