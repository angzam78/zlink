#pragma once
// zlink/server.hpp — Server framework for serving RPC calls from a real .so
//
// The server:
//   1. dlopen's the target shared library
//   2. Accepts TCP connections from clients
//   3. Receives RPC frames, dispatches to real functions
//   4. Sends responses back
//   5. Handles remote memory operations
//
// Unlike Lupine, which generates specific server code per CUDA version,
// zlink's server is fully dynamic — it works with ANY shared library.

#include <zlink/config.hpp>
#include <zlink/rpc.hpp>
#include <zlink/transport.hpp>
#include <zlink/memory.hpp>
#include <zlink/ptr_map.hpp>

#include <string>
#include <memory>
#include <thread>
#include <vector>
#include <atomic>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <dlfcn.h>

namespace zlink {

// ── Server configuration ───────────────────────────────────────────────
struct server_config {
    std::string target_library;      // e.g., "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    std::string bind_address = "0.0.0.0";
    std::uint16_t port = default_port;
    int max_connections = 64;

    // Optional: restrict which symbols to serve
    std::unordered_set<std::string> served_symbols;

    // Enable remote memory access
    bool enable_memory_ops = true;

    static server_config from_env();
    static server_config from_file(const std::string& path);
};

// ── Function registry ──────────────────────────────────────────────────
// Maps function IDs to their real implementations loaded from the .so.
// Each entry stores the real function pointer and its zpp_bits wrapper.

struct function_entry {
    std::string  name;
    void*        real_ptr = nullptr;
    std::uint32_t id = 0;         // Stable hash of the name
};

class function_registry {
public:
    // Register a function
    void register_function(const std::string& name, void* real_ptr);

    // Look up by ID
    function_entry* find(std::uint32_t id);
    const function_entry* find(std::uint32_t id) const;

    // Look up by name
    function_entry* find_by_name(const std::string& name);

    // Scan all exported symbols from a dlopen'd library
    std::size_t scan_library(void* handle);

    std::size_t size() const { return entries_.size(); }

private:
    std::unordered_map<std::uint32_t, function_entry> entries_;
    std::unordered_map<std::string, std::uint32_t> name_to_id_;
    mutable std::mutex mutex_;
};

// ── Connection handler ─────────────────────────────────────────────────
// Each client connection gets its own handler thread.
class connection_handler {
public:
    connection_handler(std::unique_ptr<transport> tp,
                       function_registry& registry,
                       ptr_map& pointers,
                       memory_server& mem_server);

    // Run the connection loop
    void run();

    // Stop this connection
    void stop() { running_.store(false); }

    bool is_running() const { return running_.load(); }

private:
    std::unique_ptr<transport> transport_;
    function_registry& registry_;
    ptr_map& pointers_;
    memory_server& mem_server_;
    std::atomic<bool> running_{true};

    // Handle a single RPC frame
    std::error_code handle_frame(frame&& f);

    // Handle a memory operation frame
    std::error_code handle_memory_op(frame&& f);
};

// ── Main server ────────────────────────────────────────────────────────
class server {
public:
    explicit server(const server_config& cfg);
    ~server();

    // Start the server (blocking)
    void start();

    // Start in background
    void start_async();

    // Stop the server
    void stop();

    // Get the function registry (for adding custom handlers)
    function_registry& registry() { return registry_; }

    // Get the memory server (for registering memory regions)
    memory_server& memory() { return mem_server_; }

private:
    server_config cfg_;
    function_registry registry_;
    memory_server mem_server_;
    ptr_map global_ptrs_;

    void* real_lib_handle_ = nullptr;
    std::unique_ptr<transport> listen_transport_;
    std::vector<std::unique_ptr<connection_handler>> connections_;
    std::vector<std::thread> connection_threads_;
    std::atomic<bool> running_{false};
    std::mutex connections_mutex_;

    // Accept connections and spawn handlers
    void accept_loop();
};

} // namespace zlink
