// zlink/server.cpp — Server framework implementation

#include <zlink/server.hpp>
#include <zlink/tcp_transport.hpp>
#include <iostream>
#include <cstring>
#include <dlfcn.h>

namespace zlink {

// ── function_registry implementation ──────────────────────────────────
void function_registry::register_function(const std::string& name, void* real_ptr) {
    std::lock_guard lock(mutex_);
    std::uint32_t id = static_cast<std::uint32_t>(std::hash<std::string>{}(name));
    function_entry entry{name, real_ptr, id};
    entries_[id] = entry;
    name_to_id_[name] = id;
}

function_entry* function_registry::find(std::uint32_t id) {
    auto it = entries_.find(id);
    return it != entries_.end() ? &it->second : nullptr;
}

const function_entry* function_registry::find(std::uint32_t id) const {
    auto it = entries_.find(id);
    return it != entries_.end() ? &it->second : nullptr;
}

function_entry* function_registry::find_by_name(const std::string& name) {
    auto it = name_to_id_.find(name);
    if (it == name_to_id_.end()) return nullptr;
    return find(it->second);
}

std::size_t function_registry::scan_library(void* handle) {
    // TODO: Use linkmap or Elf64 to enumerate symbols
    // For now, this is a placeholder
    return 0;
}

// ── connection_handler implementation ─────────────────────────────────
connection_handler::connection_handler(std::unique_ptr<transport> tp,
                                       function_registry& registry,
                                       ptr_map& pointers,
                                       memory_server& mem_server)
    : transport_(std::move(tp))
    , registry_(registry)
    , pointers_(pointers)
    , mem_server_(mem_server) {}

void connection_handler::run() {
    while (running_.load() && transport_->is_connected()) {
        frame req;
        auto ec = transport_->receive(req);
        if (ec) break;

        // Handle memory ops
        if (req.type == frame_type::memory_op) {
            ec = handle_memory_op(std::move(req));
            if (ec) break;
            continue;
        }

        // Handle RPC requests
        ec = handle_frame(std::move(req));
        if (ec) break;
    }
}

std::error_code connection_handler::handle_frame(frame&& f) {
    // TODO: Dynamic dispatch to registered functions
    frame resp;
    resp.call_id = f.call_id;
    resp.type = frame_type::response;
    resp.payload = std::move(f.payload); // Echo for now

    return transport_->send(resp);
}

std::error_code connection_handler::handle_memory_op(frame&& f) {
    // TODO: Parse and dispatch memory operations
    return {};
}

// ── server implementation ─────────────────────────────────────────────
server::server(const server_config& cfg) : cfg_(cfg) {}

server::~server() {
    stop();
}

void server::start() {
    running_.store(true);

    // Open the target library
    real_lib_handle_ = dlopen(cfg_.target_library.c_str(), RTLD_LAZY);
    if (!real_lib_handle_) {
        std::cerr << "zlink: dlopen failed: " << dlerror() << "\n";
        throw std::runtime_error("Failed to open target library: " + cfg_.target_library);
    }

    // Scan and register all exported symbols
    std::size_t n = registry_.scan_library(real_lib_handle_);
    std::cout << "zlink server: Registered " << n << " symbols from "
              << cfg_.target_library << "\n";

    listen_transport_ = make_transport(transport_kind::tcp);
    auto ec = listen_transport_->listen(cfg_.bind_address, cfg_.port);
    if (ec) {
        throw std::system_error(ec);
    }

    std::cout << "zlink server: Listening on " << cfg_.bind_address << ":"
              << cfg_.port << "\n";

    accept_loop();
}

void server::start_async() {
    // TODO: Start in a background thread
    start();
}

void server::stop() {
    running_.store(false);

    for (auto& t : connection_threads_) {
        if (t.joinable()) t.join();
    }
    connections_.clear();
    connection_threads_.clear();

    if (listen_transport_) {
        listen_transport_->close();
    }
    if (real_lib_handle_) {
        dlclose(real_lib_handle_);
        real_lib_handle_ = nullptr;
    }
}

void server::accept_loop() {
    while (running_.load()) {
        // TODO: Accept connections from listen transport and spawn handlers
        break; // Placeholder — needs transport accept API
    }
}

} // namespace zlink
