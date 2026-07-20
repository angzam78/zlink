#pragma once
// zlink/client.hpp — Client framework for connecting to a zlink server
//
// The client provides:
//   1. A typed RPC client for making remote calls
//   2. A pointer map for handle translation
//   3. A memory client for remote memory operations
//   4. A shim layer for LD_PRELOAD interception
//
// Typical usage:
//   zlink::client cl("gpu-server:14833", "libcuda.so.1");
//   cl.connect();
//   auto result = cl.call<CUDA_API_cuInit>(0);

#include <zlink/config.hpp>
#include <zlink/rpc.hpp>
#include <zlink/transport.hpp>
#include <zlink/ptr_map.hpp>
#include <zlink/memory.hpp>
#include <zlink/shim.hpp>

#include <string>
#include <memory>
#include <atomic>

namespace zlink {

struct client_config {
    std::string server_host;
    std::uint16_t server_port = default_port;
    std::string target_library;
    bool auto_reconnect = true;
    int reconnect_timeout_ms = 5000;

    static client_config from_env();
};

class client {
public:
    explicit client(const client_config& cfg);
    ~client();

    // Connect to the server
    std::error_code connect();

    // Disconnect
    void disconnect();

    // Check if connected
    bool is_connected() const;

    // Get the underlying transport
    transport& get_transport() { return *transport_; }

    // Get the RPC client base (for dynamic dispatch)
    rpc_client_base& rpc() { return *rpc_client_; }

    // Get the pointer map
    ptr_map& pointers() { return ptrs_; }

    // Get the memory client
    memory_client& memory() { return *mem_client_; }

    // Make a typed RPC call
    template<typename RpcDef, std::size_t FuncIndex, typename... Args>
    auto call(Args&&... args) {
        auto* typed = dynamic_cast<rpc_client<RpcDef>*>(rpc_client_.get());
        if (!typed) {
            throw std::runtime_error("RPC client type mismatch");
        }
        return typed->template call<FuncIndex>(std::forward<Args>(args)...);
    }

private:
    client_config cfg_;
    std::unique_ptr<transport> transport_;
    std::unique_ptr<rpc_client_base> rpc_client_;
    ptr_map ptrs_;
    std::unique_ptr<memory_client> mem_client_;
    std::atomic<bool> connected_{false};
};

} // namespace zlink
