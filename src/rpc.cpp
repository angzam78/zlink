// zlink/rpc.cpp — RPC engine implementation

#include <zlink/rpc.hpp>
#include <zlink/transport.hpp>

#include <cstring>
#include <system_error>
#include <stdexcept>
#include <mutex>

namespace zlink {

// ── rpc_client_base implementation ────────────────────────────────────
std::error_code rpc_client_base::send_request(std::span<const std::byte> request_data,
                                                std::vector<std::byte>& response_data) {
    std::lock_guard lock(pending_mutex_);

    std::uint32_t call_id = next_call_id_.fetch_add(1);

    frame req;
    req.call_id = call_id;
    req.type = frame_type::request;
    req.payload.assign(request_data.begin(), request_data.end());

    auto ec = transport_.send(req);
    if (ec) return ec;

    frame resp;
    ec = transport_.receive(resp);
    if (ec) return ec;

    response_data.assign(resp.payload.begin(), resp.payload.end());
    return {};
}

std::error_code rpc_client_base::send_request_async(std::span<const std::byte> request_data,
                                                      std::uint32_t& out_call_id) {
    // TODO: Implement async send with pending call tracking
    out_call_id = next_call_id_.fetch_add(1);
    return std::make_error_code(std::errc::operation_not_supported);
}

std::error_code rpc_client_base::wait_response(std::uint32_t call_id,
                                                 std::vector<std::byte>& response_data,
                                                 int timeout_ms) {
    // TODO: Implement response waiting
    return std::make_error_code(std::errc::operation_not_supported);
}

void rpc_client_base::start_receiver() {
    // TODO: Start background receiver thread for async operations
}

void rpc_client_base::stop_receiver() {
    running_.store(false);
}

// ── rpc_server_base implementation ────────────────────────────────────
std::error_code rpc_server_base::serve_one() {
    frame req;
    auto ec = transport_.receive(req);
    if (ec) return ec;

    std::vector<std::byte> response_data;
    ec = dispatch_request(req.call_id, req.payload, response_data);
    if (ec) return ec;

    frame resp;
    resp.call_id = req.call_id;
    resp.type = frame_type::response;
    resp.payload = std::move(response_data);

    return transport_.send(resp);
}

void rpc_server_base::serve_forever() {
    while (running_.load()) {
        auto ec = serve_one();
        if (ec) break;
    }
}

} // namespace zlink
