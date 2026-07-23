// zlink/rpc.cpp — RPC engine implementation

#include <zlink/rpc.hpp>
#include <zlink/transport.hpp>

#include <cstring>
#include <system_error>

namespace zlink {

// ── rpc_client_base implementation ────────────────────────────────────
std::error_code rpc_client_base::send_request(std::span<const std::byte> request_data,
                                                std::vector<std::byte>& response_data) {
    std::lock_guard lock(send_mutex_);

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

} // namespace zlink
