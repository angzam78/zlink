// zlink/examples/libmath/math_client.cpp
//
// Client that calls remote math functions via zlink RPC.
// Connect to math_server running on another machine.
//
// This demonstrates that:
//   1. The client has ZERO dependency on libm — all computation is remote
//   2. Type safety is preserved — you get back double, not raw bytes
//   3. The API is identical to calling the function locally

#include <zlink/rpc.hpp>
#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/config.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>

// ── Same API declaration as the server ─────────────────────────────────
namespace math_api {
    double sin(double x);
    double cos(double x);
    double tan(double x);
    double sqrt(double x);
    double pow(double base, double exp);
    double exp(double x);
    double log(double x);
    double log2(double x);
    double fabs(double x);
    double fmod(double x, double y);
    double ceil(double x);
    double floor(double x);
}

using math_rpc = zpp::bits::rpc<
    zpp::bits::bind<&math_api::sin, 0>,
    zpp::bits::bind<&math_api::cos, 1>,
    zpp::bits::bind<&math_api::tan, 2>,
    zpp::bits::bind<&math_api::sqrt, 3>,
    zpp::bits::bind<&math_api::pow, 4>,
    zpp::bits::bind<&math_api::exp, 5>,
    zpp::bits::bind<&math_api::log, 6>,
    zpp::bits::bind<&math_api::log2, 7>,
    zpp::bits::bind<&math_api::fabs, 8>,
    zpp::bits::bind<&math_api::fmod, 9>,
    zpp::bits::bind<&math_api::ceil, 10>,
    zpp::bits::bind<&math_api::floor, 11>
>;

// ── Call helper ────────────────────────────────────────────────────────
// Each call creates a fresh zpp_bits context — the proven pattern.
template<auto FuncIndex, typename... Args>
auto remote_call(zlink::transport& tp, Args&&... args) {
    using namespace zpp::bits;

    // Serialize the request (fresh context)
    auto [req_data, req_in, req_out] = data_in_out();
    math_rpc::client req_client{req_in, req_out};
    req_client.template request<FuncIndex>(std::forward<Args>(args)...).or_throw();

    // Send the request frame
    zlink::frame req_frame;
    req_frame.call_id = 1;
    req_frame.type = zlink::frame_type::request;
    req_frame.payload.assign(req_data.begin(), req_data.end());

    auto ec = tp.send(req_frame);
    if (ec) throw std::system_error(ec);

    // Receive the response frame
    zlink::frame resp_frame;
    ec = tp.receive(resp_frame);
    if (ec) throw std::system_error(ec);

    // Deserialize the response (fresh context)
    auto [resp_data, resp_in, resp_out] = data_in_out();
    resp_data.assign(resp_frame.payload.begin(), resp_frame.payload.end());
    math_rpc::client resp_client{resp_in, resp_out};

    auto result = resp_client.template response<FuncIndex>().or_throw();
    return result;
}

int main(int argc, char* argv[]) {
    std::string host = "127.0.0.1";
    std::uint16_t port = zlink::default_port;

    if (argc > 1) host = argv[1];
    if (argc > 2) port = static_cast<std::uint16_t>(std::stoi(argv[2]));

    auto tp = zlink::make_transport(zlink::transport_kind::tcp);
    std::cout << "Connecting to " << host << ":" << port << "...\n";

    auto ec = tp->connect(host, port);
    if (ec) {
        std::cerr << "Failed to connect: " << ec.message() << "\n";
        return 1;
    }

    std::cout << "Connected! Calling remote math functions...\n\n";
    std::cout << std::fixed << std::setprecision(8);

    double result;

    result = remote_call<0>(*tp, 0.5235987755982988);
    std::cout << "remote sin(pi/6)  = " << result << "  (expected ~0.5)\n";

    result = remote_call<1>(*tp, 0.0);
    std::cout << "remote cos(0)     = " << result << "  (expected 1.0)\n";

    result = remote_call<3>(*tp, 2.0);
    std::cout << "remote sqrt(2)    = " << result << "  (expected ~1.41421356)\n";

    result = remote_call<4>(*tp, 2.0, 10.0);
    std::cout << "remote pow(2,10)  = " << result << "  (expected 1024.0)\n";

    result = remote_call<5>(*tp, 1.0);
    std::cout << "remote exp(1)     = " << result << "  (expected ~2.71828182)\n";

    result = remote_call<6>(*tp, 2.718281828459045);
    std::cout << "remote log(e)     = " << result << "  (expected ~1.0)\n";

    result = remote_call<10>(*tp, 3.14);
    std::cout << "remote ceil(3.14) = " << result << "  (expected 4.0)\n";

    result = remote_call<11>(*tp, 3.14);
    std::cout << "remote floor(3.14)= " << result << "  (expected 3.0)\n";

    std::cout << "\nAll calls completed successfully!\n";

    tp->close();
    return 0;
}
