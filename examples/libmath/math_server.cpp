// zlink/examples/libmath/math_server.cpp
//
// A minimal server that exposes libm (math) functions over the network.
// Run this on any machine, then connect with math_client from another.
//
// This demonstrates zlink's core value proposition:
//   - No codegen needed — just register functions
//   - Type-safe serialization via zpp_bits
//   - Works with ANY shared library

#include <zlink/rpc.hpp>
#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/config.hpp>

#include <cmath>
#include <iostream>
#include <cstdlib>

// ── Step 1: Define the remote API surface ──────────────────────────────
// Using zpp_bits's native RPC binding — no codegen, no config files.
// Just declare which functions to expose.

namespace math_api {

double sin(double x) { return std::sin(x); }
double cos(double x) { return std::cos(x); }
double tan(double x) { return std::tan(x); }
double sqrt(double x) { return std::sqrt(x); }
double pow(double base, double exp) { return std::pow(base, exp); }
double exp(double x) { return std::exp(x); }
double log(double x) { return std::log(x); }
double log2(double x) { return std::log2(x); }
double fabs(double x) { return std::fabs(x); }
double fmod(double x, double y) { return std::fmod(x, y); }
double ceil(double x) { return std::ceil(x); }
double floor(double x) { return std::floor(x); }

} // namespace math_api

// ── Step 2: Create the RPC type ────────────────────────────────────────
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

// ── Step 3: Serve ─────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::uint16_t port = zlink::default_port;
    if (argc > 1) port = static_cast<std::uint16_t>(std::stoi(argv[1]));

    std::cout << "zlink math server — exposing libm functions over RPC\n"
              << "Listening on port " << port << "...\n" << std::endl;

    auto tp = zlink::make_transport(zlink::transport_kind::tcp);
    auto ec = tp->listen("0.0.0.0", port);
    if (ec) {
        std::cerr << "Failed to listen: " << ec.message() << "\n";
        return 1;
    }

    ec = tp->accept();
    if (ec) {
        std::cerr << "Failed to accept: " << ec.message() << "\n";
        return 1;
    }

    std::cout << "Client connected. Serving math RPC calls...\n";

    // Serve requests in a loop.
    // CRITICAL: Create fresh data_in_out() per request — zpp_bits archives
    // track internal positions that become stale when data is reassigned.
    while (tp->is_connected()) {
        try {
            // Receive a frame from the client
            zlink::frame req_frame;
            ec = tp->receive(req_frame);
            if (ec) {
                std::cerr << "Receive error: " << ec.message() << "\n";
                break;
            }

            // Create fresh zpp_bits context for this request
            auto [data, in, out] = zpp::bits::data_in_out();
            data.assign(req_frame.payload.begin(), req_frame.payload.end());

            math_rpc::server server{in, out};
            auto result = server.serve();
            if (zpp::bits::failure(result)) {
                std::cerr << "Serve error\n";
                break;
            }

            // Send the response back
            zlink::frame resp_frame;
            resp_frame.call_id = req_frame.call_id;
            resp_frame.type = zlink::frame_type::response;
            resp_frame.payload.assign(data.begin(), data.end());

            ec = tp->send(resp_frame);
            if (ec) {
                std::cerr << "Send error: " << ec.message() << "\n";
                break;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            break;
        }
    }

    std::cout << "Server shutting down.\n";
    return 0;
}
