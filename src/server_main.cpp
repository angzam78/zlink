// zlink/server_main.cpp — zlink server binary entry point
//
// Usage:
//   zlink_server --library libcuda.so.1 --port 14833
//
// The server opens the target shared library, listens for TCP connections,
// and serves RPC calls by forwarding them to the real library functions.

#include <zlink/server.hpp>
#include <zlink/config.hpp>
#include <zlink/transport.hpp>

#include <iostream>
#include <string>
#include <cstdlib>
#include <csignal>
#include <cstring>

static zlink::server* g_server = nullptr;

static void signal_handler(int sig) {
    if (g_server) {
        std::cout << "\nReceived signal " << sig << ", shutting down...\n";
        g_server->stop();
    }
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  --library PATH   Path to the shared library to serve\n"
              << "  --port PORT      TCP port to listen on (default: "
              << zlink::default_port << ")\n"
              << "  --bind ADDR      Address to bind to (default: 0.0.0.0)\n"
              << "  --max-conn N     Maximum concurrent connections (default: 64)\n"
              << "  --help           Show this help message\n"
              << "\nEnvironment variables:\n"
              << "  ZLINK_LIBRARY    Target library path\n"
              << "  ZLINK_PORT       Listen port\n"
              << "  ZLINK_BIND       Bind address\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    zlink::server_config cfg;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--library" && i + 1 < argc) {
            cfg.target_library = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            cfg.port = static_cast<std::uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--bind" && i + 1 < argc) {
            cfg.bind_address = argv[++i];
        } else if (arg == "--max-conn" && i + 1 < argc) {
            cfg.max_connections = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Fall back to environment variables
    if (cfg.target_library.empty()) {
        const char* env_lib = std::getenv("ZLINK_LIBRARY");
        if (env_lib) cfg.target_library = env_lib;
    }
    if (cfg.port == zlink::default_port) {
        const char* env_port = std::getenv("ZLINK_PORT");
        if (env_port) cfg.port = static_cast<std::uint16_t>(std::stoi(env_port));
    }

    // Validate
    if (cfg.target_library.empty()) {
        std::cerr << "Error: No target library specified. Use --library or ZLINK_LIBRARY.\n";
        return 1;
    }

    // Set up signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        zlink::server srv(cfg);
        g_server = &srv;

        std::cout << "zlink server v0.1.0\n"
                  << "  Library: " << cfg.target_library << "\n"
                  << "  Listening on: " << cfg.bind_address << ":" << cfg.port << "\n"
                  << "  Max connections: " << cfg.max_connections << "\n"
                  << std::endl;

        srv.start();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    g_server = nullptr;
    return 0;
}
