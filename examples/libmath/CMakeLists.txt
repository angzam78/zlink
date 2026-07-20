cmake_minimum_required(VERSION 3.20)

# ── Math example: server ────────────────────────────────────────────────
add_executable(math_server math_server.cpp)
target_link_libraries(math_server PRIVATE zlink_core m)

# ── Math example: client ───────────────────────────────────────────────
add_executable(math_client math_client.cpp)
target_link_libraries(math_client PRIVATE zlink_core m)
