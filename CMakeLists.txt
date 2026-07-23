cmake_minimum_required(VERSION 3.20)
project(zlink VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Dependencies ────────────────────────────────────────────────────────
include(cmake/FetchZppBits.cmake)

# ── LZ4 compression ────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED IMPORTED_TARGET liblz4)

# ── Core library ───────────────────────────────────────────────────────
add_library(zlink_core STATIC
    src/tcp_transport.cpp
    src/multiplexed_transport.cpp
    src/rpc.cpp
    src/memory_region.cpp
    src/chunk_cache.cpp
    src/ptr_map.cpp
    src/memory_page_tracker.cpp
)
target_include_directories(zlink_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(zlink_core PUBLIC
    zpp_bits
    PkgConfig::LZ4
    Threads::Threads
)
target_compile_options(zlink_core PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O2>
)

# ── Examples ───────────────────────────────────────────────────────────
add_subdirectory(examples/libmath)

# ── CUDA examples (optional — requires CUDA toolkit) ───────────────────
# Build with: cmake -DZLINK_CUDA_EXAMPLES=ON ..
option(ZLINK_CUDA_EXAMPLES "Build CUDA RPC examples (requires CUDA toolkit)" OFF)
if(ZLINK_CUDA_EXAMPLES)
    add_subdirectory(examples/cuda)
endif()

# ── Tests ──────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
