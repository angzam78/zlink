cmake_minimum_required(VERSION 3.20)

# ── CUDA examples ─────────────────────────────────────────────────────
# Single canonical server (codegen-based, 269+ CUDA Driver API functions)
# Single canonical shim (codegen-based, 269+ functions, replaces libcuda.so.1)
# Plus: NVML shim, CUDA Runtime shim, pipeline client
#
# Build scenarios:
#   GPU machine:  cmake -DZLINK_CUDA_EXAMPLES=ON ..  (builds server + client)
#   CPU machine:  cmake -DZLINK_CUDA_EXAMPLES=ON ..  (builds shim + client)

find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    # ── CUDA server (codegen-based, 269+ functions) ────────────────────
    # Runs on the GPU machine. Receives RPC calls from the shim/client
    # and executes them on real CUDA hardware.
    add_executable(cuda_server cuda_server.cpp)
    target_link_libraries(cuda_server PRIVATE
        zlink_core
        CUDA::cuda_driver
    )
    target_include_directories(cuda_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # ── CUDA pipeline client ──────────────────────────────────────────
    # Uses virtual handles for batching, eliminates barriers between
    # enqueued operations. Demonstrates the key zlink performance win.
    add_executable(cuda_client cuda_client.cpp)
    target_link_libraries(cuda_client PRIVATE
        zlink_core
    )
    target_include_directories(cuda_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
else()
    message(STATUS "CUDA toolkit not found — server target skipped")
    message(STATUS "Build the server on the GPU machine with CUDA installed")

    # Client can still build on CPU-only machines (doesn't need CUDA)
    add_executable(cuda_client cuda_client.cpp)
    target_link_libraries(cuda_client PRIVATE
        zlink_core
    )
    target_include_directories(cuda_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# ════════════════════════════════════════════════════════════════════════
# libcuda.so.1 — zlink CUDA Driver API shim
# ════════════════════════════════════════════════════════════════════════
# This is the KEY deliverable: a shared library that replaces libcuda.so.1
# on the CPU client machine. When PyTorch or any CUDA application calls
# dlopen("libcuda.so.1"), the dynamic linker finds THIS library instead
# (because the directory containing it is first in LD_LIBRARY_PATH).
#
# The shim:
#   - Exports all 269+ cu* symbols (generated from codegen/gen_shim.inc)
#   - Intercepts every call, translates virtual handles, and forwards
#     via zlink RPC to cuda_server on the remote GPU machine
#   - Does NOT link against the real CUDA driver — it REPLACES it
#   - Defines its own CUDA type stubs (no cuda.h dependency)
#   - Can be built on CPU-only machines (no CUDA toolkit needed)
#
# Usage on the CPU client:
#   export ZLINK_SERVER=gpu-machine:14833
#   export LD_LIBRARY_PATH=/path/to/zlink-cuda-runtime:$LD_LIBRARY_PATH
#   python my_script.py
#
# The zlink-cuda-runtime directory contains:
#   libcuda.so.1       (this shim)
#   libcuda.so         → libcuda.so.1   (symlink)
#   libnvidia-ml.so.1  (NVML shim)
#   libnvidia-ml.so    → libnvidia-ml.so.1 (symlink)
#   libcudart.so.12    (CUDA Runtime shim)
#   libcudart.so       → libcudart.so.12  (symlink)
add_library(zlink_cuda_shim SHARED cuda_shim.cpp)
target_link_libraries(zlink_cuda_shim PRIVATE
    zlink_core
)
target_include_directories(zlink_cuda_shim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# The shim must export all cu* symbols with default visibility
# so that PyTorch can dlsym them after dlopen("libcuda.so.1")
set_target_properties(zlink_cuda_shim PROPERTIES
    OUTPUT_NAME "cuda"
    PREFIX "lib"
    SOVERSION "1"
    VERSION "1.0.0"
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET "default"
    VISIBILITY_INLINES_HIDDEN OFF
)
# Allow undefined symbols — the shim replaces libcuda.so, it doesn't link to it
target_link_options(zlink_cuda_shim PRIVATE
    -Wl,--allow-shlib-undefined
)

# ════════════════════════════════════════════════════════════════════════
# libnvidia-ml.so.1 — zlink NVML shim
# ════════════════════════════════════════════════════════════════════════
# Minimal NVML (NVIDIA Management Library) shim. PyTorch and tools like
# nvidia-smi load libnvidia-ml.so.1 to query GPU status, memory usage,
# device names, etc. Without this shim, those queries fail silently or
# raise errors.
#
# This shim returns reasonable defaults (1 GPU, server-reported memory)
# and forwards queries to the zlink server where possible.
add_library(zlink_nvidia_ml_shim SHARED nvidia_ml_shim.cpp)
target_link_libraries(zlink_nvidia_ml_shim PRIVATE
    zlink_core
)
target_include_directories(zlink_nvidia_ml_shim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(zlink_nvidia_ml_shim PROPERTIES
    OUTPUT_NAME "nvidia-ml"
    PREFIX "lib"
    SOVERSION "1"
    VERSION "1.0.0"
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET "default"
    VISIBILITY_INLINES_HIDDEN OFF
)
target_link_options(zlink_nvidia_ml_shim PRIVATE
    -Wl,--allow-shlib-undefined
)

# ════════════════════════════════════════════════════════════════════════
# libcudart.so.12 — CUDA Runtime API shim
# ════════════════════════════════════════════════════════════════════════
# PyTorch and CUDA applications can also load libcudart.so.12 (the
# CUDA Runtime API library). This shim translates cuda* Runtime API
# calls into cu* Driver API calls, which then go through the driver shim.
#
# Chain: PyTorch → libcudart.so.12 (this shim) → libcuda.so.1 (driver shim)
#        → zlink RPC → cuda_server → real GPU
add_library(zlink_cudart_shim SHARED cudart_shim.cpp)
target_link_libraries(zlink_cudart_shim PRIVATE
    zlink_core
)
target_include_directories(zlink_cudart_shim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(zlink_cudart_shim PROPERTIES
    OUTPUT_NAME "cudart"
    PREFIX "lib"
    SOVERSION "12"
    VERSION "12.0.0"
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET "default"
    VISIBILITY_INLINES_HIDDEN OFF
)
target_link_options(zlink_cudart_shim PRIVATE
    -Wl,--allow-shlib-undefined
    -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/libcudart.version
)

# ════════════════════════════════════════════════════════════════════════
# Install target: creates the zlink-cuda-runtime directory
# ════════════════════════════════════════════════════════════════════════
# After `cmake --install .`, the user gets:
#   ${CMAKE_INSTALL_PREFIX}/zlink-cuda-runtime/
#     libcuda.so.1        (CUDA Driver shim)
#     libcuda.so          → libcuda.so.1
#     libnvidia-ml.so.1   (NVML shim)
#     libnvidia-ml.so     → libnvidia-ml.so.1
#     libcudart.so.12     (CUDA Runtime shim)
#     libcudart.so        → libcudart.so.12
#
# Deploy this directory to the CPU client machine and:
#   export LD_LIBRARY_PATH=/path/to/zlink-cuda-runtime:$LD_LIBRARY_PATH
#   export ZLINK_SERVER=gpu-host:14833
#   python my_script.py

install(TARGETS zlink_cuda_shim zlink_nvidia_ml_shim zlink_cudart_shim
    LIBRARY DESTINATION zlink-cuda-runtime
)

# Create versioned symlinks at install time
install(CODE "
    set(_dir \"\${CMAKE_INSTALL_PREFIX}/zlink-cuda-runtime\")
    # libcuda.so → libcuda.so.1
    execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
        libcuda.so.1 \"\${_dir}/libcuda.so\")
    # libnvidia-ml.so → libnvidia-ml.so.1
    execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
        libnvidia-ml.so.1 \"\${_dir}/libnvidia-ml.so\")
    # libcudart.so → libcudart.so.12
    execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
        libcudart.so.12 \"\${_dir}/libcudart.so\")
    message(STATUS \"Created symlinks in \${_dir}\")
    message(STATUS \"Deploy this directory to CPU client and set LD_LIBRARY_PATH\")
")
