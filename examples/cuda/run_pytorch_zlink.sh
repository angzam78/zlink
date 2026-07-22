#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# run_pytorch_zlink.sh — Run PyTorch GPU workloads over zlink
#
# This script sets up LD_LIBRARY_PATH so that PyTorch finds the
# zlink CUDA shims (libcuda.so.1, libnvidia-ml.so.1) instead of
# the real NVIDIA libraries. The shims intercept all CUDA Driver API
# and NVML calls and forward them to a remote GPU server.
#
# PREREQUISITES:
#   1. PyTorch GPU edition installed (pip install torch --index-url ...)
#   2. zlink built with CUDA examples: cmake -DZLINK_CUDA_EXAMPLES=ON ..
#   3. cuda_server running on the GPU machine
#
# USAGE:
#   # Start the server on the GPU machine:
#   ./cuda_server
#
#   # On the CPU client machine:
#   export ZLINK_SERVER=gpu-machine:14833
#   ./run_pytorch_zlink.sh python my_script.py
#   ./run_pytorch_zlink.sh python test_pytorch_zlink.py
#
# ARCHITECTURE:
#   ┌──────────────────┐         TCP          ┌──────────────────┐
#   │ CPU Client       │ ───────────────────── │ GPU Server       │
#   │                  │                       │                  │
#   │  PyTorch GPU     │                       │  cuda_server     │
#   │    │             │                       │    │             │
#   │  libcuda.so.1    │   RPC over TCP        │  real libcuda.so │
#   │  (zlink shim)    │ ═════════════════════ │  (CUDA driver)   │
#   │    │             │                       │    │             │
#   │  libnvidia-ml    │                       │  real GPU HW     │
#   │  (zlink shim)    │                       │                  │
#   └──────────────────┘                       └──────────────────┘
#
# The LD_LIBRARY_PATH approach is superior to LD_PRELOAD because:
#   - Works on machines with NO CUDA installed at all
#   - PyTorch naturally finds "libcuda.so.1" via the dynamic linker
#   - No need for fake library symlinks or dlopen tricks
#   - Multiple processes inherit the environment automatically
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../build"

# ── Find or create the zlink-cuda-runtime directory ────────────────────
# This directory contains:
#   libcuda.so.1        — CUDA Driver API shim
#   libcuda.so          → libcuda.so.1
#   libnvidia-ml.so.1   — NVML shim
#   libnvidia-ml.so     → libnvidia-ml.so.1

RUNTIME_DIR=""

# First, check if the install target has been run
if [ -d "${BUILD_DIR}/zlink-cuda-runtime" ]; then
    RUNTIME_DIR="${BUILD_DIR}/zlink-cuda-runtime"
fi

# If not, look for the individual libraries and create the runtime dir
if [ -z "${RUNTIME_DIR}" ]; then
    LIBCUDA=$(find "${BUILD_DIR}" -name "libcuda.so.1" -print -quit 2>/dev/null || true)
    LIBNVML=$(find "${BUILD_DIR}" -name "libnvidia-ml.so.1" -print -quit 2>/dev/null || true)

    if [ -z "${LIBCUDA}" ]; then
        # Fallback: broader search pattern
        LIBCUDA=$(find "${BUILD_DIR}" -name "libcuda.so*" -print -quit 2>/dev/null || true)
    fi

    if [ -z "${LIBCUDA}" ]; then
        echo "ERROR: libcuda.so.1 (zlink CUDA shim) not found in ${BUILD_DIR}"
        echo "Build with: cd build && cmake .. -DZLINK_CUDA_EXAMPLES=ON && make"
        exit 1
    fi

    # Create a runtime directory with proper names and symlinks
    RUNTIME_DIR=$(mktemp -d)
    trap 'rm -rf "${RUNTIME_DIR}"' EXIT

    LIBCUDA_DIR=$(dirname "${LIBCUDA}")

    # Copy or symlink libcuda.so.1
    if [[ "${LIBCUDA}" == *".so.1" ]]; then
        ln -sf "${LIBCUDA}" "${RUNTIME_DIR}/libcuda.so.1"
    else
        # Non-standard name: copy as libcuda.so.1
        cp "${LIBCUDA}" "${RUNTIME_DIR}/libcuda.so.1"
    fi

    # Create libcuda.so symlink
    ln -sf libcuda.so.1 "${RUNTIME_DIR}/libcuda.so"

    # Copy or symlink libnvidia-ml.so.1 if found
    if [ -n "${LIBNVML}" ]; then
        ln -sf "${LIBNVML}" "${RUNTIME_DIR}/libnvidia-ml.so.1"
        ln -sf libnvidia-ml.so.1 "${RUNTIME_DIR}/libnvidia-ml.so"
    else
        # Try to find it in the same directory as libcuda
        if [ -f "${LIBCUDA_DIR}/libnvidia-ml.so.1" ]; then
            ln -sf "${LIBCUDA_DIR}/libnvidia-ml.so.1" "${RUNTIME_DIR}/libnvidia-ml.so.1"
            ln -sf libnvidia-ml.so.1 "${RUNTIME_DIR}/libnvidia-ml.so"
        fi
        # If not found, NVML queries will fail but CUDA still works
    fi
fi

# ── Validate environment ──────────────────────────────────────────────
if [ -z "${ZLINK_SERVER:-}" ]; then
    echo "ERROR: ZLINK_SERVER environment variable not set"
    echo "Usage: export ZLINK_SERVER=hostname:port"
    echo "Example: export ZLINK_SERVER=192.168.1.100:14833"
    exit 1
fi

# ── Verify runtime directory contents ─────────────────────────────────
if [ ! -f "${RUNTIME_DIR}/libcuda.so.1" ]; then
    echo "ERROR: libcuda.so.1 not found in ${RUNTIME_DIR}"
    exit 1
fi

echo "zlink PyTorch Runner"
echo "  Runtime dir:  ${RUNTIME_DIR}"
echo "  libcuda:      $(ls -la "${RUNTIME_DIR}/libcuda.so.1" 2>/dev/null | awk '{print $NF}')"
echo "  libnvidia-ml: $(ls -la "${RUNTIME_DIR}/libnvidia-ml.so.1" 2>/dev/null | awk '{print $NF}' || echo 'not found (optional)')"
echo "  Server:       ${ZLINK_SERVER}"
echo "  Command:      $@"
echo ""

# ── Set up LD_LIBRARY_PATH ────────────────────────────────────────────
# Prepend the runtime directory so the dynamic linker finds our
# libcuda.so.1 and libnvidia-ml.so.1 BEFORE the real NVIDIA libraries.
# This works even on machines with NO CUDA installed at all — PyTorch's
# dlopen("libcuda.so.1") will find our shim instead.
export LD_LIBRARY_PATH="${RUNTIME_DIR}:${LD_LIBRARY_PATH:-}"

# ── Run the command ───────────────────────────────────────────────────
echo "Running: $@"
echo ""

"$@"
