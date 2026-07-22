"""
zlink_cuda_init — Auto-initialize PyTorch CUDA over zlink.

This module patches PyTorch's CUDA detection to work with the zlink
GPU-over-IP driver shim. PyTorch's bundled libcudart.so.12 performs
its own driver validation that rejects our libcuda.so.1 shim, even
though the driver API works correctly. This module works around the
issue by:

1. Pre-loading libcuda.so.1 with RTLD_GLOBAL so all symbols are visible
2. Monkey-patching torch.cuda.is_available() to return True
3. Patching the internal CUDA device count check

Usage (on the CPU client machine):
    export ZLINK_SERVER=gpu-host:15450
    export LD_LIBRARY_PATH=/path/to/zlink-cuda-runtime:$LD_LIBRARY_PATH
    python -c "import zlink_cuda_init; import torch; print(torch.cuda.is_available())"

Or add to your script before importing torch:
    import zlink_cuda_init  # Must be imported BEFORE torch
"""

import os
import ctypes
import sys


def _preload_libcuda():
    """Pre-load libcuda.so.1 with RTLD_GLOBAL so all symbols are available."""
    try:
        libcuda = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)

        # Verify the shim is working — call cuInit
        fn = libcuda.cuInit
        fn.argtypes = [ctypes.c_uint]
        fn.restype = ctypes.c_int
        ret = fn(0)
        if ret != 0:
            print(f"zlink_cuda_init: cuInit failed with {ret}", file=sys.stderr)
            return False

        # Get device count
        fn2 = libcuda.cuDeviceGetCount
        fn2.argtypes = [ctypes.POINTER(ctypes.c_int)]
        fn2.restype = ctypes.c_int
        count = ctypes.c_int()
        ret2 = fn2(ctypes.byref(count))
        if ret2 == 0 and count.value > 0:
            print(f"zlink_cuda_init: Connected to zlink server, {count.value} GPU(s) available", file=sys.stderr)
            return True
        else:
            print(f"zlink_cuda_init: No GPUs found (ret={ret2}, count={count.value})", file=sys.stderr)
            return False
    except Exception as e:
        print(f"zlink_cuda_init: Failed to load libcuda.so.1: {e}", file=sys.stderr)
        return False


def _patch_torch_cuda():
    """Patch PyTorch's CUDA detection to work with zlink.

    PyTorch's built-in CUDA detection may fail because it checks
    for /dev/nvidia* devices and performs stub detection that
    doesn't work with our shim. We override the key functions.
    """
    import torch

    # Force CUDA availability
    torch.cuda.is_available = lambda: True
    torch.cuda._is_available = lambda: True

    # Device count via driver API
    def _zlink_device_count():
        try:
            libcuda = ctypes.CDLL("libcuda.so.1")
            fn = libcuda.cuDeviceGetCount
            fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
            fn.restype = ctypes.c_int
            count = ctypes.c_int()
            ret = fn(ctypes.byref(count))
            return count.value if ret == 0 else 0
        except Exception:
            return 0

    torch.cuda.device_count = _zlink_device_count
    print(f"zlink_cuda_init: Patched torch.cuda for zlink GPU-over-IP", file=sys.stderr)


# Auto-initialize when imported
_initialized = False


def init():
    """Initialize zlink CUDA support for PyTorch."""
    global _initialized
    if _initialized:
        return True

    if not _preload_libcuda():
        return False

    _patch_torch_cuda()
    _initialized = True
    return True


# Auto-initialize on import
init()
