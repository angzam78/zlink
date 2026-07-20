#!/usr/bin/env python3
"""
zlink PyTorch test — Verify that PyTorch GPU works over zlink.

This script tests the full PyTorch → CUDA Driver API → zlink → GPU pipeline.

Usage (on the CPU client machine):
    export ZLINK_SERVER=gpu-server:9876
    LD_PRELOAD=/path/to/cuda_shim_pytorch.so python test_pytorch_zlink.py

This script should be run AFTER starting cuda_server_v4 on the GPU machine:
    ./cuda_server_v4 9876
"""

import torch
import sys

def test_basic_cuda():
    """Test 1: Basic CUDA initialization and device query"""
    print("=" * 60)
    print("Test 1: Basic CUDA initialization")
    print("=" * 60)

    assert torch.cuda.is_available(), "CUDA not available via zlink!"
    print(f"  ✓ torch.cuda.is_available() = True")

    device_count = torch.cuda.device_count()
    print(f"  ✓ Device count: {device_count}")

    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        print(f"  ✓ Device {i}: {name}")

    cap = torch.cuda.get_device_capability(0)
    print(f"  ✓ Device capability: {cap}")

    mem_alloc = torch.cuda.memory_allocated(0)
    mem_resv = torch.cuda.memory_reserved(0)
    print(f"  ✓ Memory: allocated={mem_alloc}, reserved={mem_resv}")

    print("Test 1: PASSED\n")
    return True


def test_tensor_operations():
    """Test 2: Tensor creation and basic operations on GPU"""
    print("=" * 60)
    print("Test 2: Tensor operations on GPU")
    print("=" * 60)

    # Create tensors on GPU
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    print(f"  ✓ Created tensor a on cuda: {a}")

    b = torch.tensor([5.0, 6.0, 7.0, 8.0], device='cuda')
    print(f"  ✓ Created tensor b on cuda: {b}")

    # Basic arithmetic
    c = a + b
    print(f"  ✓ a + b = {c}")

    d = a * b
    print(f"  ✓ a * b = {d}")

    e = torch.matmul(a.unsqueeze(0), b.unsqueeze(0).T)
    print(f"  ✓ matmul = {e.item()}")

    # Transfer back to CPU
    c_cpu = c.cpu()
    expected = torch.tensor([6.0, 8.0, 10.0, 12.0])
    assert torch.allclose(c_cpu, expected), f"Expected {expected}, got {c_cpu}"
    print(f"  ✓ Result verified on CPU: {c_cpu}")

    print("Test 2: PASSED\n")
    return True


def test_matrix_multiplication():
    """Test 3: Larger matrix multiplication (triggers cuLaunchKernel)"""
    print("=" * 60)
    print("Test 3: Matrix multiplication")
    print("=" * 60)

    size = 256
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')

    print(f"  ✓ Created {size}x{size} matrices on cuda")

    c = torch.mm(a, b)
    print(f"  ✓ Matrix multiply completed")

    # Verify dimensions
    assert c.shape == (size, size), f"Expected ({size},{size}), got {c.shape}"
    print(f"  ✓ Output shape: {c.shape}")

    # Spot check: transfer a few values
    c_cpu = c.cpu()
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    expected = torch.mm(a_cpu, b_cpu)
    assert torch.allclose(c_cpu, expected, atol=1e-3), "Result mismatch!"
    print(f"  ✓ Results match CPU computation (atol=1e-3)")

    print("Test 3: PASSED\n")
    return True


def test_memory_operations():
    """Test 4: Memory allocation, copy, and synchronization"""
    print("=" * 60)
    print("Test 4: Memory operations")
    print("=" * 60)

    # Allocate and fill
    x = torch.zeros(1024, device='cuda')
    print(f"  ✓ Allocated zeros: shape={x.shape}")

    x.fill_(42.0)
    print(f"  ✓ Filled with 42.0")

    # Copy to CPU
    x_cpu = x.cpu()
    assert (x_cpu == 42.0).all(), "Fill value mismatch!"
    print(f"  ✓ Verified fill on CPU")

    # Copy CPU → GPU
    y = torch.tensor([1.0, 2.0, 3.0], device='cpu')
    y_gpu = y.to('cuda')
    print(f"  ✓ CPU → GPU copy: {y_gpu}")

    # Copy GPU → CPU
    z = y_gpu.to('cpu')
    assert torch.allclose(y, z), "Round-trip copy mismatch!"
    print(f"  ✓ GPU → CPU copy verified")

    # Free memory
    del x, y_gpu
    torch.cuda.empty_cache()
    print(f"  ✓ Memory freed and cache emptied")

    print("Test 4: PASSED\n")
    return True


def test_stream_operations():
    """Test 5: CUDA stream operations"""
    print("=" * 60)
    print("Test 5: Stream operations")
    print("=" * 60)

    # Default stream
    s0 = torch.cuda.default_stream(0)
    print(f"  ✓ Default stream: {s0}")

    # Create a new stream
    s1 = torch.cuda.Stream()
    print(f"  ✓ Created stream: {s1}")

    # Use the stream
    with torch.cuda.stream(s1):
        a = torch.randn(128, 128, device='cuda')
        b = torch.randn(128, 128, device='cuda')
        c = torch.mm(a, b)
    print(f"  ✓ Computation on custom stream completed")

    # Synchronize
    s1.synchronize()
    print(f"  ✓ Stream synchronized")

    print("Test 5: PASSED\n")
    return True


def test_neural_network():
    """Test 6: Simple neural network forward/backward pass"""
    print("=" * 60)
    print("Test 6: Neural network forward/backward pass")
    print("=" * 60)

    # Simple linear layer
    model = torch.nn.Linear(64, 10).cuda()
    print(f"  ✓ Created Linear(64, 10) on cuda")

    x = torch.randn(32, 64, device='cuda')
    print(f"  ✓ Created input: shape={x.shape}")

    # Forward pass
    y = model(x)
    print(f"  ✓ Forward pass: output shape={y.shape}")

    # Loss and backward
    loss = y.sum()
    loss.backward()
    print(f"  ✓ Backward pass completed, loss={loss.item():.4f}")

    # Check gradients exist
    assert model.weight.grad is not None, "No weight gradient!"
    print(f"  ✓ Weight gradient: shape={model.weight.grad.shape}")

    print("Test 6: PASSED\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("zlink PyTorch Integration Test")
    print("=" * 60 + "\n")

    tests = [
        test_basic_cuda,
        test_tensor_operations,
        test_matrix_multiplication,
        test_memory_operations,
        test_stream_operations,
        test_neural_network,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed! zlink + PyTorch is working correctly.\n")


if __name__ == "__main__":
    main()
