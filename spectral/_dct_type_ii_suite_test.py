import torch
import pytest
import numpy as np
from torch.autograd import gradcheck
from spectral import dct2, idct2, dct2_2d, idct2_2d, LinearDCT, DCTSpectralPooling

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dtype():
    return torch.float64

def assert_tensors_close(t1, t2, atol=1e-6, rtol=1e-5):
    torch.testing.assert_close(t1, t2, atol=atol, rtol=rtol)

# -----------------------------------------------------------------------------
# 1. FUNCTIONAL TESTS (DIMENSIONALITY & CORRECTNESS)
# -----------------------------------------------------------------------------

def test_dct2_idct2_reversibility(device, dtype):
    B, N = 4, 128
    x = torch.randn(B, N, device=device, dtype=dtype)
    y = dct2(x, norm="ortho")
    x_hat = idct2(y, norm="ortho")

    assert x.shape == y.shape == x_hat.shape, "Dimensionality check failed."
    assert_tensors_close(x, x_hat)

def test_dct2_2d_idct2_2d_reversibility(device, dtype):
    B, C, H, W = 2, 3, 32, 32
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)
    y = dct2_2d(x, norm="ortho")
    x_hat = idct2_2d(y, norm="ortho")
    
    assert x.shape == y.shape == x_hat.shape, "2D Dimensionality check failed."
    assert_tensors_close(x, x_hat)

def test_dct2_orthogonality(device, dtype):
    N = 100
    x = torch.randn(N, device=device, dtype=dtype)
    y = dct2(x, norm="ortho")
    energy_x = torch.sum(x ** 2)
    energy_y = torch.sum(y ** 2)
    
    assert_tensors_close(energy_x, energy_y)

# -----------------------------------------------------------------------------
# 2. CPU-GPU INTERFACE SAFETY & MODULE TESTS
# -----------------------------------------------------------------------------

def test_lineardct_device_movement(device, dtype):
    N = 64
    layer = LinearDCT(N, type='dct', norm='ortho').to(dtype)
    layer = layer.to(device)
    x = torch.randn(10, N, device=device, dtype=dtype)
    
    try:
        y = layer(x)
    except RuntimeError as e:
        pytest.fail(f"Device mismatch error detected: {e}")

    assert y.device == x.device, "Output tensor is not on the correct device."
    assert layer.weight.device == x.device, "Layer weights did not move to the correct device."

def test_dct_spectral_pooling_buffer_persistence(device, dtype):
    H, W = 32, 32
    keep_h, keep_w = 16, 16
    pool = DCTSpectralPooling(H, W, keep_h, keep_w).to(dtype)
    pool = pool.to(device)
    x = torch.randn(1, 1, H, W, device=device, dtype=dtype)
    
    try:
        out = pool(x)
    except RuntimeError as e:
        pytest.fail(f"Spectral Pooling Device mismatch: {e}")
        
    assert out.shape == x.shape
    assert pool.mask.device.type == device.type

# -----------------------------------------------------------------------------
# 3. AUTOGRAD & GRADIENT TESTS
# -----------------------------------------------------------------------------

def test_dct2_gradcheck(device):
    x = torch.randn(4, 16, device=device, dtype=torch.float64, requires_grad=True)
    test_func = lambda input: dct2(input, norm="ortho")
    
    assert gradcheck(test_func, (x,), eps=1e-6, atol=1e-4)

def test_idct2_gradcheck(device):
    x = torch.randn(4, 16, device=device, dtype=torch.float64, requires_grad=True)
    test_func = lambda input: idct2(input, norm="ortho")
    
    assert gradcheck(test_func, (x,), eps=1e-6, atol=1e-4)

def test_lineardct_frozen_weights(device, dtype):
    N = 32
    layer = LinearDCT(N, type='dct', norm='ortho').to(device).to(dtype)
    x = torch.randn(2, N, device=device, dtype=dtype, requires_grad=True)
    initial_weight = layer.weight.clone()
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None, "Input gradient not computed through LinearDCT."
    assert torch.norm(x.grad) > 0, "Input gradient is zero (unexpected)."
    assert layer.weight.grad is None, "LinearDCT weights should not receive gradients."
    assert_tensors_close(layer.weight, initial_weight)

def test_spectral_pooling_end_to_end_train_step(device, dtype):
    H, W = 16, 16
    pool = DCTSpectralPooling(H, W, 8, 8).to(device).to(dtype)
    x = torch.randn(2, H, W, device=device, dtype=dtype, requires_grad=True)
    out = pool(x)
    loss = out.mean()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

# -----------------------------------------------------------------------------
# 4. EQUIVALENCE TESTS
# -----------------------------------------------------------------------------

def test_functional_vs_linear_equivalence(device, dtype):
    N = 32
    x = torch.randn(10, N, device=device, dtype=dtype)
    y_func = dct2(x, norm='ortho')
    layer = LinearDCT(N, type='dct', norm='ortho').to(device).to(dtype)
    y_linear = layer(x)
    
    assert_tensors_close(y_func, y_linear)