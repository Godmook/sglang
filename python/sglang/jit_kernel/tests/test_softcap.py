import sys

import pytest
import torch

from sglang.jit_kernel.softcap import softcap_inplace, softcap_out_fp32
from sglang.srt.layers.elementwise import fused_softcap
from sglang.srt.layers.elementwise import softcap_inplace as dispatch_softcap_inplace
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=180, suite="nightly-kernel-1-gpu", nightly=True)


def _ref(x: torch.Tensor, c: float) -> torch.Tensor:
    return torch.tanh(x.float() / c) * c


# ---- JIT kernel unit tests ------------------------------------------------


@pytest.mark.parametrize("n", [1, 7, 127, 1024, 1025, 4096, 131072])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("c", [1.0, 30.0, 50.0])
def test_softcap_out_fp32(n: int, dtype: torch.dtype, c: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=dtype) * 0.5
    out = torch.empty(n, device="cuda", dtype=torch.float32)
    softcap_out_fp32(x, out, c)
    ref = _ref(x, c)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (
        f"max err={(out - ref).abs().max().item()}"
    )


@pytest.mark.parametrize("n", [1, 7, 1024, 1025, 131072])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("c", [1.0, 30.0])
def test_softcap_inplace(n: int, dtype: torch.dtype, c: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(1)
    x = torch.randn(n, device="cuda", dtype=dtype) * 0.5
    ref = _ref(x, c).to(dtype)
    softcap_inplace(x, c)
    assert torch.allclose(x, ref, atol=1e-3, rtol=1e-3), (
        f"max err={(x.float() - ref.float()).abs().max().item()}"
    )


# ---- 2D shape (realistic logits: [batch, vocab]) --------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 256256), (4, 128256), (32, 256256), (1, 1)],
    ids=["1x256k", "4x128k", "32x256k", "1x1"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_softcap_out_fp32_2d(shape: tuple, dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=dtype) * 0.5
    out = torch.empty(shape, device="cuda", dtype=torch.float32)
    softcap_out_fp32(x, out, 30.0)
    ref = _ref(x, 30.0)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (
        f"max err={(out - ref).abs().max().item()}"
    )


@pytest.mark.parametrize("shape", [(4, 128256), (1, 256256)])
def test_softcap_inplace_2d(shape: tuple) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(7)
    x = torch.randn(shape, device="cuda", dtype=torch.float32) * 0.5
    ref = _ref(x, 30.0)
    softcap_inplace(x, 30.0)
    assert torch.allclose(x, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", [(2, 4096)])
def test_softcap_inplace_2d_bf16(shape: tuple) -> None:
    """2D in-place bf16 (common lm_head dtype path)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(11)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    ref = _ref(x, 30.0).to(torch.bfloat16)
    softcap_inplace(x, 30.0)
    assert torch.allclose(x, ref, atol=1e-2, rtol=1e-2)


# ---- 3D tensor: view(-1) flatten in Python binding ----------------------


def test_softcap_3d_contiguous() -> None:
    """Arbitrary rank contiguous tensor is flattened to 1D in jit_kernel."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(3)
    x = torch.randn(2, 4, 512, device="cuda", dtype=torch.float32) * 0.5
    out = torch.empty_like(x, dtype=torch.float32)
    ref = _ref(x, 25.0)
    softcap_out_fp32(x, out, 25.0)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    xi = x.clone()
    softcap_inplace(xi, 25.0)
    assert torch.allclose(xi, ref, atol=1e-4, rtol=1e-4)


# ---- fp32: n in (2,3) is tail-only when kVecSize is 4 (pre-Blackwell) ------


@pytest.mark.parametrize("n", [2, 3])
def test_softcap_fp32_small_n_tail_region(n: int) -> None:
    """Regression: grid must be >=1 when n < kVecSize (no full vectors)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    ref = _ref(x, 15.0)
    out = torch.empty(n, device="cuda", dtype=torch.float32)
    softcap_out_fp32(x, out, 15.0)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    xi = x.clone()
    softcap_inplace(xi, 15.0)
    assert torch.allclose(xi, ref, atol=1e-4, rtol=1e-4)


# ---- NaN / Inf edge cases -------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_softcap_no_nan_on_large_input(dtype: torch.dtype) -> None:
    """tanh saturates to +/-1 for large inputs; result should be +/-c, never NaN."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    max_val = 65000.0 if dtype == torch.float16 else 1e6
    x = torch.tensor(
        [-max_val, -100.0, -1.0, 0.0, 1.0, 100.0, max_val],
        device="cuda",
        dtype=dtype,
    )
    out = torch.empty_like(x, dtype=torch.float32)
    softcap_out_fp32(x, out, 30.0)
    assert not torch.isnan(out).any(), f"NaN detected: {out}"
    assert not torch.isinf(out).any(), f"Inf detected: {out}"
    assert (out.abs() <= 30.0 + 1e-3).all(), f"Output exceeds softcap: {out}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_softcap_inplace_no_nan_on_large_input(dtype: torch.dtype) -> None:
    """In-place path must match out-of-place stability for extreme logits."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    max_val = 65000.0 if dtype == torch.float16 else 1e6
    x = torch.tensor(
        [-max_val, -100.0, -1.0, 0.0, 1.0, 100.0, max_val],
        device="cuda",
        dtype=dtype,
    )
    ref = _ref(x, 30.0).to(dtype)
    softcap_inplace(x, 30.0)
    assert not torch.isnan(x.float()).any(), f"NaN: {x}"
    assert not torch.isinf(x.float()).any(), f"Inf: {x}"
    assert torch.allclose(x, ref, atol=1e-3, rtol=1e-3)


# ---- Zero-element tensor ---------------------------------------------------


def test_softcap_zero_elements() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.empty(0, device="cuda", dtype=torch.float32)
    out = torch.empty(0, device="cuda", dtype=torch.float32)
    softcap_out_fp32(x, out, 30.0)
    softcap_inplace(x, 30.0)


# ---- Integration: elementwise.py dispatch ----------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_softcap_integration(dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(99)
    x = torch.randn(4096, device="cuda", dtype=dtype)
    out = fused_softcap(x, 30.0)
    ref = _ref(x, 30.0)
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_softcap_inplace_integration(dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(99)
    x = torch.randn(4096, device="cuda", dtype=dtype)
    ref = _ref(x, 30.0).to(dtype)
    dispatch_softcap_inplace(x, 30.0)
    assert torch.allclose(x, ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
