"""Softmax kernel for LLM sampling with large vocabulary sizes.

Two execution paths dispatched at runtime via `num_splits`:
  - **Fused** (num_splits=None): single-block per row
  - **Split** (num_splits>1): multi-block per row with merge

out dtype matches input dtype.  All internal computation is in fp32.

**Backend selection** (CUDA only): set environment variable
``SGLANG_FUSED_SOFTMAX_BACKEND`` to ``jit`` (default, C++/CUDA JIT) or
``triton`` (fused Triton kernels in ``jit_kernel/triton/``). The Triton path
allocates fp32 internally then casts to the output dtype to match the JIT API.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# "jit" = C++/CUDA JIT (default). "triton" = fused Triton kernels.
FUSED_SOFTMAX_BACKEND_ENV = "SGLANG_FUSED_SOFTMAX_BACKEND"


def fused_softmax_backend() -> str:
    v = os.environ.get(FUSED_SOFTMAX_BACKEND_ENV, "jit").strip().lower()
    if v == "triton":
        return "triton"
    return "jit"


@lru_cache(maxsize=1)
def _triton_fused_module():
    try:
        from sglang.jit_kernel.triton import fused_temperature_softmax as m

        return m
    except ImportError as e:
        logger.warning(
            "Triton fused softmax not available (%s); falling back to JIT softmax.",
            e,
        )
        return None


@cache_once
def _jit_softmax_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(is_arch_support_pdl(), dtype)
    return load_jit(
        "softmax",
        *args,
        cuda_files=["elementwise/softmax.cuh"],
        cuda_wrappers=[("softmax", f"SoftmaxKernel<{args}>::run")],
    )


def can_use_softmax_sampling(logits: torch.Tensor) -> bool:
    dtype = logits.dtype
    return (
        logits.is_cuda
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
        and (logits.shape[-1] * dtype.itemsize) % 16 == 0
    )


def triton_softmax_sampling(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton fused temperature + softmax. Output dtype matches ``out`` / logits.

    Uses fp32 internally then casts into ``out`` so behavior matches
    :func:`softmax_sampling` for float16/bfloat16 logits.
    """
    mod = _triton_fused_module()
    if mod is None:
        raise RuntimeError("Triton fused softmax is not available (import failed).")

    if out is None:
        out = torch.empty_like(logits)

    probs_fp32 = mod.fused_temperature_softmax(logits, temperatures)
    out.copy_(probs_fp32)
    return out


def maybe_warmup_triton_fused_softmax(
    *,
    vocab_size: int,
    logits_dtype: torch.dtype,
    tp_group=None,
) -> None:
    """Compile/autotune Triton fused softmax when that backend is selected."""
    if fused_softmax_backend() != "triton":
        return
    mod = _triton_fused_module()
    if mod is None:
        return
    mod.warmup_fused_temperature_softmax(
        vocab_size,
        device=torch.cuda.current_device(),
        logits_dtype=logits_dtype,
        tp_group=tp_group,
    )


def softmax_sampling(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute softmax with fused temperature scaling for sampling.

    Parameters
    ----------
    logits : torch.Tensor
        Input logits of shape ``(batch_size, vocab_size)``.
        Supported dtypes: float16, bfloat16, float32.
    temperatures : torch.Tensor
        Per-row temperature of shape ``(batch_size,)`` in float32.
        Must be > 0 for all elements.
    out : torch.Tensor, optional
        Pre-allocated out tensor of shape ``(batch_size, vocab_size)``
        with the same dtype as logits. If None, one is allocated.

    Returns
    -------
    torch.Tensor
        Probability distribution of shape ``(batch_size, vocab_size)``
        with the same dtype as logits.
    """
    if fused_softmax_backend() == "triton":
        mod = _triton_fused_module()
        if mod is not None:
            return triton_softmax_sampling(logits, temperatures, out=out)

    if out is None:
        out = torch.empty_like(logits)
    module = _jit_softmax_module(logits.dtype)
    module.softmax(logits, out, temperatures, 0.0, 0)
    return out
