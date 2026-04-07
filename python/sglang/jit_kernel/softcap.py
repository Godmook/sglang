"""JIT CUDA softcap: out = tanh(x / c) * c.

Two entry points:
  - ``softcap_inplace``:  in-place, preserves dtype
  - ``softcap_out_fp32``: out-of-place, always writes float32
"""

from __future__ import annotations

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


@cache_once
def _jit_softcap_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "softcap",
        *args,
        cuda_files=["elementwise/softcap.cuh"],
        cuda_wrappers=[
            ("softcap_inplace", f"softcap_inplace<{args}>"),
            ("softcap_out_fp32", f"softcap_out_fp32<{args}>"),
        ],
        extra_cuda_cflags=["--use_fast_math"],
    )


def softcap_inplace(tensor: torch.Tensor, softcap_const: float) -> None:
    """In-place softcap on a contiguous CUDA tensor (same dtype preserved).

    Args:
        tensor: CUDA tensor, contiguous, fp16 / bf16 / fp32.
        softcap_const: Positive scale (e.g. Gemma2 final_logit_softcapping).
    """
    flat = tensor.view(-1)
    module = _jit_softcap_module(flat.dtype)
    module.softcap_inplace(flat, float(softcap_const))


def softcap_out_fp32(
    input: torch.Tensor, output: torch.Tensor, softcap_const: float
) -> None:
    """Write softcap(input) into ``output`` in fp32 (out-of-place).

    Args:
        input: CUDA tensor, contiguous, fp16 / bf16 / fp32.
        output: Pre-allocated float32 tensor, same numel, contiguous on CUDA.
        softcap_const: Positive scale.
    """
    in_flat = input.view(-1)
    out_flat = output.view(-1)
    module = _jit_softcap_module(in_flat.dtype)
    module.softcap_out_fp32(in_flat, out_flat, float(softcap_const))
