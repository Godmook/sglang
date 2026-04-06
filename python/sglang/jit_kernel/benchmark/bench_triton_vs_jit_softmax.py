"""Benchmark: Triton fused_temperature_softmax vs JIT softmax_sampling vs baselines.

Compares the two fused softmax kernels head-to-head on the same GPU, same inputs.
Uses torch.cuda.Event timing for accurate measurement.

Usage:
  python -m sglang.jit_kernel.benchmark.bench_triton_vs_jit_softmax [--iters 200] [--warmup 50] [--quick]

This repository layout:
  - Triton: ``sglang.jit_kernel.triton.fused_temperature_softmax.fused_temperature_softmax``
  - JIT: ``sglang.jit_kernel.softmax.softmax_sampling``

Alternative setups (other branches / copies):
  Option A) JIT branch + copy Godmook's ``fused_sampling.py`` beside this script
    - Imports Triton from local ``fused_sampling`` if package path missing

  Option B) fused_sampling branch + full JIT CUDA build (heavier)

  Option C) ``pip install -e .`` from a tree that contains both kernels under
    ``jit_kernel/triton/`` and ``jit_kernel/softmax.py`` (this repo).
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

import torch

# ============================================================================
# Kernel imports - adjust these based on your setup
# ============================================================================

HAVE_TRITON_KERNEL = False
HAVE_JIT_KERNEL = False
HAVE_FLASHINFER = False

# --- Triton kernel (in-repo: jit_kernel/triton; or srt.layers; or local file) ---
try:
    from sglang.jit_kernel.triton.fused_temperature_softmax import (
        fused_temperature_softmax,
    )

    HAVE_TRITON_KERNEL = True
    print("[OK] Triton kernel: from sglang.jit_kernel.triton.fused_temperature_softmax")
except ImportError:
    try:
        from sglang.srt.layers.fused_sampling import (  # type: ignore
            fused_temperature_softmax,
        )

        HAVE_TRITON_KERNEL = True
        print("[OK] Triton kernel: from sglang.srt.layers.fused_sampling")
    except ImportError:
        try:
            from fused_sampling import fused_temperature_softmax  # type: ignore

            HAVE_TRITON_KERNEL = True
            print("[OK] Triton kernel: from local fused_sampling.py")
        except ImportError:
            print("[--] Triton kernel: NOT FOUND (skipping)")

# --- JIT kernel ---
try:
    from sglang.jit_kernel.softmax import softmax_sampling

    HAVE_JIT_KERNEL = True
    print("[OK] JIT kernel: from sglang.jit_kernel.softmax")
except ImportError:
    softmax_sampling = None  # type: ignore
    print("[--] JIT kernel: NOT FOUND (skipping)")

# --- FlashInfer baseline ---
# Import can raise RuntimeError (e.g. flashinfer vs flashinfer-cubin version mismatch).
try:
    from flashinfer.sampling import softmax as flashinfer_softmax

    HAVE_FLASHINFER = True
    print("[OK] FlashInfer: available")
except ImportError:
    flashinfer_softmax = None  # type: ignore
    print("[--] FlashInfer: NOT FOUND (skipping)")
except RuntimeError as e:
    flashinfer_softmax = None  # type: ignore
    print("[--] FlashInfer: skipped —", e)
    print(
        "    Fix: pip install matching `flashinfer` and `flashinfer-cubin`, or set "
        "FLASHINFER_DISABLE_VERSION_CHECK=1 (not recommended for production)."
    )

print()

if not HAVE_TRITON_KERNEL and not HAVE_JIT_KERNEL:
    print("ERROR: At least one of Triton or JIT kernel must be available.")
    print("See the docstring at the top of this file for setup instructions.")
    sys.exit(1)


# ============================================================================
# Timing utility
# ============================================================================


def benchmark_fn(fn: Callable[[], None], warmup: int = 50, iters: int = 200) -> float:
    """Time a zero-arg callable using CUDA events. Returns microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # ms -> us


# ============================================================================
# Benchmark configs
# ============================================================================

CONFIGS = [
    # (batch_size, vocab_size, dtype)
    # Small vocab (single-pass for Triton)
    (1, 32000, torch.bfloat16),
    (8, 32000, torch.bfloat16),
    (32, 32000, torch.bfloat16),
    (64, 32000, torch.bfloat16),
    (128, 32000, torch.bfloat16),
    (256, 32000, torch.bfloat16),
    (512, 32000, torch.bfloat16),
    # Large vocab (multi-pass for Triton, split for JIT)
    (1, 128256, torch.bfloat16),
    (8, 128256, torch.bfloat16),
    (32, 128256, torch.bfloat16),
    (64, 128256, torch.bfloat16),
    (128, 128256, torch.bfloat16),
    (256, 128256, torch.bfloat16),
    (512, 128256, torch.bfloat16),
    # Very large vocab
    (1, 151936, torch.bfloat16),
    (32, 151936, torch.bfloat16),
    (128, 151936, torch.bfloat16),
    (512, 151936, torch.bfloat16),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Triton vs JIT softmax")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only a subset of configs for quick testing",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required.")
        sys.exit(1)

    configs = CONFIGS
    if args.quick:
        configs = [
            (1, 32000, torch.bfloat16),
            (32, 32000, torch.bfloat16),
            (128, 32000, torch.bfloat16),
            (1, 128256, torch.bfloat16),
            (32, 128256, torch.bfloat16),
            (128, 128256, torch.bfloat16),
        ]

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print()

    cols = ["bs", "vocab", "dtype"]
    col_widths = [5, 7, 8]

    cols.append("torch (us)")
    col_widths.append(12)

    if HAVE_TRITON_KERNEL:
        cols.append("triton (us)")
        col_widths.append(12)

    if HAVE_JIT_KERNEL:
        cols.append("jit (us)")
        col_widths.append(12)

    if HAVE_FLASHINFER:
        cols.append("fi (us)")
        col_widths.append(12)

    if HAVE_TRITON_KERNEL:
        cols.append("tri/torch")
        col_widths.append(10)

    if HAVE_JIT_KERNEL:
        cols.append("jit/torch")
        col_widths.append(10)

    if HAVE_TRITON_KERNEL and HAVE_JIT_KERNEL:
        cols.append("jit/tri")
        col_widths.append(10)

    if HAVE_FLASHINFER and HAVE_TRITON_KERNEL:
        cols.append("tri/fi")
        col_widths.append(10)

    if HAVE_FLASHINFER and HAVE_JIT_KERNEL:
        cols.append("jit/fi")
        col_widths.append(10)

    header = "  ".join(f"{c:>{w}}" for c, w in zip(cols, col_widths))
    print(header)
    print("-" * len(header))

    for bs, vocab, dtype in configs:
        logits_src = torch.randn(bs, vocab, dtype=dtype, device="cuda")
        temps_2d = torch.rand(bs, 1, dtype=torch.float32, device="cuda") * 1.5 + 0.1
        temps_1d = temps_2d.view(-1)

        results: dict[str, float] = {}

        def run_torch(src=logits_src, t=temps_2d):
            l = src.clone()
            l.div_(t)
            torch.softmax(l, dim=-1)

        results["torch"] = benchmark_fn(run_torch, args.warmup, args.iters)

        if HAVE_TRITON_KERNEL:
            def run_triton(src=logits_src, t=temps_2d):
                fused_temperature_softmax(src.clone(), t)

            results["triton"] = benchmark_fn(run_triton, args.warmup, args.iters)

        if HAVE_JIT_KERNEL:
            assert softmax_sampling is not None

            def run_jit(src=logits_src, t=temps_1d):
                softmax_sampling(src.clone(), t)

            results["jit"] = benchmark_fn(run_jit, args.warmup, args.iters)

        if HAVE_FLASHINFER:
            assert flashinfer_softmax is not None

            def run_fi(src=logits_src, t=temps_1d):
                flashinfer_softmax(src.clone(), temperature=t)

            results["fi"] = benchmark_fn(run_fi, args.warmup, args.iters)

        row_vals = [
            f"{bs:>5}",
            f"{vocab:>7}",
            f"{str(dtype):>8}",
            f"{results['torch']:>12.1f}",
        ]

        if HAVE_TRITON_KERNEL:
            row_vals.append(f"{results['triton']:>12.1f}")
        if HAVE_JIT_KERNEL:
            row_vals.append(f"{results['jit']:>12.1f}")
        if HAVE_FLASHINFER:
            row_vals.append(f"{results['fi']:>12.1f}")

        t_torch = results["torch"]

        if HAVE_TRITON_KERNEL:
            sp = t_torch / results["triton"]
            row_vals.append(f"{sp:>9.2f}x")

        if HAVE_JIT_KERNEL:
            sp = t_torch / results["jit"]
            row_vals.append(f"{sp:>9.2f}x")

        if HAVE_TRITON_KERNEL and HAVE_JIT_KERNEL:
            ratio = results["triton"] / results["jit"]
            row_vals.append(f"{ratio:>9.2f}x")

        if HAVE_FLASHINFER and HAVE_TRITON_KERNEL:
            ratio = results["fi"] / results["triton"]
            row_vals.append(f"{ratio:>9.2f}x")

        if HAVE_FLASHINFER and HAVE_JIT_KERNEL:
            ratio = results["fi"] / results["jit"]
            row_vals.append(f"{ratio:>9.2f}x")

        print("  ".join(row_vals))

    print()
    print("Speedup ratios:")
    print("  tri/torch, jit/torch  : >1x means faster than PyTorch baseline")
    print("  jit/tri               : >1x means JIT is faster than Triton")
    print("  tri/fi, jit/fi        : >1x means the kernel is faster than FlashInfer")


if __name__ == "__main__":
    main()
