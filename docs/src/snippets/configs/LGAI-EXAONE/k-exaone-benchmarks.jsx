// One entry per cell `match` tuple. Speed: P50 TTFT/TPOT; tokens_per_sec_per_gpu is total
// (in+out) tok/s/GPU = output tok/s / GPUs * (isl+osl)/osl.
// Measured on a 16x A100 40GB host with the K-EXAONE serving PR stack
// (github.com/Godmook/sglang kexaone/2-mtp-head @ 03acefe565 on main 76ccf0850c).

export const benchmarks = [
  {
    match: { hw: "a100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "03acefe565",
    speed: [
      { workload: { dataset: "random-ids", isl: 1024, osl: 512, max_concurrency: 16 },
        ttft_ms: 146.5, tpot_ms: 16.4, tokens_per_sec_per_gpu: 152.6 },
    ],
    accuracy: { gsm8k_pct: 90.0 },
  },
];
