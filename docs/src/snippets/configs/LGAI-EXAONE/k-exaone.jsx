// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// K-EXAONE note: the verified recipes were measured on a 16x A100 40GB host. BF16 weights
// (474GB) need all 16 GPUs; the bundled MTP head is the drafter (one draft token is the
// best operating point on A100 at every concurrency measured). The triton attention
// backend is the one that is correct on A100 for this hybrid sliding-window model.

export const config = {
  modelName: "K-EXAONE",

  supportedHardware: ["a100"],
  hardware: [
    { id: "a100", label: "A100 40GB", vram: "40GB", vendor: "nvidia" },
  ],

  variants: [
    { id: "default", label: "236B-A23B" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
  ],
  strategies: [
    { id: "balanced", label: "Balanced" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16": "LGAI-EXAONE/K-EXAONE-236B-A23B",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"  },
    PORT:      { target: "command", label: "Bind port",         default: "30000"    },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"     },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --flush-cache`,
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 32`,
    },
    numPromptsByConc: { 1: 8, 16: 64, 32: 64 },
  },

  defaultAccuracy: {
    default: { gsm8k_pct: 90.0 },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  dockerImages: {
    a100: "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "LGAI-EXAONE/K-EXAONE-236B-A23B",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 8, 16] },
        { id: "dpAttn", label: "DP-Attention", values: [null, false, 2],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },
    moe: {
      backend: {
        options: [
          { id: null, label: "Inherited" },
        ],
      },
      ep: { label: "EP", values: [null, 1, 8, 16] },
    },
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser qwen3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser qwen25" },
      ],
    },
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "MTP (1 draft token)",
          flags: ["--speculative-algorithm EAGLE",
                  "--speculative-draft-model-path {{MODEL_NAME}}",
                  "--speculative-num-steps 1",
                  "--speculative-eagle-topk 1",
                  "--speculative-num-draft-tokens 2"] },
        { id: "ngram",   label: "NGRAM",
          flags: ["--speculative-algorithm NGRAM",
                  "--speculative-num-draft-tokens 16",
                  "--speculative-ngram-max-bfs-breadth 10"],
          disable: { dpAttnOn: [true] },
          disableReason: "NGRAM is incompatible with DP-Attention. Turn DP-Attention off in the Attention card above to use NGRAM." },
      ],
    },
    hicache: {
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
        { id: "hf3fs",    label: "HF3FS" },
        { id: "nixl",     label: "NiXL" },
      ],
      writePolicies: [
        { id: "auto",                    label: "Auto" },
        { id: "write_through",           label: "Write-through" },
        { id: "write_back",              label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },
    flagSelects: [
      {
        id: "swaRatio",
        title: "SWA / full KV ratio (long context)",
        stripPrefixes: ["--swa-full-tokens-ratio"],
        options: [
          { id: "default", label: "Default (0.8)" },
          { id: "0.3", label: "0.3", flags: ["--swa-full-tokens-ratio 0.3"] },
          { id: "0.1", label: "0.1 (max full-attention KV)", flags: ["--swa-full-tokens-ratio 0.1"] },
        ],
      },
    ],
  },

  cells: [
    // ==== NVIDIA A100 40GB x16 + BF16 (single node, triton attention, MTP) ====
    {
      match: { hw: "a100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--attention-backend triton",
        "--speculative-algorithm EAGLE",
        "--speculative-draft-model-path {{MODEL_NAME}}",
        "--speculative-num-steps 1",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 2",
        "--cuda-graph-max-bs-decode 32",
        "--mem-fraction-static 0.85",
        "--reasoning-parser qwen3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
