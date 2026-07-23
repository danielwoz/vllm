# MiMo-V2.5 nvfp4 serve — optimization campaign log

Standard bench: warmed single-stream (2x500 tok) + C=4 (4x500 tok) wall
tok/s, cross-checked against engine-logged generation throughput.
Baseline config: TP2, nvfp4 KV, max-model-len 16384, max-num-seqs 32,
gpu-memory-utilization 0.95, Marlin MXFP8 linear, VLLM_CUTLASS NVFP4 MoE,
TRITON_ATTN_DIFFKV.

Entry baselines (from bring-up): bf16 KV 116 / 267 tok/s; fp8 KV 116 /
261 tok/s; nvfp4 KV 113 / 231 tok/s.

## Step 1: Profiling (no changes) — 2026-07-23

Baseline this session (nvfp4, 16k): 114.2 tok/s single, 211.5 tok/s C=4
(note: C=4 shows +/-10% run variance vs yesterday's 231).

torch.profiler kernel breakdown, self-CUDA % of decode window:

| Kernel group                                   | C=1    | C=4    |
|------------------------------------------------|--------|--------|
| NVFP4 MoE grouped GEMM (cutlass GemmUniversal)  | 31.0%  | 44.6%  |
| MoE overhead (act fp4 quant, blockscale offsets,
|   topk, shuffles, problem sizes)                | ~15%   | ~9%    |
| bf16 dense GEMV/GEMM (o_proj, lm_head, gates)   | ~25%   | ~20%   |
| Marlin MXFP8 (qkv_proj + dense MLP)             | 13.3%  |  8.8%  |
| TP2 allreduce (custom 1-stage)                  |  6.9%  |  6.3%  |
| DiffKV attention (unified + reduce + cache)     |  5.1%  | ~3.5%  |

- GPU util at C=1: 100% SM-active, ~59% memory util on both GPUs
  (decode is GPU-bound; host overhead negligible).
- fp8-KV comparison at C=1: total window 2.586s vs nvfp4 2.621s over
  300 decode steps -> nvfp4 attention overhead ~1.3% end-to-end.

Verdict: the dominant cost at both concurrencies is the MoE path
(GEMM + overheads ~46% at C=1, ~54% at C=4). Attention is a minor
cost; steps 4/5 (fp4 decode fast path, kernel tuning) are bounded at
~1-3% end-to-end. Step 3 (MoE backend A/B) carries the leverage.
