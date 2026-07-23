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

## Step 2: SWA KV allocation check — no change needed

Evidence that the hybrid manager gives the 39 SWA layers window-sized
allocations (fp8 server, pool 17,184 blocks):

- Boot: specs are SlidingWindowSpec(sliding_window=128) (runtime spec
  dump from the fp8 phase); 6 groups formed (1x9 full + 5 SWA groups,
  "Add 6 padding layers, may waste at most 15.38%").
- Runtime: during a 3.5k-token greedy decode, used blocks grew
  79 -> 190 over 832 -> 3,588 tokens = 0.040 blocks/token, matching the
  full-attention-only slope (1/32 = 0.031 + bookkeeping). If SWA layers
  were full-length the slope would be 0.344 blocks/token (8.5x).
- Absolute usage at 3.6k tokens: 190 blocks vs ~165 theoretical
  (window-bounded SWA + full growth): ~15% padding overhead, matching
  the declared padding-layer waste.

The pessimistic boot-time "Maximum concurrency" metric comes from the
per-request SWA bound min(window + max_num_batched_tokens, max_model_len)
with the default 16k prefill chunk — a REPORTING/boot-check artifact,
not an allocation problem. Actionable lever found for step 6: lowering
--max-num-batched-tokens shrinks that bound and should raise the max
bootable context materially.

Decision: keep as-is (already optimal); carry the chunk-size lever to
step 6.

## Step 3: NVFP4 MoE backend A/B — KEEP --moe-backend marlin

Same-session bench (nvfp4 KV, 16k, TP2), warmed, 500-tok gens:

| MoE backend        | single tok/s | C=4 tok/s | notes |
|--------------------|--------------|-----------|-------|
| VLLM_CUTLASS (auto)| 114.2        | 211.5     | baseline; W4A4 + act-quant overhead |
| MARLIN             | 141.5 (+24%) | 240.4 (+14%) | W4A16; winner |
| FLASHINFER_CUTLASS | 135.5 (+19%) | 236.1 (+12%) | needs nvcc on PATH (JIT) |
| FLASHINFER_B12X    | boot FAIL    | -         | torch stable-ABI dispatch error in flashinfer prebuilt module |

- Also tried FlashInfer CUTLASS MXFP8 linear kernel (nvcc on PATH) atop
  marlin MoE: boot FAIL — "'QKVParallelLinear' object has no attribute
  'workspace'" in the flashinfer MXFP8 path. Separate upstream bug;
  discarded (Marlin MXFP8 stays).
- Quality on marlin MoE: gsm8k n=20 100%, humaneval n=20 90% (within
  the 90-100% band seen across all backends at n=20, temp 0.6).
- Matches profiling: marlin removes the fp4 activation-quant +
  blockscale bookkeeping (~15% of C=1 time) and its W4A16 GEMV is
  stronger at decode batch sizes.

Decision: KEEP `--moe-backend marlin` in the serve config.

## Step 4: FP4 decode fast path — KEEP (single-pass loads; PTX cvt blocked)

- Native PTX path FAILED: `cvt.rn.f16x2.e2m1x2` assembles for sm_120a
  but the low fp16 lane always returns 0 (hi lane correct; byte staging
  verified by round-trip). Looks like a ptxas-blackwell/triton-3.6
  bundle bug; documented and abandoned.
- Fallback implemented: single-pass tile loads (each packed byte loaded
  once, both nibbles decoded, tl.interleave into head-dim order; block
  scales loaded once per 16 dims and broadcast). K built token-major
  then transposed.
- Found + fixed en route: tl.interleave miscompiles at 16-row tiles
  (3D decode path produced garbage; caught by extending unit tests to
  the 3D and prefill shapes after e2e gibberish). NVFP4 now pins
  32-token tiles in the 3D path.
- Kernel microbench (8x4k decode): 546us -> 322us (-41%).
- E2E: single 143.0 (vs 141.5, +1%), C=4 333-347 (vs 240.4, +39%).
- Quality: gsm8k n=20 95%; sanity outputs coherent (391/prose/code).

Decision: KEEP.

## Step 5: Kernel tuning sweep — KEEP split policy

Sweep (kernel-level, decode/prefill shapes; stages had no effect):

- 2D grid: tile64/warps8 wins: C1-shape 639->488us, C4-shape
  322->246us (-24%) vs tile32/warps4.
- 3D (split-KV decode) grid: tile32/warps4 best (63.9us C1); tile64 or
  warps8 regress up to +28% there.

Applied: NVFP4 -> 2D tile64/warps8, 3D tile32/warps4. All unit tests
pass (cos unchanged).

E2E: C=4 330-347 tok/s (unchanged vs step 4 — decode config identical);
single-stream measured 136-143 tok/s with ~5% boot-to-boot variance
(within-boot spread <1%); prefill benefits deferred to step 6 numbers.

Decision: KEEP (2D win is real at kernel level; 3D unchanged).

## Step 6: Context push — native 1M context boots

Lever from step 2 (--max-num-batched-tokens 8192 shrinks the
per-request SWA boot estimate) + nvfp4 KV + marlin MoE:

| max-model-len | pool tokens | concurrency | note |
|---------------|-------------|-------------|------|
| 262,144       | 806,470     | 3.08x       | needle EXACT @ 207,059 tok (160s, ~1.3k tok/s prefill) |
| 524,288       | 1,003,541   | 1.91x       | needle EXACT @ 405,059 tok (577s, ~700 tok/s) |
| 786,432       | 1,121,011   | 1.43x       | boot probe |
| 1,048,576     | 1,174,682   | 1.12x       | NATIVE MAX — boots, sanity coherent |

(bf16 KV baseline could not even boot 32k at 0.95 utilization.)
Follow-up: 1M-range needle (~25 min prefill) not run; 405k retrieval
was exact.
