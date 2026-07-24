# MiMo-V2.5 on vLLM — upstream PR breakdown

Branch `mimo-v25-support`, based on vLLM master `b44311b6e` (2026-07-22).
All changes are Python-only (model, config, quantization, Triton kernels).

vLLM master already carries most of the MiMo-V2 machinery (model classes,
DiffKV attention backend for the asymmetric qk=192 / v=128 head dims, ModelOpt
`MIXED_PRECISION` routing). This branch makes the `lukealonso/MiMo-V2.5-NVFP4`
checkpoint actually load and serve on consumer Blackwell (SM120), then adds
FP8/NVFP4 KV cache and decode-throughput work.

Suggested PRs, in dependency order. Each is independently reviewable; later
ones build on earlier ones.

---

## PR 1 — Register `MiMoV2Config` for `mimo_v2` checkpoints
Commit: `79284f7d7`
Files: `vllm/transformers_utils/config.py`, `configs/__init__.py`, `configs/mimo_v2.py`

`model_type: mimo_v2` ships in Xiaomi's transformers fork; upstream transformers
does not know it and the checkpoint bundles no remote code, so config loading
fails outright. Adds a `MiMoV2Config` (ported field-for-field from the reference)
and registers it. Small, self-contained, unblocks everything else.
Optionally squash the tiny `cache_config` bugfix (PR 3) into this one.

## PR 2 — Load ModelOpt MXFP8 fused `qkv_proj` in MiMo-V2 (+ FlashInfer gating)
Commits: `f991ba077`, `8660cf779`
Files: `vllm/model_executor/models/mimo_v2.py`,
`vllm/model_executor/kernels/linear/mxfp8/flashinfer.py`

The checkpoint stores the fused qkv in plain `[Q|K|V]` row order with uint8
E8M0 group-32 scales named `weight_scale_inv`, but the MXFP8 method registers
the parameter as `weight_scale` and the existing path assumed the Pro-format
interleaved layout. Fixes the loader to detect the format, and gates the
FlashInfer CUTLASS MXFP8 kernel on `has_flashinfer()` (it was auto-selected
without cubins/nvcc and crashed the first forward). Both are genuine bring-up
bugs for this checkpoint class.

## PR 3 — Pass `cache_config` to MiMo-V2 attention layers
Commits: `dab6adab7` (target), `fcf79c2e8` (MTP draft)
Files: `mimo_v2.py`, `mimo_v2_mtp.py`

The decoder layers never forwarded `cache_config` into their attention modules,
so `--kv-cache-dtype` was silently ignored — every layer ran the default dtype
while the engine banner reported the requested one. Correctness fix on its own;
also a prerequisite for the KV-cache PRs. (The MTP-side copy of this bug also
zeroed speculative acceptance — see PR 6.)

## PR 4 — FP8 per-tensor KV cache in the Triton DiffKV backend
Commit: `73204a622`
Files: `vllm/v1/attention/backends/triton_attn_diffkv.py`,
`ops/triton_reshape_and_cache_flash.py`, `ops/triton_unified_attention_diffkv.py`

Adds an FP8 read path (`KV_QUANT_MODE` + per-tensor descales) to the DiffKV
unified kernel, reusing the parent backend's cast helper; store side already
quantized. FA3/FA4 are unavailable on SM120, so DiffKV is the only KV-quant
route for these head shapes there. 2x KV capacity, throughput parity, quality
unchanged. Include the kernel unit test.

## PR 5 — NVFP4 KV cache in the Triton DiffKV backend
Commits: `66038c8a3`, `92e942660`, `81c097b7c`
Files: same three attention files

NVFP4 store + read for DiffKV: packed E2M1 data with inline fp8-e4m3 per-16
block scales, in-register dequant fused into the attention kernel (no
whole-buffer dequant). Follows the FlashInfer NVFP4-KV layout conventions.
~3.5x KV capacity vs bf16, quality within noise, needle exact at 405k tokens.
Depends on PR 4's kernel scaffolding. Include unit tests (2D + 3D + prefill
shapes — the 3D path has a distinct tile constraint).

## PR 6 — Optional online FP8 for unquantized modules in ModelOpt mixed checkpoints
Commit: `3667e1269`
Files: `vllm/model_executor/layers/quantization/modelopt.py`

ModelOpt mixed checkpoints leave attention `o_proj` and `lm_head` in bf16.
Adds an opt-in to quantize named module suffixes to online per-tensor FP8 at
load. +12% single-stream, quality at ceiling. Promote the env var
(`VLLM_MODELOPT_MIXED_ONLINE_FP8`) to a proper config field before submission;
scope the suffix match so it does not catch multimodal towers.

## PR 7 — Widen DiffKV split-KV decode parallelism
Commit: `2acff4c2b`
Files: `triton_attn_diffkv.py`, `triton_unified_attention_diffkv.py`

DiffKV models keep very few KV heads per rank (MiMo: 2 at TP2), so the 3D
split-KV grid at the default 16 segments launches ~32 blocks on ~188 SMs and
starves the GPU at long context. Raises the DiffKV segment count to 128
(scoped to the subclass — does not touch the global default) and admits short
speculative-verify queries (q_len<=2) into the 3D path. 6.7x kernel speedup at
1M depth; end-to-end decode at 200k context 13 -> 112 tok/s. Consider proposing
the adaptive form (`segments ~= sm_count / (kv_heads * batch)`) in review.

---

## Not for PR
- `mimo_v2_mtp.py` MTP loader fix (`a541f6223`) ships with PR 2/3 as the draft
  counterpart of the same bugs.
- `tools/mimo_v25_offline_validation.py` — reshape into a weight-loading test
  under `tests/` rather than shipping as a script.
- `OPTLOG.md` / `WORKPLAN.md` / this file — local notes, not for upstream.

## Per AGENTS.md
vLLM requires AI-assistance disclosure and a human submitter who reviews every
line and runs the tests; PR descriptions must state tests run and eval results.
Model-affecting PRs (4–7) need an eval table — the n=40 sweep in `OPTLOG.md`
(gsm8k 100 / humaneval 92.5 / math500 70 / mmlu-cs 75) is the source.
