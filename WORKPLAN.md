# MiMo-V2.5 (NVFP4) support in vLLM — work plan

Target: serve `lukealonso/MiMo-V2.5-NVFP4` (architecture `MiMoV2ForCausalLM`,
`model_type: mimo_v2`) on 2x RTX PRO 6000 Blackwell (SM120, CUDA 13, no
NVLink). Text generation first; multimodal, MTP spec-decode, and FP4 KV cache
as later phases.

## Checkpoint facts (verified against the snapshot)

- 48 decoder layers; `layer_types`: 9 `full_attention`
  (indices 0, 5, 11, 17, 23, 29, 35, 41, 47) + 39 `sliding_attention`
  (window 128, from `hybrid_layer_pattern` / `sliding_window_size`).
- Full-attention layers: 64 Q heads, 4 KV heads, rope_theta 1e7.
  SWA layers: 64 Q heads, 8 KV heads, rope_theta 1e4.
  Both: qk head_dim 192, v_head_dim 128 (asymmetric),
  partial_rotary_factor 0.334 (rotary_dim 64), `attention_value_scale`
  0.707 applied to V, attention sink bias on SWA layers only (`[64]` per
  layer, TP-narrowed).
- MoE: layer 0 dense (intermediate 16384), layers 1-47 routed MoE with 256
  experts, top-8, moe_intermediate 2048, sigmoid scoring, `noaux_tc` with
  `gate.e_score_correction_bias`, n_group=topk_group=1, norm_topk_prob.
  No shared experts (`n_shared_experts: null`).
- max_position_embeddings 1,048,576; vocab 152,576; ~310B params, 183 GB
  on disk.
- Quantization (ModelOpt `MIXED_PRECISION`, `quant_method: modelopt`):
  - `qkv_proj` (48 layers + 3 MTP layers) and dense MLP projections
    (layer 0 + MTP): **MXFP8** — fp8-e4m3 weight in plain `[Q|K|V]` row
    order + uint8 E8M0 scales (`weight_scale_inv`, group 32 along input).
  - Routed experts (47 x 256 x 3 projections): **NVFP4** — packed uint8
    weight, fp8 per-16-group `weight_scale`, f32 `weight_scale_2` /
    `input_scale`.
  - BF16: `o_proj`, router gates, embeddings, `lm_head`, norms, sinks.
  - No KV-cache quant scales in the checkpoint.
- Extra top-level weights in the index: `model.mtp.layers.{0,1,2}.*`
  (3 MTP draft layers), `visual.*`, `audio_encoder.*`,
  `speech_embeddings.*`. `amax_checkpoint.safetensors` and
  `model-inputscales.safetensors` are auxiliary (not in the index) and are
  ignored.
- The config carries non-empty `vision_config`/`audio_config`; vLLM's
  `MimoV2ModelArchConfigConvertor` then forces
  `architectures=["MiMoV2OmniForCausalLM"]`. Text-only serving uses
  `--hf-overrides '{"vision_config": null, "audio_config": null}'`.

## What upstream vLLM already provides (verified on master b44311b6e)

- Model: `vllm/model_executor/models/mimo_v2.py` (text), `mimo_v2_omni.py`
  (vision/audio), `mimo_v2_mtp.py` (MTP draft), registry entries for all
  three, arch-config convertors for `mimo_v2` / `mimo_v2_mtp`.
- Attention: `Attention(head_size_v=...)` supports qk!=v head dims;
  dedicated `FLASH_ATTN_DIFFKV` / `TRITON_ATTN_DIFFKV` backends pack
  K(192)+V(128) in the KV-cache content dim; sinks and per-layer sliding
  window are supported; hybrid KV-cache manager handles the mixed
  full/SWA groups with differing KV-head counts (page bytes are unified
  by block-size scaling: SWA per-token bytes are exactly 2x full-attn).
- Quantization: `ModelOptMixedPrecisionConfig` (`modelopt_mixed`) routes
  per-module to MXFP8 / NVFP4 / FP8 / unquantized from the
  `quantized_layers` map; `ModelOptMxFp8LinearMethod` (kernels:
  FlashInfer CuteDSL/Cutlass, Marlin, emulation) and
  `ModelOptNvFp4FusedMoE` (SM120-capable backends: FLASHINFER_CUTLASS,
  VLLM_CUTLASS, MARLIN; `FLASHINFER_B12X` is a dedicated SM12x kernel
  behind explicit `moe_backend="flashinfer_b12x"`).

## Done in this branch

1. `[Model] Register MiMoV2Config for mimo_v2 checkpoints`
   (`vllm/transformers_utils/configs/mimo_v2.py` + registry entries).
   Upstream transformers does not know `model_type: mimo_v2` (checkpoints
   come from a transformers fork and ship no remote code), so config
   loading previously failed outright.
2. `[Model] Load ModelOpt MXFP8 fused qkv_proj in MiMo-V2`
   (`vllm/model_executor/models/mimo_v2.py`):
   - The fp8 qkv_proj interception now only applies to Pro-format
     block-FP8 layers (identified by their `weight_scale_inv` parameter);
     MXFP8 fused qkv_proj loads via the standard fused-QKV loader, which
     already handles the `[Q|K|V]` split, `v_head_size`, and TP.
   - uint8 (E8M0) `weight_scale_inv` checkpoint tensors are remapped to
     the `weight_scale` parameter registered by the MXFP8 linear method.
   - Text-only load skips `visual.` / `audio_encoder.` /
     `speech_embeddings.` weights.
3. `[Misc] Add offline validation tool`
   (`tools/mimo_v25_offline_validation.py`).

### Offline validation results (CPU-only, meta device, emulated SM120)

TP=1 and TP=2 (torchrun, gloo) both PASS:

- 145,276 checkpoint tensors fed through the real `load_weights`;
  510 intentionally skipped (MTP + multimodal towers); all 756 model
  parameters loaded on every rank; shape/dtype checked by the real
  weight loaders (meta-device copies still validate shapes).
- Quantization resolution per module: 50 `ModelOptMxFp8LinearMethod`
  (48 qkv_proj + layer-0 gate_up_proj/down_proj), 47
  `ModelOptNvFp4FusedMoE`, 49 `UnquantizedLinearMethod` (48 o_proj +
  lm_head), 1 `UnquantizedEmbeddingMethod`.
- Config pipeline resolves architecture `MiMoV2ForCausalLM` and
  quantization `modelopt_mixed` without trust_remote_code.

## Remaining work

### Phase 1: GPU bring-up — DONE (2026-07-23)

Bring-up succeeded on the 2x RTX PRO 6000 (TP=2). Verified serve command:

```bash
source ~/vllm-mimo/.venv/bin/activate
HF_HOME=/data/models/hf-cache vllm serve lukealonso/MiMo-V2.5-NVFP4 \
    --tensor-parallel-size 2 \
    --hf-overrides '{"vision_config": null, "audio_config": null}' \
    --max-model-len 32768
```

Additional required flags found during bring-up: `--max-model-len 16384
--max-num-seqs 32 --gpu-memory-utilization 0.95 --reasoning-parser mimo
--tool-call-parser mimo --enable-auto-tool-choice`, and launch with
`VLLM_API_KEY` unset if the shell profile exports one.

Results:

- Backends: `TRITON_ATTN_DIFFKV` attention, `MarlinMxfp8LinearKernel`
  for MXFP8 GEMM (qkv_proj + dense MLP), `VLLM_CUTLASS` NVFP4 MoE.
- Memory: weights ~89 GB/GPU; KV cache 2.95 GiB (24,981 tokens) at
  0.95 utilization and 16k max len. 0.92 utilization is 0.18 GiB short.
- Coherence: greedy sanity prompts correct (arithmetic, prose, code).
- Throughput (steady state): single-stream 116 tok/s; C=4 aggregate
  ~265-293 tok/s (engine-log cross-checked). Above the sglang stack on
  the same hardware.
- Quality spots (temperature 0.6, max-tokens 1024, harness defaults):
  gsm8k n=20 = 95% (sglang reference 95%), humaneval n=20 = 100%
  (sglang reference 95%).
- Fixed during bring-up: FlashInfer CUTLASS MXFP8 kernel was selected
  when flashinfer could not actually run (no cubins, no nvcc on PATH),
  crashing in the first compiled forward; its `is_supported` now
  requires `has_flashinfer()`.
- Still open from this phase:
  - Validate long-context needle behavior across the SWA/full-attention
    mix (needs a longer max-model-len profile).
  - Try the FlashInfer MXFP8 CUTLASS kernel with `/usr/local/cuda/bin`
    on PATH (enables JIT) and `flashinfer_b12x` MoE backend; A/B against
    Marlin/VLLM_CUTLASS.
  - Larger eval sweeps (n=40+: mmlu, math500, gpqa) vs the sglang
    references.

### Phase 2: FP8 KV cache — DONE (2026-07-23)

`--kv-cache-dtype fp8_e4m3` now works with `TRITON_ATTN_DIFFKV`.

- Root-caused and fixed a silent pre-existing bug first:
  `MiMoV2FlashDecoderLayer` never passed `cache_config` into its
  attention layers, so any `--kv-cache-dtype` was ignored (specs stayed
  bf16/KVQuantMode.NONE while the engine banner claimed fp8).
- DiffKV read side extended for FP8 per-tensor KV: `KV_QUANT_MODE`
  constexpr + per-tensor descales in `kernel_unified_attention_diffkv`
  (shared `_cast_kv_tile` dequant); impl views the packed cache as fp8
  and passes `layer._k_scale`/`_v_scale`; query quantization disabled
  (kernel dequantizes K/V against the bf16 query). The store kernel
  (`reshape_and_cache_flash_diffkv`) already handled fp8.
- Validation on the 2x PRO 6000 (TP2):
  - GPU unit test: fp8 vs bf16 kernel outputs cosine 0.9994 (2D + 3D
    paths, sliding window, non-unit descale).
  - KV capacity doubled: 24,981 -> 49,965 tokens at 16k max len
    (concurrency 1.52x -> 3.05x); 32k max len boots with 91,018 tokens
    (2.78x) — bf16 could not hold 32k at 0.95 utilization.
  - Coherence (greedy 391/prose/code) and 13.5k-token needle recall
    pass; gsm8k n=20 95%, humaneval n=20 95% (bf16: 95%/100%; one
    sample at n=20 is within temperature-0.6 noise).
  - Throughput parity with bf16: 116 tok/s single-stream, ~261 tok/s
    at C=4.
- Still open: FA4-on-SM12x investigation for `FLASH_ATTN_DIFFKV`
  remains a possible perf upgrade over the triton path.

### Phase 2b: NVFP4 KV cache — DONE (2026-07-23)

`--kv-cache-dtype nvfp4` works with `TRITON_ATTN_DIFFKV`. Layout per
(token, head, side): E2M1 data (2 dims/byte) + inline fp8-e4m3 block
scales (1 per 16 dims), matching the FlashInfer NVFP4 KV conventions
(`nvfp4_kv_cache_full_dim`); content dim 108+72 = 180 bytes vs 320
(fp8) / 640 (bf16).

Results (2x PRO 6000, TP2, runtime spec dump confirmed
`uint8 / KVQuantMode.NVFP4`):

- GPU unit test: store+read round trip vs bf16, cosine 0.991-0.992
  (windowed + non-unit tensor scales; ~4x the fp8 error, as expected
  for 4-bit).
- Capacity: 89,180 tokens at 16k max len (3.57x bf16, 1.78x fp8 —
  exactly the 640:320:180 byte ratio). New context ceiling: 64k boots
  at 4.25x concurrency (278,780 tokens), 128k boots at 3.31x
  (433,976 tokens). bf16 could not hold 32k.
- Coherence pass (391 / prose / code, also at the 128k config); needle
  recall exact at 13.5k and 55.5k tokens.
- Throughput: 113 tok/s single-stream (-2.5% vs bf16/fp8), 231 tok/s
  at C=4 (-12% vs fp8) — in-register nibble decode cost.
- Quality: gsm8k n=20 95%, humaneval n=20 100%, math500 n=20 50% at
  max-tokens 1024 but 70% at 2048 (above the sglang n=40 reference of
  65%) — the low first reading was token-budget truncation, not fp4
  degradation.

### Phase 2b original assessment (for reference)

Groundwork already in vLLM: `KVQuantMode.NVFP4` exists,
`AttentionSpec.real_page_size_bytes` / the DiffKV spec compute packed
fp4 page sizes via `nvfp4_kv_cache_full_dim` (fp4 data + inline fp8
block scales), and `CacheDType` accepts "nvfp4" (used by the
non-DiffKV nvfp4 path). Remaining work for DiffKV:

1. Store: add an NVFP4 branch to `reshape_and_cache_flash_diffkv`
   (per-16-element group amax -> block scale, pack two E2M1 nibbles per
   byte, write scales inline after the packed data).
2. Load: add a `KV_QUANT_MODE == NVFP4` branch to
   `kernel_unified_attention_diffkv`: load packed bytes + block scales,
   branchless E2M1 nibble decode, dequant per 16-element group. The
   proven portable math (nibble LUT `_dequant_fp4_nibble`, lo/hi nibble
   indexing `byte d//2`, uint8 exponent scales per 16 dims) exists in
   the working sglang-side kernel
   (~/mimo-v25-sglang/patches/extend_attention.py) and in the earlier
   DeepSeek sparse-MLA NVFP4-KV port (~/vllm-nvfp4-port).
3. Layout: `TritonAttentionDiffKVBackend.get_kv_cache_shape` must
   switch the content dim to
   `nvfp4_kv_cache_full_dim(head_size) + nvfp4_kv_cache_full_dim(head_size_v)`
   and the impl's K/V view slicing must follow the packed offsets;
   match the existing non-DiffKV nvfp4 backend's inline-scale layout.
4. Gating: advertise "nvfp4" in `supported_kv_cache_dtypes` and reject
   it for backends without the branch.

Effort estimate: comparable to the fp8 step plus the two kernel
branches; quality risk is higher (4-bit V), so gate on needle tests and
a full eval sweep vs fp8.

### Phase 3: Multimodal (omni)

- Drop the `--hf-overrides` nulling and exercise
  `MiMoV2OmniForCausalLM` (vision tower + audio encoder are BF16 and
  already excluded from quantization). Re-run the offline validation
  against the omni architecture (extend the tool to omit the MM skip
  prefixes in that mode).

### Phase 4: MTP speculative decoding

- `MiMoV2MTPModel` is registered; the checkpoint has 3 MTP layers
  (`model.mtp.layers.{0,1,2}`, MXFP8 qkv/MLP). Wire up
  `--speculative-config` for `mimo_v2_mtp`, confirm the draft loader
  consumes the `model.mtp.*` weights the target model skips.

## Upstream PR series (proposed order)

1. `[Model] Register MiMoV2Config for mimo_v2 checkpoints` — standalone,
   unblocks all mimo_v2 checkpoints without remote code.
2. `[Bugfix] Pass cache_config to MiMo-V2 attention layers` — standalone
   correctness fix (kv-cache dtype silently ignored); small and easy to
   review; could be squashed with (1) into one MiMo-V2 fixes PR.
3. `[Model] Load ModelOpt MXFP8 fused qkv_proj in MiMo-V2` +
   `[Kernel] Gate FlashInfer CUTLASS MXFP8 kernel on FlashInfer
   availability` — one PR: makes the NVFP4/MXFP8 checkpoint load and run
   (the kernel gating fix is what its bring-up exposed).
4. `[Attention] Support FP8 per-tensor KV cache in Triton DiffKV
   backend` — one PR with unit tests ported from the bring-up scripts
   (tests/kernels/attention).
5. `[Attention] Support NVFP4 KV cache in Triton DiffKV backend` —
   follow-up PR on (4), same test harness extended.
- Not for upstream as-is: the offline validation tool (reshape into a
  weight-loading test) and the WORKPLAN itself.

## Upstream PR notes

- Candidate upstream changes: the two `[Model]` commits (config
  registration + MXFP8 loader). The validation tool can be dropped from
  the PR or reshaped into a test; prefer extending
  `tests/models/registry.py` usage plus a weight-loading unit test.
- `tests/models/registry.py` currently points `MiMoV2ForCausalLM` at
  `XiaomiMiMo/MiMo-V2.5-Pro` with `trust_remote_code=True`; with
  `MiMoV2Config` registered, trust_remote_code should no longer be
  needed — verify against the Pro checkpoint before changing it.
- Follow repo policy in AGENTS.md: check for duplicate PRs first
  (`gh pr list --repo vllm-project/vllm --search "mimo"`), run
  `pre-commit run --all-files` and targeted tests, and include eval
  results for model-affecting changes.
