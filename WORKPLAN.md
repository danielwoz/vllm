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

### Phase 1: GPU bring-up (blocked until the benchmark container is done)

```bash
source ~/vllm-mimo/.venv/bin/activate
HF_HOME=/data/models/hf-cache vllm serve lukealonso/MiMo-V2.5-NVFP4 \
    --tensor-parallel-size 2 \
    --hf-overrides '{"vision_config": null, "audio_config": null}' \
    --max-model-len 32768
```

- Confirm attention backend: expect `TRITON_ATTN_DIFFKV` on SM120
  (FLASH_ATTN_DIFFKV needs FA3/FA4; `get_flash_attn_version` returns FA2
  for capability major 12).
- Confirm NVFP4 MoE backend selection (flashinfer not installed:
  VLLM_CUTLASS, else MARLIN; try `flashinfer_b12x` explicitly later) and
  MXFP8 linear kernel (likely Marlin without flashinfer; the emulation
  kernel dequantizes to BF16 at load, +~3 GB weights — avoid).
- Memory: 183 GB checkpoint on 2x96 GB is tight; start with modest
  `--max-model-len` / `--gpu-memory-utilization`, then grow.
- Quality: greedy sanity prompts, then compare against the sglang
  reference serve on the same checkpoint (quality benchmark already run
  on this host).
- Validate long-context needle behavior across the SWA/full-attention mix.

### Phase 2: FP8 KV cache (goal) and FP4 KV cache (stretch)

- Gap: `TRITON_ATTN_DIFFKV` currently allows only `auto`/`bfloat16` KV
  (`supported_kv_cache_dtypes`); `--kv-cache-dtype fp8_e4m3` will be
  rejected on SM120. Options, in order of preference:
  1. Extend the DiffKV triton kernels to fp8 KV (the parent
     `TRITON_ATTN` already supports fp8; the diffkv variant only changes
     K/V packing in the content dim, so this is mostly plumbing scale
     handling through the packed layout).
  2. Investigate FA4 on SM12x for `FLASH_ATTN_DIFFKV` (fa_utils
     currently restricts FA4 to capability major 10).
- FP4 KV: no generic fp4 KV path exists in vLLM. A DiffKV fp4 cache
  would need a packed 4-bit content dim + per-block scales and
  quant/dequant in the triton kernels, mirroring the earlier
  DeepSeek sparse-MLA NVFP4-KV port (~/vllm-nvfp4-port). Document-only
  for now; do not block bring-up on it.

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
