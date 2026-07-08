# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NVFP4 (E2M1) store for the DeepSeek-V4 sparse-MLA *main* 512-head KV cache.

Triton replacement for the KV side of the fp8_ds_mla C++ op
``torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``. Writes the
360-byte NVFP4 page the matching CUDA decode kernel reads.

LOCKED PAGE LAYOUT (per token, 360 bytes; footer structure, pbs tokens/block):
  data region (352 bytes / token, laid out contiguously for all pbs tokens):
    [0   : 224)  448 E2M1 (NVFP4) nibbles of the NoPE latent, 2 nibbles/byte
                 byte b  = low nibble: dim 2b, high nibble: dim 2b+1
    [224 : 352)  64 bf16 RoPE values (128 B), GPT-J rotated, copied unchanged
  footer region (after all pbs tokens' 352-byte data blocks):
    for token t:  offset = pbs*352 + t*8  →  8 bytes =
                  7 UE8M0 per-64-block scales + 1 zero pad byte

The DSV4 fp8 layout stores nope=448 fp8 @ [0:448) + rope=128B @ [448:576) with
an 8B UE8M0 footer; here the NoPE region is 4-bit E2M1 (224 B), so the per-token
data stride is 352 instead of 576 and the total page is 360 vs 584.

QUANTIZATION (must match how the decode kernel dequantizes):
  For each 64-element NoPE block choose a UE8M0 (power-of-2) scale
      s = 2^ceil(log2(amax / 6.0))         (6.0 == E2M1 max magnitude)
  store the exponent byte  e = (ceil(log2(amax/6.0)) + 127)  and quantize each
  value to the nearest E2M1 code of (val / s) via the hardware
  ``cvt.rn.satfinite.e2m1x2.f32`` instruction. RoPE stays bf16.

Round-trip (decode): E2M1 code c with UE8M0 byte e reads back as
  magnitude(c) * 2^(e - 127) == c*s  ≈  val.
"""

import torch

from vllm.triton_utils import tl, triton

# ---------------------------------------------------------------------------
# E2M1 packer. Packs two fp32 lanes into one uint8: low nibble = x_lo,
# high nibble = x_hi (E2M1: sign<<3 | exp<<1 | mant; magnitudes
# {0,0.5,1,1.5,2,3,4,6}). ``satfinite`` clamps |x|>6 to the ±6 code, so no
# explicit clamp is needed after dividing by the per-block scale.
# ---------------------------------------------------------------------------
@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # NOTE: $1 is high nibble, $2 is low nibble
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _store_nvfp4_kv_kernel(
    # ── input latent (already qnorm'd; rope half NOT yet rotated) ──
    kv_ptr,               # [num_tokens, 512] bf16/fp32, contiguous
    kv_stride0,
    # ── metadata ──
    slot_mapping_ptr,     # [num_tokens] int64
    positions_ptr,        # [num_tokens] int64
    # ── RoPE ──
    cos_sin_cache_ptr,    # [max_pos, ROPE_DIM] fp32 (cos||sin, each HALF_ROPE)
    cos_sin_stride,
    # ── KV cache output (flattened uint8 view [num_blocks, block_stride]) ──
    kv_cache_ptr,
    kv_cache_block_size,  # pbs: tokens per paged cache block
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,       # 512
    NOPE_DIM: tl.constexpr,        # 448
    ROPE_DIM: tl.constexpr,        # 64
    HALF_ROPE: tl.constexpr,       # 32
    QUANT_BLOCK: tl.constexpr,     # 64
    N_QUANT_BLOCKS: tl.constexpr,  # 8  (= HEAD_SIZE // QUANT_BLOCK; row 7 = rope, unstored)
    N_NOPE_BLOCKS: tl.constexpr,   # 7  (= NOPE_DIM // QUANT_BLOCK)
    NOPE_PACKED_BYTES: tl.constexpr,  # 224 (= NOPE_DIM // 2)
    TOKEN_STRIDE: tl.constexpr,    # 352 (= 224 packed E2M1 + 128 bf16 rope)
    SCALE_DIM: tl.constexpr,       # 8  (7 UE8M0 + 1 pad)
    TRITON_BLOCK_SIZE: tl.constexpr,  # 512
    KV_BLOCK_STRIDE: tl.constexpr,    # kv_cache.stride(0), bytes per paged block
    E2M1_MAX: tl.constexpr,        # 6.0
):
    """One program per token. Writes the 360-byte NVFP4 page for its slot.

    Fused: GPT-J RoPE on the rope half (dims 448:512) + per-64 UE8M0 E2M1 quant
    of the NoPE half (dims 0:448) + nibble pack + footer scale write.
    """
    token_idx = tl.program_id(0)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        return

    kv_block_idx = slot_idx // kv_cache_block_size
    kv_pos_in_block = slot_idx % kv_cache_block_size

    # int64: block_idx * block_stride can exceed 2^31 with many KV-cache blocks.
    cache_block_ptr = kv_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE

    # Token data pointer: contiguous 352-byte data blocks at start of the page.
    val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE          # E2M1 @ [0:224)
    rope_ptr = val_ptr + NOPE_PACKED_BYTES                              # bf16  @ [224:352)
    # Footer: after ALL pbs data blocks, 8 bytes/token.
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    # ── Load latent [512] ─────────────────────────────────────────────
    block = tl.arange(0, TRITON_BLOCK_SIZE)
    x = tl.load(kv_ptr + token_idx * kv_stride0 + block).to(tl.float32)

    # ── Even/odd split (dim 2i = even, dim 2i+1 = odd) ────────────────
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2           # 256
    NOPE_PAIRS: tl.constexpr = NOPE_DIM // 2                   # 224
    pair_2d = tl.reshape(x, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)  # each [256] fp32

    # ── GPT-J forward RoPE on the rope pairs (pair_idx >= 224) ────────
    # Uses the raw token position (no compressor decimation) and the SWA main cache.
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    position = tl.load(positions_ptr + token_idx)
    cache_base = cos_sin_cache_ptr + position * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)  # [512] fp32, rope pairs rotated

    # Store rotated rope portion (dims 448:512) as bf16 into [224:352).
    bf16_ptr = rope_ptr.to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_DIM
    is_rope = block >= NOPE_DIM
    tl.store(bf16_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)

    # ── Per-64 UE8M0 E2M1 quant of the NoPE half ──────────────────────
    # bf16-roundtrip the quant input for numeric parity with the fp8 store. The
    # nope half uses the UN-rotated even/odd (rope only rotates pairs >= 224,
    # which are masked out below).
    even_q = even.to(tl.bfloat16).to(tl.float32)
    odd_q = odd.to(tl.bfloat16).to(tl.float32)

    # Tile into (N_QUANT_BLOCKS, HALF_BLOCK): block j = pairs [32j..32j+31]
    # = dims [64j..64j+63], exactly one UE8M0 block. Row 7 covers the rope
    # dims and is computed but never stored (masked by N_NOPE_BLOCKS / 224 B).
    HALF_BLOCK: tl.constexpr = QUANT_BLOCK // 2                # 32
    even_2d = tl.reshape(even_q, (N_QUANT_BLOCKS, HALF_BLOCK))
    odd_2d = tl.reshape(odd_q, (N_QUANT_BLOCKS, HALF_BLOCK))

    amax = tl.maximum(
        tl.max(tl.abs(even_2d), axis=1),
        tl.max(tl.abs(odd_2d), axis=1),
    )  # [N_QUANT_BLOCKS]
    # 6 * 2^-126 guard: keeps log2 finite for an all-zero block and produces the
    # minimum UE8M0 exponent.
    amax = tl.maximum(amax, E2M1_MAX * (2.0**-126))

    # UE8M0 block scale: s = 2^ceil(log2(amax / 6.0)); byte = exp + 127.
    log2_ratio = tl.ceil(tl.log2(amax * (1.0 / E2M1_MAX)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    inv_scale = tl.exp2(-log2_ratio)                          # 1 / s
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)                 # [N_QUANT_BLOCKS]

    inv_scale_col = tl.reshape(inv_scale, (N_QUANT_BLOCKS, 1))
    # low nibble = even (dim 2b), high nibble = odd (dim 2b+1). ``satfinite``
    # clamps |val/s| > 6 to the ±6 code — no explicit clamp needed.
    packed = _fp32x2_to_fp4x2(even_2d * inv_scale_col, odd_2d * inv_scale_col)
    packed_flat = tl.reshape(packed, (TRITON_BLOCK_SIZE // 2,))  # [256] uint8

    # Store the first 224 packed bytes (7 nope blocks). Row 7's 32 bytes (rope)
    # are masked off.
    byte_idx = tl.arange(0, TRITON_BLOCK_SIZE // 2)
    tl.store(val_ptr + byte_idx, packed_flat, mask=byte_idx < NOPE_PACKED_BYTES)

    # Footer: 7 UE8M0 scale bytes + 1 zero pad byte.
    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    tl.store(scale_ptr + scale_idx, ue8m0, mask=scale_idx < N_NOPE_BLOCKS)
    tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))


def store_nvfp4_kv(
    kv: torch.Tensor,             # [num_tokens, 512] bf16 (post-qnorm latent)
    kv_cache: torch.Tensor,       # [num_blocks, block_stride] uint8 (flattened)
    slot_mapping: torch.Tensor,   # [num_tokens] int64
    positions: torch.Tensor,      # [num_tokens] int64
    cos_sin_cache: torch.Tensor,  # [max_pos, 64] fp32 (cos||sin)
    block_size: int,              # pbs: tokens per paged cache block
) -> None:
    """Quantize the DSV4 main-KV latent to NVFP4 and insert into the paged cache.

    Inputs match the KV side of the fp8_ds_mla C++ op
    ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``:
    the post-qnorm latent ``kv`` (nope[:448] already RMSNorm'd, rope[448:512]
    NOT yet rotated), the paged uint8 cache, the slot mapping, token positions,
    and the rope cos/sin cache. GPT-J RoPE, E2M1 quant, nibble pack and the
    footer UE8M0 scales are all done inside the kernel.

    ``kv_cache`` must be the *flattened* per-block byte view, i.e. shape
    ``[num_blocks, block_stride]`` uint8 with ``block_stride`` >= pbs*352 +
    pbs*8 = pbs*360.
    """
    assert kv.dim() == 2 and kv.shape[1] == 512, (
        f"kv must be [num_tokens, 512], got {tuple(kv.shape)}"
    )
    assert kv_cache.dtype == torch.uint8, "kv_cache must be uint8"
    assert positions.dtype == torch.int64
    assert cos_sin_cache.dtype == torch.float32
    assert cos_sin_cache.shape[-1] == 64, (
        f"cos_sin_cache must be [max_pos, 64], got {tuple(cos_sin_cache.shape)}"
    )

    HEAD_SIZE = 512
    NOPE_DIM = 448
    ROPE_DIM = 64
    QUANT_BLOCK = 64
    NOPE_PACKED_BYTES = NOPE_DIM // 2          # 224
    ROPE_BYTES = ROPE_DIM * 2                   # 128
    TOKEN_STRIDE = NOPE_PACKED_BYTES + ROPE_BYTES  # 352
    SCALE_DIM = 8                               # 7 UE8M0 + 1 pad
    N_QUANT_BLOCKS = HEAD_SIZE // QUANT_BLOCK   # 8
    N_NOPE_BLOCKS = NOPE_DIM // QUANT_BLOCK     # 7

    # DP padding: slot_mapping may be shorter than kv.
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return

    block_stride = kv_cache.stride(0)           # bytes per paged block
    assert block_stride >= block_size * (TOKEN_STRIDE + SCALE_DIM), (
        f"block_stride {block_stride} < pbs*360 "
        f"({block_size * (TOKEN_STRIDE + SCALE_DIM)})"
    )

    grid = (num_tokens,)
    _store_nvfp4_kv_kernel[grid](
        kv,
        kv.stride(0),
        slot_mapping,
        positions,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        block_size,
        HEAD_SIZE=HEAD_SIZE,
        NOPE_DIM=NOPE_DIM,
        ROPE_DIM=ROPE_DIM,
        HALF_ROPE=ROPE_DIM // 2,
        QUANT_BLOCK=QUANT_BLOCK,
        N_QUANT_BLOCKS=N_QUANT_BLOCKS,
        N_NOPE_BLOCKS=N_NOPE_BLOCKS,
        NOPE_PACKED_BYTES=NOPE_PACKED_BYTES,
        TOKEN_STRIDE=TOKEN_STRIDE,
        SCALE_DIM=SCALE_DIM,
        TRITON_BLOCK_SIZE=HEAD_SIZE,
        KV_BLOCK_STRIDE=block_stride,
        E2M1_MAX=6.0,
        num_warps=4,
    )
