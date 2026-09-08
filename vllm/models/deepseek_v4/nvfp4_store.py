# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 store for the DeepSeek-V4 sparse-MLA *main* 512-head KV cache.

Replaces the KV side of the fp8_ds_mla C++ op
``torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`` for the
nvfp4_ds_mla cache dtype. The page ABI (384 bytes/token: group-16 E2M1 NoPE
with E4M3 scales + bf16 RoPE, page-major data/scale split) is owned by
FlashInfer; this module only applies GPT-J RoPE to the latent's rope half and
hands the roped rows to ``nvfp4_quantize_append_sparse_mla_cache``, which
quantizes and writes slots in place. Negative slot-mapping entries are padding
and are skipped by the append kernel.
"""

import torch

from vllm.utils.flashinfer import flashinfer_nvfp4_quantize_append_sparse_mla_cache

NOPE_DIM = 448
ROPE_DIM = 64
HALF_ROPE = ROPE_DIM // 2


def gptj_rope_latent(
    kv: torch.Tensor,  # [num_tokens, 512] bf16 (post-qnorm latent)
    positions: torch.Tensor,  # [num_tokens] int64
    cos_sin_cache: torch.Tensor,  # [max_pos, 64] fp32 (cos||sin, 32 each)
) -> torch.Tensor:
    """GPT-J-rotate the rope half (dims 448:512) of the DSV4 latent.

    Pairs are interleaved: pair k lives at dims (448+2k, 449+2k). Matches the
    rotation in the fused fp8 insert op: computed in fp32, stored bf16.
    """
    n = kv.shape[0]
    rope = kv[:, NOPE_DIM:].to(torch.float32).view(n, HALF_ROPE, 2)
    cs = cos_sin_cache.index_select(0, positions)
    cos = cs[:, :HALF_ROPE]
    sin = cs[:, HALF_ROPE:]
    even = rope[..., 0]
    odd = rope[..., 1]
    rotated = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    ).view(n, ROPE_DIM)
    latent = torch.empty_like(kv)
    latent[:, :NOPE_DIM] = kv[:, :NOPE_DIM]
    latent[:, NOPE_DIM:] = rotated.to(kv.dtype)
    return latent


def store_nvfp4_kv(
    kv: torch.Tensor,  # [num_tokens, 512] bf16 (post-qnorm latent)
    kv_cache: torch.Tensor,  # [num_blocks, block_size, 384] uint8
    slot_mapping: torch.Tensor,  # [num_tokens] int32/int64
    positions: torch.Tensor,  # [num_tokens] int64
    cos_sin_cache: torch.Tensor,  # [max_pos, 64] fp32 (cos||sin)
) -> None:
    """RoPE + NVFP4-quantize the DSV4 main-KV latent into the paged cache.

    Input contract matches the KV side of the fp8_ds_mla C++ insert op: the
    post-qnorm latent ``kv`` (nope[:448] RMSNorm'd, rope[448:512] NOT yet
    rotated), the paged uint8 cache, slot mapping and rope caches. slot_mapping
    may be shorter than kv (DP padding); extra rows are ignored.
    """
    assert kv.dim() == 2 and kv.shape[1] == NOPE_DIM + ROPE_DIM, (
        f"kv must be [num_tokens, 512], got {tuple(kv.shape)}"
    )
    assert kv_cache.dtype == torch.uint8, "kv_cache must be uint8"
    assert cos_sin_cache.dtype == torch.float32
    assert cos_sin_cache.shape[-1] == ROPE_DIM, (
        f"cos_sin_cache must be [max_pos, 64], got {tuple(cos_sin_cache.shape)}"
    )

    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    latent = gptj_rope_latent(
        kv[:num_tokens], positions[:num_tokens], cos_sin_cache
    ).contiguous()
    flashinfer_nvfp4_quantize_append_sparse_mla_cache(
        latent, slot_mapping.contiguous(), kv_cache
    )
