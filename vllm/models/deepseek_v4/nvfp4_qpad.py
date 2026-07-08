# SPDX-License-Identifier: Apache-2.0
"""Standalone Q-side transform for the DeepSeek-V4 nvfp4_ds_mla path.

Produces the Q half of the fused fp8 C++ op
``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``:
  - per-head RMSNorm (no weight) over the full head_dim=512
  - GPT-J (interleaved-pair) RoPE on the last 64 dims, raw positions,
    cos_sin_cache layout [max_pos, 64] = [cos(32) | sin(32)]
  - zero-padding from n_local_heads up to padded_heads
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _qnorm_rope_pad_kernel(
    q_ptr,        # [num_tokens, n_heads, 512] bf16
    out_ptr,      # [num_tokens, padded_heads, 512] bf16 (pre-zeroed pad rows OR full store)
    positions_ptr,  # [num_tokens] int64
    cos_sin_ptr,  # [max_pos, 64] fp32: [:32]=cos, [32:]=sin
    n_heads, padded_heads,
    eps,
    HEAD_DIM: tl.constexpr,   # 512
    ROPE_DIM: tl.constexpr,   # 64
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    out_base = out_ptr + (token * padded_heads + head) * HEAD_DIM
    d = tl.arange(0, HEAD_DIM)

    if head >= n_heads:
        # zero-fill padding head slots
        tl.store(out_base + d, tl.zeros([HEAD_DIM], dtype=tl.bfloat16))
        return

    x = tl.load(q_ptr + (token * n_heads + head) * HEAD_DIM + d).to(tl.float32)

    # per-head RMSNorm, no weight
    rms = tl.sqrt(tl.sum(x * x) / HEAD_DIM + eps)
    x = x / rms

    # GPT-J interleaved RoPE on dims [448:512): pairs (448+2i, 448+2i+1)
    pos = tl.load(positions_ptr + token)
    i = tl.arange(0, ROPE_DIM // 2)  # 32
    cos = tl.load(cos_sin_ptr + pos * ROPE_DIM + i)
    sin = tl.load(cos_sin_ptr + pos * ROPE_DIM + 32 + i)
    nope: tl.constexpr = HEAD_DIM - ROPE_DIM
    even = tl.load(q_ptr + (token * n_heads + head) * HEAD_DIM + nope + 2 * i).to(tl.float32) / rms
    odd = tl.load(q_ptr + (token * n_heads + head) * HEAD_DIM + nope + 2 * i + 1).to(tl.float32) / rms
    r_even = even * cos - odd * sin
    r_odd = even * sin + odd * cos

    # store: nope dims normed, rope dims rotated
    tl.store(out_base + d, x.to(tl.bfloat16), mask=d < nope)
    tl.store(out_base + nope + 2 * i, r_even.to(tl.bfloat16))
    tl.store(out_base + nope + 2 * i + 1, r_odd.to(tl.bfloat16))


def qnorm_rope_pad(q: torch.Tensor, positions: torch.Tensor,
                   cos_sin_cache: torch.Tensor, padded_heads: int,
                   eps: float) -> torch.Tensor:
    """q: [T, H, 512] bf16 -> [T, padded_heads, 512] bf16 (normed+roped+padded)."""
    t, h, hd = q.shape
    assert hd == 512 and q.dtype == torch.bfloat16
    out = torch.empty((t, padded_heads, hd), dtype=q.dtype, device=q.device)
    _qnorm_rope_pad_kernel[(t, padded_heads)](
        q, out, positions, cos_sin_cache.to(torch.float32),
        h, padded_heads, eps, HEAD_DIM=hd, ROPE_DIM=64,
    )
    return out
