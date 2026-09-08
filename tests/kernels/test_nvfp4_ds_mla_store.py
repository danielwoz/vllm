# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the nvfp4_ds_mla (SM120 FlashInfer sparse) KV store path.

The fused fp8 C++ op `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` is
replaced on this path by `qnorm_rope_pad` (Q side, triton) and
`store_nvfp4_kv` (KV side, rope in torch + FlashInfer quantize-append into
its opaque 384 B/token page). These tests check the vLLM-side transforms
against the PyTorch references used for the fused op, and the slot-mapping
contract of the store; NVFP4 quantization fidelity and the page byte layout
are owned and tested by FlashInfer.
"""

import pytest
import torch

from tests.kernels.utils import bf16_ulp_distance
from vllm.platforms import current_platform

from .test_fused_deepseek_v4_qnorm_rope_kv_insert import (
    apply_rope_gptj_last_k,
    make_cos_sin_cache,
    rmsnorm_no_weight,
)

HEAD_DIM = 512
ROPE_DIM = 64
NVFP4_PAGE_BYTES = 384
MAX_POS = 8192
EPS = 1e-6


def _nvfp4_append_available() -> bool:
    if not current_platform.is_cuda():
        return False
    if current_platform.get_device_capability().major not in (10, 12):
        return False
    from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_nvfp4

    return has_flashinfer_sparse_mla_sm120_nvfp4()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("num_tokens", [1, 17, 2048])
@pytest.mark.parametrize("n_heads,padded_heads", [(16, 16), (16, 64), (64, 128)])
def test_qnorm_rope_pad_matches_reference(
    num_tokens: int, n_heads: int, padded_heads: int
):
    from vllm.models.deepseek_v4.nvfp4_qpad import qnorm_rope_pad

    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.randint(0, MAX_POS, (num_tokens,), device=device)
    cos_sin = make_cos_sin_cache(MAX_POS, ROPE_DIM, torch.float32, device)

    out = qnorm_rope_pad(q, positions, cos_sin, padded_heads, EPS)

    ref = rmsnorm_no_weight(q, EPS)
    ref = apply_rope_gptj_last_k(ref, positions, cos_sin).to(torch.bfloat16)

    assert out.shape == (num_tokens, padded_heads, HEAD_DIM)
    assert bf16_ulp_distance(out[:, :n_heads], ref).max() <= 2
    assert (out[:, n_heads:] == 0).all()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("num_tokens", [1, 17, 2048])
def test_gptj_rope_latent_matches_reference(num_tokens: int):
    from vllm.models.deepseek_v4.nvfp4_store import NOPE_DIM, gptj_rope_latent

    torch.manual_seed(1)
    device = "cuda"
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.randint(0, MAX_POS, (num_tokens,), device=device)
    cos_sin = make_cos_sin_cache(MAX_POS, ROPE_DIM, torch.float32, device)

    out = gptj_rope_latent(kv, positions, cos_sin)

    ref = apply_rope_gptj_last_k(kv.float(), positions, cos_sin).to(torch.bfloat16)
    assert (out[:, :NOPE_DIM] == kv[:, :NOPE_DIM]).all()
    assert bf16_ulp_distance(out, ref).max() <= 1


@pytest.mark.skipif(
    not _nvfp4_append_available(),
    reason="requires FlashInfer NVFP4 sparse MLA modules on SM10x/SM12x",
)
@pytest.mark.parametrize("block_size", [64, 256])
def test_store_nvfp4_kv_slot_contract(block_size: int):
    """Valid slots are written, negative slots and rows past the slot mapping
    are ignored, and untargeted cache entries are untouched."""
    from vllm.models.deepseek_v4.nvfp4_store import store_nvfp4_kv

    torch.manual_seed(2)
    device = "cuda"
    num_blocks = 4
    num_tokens, pad = 13, 3
    kv = torch.randn(num_tokens + pad, HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.randint(0, MAX_POS, (num_tokens + pad,), device=device)
    cos_sin = make_cos_sin_cache(MAX_POS, ROPE_DIM, torch.float32, device)

    sentinel = 0xAB
    kv_cache = torch.full(
        (num_blocks, block_size, NVFP4_PAGE_BYTES),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )
    total_slots = num_blocks * block_size
    slots = torch.randperm(total_slots, device=device)[:num_tokens].to(torch.int64)
    slot_mapping = slots.clone()
    slot_mapping[::4] = -1  # DP/profiling padding entries

    store_nvfp4_kv(kv, kv_cache, slot_mapping, positions, cos_sin)

    flat = kv_cache.view(total_slots, NVFP4_PAGE_BYTES)
    written = torch.zeros(total_slots, dtype=torch.bool, device=device)
    valid = slot_mapping[slot_mapping >= 0]
    written[valid] = True
    changed = (flat != sentinel).any(dim=-1)
    assert (changed == written).all(), (
        "store must write exactly the non-negative slot-mapping targets"
    )
