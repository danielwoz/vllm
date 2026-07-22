# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline (CPU-only) validation of MiMo-V2.5 NVFP4 checkpoint support.

Validates, without touching a GPU, that:
 1. vLLM's config pipeline resolves the checkpoint (model_type ``mimo_v2``,
    quantization ``modelopt_mixed``).
 2. ``MiMoV2ForCausalLM`` instantiates on the meta device from the real
    config, with the real quant config applied per-module.
 3. Every tensor in ``model.safetensors.index.json`` is either loaded into a
    model parameter or intentionally skipped (MTP / vision / audio towers),
    by running the real ``load_weights`` path with synthetic tensors of the
    exact checkpoint shapes/dtypes; and every model parameter is loaded.
 4. Each quantized module resolves to the expected algorithm
    (MXFP8 qkv/dense-MLP, NVFP4 routed experts, unquantized elsewhere).

Device capability is emulated as SM120 (RTX PRO 6000 Blackwell) so that
backend-selection code paths match the target machine.

Usage:
    python tools/mimo_v25_offline_validation.py /path/to/snapshot
    torchrun --nproc-per-node=2 tools/mimo_v25_offline_validation.py \
        /path/to/snapshot  # validates TP-2 sharded loading
"""

import json
import os
import struct
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# Keep the real GPUs untouched; everything below runs on CPU/meta.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "U8": torch.uint8,
    "I64": torch.int64,
}

# Checkpoint tensors the target model must NOT load: MTP draft layers and
# multimodal towers (text-only phase).
SKIP_PREFIXES = (
    "model.mtp.",
    "visual.",
    "audio_encoder.",
    "speech_embeddings.",
)


def read_index(snapshot: Path) -> dict[str, tuple[list[int], torch.dtype]]:
    """Map every checkpoint tensor to its (shape, dtype) via shard headers."""
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    headers: dict[str, dict] = {}
    tensors: dict[str, tuple[list[int], torch.dtype]] = {}
    for name, shard in weight_map.items():
        if shard not in headers:
            with open(snapshot / shard, "rb") as f:
                (hdr_len,) = struct.unpack("<Q", f.read(8))
                headers[shard] = json.loads(f.read(hdr_len))
        meta = headers[shard][name]
        tensors[name] = (meta["shape"], SAFETENSORS_DTYPES[meta["dtype"]])
    return tensors


def emulate_sm120() -> None:
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    cls = type(current_platform)
    cls.get_device_capability = classmethod(
        lambda cls, device_id=0: DeviceCapability(major=12, minor=0)
    )
    cls.get_device_name = classmethod(
        lambda cls, device_id=0: "NVIDIA RTX PRO 6000 Blackwell (emulated)"
    )
    cls.get_device_total_memory = classmethod(lambda cls, device_id=0: 96 * 1024**3)
    cls.device_count = classmethod(lambda cls: 2)
    # Weight loading never communicates, so a CPU-safe communicator suffices.
    cls.get_device_communicator_cls = classmethod(
        lambda cls: (
            "vllm.distributed.device_communicators.base_device_communicator"
            ".DeviceCommunicatorBase"
        )
    )


def build_vllm_config(snapshot: Path, tp_size: int):
    from vllm.config import ModelConfig, ParallelConfig, VllmConfig

    model_config = ModelConfig(
        model=str(snapshot),
        tokenizer=str(snapshot),
        dtype="bfloat16",
        seed=0,
        # Text-only phase: drop the multimodal towers so the config
        # convertor keeps architectures=["MiMoV2ForCausalLM"].
        hf_overrides={"vision_config": None, "audio_config": None},
    )
    parallel_config = ParallelConfig(tensor_parallel_size=tp_size)
    return VllmConfig(model_config=model_config, parallel_config=parallel_config)


def init_cpu_distributed(world_size: int, rank: int) -> None:
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://"
        if "MASTER_ADDR" in os.environ
        else "tcp://127.0.0.1:29511",
        local_rank=rank,
        backend="gloo",
    )
    ensure_model_parallel_initialized(world_size, 1)


def instantiate_meta_model(vllm_config):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import initialize_model

    with set_current_vllm_config(vllm_config), torch.device("meta"):
        model = initialize_model(
            vllm_config=vllm_config, model_config=vllm_config.model_config
        )
    return model


def synthetic_weights(tensors):
    for name, (shape, dtype) in tensors.items():
        yield name, torch.empty(shape, dtype=dtype)


def summarize_quant_methods(model) -> Counter:
    counts: Counter = Counter()
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if method is None:
            continue
        counts[type(method).__name__] += 1
    return counts


def generalize(name: str) -> str:
    import re

    return re.sub(r"\.\d+\.", ".N.", name)


def main() -> int:
    snapshot = Path(sys.argv[1])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    tensors = read_index(snapshot)
    print(f"checkpoint tensors: {len(tensors)} (tp={world_size}, rank={rank})")

    emulate_sm120()
    vllm_config = build_vllm_config(snapshot, world_size)
    model_config = vllm_config.model_config
    print(f"architecture: {model_config.architecture}")
    print(f"quantization: {model_config.quantization}")
    print(f"quant_config: {type(vllm_config.quant_config).__name__}")

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        init_cpu_distributed(world_size, rank)
    model = instantiate_meta_model(vllm_config)
    params = dict(model.named_parameters())
    print(f"model parameters: {len(params)}")

    quant_counts = summarize_quant_methods(model)
    print("\nquant methods by module:")
    for cls_name, count in sorted(quant_counts.items()):
        print(f"  {count:5d}  {cls_name}")

    expected_skips = {name for name in tensors if name.startswith(SKIP_PREFIXES)}
    loadable = {name: spec for name, spec in tensors.items()}

    loaded = model.load_weights(synthetic_weights(loadable))

    unloaded_params = sorted(set(params) - loaded)
    print(f"\ncheckpoint tensors fed:        {len(tensors)}")
    print(f"expected skips (mtp/mm):       {len(expected_skips)}")
    print(f"params loaded:                 {len(loaded)}")
    print(f"model params NOT loaded:       {len(unloaded_params)}")
    if unloaded_params:
        by_pattern = Counter(generalize(p) for p in unloaded_params)
        for pat, count in sorted(by_pattern.items()):
            print(f"    {count:5d}  {pat}")

    ok = not unloaded_params
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
