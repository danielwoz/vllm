# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    filter_safetensors_files_by_weight_name_prefixes,
    safetensors_weights_iterator,
)


def test_filter_duplicate_safetensors_files_missing_weight():
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, "model-00001-of-00002.safetensors")
        with open(existing_file, "wb") as f:
            f.write(b"")

        existing_file2 = os.path.join(tmpdir, "model-00002-of-00002.safetensors")
        with open(existing_file2, "wb") as f:
            f.write(b"")

        index_file = os.path.join(tmpdir, "model.safetensors.index.json")
        index_content = {
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
                "layer.2.weight": "model-00003-of-00002.safetensors",
            }
        }
        with open(index_file, "w") as f:
            json.dump(index_content, f)

        hf_weights_files = [
            os.path.join(tmpdir, "model-00001-of-00002.safetensors"),
            os.path.join(tmpdir, "model-00002-of-00002.safetensors"),
        ]

        with pytest.raises(FileNotFoundError) as exc_info:
            filter_duplicate_safetensors_files(
                hf_weights_files=hf_weights_files,
                hf_folder=tmpdir,
                index_file="model.safetensors.index.json",
            )

        assert "model-00003-of-00002.safetensors" in str(exc_info.value)


def test_filter_duplicate_safetensors_files_all_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_files = []
        for i in range(1, 3):
            file_path = os.path.join(tmpdir, f"model-0000{i}-of-00002.safetensors")
            with open(file_path, "wb") as f:
                f.write(b"")
            existing_files.append(file_path)

        index_file = os.path.join(tmpdir, "model.safetensors.index.json")
        index_content = {
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            }
        }
        with open(index_file, "w") as f:
            json.dump(index_content, f)

        filter_duplicate_safetensors_files(
            hf_weights_files=existing_files,
            hf_folder=tmpdir,
            index_file="model.safetensors.index.json",
        )


def test_safetensors_prefix_filter_uses_index_and_skips_other_tensors(tmp_path):
    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"model.layers.0.weight": torch.ones(1)}, shard_a)
    save_file(
        {
            "mtp.0.weight": torch.ones(1),
            "model.layers.1.weight": torch.ones(1),
        },
        shard_b,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.layers.0.weight": shard_a.name,
                    "mtp.0.weight": shard_b.name,
                    "model.layers.1.weight": shard_b.name,
                },
            }
        )
    )

    selected_files = filter_safetensors_files_by_weight_name_prefixes(
        [str(shard_a), str(shard_b)],
        str(tmp_path),
        "model.safetensors.index.json",
        ("mtp.",),
    )

    assert selected_files == [str(shard_b)]
    weights = dict(
        safetensors_weights_iterator(
            selected_files,
            use_tqdm_on_load=False,
            weight_name_prefixes=("mtp.",),
        )
    )
    assert set(weights) == {"mtp.0.weight"}


def test_safetensors_prefix_filter_fails_when_index_has_no_matches(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file({"model.layers.0.weight": torch.ones(1)}, shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.layers.0.weight": shard.name,
                },
            }
        )
    )

    with pytest.raises(RuntimeError, match="matching prefixes"):
        filter_safetensors_files_by_weight_name_prefixes(
            [str(shard)],
            str(tmp_path),
            "model.safetensors.index.json",
            ("mtp.",),
        )


if __name__ == "__main__":
    test_filter_duplicate_safetensors_files_missing_weight()
    test_filter_duplicate_safetensors_files_all_exist()
