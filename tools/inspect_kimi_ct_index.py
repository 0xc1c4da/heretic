# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tiny verification script for compressed-tensors checkpoint inspection.

This does NOT load model weights. It only downloads/reads:
- `config.json`
- `model.safetensors.index.json`
and verifies that Kimi K2.5 looks like a pre-compressed pack-quantized checkpoint.
"""

from __future__ import annotations

import json

from huggingface_hub import hf_hub_download

from pathlib import Path

from heretic.runtime.ct_checkpoint_inspector import (
    CheckpointKind,
    infer_checkpoint_kind,
    load_weight_map_keys,
)


def main() -> None:
    repo_id = "moonshotai/Kimi-K2.5"
    cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    idx_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")

    cfg = json.loads(open(cfg_path, "r", encoding="utf-8").read())
    text_cfg = cfg.get("text_config", {}) if isinstance(cfg, dict) else {}
    qcfg = text_cfg.get("quantization_config", {}) if isinstance(text_cfg, dict) else {}

    expected_format = qcfg.get("format") if isinstance(qcfg, dict) else None
    quant_status = qcfg.get("quantization_status") if isinstance(qcfg, dict) else None
    expect_precompressed = bool(
        isinstance(quant_status, str) and quant_status.strip().lower() == "compressed"
    )

    keys = load_weight_map_keys(Path(idx_path))
    res = infer_checkpoint_kind(
        keys=keys,
        expected_format=expected_format,
        expect_precompressed=expect_precompressed,
    )

    print("repo_id", repo_id)
    print("index", idx_path)
    print("expected_format", expected_format)
    print("expect_precompressed", expect_precompressed)
    print("n_total_keys", res.n_total_keys)
    print("n_dense_weight", res.n_dense_weight)
    print("n_expected_artifacts", res.n_expected_artifacts)
    print("artifacts_suffixes", list(res.artifacts_suffixes))
    print("kind", res.kind)

    assert res.kind == CheckpointKind.PRECOMPRESSED, "Expected Kimi checkpoint to be precompressed"


if __name__ == "__main__":
    main()

