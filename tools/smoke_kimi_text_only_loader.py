"""
Smoke test: Kimi K2.5 text-only loader path.

This script exercises Heretic's text-only loader for wrapper models that provide
`text_config` and store LM weights under `language_model.*`.

It uses the *tiny mock checkpoint* mechanism: point `--source-dir` at a cached Kimi
snapshot directory (containing remote-code files), materialize a tiny checkpoint, then
load it through `heretic.model.Model`.

Run:
  uv run python tools/smoke_kimi_text_only_loader.py --source-dir /path/to/kimi/snapshot
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

from heretic.model import Model
from heretic.utils import Prompt, print


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, help="Local Kimi K2.5 snapshot dir (remote-code style)")
    args = ap.parse_args()

    # Make the test deterministic and lightweight.
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "2")

    # Avoid constructing full Pydantic Settings (it reads config.toml). We only need
    # the fields accessed by `heretic.model.Model.__init__` for this smoke.
    settings = SimpleNamespace(
        model=str(args.source_dir),
        evaluate_model=None,
        trust_remote_code=True,
        dtypes=["float32"],
        device_map="cpu",
        max_memory=None,
        # quantization config (use none for tiny checkpoint)
        quantization="none",
        quantization_config_type=None,
        quantization_kwargs=None,
        precision_fallback_dtype="auto",
        precision_debug=False,
        # compressed-tensors knobs
        ct_fast_load=False,
        ct_loading_info=False,
        # tiny mock checkpoint knobs
        mock_tiny_model=True,
        mock_tiny_out_dir="~/.cache/heretic/mock_models",
        mock_tiny_hidden_size=64,
        mock_tiny_intermediate_size=256,
        mock_tiny_num_hidden_layers=2,
        mock_tiny_num_attention_heads=4,
        mock_tiny_num_key_value_heads=4,
        mock_tiny_max_position_embeddings=512,
        mock_tiny_sliding_window=256,
        mock_tiny_num_experts_per_tok=1,
        mock_tiny_num_local_experts=2,
        mock_tiny_seed=0,
        # generation knobs used by Model helpers
        system_prompt="You are a helpful assistant.",
        batch_size=1,
        max_response_length=8,
    )

    m = Model(settings)

    # Assert we did NOT load the multimodal wrapper.
    cls_name = m.model.__class__.__name__
    assert "KimiK25" not in cls_name, f"Expected LM backbone, got wrapper class {cls_name}"

    # Assert vision tower isn't present in module tree.
    names = [n for n, _ in m.model.named_modules()]
    forbidden = ("vision", "mm", "projector", "vit", "MoonViT")
    hits = [n for n in names if any(tok in n.lower() for tok in forbidden)]
    assert not hits, f"Found vision-ish modules in text-only model: {hits[:20]}"

    # Minimal generation sanity check.
    out = m.get_responses(
        [
            # Keep it short; we only test that generation runs.
            # system_prompt comes from Settings default.
            # Prompt format is internal; `get_responses` handles it.
            Prompt(system=settings.system_prompt, user="What is 1+1?"),
        ],
        max_new_tokens=1,
        use_cache=False,
    )
    assert isinstance(out, list) and len(out) == 1

    print("[green]OK[/] text-only loader smoke passed")


if __name__ == "__main__":
    main()

