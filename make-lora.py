#!/usr/bin/env python3
"""
Reproduce a specific Heretic trial by directly applying known parameters
to build a LoRA adapter (no Optuna trials), using the LOCAL backend.

This script:
1) Loads the model (HF local backend)
2) Loads the "good" + "bad" prompt datasets from Settings (config.toml / env / defaults)
3) Computes per-layer refusal directions (same logic as heretic.main)
4) Applies trial parameters via Model.abliterate()
5) Saves the resulting adapter (or merged model, optionally)

Usage:
  python make-lora.py --model "..." --out ./adapter-out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# Allow running from repo root without installation (imports from ./src).
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists():
    sys.path.insert(0, str(_SRC_DIR))


from heretic.config import BackendType, RowNormalization, Settings  # noqa: E402
from heretic.model import AbliterationParameters, Model  # noqa: E402
from heretic.utils import empty_cache, load_prompts, print  # noqa: E402


TRIAL_319_DIRECTION_INDEX = 36.64
TRIAL_319_PARAMETERS = {
    "attn.o_proj": AbliterationParameters(
        max_weight=3.87,
        max_weight_position=56.90,
        min_weight=1.83,
        min_weight_distance=49.40,
    ),
    "mlp.down_proj": AbliterationParameters(
        max_weight=2.29,
        max_weight_position=55.31,
        min_weight=0.66,
        min_weight_distance=31.20,
    ),
}


def _build_settings(model: str, *, full_normalization_lora_rank: int | None) -> Settings:
    """
    Force the knobs requested by the user while keeping everything else
    configurable via config.toml / env vars / defaults.
    """

    overrides: dict[str, object] = {
        "model": model,
        "backend": BackendType.LOCAL,
        "orthogonalize_direction": True,
        "row_normalization": RowNormalization.FULL,
    }
    if full_normalization_lora_rank is not None:
        overrides["full_normalization_lora_rank"] = full_normalization_lora_rank

    return Settings(**overrides)  # type: ignore[arg-type]


@torch.no_grad()
def _compute_refusal_directions(model: Model, settings: Settings) -> torch.Tensor:
    """
    Mirrors the logic in heretic.main for refusal direction computation.
    """
    print()
    print("Loading prompt datasets...")
    good_prompts = load_prompts(settings.good_prompts)
    bad_prompts = load_prompts(settings.bad_prompts)

    print()
    print("Computing per-layer refusal directions...")
    print("* Obtaining residuals for good prompts...")
    good_residuals = model.get_residuals_batched(good_prompts)
    print("* Obtaining residuals for bad prompts...")
    bad_residuals = model.get_residuals_batched(bad_prompts)

    good_means = good_residuals.mean(dim=0)
    bad_means = bad_residuals.mean(dim=0)

    refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    if settings.orthogonalize_direction:
        # Implements https://huggingface.co/blog/grimjim/projected-abliteration
        good_directions = F.normalize(good_means, p=2, dim=1)
        projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
        refusal_directions = refusal_directions - projection_vector.unsqueeze(1) * good_directions
        refusal_directions = F.normalize(refusal_directions, p=2, dim=1)

    del good_residuals, bad_residuals
    empty_cache()

    return refusal_directions


def main() -> int:
    ap = argparse.ArgumentParser(description="Directly build Heretic LoRA for trial 319 (local backend).")
    ap.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id or local path (same meaning as heretic --model).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output directory to save the adapter or merged model.",
    )
    ap.add_argument(
        "--save-strategy",
        choices=["adapter", "merge"],
        default="adapter",
        help="Save only the adapter (default) or save merged model weights.",
    )
    ap.add_argument(
        "--full-normalization-lora-rank",
        type=int,
        default=None,
        help="Override Settings.full_normalization_lora_rank (only used when row_normalization='full').",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = _build_settings(
        args.model,
        full_normalization_lora_rank=args.full_normalization_lora_rank,
    )

    print()
    print("=== make-lora.py ===")
    print(f"* model: [bold]{settings.model}[/]")
    print(f"* backend: [bold]{settings.backend}[/] (forced)")
    print(f"* orthogonalize_direction: [bold]{settings.orthogonalize_direction}[/] (forced)")
    print(f"* row_normalization: [bold]{settings.row_normalization}[/] (forced)")
    if settings.row_normalization == RowNormalization.FULL:
        print(f"* full_normalization_lora_rank: [bold]{settings.full_normalization_lora_rank}[/]")

    print()
    print("Loading model (local backend)...")
    model = Model(settings)

    refusal_directions = _compute_refusal_directions(model, settings)

    n_layers = len(model.get_layers())
    if not (0.0 <= TRIAL_319_DIRECTION_INDEX < (n_layers - 1)):
        raise ValueError(
            f"direction_index={TRIAL_319_DIRECTION_INDEX} out of range for n_layers={n_layers} "
            f"(must be in [0, {n_layers - 1}))."
        )

    print()
    print("Applying trial 319 parameters...")
    print(f"* direction_index = [bold]{TRIAL_319_DIRECTION_INDEX}[/]")
    for comp, params in TRIAL_319_PARAMETERS.items():
        print(f"* {comp}: [bold]{params}[/]")

    model.abliterate(
        refusal_directions=refusal_directions,
        direction_index=TRIAL_319_DIRECTION_INDEX,
        parameters=TRIAL_319_PARAMETERS,
    )

    print()
    print("Saving...")
    if args.save_strategy == "adapter":
        # For PEFT models this writes adapter_config.json + adapter_model.safetensors, etc.
        model.model.save_pretrained(str(out_dir))
    else:
        merged = model.get_merged_model()
        merged.save_pretrained(str(out_dir))
        del merged
        empty_cache()

    model.tokenizer.save_pretrained(str(out_dir))
    print(f"Saved to [bold]{out_dir}[/].")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

