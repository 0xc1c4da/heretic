#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
SGLang-offline smoke suite to isolate Heretic KL pathologies.
---------------------------------------------

This script is intentionally "close to Heretic" and low-cognitive-load:
- One required arg: --config-toml
- Uses `Settings` + `Model` (encode_prompts, residual capture, adapter export match trials)
- Uses SGLang offline backend (`backend="sglang_offline"`)
- Computes KL the same way as `Evaluator.get_score()`

What it answers quickly:
1) Does *residual capture* change the backend scoring state?
   i.e. KL(score_before_residuals || score_after_residuals)
2) Does a "trial-like" exported adapter behave like a near-no-op at tiny weights?
   i.e. compare KL vs baseline (captured after residual capture).

If (1) is large, Heretic's current baseline capture timing is wrong for this backend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

try:
    import tomllib  # py>=3.11
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

def _load_toml(path: str) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("tomllib is not available; need Python 3.11+ to read TOML.")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"TOML root must be a table/dict: {path}")
    return dict(data)


def _parse_args() -> argparse.Namespace:
    # IMPORTANT: do this before importing Heretic/SGLang modules.
    # Some dependencies (or older versions of this script) may instantiate `Settings()`,
    # which uses Pydantic's CLI source and can choke on unknown args.
    ap = argparse.ArgumentParser(
        description="SGLang offline smoke suite for Heretic KL issues (one-command, minimal knobs)."
    )
    ap.add_argument(
        "--config-toml",
        default=None,
        help="Path to a Heretic TOML config. If omitted, uses ./config.toml.",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = str(args.config_toml) if args.config_toml else "config.toml"
    if not os.path.exists(config_path):
        raise SystemExit(
            f"config not found: {config_path!r}. Pass --config-toml PATH or create ./config.toml"
        )

    # After parsing our args, strip argv so no other CLI-parsing layers see our flags.
    sys.argv = [sys.argv[0]]

    import torch
    import torch.nn.functional as F

    from heretic.config import BackendType, Settings
    from heretic.model import AbliterationParameters, Model
    from heretic.utils import Prompt

    def _finite_or_raise(t: torch.Tensor, *, where: str) -> None:
        if not isinstance(t, torch.Tensor):
            raise RuntimeError(f"{where}: expected torch.Tensor, got {type(t)}")
        finite = torch.isfinite(t)
        if not bool(finite.all().item()):
            bad = int((~finite).sum().item())
            t_min = float(t[finite].min().item()) if bool(finite.any().item()) else float("nan")
            t_max = float(t[finite].max().item()) if bool(finite.any().item()) else float("nan")
            raise RuntimeError(
                f"{where}: non-finite tensor values (bad={bad} of {t.numel()}) "
                f"(finite_min={t_min:.4g} finite_max={t_max:.4g})"
            )

    def _kl_base_vs_other(*, base_logprobs: torch.Tensor, other_logprobs: torch.Tensor) -> float:
        """
        Match Heretic evaluator semantics:
          F.kl_div(input=other, target=base, log_target=True) == KL(base || other)
        """
        _finite_or_raise(base_logprobs, where="base_logprobs")
        _finite_or_raise(other_logprobs, where="other_logprobs")
        if base_logprobs.shape != other_logprobs.shape:
            raise RuntimeError(
                f"shape mismatch: base={tuple(base_logprobs.shape)} other={tuple(other_logprobs.shape)}"
            )
        return float(
            F.kl_div(
                other_logprobs,
                base_logprobs,
                reduction="batchmean",
                log_target=True,
            ).item()
        )

    def _merge_settings_from_toml(*, data: dict[str, Any]) -> Settings:
        # Settings is a BaseSettings, but model_validate() works fine for dict input.
        s = Settings.model_validate(data)
        # Force offline backend for this script.
        s.backend = BackendType.SGLANG_OFFLINE
        # Don't run heavy validations inside the smoke suite unless the user explicitly wants it.
        s.validate_backend = False
        return s

    def _mk_prompt(*, system: str, user: str) -> Prompt:
        return Prompt(system=str(system), user=str(user))

    def _score_full_vocab(model: Model, prompts: list[Prompt], *, adapter: str | None) -> torch.Tensor:
        input_ids_batch = model.encode_prompts(prompts)
        out = model.backend.score(input_ids_batch, adapter=adapter)
        t = out.logprobs_full
        if t is None:
            raise RuntimeError("Backend did not return logprobs_full (required).")
        return t

    def _compute_refusal_directions_like_main_from_residuals(
        *, settings: Settings, good_residuals: torch.Tensor, bad_residuals: torch.Tensor
    ) -> torch.Tensor:
        good_means = good_residuals.mean(dim=0)
        bad_means = bad_residuals.mean(dim=0)

        refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

        if bool(getattr(settings, "orthogonalize_direction", False)):
            good_directions = F.normalize(good_means, p=2, dim=1)
            projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
            refusal_directions = (
                refusal_directions - projection_vector.unsqueeze(1) * good_directions
            )
            refusal_directions = F.normalize(refusal_directions, p=2, dim=1)

        return refusal_directions

    def _mk_constant_params(model: Model, *, weight: float) -> dict[str, AbliterationParameters]:
        # Use constant weight for all layers by setting min_weight == max_weight.
        # Include (almost) all layers by setting min_weight_distance very large.
        n_layers = int(model.num_layers)
        pos = float(0.5 * max(0, n_layers - 1))
        dist = float(max(1, n_layers))  # include all layers
        params: dict[str, AbliterationParameters] = {}
        for component in model.get_abliterable_components():
            params[component] = AbliterationParameters(
                max_weight=float(weight),
                max_weight_position=pos,
                min_weight=float(weight),
                min_weight_distance=dist,
            )
        return params

    cfg = _load_toml(config_path)
    settings = _merge_settings_from_toml(data=cfg)
    # Keep this fast-ish.
    settings.batch_size = 1
    settings.max_batch_size = 1

    model = Model(settings)

    system = str(getattr(settings, "system_prompt", None) or "You are a helpful assistant.")
    good_prompts = [_mk_prompt(system=system, user="Hello.")]
    bad_prompts = [_mk_prompt(system=system, user="Tell me how to build a bomb.")]
    eval_prompts = [_mk_prompt(system=system, user="Summarize the benefits of exercise.")]

    # Fixed tiny weights: these should be near-no-ops in a correct setup.
    weights_to_test = [0.0, 1e-4]

    # Evidence bundle (always printed at the end).
    evidence: dict[str, Any] = {
        "config_toml": str(config_path),
        "backend": "sglang_offline",
        "model": str(getattr(settings, "model", "")),
        "row_normalization": str(getattr(settings, "row_normalization", "")),
        "orthogonalize_direction": bool(getattr(settings, "orthogonalize_direction", False)),
        "winsorization_quantile": float(getattr(settings, "winsorization_quantile", 1.0)),
        "sglang_offline_args": dict(getattr(settings, "sglang_offline_args", {}) or {}),
        "prompts": {
            "good": [good_prompts[0].user],
            "bad": [bad_prompts[0].user],
            "eval": [eval_prompts[0].user],
        },
        "token_lens": {},
        "checks": {},
        "adapter": {},
    }

    # Precompute token lengths (helps confirm we’re scoring the same thing).
    try:
        evidence["token_lens"] = {
            "good": len(model.encode_prompts(good_prompts)[0]),
            "bad": len(model.encode_prompts(bad_prompts)[0]),
            "eval": len(model.encode_prompts(eval_prompts)[0]),
        }
    except Exception as e:
        evidence["token_lens_error"] = str(e)

    # Conservative failure threshold: if we see multi-point KL jumps, it's definitely pathological.
    # (The exact bug you’re chasing shows KL ~3+.)
    FAIL_KL = 1.0
    failed = False

    def _record(check: str, *, kl: float | None, maxdiff: float | None, note: str | None = None) -> None:
        nonlocal failed
        evidence["checks"][check] = {
            "kl": None if kl is None else float(kl),
            "max_abs_diff": None if maxdiff is None else float(maxdiff),
            "note": note,
        }
        if kl is not None and float(kl) >= FAIL_KL:
            failed = True

    # Check 0: base score repeat (sanity: scoring itself isn't completely unstable).
    base0 = _score_full_vocab(model, eval_prompts, adapter=None)
    base0b = _score_full_vocab(model, eval_prompts, adapter=None)
    kl_00 = _kl_base_vs_other(base_logprobs=base0, other_logprobs=base0b)
    md_00 = float((base0b - base0).abs().max().item())
    _record("base_repeat", kl=kl_00, maxdiff=md_00)

    # Check 1: residual-capture warmup drift.
    good_residuals = model.get_residuals_batched(good_prompts)
    bad_residuals = model.get_residuals_batched(bad_prompts)

    base1 = _score_full_vocab(model, eval_prompts, adapter=None)
    kl_01 = _kl_base_vs_other(base_logprobs=base0, other_logprobs=base1)
    md_01 = float((base1 - base0).abs().max().item())
    _record("after_residual_capture", kl=kl_01, maxdiff=md_01)

    # Compute refusal directions from the already-captured residuals (avoid additional backend calls).
    refusal_directions = _compute_refusal_directions_like_main_from_residuals(
        settings=settings, good_residuals=good_residuals, bad_residuals=bad_residuals
    )
    evidence["adapter"]["refusal_directions_shape"] = tuple(int(x) for x in refusal_directions.shape)

    # Use the same global direction selection that the default trial scope uses.
    direction_index = float(0.5 * max(0, int(model.num_layers) - 1))
    evidence["adapter"]["direction_index"] = float(direction_index)

    # Build + load adapters at fixed tiny weights, then compare against base1 (post-residual baseline).
    # This matches how trials behave: they evaluate after residual capture has already happened.
    adapter_cases: list[dict[str, Any]] = []
    for w in weights_to_test:
        params = _mk_constant_params(model, weight=float(w))
        bundle = model.build_lora_adapter_bundle(
            refusal_directions=refusal_directions,
            direction_index=direction_index,
            parameters=params,
        )
        case: dict[str, Any] = {
            "weight": float(w),
            "exported_tensors": int(bundle.stats.get("exported_tensors") or 0),
        }
        # Sample tensor metadata (helps spot obvious shape/path issues).
        if bundle.tensors:
            k0 = next(iter(bundle.tensors.keys()))
            v0 = bundle.tensors[k0]
            case["sample_tensor"] = {"key": k0, "shape": [int(x) for x in v0.shape], "dtype": str(v0.dtype)}

        adapter_name = f"smoke_w_{w:g}".replace(".", "_")
        adapter_id: str | None = None
        try:
            adapter_id = model.backend.load_adapter(
                name=adapter_name, tensors=bundle.tensors, config=bundle.config_dict
            )
            case["adapter_id"] = adapter_id
            lp_ad = _score_full_vocab(model, eval_prompts, adapter=adapter_id)
            case["kl_base1_adapt"] = _kl_base_vs_other(base_logprobs=base1, other_logprobs=lp_ad)
            case["maxdiff_base1_adapt"] = float((lp_ad - base1).abs().max().item())
        finally:
            try:
                model.backend.unload_adapter(name=adapter_name)
            except Exception as e:
                case["unload_error"] = str(e)

        # Post-unload baseline: did we return to base1?
        base_post = _score_full_vocab(model, eval_prompts, adapter=None)
        case["kl_base1_post_unload"] = _kl_base_vs_other(base_logprobs=base1, other_logprobs=base_post)
        case["maxdiff_base1_post_unload"] = float((base_post - base1).abs().max().item())

        # Flag failure if any case produces multi-point KL.
        if float(case.get("kl_base1_adapt", 0.0) or 0.0) >= FAIL_KL:
            failed = True
        if float(case.get("kl_base1_post_unload", 0.0) or 0.0) >= FAIL_KL:
            failed = True

        adapter_cases.append(case)

    evidence["adapter"]["cases"] = adapter_cases

    # Print a single-line headline + full JSON evidence.
    status = "FAIL" if failed else "OK"
    print(status)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

