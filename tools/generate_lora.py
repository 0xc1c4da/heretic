#!/usr/bin/env python3
"""
Reconstruct and export a Heretic LoRA adapter without running optimization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    import tomli as tomllib  # type: ignore[import-not-found]

# Make repository root importable regardless of invocation cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if _REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _REPO_ROOT.as_posix())

from src.heretic.config import BackendType, Settings  # noqa: E402
from src.heretic.model import AbliterationParameters, Model  # noqa: E402
from src.heretic.peft_packed_moe import (  # noqa: E402
    inject_exported_packed_w2_factors_into_bundle,
    register_packed_w2_full_builds,
)
from src.heretic.refusal_cache import (  # noqa: E402
    compute_identity as compute_refusal_cache_identity,
    get_cache_dir as get_refusal_cache_dir,
    identity_hash as refusal_cache_identity_hash,
    probe_refusal_cache,
    save_refusal_directions,
)
from src.heretic.utils import empty_cache, load_prompts  # noqa: E402

_ABLITERATION_FIELDS = (
    "max_weight",
    "max_weight_position",
    "min_weight",
    "min_weight_distance",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a PEFT LoRA adapter from Heretic config + trial parameters.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to Heretic TOML config file.")
    parser.add_argument("--output", type=str, required=True, help="Output adapter directory.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional override for settings.model.",
    )
    parser.add_argument(
        "--study-jsonl",
        type=str,
        default=None,
        help="Path to Optuna JournalStorage JSONL file.",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=None,
        help="Trial index to extract from study user_attrs['index'].",
    )
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help="JSON file containing direction_index + component parameters.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Repeatable KEY=VALUE parameter (e.g. attn.o_proj.max_weight=2.58).",
    )
    parser.add_argument(
        "--direction-index",
        type=float,
        default=None,
        help="Optional global direction index for --param mode (omit for per-layer).",
    )
    parser.add_argument(
        "--save-tokenizer",
        action="store_true",
        help="Also save tokenizer files in output directory.",
    )
    return parser


def _load_settings_from_toml(config_path: str, *, model_override: str | None) -> Settings:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_path!r} must parse to a TOML table.")
    if model_override:
        raw["model"] = model_override
    return Settings.model_validate(raw)


def _sanitized_model_study_name(model_name: str) -> str:
    return "".join((c if (c.isalnum() or c in ["_", "-"]) else "--") for c in str(model_name))


def _derive_study_jsonl_path(settings: Settings) -> str:
    return os.path.join(
        str(settings.study_checkpoint_dir),
        _sanitized_model_study_name(str(settings.model)) + ".jsonl",
    )


def _to_float(value: Any, *, where: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Expected numeric value for {where}, got {value!r}.") from exc
    if not torch.isfinite(torch.tensor(out)):
        raise ValueError(f"Expected finite numeric value for {where}, got {value!r}.")
    return out


def _build_component_parameters(component: str, payload: dict[str, Any]) -> AbliterationParameters:
    missing = [name for name in _ABLITERATION_FIELDS if name not in payload]
    unknown = [name for name in payload.keys() if name not in _ABLITERATION_FIELDS]
    if missing or unknown:
        raise ValueError(
            f"Invalid parameter block for component {component!r}: "
            f"missing={missing or 'none'}, unknown={unknown or 'none'}."
        )
    return AbliterationParameters(
        max_weight=_to_float(payload["max_weight"], where=f"{component}.max_weight"),
        max_weight_position=_to_float(
            payload["max_weight_position"],
            where=f"{component}.max_weight_position",
        ),
        min_weight=_to_float(payload["min_weight"], where=f"{component}.min_weight"),
        min_weight_distance=_to_float(
            payload["min_weight_distance"],
            where=f"{component}.min_weight_distance",
        ),
    )


def _parse_params_json_payload(payload: dict[str, Any]) -> tuple[float | None, dict[str, AbliterationParameters]]:
    if not isinstance(payload, dict):
        raise ValueError("Parameter JSON must be an object.")
    if "parameters" not in payload:
        raise ValueError("Parameter JSON is missing required key: 'parameters'.")
    raw_direction_index = payload.get("direction_index")
    direction_index = None
    if raw_direction_index is not None:
        direction_index = _to_float(raw_direction_index, where="direction_index")

    raw_parameters = payload["parameters"]
    if not isinstance(raw_parameters, dict) or not raw_parameters:
        raise ValueError("'parameters' must be a non-empty object.")

    parameters: dict[str, AbliterationParameters] = {}
    for component, component_values in raw_parameters.items():
        if not isinstance(component, str) or not component:
            raise ValueError(f"Invalid component key in parameters: {component!r}")
        if not isinstance(component_values, dict):
            raise ValueError(f"Expected object for component {component!r}, got {type(component_values).__name__}.")
        parameters[component] = _build_component_parameters(component, component_values)
    return direction_index, parameters


def _parse_cli_params(
    raw_items: list[str],
    *,
    direction_index: float | None,
) -> tuple[float | None, dict[str, AbliterationParameters]]:
    grouped: dict[str, dict[str, float]] = {}

    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --param value {item!r}; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid --param value {item!r}; empty key.")
        if key == "direction_index":
            raise ValueError("Use --direction-index instead of --param direction_index=...")

        parts = key.split(".")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid parameter key {key!r}; expected dotted form COMPONENT.FIELD (e.g. attn.o_proj.max_weight)."
            )
        component = ".".join(parts[:-1])
        field = parts[-1]

        grouped.setdefault(component, {})
        if field in grouped[component]:
            raise ValueError(f"Duplicate parameter provided: {key}")
        grouped[component][field] = _to_float(value, where=key)

    if not grouped:
        raise ValueError("No component parameters provided.")

    parameters = {component: _build_component_parameters(component, values) for component, values in grouped.items()}
    return direction_index, parameters


def _extract_trial_from_trials(
    trials: list[Any],
    *,
    trial_index: int,
) -> tuple[float | None, dict[str, AbliterationParameters]]:
    for trial in trials:
        if trial.user_attrs.get("index") == trial_index:
            raw_direction_index = trial.user_attrs.get("direction_index")
            direction_index = (
                None
                if raw_direction_index is None
                else _to_float(raw_direction_index, where=f"trial[{trial_index}].direction_index")
            )
            raw_params = trial.user_attrs.get("parameters", {})
            if not isinstance(raw_params, dict) or not raw_params:
                raise ValueError(
                    f"Trial index={trial_index} has no valid 'parameters' payload in user_attrs."
                )
            parameters: dict[str, AbliterationParameters] = {}
            for component, component_values in raw_params.items():
                if not isinstance(component_values, dict):
                    raise ValueError(
                        f"Trial index={trial_index} has invalid parameter block for {component!r}: "
                        f"{type(component_values).__name__}."
                    )
                parameters[str(component)] = _build_component_parameters(
                    str(component),
                    component_values,
                )
            return direction_index, parameters

    available = sorted(
        int(i)
        for i in (
            t.user_attrs.get("index")
            for t in trials
        )
        if isinstance(i, int)
    )
    raise ValueError(f"Trial index={trial_index} not found. Available: {available}")


def extract_trial_from_study(jsonl_path: str, trial_index: int) -> tuple[float | None, dict[str, AbliterationParameters]]:
    lock_obj = JournalFileOpenLock(jsonl_path)
    backend = JournalFileBackend(jsonl_path, lock_obj=lock_obj)
    storage = JournalStorage(backend)

    studies = storage.get_all_studies()
    if not studies:
        raise ValueError(f"No studies found in {jsonl_path}")

    study_id = studies[0]._study_id
    trials = storage.get_all_trials(study_id)
    return _extract_trial_from_trials(trials, trial_index=trial_index)


def _resolve_parameter_source(
    args: argparse.Namespace,
    settings: Settings,
) -> tuple[float | None, dict[str, AbliterationParameters]]:
    using_trial = args.trial is not None
    using_json = args.params_json is not None
    using_param = bool(args.param)

    sources_count = int(using_trial) + int(using_json) + int(using_param)
    if sources_count != 1:
        raise ValueError(
            "Provide exactly one parameter source: "
            "--study-jsonl/--trial, or --params-json, or one/more --param."
        )
    if args.study_jsonl and not using_trial:
        raise ValueError("--study-jsonl requires --trial.")
    if args.direction_index is not None and not using_param:
        raise ValueError("--direction-index can only be used with --param.")

    if using_trial:
        study_path = args.study_jsonl or _derive_study_jsonl_path(settings)
        if not os.path.exists(study_path):
            raise ValueError(
                f"Study JSONL file not found: {study_path}. "
                "Provide --study-jsonl explicitly or ensure settings.study_checkpoint_dir contains the derived file."
            )
        return extract_trial_from_study(study_path, int(args.trial))

    if using_json:
        with open(args.params_json, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"--params-json must contain an object, got {type(payload).__name__}.")
        return _parse_params_json_payload(payload)

    return _parse_cli_params(args.param, direction_index=args.direction_index)


def _load_or_compute_refusal_directions(settings: Settings, model: Model) -> torch.Tensor:
    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)

    refusal_directions: torch.Tensor | None = None
    cache_identity: dict[str, Any] | None = None
    cache_dir: str | None = None

    if getattr(settings, "refusal_cache", True):
        try:
            cache_identity = compute_refusal_cache_identity(
                settings=settings,
                model=model,
                good_prompts=good_prompts,
                bad_prompts=bad_prompts,
            )
            cache_dir = get_refusal_cache_dir(settings)
            cached, reason = probe_refusal_cache(cache_dir, cache_identity)
            if cached is not None:
                refusal_directions = cached
                print(f"Refusal cache hit ({refusal_cache_identity_hash(cache_identity)[:12]})")
            else:
                print(f"Refusal cache miss ({reason}); computing residuals.")
        except Exception as exc:
            print(f"Refusal cache disabled (identity error: {exc})")
            cache_identity = None
            cache_dir = None

    if refusal_directions is None:
        good_residuals = model.get_residuals_batched(good_prompts)
        bad_residuals = model.get_residuals_batched(bad_prompts)

        good_means = good_residuals.mean(dim=0)
        bad_means = bad_residuals.mean(dim=0)
        refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

        if settings.orthogonalize_direction:
            good_directions = F.normalize(good_means, p=2, dim=1)
            projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
            refusal_directions = refusal_directions - projection_vector.unsqueeze(1) * good_directions
            refusal_directions = F.normalize(refusal_directions, p=2, dim=1)

        del good_residuals, bad_residuals
        empty_cache()

        if getattr(settings, "refusal_cache", True) and cache_identity is not None and cache_dir is not None:
            try:
                save_refusal_directions(cache_dir, cache_identity, refusal_directions)
                print(f"Refusal cache saved ({refusal_cache_identity_hash(cache_identity)[:12]})")
            except Exception as exc:
                print(f"Refusal cache save failed: {exc}")

    return refusal_directions


def _materialize_packed_moe_factors(model: Model, bundle: Any) -> None:
    builds = getattr(bundle, "packed_w2_full_builds", None)
    if not builds:
        return

    adapter_name = "reconstruct_export"
    adapter_loaded = False
    adapter_id = None
    try:
        adapter_id = model.backend.load_adapter(
            name=adapter_name,
            tensors=bundle.tensors,
            config=bundle.config_dict,
        )
        adapter_loaded = True
        if adapter_id is None:
            raise RuntimeError("Backend returned empty adapter ID while packed-MoE export is required.")

        # Respect the build dict recorded in the bundle (authoritative).
        register_packed_w2_full_builds(
            backend=model.backend,
            adapter_id=str(adapter_id),
            builds=list(builds),
        )

        inject_exported_packed_w2_factors_into_bundle(
            bundle=bundle,
            backend=model.backend,
            adapter_id=str(adapter_id),
            expert_down_proj_leaf="down_proj",
        )
    finally:
        if adapter_loaded:
            try:
                model.backend.unload_adapter(name=adapter_name)
            except Exception as exc:
                print(f"Warning: failed to unload adapter {adapter_name!r}: {exc}")


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = _load_settings_from_toml(args.config, model_override=args.model)
    backend_type = getattr(settings, "backend", BackendType.LOCAL)
    if backend_type not in (BackendType.SGLANG, BackendType.SGLANG_OFFLINE):
        raise RuntimeError(
            "This tool exports LoRA bundles via SGLang backend paths only. "
            f"Unsupported backend={backend_type!s}."
        )

    model = Model(settings)
    refusal_directions = _load_or_compute_refusal_directions(settings, model)
    direction_index, parameters = _resolve_parameter_source(args, settings)

    bundle = model.build_lora_adapter_bundle(
        refusal_directions,
        direction_index,
        parameters,
    )
    _materialize_packed_moe_factors(model, bundle)

    tokenizer = model.tokenizer if bool(args.save_tokenizer) else None
    bundle.save_pretrained(str(args.output), tokenizer=tokenizer)
    print(f"Adapter exported to {args.output}")
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
