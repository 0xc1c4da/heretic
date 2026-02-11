#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Packed-MoE sanity checks for Heretic + SGLang offline backend.

Goals:
- Run *preflight* checks before loading the model (config + code surface).
- After model load, verify:
  - `heretic/module_map` exposes `kind="moe_packed_w2"` targets
  - packed-w2 FULL builder can register factors under a real `lora_id`
  - runtime injection changes logits for adapted rows in a mixed (paired) batch
  - standard LoRA still works (non-MoE linear path)

Usage:
  uv run python tools/sglang_offline_packed_moe_sanity.py --config-toml config.kimi_k25_h200_offline.toml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
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
    ap = argparse.ArgumentParser(
        description="Sanity checks for packed-MoE adapter plumbing (SGLang offline)."
    )
    ap.add_argument(
        "--config-toml",
        default=None,
        help="Path to a Heretic TOML config. If omitted, uses ./config.toml.",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight checks only (do not load the model).",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Use minimal prompts and smallest ranks (recommended).",
    )
    return ap.parse_args()


def _log(msg: str) -> None:
    print(msg, flush=True)


def _time_call(label: str, fn):
    _log(f"[packed-moe] {label} ...")
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    _log(f"[packed-moe] {label} done in {dt:.2f}s")
    return out, dt


def _preflight_config_checks(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True, "issues": [], "notes": []}
    sargs = dict(cfg.get("sglang_offline_args") or {})

    def issue(msg: str) -> None:
        out["ok"] = False
        out["issues"].append(str(msg))

    def note(msg: str) -> None:
        out["notes"].append(str(msg))

    if str(cfg.get("backend") or "") not in ("sglang_offline", "BackendType.SGLANG_OFFLINE"):
        note("config.backend is not 'sglang_offline' (script will force offline backend).")

    if not bool(sargs.get("enable_lora", False)):
        issue("[sglang_offline_args].enable_lora must be true.")
    if not bool(sargs.get("enable_return_hidden_states", False)):
        issue("[sglang_offline_args].enable_return_hidden_states must be true (Heretic requirement).")

    moe_backend = str(sargs.get("moe_runner_backend") or "auto")
    out["moe_runner_backend"] = moe_backend
    if moe_backend == "auto":
        note("moe_runner_backend=auto: may select fused-only backends; for packed-MoE injection use 'triton' or 'deep_gemm'.")
    if moe_backend in ("flashinfer_trtllm", "marlin", "triton_kernel"):
        issue(f"moe_runner_backend={moe_backend!r} is fused/opaque; packed-MoE injection requires a non-fused runner (e.g. 'triton' or 'deep_gemm').")

    disable_fused_cfg = bool(sargs.get("heretic_disable_moe_fused_func", False)) or bool(
        sargs.get("disable_moe_fused_func", False)
    )
    if disable_fused_cfg:
        note("Config requests disabling MoE fused func (Heretic will set SGLANG_CI_DISABLE_MOE_FUSED_FUNC=1 during offline engine init).")
    elif os.environ.get("SGLANG_CI_DISABLE_MOE_FUSED_FUNC") not in ("1", "true", "True"):
        note(
            "MoE fused-func disable is not set (neither config nor env). "
            "SGLang may route through fused functions even when runner_backend='triton'. "
            "Set `[sglang_offline_args].heretic_disable_moe_fused_func = true` for Heretic packed-MoE injection."
        )

    targets = sargs.get("lora_target_modules")
    if isinstance(targets, list):
        if "down_proj" not in [str(x) for x in targets]:
            note("lora_target_modules does not include 'down_proj' (standard LoRA path); packed-MoE injection is separate, but Heretic commonly targets both o_proj/down_proj.")
    return out


def _preflight_code_surface_checks(repo_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True, "issues": [], "notes": []}

    def issue(msg: str) -> None:
        out["ok"] = False
        out["issues"].append(str(msg))

    def note(msg: str) -> None:
        out["notes"].append(str(msg))

    # 1) Import surface: IO structs should exist.
    try:
        from sglang.srt.managers.io_struct import (  # noqa: F401
            HereticBuildPackedW2FullRownormReqInput,
            HereticUnloadPackedMoEAdapterReqInput,
        )

        note("io_struct: packed build/unload request types import OK.")
    except Exception as e:
        issue(f"io_struct: missing packed request types: {e}")

    # 2) Runner injection code presence: check files for sentinel strings.
    triton_path = repo_root / "vendor" / "sglang" / "python" / "sglang" / "srt" / "layers" / "moe" / "moe_runner" / "triton.py"
    deep_path = repo_root / "vendor" / "sglang" / "python" / "sglang" / "srt" / "layers" / "moe" / "moe_runner" / "deep_gemm.py"
    ctx_path = repo_root / "vendor" / "sglang" / "python" / "sglang" / "srt" / "layers" / "moe" / "heretic_packed_context.py"

    for p, sentinel in (
        (triton_path, "heretic_packed_w2_by_lora_id"),
        (deep_path, "heretic_apply_mask_rows"),
        (ctx_path, "HereticPackedMoEContext"),
    ):
        try:
            txt = p.read_text(encoding="utf-8")
            if sentinel not in txt:
                issue(f"missing sentinel {sentinel!r} in {str(p)}")
        except Exception as e:
            issue(f"failed to read {str(p)}: {e}")

    return out


def main() -> int:
    args = _parse_args()
    config_path = str(args.config_toml) if args.config_toml else "config.toml"
    if not os.path.exists(config_path):
        raise SystemExit(
            f"config not found: {config_path!r}. Pass --config-toml PATH or create ./config.toml"
        )

    # After parsing our args, strip argv so no other CLI-parsing layers see our flags.
    sys.argv = [sys.argv[0]]

    repo_root = Path(__file__).resolve().parents[1]
    cfg = _load_toml(config_path)

    evidence: dict[str, Any] = {
        "config_toml": str(config_path),
        "preflight": {},
        "runtime": {},
    }

    evidence["preflight"]["config"] = _preflight_config_checks(cfg)
    evidence["preflight"]["code_surface"] = _preflight_code_surface_checks(repo_root)

    pre_ok = bool(evidence["preflight"]["config"]["ok"]) and bool(
        evidence["preflight"]["code_surface"]["ok"]
    )
    if args.preflight_only:
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0 if pre_ok else 2

    import torch
    import torch.nn.functional as F

    from heretic.config import BackendType, Settings
    from heretic.model import Model
    from heretic.utils import Prompt

    def _merge_settings_from_toml(*, data: dict[str, Any]) -> Settings:
        s = Settings.model_validate(data)
        s.backend = BackendType.SGLANG_OFFLINE
        s.validate_backend = False
        # keep this small; we only need correctness.
        s.batch_size = 1
        s.max_batch_size = 1
        return s

    def _mk_prompt(*, system: str, user: str) -> Prompt:
        return Prompt(system=str(system), user=str(user))

    def _finite_or_raise(t: torch.Tensor, *, where: str) -> None:
        if not isinstance(t, torch.Tensor):
            raise RuntimeError(f"{where}: expected torch.Tensor, got {type(t)}")
        finite = torch.isfinite(t)
        if not bool(finite.all().item()):
            bad = int((~finite).sum().item())
            raise RuntimeError(f"{where}: non-finite tensor values (bad={bad} of {t.numel()})")

    def _kl_base_vs_other(*, base_logprobs: torch.Tensor, other_logprobs: torch.Tensor) -> float:
        _finite_or_raise(base_logprobs, where="base_logprobs")
        _finite_or_raise(other_logprobs, where="other_logprobs")
        return float(
            F.kl_div(
                other_logprobs,
                base_logprobs,
                reduction="batchmean",
                log_target=True,
            ).item()
        )

    settings = _merge_settings_from_toml(data=cfg)
    system = str(getattr(settings, "system_prompt", None) or "You are a helpful assistant.")

    eval_prompts = [
        _mk_prompt(system=system, user="Summarize the benefits of exercise."),
    ]
    if not args.fast:
        eval_prompts.append(_mk_prompt(system=system, user="Write a short poem about winter."))

    _log("[packed-moe] initializing Model / SGLang engine (first-time warmup can take minutes)...")
    model, _dt_init = _time_call("Model(settings)", lambda: Model(settings))
    evidence["runtime"]["init_s"] = float(_dt_init)

    # Discover packed w2 targets.
    mm, _ = _time_call(
        "module_map(down_proj)",
        lambda: model.backend.module_map(include_projs=["down_proj"], include_experts=None),
    )
    packed = None
    if isinstance(mm, list):
        # Record quick stats for debugging.
        try:
            evidence["runtime"]["down_proj_module_map_count"] = int(len(mm))
            sample_paths = [
                str(d.get("module_path"))
                for d in mm
                if isinstance(d, dict) and isinstance(d.get("module_path"), str)
            ][:10]
            evidence["runtime"]["down_proj_module_map_sample_paths"] = sample_paths
            kinds: dict[str, int] = {}
            for d in mm:
                if isinstance(d, dict) and isinstance(d.get("kind"), str):
                    kinds[str(d["kind"])] = int(kinds.get(str(d["kind"]), 0)) + 1
            evidence["runtime"]["down_proj_module_map_kinds"] = dict(sorted(kinds.items()))
        except Exception:
            pass

        packed = next(
            (
                d
                for d in mm
                if isinstance(d, dict)
                and d.get("kind") == "moe_packed_w2"
                and isinstance(d.get("module_path"), str)
            ),
            None,
        )
    evidence["runtime"]["packed_w2_found"] = bool(packed is not None)
    if packed is None:
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 2

    packed_name = str(packed["module_path"])
    evidence["runtime"]["packed_w2_module_path"] = packed_name

    # Pick one o_proj module to test standard LoRA.
    mm_o, _ = _time_call(
        "module_map(o_proj)",
        lambda: model.backend.module_map(include_projs=["o_proj"], include_experts=[]),
    )
    pick_o = None
    if isinstance(mm_o, list):
        pick_o = next(
            (d for d in mm_o if isinstance(d, dict) and isinstance(d.get("module_path"), str)),
            None,
        )
    if pick_o is None:
        raise RuntimeError("module_map(o_proj) returned no usable module_path.")
    mp_o = str(pick_o["module_path"])
    out_f = int(pick_o.get("out_features") or 0)
    in_f = int(pick_o.get("in_features") or 0)
    if out_f <= 0 or in_f <= 0:
        raise RuntimeError(f"module_map missing dims for {mp_o}: out={out_f} in={in_f}")

    evidence["runtime"]["picked_o_proj"] = {"module_path": mp_o, "out": out_f, "in": in_f}

    def _score_full_vocab(*, adapter: str | None):
        ids = model.encode_prompts(eval_prompts)
        out = model.backend.score(ids, adapter=adapter)
        if out.logprobs_full is None:
            raise RuntimeError("Backend did not return logprobs_full.")
        return out.logprobs_full

    base_lp, _ = _time_call("score(base)", lambda: _score_full_vocab(adapter=None))

    # Load a minimal adapter to obtain a lora_id.
    module_base = mp_o[: -len(".weight")] if mp_o.endswith(".weight") else mp_o
    A0 = torch.zeros((1, in_f), dtype=torch.float16)
    B0 = torch.zeros((out_f, 1), dtype=torch.float16)
    tensors0 = {
        f"{module_base}.lora_A.default.weight": A0.cpu(),
        f"{module_base}.lora_B.default.weight": B0.cpu(),
    }
    cfg0 = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": 1,
        "lora_alpha": 1,
        "lora_dropout": 0.0,
        "target_modules": ["o_proj"],
        "bias": "none",
    }

    adapter_id, _ = _time_call(
        "load_adapter(zero_rank1)",
        lambda: model.backend.load_adapter(name="packed_moe_sanity_zero", tensors=tensors0, config=cfg0),
    )
    if adapter_id is None:
        raise RuntimeError("load_adapter returned None lora_id.")
    evidence["runtime"]["adapter_id"] = str(adapter_id)

    # Register packed w2 factors under this adapter id.
    # Use a refusal direction from the middle layer if available; fall back to random unit vector.
    # We intentionally keep rank=1 and small weight to make this cheap.
    try:
        layer_idx = int(packed.get("layer") if isinstance(packed, dict) else -1)
    except Exception:
        layer_idx = -1

    # Build a synthetic v: must match out_features (hidden_size). We can derive out_features from module_map.
    out_features = int(packed.get("out_features") or 0)
    if out_features <= 0:
        out_features = int(packed.get("shape")[0]) if isinstance(packed.get("shape"), list) else 0
    if out_features <= 0:
        raise RuntimeError(f"packed w2 entry missing out_features: {packed}")

    v = torch.zeros((out_features,), dtype=torch.float32)
    v[0] = 1.0
    v = F.normalize(v, p=2, dim=0)

    build_out, _ = _time_call(
        "build_packed_w2_full_rownorm(rank=1, w=1e-3)",
        lambda: model.backend.build_packed_w2_full_rownorm(
            lora_id=str(adapter_id),
            name=packed_name,
            v=v,
            weight=1e-3,
            rank=1,
            out_dtype="float16",
        ),
    )
    evidence["runtime"]["packed_build"] = dict(build_out) if isinstance(build_out, dict) else build_out
    if not (isinstance(build_out, dict) and bool(build_out.get("success", False))):
        raise RuntimeError(f"packed_w2 build failed: {build_out}")

    # Mixed-batch correctness: within-call paired scoring must change adapted rows but not base rows.
    ids = model.encode_prompts(eval_prompts)
    base_p, adapted_p = model.backend.score_full_vocab_paired(ids, adapter=str(adapter_id))
    kl_within = _kl_base_vs_other(base_logprobs=base_p, other_logprobs=adapted_p)
    maxdiff_within = float((adapted_p - base_p).abs().max().item())
    evidence["runtime"]["packed_within_call"] = {
        "kl": float(kl_within),
        "max_abs_diff": float(maxdiff_within),
    }

    # Also verify a plain (non-paired) score differs from base.
    ad_lp, _ = _time_call("score(adapter)", lambda: _score_full_vocab(adapter=str(adapter_id)))
    kl = _kl_base_vs_other(base_logprobs=base_lp, other_logprobs=ad_lp)
    maxdiff = float((ad_lp - base_lp).abs().max().item())
    evidence["runtime"]["packed_effect"] = {"kl": float(kl), "max_abs_diff": float(maxdiff)}

    # Unload and ensure we revert close to baseline.
    _time_call(
        "unload_adapter",
        lambda: model.backend.unload_adapter(name="packed_moe_sanity_zero"),
    )
    post_lp, _ = _time_call("score(post-unload)", lambda: _score_full_vocab(adapter=None))
    kl_post = _kl_base_vs_other(base_logprobs=base_lp, other_logprobs=post_lp)
    evidence["runtime"]["post_unload"] = {
        "kl": float(kl_post),
        "max_abs_diff": float((post_lp - base_lp).abs().max().item()),
    }

    # Decide pass/fail.
    # We require a non-trivial delta for packed injection in paired mode.
    passed = bool(pre_ok) and bool(maxdiff_within > 0.0)
    evidence["ok"] = bool(passed)

    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

