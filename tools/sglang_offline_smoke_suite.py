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
import time
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

    def _log(msg: str) -> None:
        print(msg, flush=True)

    def _time_call(label: str, fn):
        _log(f"[smoke] {label} ...")
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        _log(f"[smoke] {label} done in {dt:.2f}s")
        return out, dt

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

    def _score_full_vocab(model: Model, prompts: list[Prompt], *, adapter: str | None):
        input_ids_batch = model.encode_prompts(prompts)
        out = model.backend.score(input_ids_batch, adapter=adapter)
        t = out.logprobs_full
        if t is None:
            raise RuntimeError("Backend did not return logprobs_full (required).")
        return out

    def _get_last_prompt_sha(out, *, idx: int) -> str | None:
        try:
            meta = getattr(out, "meta", None) or {}
            sha_list = meta.get("heretic_input_ids_sha256")
            if isinstance(sha_list, list) and 0 <= int(idx) < len(sha_list):
                v = sha_list[int(idx)]
                return str(v) if isinstance(v, str) else None
        except Exception:
            return None
        return None

    def _get_meta_int(out, *, key: str, idx: int) -> int | None:
        try:
            meta = getattr(out, "meta", None) or {}
            xs = meta.get(str(key))
            if isinstance(xs, list) and 0 <= int(idx) < len(xs):
                v = xs[int(idx)]
                if v is None:
                    return None
                return int(v)
        except Exception:
            return None
        return None

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
        # NOTE: For FULL rownorm this can be extremely expensive if it includes many layers.
        # This helper remains for parity with Heretic, but the smoke suite below does NOT
        # build full bundles by default (it uses per-module tests instead).
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

    _log("[smoke] initializing Model / SGLang engine (first-time warmup can take minutes)...")
    model, _dt_init = _time_call("Model(settings)", lambda: Model(settings))

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

    # Conservative failure threshold: if we see multi-point KL jumps in *within-call* diagnostics,
    # it's definitely pathological. Cross-call drift is expected/allowed on some SGLang stacks and
    # should not fail the suite when paired scoring is available.
    FAIL_KL = 1.0
    failed = False

    supports = model.backend.get_metadata().supports
    supports_paired = bool(supports.get("score_full_vocab_paired", False))
    supports_noise = bool(supports.get("score_full_vocab_paired_with_noise", False))

    def _record(
        check: str,
        *,
        kl: float | None,
        maxdiff: float | None,
        note: str | None = None,
        counts_for_fail: bool = True,
    ) -> None:
        nonlocal failed
        evidence["checks"][check] = {
            "kl": None if kl is None else float(kl),
            "max_abs_diff": None if maxdiff is None else float(maxdiff),
            "note": note,
        }
        if counts_for_fail and kl is not None and float(kl) >= FAIL_KL:
            failed = True

    # Check 0: base score repeat (cross-call; may be meaningless under drift).
    base0r, _ = _time_call("score(base) #1", lambda: _score_full_vocab(model, eval_prompts, adapter=None))
    base0br, _ = _time_call("score(base) #2", lambda: _score_full_vocab(model, eval_prompts, adapter=None))
    base0 = base0r.logprobs_full
    base0b = base0br.logprobs_full
    kl_00 = _kl_base_vs_other(base_logprobs=base0, other_logprobs=base0b)
    md_00 = float((base0b - base0).abs().max().item())
    _record("base_repeat", kl=kl_00, maxdiff=md_00, counts_for_fail=not supports_paired)
    _log(f"[smoke] KL(base#1 || base#2) = {kl_00:.6g} (max|diff|={md_00:.6g})")

    # Check 0b: within-call repeatability (single request, duplicated batch).
    if supports_paired:
        base0_pair, _ = _time_call(
            "score(base) within-call (duplicated batch)",
            lambda: _score_full_vocab(model, eval_prompts + eval_prompts, adapter=None),
        )
        base0_pair_t = base0_pair.logprobs_full
        if base0_pair_t is not None and base0_pair_t.ndim == 2 and base0_pair_t.shape[0] == 2:
            # Hard invariant: duplicated prompts must have identical prompt-id hash at capture.
            sha0 = _get_last_prompt_sha(base0_pair, idx=0)
            sha1 = _get_last_prompt_sha(base0_pair, idx=1)
            tp0 = _get_meta_int(base0_pair, key="heretic_tp_rank", idx=0)
            tp1 = _get_meta_int(base0_pair, key="heretic_tp_rank", idx=1)
            vd0 = _get_meta_int(base0_pair, key="heretic_vocab_dim", idx=0)
            vd1 = _get_meta_int(base0_pair, key="heretic_vocab_dim", idx=1)
            _log(
                f"[smoke] duplicated meta: sha=({sha0},{sha1}) tp_rank=({tp0},{tp1}) vocab_dim=({vd0},{vd1})"
            )
            if sha0 is not None and sha1 is not None and sha0 != sha1:
                raise RuntimeError(
                    f"Duplicated scoring prompts have different prompt hashes: {sha0} != {sha1}"
                )
            base0a = base0_pair_t[:1]
            base0c = base0_pair_t[1:]
            kl_00w = _kl_base_vs_other(base_logprobs=base0a, other_logprobs=base0c)
            md_00w = float((base0c - base0a).abs().max().item())
            _record("base_repeat_within_call", kl=kl_00w, maxdiff=md_00w, counts_for_fail=True)
            _log(
                f"[smoke] KL(base||base) within-call = {kl_00w:.6g} (max|diff|={md_00w:.6g})"
            )

    # Check 1: residual-capture warmup drift.
    good_residuals, _ = _time_call(
        "capture residuals (good)",
        lambda: model.get_residuals_batched(good_prompts),
    )
    bad_residuals, _ = _time_call(
        "capture residuals (bad)",
        lambda: model.get_residuals_batched(bad_prompts),
    )

    base1r, _ = _time_call("score(base after residuals)", lambda: _score_full_vocab(model, eval_prompts, adapter=None))
    base1 = base1r.logprobs_full
    kl_01 = _kl_base_vs_other(base_logprobs=base0, other_logprobs=base1)
    md_01 = float((base1 - base0).abs().max().item())
    _record("after_residual_capture", kl=kl_01, maxdiff=md_01, counts_for_fail=not supports_paired)
    _log(f"[smoke] KL(base_pre || base_post_residuals) = {kl_01:.6g} (max|diff|={md_01:.6g})")

    # Compute refusal directions from the already-captured residuals (avoid additional backend calls).
    refusal_directions, _ = _time_call(
        "compute refusal_directions",
        lambda: _compute_refusal_directions_like_main_from_residuals(
            settings=settings, good_residuals=good_residuals, bad_residuals=bad_residuals
        ),
    )
    evidence["adapter"]["refusal_directions_shape"] = tuple(int(x) for x in refusal_directions.shape)

    # Use the same global direction selection that the default trial scope uses.
    direction_index = float(0.5 * max(0, int(model.num_layers) - 1))
    evidence["adapter"]["direction_index"] = float(direction_index)

    # Adapter lifecycle check without building a huge bundle:
    # - Pick ONE o_proj weight near the chosen direction layer
    # - Load a true no-op adapter (rank=1, A=B=0) to test "LoRA path" and unload cleanliness
    # - Optionally test SGLang's FULL rownorm builder for that same module at 0 and 1e-4
    #
    # This keeps the suite faithful to the backend mechanics while staying fast.
    mm, _ = _time_call(
        "module_map (pick single o_proj)",
        lambda: model.backend.module_map(
            include_projs=["o_proj"],
            include_layers=[int(round(direction_index))],
            include_experts=[],
            max_experts_per_layer=1,
            expert_strategy="first",
        ),
    )
    if isinstance(mm, list) and mm and isinstance(mm[0], list):
        mm = mm[0]
    if not isinstance(mm, list) or not mm:
        # Fallback: ask without include_layers.
        mm, _ = _time_call(
            "module_map (fallback, no include_layers)",
            lambda: model.backend.module_map(
                include_projs=["o_proj"],
                include_layers=None,
                include_experts=[],
                max_experts_per_layer=1,
                expert_strategy="first",
            ),
        )
        if isinstance(mm, list) and mm and isinstance(mm[0], list):
            mm = mm[0]
    if not isinstance(mm, list) or not mm:
        raise RuntimeError("module_map returned no modules; cannot run adapter lifecycle checks.")

    pick = next((d for d in mm if isinstance(d, dict) and isinstance(d.get("module_path"), str)), None)
    if pick is None:
        raise RuntimeError("module_map did not include a usable module_path entry.")
    mp = str(pick["module_path"])
    out_f = int(pick.get("out_features") or 0)
    in_f = int(pick.get("in_features") or 0)
    if out_f <= 0 or in_f <= 0:
        raise RuntimeError(f"module_map missing in/out dims for {mp}: out={out_f} in={in_f}")
    evidence["adapter"]["picked_module_path"] = mp
    evidence["adapter"]["picked_dims"] = {"out_features": out_f, "in_features": in_f}

    # Optional: discover packed MoE w2 targets (truthful fused expert weights).
    packed_mm, _ = _time_call(
        "module_map (discover packed w2_weight)",
        lambda: model.backend.module_map(
            include_projs=["down_proj"],
            include_layers=[int(round(direction_index))],
            include_experts=None,
            max_experts_per_layer=None,
            expert_strategy="first",
        ),
    )
    if isinstance(packed_mm, list) and packed_mm and isinstance(packed_mm[0], list):
        packed_mm = packed_mm[0]
    packed_w2 = None
    if isinstance(packed_mm, list):
        packed_w2 = next(
            (
                d
                for d in packed_mm
                if isinstance(d, dict)
                and d.get("kind") == "moe_packed_w2"
                and isinstance(d.get("module_path"), str)
            ),
            None,
        )
    evidence["adapter"]["packed_w2_found"] = bool(packed_w2 is not None)
    if packed_w2 is not None:
        evidence["adapter"]["packed_w2_module_path"] = str(packed_w2["module_path"])

    def _run_adapter_case(
        *,
        name: str,
        tensors: dict[str, torch.Tensor],
        config_dict: dict[str, Any],
        note: str,
        post_load_fn=None,
    ) -> None:
        adapter_id, _ = _time_call(
            f"load_adapter({name})",
            lambda: model.backend.load_adapter(name=name, tensors=tensors, config=config_dict),
        )
        if post_load_fn is not None and adapter_id is not None:
            _time_call(
                f"{name}: post_load_fn",
                lambda: post_load_fn(str(adapter_id)),
            )
        lp_adr, _ = _time_call(
            f"score(adapter {name})",
            lambda: _score_full_vocab(model, eval_prompts, adapter=adapter_id),
        )
        lp_ad = lp_adr.logprobs_full
        kl = _kl_base_vs_other(base_logprobs=base1, other_logprobs=lp_ad)
        md = float((lp_ad - base1).abs().max().item())
        _record(f"{name}_kl", kl=kl, maxdiff=md, note=note, counts_for_fail=not supports_paired)
        _log(f"[smoke] KL(base1 || {name}) = {kl:.6g} (max|diff|={md:.6g})")

        # Within-call paired KL (architecturally meaningful on drift-y backends).
        if supports_noise and adapter_id is not None:
            ids = model.encode_prompts(eval_prompts)
            base_p, adapted_p, base2_p = model.backend.score_full_vocab_paired_with_noise(
                ids, adapter=str(adapter_id)
            )
            # Hard invariant: all rows in the paired call must share the same prompt hash.
            # We read it from the backend side-channel set during scoring calls.
            sha_list = getattr(model.backend, "_last_score_prompt_sha256", None)
            if isinstance(sha_list, list) and len(sha_list) >= 3:
                sha_base1 = sha_list[0]
                sha_ad = sha_list[len(ids)] if len(ids) < len(sha_list) else None
                sha_base2 = sha_list[2 * len(ids)] if 2 * len(ids) < len(sha_list) else None
                if (
                    isinstance(sha_base1, str)
                    and isinstance(sha_ad, str)
                    and isinstance(sha_base2, str)
                    and not (sha_base1 == sha_ad == sha_base2)
                ):
                    raise RuntimeError(
                        f"Paired scoring prompt hash mismatch: base1={sha_base1} adapted={sha_ad} base2={sha_base2}"
                    )
            kl_p = _kl_base_vs_other(base_logprobs=base_p, other_logprobs=adapted_p)
            md_p = float((adapted_p - base_p).abs().max().item())
            kl_noise = _kl_base_vs_other(base_logprobs=base_p, other_logprobs=base2_p)
            md_noise = float((base2_p - base_p).abs().max().item())
            _record(f"{name}_kl_within_call", kl=kl_p, maxdiff=md_p, note=note, counts_for_fail=True)
            _record(
                f"{name}_kl_noise_within_call",
                kl=kl_noise,
                maxdiff=md_noise,
                note=note,
                counts_for_fail=True,
            )
            _log(
                f"[smoke] KL(base||{name}) within-call = {kl_p:.6g} (max|diff|={md_p:.6g})"
            )
            _log(
                f"[smoke] KL_noise(base||base2) within-call = {kl_noise:.6g} (max|diff|={md_noise:.6g})"
            )
        elif supports_paired and adapter_id is not None:
            ids = model.encode_prompts(eval_prompts)
            base_p, adapted_p = model.backend.score_full_vocab_paired(ids, adapter=str(adapter_id))
            kl_p = _kl_base_vs_other(base_logprobs=base_p, other_logprobs=adapted_p)
            md_p = float((adapted_p - base_p).abs().max().item())
            _record(f"{name}_kl_within_call", kl=kl_p, maxdiff=md_p, note=note, counts_for_fail=True)
            _log(
                f"[smoke] KL(base||{name}) within-call = {kl_p:.6g} (max|diff|={md_p:.6g})"
            )
        _, _ = _time_call(
            f"unload_adapter({name})",
            lambda: model.backend.unload_adapter(name=name),
        )
        base_postr, _ = _time_call(
            f"score(post-unload {name})",
            lambda: _score_full_vocab(model, eval_prompts, adapter=None),
        )
        base_post = base_postr.logprobs_full
        kl_post = _kl_base_vs_other(base_logprobs=base1, other_logprobs=base_post)
        md_post = float((base_post - base1).abs().max().item())
        _record(f"{name}_post_unload", kl=kl_post, maxdiff=md_post, note=note, counts_for_fail=not supports_paired)
        _log(f"[smoke] KL(base1 || post-unload {name}) = {kl_post:.6g} (max|diff|={md_post:.6g})")

    # Case A: explicit true no-op (rank=1) to test "LoRA path" + unload cleanliness.
    module_base = mp[: -len(".weight")] if mp.endswith(".weight") else mp
    rank0 = 1
    A0 = torch.zeros((rank0, in_f), dtype=torch.float16)
    B0 = torch.zeros((out_f, rank0), dtype=torch.float16)
    tensors0 = {
        f"{module_base}.lora_A.default.weight": A0.cpu(),
        f"{module_base}.lora_B.default.weight": B0.cpu(),
    }
    cfg0 = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": int(rank0),
        "lora_alpha": int(rank0),
        "lora_dropout": 0.0,
        "target_modules": ["o_proj"],
        "bias": "none",
    }
    _run_adapter_case(
        name="explicit_zero_rank1",
        tensors=tensors0,
        config_dict=cfg0,
        note=f"module_path={mp} rank=1",
    )

    # Case A2: packed-w2 FULL builder + injection, registered under a standard lora_id.
    if packed_w2 is not None:
        packed_name = str(packed_w2["module_path"])
        packed_layer = packed_w2.get("layer")
        if not isinstance(packed_layer, int):
            packed_layer = int(round(direction_index))

        def _register_packed(adapter_id: str) -> None:
            v_local = refusal_directions[int(packed_layer)].detach().to(torch.float32).cpu()
            model.backend.build_packed_w2_full_rownorm(
                lora_id=str(adapter_id),
                name=str(packed_name),
                v=v_local,
                weight=1e-3,
                rank=1,
                out_dtype="float16",
            )

        _run_adapter_case(
            name="packed_w2_full_rank1",
            tensors=tensors0,
            config_dict=cfg0,
            note=f"packed_name={packed_name} rank=1 weight=1e-3",
            post_load_fn=_register_packed,
        )

    # Case B: FULL rownorm builder for one module (rank reduced for smoke speed).
    # This exercises the same SGLang primitive Heretic uses under row_normalization=full,
    # without exporting a huge bundle across layers.
    FULL_RANK_SMOKE = 16
    v_vec = refusal_directions[int(round(direction_index))].to(torch.float32)
    for w in (0.0, 1e-4):
        A, B = model.backend.build_full_rownorm_lora(
            name=mp,
            v=v_vec,
            weight=float(w),
            rank=int(FULL_RANK_SMOKE),
            out_dtype="float16",
        )
        tensors = {
            f"{module_base}.lora_A.default.weight": A.to(torch.float16).cpu(),
            f"{module_base}.lora_B.default.weight": B.to(torch.float16).cpu(),
        }
        cfg = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "inference_mode": True,
            "r": int(FULL_RANK_SMOKE),
            "lora_alpha": int(FULL_RANK_SMOKE),
            "lora_dropout": 0.0,
            "target_modules": ["o_proj"],
            "bias": "none",
        }
        _run_adapter_case(
            name=f"full_rownorm_rank{FULL_RANK_SMOKE}_w{w:g}".replace(".", "_"),
            tensors=tensors,
            config_dict=cfg,
            note=f"module_path={mp} rank={FULL_RANK_SMOKE} weight={w:g}",
        )

    # Print a single-line headline + full JSON evidence.
    status = "FAIL" if failed else "OK"
    print(status)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

