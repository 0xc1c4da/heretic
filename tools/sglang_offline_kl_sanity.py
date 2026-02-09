#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
SGLang offline KL sanity checks.

Why this script exists:
- SGLang offline engine launches multiprocessing children (often using "spawn").
- Running via stdin / `python -c` can crash children with:
    FileNotFoundError: ... '/mnt/vdb/heretic/<stdin>'
- A real .py file with a proper __main__ entrypoint avoids that.

What it checks:
1) Determinism / state-leakage: KL(base || base) for the same prompts twice.
2) LoRA no-op correctness: build a FULL rownorm adapter with weight≈0 for one o_proj
   module, load it, then compute KL(base || adapter). This should be ~0.

Adapter modes:
- full_rownorm_zero: uses SGLang's heretic_build_full_rownorm_lora with weITight=0.0
- explicit_zero: bypasses FULL rownorm builder and directly loads A=0 and B=0 tensors
  (isolates LoRA application path from the builder math)

To minimize cost/time:
- Use --suite to run multiple adapter checks in one engine instance.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

import torch
import torch.nn.functional as F

from heretic.backend.sglang_offline import SGLangOfflineBackend
from heretic.hf_resolve import resolve_model_dir


_LAYER_RE = re.compile(r"layers\.(\d+)\.")

try:
    import tomllib  # py>=3.11
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


def _kl_base_vs_other(*, base_logprobs: torch.Tensor, other_logprobs: torch.Tensor) -> float:
    """
    Match Heretic's evaluator semantics:
      F.kl_div(input=other, target=base, log_target=True) == KL(base || other)
    """
    return float(
        F.kl_div(
            other_logprobs,
            base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
    )


def _tensor_summary(name: str, t: torch.Tensor) -> str:
    finite = torch.isfinite(t)
    bad = int((~finite).sum().item())
    if bad == 0:
        mn = float(t.min().item())
        mx = float(t.max().item())
        return f"{name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} min={mn:.6g} max={mx:.6g} finite=all"
    nan = int(torch.isnan(t).sum().item())
    posinf = int(torch.isposinf(t).sum().item())
    neginf = int(torch.isneginf(t).sum().item())
    # Avoid reductions over empty finite set.
    if bool(finite.any().item()):
        mn = float(t[finite].min().item())
        mx = float(t[finite].max().item())
        mm = f"finite_min={mn:.6g} finite_max={mx:.6g}"
    else:
        mm = "finite_min=nan finite_max=nan"
    return (
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
        f"nonfinite={bad} (nan={nan} +inf={posinf} -inf={neginf}) {mm}"
    )


def _select_o_proj_module_path(mods: list[dict[str, Any]]) -> str:
    # Prefer a path that includes layers.<idx>. because SGLang LoRA loader uses get_layer_id().
    for d in mods:
        p = d.get("module_path")
        if isinstance(p, str) and "o_proj" in p and p.endswith(".weight") and _LAYER_RE.search(p):
            return p
    # Fallback: any .weight path.
    for d in mods:
        p = d.get("module_path")
        if isinstance(p, str) and p.endswith(".weight"):
            return p
    raise RuntimeError("module_map() returned no usable module_path entries.")

def _select_o_proj_module_desc(mods: list[dict[str, Any]]) -> dict[str, Any]:
    # Prefer a path that includes layers.<idx>. because SGLang LoRA loader uses get_layer_id().
    for d in mods:
        if not isinstance(d, dict):
            continue
        p = d.get("module_path")
        if isinstance(p, str) and "o_proj" in p and p.endswith(".weight") and _LAYER_RE.search(p):
            return d
    for d in mods:
        if not isinstance(d, dict):
            continue
        p = d.get("module_path")
        if isinstance(p, str) and p.endswith(".weight"):
            return d
    raise RuntimeError("module_map() returned no usable module descriptors.")


def _get_out_in_features(desc: dict[str, Any]) -> tuple[int, int]:
    out_f = desc.get("out_features")
    in_f = desc.get("in_features")
    if isinstance(out_f, int) and isinstance(in_f, int) and out_f > 0 and in_f > 0:
        return int(out_f), int(in_f)
    shape = desc.get("shape")
    if isinstance(shape, list) and len(shape) == 2 and all(isinstance(x, int) for x in shape):
        out_f = int(shape[0])
        in_f = int(shape[1])
        if out_f > 0 and in_f > 0:
            return out_f, in_f
    raise RuntimeError(f"module_map descriptor missing usable out/in features: keys={sorted(desc.keys())} desc={desc}")


def _load_engine_args_from_toml(path: str) -> tuple[str | None, bool | None, dict[str, Any]]:
    """
    Best-effort parse of Heretic TOML configs.

    Returns (model, trust_remote_code, sglang_offline_args_dict).
    """
    if tomllib is None:
        raise RuntimeError("tomllib is not available; need Python 3.11+ to read TOML.")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    model = data.get("model")
    trust_remote_code = data.get("trust_remote_code")
    args = data.get("sglang_offline_args") or {}
    if not isinstance(args, dict):
        raise RuntimeError(f"[sglang_offline_args] must be a table/dict in {path}")
    return (str(model) if isinstance(model, str) else None, bool(trust_remote_code) if isinstance(trust_remote_code, bool) else None, dict(args))


def _parse_engine_args_json(s: str) -> dict[str, Any]:
    try:
        data = json.loads(s)
    except Exception as e:
        raise RuntimeError(f"--engine-args-json must be valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError("--engine-args-json must decode to an object/dict.")
    return dict(data)


def _build_full_rownorm_adapter(
    *,
    backend: SGLangOfflineBackend,
    module_path_weight: str,
    rank: int,
    hidden_size: int,
    weight: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Build a minimal adapter (PEFT-style keys) for one module using SGLang's FULL rownorm builder.
    """
    if hidden_size <= 0:
        raise RuntimeError(f"Invalid hidden_size={hidden_size}")

    module_base = module_path_weight[: -len(".weight")] if module_path_weight.endswith(".weight") else module_path_weight

    # Deterministic unit vector; any nonzero vector is fine for weight=0.0, but keep it stable.
    v = torch.zeros((hidden_size,), dtype=torch.float32)
    v[0] = 1.0

    A, B = backend.build_full_rownorm_lora(
        name=module_path_weight,
        v=v,
        weight=float(weight),
        rank=int(rank),
        out_dtype="float16",
    )

    # Fail fast if the backend produced invalid factors (this would poison scoring/KL).
    if not bool(torch.isfinite(A).all().item()):
        raise RuntimeError("Non-finite values in LoRA factor A. " + _tensor_summary("A", A))
    if not bool(torch.isfinite(B).all().item()):
        raise RuntimeError("Non-finite values in LoRA factor B. " + _tensor_summary("B", B))

    tensors = {
        f"{module_base}.lora_A.default.weight": A.to(torch.float16).cpu(),
        f"{module_base}.lora_B.default.weight": B.to(torch.float16).cpu(),
    }

    # Keep scaling=1.0 by setting lora_alpha == r (SGLang: scaling = lora_alpha / r).
    config_dict: dict[str, Any] = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": int(rank),
        "lora_alpha": int(rank),
        "lora_dropout": 0.0,
        "target_modules": ["o_proj"],
        "bias": "none",
    }

    return tensors, config_dict


def _build_explicit_zero_adapter(
    *,
    module_path_weight: str,
    out_features: int,
    in_features: int,
    rank: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Build a minimal adapter with A=0 and B=0 (true no-op delta).

    This bypasses SGLang's FULL rownorm builder to isolate LoRA application/parity issues.
    """
    if out_features <= 0 or in_features <= 0:
        raise RuntimeError(f"Invalid module dims: out_features={out_features} in_features={in_features}")
    if rank <= 0:
        raise RuntimeError(f"Invalid rank={rank}")

    module_base = (
        module_path_weight[: -len(".weight")] if module_path_weight.endswith(".weight") else module_path_weight
    )

    A = torch.zeros((int(rank), int(in_features)), dtype=torch.float16)
    B = torch.zeros((int(out_features), int(rank)), dtype=torch.float16)

    tensors = {
        f"{module_base}.lora_A.default.weight": A.cpu(),
        f"{module_base}.lora_B.default.weight": B.cpu(),
    }

    config_dict: dict[str, Any] = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": int(rank),
        # scaling = lora_alpha / r; keep scaling=1.0
        "lora_alpha": int(rank),
        "lora_dropout": 0.0,
        "target_modules": ["o_proj"],
        "bias": "none",
    }
    return tensors, config_dict


def _run_adapter_case(
    *,
    backend: SGLangOfflineBackend,
    adapter_name: str,
    input_ids_batch: list[list[int]],
    base_logprobs: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    adapter_id: str | None = None
    try:
        adapter_id = backend.load_adapter(name=adapter_name, tensors=tensors, config=config_dict)
        out: dict[str, Any] = {"adapter_id": adapter_id}

        # Cross-call diagnostic (may be meaningless under drift).
        lp_ad = backend.score(input_ids_batch, adapter=adapter_id).logprobs_full

        if not bool(torch.isfinite(lp_ad).all().item()):
            out["ok"] = False
            out["base_summary"] = _tensor_summary("base", base_logprobs)
            out["adapted_summary"] = _tensor_summary("adapted", lp_ad)
            return out

        kl = _kl_base_vs_other(base_logprobs=base_logprobs, other_logprobs=lp_ad)
        max_diff = float((lp_ad - base_logprobs).abs().max().item())
        out.update({"ok": True, "kl": float(kl), "max_diff": float(max_diff)})

        # Within-call paired diagnostic (architecturally meaningful on drift-y backends).
        supports = backend.get_metadata().supports
        if bool(supports.get("score_full_vocab_paired_with_noise", False)) and adapter_id is not None:
            base_p, adapted_p, base2_p = backend.score_full_vocab_paired_with_noise(
                input_ids_batch, adapter=str(adapter_id)
            )
            if (
                bool(torch.isfinite(base_p).all().item())
                and bool(torch.isfinite(adapted_p).all().item())
                and bool(torch.isfinite(base2_p).all().item())
            ):
                kl_p = _kl_base_vs_other(base_logprobs=base_p, other_logprobs=adapted_p)
                max_diff_p = float((adapted_p - base_p).abs().max().item())
                kl_noise = _kl_base_vs_other(base_logprobs=base_p, other_logprobs=base2_p)
                max_diff_noise = float((base2_p - base_p).abs().max().item())
                out.update(
                    {
                        "kl_paired": float(kl_p),
                        "max_diff_paired": float(max_diff_p),
                        "kl_noise_within_call": float(kl_noise),
                        "max_diff_noise_within_call": float(max_diff_noise),
                    }
                )
        elif bool(supports.get("score_full_vocab_paired", False)) and adapter_id is not None:
            base_p, adapted_p = backend.score_full_vocab_paired(
                input_ids_batch, adapter=str(adapter_id)
            )
            if bool(torch.isfinite(base_p).all().item()) and bool(torch.isfinite(adapted_p).all().item()):
                kl_p = _kl_base_vs_other(base_logprobs=base_p, other_logprobs=adapted_p)
                max_diff_p = float((adapted_p - base_p).abs().max().item())
                out.update({"kl_paired": float(kl_p), "max_diff_paired": float(max_diff_p)})
        return out
    finally:
        # Always attempt unload; don't mask the real failure if unload fails.
        try:
            backend.unload_adapter(name=adapter_name)
        except Exception as e:
            # Report in-band, but keep the suite moving.
            print(f"[warn] unload_adapter failed for {adapter_name!r}: {e}")


def _parse_float_list_csv(s: str) -> list[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise RuntimeError("--suite-weights must be a non-empty comma-separated list.")
    out: list[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception as e:
            raise RuntimeError(f"Invalid float in --suite-weights: {p!r} ({e})") from e
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="SGLang offline KL sanity checks (spawn-safe).")
    ap.add_argument(
        "--config-toml",
        default=None,
        help="Optional Heretic config TOML path. If provided, reads `model`, `trust_remote_code`, and `[sglang_offline_args]`.",
    )
    ap.add_argument("--model", default="moonshotai/Kimi-K2.5", help="HF model id (or local path). Ignored if --config-toml sets model.")
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for model loading. Ignored if --config-toml sets trust_remote_code.",
    )
    ap.add_argument("--tp-size", type=int, default=8, help="SGLang tp_size (overrides TOML if set).")
    ap.add_argument("--attention-backend", default="flashinfer", help="SGLang attention_backend (overrides TOML if set).")
    ap.add_argument(
        "--engine-args-json",
        default=None,
        help="JSON object of additional SGLang Engine args (merged last, overrides TOML + flags).",
    )
    ap.add_argument("--rank", type=int, default=64, help="LoRA rank for FULL rownorm sanity adapter.")
    ap.add_argument(
        "--adapter-mode",
        choices=["full_rownorm_zero", "explicit_zero"],
        default="full_rownorm_zero",
        help="How to construct the 'zero' adapter for the sanity check.",
    )
    ap.add_argument(
        "--adapter-weight",
        type=float,
        default=0.0,
        help=(
            "Adapter weight for full_rownorm_zero mode (passed to heretic_build_full_rownorm_lora). "
            "Ignored for explicit_zero. Use small values like 1e-4 to test numerical stability."
        ),
    )
    ap.add_argument(
        "--suite",
        action="store_true",
        help=(
            "Run a small suite of adapter sanity checks in one engine instance "
            "(explicit_zero + full_rownorm for multiple weights + post-unload base check)."
        ),
    )
    ap.add_argument(
        "--suite-weights",
        default="0.0,1e-4",
        help="Comma-separated weights for FULL rownorm cases when --suite is set.",
    )
    ap.add_argument("--repeat", type=int, default=2, help="Number of base scoring repeats for determinism check (>=2).")
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=["Hello."],
        help="User prompt texts. Will be wrapped with a minimal system prompt.",
    )
    ap.add_argument("--system", default="You are a helpful assistant.")
    args = ap.parse_args()

    toml_model: str | None = None
    toml_trust: bool | None = None
    engine_args: dict[str, Any] = {}
    if args.config_toml:
        toml_model, toml_trust, engine_args = _load_engine_args_from_toml(str(args.config_toml))

    model = toml_model or str(args.model)
    trust_remote_code = toml_trust if toml_trust is not None else bool(args.trust_remote_code)

    # Merge/override with explicit flags.
    engine_args = dict(engine_args)
    engine_args["tp_size"] = int(args.tp_size)
    engine_args["attention_backend"] = str(args.attention_backend)
    if args.engine_args_json:
        engine_args.update(_parse_engine_args_json(str(args.engine_args_json)))

    resolved = resolve_model_dir(model)

    backend = SGLangOfflineBackend(
        model_path=resolved.resolved_dir,
        trust_remote_code=trust_remote_code,
        engine_args=engine_args,
    )

    # Build input_ids using the backend's own chat tokenization (keeps it close to Heretic behavior).
    chats = [
        [
            {"role": "system", "content": str(args.system)},
            {"role": "user", "content": str(p)},
        ]
        for p in list(args.prompts)
    ]
    toks = backend.tokenize_chat(chats)
    # Heretic backend contract: TokenizeChatResult.token_ids
    input_ids_batch = getattr(toks, "token_ids", None)
    if input_ids_batch is None:
        # Defensive: if the return type changes, print something actionable.
        raise RuntimeError(
            "tokenize_chat() returned an unexpected object without `.token_ids`. "
            f"type={type(toks).__name__} attrs_sample={sorted([a for a in dir(toks) if not a.startswith('_')])[:30]}"
        )
    if not isinstance(input_ids_batch, list) or not input_ids_batch:
        raise RuntimeError("tokenize_chat() returned empty input_ids_batch.")

    print("\n== KL(base || base) determinism check ==")
    repeat = int(args.repeat)
    if repeat < 2:
        raise RuntimeError("--repeat must be >= 2")
    lps: list[torch.Tensor] = []
    for _ in range(repeat):
        lps.append(backend.score(input_ids_batch, adapter=None).logprobs_full)
    lp1 = lps[0]
    if not bool(torch.isfinite(lp1).all().item()):
        raise RuntimeError("Base logprobs contain non-finite values. " + _tensor_summary("base", lp1))
    worst_kl = -1.0
    worst_maxdiff = -1.0
    for lp in lps[1:]:
        if not bool(torch.isfinite(lp).all().item()):
            raise RuntimeError("Base repeat logprobs contain non-finite values. " + _tensor_summary("base_repeat", lp))
        worst_kl = max(worst_kl, _kl_base_vs_other(base_logprobs=lp1, other_logprobs=lp))
        # max|diff| over finite entries (should be all finite here).
        worst_maxdiff = max(worst_maxdiff, float((lp - lp1).abs().max().item()))
    print(f"repeat        : {repeat}")
    # KL is theoretically >= 0; clamp tiny negative due to fp error for readability.
    if worst_kl < 0 and abs(worst_kl) < 1e-6:
        worst_kl = 0.0
    print(f"KL(base||base): {worst_kl:.8f}")
    print(f"max|diff|     : {worst_maxdiff:.8e}")
    # Within-call repeatability (single request, duplicated batch).
    supports = backend.get_metadata().supports
    if bool(supports.get("score_full_vocab_paired", False)):
        try:
            doubled = list(input_ids_batch) + list(input_ids_batch)
            lp2 = backend.score(doubled, adapter=None).logprobs_full
            if lp2 is not None and lp2.ndim == 2 and lp2.shape[0] == 2 * len(input_ids_batch):
                lp_a = lp2[: len(input_ids_batch)]
                lp_b = lp2[len(input_ids_batch) :]
                kl_within = _kl_base_vs_other(base_logprobs=lp_a, other_logprobs=lp_b)
                md_within = float((lp_b - lp_a).abs().max().item())
                if kl_within < 0 and abs(kl_within) < 1e-6:
                    kl_within = 0.0
                print(f"KL(base||base) within-call: {float(kl_within):.8f}")
                print(f"max|diff| within-call     : {float(md_within):.8e}")
        except Exception as e:
            print(f"[warn] within-call repeatability check failed: {e}")

    print("\n== KL(base || zero-adapter) correctness check ==")
    meta = backend.get_metadata()
    hidden_size = int(getattr(meta, "hidden_size", None) or 0)
    mods = backend.module_map(include_projs=["o_proj"], include_experts=[])
    desc = _select_o_proj_module_desc(mods)
    module_path = desc.get("module_path")
    if not isinstance(module_path, str):
        raise RuntimeError(f"Malformed module_map descriptor (missing module_path): {desc}")
    print(f"selected module_path: {module_path}")

    def run_one(mode: str, *, weight: float | None = None) -> None:
        if mode == "explicit_zero":
            out_f, in_f = _get_out_in_features(desc)
            tensors, config_dict = _build_explicit_zero_adapter(
                module_path_weight=module_path,
                out_features=out_f,
                in_features=in_f,
                rank=int(args.rank),
            )
            label = "explicit_zero"
        else:
            w = float(weight if weight is not None else float(args.adapter_weight))
            tensors, config_dict = _build_full_rownorm_adapter(
                backend=backend,
                module_path_weight=module_path,
                rank=int(args.rank),
                hidden_size=hidden_size,
                weight=w,
            )
            label = f"full_rownorm(weight={w:g})"

        adapter_name = f"debug_{label}".replace("(", "_").replace(")", "_").replace("=", "_").replace(",", "_")
        result = _run_adapter_case(
            backend=backend,
            adapter_name=adapter_name,
            input_ids_batch=input_ids_batch,
            base_logprobs=lp1,
            tensors=tensors,
            config_dict=config_dict,
        )
        print(f"\n-- case: {label} --")
        print(f"adapter_id    : {result.get('adapter_id')}")
        if not result.get("ok", False):
            print(result.get("base_summary"))
            print(result.get("adapted_summary"))
            raise RuntimeError("Adapted logprobs contain non-finite values; cannot compute KL reliably.")
        print(f"KL(base||case): {float(result['kl']):.8f}")
        print(f"max|diff|     : {float(result['max_diff']):.8e}")
        if "kl_paired" in result:
            print(f"KL(base||case) within-call: {float(result['kl_paired']):.8f}")
            print(f"max|diff| within-call     : {float(result['max_diff_paired']):.8e}")
        if "kl_noise_within_call" in result:
            print(f"KL_noise(base||base2) within-call: {float(result['kl_noise_within_call']):.8f}")
            print(
                f"max|diff| noise within-call     : {float(result['max_diff_noise_within_call']):.8e}"
            )

        # Post-unload base parity check: ensure returning to adapter=None doesn't drift badly.
        lp_post = backend.score(input_ids_batch, adapter=None).logprobs_full
        if not bool(torch.isfinite(lp_post).all().item()):
            print(_tensor_summary("base_post", lp_post))
            raise RuntimeError("Post-unload base logprobs contain non-finite values.")
        kl_post = _kl_base_vs_other(base_logprobs=lp1, other_logprobs=lp_post)
        if kl_post < 0 and abs(kl_post) < 1e-6:
            kl_post = 0.0
        print(f"KL(base||post_unload_base): {kl_post:.8f}")

    if args.suite:
        weights = _parse_float_list_csv(str(args.suite_weights))
        # Always run explicit zero first (isolates LoRA-path parity).
        run_one("explicit_zero")
        # Then run FULL rownorm for each requested weight.
        for w in weights:
            run_one("full_rownorm", weight=float(w))
    else:
        # Single-case behavior.
        run_one(str(args.adapter_mode), weight=float(args.adapter_weight))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

