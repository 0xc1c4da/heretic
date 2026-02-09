#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Quick sanity check for top-k Jensen–Shannon damage metric on SGLang offline backend.

What it tests:
- Generates a short base continuation for a harmless prompt.
- Uses paired-within-call top-k logprobs for (base1, adapted, base2) on that continuation.
- Computes:
  - damage = mean_t JS(base1_t || adapted_t) over first N continuation positions
  - noise  = mean_t JS(base1_t || base2_t) over same positions
- Repeats the paired measurement twice to check repeatability.

This is intended to validate that "topk_js" is stable enough to optimize on, even when
full-vocab next-token KL is unreliable.
"""

from __future__ import annotations

import argparse
import math
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
    ap = argparse.ArgumentParser(description="SGLang-offline top-k JS sanity check")
    ap.add_argument("--config-toml", default=None)
    ap.add_argument("--positions", type=int, default=32, help="Continuation positions to score")
    ap.add_argument("--k", type=int, default=128, help="Top-k to request at each position")
    return ap.parse_args()


def _js_other_bucket(p_log: dict[int, float], q_log: dict[int, float]) -> float:
    keys = set(p_log.keys()) | set(q_log.keys())

    def _collect(d: dict[int, float]) -> tuple[dict[int, float], float]:
        probs: dict[int, float] = {}
        s = 0.0
        for k in keys:
            lp = d.get(k)
            if lp is None:
                continue
            pk = math.exp(float(lp))
            if pk <= 0.0:
                continue
            probs[int(k)] = pk
            s += pk
        other = max(0.0, 1.0 - s)
        return probs, other

    p_probs, p_other = _collect(p_log)
    q_probs, q_other = _collect(q_log)

    def _kl(a: dict[int, float], a_other: float, b: dict[int, float], b_other: float) -> float:
        out = 0.0
        for k, ap in a.items():
            if ap <= 0.0:
                continue
            bp = b.get(k, 0.0)
            if bp <= 0.0:
                return float("inf")
            out += ap * (math.log(ap) - math.log(bp))
        if a_other > 0.0:
            if b_other <= 0.0:
                return float("inf")
            out += a_other * (math.log(a_other) - math.log(b_other))
        return out

    m: dict[int, float] = {}
    for k in keys:
        m[k] = 0.5 * p_probs.get(k, 0.0) + 0.5 * q_probs.get(k, 0.0)
    m_other = 0.5 * p_other + 0.5 * q_other

    kl_pm = _kl(p_probs, p_other, m, m_other)
    kl_qm = _kl(q_probs, q_other, m, m_other)
    if not math.isfinite(kl_pm) or not math.isfinite(kl_qm):
        return float("inf")
    return 0.5 * kl_pm + 0.5 * kl_qm


def main() -> int:
    args = _parse_args()
    config_path = str(args.config_toml) if args.config_toml else "config.toml"
    if not os.path.exists(config_path):
        raise SystemExit(f"config not found: {config_path!r}")

    if int(args.positions) <= 0:
        raise SystemExit("--positions must be > 0")
    if int(args.k) <= 0:
        raise SystemExit("--k must be > 0")

    # Strip argv so Settings CLI parsing won't see our flags.
    sys.argv = [sys.argv[0]]

    import torch

    from heretic.config import BackendType, Settings
    from heretic.model import Model
    from heretic.utils import Prompt

    cfg = _load_toml(config_path)
    settings = Settings.model_validate(cfg)
    settings.backend = BackendType.SGLANG_OFFLINE
    settings.validate_backend = False
    settings.batch_size = 1
    settings.max_batch_size = 1
    settings.damage_metric = "topk_js"
    settings.topk_js_k = int(args.k)
    settings.topk_js_positions = int(args.positions)
    # Keep continuation cache deterministic for the sanity tool.
    settings.delta_nll_num_refs = 1
    settings.delta_nll_ref_temperatures = [0.0]
    settings.delta_nll_continuation_tokens = int(args.positions)

    print("[topk_js] initializing Model...", flush=True)
    t0 = time.perf_counter()
    model = Model(settings)
    print(f"[topk_js] Model ready in {time.perf_counter()-t0:.2f}s", flush=True)

    system = str(getattr(settings, "system_prompt", None) or "You are a helpful assistant.")
    prompts = [Prompt(system=system, user="Summarize the benefits of exercise.")]

    prompt_ids = model.encode_prompts(prompts)
    backend = model.backend
    cont = backend.generate_token_ids(  # type: ignore[attr-defined]
        prompt_ids, max_new_tokens=int(args.positions), adapter=None, temperature=0.0, top_k=1
    )
    print(f"[topk_js] cached continuation len={len(cont[0])}", flush=True)

    # Build an explicit no-op adapter (rank=1, A=B=0) to exercise the adapter path.
    module_descs = backend.module_map(  # type: ignore[attr-defined]
        include_projs=["o_proj", "down_proj"],
        include_experts=[],
    )
    picked = None
    for d in module_descs:
        if not isinstance(d, dict):
            continue
        path = d.get("module_path") or d.get("name")
        shape = d.get("shape")
        if not isinstance(path, str) or not isinstance(shape, list) or len(shape) != 2:
            continue
        if not all(isinstance(x, int) for x in shape):
            continue
        if not path.endswith(".weight"):
            continue
        out_f, in_f = int(shape[0]), int(shape[1])
        if out_f > 0 and in_f > 0:
            picked = (path, out_f, in_f)
            break
    if picked is None:
        raise RuntimeError("Could not pick a linear weight with (out_f,in_f) shape from backend.module_map().")
    path, out_f, in_f = picked
    module_base = path[: -len(".weight")]
    rank = 1
    A0 = torch.zeros((rank, in_f), dtype=torch.float16)
    B0 = torch.zeros((out_f, rank), dtype=torch.float16)
    tensors0 = {
        f"{module_base}.lora_A.default.weight": A0.cpu(),
        f"{module_base}.lora_B.default.weight": B0.cpu(),
    }
    cfg0 = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": int(rank),
        "lora_alpha": int(rank),
        "lora_dropout": 0.0,
        "target_modules": ["o_proj", "down_proj"],
        "bias": "none",
    }
    adapter_id = backend.load_adapter(  # type: ignore[attr-defined]
        name="topk_js_explicit_zero_rank1",
        tensors=tensors0,
        config=cfg0,
    )

    def _score_once() -> tuple[float, float]:
        b1, ad, b2 = backend.score_continuation_topk_paired_with_noise(  # type: ignore[attr-defined]
            prompt_ids_batch=prompt_ids,
            continuation_ids_batch=cont,
            adapter=str(adapter_id),
            top_k=int(args.k),
        )
        # batch=1
        b1_pos = b1[0]
        ad_pos = ad[0]
        b2_pos = b2[0]
        npos = min(int(args.positions), len(b1_pos), len(ad_pos), len(b2_pos))
        if npos <= 0:
            return 0.0, 0.0
        dmg = 0.0
        noi = 0.0
        for t in range(npos):
            dmg += float(_js_other_bucket(b1_pos[t], ad_pos[t]))
            noi += float(_js_other_bucket(b1_pos[t], b2_pos[t]))
        return float(dmg / npos), float(noi / npos)

    a_dmg, a_noise = _score_once()
    b_dmg, b_noise = _score_once()
    print(f"[topk_js] paired #1 damage/noise: {a_dmg:.6g} / {a_noise:.6g}")
    print(f"[topk_js] paired #2 damage/noise: {b_dmg:.6g} / {b_noise:.6g}")
    print(f"[topk_js] |Δdamage|         : {abs(a_dmg-b_dmg):.6g}")
    print(f"[topk_js] |Δnoise|          : {abs(a_noise-b_noise):.6g}")

    backend.unload_adapter(name="topk_js_explicit_zero_rank1")  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

