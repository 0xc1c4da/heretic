#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Quick sanity check for ΔNLL damage metric on SGLang offline backend.

What it tests:
- Generates a short base continuation for a harmless prompt.
- Computes paired (within-call) continuation NLLs with a zero-effect adapter and reports:
  - damage = (adapted_nll - base1_nll)
  - noise  = |base2_nll - base1_nll|

This is intended to be the replacement "trustability" check when full-vocab KL is unstable.
"""

from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser(description="SGLang-offline ΔNLL sanity check")
    ap.add_argument("--config-toml", default=None)
    ap.add_argument("--tokens", type=int, default=32, help="Continuation tokens to cache")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = str(args.config_toml) if args.config_toml else "config.toml"
    if not os.path.exists(config_path):
        raise SystemExit(f"config not found: {config_path!r}")

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
    settings.damage_metric = "paired_delta_nll"
    settings.delta_nll_continuation_tokens = int(args.tokens)
    settings.delta_nll_num_refs = 1
    settings.delta_nll_ref_temperatures = [0.0]
    settings.damage_noise_threshold = 0.05
    settings.damage_retry_count = 0

    print("[delta_nll] initializing Model...", flush=True)
    t0 = time.perf_counter()
    model = Model(settings)
    print(f"[delta_nll] Model ready in {time.perf_counter()-t0:.2f}s", flush=True)

    system = str(getattr(settings, "system_prompt", None) or "You are a helpful assistant.")
    prompts = [Prompt(system=system, user="Summarize the benefits of exercise.")]

    prompt_ids = model.encode_prompts(prompts)
    backend = model.backend
    cont = backend.generate_token_ids(  # type: ignore[attr-defined]
        prompt_ids, max_new_tokens=int(args.tokens), adapter=None, temperature=0.0, top_k=1
    )
    print(f"[delta_nll] cached continuation len={len(cont[0])}", flush=True)

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
        name="delta_nll_explicit_zero_rank1",
        tensors=tensors0,
        config=cfg0,
    )

    def _score_paired():
        b1, ad, b2 = backend.score_continuation_nll_paired_with_noise(  # type: ignore[attr-defined]
            prompt_ids_batch=prompt_ids,
            continuation_ids_batch=cont,
            adapter=str(adapter_id),
        )
        base1 = float(b1[0])
        adapt = float(ad[0])
        base2 = float(b2[0])
        delta = float(adapt - base1)
        noise = float(abs(base2 - base1))
        return base1, adapt, base2, delta, noise

    a = _score_paired()
    b = _score_paired()
    print(f"[delta_nll] paired #1 base1/adapt/base2: {a[0]:.6g} / {a[1]:.6g} / {a[2]:.6g}")
    print(f"[delta_nll] paired #1 damage/noise      : {a[3]:.6g} / {a[4]:.6g}")
    print(f"[delta_nll] paired #2 base1/adapt/base2: {b[0]:.6g} / {b[1]:.6g} / {b[2]:.6g}")
    print(f"[delta_nll] paired #2 damage/noise      : {b[3]:.6g} / {b[4]:.6g}")
    print(f"[delta_nll] |Δdamage|                 : {abs(a[3]-b[3]):.6g}")
    print(f"[delta_nll] |Δnoise|                  : {abs(a[4]-b[4]):.6g}")
    backend.unload_adapter(name="delta_nll_explicit_zero_rank1")  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

