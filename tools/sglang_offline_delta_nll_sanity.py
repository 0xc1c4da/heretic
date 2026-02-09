#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Quick sanity check for ΔNLL damage metric on SGLang offline backend.

What it tests:
- Generates a short base continuation for a harmless prompt.
- Computes teacher-forced NLL on that continuation twice (same adapter) and reports drift.

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
    settings.damage_metric = "delta_nll"
    settings.delta_nll_continuation_tokens = int(args.tokens)

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

    def _score():
        return backend.score_sequence_delta_nll(  # type: ignore[attr-defined]
            prompt_ids_batch=prompt_ids,
            continuation_ids_batch=cont,
            adapter=None,
        )[0]

    a = float(_score())
    b = float(_score())
    print(f"[delta_nll] base mean NLL #1: {a:.6g}")
    print(f"[delta_nll] base mean NLL #2: {b:.6g}")
    print(f"[delta_nll] |diff|        : {abs(a-b):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

