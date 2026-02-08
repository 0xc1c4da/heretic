#!/usr/bin/env python3
"""Static (no-deps) smoke test for long-run robustness hooks.

This test intentionally does NOT import Heretic modules (torch may be unavailable).
It verifies that key guardrails remain present in source:
- Optuna optimize uses catch= to prevent single-trial crashes
- Evaluator validates finite logprobs before KL
"""

from __future__ import annotations

from pathlib import Path


def must_contain(path: Path, needles: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [n for n in needles if n not in text]
    if missing:
        raise AssertionError(f"{path}: missing expected substrings: {missing}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    must_contain(
        root / "src" / "heretic" / "main.py",
        [
            "catch=(Exception,)",
            "trial.set_user_attr(\"error_type\"",
            "trial.set_user_attr(\"error\"",
        ],
    )
    must_contain(
        root / "src" / "heretic" / "evaluator.py",
        [
            "torch.isfinite",
            "NonFiniteLogprobsError",
            "_validate_logprobs_tensor",
        ],
    )
    must_contain(
        root / "src" / "heretic" / "model.py",
        [
            "def num_layers",
        ],
    )
    print("ok: optuna robustness static checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

