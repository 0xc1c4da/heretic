"""CPU-only regression test for SGLang heretic_module_map parsing helpers.

This intentionally does NOT load any model or require GPUs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    # Import the dependency-light helper directly by file path.
    repo_root = Path(__file__).resolve().parents[1]
    helper_path = repo_root / "vendor" / "sglang" / "python" / "sglang" / "srt" / "heretic_utils.py"
    spec = importlib.util.spec_from_file_location("sglang_srt_heretic_utils", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {helper_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    parse = getattr(mod, "heretic_parse_layer_expert", None)
    if parse is None:
        raise RuntimeError("Missing heretic_parse_layer_expert in heretic_utils.")

    cases: list[tuple[str, tuple[int | None, int | None]]] = [
        (
            "language_model.model.layers.12.self_attn.o_proj.weight",
            (12, None),
        ),
        (
            "language_model.model.layers.12.mlp.shared_experts.down_proj.weight",
            (12, None),
        ),
        (
            "language_model.model.layers.12.mlp.experts.7.down_proj.weight",
            (12, 7),
        ),
        (
            "model.layers.0.mlp.experts.0.gate_up_proj.weight",
            (0, 0),
        ),
        (
            "something.without.layers.or.experts.weight",
            (None, None),
        ),
    ]

    for name, expected in cases:
        got = parse(name)
        if got != expected:
            raise AssertionError(f"{name}: expected {expected}, got {got}")

    print("ok: module_map parsing")


if __name__ == "__main__":
    main()

