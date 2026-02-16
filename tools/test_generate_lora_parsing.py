#!/usr/bin/env python3
from __future__ import annotations

# Lightweight parser/path tests for tools/generate_lora.py.

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_generate_lora_module():
    script_path = Path(__file__).resolve().parents[0] / "generate_lora.py"
    spec = importlib.util.spec_from_file_location("generate_lora", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = _load_generate_lora_module()

    # --param dotted parsing.
    direction_index, parameters = mod._parse_cli_params(
        [
            "attn.o_proj.max_weight=2.58",
            "attn.o_proj.max_weight_position=40.80",
            "attn.o_proj.min_weight=2.14",
            "attn.o_proj.min_weight_distance=26.79",
        ],
        direction_index=26.74,
    )
    assert abs(direction_index - 26.74) < 1e-9
    assert "attn.o_proj" in parameters
    assert abs(parameters["attn.o_proj"].max_weight - 2.58) < 1e-9

    # JSON payload parsing with per-layer mode.
    payload = {
        "direction_index": None,
        "parameters": {
            "mlp.down_proj": {
                "max_weight": 2.18,
                "max_weight_position": 52.83,
                "min_weight": 1.97,
                "min_weight_distance": 29.08,
            }
        },
    }
    direction_index2, parameters2 = mod._parse_params_json_payload(payload)
    assert direction_index2 is None
    assert abs(parameters2["mlp.down_proj"].min_weight_distance - 29.08) < 1e-9

    # Study path sanitization parity.
    fake_settings = SimpleNamespace(
        study_checkpoint_dir="checkpoints",
        model="moonshotai/Kimi-K2.5",
    )
    derived = mod._derive_study_jsonl_path(fake_settings)
    assert derived == "checkpoints/moonshotai--Kimi-K2--5.jsonl"

    # Trial lookup behavior (found and not-found list).
    t0 = SimpleNamespace(
        user_attrs={
            "index": 41,
            "direction_index": 24.0,
            "parameters": {
                "attn.o_proj": {
                    "max_weight": 2.0,
                    "max_weight_position": 30.0,
                    "min_weight": 1.5,
                    "min_weight_distance": 20.0,
                }
            },
        }
    )
    t1 = SimpleNamespace(
        user_attrs={
            "index": 42,
            "direction_index": None,
            "parameters": {
                "mlp.down_proj": {
                    "max_weight": 2.2,
                    "max_weight_position": 40.0,
                    "min_weight": 1.7,
                    "min_weight_distance": 25.0,
                }
            },
        }
    )
    d3, p3 = mod._extract_trial_from_trials([t0, t1], trial_index=42)
    assert d3 is None
    assert "mlp.down_proj" in p3

    try:
        mod._extract_trial_from_trials([t0, t1], trial_index=999)
        raise AssertionError("Expected missing-trial error.")
    except ValueError as exc:
        msg = str(exc)
        assert "Available: [41, 42]" in msg

    print("[ok] generate_lora parser/path tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
