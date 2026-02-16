"""CPU-only regression test for packed-MoE factor materialization into PEFT keys.

This intentionally does NOT load any model or require GPUs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "src" / "heretic" / "peft_packed_moe.py"
    spec = importlib.util.spec_from_file_location("heretic_peft_packed_moe", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fn = getattr(mod, "materialize_packed_w2_factors_to_peft_tensors", None)
    if fn is None:
        raise RuntimeError("Missing materialize_packed_w2_factors_to_peft_tensors.")

    packed = "language_model.model.layers.17.mlp.experts.w2_weight"
    expert_ids = [382, 383]
    r = 3
    in_f = 2048
    out_f = 7168
    A = torch.zeros((len(expert_ids), r, in_f), dtype=torch.float16)
    B = torch.zeros((len(expert_ids), out_f, r), dtype=torch.float16)

    out = fn(
        packed_w2_param_name=packed,
        expert_ids=expert_ids,
        A_stack=A,
        B_stack=B,
        expert_down_proj_leaf="down_proj",
    )

    # Two experts -> 4 tensors (A/B per expert).
    if len(out) != 4:
        raise AssertionError(f"Expected 4 tensors, got {len(out)} keys={list(out)[:5]}")

    for gid in expert_ids:
        base = f"language_model.model.layers.17.mlp.experts.{gid}.down_proj"
        kA = f"{base}.lora_A.default.weight"
        kB = f"{base}.lora_B.default.weight"
        if kA not in out or kB not in out:
            raise AssertionError(f"Missing keys for expert {gid}: {kA in out=} {kB in out=}")
        if tuple(out[kA].shape) != (r, in_f):
            raise AssertionError(f"{kA} shape mismatch: {tuple(out[kA].shape)}")
        if tuple(out[kB].shape) != (out_f, r):
            raise AssertionError(f"{kB} shape mismatch: {tuple(out[kB].shape)}")

    print("ok: packed-w2 PEFT materialization")


if __name__ == "__main__":
    main()

