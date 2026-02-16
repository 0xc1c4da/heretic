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

    recon = getattr(mod, "reconstruct_moe_tp_blockdiag_factors", None)
    if recon is None:
        raise RuntimeError("Missing reconstruct_moe_tp_blockdiag_factors.")

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

    # TP reconstruction regression: ensure we can exactly reconstruct full-width factors
    # from TP-local shards without producing TP-local-width "A" factors.
    E = 2
    tp = 4
    r = 3
    in_local = 5
    out_f = 7
    gen = torch.Generator().manual_seed(1234)
    A_shards = [
        torch.randn((E, r, in_local), dtype=torch.float32, generator=gen) for _ in range(tp)
    ]
    B_shards = [
        torch.randn((E, out_f, r), dtype=torch.float32, generator=gen) for _ in range(tp)
    ]

    A_full, B_full = recon(A_shards=A_shards, B_shards=B_shards)
    if tuple(A_full.shape) != (E, r * tp, in_local * tp):
        raise AssertionError(f"A_full shape mismatch: {tuple(A_full.shape)}")
    if tuple(B_full.shape) != (E, out_f, r * tp):
        raise AssertionError(f"B_full shape mismatch: {tuple(B_full.shape)}")

    # Exactness check against explicit horizontal concatenation of deltas.
    for e in range(E):
        delta_expected = torch.cat(
            [B_shards[k][e] @ A_shards[k][e] for k in range(tp)], dim=1
        )
        delta_got = B_full[e] @ A_full[e]
        if not torch.allclose(delta_got, delta_expected, atol=1e-5, rtol=1e-5):
            raise AssertionError("TP block-diag reconstruction mismatch.")

    print("ok: packed-w2 TP block-diag reconstruction")


if __name__ == "__main__":
    main()

