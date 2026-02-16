"""Helpers to persist packed-MoE LoRA factors into PEFT adapters.

Heretic's SGLang backend can apply routed-expert down-projection updates via a packed
`w2_weight` tensor (3D [E_local, out, in]) plus per-expert low-rank factors A/B.

For Option B (offline merge into the original HuggingFace checkpoint), we need to
materialize those packed factors into standard PEFT LoRA keys that target per-expert
modules like:
  ...layers.L.mlp.experts.<expert_id>.down_proj
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def materialize_packed_w2_factors_to_peft_tensors(
    *,
    packed_w2_param_name: str,
    expert_ids: List[int],
    A_stack: torch.Tensor,
    B_stack: torch.Tensor,
    expert_down_proj_leaf: str = "down_proj",
) -> Dict[str, torch.Tensor]:
    """Convert packed [E, r, in]/[E, out, r] factors to PEFT-style per-expert keys.

    Args:
        packed_w2_param_name: Full parameter name of packed tensor (e.g.
            "...mlp.experts.w2_weight" / "...mlp.experts.w2_qweight_packed").
            We derive the experts module prefix by stripping the final segment.
        expert_ids: Global expert ids corresponding to dim0 of A_stack/B_stack.
        A_stack: [E, r, in_features] fp16/bf16/float32 tensor (CPU preferred).
        B_stack: [E, out_features, r] fp16/bf16/float32 tensor (CPU preferred).
        expert_down_proj_leaf: Leaf module name in the base model (default: "down_proj").

    Returns:
        Dict of PEFT tensor keys -> 2D tensors:
          "<experts_prefix>.<expert_id>.<down_proj>.lora_A.default.weight": [r, in]
          "<experts_prefix>.<expert_id>.<down_proj>.lora_B.default.weight": [out, r]
    """
    if not isinstance(packed_w2_param_name, str) or not packed_w2_param_name:
        raise ValueError("packed_w2_param_name must be a non-empty string.")
    if not isinstance(expert_down_proj_leaf, str) or not expert_down_proj_leaf:
        raise ValueError("expert_down_proj_leaf must be a non-empty string.")

    if not isinstance(expert_ids, list) or not all(isinstance(x, int) for x in expert_ids):
        raise ValueError("expert_ids must be a list[int].")

    if not isinstance(A_stack, torch.Tensor) or not isinstance(B_stack, torch.Tensor):
        raise ValueError("A_stack and B_stack must be torch.Tensors.")
    if A_stack.ndim != 3 or B_stack.ndim != 3:
        raise ValueError(
            f"Expected A_stack/B_stack to be 3D; got A.ndim={A_stack.ndim} B.ndim={B_stack.ndim}"
        )

    E = int(A_stack.shape[0])
    if int(B_stack.shape[0]) != E:
        raise ValueError(
            f"Expert dim mismatch: A_stack.shape[0]={E} B_stack.shape[0]={int(B_stack.shape[0])}"
        )
    if len(expert_ids) != E:
        raise ValueError(f"expert_ids length mismatch: {len(expert_ids)} != {E}")

    # Derive the experts module prefix from the packed parameter name:
    # "...mlp.experts.w2_weight" -> "...mlp.experts"
    experts_prefix = packed_w2_param_name.rsplit(".", 1)[0]

    out: Dict[str, torch.Tensor] = {}
    for gid, A_e, B_e in zip(expert_ids, A_stack, B_stack):
        base = f"{experts_prefix}.{int(gid)}.{expert_down_proj_leaf}"
        out[f"{base}.lora_A.default.weight"] = A_e.contiguous()
        out[f"{base}.lora_B.default.weight"] = B_e.contiguous()
    return out

