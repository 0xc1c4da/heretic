"""Helpers to persist packed-MoE LoRA factors into PEFT adapters.

Heretic's SGLang backend can apply routed-expert down-projection updates via a packed
`w2_weight` tensor (3D [E_local, out, in]) plus per-expert low-rank factors A/B.

For Option B (offline merge into the original HuggingFace checkpoint), we need to
materialize those packed factors into standard PEFT LoRA keys that target per-expert
modules like:
  ...layers.L.mlp.experts.<expert_id>.down_proj
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def register_packed_w2_full_builds(
    *,
    backend: Any,
    adapter_id: str,
    builds: list[dict[str, Any]] | None,
) -> None:
    """Register packed-MoE w2 FULL row-norm factors under a loaded adapter id.

    This is the *authoritative* consumer of `bundle.packed_w2_full_builds`.
    Callers must not substitute settings/hardcoded defaults when `builds` provides them.
    """
    if backend is None:
        raise ValueError("backend is required")
    if not isinstance(adapter_id, str) or not adapter_id:
        raise ValueError("adapter_id must be a non-empty string")
    if not builds:
        return
    if not isinstance(builds, list):
        raise ValueError("builds must be a list when provided")

    for it in builds:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "")
        if not name:
            continue

        v = it.get("v")
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"packed_w2_full_builds[{name!r}].v must be a torch.Tensor")

        svd_q_raw = it.get("svd_q", None)
        max_experts_raw = it.get("max_experts", None)

        backend.build_packed_w2_full_rownorm(
            lora_id=str(adapter_id),
            name=name,
            v=v,
            weight=float(it.get("weight")),
            rank=int(it.get("rank")),
            svd_q=(int(svd_q_raw) if svd_q_raw is not None else None),
            svd_niter=int(it.get("svd_niter", 6)),
            build_device=str(it.get("build_device", "auto")),
            expert_chunk_size=int(it.get("expert_chunk_size", 8)),
            max_experts=(int(max_experts_raw) if max_experts_raw is not None else None),
            max_identity_k=int(it.get("max_identity_k", 2048)),
            out_dtype=str(it.get("out_dtype") or "float16"),
        )


def reconstruct_moe_tp_blockdiag_factors(
    *,
    A_shards: List[torch.Tensor],
    B_shards: List[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct full-width expert factors from MoE-TP shards exactly.

    Each MoE-TP rank k holds factors:
      A_k: [E, r, in_k]
      B_k: [E, out, r]

    The full delta is the horizontal concatenation:
      ΔW = [B_0@A_0 | ... | B_{tp-1}@A_{tp-1}]

    We encode this as a single LoRA with rank r*tp:
      B_full = cat(B_k, dim=rank)                 -> [E, out, r*tp]
      A_full = block_diag(A_0..A_{tp-1})          -> [E, r*tp, in_full]
    """
    if not A_shards or not B_shards:
        raise ValueError("A_shards and B_shards must be non-empty.")
    if len(A_shards) != len(B_shards):
        raise ValueError(
            f"Shard list length mismatch: {len(A_shards)} != {len(B_shards)}"
        )

    tp = len(A_shards)
    A0 = A_shards[0]
    B0 = B_shards[0]
    if not isinstance(A0, torch.Tensor) or not isinstance(B0, torch.Tensor):
        raise ValueError("Shards must be torch.Tensors.")
    if A0.ndim != 3 or B0.ndim != 3:
        raise ValueError(
            f"Expected 3D shards; got A.ndim={A0.ndim} B.ndim={B0.ndim}"
        )
    E, r, _ = (int(A0.shape[0]), int(A0.shape[1]), int(A0.shape[2]))
    if int(B0.shape[0]) != E:
        raise ValueError("A/B expert dim mismatch in shard 0.")
    out = int(B0.shape[1])
    if int(B0.shape[2]) != r:
        raise ValueError("A/B rank mismatch in shard 0.")

    in_sizes: list[int] = []
    for i, (A, B) in enumerate(zip(A_shards, B_shards)):
        if A.ndim != 3 or B.ndim != 3:
            raise ValueError(f"Shard {i} has wrong ndim: A.ndim={A.ndim} B.ndim={B.ndim}")
        if int(A.shape[0]) != E or int(B.shape[0]) != E:
            raise ValueError(f"Shard {i} has wrong expert dim.")
        if int(A.shape[1]) != r or int(B.shape[2]) != r:
            raise ValueError(f"Shard {i} has wrong rank.")
        if int(B.shape[1]) != out:
            raise ValueError(f"Shard {i} has wrong out dim.")
        in_sizes.append(int(A.shape[2]))

    in_full = int(sum(in_sizes))
    r_full = int(r * tp)

    B_full = torch.cat(B_shards, dim=2).contiguous()  # [E, out, r_full]
    A_full = torch.zeros(
        (E, r_full, in_full),
        dtype=A0.dtype,
        device=A0.device,
    )
    col = 0
    for k, A_k in enumerate(A_shards):
        in_k = int(A_k.shape[2])
        A_full[:, k * r : (k + 1) * r, col : col + in_k] = A_k
        col += in_k
    return A_full.contiguous(), B_full


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


def inject_exported_packed_w2_factors_into_bundle(
    *,
    bundle: Any,
    backend: Any,
    adapter_id: str,
    expert_down_proj_leaf: str = "down_proj",
) -> None:
    """Export packed-MoE w2 factors from backend and inject into a PEFT bundle.

    This mutates `bundle.tensors` and `bundle.config_dict` in-place.

    - Adds per-expert PEFT keys via `materialize_packed_w2_factors_to_peft_tensors`.
    - If the exported expert rank differs from the bundle default `r`, adds a
      `rank_pattern`/`alpha_pattern` entry for routed experts so PEFT can load/merge.
    """
    if bundle is None:
        raise ValueError("bundle is required")
    if backend is None:
        raise ValueError("backend is required")
    if not isinstance(adapter_id, str) or not adapter_id:
        raise ValueError("adapter_id must be a non-empty string")

    builds = getattr(bundle, "packed_w2_full_builds", None)
    if not builds:
        return

    cfg = getattr(bundle, "config_dict", None)
    if not isinstance(cfg, dict):
        raise ValueError("bundle.config_dict must be a dict")

    default_r = cfg.get("r", None)
    default_alpha = cfg.get("lora_alpha", None)
    try:
        default_r_i = int(default_r)
    except Exception:
        default_r_i = -1

    expert_rank_seen: int | None = None

    for it in builds or []:
        packed_name = str(it.get("name") or "")
        if not packed_name:
            continue

        expert_ids, A_stack, B_stack = backend.export_packed_w2_factors(
            lora_id=str(adapter_id),
            name=packed_name,
        )
        if not isinstance(A_stack, torch.Tensor) or A_stack.ndim != 3:
            raise RuntimeError(
                f"export_packed_w2_factors returned invalid A_stack for {packed_name}: "
                f"{type(A_stack).__name__} ndim={getattr(A_stack, 'ndim', None)}"
            )
        if expert_rank_seen is None:
            expert_rank_seen = int(A_stack.shape[1])
        else:
            expert_rank_seen = max(expert_rank_seen, int(A_stack.shape[1]))

        peft_tensors = materialize_packed_w2_factors_to_peft_tensors(
            packed_w2_param_name=packed_name,
            expert_ids=list(expert_ids),
            A_stack=A_stack,
            B_stack=B_stack,
            expert_down_proj_leaf=expert_down_proj_leaf,
        )
        bundle.tensors.update(peft_tensors)

    # If experts have a larger rank (e.g. r*moe_tp_size), ensure PEFT can load them.
    if expert_rank_seen is not None and default_r_i > 0 and int(expert_rank_seen) != int(default_r_i):
        # Match routed expert down-proj modules while avoiding shared_experts.
        # PEFT's rank_pattern keys are matched against module names; using a narrow substring
        # is the most robust across model wrappers/prefixes.
        key = ".mlp.experts."

        rank_pattern = cfg.get("rank_pattern")
        if not isinstance(rank_pattern, dict):
            rank_pattern = {}
        rank_pattern.setdefault(key, int(expert_rank_seen))
        cfg["rank_pattern"] = rank_pattern

        if isinstance(default_alpha, (int, float)) and default_r_i > 0:
            # Preserve alpha/r scaling by scaling alpha linearly with rank.
            alpha_expert = float(default_alpha) * (float(expert_rank_seen) / float(default_r_i))
            alpha_expert_out: int | float
            alpha_expert_out = int(round(alpha_expert)) if isinstance(default_alpha, int) else alpha_expert

            alpha_pattern = cfg.get("alpha_pattern")
            if not isinstance(alpha_pattern, dict):
                alpha_pattern = {}
            alpha_pattern.setdefault(key, alpha_expert_out)
            cfg["alpha_pattern"] = alpha_pattern

