#!/usr/bin/env python3
"""
Production quantization-preserving LoRA merger.

This tool merges a PEFT LoRA adapter into a sharded safetensors checkpoint while:
- preserving shard layout and index mapping
- preserving each weight's quantization format
- avoiding full model instantiation
"""

from __future__ import annotations

import argparse
import enum
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from safetensors.torch import safe_open
from safetensors.torch import save_file

# Make repository root importable regardless of invocation cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if _REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _REPO_ROOT.as_posix())

# Reuse battle-tested baseline helpers from the existing streaming merger.
from stream_merge_lora_safetensors import (  # type: ignore[attr-defined]
    LoraForWeight,
    _adapter_weights_path,
    _apply_lora_blockwise as _apply_lora_blockwise_base,
    _build_lora_maps,
    _compile_patterns,
    _copy_support_files,
    _detect_base_layout,
    _die,
    _dump_json,
    _is_subpath,
    _load_json,
    _link_unchanged_shard,
    _parse_adapter_spec,
    _parse_device,
    _parse_dtype,
    _prepare_out_dir,
    _resolve_alpha_for_module,
    _scaling,
)


class QuantFormat(enum.Enum):
    FLOAT = "float"
    FP8_BLOCK_INV = "fp8_block_inv"
    # FP8 block quant with UE8M0-packed int32 scale tensor (DeepGEMM-style layout).
    FP8_BLOCK_INV_UE8M0 = "fp8_block_inv_ue8m0"
    CT_FP8_BLOCK = "ct_fp8_block"
    INT4_CT = "int4_ct"
    MXFP4 = "mxfp4"


@dataclass(frozen=True)
class QuantParams:
    group_size: int
    block_size: tuple[int, int]
    num_bits: int
    mxfp4_block_size: int


def _require_quant_deps() -> None:
    """
    Fail fast with actionable messages if quant codec deps are missing.
    """
    missing: list[str] = []
    try:
        from compressed_tensors.compressors.quantized_compressors.pack_quantized import (  # noqa: F401
            pack_to_int32,
            unpack_from_int32,
        )
    except Exception:
        missing.append(
            "compressed_tensors.compressors.quantized_compressors.pack_quantized"
        )

    try:
        from sglang.srt.layers.quantization.fp8_utils import (  # noqa: F401
            block_quant_dequant,
            per_block_cast_to_fp8,
        )
        from sglang.srt.layers.quantization.mxfp4_tensor import (  # noqa: F401
            MXFP4QuantizeUtil,
        )
    except Exception:
        missing.append("sglang.srt.layers.quantization.[fp8_utils|mxfp4_tensor]")

    if missing:
        joined = ", ".join(missing)
        _die(
            "Required quantization dependencies are unavailable: "
            f"{joined}. Install heretic inference dependencies (including sglang "
            "and compressed-tensors) before running this tool."
        )


def _parse_block_size(s: Optional[str]) -> Optional[tuple[int, int]]:
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        _die(f"Invalid --block-size {s!r}. Expected format INT,INT (e.g. 128,128).")
    try:
        a = int(parts[0])
        b = int(parts[1])
    except Exception:
        _die(f"Invalid --block-size {s!r}. Expected integers.")
    if a <= 0 or b <= 0:
        _die(f"Invalid --block-size {s!r}. Block sizes must be positive.")
    return (a, b)


def _extract_quant_params(config: Mapping[str, Any]) -> QuantParams:
    group_size = 32
    block_size = (128, 128)
    num_bits = 4
    mxfp4_block_size = 32

    roots = [config]
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        roots.append(text_cfg)

    found_qc = False
    for root in roots:
        if not isinstance(root, dict):
            continue
        qc = root.get("quantization_config")
        if not isinstance(qc, dict) or not qc:
            continue
        found_qc = True

        wb = qc.get("weight_block_size")
        if isinstance(wb, list) and len(wb) == 2:
            if all(isinstance(x, int) and x > 0 for x in wb):
                block_size = (int(wb[0]), int(wb[1]))

        cgroups = qc.get("config_groups")
        if isinstance(cgroups, dict):
            for g in cgroups.values():
                if not isinstance(g, dict):
                    continue
                weights = g.get("weights")
                if not isinstance(weights, dict):
                    continue
                gs = weights.get("group_size")
                if isinstance(gs, int) and gs > 0:
                    group_size = int(gs)
                nb = weights.get("num_bits")
                if isinstance(nb, int) and nb > 0:
                    num_bits = int(nb)
            # First quantization_config wins.
            break

    if not found_qc:
        print(
            "[merge] warning: quantization_config not found in config.json; "
            "using defaults group_size=32 block_size=(128,128) num_bits=4"
        )

    return QuantParams(
        group_size=group_size,
        block_size=block_size,
        num_bits=num_bits,
        mxfp4_block_size=mxfp4_block_size,
    )


def _is_float8(dtype: torch.dtype) -> bool:
    float8_names = {
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    }
    return getattr(dtype, "__str__", lambda: "")() in {f"torch.{n}" for n in float8_names}


def _is_floating_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    } or _is_float8(dtype)


def _dtype_name(dtype: Any) -> str:
    return str(dtype)


def detect_quant_format(
    *,
    weight_key: str,
    shard_keys: set[str],
    dtype: torch.dtype,
    shard_f: Any,
) -> QuantFormat:
    base = weight_key.removesuffix(".weight").removesuffix(".weight_packed")
    scale_inv_key = f"{base}.weight_scale_inv"
    scale_key = f"{base}.weight_scale"
    shape_key = f"{base}.weight_shape"

    if _is_float8(dtype):
        if scale_inv_key in shard_keys:
            scale_slice = shard_f.get_slice(scale_inv_key)
            shape = tuple(int(x) for x in scale_slice.get_shape())
            if len(shape) != 2:
                raise ValueError(
                    f"FP8 tensor {weight_key} has invalid companion shape for "
                    f"{scale_inv_key}: {shape}"
                )
            sd = scale_slice.dtype
            if sd == torch.int32:
                return QuantFormat.FP8_BLOCK_INV_UE8M0
            if _is_floating_dtype(sd):
                return QuantFormat.FP8_BLOCK_INV
            raise ValueError(
                f"FP8 tensor {weight_key} has unsupported companion dtype for "
                f"{scale_inv_key}: {_dtype_name(sd)}"
            )
        if scale_key in shard_keys:
            scale_slice = shard_f.get_slice(scale_key)
            shape = tuple(int(x) for x in scale_slice.get_shape())
            sd = scale_slice.dtype
            if not _is_floating_dtype(sd):
                raise ValueError(
                    f"FP8 tensor {weight_key} has non-float companion dtype for "
                    f"{scale_key}: {_dtype_name(sd)}"
                )
            if len(shape) == 2 and shape[0] > 1 and shape[1] > 1:
                return QuantFormat.CT_FP8_BLOCK
            raise ValueError(
                f"FP8 tensor {weight_key} has unsupported {scale_key} shape {shape}; "
                "only block FP8 is supported."
            )
        raise ValueError(
            f"FP8 tensor {weight_key} missing companion scale tensor: "
            f"tried {scale_inv_key} and {scale_key}"
        )

    if weight_key.endswith(".weight_packed") and dtype == torch.int32:
        if scale_key in shard_keys and shape_key in shard_keys:
            return QuantFormat.INT4_CT
        raise ValueError(
            f"Packed INT4 tensor {weight_key} missing companions: "
            f"required {scale_key} and {shape_key}"
        )

    if dtype == torch.uint8:
        if scale_key in shard_keys:
            sd = shard_f.get_slice(scale_key).dtype
            if sd == torch.uint8:
                return QuantFormat.MXFP4
        raise ValueError(
            f"uint8 tensor {weight_key} does not match MXFP4 pattern "
            f"(requires {scale_key} with dtype uint8)"
        )

    if _is_floating_dtype(dtype):
        return QuantFormat.FLOAT

    raise ValueError(f"Unsupported tensor format for {weight_key}: dtype={dtype}")


def get_companion_keys(weight_key: str, fmt: QuantFormat, shard_keys: set[str]) -> list[str]:
    base = weight_key.removesuffix(".weight").removesuffix(".weight_packed")
    if fmt in {QuantFormat.FP8_BLOCK_INV, QuantFormat.FP8_BLOCK_INV_UE8M0}:
        return [f"{base}.weight_scale_inv"]
    if fmt == QuantFormat.CT_FP8_BLOCK:
        return [f"{base}.weight_scale"]
    if fmt == QuantFormat.INT4_CT:
        keys = [f"{base}.weight_scale", f"{base}.weight_shape"]
        for suffix in (".weight_zero_point", ".weight_g_idx"):
            k = f"{base}{suffix}"
            if k in shard_keys:
                keys.append(k)
        return keys
    if fmt == QuantFormat.MXFP4:
        return [f"{base}.weight_scale"]
    return []


def _resolve_base_weight_key(module_key: str, base_keys: set[str]) -> str:
    candidates = [f"{module_key}.weight", f"{module_key}.weight_packed"]
    for c in candidates:
        if c in base_keys:
            return c
    raise KeyError(f"No base weight found for module_key={module_key}")


def dequant_fp8_block_inv(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

    return block_quant_dequant(weight_fp8, scale_inv, list(block_size), torch.float32)


def dequant_ct_fp8_block(
    weight_fp8: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

    return block_quant_dequant(weight_fp8, scale, list(block_size), torch.float32)


def requant_fp8_block(
    weight_f32: torch.Tensor,
    *,
    block_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8

    # Vendored sglang supports parameterized block sizes.
    # Keep weight_f32 as float32 to match kernel expectations.
    w_fp8, sf = per_block_cast_to_fp8(
        weight_f32.to(torch.float32),
        block_n=int(block_size[0]),
        block_k=int(block_size[1]),
    )
    return w_fp8.contiguous(), sf.contiguous()


def dequant_int4_ct(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
        unpack_from_int32,
    )

    out_features = int(weight_shape[0].item())
    in_features = int(weight_shape[1].item())
    w_int = unpack_from_int32(weight_packed, num_bits=4, shape=(out_features, in_features))
    w_int = w_int.to(torch.float32)
    expanded = weight_scale.to(torch.float32).repeat_interleave(group_size, dim=1)[:, :in_features]
    return (w_int * expanded).contiguous()


def requant_int4_ct(
    weight_f32: torch.Tensor,
    *,
    group_size: int,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
        pack_to_int32,
    )

    out_features, in_features = int(weight_f32.shape[0]), int(weight_f32.shape[1])
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group_size}"
        )
    grouped = weight_f32.to(torch.float32).reshape(out_features, -1, group_size)
    scales = grouped.abs().amax(dim=-1) / 7.0
    scales = scales.clamp(min=1e-10)
    q = (grouped / scales.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    q = q.reshape(out_features, in_features)
    packed = pack_to_int32(q, num_bits=4).contiguous()
    return packed, scales.to(scale_dtype).contiguous()


def dequant_mxfp4(weight_packed: torch.Tensor, weight_scale: torch.Tensor, *, block_size: int) -> torch.Tensor:
    from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

    return MXFP4QuantizeUtil.dequantize(
        quantized_data=weight_packed,
        dtype=torch.float32,
        scale=weight_scale,
        block_sizes=[block_size],
    ).contiguous()


def requant_mxfp4(
    weight_f32: torch.Tensor,
    *,
    block_size: int,
    base_packed_shape: torch.Size,
    base_scale_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

    packed, scale = MXFP4QuantizeUtil.quantize(weight_f32.to(torch.float32), block_size=block_size)
    packed = packed.contiguous()
    scale = scale.contiguous()
    if packed.shape != base_packed_shape:
        raise ValueError(
            f"MXFP4 packed shape mismatch: got {tuple(packed.shape)} expected {tuple(base_packed_shape)}"
        )
    if scale.shape != base_scale_shape:
        raise ValueError(
            f"MXFP4 scale shape mismatch: got {tuple(scale.shape)} expected {tuple(base_scale_shape)}"
        )
    return packed, scale


def _is_block_diagonal(A: torch.Tensor, r_base: int, tp: int) -> bool:
    if tp <= 1:
        return False
    if A.ndim != 2:
        return False
    if int(A.shape[0]) != r_base * tp:
        return False
    if int(A.shape[1]) % tp != 0:
        return False
    in_per = int(A.shape[1]) // tp
    for k in range(tp):
        for j in range(tp):
            if k == j:
                continue
            block = A[k * r_base : (k + 1) * r_base, j * in_per : (j + 1) * in_per]
            if block.any():
                return False
    return True


def _infer_tp_from_rank_pattern(
    module_key: str,
    *,
    r_default: int,
    rank_pattern: Optional[dict[str, int]],
) -> int:
    if not rank_pattern:
        return 1
    compiled = _compile_patterns(rank_pattern)
    candidates = (module_key, "base_model.model." + module_key)
    for c in candidates:
        for pat, val in compiled:
            if pat.search(c):
                try:
                    rv = int(val)
                    if rv > 0 and r_default > 0 and rv % r_default == 0:
                        return max(1, rv // r_default)
                except Exception:
                    return 1
    return 1


def _apply_lora_blockwise(
    *,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float,
    fan_in_fan_out: bool,
    device: torch.device,
    compute_dtype: torch.dtype,
    chunk_mib: float,
    verify: bool,
    tp_hint: int,
) -> torch.Tensor:
    """
    Wrapper around the baseline implementation with optional TP block-diagonal fast path.
    """
    if W.ndim != 2 or A.ndim != 2 or B.ndim != 2:
        _die(f"LoRA merge requires 2D tensors, got W={tuple(W.shape)} A={tuple(A.shape)} B={tuple(B.shape)}")

    r = int(A.shape[0])
    if int(B.shape[1]) != r:
        _die(f"LoRA shape mismatch A={tuple(A.shape)} B={tuple(B.shape)}")

    if tp_hint > 1 and r % tp_hint == 0 and not fan_in_fan_out:
        r_base = r // tp_hint
        if _is_block_diagonal(A, r_base, tp_hint):
            in_per = int(A.shape[1]) // tp_hint
            W_out = W.clone()
            A_dev = A.to(device=device, dtype=compute_dtype)
            B_dev = B.to(device=device, dtype=compute_dtype)
            for k in range(tp_hint):
                c0 = k * in_per
                c1 = c0 + in_per
                A_k = A_dev[k * r_base : (k + 1) * r_base, c0:c1]
                B_k = B_dev[:, k * r_base : (k + 1) * r_base]
                delta = (B_k @ A_k).to(device="cpu", dtype=W_out.dtype)
                W_out[:, c0:c1].add_(delta, alpha=float(scaling))
            if verify:
                # Reuse baseline verifier by comparing against generic path on tiny slices.
                ref = _apply_lora_blockwise_base(
                    W=W,
                    A=A,
                    B=B,
                    scaling=scaling,
                    fan_in_fan_out=fan_in_fan_out,
                    device=device,
                    compute_dtype=compute_dtype,
                    chunk_mib=min(chunk_mib, 16.0),
                    verify=True,
                )
                err = (W_out - ref).abs().max().item()
                if not math.isfinite(err) or err > 1e-2:
                    _die(f"Block-diagonal optimization verification failed: max_abs_err={err}")
            return W_out

    return _apply_lora_blockwise_base(
        W=W,
        A=A,
        B=B,
        scaling=scaling,
        fan_in_fan_out=fan_in_fan_out,
        device=device,
        compute_dtype=compute_dtype,
        chunk_mib=chunk_mib,
        verify=verify,
    )


def _apply_lora_with_padding_support(
    *,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float,
    fan_in_fan_out: bool,
    device: torch.device,
    compute_dtype: torch.dtype,
    chunk_mib: float,
    verify: bool,
    tp_hint: int,
) -> torch.Tensor:
    """
    Apply LoRA while tolerating padded quantized matrices (e.g. MXFP4).
    """
    if A.ndim != 2 or B.ndim != 2:
        _die(f"Expected 2D LoRA A/B, got A={tuple(A.shape)} B={tuple(B.shape)}")

    r = int(A.shape[0])
    in_features = int(A.shape[1])
    out_features = int(B.shape[0])
    expected = (in_features, out_features) if fan_in_fan_out else (out_features, in_features)
    if tuple(W.shape) == expected:
        return _apply_lora_blockwise(
            W=W,
            A=A,
            B=B,
            scaling=scaling,
            fan_in_fan_out=fan_in_fan_out,
            device=device,
            compute_dtype=compute_dtype,
            chunk_mib=chunk_mib,
            verify=verify,
            tp_hint=tp_hint,
        )

    if W.ndim != 2:
        _die(f"Expected 2D weight for padded merge support, got {tuple(W.shape)}")
    if int(W.shape[0]) < expected[0] or int(W.shape[1]) < expected[1]:
        _die(
            "Padded merge expected base tensor to be at least logical shape: "
            f"base={tuple(W.shape)} logical={expected}"
        )

    out = W.clone()
    region = out[: expected[0], : expected[1]]
    merged_region = _apply_lora_blockwise(
        W=region,
        A=A,
        B=B,
        scaling=scaling,
        fan_in_fan_out=fan_in_fan_out,
        device=device,
        compute_dtype=compute_dtype,
        chunk_mib=chunk_mib,
        verify=verify,
        tp_hint=tp_hint,
    )
    out[: expected[0], : expected[1]] = merged_region
    return out


def dequantize(
    *,
    fmt: QuantFormat,
    weight_key: str,
    weight_q: torch.Tensor,
    companions: Mapping[str, torch.Tensor],
    qparams: QuantParams,
) -> torch.Tensor:
    base = weight_key.removesuffix(".weight").removesuffix(".weight_packed")
    if fmt == QuantFormat.FLOAT:
        return weight_q.to(torch.float32)
    if fmt == QuantFormat.FP8_BLOCK_INV:
        return dequant_fp8_block_inv(weight_q, companions[f"{base}.weight_scale_inv"], qparams.block_size)
    if fmt == QuantFormat.FP8_BLOCK_INV_UE8M0:
        from sglang.srt.layers.quantization.fp8_utils import (
            block_quant_dequant,
            inverse_transform_scale_ue8m0,
        )

        scale_packed = companions[f"{base}.weight_scale_inv"]
        scale = inverse_transform_scale_ue8m0(
            scale_packed,
            mn=int(weight_q.shape[-2]),
            block_n=int(qparams.block_size[0]),
        )
        return block_quant_dequant(weight_q, scale, list(qparams.block_size), torch.float32)
    if fmt == QuantFormat.CT_FP8_BLOCK:
        return dequant_ct_fp8_block(weight_q, companions[f"{base}.weight_scale"], qparams.block_size)
    if fmt == QuantFormat.INT4_CT:
        g_idx_key = f"{base}.weight_g_idx"
        if g_idx_key in companions:
            raise ValueError(
                f"{weight_key}: weight_g_idx is present (actorder). This is unsupported in v1."
            )
        return dequant_int4_ct(
            weight_q,
            companions[f"{base}.weight_scale"],
            companions[f"{base}.weight_shape"],
            group_size=qparams.group_size,
        )
    if fmt == QuantFormat.MXFP4:
        return dequant_mxfp4(
            weight_q,
            companions[f"{base}.weight_scale"],
            block_size=qparams.mxfp4_block_size,
        )
    raise AssertionError(f"Unhandled format: {fmt}")


def requantize(
    *,
    fmt: QuantFormat,
    weight_key: str,
    merged_f32: torch.Tensor,
    companions: Mapping[str, torch.Tensor],
    qparams: QuantParams,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base = weight_key.removesuffix(".weight").removesuffix(".weight_packed")
    if fmt == QuantFormat.FLOAT:
        # Caller handles cast back to original dtype.
        return merged_f32, {}
    if fmt in {QuantFormat.FP8_BLOCK_INV, QuantFormat.CT_FP8_BLOCK}:
        w, sf = requant_fp8_block(merged_f32, block_size=qparams.block_size)
        ck = f"{base}.weight_scale_inv" if fmt == QuantFormat.FP8_BLOCK_INV else f"{base}.weight_scale"
        return w, {ck: sf.to(companions[ck].dtype)}
    if fmt == QuantFormat.FP8_BLOCK_INV_UE8M0:
        from sglang.srt.layers.quantization.fp8_utils import transform_scale_ue8m0

        w, sf = requant_fp8_block(merged_f32, block_size=qparams.block_size)
        # Preserve UE8M0 packed int32 scale format used by the base checkpoint.
        ck = f"{base}.weight_scale_inv"
        packed = transform_scale_ue8m0(
            sf.to(torch.float32),
            mn=int(merged_f32.shape[0]),
            block_n=int(qparams.block_size[0]),
            # Prefer the reference torch implementation for portability.
            use_torch_impl=True,
        )
        return w, {ck: packed.to(dtype=companions[ck].dtype)}
    if fmt == QuantFormat.INT4_CT:
        scale_key = f"{base}.weight_scale"
        packed, scales = requant_int4_ct(
            merged_f32,
            group_size=qparams.group_size,
            scale_dtype=companions[scale_key].dtype,
        )
        return packed, {scale_key: scales}
    if fmt == QuantFormat.MXFP4:
        scale_key = f"{base}.weight_scale"
        packed, scale = requant_mxfp4(
            merged_f32,
            block_size=qparams.mxfp4_block_size,
            base_packed_shape=companions[weight_key].shape,
            base_scale_shape=companions[scale_key].shape,
        )
        return packed, {scale_key: scale}
    raise AssertionError(f"Unhandled format: {fmt}")


def _collect_all_tensor_meta(model_dir: Path, weight_map: Mapping[str, str]) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(k)

    out: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
    for shard, keys in by_shard.items():
        with safe_open((model_dir / shard).as_posix(), framework="pt", device="cpu") as f:
            for k in keys:
                sl = f.get_slice(k)
                out[k] = (sl.dtype, tuple(int(x) for x in sl.get_shape()))
    return out


def _validate_structural(
    *,
    base_layout: Any,
    out_dir: Path,
    weight_map_out: Mapping[str, str],
) -> None:
    idx_path = out_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        _die(f"Structural validation failed: missing index {idx_path}")
    try:
        idx = _load_json(idx_path)
    except Exception as e:
        _die(f"Structural validation failed: invalid index JSON: {e}")

    wm = idx.get("weight_map")
    if not isinstance(wm, dict):
        _die("Structural validation failed: index missing weight_map object")

    for shard in sorted(set(str(v) for v in wm.values())):
        if not (out_dir / shard).exists():
            _die(f"Structural validation failed: shard referenced but missing: {shard}")

    base_meta = _collect_all_tensor_meta(base_layout.base_dir, base_layout.weight_map)
    out_meta = _collect_all_tensor_meta(out_dir, weight_map_out)

    base_keys = set(base_meta.keys())
    out_keys = set(out_meta.keys())
    if base_keys != out_keys:
        miss = sorted(base_keys - out_keys)[:5]
        extra = sorted(out_keys - base_keys)[:5]
        _die(
            "Structural validation failed: output key set mismatch "
            f"(missing={miss} extra={extra})"
        )

    for k in sorted(base_keys):
        bd, bs = base_meta[k]
        od, osz = out_meta[k]
        if bd != od:
            _die(f"Structural validation failed: dtype mismatch for {k}: base={bd} out={od}")
        if bs != osz:
            _die(
                f"Structural validation failed: shape mismatch for {k}: "
                f"base={bs} out={osz}"
            )


def _sample_items(items: Sequence[str], n: int, *, seed: int = 0) -> list[str]:
    if not items:
        return []
    rng = random.Random(seed)
    if len(items) <= n:
        return list(items)
    return rng.sample(list(items), n)


def _run_verify_checks(
    *,
    base_dir: Path,
    out_dir: Path,
    weight_map: Mapping[str, str],
    lora_by_base_weight: Mapping[str, LoraForWeight],
    adapter_weights_path: Path,
    adapter_spec: Any,
    qparams: QuantParams,
    samples: int = 6,
) -> None:
    """
    Verification suite:
    - numerical spot-check on merged keys
    - codec roundtrip on unmodified keys
    """
    print("[verify] starting numerical spot-check and codec roundtrip checks...")
    t0 = time.monotonic()

    targeted = [k for k in lora_by_base_weight.keys() if k in weight_map]
    non_targeted = [k for k in weight_map.keys() if k not in lora_by_base_weight]
    sample_targeted = _sample_items(targeted, samples, seed=123)
    sample_non = _sample_items(non_targeted, samples, seed=456)

    with safe_open(adapter_weights_path.as_posix(), framework="pt", device="cpu") as af:
        for k in sample_targeted:
            shard = weight_map[k]
            with safe_open((base_dir / shard).as_posix(), framework="pt", device="cpu") as bf:
                with safe_open((out_dir / shard).as_posix(), framework="pt", device="cpu") as of:
                    shard_keys = set(bf.keys())
                    dtype = bf.get_slice(k).dtype
                    fmt = detect_quant_format(weight_key=k, shard_keys=shard_keys, dtype=dtype, shard_f=bf)
                    ckeys = get_companion_keys(k, fmt, shard_keys)
                    comp_base = {ck: bf.get_tensor(ck) for ck in ckeys}
                    comp_out = {ck: of.get_tensor(ck) for ck in ckeys}
                    if fmt == QuantFormat.MXFP4:
                        comp_base[k] = bf.get_tensor(k)
                        comp_out[k] = of.get_tensor(k)

                    w_base_q = bf.get_tensor(k)
                    w_out_q = of.get_tensor(k)
                    w_base = dequantize(
                        fmt=fmt, weight_key=k, weight_q=w_base_q, companions=comp_base, qparams=qparams
                    )
                    w_out = dequantize(
                        fmt=fmt, weight_key=k, weight_q=w_out_q, companions=comp_out, qparams=qparams
                    )

                    lora = lora_by_base_weight[k]
                    A = af.get_tensor(lora.a_key)
                    B = af.get_tensor(lora.b_key)
                    alpha_patterns = _compile_patterns(adapter_spec.alpha_pattern)
                    alpha = _resolve_alpha_for_module(
                        adapter_spec=adapter_spec,
                        module_key=lora.module_key,
                        alpha_patterns=[(p, float(v)) for p, v in alpha_patterns],
                    )
                    scale = _scaling(alpha, int(A.shape[0]), adapter_spec.use_rslora)
                    delta = (B.to(torch.float32) @ A.to(torch.float32)) * float(scale)
                    # Allow padded quantized tensors.
                    drows = min(delta.shape[0], w_out.shape[0])
                    dcols = min(delta.shape[1], w_out.shape[1])
                    err = (w_out[:drows, :dcols] - w_base[:drows, :dcols] - delta[:drows, :dcols]).abs().max().item()

                    tol = 1e-3
                    if fmt in {
                        QuantFormat.FP8_BLOCK_INV,
                        QuantFormat.FP8_BLOCK_INV_UE8M0,
                        QuantFormat.CT_FP8_BLOCK,
                    }:
                        tol = 5e-2
                    elif fmt in {QuantFormat.INT4_CT, QuantFormat.MXFP4}:
                        tol = 5e-1
                    if not math.isfinite(err) or err > tol:
                        _die(
                            f"[verify] numerical spot-check failed for {k}: "
                            f"fmt={fmt.value} max_abs_err={err} tol={tol}"
                        )

        for k in sample_non:
            shard = weight_map[k]
            with safe_open((base_dir / shard).as_posix(), framework="pt", device="cpu") as bf:
                shard_keys = set(bf.keys())
                dtype = bf.get_slice(k).dtype
                try:
                    fmt = detect_quant_format(weight_key=k, shard_keys=shard_keys, dtype=dtype, shard_f=bf)
                except Exception:
                    continue
                if fmt == QuantFormat.FLOAT:
                    continue
                ckeys = get_companion_keys(k, fmt, shard_keys)
                companions = {ck: bf.get_tensor(ck) for ck in ckeys}
                if fmt == QuantFormat.MXFP4:
                    companions[k] = bf.get_tensor(k)
                wq = bf.get_tensor(k)
                w_f = dequantize(fmt=fmt, weight_key=k, weight_q=wq, companions=companions, qparams=qparams)
                wq_rt, comp_rt = requantize(
                    fmt=fmt, weight_key=k, merged_f32=w_f, companions=companions, qparams=qparams
                )

                if fmt == QuantFormat.INT4_CT:
                    if not torch.equal(wq_rt, wq):
                        _die(f"[verify] INT4 roundtrip packed mismatch for {k}")
                else:
                    comp2 = dict(companions)
                    comp2.update(comp_rt)
                    if fmt == QuantFormat.MXFP4:
                        comp2[k] = wq_rt
                    w_f_rt = dequantize(fmt=fmt, weight_key=k, weight_q=wq_rt, companions=comp2, qparams=qparams)
                    max_err = (w_f - w_f_rt).abs().max().item()
                    tol = (
                        5e-2
                        if fmt
                        in {
                            QuantFormat.FP8_BLOCK_INV,
                            QuantFormat.FP8_BLOCK_INV_UE8M0,
                            QuantFormat.CT_FP8_BLOCK,
                        }
                        else 5e-1
                    )
                    if not math.isfinite(max_err) or max_err > tol:
                        _die(f"[verify] codec roundtrip failed for {k}: fmt={fmt.value} max_err={max_err} tol={tol}")

    dt = time.monotonic() - t0
    print(f"[verify] completed in {dt:.1f}s")


def _remap_lora_targets_to_base_keys(
    *,
    base_keys: set[str],
    lora_by_base_weight: Mapping[str, LoraForWeight],
) -> dict[str, LoraForWeight]:
    out: dict[str, LoraForWeight] = {}
    for l in lora_by_base_weight.values():
        real_key = _resolve_base_weight_key(l.module_key, base_keys)
        out[real_key] = LoraForWeight(
            base_weight_key=real_key,
            a_key=l.a_key,
            b_key=l.b_key,
            b_bias_key=l.b_bias_key,
            module_key=l.module_key,
        )
    return out


def merge_lora_quantized(
    *,
    base_dir: Path,
    adapter_dir: Path,
    out_dir: Path,
    adapter_name: str,
    device: torch.device,
    compute_dtype: torch.dtype,
    chunk_mib: float,
    verify: bool,
    link_unchanged: str = "auto",
    overwrite: bool = False,
    group_size_override: Optional[int] = None,
    block_size_override: Optional[tuple[int, int]] = None,
) -> None:
    _require_quant_deps()
    adapter_spec = _parse_adapter_spec(adapter_dir)
    base_layout = _detect_base_layout(base_dir)

    config = _load_json(base_dir / "config.json") if (base_dir / "config.json").exists() else {}
    qparams = _extract_quant_params(config if isinstance(config, dict) else {})
    if group_size_override is not None:
        qparams = QuantParams(
            group_size=group_size_override,
            block_size=qparams.block_size,
            num_bits=qparams.num_bits,
            mxfp4_block_size=qparams.mxfp4_block_size,
        )
    if block_size_override is not None:
        qparams = QuantParams(
            group_size=qparams.group_size,
            block_size=block_size_override,
            num_bits=qparams.num_bits,
            mxfp4_block_size=qparams.mxfp4_block_size,
        )

    lora_by_base_weight_raw, overrides, bias_updates = _build_lora_maps(
        adapter_dir=adapter_dir,
        adapter_spec=adapter_spec,
        adapter_name=adapter_name,
    )
    base_keys = set(base_layout.weight_map.keys())
    lora_by_base_weight = _remap_lora_targets_to_base_keys(
        base_keys=base_keys,
        lora_by_base_weight=lora_by_base_weight_raw,
    )

    missing_bias_targets = [k for k in bias_updates.keys() if k not in base_keys]
    if missing_bias_targets:
        _die(
            "Adapter requires bias updates but base checkpoint is missing targets. "
            f"Example: {missing_bias_targets[:5]}"
        )

    base_override_keys = set(overrides.keys()) & base_keys
    affected_keys = set(lora_by_base_weight.keys()) | base_override_keys | set(bias_updates.keys())
    affected_keys = {k for k in affected_keys if k in base_keys}
    changed_shards = {base_layout.weight_map[k] for k in affected_keys}

    if link_unchanged != "off":
        if _is_subpath(out_dir, base_dir) or _is_subpath(base_dir, out_dir):
            _die(
                "Refusing to link unchanged shards with overlapping base/out trees: "
                f"base_dir={base_dir} out_dir={out_dir}"
            )

    _prepare_out_dir(out_dir, overwrite=overwrite)

    if link_unchanged not in {"auto", "hardlink", "symlink", "off"}:
        _die(f"Unsupported --link-unchanged mode: {link_unchanged}")

    print(
        f"[merge] qparams group_size={qparams.group_size} block_size={qparams.block_size} "
        f"num_bits={qparams.num_bits}"
    )
    print(
        f"[merge] shards_total={len(base_layout.shard_files)} "
        f"shards_changed={len(changed_shards)} "
        f"shards_linked={len(base_layout.shard_files) - len(changed_shards)} "
        f"link_mode={link_unchanged}"
    )

    adapter_weights_path = _adapter_weights_path(adapter_dir)
    adapter_f = safe_open(adapter_weights_path.as_posix(), framework="pt", device="cpu")

    t_start = time.monotonic()
    try:
        for shard_path in base_layout.shard_files:
            shard_name = shard_path.name
            out_shard_path = out_dir / shard_name
            if link_unchanged != "off" and shard_name not in changed_shards:
                _link_unchanged_shard(
                    src=shard_path,
                    dst=out_shard_path,
                    mode=link_unchanged,
                    overwrite=overwrite,
                )
                print(f"[merge] shard {shard_name} -> linked")
                continue

            print(f"[merge] shard {shard_name} -> rewritten")
            with safe_open(shard_path.as_posix(), framework="pt", device="cpu") as base_f:
                shard_keys = set(base_f.keys())
                out_tensors: dict[str, torch.Tensor] = {}
                for k in base_f.keys():
                    # Direct override wins.
                    if k in overrides:
                        src_k = overrides[k]
                        base_dtype = base_f.get_slice(k).dtype
                        out_tensors[k] = adapter_f.get_tensor(src_k).to(dtype=base_dtype, device="cpu").contiguous()
                        continue

                    # Bias update for lora_bias=True.
                    if k in bias_updates:
                        b_bias_key, scale = bias_updates[k]
                        W = base_f.get_tensor(k)
                        delta_bias = adapter_f.get_tensor(b_bias_key).to(dtype=W.dtype, device="cpu")
                        out_tensors[k] = W.clone().add_(delta_bias, alpha=float(scale)).contiguous()
                        continue

                    lora = lora_by_base_weight.get(k)
                    if lora is None:
                        out_tensors[k] = base_f.get_tensor(k).contiguous()
                        continue

                    A = adapter_f.get_tensor(lora.a_key)
                    B = adapter_f.get_tensor(lora.b_key)
                    alpha_patterns = _compile_patterns(adapter_spec.alpha_pattern)
                    alpha = _resolve_alpha_for_module(
                        adapter_spec=adapter_spec,
                        module_key=lora.module_key,
                        alpha_patterns=[(p, float(v)) for p, v in alpha_patterns],
                    )
                    scale = _scaling(alpha, int(A.shape[0]), adapter_spec.use_rslora)
                    dtype = base_f.get_slice(k).dtype
                    fmt = detect_quant_format(weight_key=k, shard_keys=shard_keys, dtype=dtype, shard_f=base_f)
                    ckeys = get_companion_keys(k, fmt, shard_keys)
                    companions = {ck: base_f.get_tensor(ck) for ck in ckeys}
                    if fmt == QuantFormat.MXFP4:
                        companions[k] = base_f.get_tensor(k)

                    if fmt == QuantFormat.FLOAT:
                        W = base_f.get_tensor(k)
                        tp_hint = _infer_tp_from_rank_pattern(
                            lora.module_key,
                            r_default=adapter_spec.r_default,
                            rank_pattern=adapter_spec.rank_pattern,
                        )
                        merged = _apply_lora_with_padding_support(
                            W=W,
                            A=A,
                            B=B,
                            scaling=scale,
                            fan_in_fan_out=adapter_spec.fan_in_fan_out,
                            device=device,
                            compute_dtype=compute_dtype,
                            chunk_mib=chunk_mib,
                            verify=verify,
                            tp_hint=tp_hint,
                        )
                        out_tensors[k] = merged.to(dtype=W.dtype, device="cpu").contiguous()
                        continue

                    W_q = base_f.get_tensor(k)
                    W_f = dequantize(
                        fmt=fmt,
                        weight_key=k,
                        weight_q=W_q,
                        companions=companions,
                        qparams=qparams,
                    )
                    tp_hint = _infer_tp_from_rank_pattern(
                        lora.module_key,
                        r_default=adapter_spec.r_default,
                        rank_pattern=adapter_spec.rank_pattern,
                    )
                    merged_f = _apply_lora_with_padding_support(
                        W=W_f,
                        A=A,
                        B=B,
                        scaling=scale,
                        fan_in_fan_out=adapter_spec.fan_in_fan_out,
                        device=device,
                        compute_dtype=compute_dtype,
                        chunk_mib=chunk_mib,
                        verify=verify,
                        tp_hint=tp_hint,
                    )
                    new_weight, updated_companions = requantize(
                        fmt=fmt,
                        weight_key=k,
                        merged_f32=merged_f,
                        companions=companions,
                        qparams=qparams,
                    )
                    out_tensors[k] = new_weight.contiguous()
                    for ck, cv in updated_companions.items():
                        out_tensors[ck] = cv.contiguous()
                    # Preserve non-updated companion tensors.
                    for ck in ckeys:
                        if ck not in out_tensors:
                            out_tensors[ck] = companions[ck].contiguous()

                save_file(out_tensors, out_shard_path.as_posix(), metadata={"format": "pt"})
                del out_tensors
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Preserve index shape and mapping exactly.
        if base_layout.index_json is not None:
            out_index = dict(base_layout.index_json)
            out_index["weight_map"] = dict(base_layout.weight_map)
            _dump_json(out_dir / "model.safetensors.index.json", out_index)
        else:
            out_index = {"metadata": {"format": "pt"}, "weight_map": dict(base_layout.weight_map)}
            _dump_json(out_dir / "model.safetensors.index.json", out_index)

        _copy_support_files(base_dir, out_dir)

    finally:
        try:
            adapter_f.close()
        except Exception:
            pass

    elapsed = time.monotonic() - t_start
    print(f"[merge] merge completed in {elapsed:.1f}s")

    # Always-on structural validation.
    _validate_structural(base_layout=base_layout, out_dir=out_dir, weight_map_out=base_layout.weight_map)
    print("[merge] structural validation passed")

    if verify:
        _run_verify_checks(
            base_dir=base_dir,
            out_dir=out_dir,
            weight_map=base_layout.weight_map,
            lora_by_base_weight=lora_by_base_weight,
            adapter_weights_path=adapter_weights_path,
            adapter_spec=adapter_spec,
            qparams=qparams,
        )


def _self_test() -> None:
    """
    Synthetic codec sanity tests; does not require large checkpoints.
    """
    _require_quant_deps()
    torch.manual_seed(0)

    # INT4 roundtrip test
    x = torch.randn(64, 128, dtype=torch.float32) * 0.3
    packed, scale = requant_int4_ct(x, group_size=32, scale_dtype=torch.float32)
    x_shape = torch.tensor([x.shape[0], x.shape[1]], dtype=torch.int64)
    x_rt = dequant_int4_ct(packed, scale, x_shape, group_size=32)
    err = (x - x_rt).abs().max().item()
    if not math.isfinite(err) or err > 0.7:
        _die(f"self-test INT4 roundtrip failed: max_err={err}")

    # FP8 roundtrip test
    y = torch.randn(128, 128, dtype=torch.float32)
    yq, ys = requant_fp8_block(y, block_size=(128, 128))
    y_rt = dequant_fp8_block_inv(yq, ys, (128, 128))
    err2 = (y - y_rt).abs().max().item()
    if not math.isfinite(err2) or err2 > 0.1:
        _die(f"self-test FP8 roundtrip failed: max_err={err2}")

    # FP8 roundtrip with non-128 block size (parameterization sanity).
    y2 = torch.randn(96, 160, dtype=torch.float32)
    y2q, y2s = requant_fp8_block(y2, block_size=(64, 64))
    y2_rt = dequant_fp8_block_inv(y2q, y2s, (64, 64))
    err2b = (y2 - y2_rt).abs().max().item()
    if not math.isfinite(err2b) or err2b > 0.2:
        _die(f"self-test FP8(64x64) roundtrip failed: max_err={err2b}")

    # UE8M0 pack/unpack sanity (DeepGEMM scale layout; block_n parameterized).
    try:
        from sglang.srt.layers.quantization.fp8_utils import (
            inverse_transform_scale_ue8m0,
            transform_scale_ue8m0,
        )

        packed = transform_scale_ue8m0(
            ys.to(torch.float32),
            mn=int(y.shape[0]),
            block_n=128,
            use_torch_impl=True,
        )
        unpacked = inverse_transform_scale_ue8m0(
            packed,
            mn=int(y.shape[0]),
            block_n=128,
        )
        max_err_scale = (unpacked - ys.to(torch.float32)).abs().max().item()
        if not math.isfinite(max_err_scale) or max_err_scale > 0.0:
            _die(f"self-test UE8M0 pack/unpack failed: max_err={max_err_scale}")
    except Exception as exc:
        _die(f"self-test UE8M0 pack/unpack failed: {exc}")

    # MXFP4 roundtrip test
    z = torch.randn(64, 128, dtype=torch.float32) * 0.1
    zq, zs = requant_mxfp4(
        z,
        block_size=32,
        base_packed_shape=torch.Size([64, 64]),
        base_scale_shape=torch.Size([256, 1]),
    )
    z_rt = dequant_mxfp4(zq, zs, block_size=32)
    err3 = (z - z_rt).abs().max().item()
    if not math.isfinite(err3) or err3 > 1.0:
        _die(f"self-test MXFP4 roundtrip failed: max_err={err3}")

    print("[self-test] codec checks passed")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Quantization-preserving PEFT LoRA merge for sharded safetensors checkpoints."
    )
    p.add_argument("--base-dir", type=str, required=False, help="Base model directory")
    p.add_argument("--adapter-dir", type=str, required=False, help="PEFT adapter directory")
    p.add_argument("--out-dir", type=str, required=False, help="Merged output directory")
    p.add_argument("--adapter-name", type=str, default="default")
    p.add_argument("--device", type=str, default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--compute-dtype", type=str, default="float32", help="float32|bfloat16|float16")
    p.add_argument("--chunk-mib", type=float, default=128.0, help="Delta chunk size in MiB")
    p.add_argument("--verify", action="store_true", help="Run numerical and codec verification after merge")
    p.add_argument("--group-size", type=int, default=None, help="INT4 group size override")
    p.add_argument("--block-size", type=str, default=None, help="FP8 block size override INT,INT (e.g. 128,128)")
    p.add_argument(
        "--link-unchanged",
        type=str,
        default="auto",
        help="How to handle unchanged shards: auto|hardlink|symlink|off",
    )
    p.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty out dir")
    p.add_argument("--self-test", action="store_true", help="Run synthetic codec self-test and exit")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.self_test:
        _self_test()
        return

    if not args.base_dir or not args.adapter_dir or not args.out_dir:
        p.print_help()
        _die("Missing required arguments: --base-dir, --adapter-dir, --out-dir")
    if args.group_size is not None and int(args.group_size) <= 0:
        _die("--group-size must be > 0")

    merge_lora_quantized(
        base_dir=Path(args.base_dir),
        adapter_dir=Path(args.adapter_dir),
        out_dir=Path(args.out_dir),
        adapter_name=str(args.adapter_name),
        device=_parse_device(args.device),
        compute_dtype=_parse_dtype(args.compute_dtype),
        chunk_mib=float(args.chunk_mib),
        verify=bool(args.verify),
        link_unchanged=str(args.link_unchanged),
        overwrite=bool(args.overwrite),
        group_size_override=(int(args.group_size) if args.group_size is not None else None),
        block_size_override=_parse_block_size(args.block_size),
    )


if __name__ == "__main__":
    main()
