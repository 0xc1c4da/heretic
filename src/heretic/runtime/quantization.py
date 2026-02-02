# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import transformers
from transformers import BitsAndBytesConfig, PretrainedConfig


@dataclass(frozen=True)
class QuantizationRequest:
    """
    Normalized quantization request derived from Settings.

    `requested_dtype` is the dtype string we are trying for model load (e.g. "auto", "float16").
    """

    method: str  # "none" | "auto" | "bnb_4bit" | "fp8" | "custom"
    requested_dtype: str
    config_type: str | None = None
    config_kwargs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class QuantizationInfo:
    """
    Resolved quantization information used by Model load and precision handling.

    - `load_quantization_config` is the object passed to `from_pretrained(..., quantization_config=...)`.
      For method="auto", this is typically None (Transformers uses model-provided config if present).
    - `model_provided_config` is the raw config dict from config.json (if present).
    - `method` is best-effort: request method or model-provided quant_method.
    - `compute_dtype` is the dtype used for non-quantized math (LoRA, generic ops).
    """

    is_quantized: bool
    method: str | None
    load_quantization_config: object | None
    model_provided_config: Mapping[str, Any] | None
    compute_dtype: torch.dtype
    has_model_quantization_config: bool = False
    supports_dequantize_on_load: bool = False


def get_model_provided_quantization_config(model_id_or_path: str) -> Mapping[str, Any] | None:
    config_dict, _ = PretrainedConfig.get_config_dict(model_id_or_path)
    raw = config_dict.get("quantization_config")
    return raw if isinstance(raw, Mapping) else None


def supports_dequantize_on_load(
    *,
    load_config: object | None,
    model_provided_config: Mapping[str, Any] | None,
) -> bool:
    """
    Data-driven detection of dequantize-on-load support using pinned Transformers classes.

    We only return True when the quantization config object (explicit or model-provided) exposes
    a loading attribute named `dequantize` via `get_loading_attributes()`.
    """

    def _has_dequantize_loading_attr(cfg: object) -> bool:
        get_attrs = getattr(cfg, "get_loading_attributes", None)
        if not callable(get_attrs):
            return False
        try:
            attrs = get_attrs()
        except Exception:
            return False
        return isinstance(attrs, dict) and "dequantize" in attrs

    if load_config is not None and _has_dequantize_loading_attr(load_config):
        return True

    if model_provided_config:
        try:
            from transformers.quantizers.auto import AutoQuantizationConfig

            cfg = AutoQuantizationConfig.from_dict(dict(model_provided_config))
        except Exception:
            return False
        return _has_dequantize_loading_attr(cfg)

    return False


def _infer_method_from_config_dict(cfg: Mapping[str, Any] | None) -> str | None:
    if not cfg:
        return None
    # Transformers commonly uses `quant_method` for method identifiers.
    method = cfg.get("quant_method")
    if isinstance(method, str) and method.strip():
        return method.strip().lower()
    return "unknown"


def _resolve_auto_fallback_dtype(*, requested: torch.dtype, fallback_setting: str) -> torch.dtype:
    """
    Mirror PrecisionPolicy.op_fallback_dtype behavior without depending on precision module.

    fallback_setting: "auto" | "bfloat16" | "float16" | "float32"
    """
    setting = (fallback_setting or "auto").strip().lower()
    if setting == "bfloat16":
        return torch.bfloat16
    if setting == "float16":
        return torch.float16
    if setting == "float32":
        return torch.float32

    # auto
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if requested == torch.float32:
        return torch.float32
    return torch.float16


def _resolve_compute_dtype(
    *,
    requested_dtype: str,
    is_quantized: bool,
    quantization_config: object | None,
    precision_fallback_dtype: str,
) -> torch.dtype:
    # bnb_4bit: respect explicit compute dtype if provided.
    if isinstance(quantization_config, BitsAndBytesConfig) and getattr(
        quantization_config, "load_in_4bit", False
    ):
        bnb_dtype = getattr(quantization_config, "bnb_4bit_compute_dtype", None)
        if isinstance(bnb_dtype, torch.dtype):
            return bnb_dtype

    # Quantized (fp8 / model-provided / unknown): compute should be bf16/fp16 (or configured fallback).
    if is_quantized:
        return _resolve_auto_fallback_dtype(
            requested=torch.float16,
            fallback_setting=precision_fallback_dtype,
        )

    # Non-quantized: if requested "auto", choose a sane compute dtype for adapters.
    if requested_dtype == "auto":
        return _resolve_auto_fallback_dtype(
            requested=torch.float16,
            fallback_setting=precision_fallback_dtype,
        )

    return getattr(torch, requested_dtype)


def build_quantization_config(req: QuantizationRequest) -> object | None:
    """
    Construct the transformers quantization config object to pass to `from_pretrained`.
    """
    method = (req.method or "none").strip().lower()
    kwargs = dict(req.config_kwargs or {})

    if method in {"none", ""}:
        return None

    if method == "auto":
        # Let transformers use model-provided quantization_config if present.
        return None

    if method == "bnb_4bit":
        # BitsAndBytesConfig expects a torch.dtype, not a string.
        if req.requested_dtype == "auto":
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = getattr(torch, req.requested_dtype)
        kwargs_full = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
        kwargs_full.update(kwargs)
        return BitsAndBytesConfig(**kwargs_full)

    if method == "fp8":
        try:
            from transformers import FineGrainedFP8Config
        except ImportError as exc:
            raise RuntimeError(
                "FineGrainedFP8Config is not available in this transformers version."
            ) from exc
        return FineGrainedFP8Config(**kwargs)

    if method == "custom":
        if not req.config_type:
            raise ValueError("quantization_config_type must be set when quantization='custom'.")
        cls = getattr(transformers, req.config_type, None)
        if cls is None:
            raise ValueError(f"Unknown quantization config class: {req.config_type}")
        return cls(**kwargs)

    raise ValueError(f"Unknown quantization method: {req.method}")


def resolve_quantization(
    *,
    model_id_or_path: str,
    request: QuantizationRequest,
    precision_fallback_dtype: str,
) -> QuantizationInfo:
    model_provided = get_model_provided_quantization_config(model_id_or_path)
    model_method = _infer_method_from_config_dict(model_provided)

    load_config = build_quantization_config(request)
    is_quantized = load_config is not None or model_provided is not None

    # If request explicitly sets method != auto/none, prefer that label for logs.
    req_method = (request.method or "none").strip().lower()
    if req_method in {"none", "auto"}:
        method = model_method
    else:
        method = req_method

    compute_dtype = _resolve_compute_dtype(
        requested_dtype=request.requested_dtype,
        is_quantized=is_quantized,
        quantization_config=load_config,
        precision_fallback_dtype=precision_fallback_dtype,
    )

    return QuantizationInfo(
        is_quantized=is_quantized,
        method=method,
        load_quantization_config=load_config,
        model_provided_config=model_provided,
        compute_dtype=compute_dtype,
        has_model_quantization_config=model_provided is not None,
        supports_dequantize_on_load=supports_dequantize_on_load(
            load_config=load_config,
            model_provided_config=model_provided,
        ),
    )

