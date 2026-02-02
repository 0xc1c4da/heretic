# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class WeightAccessError(RuntimeError):
    message: str
    details: dict[str, Any]

    def __str__(self) -> str:
        parts = [self.message]
        for k, v in self.details.items():
            parts.append(f"- {k}={v}")
        return "\n".join(parts)


class WeightAccess:
    """
    Weight access helper for abliteration.

    Abliteration needs an *effective* float weight matrix W to compute v^T W (and, in some modes,
    additional operations). Some quantization backends store weights in packed forms that cannot be
    naively cast to float.
    """

    @staticmethod
    def materialize_W_float32(*, base_layer: Any, component: str, layer_index: int) -> Tensor:
        """
        Materialize a float32 weight matrix W for the given base layer.

        Supported:
        - Normal floating-point weights
        - bitsandbytes 4-bit weights exposing `weight.quant_state`
        """
        w = getattr(base_layer, "weight", None)
        if not isinstance(w, Tensor):
            raise WeightAccessError(
                "Unsupported weight representation: base_layer.weight is not a Tensor.",
                {
                    "layer": layer_index,
                    "component": component,
                    "base_layer_class": type(base_layer).__name__,
                    "weight_type": type(w).__name__,
                },
            )

        quant_state = getattr(w, "quant_state", None)
        if quant_state is not None:
            # bitsandbytes 4-bit.
            try:
                import bitsandbytes as bnb  # type: ignore
            except Exception as exc:
                raise WeightAccessError(
                    "bitsandbytes is required to dequantize 4-bit weights for abliteration.",
                    {
                        "layer": layer_index,
                        "component": component,
                        "weight_dtype": str(w.dtype),
                    },
                ) from exc
            W = bnb.functional.dequantize_4bit(  # type: ignore[attr-defined]
                w.data,
                quant_state,
            )
            return W.to(torch.float32).view(W.shape[0], -1)

        if not w.is_floating_point():
            raise WeightAccessError(
                "Unsupported quantized weight tensor for abliteration (non-floating Tensor without bnb quant_state).",
                {
                    "layer": layer_index,
                    "component": component,
                    "weight_dtype": str(w.dtype),
                    "base_layer_class": type(base_layer).__name__,
                },
            )

        return w.to(torch.float32).view(w.shape[0], -1)

