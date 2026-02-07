# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LoraAdapterBundle:
    """A validated LoRA adapter artifact for disk + hot-swap backends.

    This is intentionally close to PEFT's adapter bundle format:
    - `adapter_config.json` (a PEFT-style dict)
    - `adapter_model.safetensors` (preferred) or `adapter_model.bin` (fallback)
    """

    tensors: dict[str, torch.Tensor]
    config_dict: dict[str, Any]
    base_model_name_or_path: str
    stats: dict[str, Any] | None = None

    def assert_valid(self) -> None:
        if not isinstance(self.tensors, dict) or not self.tensors:
            raise ValueError("LoRA bundle has no tensors.")

        # Basic tensor sanity.
        a = 0
        b = 0
        layer_compatible = 0
        module_bases: dict[str, set[str]] = {}
        for k, v in self.tensors.items():
            if not isinstance(k, str) or not isinstance(v, torch.Tensor):
                raise ValueError(f"Invalid LoRA tensor entry: {type(k).__name__} -> {type(v).__name__}")
            if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
                raise ValueError(f"Unexpected LoRA tensor key (missing lora_A/lora_B suffix): {k}")
            if v.ndim != 2:
                raise ValueError(f"Unexpected LoRA tensor rank (expected 2D): {k} {tuple(int(x) for x in v.shape)}")

            if re.search(r"layers\.(\d+)\.", k) is not None:
                layer_compatible += 1

            if k.endswith(".lora_A.weight"):
                a += 1
                base = k[: -len(".lora_A.weight")]
                module_bases.setdefault(base, set()).add("A")
            else:
                b += 1
                base = k[: -len(".lora_B.weight")]
                module_bases.setdefault(base, set()).add("B")

        if a == 0 or b == 0:
            raise ValueError(f"LoRA bundle missing A/B weights: {a=} {b=}")

        unpaired = [base for base, kinds in module_bases.items() if kinds != {"A", "B"}]
        if unpaired:
            raise ValueError(f"LoRA bundle contains unpaired A/B weights (sample): {unpaired[:5]}")

        if layer_compatible == 0:
            # SGLang currently infers layer ids from `layers.<idx>.` in weight names.
            raise ValueError(
                "LoRA bundle tensor keys contain no `layers.<idx>.` segments; SGLang will treat them as non-layer weights."
            )

        # Config sanity: require keys used by both PEFT and SGLang.
        cfg = self.config_dict
        if not isinstance(cfg, dict):
            raise ValueError("LoRA bundle config_dict is not a dict.")

        peft_type = cfg.get("peft_type")
        if not isinstance(peft_type, str) or peft_type.lower() != "lora":
            raise ValueError(f"LoRA bundle config_dict.peft_type must be 'LORA' (got {peft_type!r}).")

        for k in ("r", "lora_alpha", "target_modules"):
            if k not in cfg:
                raise ValueError(f"LoRA bundle config_dict missing required key: {k}")

        if not isinstance(cfg.get("target_modules"), list):
            raise ValueError("LoRA bundle config_dict.target_modules must be a list.")

    def save_pretrained(self, save_directory: str, *, tokenizer: Any | None = None) -> None:
        self.assert_valid()

        os.makedirs(save_directory, exist_ok=True)

        # Ensure PEFT-style config includes base model information.
        cfg = dict(self.config_dict)
        cfg.setdefault("base_model_name_or_path", self.base_model_name_or_path)
        cfg.setdefault("inference_mode", True)

        with open(os.path.join(save_directory, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
            f.write("\n")

        # Save tensors to disk (prefer safetensors).
        cpu_state: dict[str, torch.Tensor] = {
            k: v.detach().to("cpu").contiguous() for k, v in self.tensors.items()
        }

        try:
            from safetensors.torch import save_file as safetensors_save_file  # type: ignore[import-not-found]

            safetensors_save_file(
                cpu_state,
                os.path.join(save_directory, "adapter_model.safetensors"),
            )
        except Exception:
            torch.save(cpu_state, os.path.join(save_directory, "adapter_model.bin"))

        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)

