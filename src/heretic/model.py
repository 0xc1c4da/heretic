# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Type, cast

import bitsandbytes as bnb
import torch
import torch.linalg as LA
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,  # ty:ignore[possibly-missing-import]
)

from .config import BackendType, QuantizationMethod, RowNormalization, Settings
from .backend.hf_local import HFLocalBackend
from .backend.base import HereticBackend
from .backend.sglang import SGLangBackend
from .backend.sglang_offline import SGLangOfflineBackend
from .utils import Prompt, batchify, empty_cache, print, sha256_token_ids


def get_model_class(
    model: str,
) -> Type[AutoModelForImageTextToText] | Type[AutoModelForCausalLM]:
    configs = PretrainedConfig.get_config_dict(model)

    if any([("vision_config" in config) for config in configs]):
        return AutoModelForImageTextToText
    else:
        return AutoModelForCausalLM


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class Model:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    peft_config: LoraConfig
    backend: HereticBackend

    def __init__(self, settings: Settings):
        self.settings = settings
        self.response_prefix = ""
        self.needs_reload = False

        print()
        backend_type = getattr(settings, "backend", BackendType.LOCAL)
        self._backend_type = backend_type
        self._num_layers: int | None = None

        if backend_type == BackendType.LOCAL:
            # Local execution: delegate model loading + quantization + LoRA init to HFLocalBackend.
            self.backend = HFLocalBackend(settings)
            # Expose underlying model/tokenizer to keep the rest of the code working.
            self.model = self.backend._state.model  # ty:ignore[protected-access]
            self.tokenizer = self.backend._state.tokenizer  # ty:ignore[protected-access]
            self.peft_config = self.backend._state.peft_config  # ty:ignore[protected-access]
            # Retain these legacy fields for merge/reload paths.
            self.trusted_models = self.backend._state.trusted_models  # ty:ignore[protected-access]
            self.max_memory = (
                {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
                if settings.max_memory
                else None
            )

            print(f"Loaded local backend for [bold]{settings.model}[/].")
            print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
            print("* Abliterable components:")
            for component, modules in self.get_layer_modules(0).items():
                print(
                    f"  * [bold]{component}[/]: [bold]{len(modules)}[/] modules per layer"
                )
        elif backend_type == BackendType.SGLANG:
            # Remote execution: do NOT load HF weights. Only keep tokenizer + config locally.
            self.backend = SGLangBackend(
                base_url=settings.sglang_url,
                admin_url=settings.sglang_admin_url,
                model=settings.model,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.model,
                trust_remote_code=settings.trust_remote_code,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            cfg = PretrainedConfig.from_pretrained(
                settings.model, trust_remote_code=settings.trust_remote_code
            )
            self._num_layers = cast(
                int | None,
                getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None),
            )

            # Legacy attributes are unused in remote mode but expected to exist.
            self.model = cast(Any, None)
            self.peft_config = cast(Any, None)
            self.trusted_models = {settings.model: settings.trust_remote_code}
            self.max_memory = None

            print(f"Loaded SGLang backend for [bold]{settings.model}[/].")
            if self._num_layers is not None:
                print(f"* Transformer model with [bold]{self._num_layers}[/] layers (from config)")
        elif backend_type == BackendType.SGLANG_OFFLINE:
            # Embedded execution: do NOT load HF weights. SGLang Engine runs in-process.
            self.backend = SGLangOfflineBackend(
                model_path=settings.model,
                trust_remote_code=bool(settings.trust_remote_code),
                engine_args=getattr(settings, "sglang_offline_args", None),
            )

            # Keep a local tokenizer for prompt building / hashing (can be pushed server-side later).
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.model,
                trust_remote_code=settings.trust_remote_code,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            # Prefer backend-reported metadata.
            self._num_layers = self.backend.get_metadata().num_layers

            # Legacy attributes are unused in SGLang mode but expected to exist.
            self.model = cast(Any, None)
            self.peft_config = cast(Any, None)
            self.trusted_models = {settings.model: settings.trust_remote_code}
            self.max_memory = None

            print(f"Loaded SGLang offline backend for [bold]{settings.model}[/].")
            if self._num_layers is not None:
                print(f"* Transformer model with [bold]{self._num_layers}[/] layers (from backend)")
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    def _apply_lora(self):
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PreTrainedModel)

        # Always use LoRA adapters for abliteration (faster reload, no weight modification).
        # We use the leaf names (e.g. "o_proj") as target modules.
        # This may cause LoRA adapters to be attached to unrelated modules (e.g. "conv.o_proj"),
        # but this is harmless as we only abliterate the modules we target in `abliterate()`,
        # leaving the others at their default (identity) state.
        # NOTE: This will need to be updated when hybrid layer support (#43) is merged.
        target_modules = [
            comp.split(".")[-1] for comp in self.get_abliterable_components()
        ]

        if self.settings.row_normalization != RowNormalization.FULL:
            # Rank 1 is sufficient for directional ablation without renormalization.
            lora_rank = 1
        else:
            # Row magnitude preservation introduces nonlinear effects.
            lora_rank = self.settings.full_normalization_lora_rank

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,  # Apply adapter at full strength.
            lora_dropout=0,
            bias="none",
            # Even if we're using AutoModelForImageTextToText, this is still correct,
            # as VL models are typically just causal LMs with an added image encoder.
            task_type="CAUSAL_LM",
        )

        # self.peft_config is a LoraConfig object rather than a dictionary,
        # so the result is a PeftModel rather than a PeftMixedModel.
        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))

        print(f"* LoRA adapters initialized (targets: {', '.join(target_modules)})")

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | None:
        """
        Creates quantization config based on settings.

        Args:
            dtype: The dtype string (e.g., "auto", "bfloat16")

        Returns:
            BitsAndBytesConfig or None
        """
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # BitsAndBytesConfig expects a torch.dtype, not a string.
            if dtype == "auto":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = getattr(torch, dtype)

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return None

    def get_merged_model(self) -> PreTrainedModel:
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PeftModel)

        # Check if we need special handling for quantized models
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # Quantized models need special handling - we must reload the base model
            # in full precision to merge the LoRA adapters

            # Get the adapter state dict before we do anything
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.clone().cpu()

            # Load base model in full precision on CPU to avoid VRAM issues
            print("* Loading base model on CPU (this may take a while)...")
            base_model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=self.trusted_models.get(self.settings.model),
            )

            # Apply LoRA adapters to the CPU model
            print("* Applying LoRA adapters...")
            peft_model = get_peft_model(base_model, self.peft_config)

            # Copy the trained adapter weights
            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            # Merge and unload
            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            return merged_model
        else:
            # Non-quantized model - can merge directly
            print("* Merging LoRA adapters into base model...")
            merged_model = self.model.merge_and_unload()
            # merge_and_unload() modifies self.model in-place, destroying LoRA adapters.
            # Mark for full reload if user switches trials later.
            self.needs_reload = True
            return merged_model

    def reset_model(self):
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behavior:
        - Fast path: If the same model is loaded and doesn't need full reload,
          resets LoRA adapter weights to zero (identity transformation).
        - Slow path: If switching models or after merge_and_unload(),
          performs full model reload with quantization config.
        """
        if self._backend_type != BackendType.LOCAL:
            # Remote backends hold no in-process weights.
            return

        current_model = getattr(self.model.config, "name_or_path", None)
        if current_model == self.settings.model and not self.needs_reload:
            # Reset LoRA adapters to zero (identity transformation)
            for name, module in self.model.named_modules():
                if "lora_B" in name and hasattr(module, "weight"):
                    torch.nn.init.zeros_(module.weight)
            return

        # Slow path: rebuild the backend (e.g. after merge_and_unload()).
        self.backend = HFLocalBackend(self.settings)
        self.model = self.backend._state.model  # ty:ignore[protected-access]
        self.tokenizer = self.backend._state.tokenizer  # ty:ignore[protected-access]
        self.peft_config = self.backend._state.peft_config  # ty:ignore[protected-access]
        self.trusted_models = self.backend._state.trusted_models  # ty:ignore[protected-access]
        self.needs_reload = False

    def get_layers(self) -> ModuleList:
        if self._backend_type != BackendType.LOCAL:
            if self._num_layers is None:
                raise RuntimeError("Remote backend layer count is unknown (missing config field).")
            # Return a synthetic module list so callers can take len(...).
            return ModuleList([torch.nn.Identity() for _ in range(self._num_layers)])

        model = self.model

        # Unwrap PeftModel (always true after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers

        # Text-only models.
        return model.model.layers

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
        if self._backend_type != BackendType.LOCAL:
            raise NotImplementedError("Layer module introspection is only available for local backend.")

        layer = self.get_layers()[layer_index]

        modules = {}

        def try_add(component: str, module: Any):
            # Only add if it's a proper nn.Module (PEFT can wrap these with LoRA)
            if isinstance(module, Module):
                if component not in modules:
                    modules[component] = []
                modules[component].append(module)
            else:
                # Assert for unexpected types (catches architecture changes)
                assert not isinstance(module, Tensor), (
                    f"Unexpected Tensor in {component} - expected nn.Module"
                )

        # Exceptions aren't suppressed here, because there is currently
        # no alternative location for the attention out-projection.
        try_add("attn.o_proj", layer.self_attn.o_proj)  # ty:ignore[possibly-missing-attribute]

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.w2)  # ty:ignore[possibly-missing-attribute]

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear)  # ty:ignore[possibly-missing-attribute]

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.output_linear)  # ty:ignore[possibly-missing-attribute]

        # We need at least one module across all components for abliteration to work.
        total_modules = sum(len(mods) for mods in modules.values())
        assert total_modules > 0, "No abliterable modules found in layer"

        return modules

    def get_abliterable_components(self) -> list[str]:
        if self._backend_type != BackendType.LOCAL:
            # Keep component names stable for downstream logic; remote module selection uses /heretic/module_map.
            return ["attn.o_proj", "mlp.down_proj"]
        return list(self.get_layer_modules(0).keys())

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        *,
        export_tensors: bool = False,
    ) -> dict[str, Tensor] | None:
        if self._backend_type in (BackendType.SGLANG, BackendType.SGLANG_OFFLINE):
            if not export_tensors:
                raise ValueError(
                    "backend='sglang' requires export_tensors=True (we load adapters into the backend)."
                )

            backend = cast(Any, self.backend)

            # Build a minimal LoRA config dict compatible with SGLang's LoRAConfig.
            target_modules = ["o_proj", "down_proj"]
            if self.settings.row_normalization == RowNormalization.FULL:
                lora_rank = self.settings.full_normalization_lora_rank
            else:
                lora_rank = 1

            self.peft_config = LoraConfig(
                r=lora_rank,
                target_modules=target_modules,
                lora_alpha=lora_rank,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Select which direction to use.
            if direction_index is None:
                refusal_direction = None
            else:
                frac, idx = math.modf(direction_index)
                idx_i = int(idx)
                refusal_direction = F.normalize(
                    refusal_directions[idx_i].lerp(refusal_directions[idx_i + 1], frac),
                    p=2,
                    dim=0,
                )

            # Fetch canonical module paths from backend and group by component + layer.
            module_descs = backend.module_map(
                include_projs=["o_proj", "down_proj"],
                # Default: exclude MoE experts to keep adapter sizes tractable.
                include_experts=[],
            )
            by_layer_component: dict[tuple[int, str], list[str]] = {}
            for d in module_descs:
                layer = d.get("layer")
                proj = d.get("proj")
                path = d.get("module_path")
                if not isinstance(layer, int) or not isinstance(proj, str) or not isinstance(path, str):
                    continue
                if proj == "o_proj":
                    comp = "attn.o_proj"
                elif proj == "down_proj":
                    comp = "mlp.down_proj"
                else:
                    continue
                by_layer_component.setdefault((layer, comp), []).append(path)

            exported: dict[str, Tensor] = {}

            # Precompute v^T W for all rank-1 modules in one batch call when possible.
            vtw_by_name: dict[str, list[float]] = {}
            if self.settings.row_normalization != RowNormalization.FULL:
                items: list[dict[str, Any]] = []
                for (layer_index, comp), paths in by_layer_component.items():
                    params = parameters[comp]
                    distance = abs(layer_index - params.max_weight_position)
                    if distance > params.min_weight_distance:
                        continue
                    if refusal_direction is None:
                        v_vec = refusal_directions[layer_index]
                    else:
                        v_vec = refusal_direction
                    for p in paths:
                        items.append(
                            {
                                "name": p,
                                "v": v_vec.detach().to(torch.float32).cpu().tolist(),
                                "dtype": "float32",
                            }
                        )
                if items:
                    results = backend.compute_vtw_batch(items=items)
                    for r in results:
                        name = r.get("name")
                        vtw = r.get("vtw")
                        if isinstance(name, str) and isinstance(vtw, list):
                            vtw_by_name[name] = vtw

            for (layer_index, comp), paths in by_layer_component.items():
                params = parameters[comp]
                distance = abs(layer_index - params.max_weight_position)
                if distance > params.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight over min_weight_distance.
                w = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                if refusal_direction is None:
                    v_vec = refusal_directions[layer_index]
                else:
                    v_vec = refusal_direction

                v_vec = v_vec.to(torch.float32)

                for p in paths:
                    module_base = p[: -len(".weight")] if p.endswith(".weight") else p

                    if self.settings.row_normalization == RowNormalization.FULL:
                        A, B = backend.build_full_rownorm_lora(
                            name=p,
                            v=v_vec,
                            weight=float(w),
                            rank=int(self.settings.full_normalization_lora_rank),
                            out_dtype="float16",
                        )
                    else:
                        vtw = vtw_by_name.get(p)
                        if vtw is None:
                            raise RuntimeError(f"Missing v^T W for module {p}")
                        A = torch.tensor(vtw, dtype=torch.float32).view(1, -1)
                        B = (-float(w) * v_vec).view(-1, 1)

                    # SGLang expects weights keyed by strings that include "layers.<idx>." and "lora_A"/"lora_B".
                    exported[f"{module_base}.lora_A.weight"] = A.to(torch.float16).cpu()
                    exported[f"{module_base}.lora_B.weight"] = B.to(torch.float16).cpu()

            return exported

        if self._backend_type != BackendType.LOCAL:
            raise NotImplementedError(
                "Remote backend support is only implemented for backend='sglang'."
            )

        exported: dict[str, Tensor] | None = {} if export_tensors else None

        module_name_by_id = None
        if export_tensors:
            module_name_by_id = {id(m): n for n, m in self.model.named_modules()}

        if direction_index is None:
            refusal_direction = None
        else:
            weight, index = math.modf(direction_index)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                params = parameters[component]

                # Type inference fails here for some reason.
                distance = cast(float, abs(layer_index - params.max_weight_position))

                # Don't orthogonalize layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > params.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance.
                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                if refusal_direction is None:
                    layer_refusal_direction = refusal_directions[layer_index]
                else:
                    layer_refusal_direction = refusal_direction

                for module in modules:
                    # FIXME: This cast is potentially invalid, because the program logic
                    #        does not guarantee that the module is of type Linear, and in fact
                    #        the retrieved modules might not conform to the interface assumed
                    #        below (though they do in practice). However, this is difficult
                    #        to fix cleanly, because get_layer_modules is called twice on
                    #        different model configurations, and PEFT employs different
                    #        module types depending on the chosen quantization.
                    module = cast(Linear, module)

                    # LoRA abliteration: delta W = -lambda * v * (v^T W)
                    # lora_B = -lambda * v
                    # lora_A = v^T W

                    # Use the FP32 refusal direction directly (no downcast/upcast)
                    # and move to the correct device.
                    v = layer_refusal_direction.to(module.weight.device)

                    # Get W (dequantize if necessary).
                    #
                    # FIXME: This cast is valid only under the assumption that the original
                    #        module wrapped by the LoRA adapter has a weight attribute.
                    #        See the comment above for why this is currently not guaranteed.
                    base_weight = cast(Tensor, module.base_layer.weight)
                    quant_state = getattr(base_weight, "quant_state", None)

                    if quant_state is None:
                        W = base_weight.to(torch.float32)
                    else:
                        # 4-bit quantization.
                        # This cast is always valid. Type inference fails here because the
                        # bnb.functional module is not found by ty for some reason.
                        W = cast(
                            Tensor,
                            bnb.functional.dequantize_4bit(  # ty:ignore[possibly-missing-attribute]
                                base_weight.data,
                                quant_state,
                            ).to(torch.float32),
                        )

                    # Flatten weight matrix to (out_features, in_features).
                    W = W.view(W.shape[0], -1)

                    if self.settings.row_normalization != RowNormalization.NONE:
                        # Keep a reference to the original weight matrix so we can subtract it later.
                        W_org = W
                        # Get the row norms.
                        W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                        # Normalize the weight matrix along the rows.
                        W = F.normalize(W, p=2, dim=1)

                    # Calculate lora_A = v^T W
                    # v is (d_out,), W is (d_out, d_in)
                    # v @ W -> (d_in,)
                    lora_A = (v @ W).view(1, -1)

                    # Calculate lora_B = -weight * v
                    # v is (d_out,)
                    lora_B = (-weight * v).view(-1, 1)

                    if self.settings.row_normalization == RowNormalization.PRE:
                        # Make the LoRA adapter apply to the original weight matrix.
                        lora_B = W_row_norms * lora_B
                    elif self.settings.row_normalization == RowNormalization.FULL:
                        # Approximates https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
                        W = W + lora_B @ lora_A
                        # Normalize the adjusted weight matrix along the rows.
                        W = F.normalize(W, p=2, dim=1)
                        # Restore the original row norms of the weight matrix.
                        W = W * W_row_norms
                        # Subtract the original matrix to turn W into a delta.
                        W = W - W_org
                        # Use a low-rank SVD to get an approximation of the matrix.
                        r = self.peft_config.r
                        U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)
                        # Truncate it to the part we want to store in the LoRA adapter.
                        # Note: svd_lowrank actually returns V, so transpose it to get Vh.
                        U = U[:, :r]
                        S = S[:r]
                        Vh = Vh[:, :r].T
                        # Transfer it into the LoRA adapter components. Split the singular values
                        # evenly between the two components to keep their norms balanced and avoid
                        # potential issues with numerical stability.
                        sqrt_S = torch.sqrt(S)
                        lora_B = U @ torch.diag(sqrt_S)
                        lora_A = torch.diag(sqrt_S) @ Vh

                    # Assign to adapters. The adapter name is "default", because that's
                    # what PEFT uses when no name is explicitly specified, as above.
                    # These casts are therefore valid.
                    weight_A = cast(Tensor, module.lora_A["default"].weight)
                    weight_B = cast(Tensor, module.lora_B["default"].weight)
                    weight_A.data = lora_A.to(weight_A.dtype)
                    weight_B.data = lora_B.to(weight_B.dtype)

                    if exported is not None and module_name_by_id is not None:
                        module_name = module_name_by_id.get(id(module))
                        if module_name is None:
                            continue
                        exported[f"{module_name}.lora_A.default.weight"] = weight_A.detach().clone().cpu()
                        exported[f"{module_name}.lora_B.default.weight"] = weight_B.detach().clone().cpu()

        return exported

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        input_ids_batch = self.encode_prompts(prompts)
        _ = [sha256_token_ids(ids) for ids in input_ids_batch]
        inputs = self._pad_input_ids_batch(input_ids_batch)

        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]

        # This cast is valid because list[str] is the return type
        # for batched operation with tokenize=False.
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        if self.response_prefix:
            # Append the common response prefix to the prompts so that evaluation happens
            # at the point where responses start to differ for different prompts.
            chat_prompts = [prompt + self.response_prefix for prompt in chat_prompts]

        # Keep the original chat prompt construction for compatibility/debugging,
        # but the actual model invocation uses canonical token IDs (`inputs` above).

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

        return inputs, outputs

    def encode_prompts(self, prompts: list[Prompt]) -> list[list[int]]:
        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]

        ids = cast(
            list[list[int]],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=True,
            ),
        )

        if self.response_prefix:
            prefix_ids = cast(
                list[int],
                self.tokenizer(self.response_prefix, add_special_tokens=False)["input_ids"],
            )
            ids = [x + prefix_ids for x in ids]

        return ids

    def _pad_input_ids_batch(self, input_ids_batch: list[list[int]]) -> BatchEncoding:
        pad_id = self.tokenizer.pad_token_id
        assert pad_id is not None

        max_len = max(len(x) for x in input_ids_batch)
        padded = []
        attn = []
        for ids in input_ids_batch:
            pad_len = max_len - len(ids)
            padded.append([pad_id] * pad_len + ids)
            attn.append([0] * pad_len + [1] * len(ids))

        return BatchEncoding(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long, device=self.model.device),
                "attention_mask": torch.tensor(attn, dtype=torch.long, device=self.model.device),
            }
        )

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
        *,
        adapter: str | None = None,
    ) -> list[str]:
        if self._backend_type in (BackendType.SGLANG, BackendType.SGLANG_OFFLINE):
            # SGLang generation (remote HTTP or embedded offline).
            input_ids_batch = self.encode_prompts(prompts)
            return self.backend.generate_text(
                input_ids_batch,
                max_new_tokens=self.settings.max_response_length,
                adapter=adapter,
            )

        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        return self.tokenizer.batch_decode(
            # Extract the newly generated part.
            # This cast is valid because the input_ids property is a Tensor
            # if the tokenizer is invoked with return_tensors="pt", as above.
            outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
        *,
        adapter: str | None = None,
    ) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(
                batch,
                skip_special_tokens=skip_special_tokens,
                adapter=adapter,
            ):
                responses.append(response)

        return responses

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        """Return residual-stream vectors under the standardized contract.

        Contract: `block_input_last_token`
        - Use last token of the *prompt* (not generated token).
        - Do NOT include embeddings stream.

        Shape: (batch, layers, d_model)
        """

        input_ids_batch = self.encode_prompts(prompts)

        if self._backend_type in (BackendType.SGLANG, BackendType.SGLANG_OFFLINE):
            # Delegate capture to backend. We request all layers when available.
            if self._num_layers is None:
                # Prefer server-reported metadata when available.
                try:
                    meta = self.backend.get_metadata()
                    if isinstance(meta.num_layers, int) and meta.num_layers > 0:
                        self._num_layers = int(meta.num_layers)
                        print(
                            f"* Inferred [bold]{self._num_layers}[/] layers from SGLang /heretic/metadata"
                        )
                except Exception:
                    pass

            if self._num_layers is None:
                # Try to infer layer count from the remote module map.
                # This avoids relying on local HF config fields, which may be missing in some
                # on-disk model snapshots used purely for tokenizer/config.
                try:
                    sgl_backend = cast(Any, self.backend)
                    descs: Any = sgl_backend.module_map(
                        include_projs=["o_proj", "down_proj"],
                        # Default: exclude MoE experts for metadata inference.
                        include_experts=[],
                    )
                    # Defensive: in some multi-DP configurations, the server can return a
                    # list-of-lists; the SGLang backend normalizes this, but keep this robust.
                    if isinstance(descs, list) and descs and isinstance(descs[0], list):
                        descs = descs[0]

                    layers: list[int] = []
                    if isinstance(descs, list):
                        for d in descs:
                            if not isinstance(d, dict):
                                continue
                            layer = d.get("layer")
                            if isinstance(layer, int):
                                layers.append(layer)
                                continue
                            # Fallback: parse from module_path if server didn't populate `layer`.
                            mp = d.get("module_path")
                            if isinstance(mp, str):
                                import re

                                m = re.search(r"\.layers\.(\d+)\.", mp)
                                if m:
                                    layers.append(int(m.group(1)))
                    if layers:
                        self._num_layers = int(max(layers) + 1)
                        print(
                            f"* Inferred [bold]{self._num_layers}[/] layers from SGLang /heretic/module_map"
                        )
                except Exception as e:
                    raise RuntimeError(
                        "Cannot infer number of layers for SGLang residual capture (missing config field and module_map inference failed)."
                    ) from e

            if self._num_layers is None:
                raise RuntimeError(
                    "Cannot infer number of layers for SGLang residual capture (missing config field)."
                )
            capture_layers = list(range(self._num_layers))
            out = self.backend.capture_residuals(
                input_ids_batch,
                capture_layers=capture_layers,
                capture_point="block_input_last_token",
                adapter=None,
            )
            residuals = out.residuals
        else:
            inputs = self._pad_input_ids_batch(input_ids_batch)

            outputs = self.model(  # ty:ignore[operator]
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = cast(tuple[FloatTensor], outputs.hidden_states)

            # hidden_states[0] is embeddings; drop it to match contract.
            residuals = torch.stack([hs[:, -1, :] for hs in hidden_states[1:]], dim=1)

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        residuals = residuals.to(torch.float32)

        if 0 <= self.settings.winsorization_quantile < 1:
            # Apply symmetric winsorization to each layer of the per-prompt residuals.
            abs_residuals = torch.abs(residuals)
            # Get the (prompt, layer, 1) quantiles of the (prompt, layer, component) residuals.
            thresholds = torch.quantile(
                abs_residuals,
                self.settings.winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            return torch.clamp(residuals, -thresholds, thresholds)

        return residuals

    def get_residuals_batched(self, prompts: list[Prompt]) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch))

        return torch.cat(residuals, dim=0)

    # We work with logprobs rather than probabilities for numerical stability
    # when computing the KL divergence.
    def get_logprobs(self, prompts: list[Prompt]) -> Tensor:
        input_ids_batch = self.encode_prompts(prompts)
        if self._backend_type in (BackendType.SGLANG, BackendType.SGLANG_OFFLINE):
            scored = self.backend.score(input_ids_batch, adapter=None)
            if scored.logprobs_full is None:
                raise RuntimeError("SGLang backend did not return full-vocab logprobs.")
            return scored.logprobs_full

        # Local path: generate one token and return logprobs over vocab at that position.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

        outputs = cast(GenerateDecoderOnlyOutput, outputs)
        logits = cast(tuple[FloatTensor], outputs.scores)[0]
        return F.log_softmax(logits, dim=-1)

    def get_logprobs_batched(self, prompts: list[Prompt]) -> Tensor:
        logprobs = []

        for batch in batchify(prompts, self.settings.batch_size):
            logprobs.append(self.get_logprobs(batch))

        return torch.cat(logprobs, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        # This cast is valid because str is the return type
        # for single-chat operation with tokenize=False.
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            # The TextStreamer constructor annotates this parameter with the AutoTokenizer
            # type, which makes no sense because AutoTokenizer is a factory class,
            # not a base class that tokenizers inherit from.
            self.tokenizer,  # ty:ignore[invalid-argument-type]
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )  # ty:ignore[call-non-callable]

        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
