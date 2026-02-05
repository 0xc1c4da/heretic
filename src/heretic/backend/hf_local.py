from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear
from torch import FloatTensor, Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.generation import GenerateDecoderOnlyOutput  # ty:ignore[possibly-missing-import]

from ..config import QuantizationMethod, RowNormalization, Settings
from ..utils import empty_cache
from .base import (
    BackendMetadata,
    HereticBackend,
    ModuleRef,
    ResidualCaptureResult,
    ScoreResult,
    VTWResult,
)


def _get_model_class(
    model: str,
) -> type[AutoModelForImageTextToText] | type[AutoModelForCausalLM]:
    configs = PretrainedConfig.get_config_dict(model)
    if any(["vision_config" in x for x in configs]):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def _left_pad(input_ids_batch: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x in input_ids_batch)
    batch = []
    mask = []
    for ids in input_ids_batch:
        pad_len = max_len - len(ids)
        batch.append([pad_id] * pad_len + ids)
        mask.append([0] * pad_len + [1] * len(ids))
    return (
        torch.tensor(batch, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
    )


@dataclass
class _HFLocalState:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    peft_config: LoraConfig
    trusted_models: dict[str, bool | None]


class HFLocalBackend(HereticBackend):
    """In-process HuggingFace/Transformers backend (current Heretic behavior)."""

    def __init__(self, settings: Settings):
        self.settings = settings

        tokenizer = AutoTokenizer.from_pretrained(
            settings.model,
            trust_remote_code=settings.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
            else None
        )

        trusted_models: dict[str, bool | None] = {settings.model: settings.trust_remote_code}
        if settings.evaluate_model is not None:
            trusted_models[settings.evaluate_model] = settings.trust_remote_code

        model: PreTrainedModel | None = None
        chosen_dtype: torch.dtype | None = None

        for dtype_str in settings.dtypes:
            try:
                quantization_config = self._get_quantization_config(dtype_str)
                extra_kwargs: dict[str, Any] = {}
                if quantization_config is not None:
                    extra_kwargs["quantization_config"] = quantization_config

                chosen_dtype = (
                    torch.bfloat16 if dtype_str == "auto" else getattr(torch, dtype_str)
                )

                model = _get_model_class(settings.model).from_pretrained(
                    settings.model,
                    dtype=dtype_str,
                    device_map=settings.device_map,
                    max_memory=max_memory,
                    trust_remote_code=trusted_models.get(settings.model),
                    **extra_kwargs,
                )

                # Accept trust_remote_code if it was implicitly prompted/accepted.
                if trusted_models.get(settings.model) is None:
                    trusted_models[settings.model] = True

                # Smoke test: one-token greedy generate
                input_ids = tokenizer.encode("What is 1+1?", add_special_tokens=True)
                ids, attn = _left_pad([input_ids], tokenizer.pad_token_id)
                model.generate(
                    input_ids=ids.to(model.device),
                    attention_mask=attn.to(model.device),
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                break
            except Exception:
                model = None
                empty_cache()
                continue

        if model is None or chosen_dtype is None:
            raise RuntimeError("Failed to load model with all configured dtypes.")

        peft_config = self._build_peft_config(model=cast(PreTrainedModel, model))
        peft_model = cast(PeftModel, get_peft_model(cast(PreTrainedModel, model), peft_config))

        self._state = _HFLocalState(
            model=peft_model,
            tokenizer=tokenizer,
            peft_config=peft_config,
            trusted_models=trusted_models,
        )

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | None:
        if self.settings.quantization != QuantizationMethod.BNB_4BIT:
            return None
        compute_dtype = torch.bfloat16 if dtype == "auto" else getattr(torch, dtype)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def _build_peft_config(self, model: PreTrainedModel) -> LoraConfig:
        # Keep behavior aligned with current `Model._apply_lora()`.
        # Targets are leaf names (e.g. "o_proj", "down_proj").
        layer0 = self._get_layers(model)[0]
        target_modules = []
        if hasattr(layer0, "self_attn") and hasattr(layer0.self_attn, "o_proj"):
            target_modules.append("o_proj")
        target_modules.append("down_proj")

        if self.settings.row_normalization != RowNormalization.FULL:
            lora_rank = 1
        else:
            lora_rank = self.settings.full_normalization_lora_rank

        return LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def _get_layers(self, model: PreTrainedModel | PeftModel):
        # Unwrap PeftModel
        base = model.base_model.model if isinstance(model, PeftModel) else model
        with torch.no_grad():
            # Multimodal models
            if hasattr(base, "model") and hasattr(base.model, "language_model"):
                return base.model.language_model.layers
        return base.model.layers

    def get_metadata(self) -> BackendMetadata:
        model = self._state.model
        # Unwrap PeftModel for metadata.
        base = model.base_model.model if isinstance(model, PeftModel) else model
        model_id = getattr(base.config, "name_or_path", self.settings.model)
        return BackendMetadata(
            backend_name="hf_local",
            backend_version=None,
            model_id=model_id,
            tokenizer_id=getattr(self._state.tokenizer, "name_or_path", None),
            max_context_len=getattr(base.config, "max_position_embeddings", None),
            supports={
                "input_ids": True,
                "logprobs_full": True,
                "hidden_states_hf": True,
                "lora_inprocess": True,
                "compute_vtw": True,
            },
        )

    def score(self, input_ids_batch: list[list[int]], *, adapter: str | None = None) -> ScoreResult:
        if adapter is not None:
            raise NotImplementedError("HFLocalBackend adapter selection is not implemented yet.")

        tokenizer = self._state.tokenizer
        model = self._state.model

        input_ids, attention_mask = _left_pad(input_ids_batch, tokenizer.pad_token_id)
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        outputs = model.generate(  # ty:ignore[call-non-callable]
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        outputs = cast(GenerateDecoderOnlyOutput, outputs)
        logits = cast(tuple[FloatTensor], outputs.scores)[0]
        logprobs = F.log_softmax(logits, dim=-1)
        return ScoreResult(logprobs_full=logprobs, meta=None)

    def capture_residuals(
        self,
        input_ids_batch: list[list[int]],
        *,
        capture_layers: list[int],
        capture_point: str = "block_input_last_token",
        adapter: str | None = None,
    ) -> ResidualCaptureResult:
        if adapter is not None:
            raise NotImplementedError("HFLocalBackend adapter selection is not implemented yet.")
        if capture_point != "block_input_last_token":
            raise NotImplementedError(f"Unsupported capture_point: {capture_point}")

        tokenizer = self._state.tokenizer
        model = self._state.model

        input_ids, attention_mask = _left_pad(input_ids_batch, tokenizer.pad_token_id)
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        # Standardized contract: "block_input_last_token" for the *prompt* token.
        # Use a forward pass (not generation) so we capture prompt hidden states.
        outputs = model(  # ty:ignore[operator]
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = cast(tuple[FloatTensor], outputs.hidden_states)

        residuals_all = torch.stack([hs[:, -1, :] for hs in hidden_states], dim=1).to(
            torch.float32
        )

        # Select requested layers (caller uses 0-based indices into this stack).
        residuals = residuals_all[:, capture_layers, :]
        return ResidualCaptureResult(
            residuals=residuals,
            captured_layers=capture_layers,
            capture_point=capture_point,
            meta=None,
        )

    def compute_vtw(
        self,
        v: torch.Tensor,
        *,
        target: ModuleRef,
        adapter: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> VTWResult:
        if adapter is not None:
            raise NotImplementedError("HFLocalBackend adapter selection is not implemented yet.")
        model = self._state.model
        params = dict(model.named_parameters())
        if target.module_path not in params:
            raise KeyError(f"Unknown parameter: {target.module_path}")

        W = params[target.module_path]
        if W.ndim != 2:
            W = W.view(W.shape[0], -1)
        v = v.to(W.device, dtype=dtype)
        vtw = v @ W.to(dtype)
        return VTWResult(target=target, vtw=vtw, implementation="dense_matmul")

    def load_adapter(self, *, name: str, tensors: dict[str, torch.Tensor], config: dict) -> None:
        raise NotImplementedError("Adapter hot-swap is implemented via SGLang backend.")

    def unload_adapter(self, *, name: str) -> None:
        raise NotImplementedError("Adapter hot-swap is implemented via SGLang backend.")

