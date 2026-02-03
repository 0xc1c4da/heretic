# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import json
import math
import os
import copy
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Type, cast

import bitsandbytes as bnb
import torch
import torch.linalg as LA
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from transformers import (
    AutoConfig,
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

from .config import QuantizationMethod, RowNormalization, Settings
from .mock_models import (
    TinyKimiK25Spec,
    TinyMiniMaxM2Spec,
    looks_like_kimi_k25_source_dir,
    looks_like_minimax_m2_source_dir,
    materialize_tiny_kimi_k25_repo,
    materialize_tiny_minimax_m2_repo,
)
from .runtime.precision import PrecisionApplier, PrecisionPolicy
from .runtime.quantization import QuantizationInfo, QuantizationRequest, resolve_quantization
from .runtime.transformers_compat import ensure_compressed_tensors_fast_load
from .runtime.weight_access import WeightAccess, WeightAccessError
from .utils import Prompt, batchify, empty_cache, print


def _patch_kimi_remote_code_in_memory(*, model_id: str) -> None:
    """
    Minimal empirical shims for Kimi K2.5 remote code.

    Kimi's model class always instantiates the vision tower during __init__ (even for text-only use),
    so we patch remote-code init bugs to allow loading.
    """

    if not isinstance(model_id, str) or "Kimi-K2.5" not in model_id:
        return

    try:
        # IMPORTANT: use transformers' dynamic module loader so it sets
        # `__transformers_module_hash__`. Otherwise, `from_config()` will reload the module
        # and wipe our in-memory patch.
        from transformers.dynamic_module_utils import get_cached_module_file, get_class_in_module

        module_path = get_cached_module_file(
            model_id,
            "modeling_kimi_k25.py",
            revision=None,
        )
        MoonViT3dEncoder = get_class_in_module("MoonViT3dEncoder", module_path)
        if not hasattr(MoonViT3dEncoder, "use_deterministic_attn"):
            setattr(MoonViT3dEncoder, "use_deterministic_attn", False)
            print("* Patched Kimi remote-code: MoonViT3dEncoder.use_deterministic_attn=False")
    except Exception as exc:
        print(f"[yellow]Kimi remote-code patch skipped[/] ({exc})")


def get_model_class(
    model: str,
) -> Type[AutoModelForImageTextToText] | Type[AutoModelForCausalLM]:
    config_dict, _ = PretrainedConfig.get_config_dict(model)

    # If the config provides an `auto_map`, we must use an auto class that honors it.
    # `AutoModelForImageTextToText` does not accept unknown remote-code config classes.
    auto_map = config_dict.get("auto_map") if isinstance(config_dict, dict) else None
    if isinstance(auto_map, dict) and (
        "AutoModelForCausalLM" in auto_map or "AutoModel" in auto_map
    ):
        return AutoModelForCausalLM

    if "vision_config" in config_dict:
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

    def __init__(self, settings: Settings):
        self.settings = settings
        self.response_prefix = ""
        self.needs_reload = False
        self.quant: QuantizationInfo | None = None
        self.compute_dtype: torch.dtype | None = None
        self.lora_rank: int = None
        self.precision_policy = PrecisionPolicy.from_settings(settings)

        print()
        self._maybe_materialize_tiny_checkpoint()
        print(f"Loading model [bold]{self.settings.model}[/]...")

        # Optional: speed up loading of pre-compressed compressed-tensors checkpoints.
        ensure_compressed_tensors_fast_load(
            print, enabled=bool(getattr(self.settings, "ct_fast_load", False))
        )

        # Empirical shims for Kimi remote code (text-only still instantiates vision tower).
        _patch_kimi_remote_code_in_memory(model_id=self.settings.model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model,
            trust_remote_code=settings.trust_remote_code,
        )
        self._ensure_chat_template()

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        #           Right-padding causes empty outputs because the model sees PAD tokens
        #           after the prompt and thinks the sequence is complete.
        self.tokenizer.padding_side = "left"

        self.model = None  # ty:ignore[invalid-assignment]
        self.max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
            else None
        )
        self.trusted_models = {self.settings.model: settings.trust_remote_code}

        if self.settings.evaluate_model is not None:
            self.trusted_models[self.settings.evaluate_model] = settings.trust_remote_code

        # Detect the device we'll be using for runtime precision probing.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._print_loading_info = bool(
            getattr(self.settings, "ct_loading_info", False)
            or getattr(self.settings, "ct_fast_load", False)
        )

        # Empirical fix for first observed Kimi failure:
        # the vision tower defaults to FlashAttention2, but flash_attn may not be installed.
        # If flash_attn is unavailable, force eager attention via an explicit config.
        self._load_config: PretrainedConfig | None = None
        if isinstance(self.settings.model, str) and "Kimi-K2.5" in self.settings.model:
            try:
                from transformers.utils import is_flash_attn_2_available

                if not is_flash_attn_2_available():
                    cfg = AutoConfig.from_pretrained(
                        self.settings.model,
                        trust_remote_code=self.trusted_models.get(self.settings.model),
                    )
                    vc = getattr(cfg, "vision_config", None)
                    if vc is not None and hasattr(vc, "_attn_implementation"):
                        setattr(vc, "_attn_implementation", "eager")
                    self._load_config = cfg
                    print("* flash_attn not available; forcing eager attention for Kimi K2.5 vision tower")
            except Exception as exc:
                print(f"[yellow]Kimi config override failed[/] ({exc})")

        for dtype in self.settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]... ", end="")

            try:
                # Resolve quantization + compute dtype consistently.
                quant_req = QuantizationRequest(
                    method=getattr(self.settings.quantization, "value", str(self.settings.quantization)),
                    requested_dtype=dtype,
                    config_type=getattr(self.settings, "quantization_config_type", None),
                    config_kwargs=getattr(self.settings, "quantization_kwargs", None),
                )
                self.quant = resolve_quantization(
                    model_id_or_path=self.settings.model,
                    request=quant_req,
                    precision_fallback_dtype=getattr(
                        self.settings, "precision_fallback_dtype", "auto"
                    ),
                )

                # Build kwargs, only include quantization_config if it's not None
                # (some models like gpt-oss have issues with explicit None)
                extra_kwargs = {}
                if self.quant.load_quantization_config is not None:
                    extra_kwargs["quantization_config"] = self.quant.load_quantization_config

                load_kwargs = {
                    "torch_dtype": self.precision_policy.resolve_model_dtype(dtype),
                    "device_map": self.settings.device_map,
                    "max_memory": self.max_memory,
                    "trust_remote_code": self.trusted_models.get(self.settings.model),
                    **extra_kwargs,
                }
                if self._load_config is not None:
                    load_kwargs["config"] = self._load_config
                if self._print_loading_info:
                    load_kwargs["output_loading_info"] = True

                # FP8 models can produce NaNs in batched generation when sequences have different
                # prompt lengths (mixed padding/attention lengths). This can manifest as repeated
                # low-id tokens (often "!") from greedy argmax over NaN logits.
                # Using a more conservative attention implementation avoids this edge case.
                if (
                    self.quant is not None
                    and self.quant.is_quantized
                    and (self.quant.method or "")
                    in {"fp8", "finegrained_fp8", "fine-grained-fp8"}
                ):
                    load_kwargs["attn_implementation"] = "eager"

                loaded = get_model_class(self.settings.model).from_pretrained(
                    self.settings.model,
                    **load_kwargs,
                )
                if self._print_loading_info and isinstance(loaded, tuple) and len(loaded) == 2:
                    self.model, loading_info = loaded
                    self._log_loading_info(loading_info)
                else:
                    self.model = loaded

                # If we reach this point and the model requires trust_remote_code,
                # either the user accepted, or settings.trust_remote_code is True.
                if self.trusted_models.get(self.settings.model) is None:
                    self.trusted_models[self.settings.model] = True

                # For non-quantized models, compute should match the model's actual dtype.
                # For quantized models, compute comes from quantization resolution (bf16/fp16).
                model_dtype = getattr(self.model, "dtype", None)
                if self.quant is not None and not self.quant.is_quantized and isinstance(
                    model_dtype, torch.dtype
                ):
                    self.compute_dtype = model_dtype
                else:
                    self.compute_dtype = self.quant.compute_dtype

                if getattr(self.settings, "precision_debug", False):
                    self.precision_policy.log_probe_matrix(print)

                # Apply precision hooks to base model (covers FP8Linear, generic ops).
                PrecisionApplier(
                    self.precision_policy,
                    print,
                    compute_dtype=self.compute_dtype,
                ).apply(self.model)
            except Exception as error:
                self.model = None  # ty:ignore[invalid-assignment]
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                continue

            print("[green]Ok[/]")
            if self.quant and self.quant.is_quantized:
                label = self._format_quantization_label()
                print(f"[bold green]Model loaded with {label}.[/]")
            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        self._apply_lora()
        self._validate_lora_wrapper_contract()
        self._cast_lora_parameters_to_compute_dtype()
        ok, reason = self._can_abliterate_with_current_weights()
        if not ok:
            raise RuntimeError(
                "Abliteration requires access to effective float weights (W) for v^T W.\n"
                f"- model={self.settings.model}\n"
                f"- quant_method={getattr(self.quant, 'method', None) if self.quant else None}\n"
                f"- details:\n{reason}\n"
                "Remediation: use a quantization method that supports dequantize-on-load for float merge/abliteration, "
                "or export adapter-only."
            )

        # Re-apply precision hooks to cover LoRA wrappers (input casting) and any
        # FP8Linear outputs inside wrapped layers.
        PrecisionApplier(
            self.precision_policy,
            print,
            compute_dtype=self.compute_dtype,
        ).apply(self.model)

        # Sanity test generation to catch dtype/kernel issues early.
        try:
            self.generate(
                [
                    Prompt(
                        system=settings.system_prompt,
                        user="What is 1+1?",
                    )
                ],
                max_new_tokens=1,
            )
        except Exception as error:
            print(f"[yellow]Sanity generate failed[/] ({error})")

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        for component, modules in self.get_layer_modules(0).items():
            print(
                f"  * [bold]{component}[/]: [bold]{len(modules)}[/] modules per layer"
            )

    def _log_loading_info(self, loading_info: Any) -> None:
        """
        Print a compact summary of Transformers loading info (missing/unexpected keys).
        """
        if not isinstance(loading_info, dict):
            return
        missing = loading_info.get("missing_keys")
        unexpected = loading_info.get("unexpected_keys")
        errors = loading_info.get("error_msgs")

        def _count(x: Any) -> int:
            return len(x) if isinstance(x, list) else 0

        mc = _count(missing)
        uc = _count(unexpected)
        ec = _count(errors)
        if mc == 0 and uc == 0 and ec == 0:
            print("* loading_info: missing_keys=0 unexpected_keys=0 error_msgs=0")
            return

        def _examples(x: Any) -> list[str]:
            if not isinstance(x, list):
                return []
            out: list[str] = []
            for item in x[:5]:
                try:
                    out.append(str(item))
                except Exception:
                    continue
            return out

        print(
            "* loading_info:"
            f" missing_keys={mc}"
            f" unexpected_keys={uc}"
            f" error_msgs={ec}"
        )
        ex_m = _examples(missing)
        ex_u = _examples(unexpected)
        if ex_m:
            print(f"  - missing_examples={ex_m}")
        if ex_u:
            print(f"  - unexpected_examples={ex_u}")

    def _format_quantization_label(self) -> str:
        if not self.quant or not self.quant.is_quantized:
            return "no quantization"
        if self.quant.method:
            return f"{self.quant.method} quantization"
        return "quantization"

    def _ensure_chat_template(self) -> None:
        """
        Ensure `tokenizer.apply_chat_template(...)` works for local model dirs that ship
        `chat_template.jinja` as a separate file.
        """
        if getattr(self.tokenizer, "chat_template", None) is not None:
            return
        model_dir = self.settings.model
        if not isinstance(model_dir, str) or not os.path.isdir(model_dir):
            return
        template_path = os.path.join(model_dir, "chat_template.jinja")
        if not os.path.exists(template_path):
            return
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
        except Exception:
            return
        if not template.strip():
            return
        try:
            setattr(self.tokenizer, "chat_template", template)
            print(f"* Loaded chat template from [bold]{template_path}[/]")
        except Exception:
            return

    def _maybe_materialize_tiny_checkpoint(self) -> None:
        """
        If `mock_tiny_model` is enabled and `settings.model` points at a local MiniMax
        M2.1 *code* directory without weights, materialize a tiny checkpoint directory
        and switch `settings.model` to it.
        """
        if not getattr(self.settings, "mock_tiny_model", False):
            return
        source_dir = self.settings.model
        if not looks_like_minimax_m2_source_dir(source_dir) and not looks_like_kimi_k25_source_dir(
            source_dir
        ):
            return

        out_base = os.path.expanduser(
            os.path.expandvars(getattr(self.settings, "mock_tiny_out_dir", "~/.cache/heretic/mock_models"))
        )

        if looks_like_minimax_m2_source_dir(source_dir):
            spec = TinyMiniMaxM2Spec(
                hidden_size=getattr(self.settings, "mock_tiny_hidden_size", 64),
                intermediate_size=getattr(self.settings, "mock_tiny_intermediate_size", 256),
                num_hidden_layers=getattr(self.settings, "mock_tiny_num_hidden_layers", 2),
                num_attention_heads=getattr(self.settings, "mock_tiny_num_attention_heads", 4),
                num_key_value_heads=getattr(self.settings, "mock_tiny_num_key_value_heads", 2),
                max_position_embeddings=getattr(self.settings, "mock_tiny_max_position_embeddings", 2048),
                sliding_window=getattr(self.settings, "mock_tiny_sliding_window", 256),
                num_experts_per_tok=getattr(self.settings, "mock_tiny_num_experts_per_tok", 2),
                num_local_experts=getattr(self.settings, "mock_tiny_num_local_experts", 2),
                seed=getattr(self.settings, "mock_tiny_seed", 0),
            )
            model_family = "minimax_m2"
        else:
            # Kimi K2.5 uses very large special token ids, so vocab size stays large even
            # for the tiny checkpoint; the model stays small due to tiny hidden/layer sizes.
            hidden_size = int(getattr(self.settings, "mock_tiny_hidden_size", 64))
            num_heads = int(getattr(self.settings, "mock_tiny_num_attention_heads", 4))
            head_dim = max(1, hidden_size // max(1, num_heads))
            spec = TinyKimiK25Spec(
                hidden_size=hidden_size,
                intermediate_size=int(getattr(self.settings, "mock_tiny_intermediate_size", 256)),
                moe_intermediate_size=max(
                    1, int(getattr(self.settings, "mock_tiny_intermediate_size", 256)) // 4
                ),
                num_hidden_layers=int(getattr(self.settings, "mock_tiny_num_hidden_layers", 2)),
                num_attention_heads=num_heads,
                num_key_value_heads=int(getattr(self.settings, "mock_tiny_num_key_value_heads", 4)),
                max_position_embeddings=int(
                    getattr(self.settings, "mock_tiny_max_position_embeddings", 512)
                ),
                n_routed_experts=max(1, int(getattr(self.settings, "mock_tiny_num_local_experts", 2))),
                n_shared_experts=1,
                num_experts_per_tok=max(
                    1, int(getattr(self.settings, "mock_tiny_num_experts_per_tok", 1))
                ),
                qk_rope_head_dim=max(1, head_dim // 2),
                qk_nope_head_dim=max(1, head_dim - max(1, head_dim // 2)),
                v_head_dim=head_dim,
                seed=int(getattr(self.settings, "mock_tiny_seed", 0)),
            )
            model_family = "kimi_k25"

        key = json.dumps(
            {"source_dir": os.path.abspath(source_dir), "spec": spec.__dict__},
            sort_keys=True,
        )
        suffix = sha1(key.encode("utf-8")).hexdigest()[:10]
        out_dir = os.path.join(out_base, f"{model_family}_tiny-{suffix}")

        # If the user didn't specify trust_remote_code, force it on for this local remote-code repo.
        if self.settings.trust_remote_code is None:
            self.settings.trust_remote_code = True

        if looks_like_minimax_m2_source_dir(source_dir):
            self.settings.model = materialize_tiny_minimax_m2_repo(
                source_dir=source_dir,
                out_dir=out_dir,
                spec=cast(TinyMiniMaxM2Spec, spec),
                logger=print,
            )
        else:
            self.settings.model = materialize_tiny_kimi_k25_repo(
                source_dir=source_dir,
                out_dir=out_dir,
                spec=cast(TinyKimiK25Spec, spec),
                logger=print,
            )

    def _apply_lora(self):
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PreTrainedModel)

        # Always use LoRA adapters for abliteration.
        target_modules = self._resolve_lora_target_module_names()
        # Save for diagnostics / strict contract errors.
        self._lora_target_modules = list(target_modules)
        self._lora_plan = {
            "targets": list(target_modules),
            "created_from_layer_index": 0,
        }

        if self.settings.row_normalization != RowNormalization.FULL:
            # Rank 1 is sufficient for directional ablation without renormalization.
            self.lora_rank = 1
        else:
            # Row magnitude preservation introduces nonlinear effects. A rank of 3 is enough to explain
            # most of the variance in the delta matrix, and reduction of the spectral norm of the error
            # of the reconstructed matrix falls off at higher ranks.
            self.lora_rank = 3

        peft_config = LoraConfig(
            r=self.lora_rank,
            target_modules=target_modules,
            lora_alpha=self.lora_rank,  # Apply adapter at full strength.
            lora_dropout=0,
            bias="none",
            # Even if we're using AutoModelForImageTextToText, this is still correct, as it is (post-vision)
            # the same kind of model.
            # https://github.com/huggingface/peft/blob/622c2821cb0d7897bee53aad7914d42b5fecbf61/src/peft/auto.py#L45
            task_type="CAUSAL_LM",
        )

        # peft_config is a LoraConfig object rather than a dictionary,
        # so the result is a PeftModel rather than a PeftMixedModel.
        self.model = cast(PeftModel, get_peft_model(self.model, peft_config))

        print(
            f"[green]LoRA adapters initialized (targets: {', '.join(target_modules)})[/]"
        )

    def _lora_targets_for_merge(self) -> list[str]:
        plan = getattr(self, "_lora_plan", None)
        if isinstance(plan, dict) and isinstance(plan.get("targets"), list):
            return list(plan["targets"])
        targets = getattr(self, "_lora_target_modules", None)
        if isinstance(targets, list) and targets:
            return list(targets)
        return self._resolve_lora_target_module_names()

    def _probe_merge_correctness(
        self,
        *,
        peft_model: Any,
        tokenizer: Any,
        max_new_tokens: int = 1,
    ) -> bool:
        """
        Behavioral probe: compare a tiny forward pass before/after merge.
        Returns True if outputs match within tolerance.
        """
        try:
            device = getattr(peft_model, "device", None) or torch.device("cpu")
            inputs = tokenizer(
                "hello",
                return_tensors="pt",
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out_before = peft_model(**inputs).logits  # type: ignore[attr-defined]
            merged = peft_model.merge_and_unload()
            with torch.no_grad():
                out_after = merged(**inputs).logits  # type: ignore[attr-defined]
            # Compare a small slice for speed.
            a = out_before[..., :8].float().cpu()
            b = out_after[..., :8].float().cpu()
            return torch.allclose(a, b, rtol=1e-3, atol=1e-3)
        except Exception:
            return False

    def _can_abliterate_with_current_weights(self) -> tuple[bool, str | None]:
        """
        Capability check: can we materialize W for one module per component per layer?
        """
        try:
            n_layers = len(self.get_layers())
            for layer_index in range(n_layers):
                layer_modules = self.get_layer_modules(layer_index)
                for component, modules in layer_modules.items():
                    if not modules:
                        continue
                    module = modules[0]
                    module_any = cast(Any, module)
                    base_layer = getattr(module_any, "base_layer", None)
                    if base_layer is None:
                        return (
                            False,
                            "\n".join(
                                [
                                    "Missing base_layer for weight access.",
                                    f"- layer={layer_index}",
                                    f"- component={component}",
                                    f"- module_class={type(module).__name__}",
                                    "Remediation: ensure LoRA wrappers are properly attached.",
                                ]
                            ),
                        )
                    _ = WeightAccess.materialize_W_float32(
                        base_layer=base_layer, component=component, layer_index=layer_index
                    )
            return True, None
        except WeightAccessError as exc:
            return False, str(exc)

    def _require_lora_wrapper_contract(
        self,
        module: Any,
        *,
        layer_index: int,
        component: str,
        module_name: str | None = None,
    ) -> None:
        """
        Strictly require the LoRA wrapper contract needed by `abliterate()`.

        Contract:
        - module.base_layer.weight exists (for reading/dequantizing W)
        - module.lora_A[\"default\"].weight exists
        - module.lora_B[\"default\"].weight exists
        """
        validated = getattr(self, "_validated_lora_module_ids", None)
        if isinstance(validated, set) and id(module) in validated:
            return

        missing: list[str] = []
        base_layer = getattr(module, "base_layer", None)
        if base_layer is None:
            missing.append("base_layer")
        else:
            w = getattr(base_layer, "weight", None)
            if w is None:
                # compressed-tensors (e.g. CompressedLinear) deletes `.weight` and exposes a compressor.
                compressor = getattr(base_layer, "compressor", None)
                decompress = getattr(compressor, "decompress_module", None)
                if not callable(decompress):
                    missing.append("base_layer.weight")

        def _get_default_adapter(container: Any) -> Any | None:
            # PEFT uses different container types across versions (dict, ModuleDict, etc).
            if container is None:
                return None
            get_fn = getattr(container, "get", None)
            if callable(get_fn):
                try:
                    return get_fn("default")
                except Exception:
                    return None
            try:
                if "default" in container:  # type: ignore[operator]
                    return container["default"]  # type: ignore[index]
            except Exception:
                return None
            return None

        lora_A = getattr(module, "lora_A", None)
        sub = _get_default_adapter(lora_A)
        if sub is None or getattr(sub, "weight", None) is None:
            missing.append('lora_A["default"].weight')

        lora_B = getattr(module, "lora_B", None)
        sub = _get_default_adapter(lora_B)
        if sub is None or getattr(sub, "weight", None) is None:
            missing.append('lora_B["default"].weight')

        if missing:
            model_id = getattr(self.settings, "model", None)
            target_modules = getattr(self, "_lora_target_modules", None)
            cls_name = type(module).__name__
            name_str = f" name={module_name}" if module_name else ""
            raise RuntimeError(
                "LoRA wrapper contract not satisfied for abliteration.\n"
                f"- model={model_id}\n"
                f"- layer={layer_index}\n"
                f"- component={component}\n"
                f"- module_class={cls_name}{name_str}\n"
                f"- missing={missing}\n"
                f"- lora_target_modules={target_modules}\n"
                f"- model_dtype={getattr(self.model, 'dtype', None)}\n"
                f"- compute_dtype={self.compute_dtype}\n"
                "Remediation: PEFT did not attach LoRA to this module type/name. "
                "Adjust target_modules or ensure PEFT supports this layer wrapper."
            )

        if isinstance(validated, set):
            validated.add(id(module))

    def _validate_lora_wrapper_contract(self) -> None:
        """
        Validate LoRA wrapper contract for all modules that will be ablated.
        This is strict: raises immediately if any required wrapper is missing.
        """
        self._validated_lora_module_ids = set()
        n_layers = len(self.get_layers())
        for layer_index in range(n_layers):
            for component, modules in self.get_layer_modules(layer_index).items():
                for module in modules:
                    self._require_lora_wrapper_contract(
                        module,
                        layer_index=layer_index,
                        component=component,
                    )

    def _resolve_lora_target_module_names(self) -> list[str]:
        """
        Resolve PEFT `target_modules` from the *actual modules* selected by `get_layer_modules(0)`.

        This avoids mismatches between the component label (e.g. \"mlp.down_proj\") and the
        model's real attribute name (e.g. MiniMax/Phi-style `w2`).
        """
        assert isinstance(self.model, PreTrainedModel)
        selected = self.get_layer_modules(0)
        selected_ids = {id(m) for ms in selected.values() for m in ms}

        leaf_names: set[str] = set()
        for name, module in self.model.named_modules():
            if id(module) not in selected_ids:
                continue
            leaf_names.add(name.split(".")[-1])

        if leaf_names:
            return sorted(leaf_names)

        # Fallback: original behavior (component label suffix).
        return [comp.split(".")[-1] for comp in self.get_abliterable_components()]

    def _cast_lora_parameters_to_compute_dtype(self) -> None:
        """
        Ensure all LoRA adapter parameters live in compute dtype.

        Base weights may be quantized; LoRA should run in compute dtype for stability and
        to avoid FP8/unsupported dtype failures in adapter matmuls.
        """
        if self.compute_dtype is None:
            return
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                continue
            if param.dtype == self.compute_dtype:
                continue
            param.data = param.data.to(self.compute_dtype)

    def get_merged_model(self) -> PreTrainedModel:
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PeftModel)

        is_quantized = bool(self.quant is not None and self.quant.is_quantized)

        # Prefer preserving quantization when possible and correct.
        hf_quantizer = getattr(self.model, "hf_quantizer", None) or getattr(
            getattr(self.model, "base_model", None), "hf_quantizer", None
        )
        if is_quantized and hf_quantizer is not None and getattr(hf_quantizer, "is_serializable", lambda: False)():
            print("* Attempting in-place merge (preserve quantization)...")
            if self._probe_merge_correctness(peft_model=self.model, tokenizer=self.tokenizer):
                merged_model = self.model.merge_and_unload()
                self._prepare_model_for_saving(merged_model)
                self.needs_reload = True
                return merged_model
            print("[yellow]In-place merge probe failed; falling back to float merge if supported.[/]")

        # Float-merge path (dequantize-on-load when supported).
        if is_quantized:
            if not (self.quant and self.quant.supports_dequantize_on_load):
                raise RuntimeError(
                    "Cannot produce a float merged export for this quantized model.\n"
                    f"- model={self.settings.model}\n"
                    f"- quant_method={getattr(self.quant, 'method', None)}\n"
                    "Reason: this quantization backend does not advertise dequantize-on-load support.\n"
                    "Remediation: export adapter-only, or use a quantization method that supports dequantize-on-load."
                )

            # Capture adapter weights first.
            adapter_state = {
                name: param.data.clone().cpu()
                for name, param in self.model.named_parameters()
                if "lora_" in name
            }

            print("* Loading dequantized base model on CPU for float merge (this may take a while)...")
            # Transformers requires passing the *same quantization config class* as the model uses
            # (e.g. FineGrainedFP8Config), not a raw dict. Passing a dict can raise:
            #   ValueError: The model is quantized with FineGrainedFP8Config but you are passing a dict config.
            qcfg_obj: object | None = None
            try:
                qcfg_obj = copy.deepcopy(getattr(self.model.config, "quantization_config", None))
            except Exception:
                qcfg_obj = getattr(self.model.config, "quantization_config", None)

            if qcfg_obj is None and self.quant is not None:
                model_cfg = getattr(self.quant, "model_provided_config", None)
                if model_cfg:
                    try:
                        from transformers.quantizers.auto import AutoQuantizationConfig

                        qcfg_obj = AutoQuantizationConfig.from_dict(dict(model_cfg))
                    except Exception:
                        qcfg_obj = None

            if qcfg_obj is not None:
                to_dict = getattr(qcfg_obj, "to_dict", None)
                if callable(to_dict):
                    try:
                        d = dict(to_dict())
                        d["dequantize"] = True
                        qcfg_obj = type(qcfg_obj)(**d)
                    except Exception:
                        try:
                            setattr(qcfg_obj, "dequantize", True)
                        except Exception:
                            pass
            load_kwargs = {
                "torch_dtype": self.model.dtype,
                "device_map": "cpu",
                "trust_remote_code": self.trusted_models.get(self.settings.model),
            }
            if qcfg_obj is not None:
                load_kwargs["quantization_config"] = qcfg_obj
            base_model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                **load_kwargs,
            )

            print("* Applying LoRA adapters (float-merge path)...")
            targets = self._lora_targets_for_merge()
            peft_config = LoraConfig(
                r=self.lora_rank,
                target_modules=targets,
                lora_alpha=self.lora_rank,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            peft_model = get_peft_model(base_model, peft_config)

            # Ensure LoRA actually attached: all adapter params must exist.
            peft_param_names = {n for n, _ in peft_model.named_parameters()}
            missing = [n for n in adapter_state.keys() if n not in peft_param_names]
            if missing:
                raise RuntimeError(
                    "Float-merge path failed to attach LoRA to the reloaded base model.\n"
                    f"- model={self.settings.model}\n"
                    f"- lora_targets={targets}\n"
                    f"- missing_adapter_params_count={len(missing)}\n"
                    f"- example_missing={missing[:5]}\n"
                    "Remediation: ensure LoRA target_modules are correct for the model, or export adapter-only."
                )

            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            print("* Merging LoRA adapters into base model (float export)...")
            merged_model = peft_model.merge_and_unload()
            self._prepare_model_for_saving(merged_model)
            return merged_model

        # Non-quantized model: merge directly.
        print("* Merging LoRA adapters into base model...")
        merged_model = self.model.merge_and_unload()
        self._prepare_model_for_saving(merged_model)
        self.needs_reload = True
        return merged_model

    @staticmethod
    def _prepare_model_for_saving(model: PreTrainedModel) -> None:
        """
        Prepare a merged model for `save_pretrained()`.

        Some checkpoints are loaded via Transformers weight conversion mappings which are stored
        on the model. During `save_pretrained()`, Transformers may attempt to reverse those
        conversions; for certain conversions in the pinned Transformers version, the reverse
        operation is not implemented, leading to `NotImplementedError`.
        """
        for attr in ("_weight_conversions", "_checkpoint_conversion_mapping"):
            if hasattr(model, attr):
                try:
                    delattr(model, attr)
                except Exception:
                    try:
                        setattr(model, attr, None)
                    except Exception:
                        pass

    def reset_model(self):
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behavior:
        - Fast path: If the same model is loaded and doesn't need full reload,
          resets LoRA adapter weights to zero (identity transformation).
        - Slow path: If switching models or after merge_and_unload(),
          performs full model reload with quantization config.
        """
        current_model = getattr(self.model.config, "name_or_path", None)
        if current_model == self.settings.model and not self.needs_reload:
            # Reset LoRA adapters to zero (identity transformation)
            for name, module in self.model.named_modules():
                if "lora_B" in name and hasattr(module, "weight"):
                    torch.nn.init.zeros_(module.weight)
            return

        dtype = self.model.dtype

        # Purge existing model object from memory to make space.
        self.model = None  # ty:ignore[invalid-assignment]
        empty_cache()

        dtype_name = str(dtype).split(".")[-1]
        quant_req = QuantizationRequest(
            method=getattr(self.settings.quantization, "value", str(self.settings.quantization)),
            requested_dtype=dtype_name,
            config_type=getattr(self.settings, "quantization_config_type", None),
            config_kwargs=getattr(self.settings, "quantization_kwargs", None),
        )
        self.quant = resolve_quantization(
            model_id_or_path=self.settings.model,
            request=quant_req,
            precision_fallback_dtype=getattr(self.settings, "precision_fallback_dtype", "auto"),
        )
        self.compute_dtype = self.quant.compute_dtype

        # Build kwargs, only include quantization_config if it's not None
        extra_kwargs = {}
        if self.quant.load_quantization_config is not None:
            extra_kwargs["quantization_config"] = self.quant.load_quantization_config

        load_kwargs = {
            "torch_dtype": self.precision_policy.resolve_model_dtype(dtype_name),
            "device_map": self.settings.device_map,
            "max_memory": self.max_memory,
            "trust_remote_code": self.trusted_models.get(self.settings.model),
            **extra_kwargs,
        }
        if self._print_loading_info:
            load_kwargs["output_loading_info"] = True
        if (
            self.quant is not None
            and self.quant.is_quantized
            and (self.quant.method or "") in {"fp8", "finegrained_fp8", "fine-grained-fp8"}
        ):
            load_kwargs["attn_implementation"] = "eager"
        loaded = get_model_class(self.settings.model).from_pretrained(
            self.settings.model,
            **load_kwargs,
        )
        if self._print_loading_info and isinstance(loaded, tuple) and len(loaded) == 2:
            self.model, loading_info = loaded
            self._log_loading_info(loading_info)
        else:
            self.model = loaded

        self._apply_lora()
        self._validate_lora_wrapper_contract()
        self._cast_lora_parameters_to_compute_dtype()
        ok, reason = self._can_abliterate_with_current_weights()
        if not ok:
            raise RuntimeError(
                "Abliteration requires access to effective float weights (W) for v^T W.\n"
                f"- model={self.settings.model}\n"
                f"- quant_method={getattr(self.quant, 'method', None) if self.quant else None}\n"
                f"- details:\n{reason}\n"
                "Remediation: use a quantization method that supports dequantize-on-load for float merge/abliteration, "
                "or export adapter-only."
            )

        PrecisionApplier(
            self.precision_policy,
            print,
            compute_dtype=self.compute_dtype,
        ).apply(self.model)

        self.needs_reload = False

    def get_layers(self) -> ModuleList:
        model = self.model

        # Unwrap PeftModel (always true after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Kimi K2.5-style composite wrappers.
        with suppress(Exception):
            language_model = getattr(model, "language_model")
            return language_model.model.layers

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers

        # Text-only models.
        return model.model.layers

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
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
            experts = layer.mlp.experts  # ty:ignore[possibly-missing-attribute]
            limit = int(getattr(self.settings, "max_moe_experts_per_layer", 0) or 0)
            if limit > 0:
                experts = list(experts)[:limit]  # type: ignore[arg-type]
            for expert in experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
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
        return list(self.get_layer_modules(0).keys())

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ):
        if direction_index is None:
            refusal_direction = None
        else:
            # The index must be shifted by 1 because the first element
            # of refusal_directions is the direction for the embeddings.
            weight, index = math.modf(direction_index + 1)
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
                    # The index must be shifted by 1 because the first element
                    # of refusal_directions is the direction for the embeddings.
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                for module in modules:
                    module_any = cast(Any, module)
                    self._require_lora_wrapper_contract(
                        module_any,
                        layer_index=layer_index,
                        component=component,
                    )

                    # LoRA abliteration: delta W = -lambda * v * (v^T W)
                    # lora_B = -lambda * v
                    # lora_A = v^T W

                    # Use the FP32 refusal direction directly (no downcast/upcast)
                    # and move to the correct device.
                    v = layer_refusal_direction.to(module_any.weight.device)

                    # Get W (dequantize if necessary).
                    base_layer = module_any.base_layer
                    try:
                        W = WeightAccess.materialize_W_float32(
                            base_layer=base_layer,
                            component=component,
                            layer_index=layer_index,
                        )
                    except WeightAccessError as exc:
                        raise RuntimeError(
                            "Unsupported quantization backend for abliteration weight access.\n"
                            f"- model={self.settings.model}\n"
                            f"- quant_method={getattr(self.quant, 'method', None) if self.quant else None}\n"
                            f"{exc}\n"
                            "Remediation: use a float model, a quantization backend that supports dequantize-on-load, "
                            "or export adapter-only."
                        ) from exc

                    if self.settings.row_normalization != RowNormalization.NONE:
                        # Keep a reference to the original weight matrix so we can subtract it later.
                        W_org = W
                        # Get the row norms (cast to work around untyped LA).
                        W_row_norms = cast(
                            Tensor, LA.vector_norm(W, dim=1, keepdim=True)
                        )
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
                        r = self.lora_rank
                        U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)
                        # Truncate it to the part we want to store in the LoRA adapter.
                        # Note: svd_lowrank actually returns V, so transpose it to get Vh.
                        U = U[:, :r]
                        S = S[:r]
                        Vh = Vh[:, :r].T
                        # Transfer it into the LoRA adapter components.
                        sqrt_S = torch.sqrt(S)
                        lora_B = U @ torch.diag(sqrt_S)
                        lora_A = torch.diag(sqrt_S) @ Vh

                    # Assign to adapters. The adapter name is "default", because that's
                    # what PEFT uses when no name is explicitly specified, as above.
                    # These casts are therefore valid.
                    weight_A = cast(Tensor, module_any.lora_A["default"].weight)
                    weight_B = cast(Tensor, module_any.lora_B["default"].weight)
                    weight_A.data = lora_A.to(weight_A.dtype)
                    weight_B.data = lora_B.to(weight_B.dtype)

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
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

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            # `apply_chat_template(tokenize=False)` already emits any required special tokens.
            # Avoid double-inserting BOS/EOS (and some remote-code tokenizers mis-handle it).
            add_special_tokens=False,
        ).to(self.model.device)

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

        return inputs, outputs

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
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
    ) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(
                batch,
                skip_special_tokens=skip_special_tokens,
            ):
                responses.append(response)

        return responses

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Hidden states for the first (only) generated token.
        # This cast is valid because we passed output_hidden_states=True above.
        hidden_states = cast(tuple[tuple[FloatTensor]], outputs.hidden_states)[0]

        # The returned tensor has shape (prompt, layer, component).
        residuals = torch.stack(
            # layer_hidden_states has shape (prompt, position, component),
            # so this extracts the hidden states at the end of each prompt,
            # and stacks them up over the layers.
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        residuals = residuals.to(torch.float32)

        if 0 <= self.settings.winsorization_quantile < 1:
            # Perform symmetric magnitude winsorization on the residuals.
            abs_residuals = torch.abs(residuals)
            thresholds = torch.quantile(
                abs_residuals, self.settings.winsorization_quantile, 2, True
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
        # We only generate one token, and we return the (log) probability distributions
        # over the vocabulary at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Logits for the first (only) generated token.
        # This cast is valid because we passed output_scores=True above.
        logits = cast(tuple[FloatTensor], outputs.scores)[0]

        # The returned tensor has shape (prompt, token).
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
            add_special_tokens=False,
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
