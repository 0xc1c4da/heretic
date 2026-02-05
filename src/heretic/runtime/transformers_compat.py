# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Transformers compatibility shims.

Why this exists:
- Some environments have a `transformers` build where
  `transformers.utils.generic.check_model_inputs` is broken/incompatible with
  models that rely on it (e.g. MiniMax M2.1), yielding:

    TypeError: check_model_inputs.<locals>.wrapped_fn() got an unexpected keyword argument 'input_ids'

This patch is intentionally:
- Low cost: runs before model loading and does not touch checkpoints.
- Targeted: only patches when a self-test demonstrates breakage.
"""

from __future__ import annotations

from contextlib import suppress
from functools import wraps
import os
import sys
from types import SimpleNamespace
from typing import Any, Callable
from pathlib import Path

from .ct_checkpoint_inspector import (
    CheckpointKind,
    inspect_checkpoint,
    infer_checkpoint_kind,
    load_weight_map_keys,
    should_scope_to_language_model,
)


def ensure_transformers_checkpoint_key_prefix_filter(logger: Callable[[str], None]) -> None:
    """
    Speed up and stabilize loading of *subtree* checkpoints with a key prefix.

    Use case (text-only loading from multimodal repos):
    - Some repos store LM weights under a prefix like `language_model.*` alongside large
      non-LM weights (vision tower, projector, etc.).
    - When we instantiate the LM backbone directly and use `key_mapping` to strip the
      prefix, Transformers' loader may still:
        1) spend time computing huge `unexpected_keys` lists, and
        2) mis-detect the "task/base" prefix state because `base_model_prefix` (often
           `"model"`) does not match keys like `language_model.model.*`, which can lead
           to incorrect prefix injection (e.g. `model.model.*`).

    This shim activates only when the model config sets:
      - `config._heretic_checkpoint_key_prefix = "language_model."`  (example)

    When active, it:
    - filters checkpoint key metadata (sharded_metadata/state_dict) to only prefixed keys
    - temporarily sets `model.base_model_prefix = ""` during load to disable the special
      task/base prefix logic that would otherwise corrupt key names
    """
    try:
        from transformers.modeling_utils import PreTrainedModel  # type: ignore
    except Exception:
        return

    cm = PreTrainedModel.__dict__.get("_load_pretrained_model")
    if not isinstance(cm, classmethod):
        return
    original = cm.__func__
    if getattr(original, "_heretic_patched", False):
        return

    def _patched(  # noqa: PLR0913
        cls,  # noqa: ANN001
        model: Any,
        state_dict: Any,  # noqa: ANN401
        checkpoint_files: Any,  # noqa: ANN401
        pretrained_model_name_or_path: Any,  # noqa: ANN401
        ignore_mismatched_sizes: bool = False,
        sharded_metadata: Any = None,  # noqa: ANN401
        device_map: Any = None,  # noqa: ANN401
        disk_offload_folder: Any = None,  # noqa: ANN401
        dtype: Any = None,  # noqa: ANN401
        hf_quantizer: Any = None,  # noqa: ANN401
        keep_in_fp32_regex: Any = None,  # noqa: ANN401
        device_mesh: Any = None,  # noqa: ANN401
        key_mapping: Any = None,  # noqa: ANN401
        weights_only: bool = True,
    ):
        cfg = getattr(model, "config", None)
        prefix = getattr(cfg, "_heretic_checkpoint_key_prefix", None)
        # Fallback: if the marker doesn't survive config deepcopy, infer from key_mapping.
        if not isinstance(prefix, str) or not prefix:
            try:
                if isinstance(key_mapping, dict):
                    for pat, repl in key_mapping.items():
                        if not (isinstance(pat, str) and isinstance(repl, str)):
                            continue
                        # We only support the specific form we emit: strip `language_model.` at start.
                        if repl == "" and pat in {r"^language_model\.", r"^language_model\\."}:
                            prefix = "language_model."
                            break
            except Exception:
                prefix = None

        if not isinstance(prefix, str) or not prefix:
            return original(
                cls,
                model,
                state_dict,
                checkpoint_files,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                sharded_metadata=sharded_metadata,
                device_map=device_map,
                disk_offload_folder=disk_offload_folder,
                dtype=dtype,
                hf_quantizer=hf_quantizer,
                keep_in_fp32_regex=keep_in_fp32_regex,
                device_mesh=device_mesh,
                key_mapping=key_mapping,
                weights_only=weights_only,
            )

        # Log once per process for observability.
        if not getattr(_patched, "_heretic_logged_active", False):
            try:
                logger(f"* transformers: checkpoint key prefix filter active (prefix={prefix})")
            except Exception:
                pass
            setattr(_patched, "_heretic_logged_active", True)

        # Filter key metadata aggressively to avoid giant unexpected_keys costs.
        try:
            if isinstance(state_dict, dict):
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if isinstance(k, str) and k.startswith(prefix)
                }
        except Exception:
            pass

        try:
            if isinstance(sharded_metadata, dict):
                new_meta = dict(sharded_metadata)
                all_keys = new_meta.get("all_checkpoint_keys", None)
                filtered: list[str] | None = None
                if isinstance(all_keys, list):
                    filtered = [k for k in all_keys if isinstance(k, str) and k.startswith(prefix)]
                    new_meta["all_checkpoint_keys"] = filtered
                weight_map = new_meta.get("weight_map", None)
                if isinstance(weight_map, dict) and filtered is not None:
                    keep = set(filtered)
                    new_meta["weight_map"] = {k: v for k, v in weight_map.items() if k in keep}
                sharded_metadata = new_meta
        except Exception:
            pass

        # Prevent Transformers' task/base prefix injection from corrupting keys like
        # `language_model.model.*` when base_model_prefix is `model`.
        had_instance_attr = "base_model_prefix" in getattr(model, "__dict__", {})
        old_bmp = getattr(model, "base_model_prefix", None)
        try:
            setattr(model, "base_model_prefix", "")
            return original(
                cls,
                model,
                state_dict,
                checkpoint_files,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                sharded_metadata=sharded_metadata,
                device_map=device_map,
                disk_offload_folder=disk_offload_folder,
                dtype=dtype,
                hf_quantizer=hf_quantizer,
                keep_in_fp32_regex=keep_in_fp32_regex,
                device_mesh=device_mesh,
                key_mapping=key_mapping,
                weights_only=weights_only,
            )
        finally:
            try:
                if had_instance_attr:
                    setattr(model, "base_model_prefix", old_bmp)
                else:
                    # Restore class default lookup.
                    if "base_model_prefix" in getattr(model, "__dict__", {}):
                        delattr(model, "base_model_prefix")
            except Exception:
                pass

    setattr(_patched, "_heretic_patched", True)
    setattr(_patched, "_heretic_original", original)
    PreTrainedModel._load_pretrained_model = classmethod(_patched)  # type: ignore[assignment]
    logger("* transformers: enabled checkpoint key prefix filter shim")

def ensure_transformers_cache_api_compat(logger: Callable[[str], None]) -> None:
    """
    Patch Transformers cache API drift for remote-code models.

    Kimi/DeepSeek remote code expects `past_key_values.get_max_length()`, but Transformers 4.57.6
    renamed/standardized this as `Cache.get_max_cache_shape()`.

    We add a small alias on the Cache base class:
    - return `None` for dynamic/unbounded caches (where max_cache_shape is negative)
    - otherwise return a positive int
    """
    try:
        from transformers.cache_utils import Cache  # type: ignore
    except Exception:
        return

    if hasattr(Cache, "get_max_length"):
        return

    def _get_max_length(self: Any, layer_idx: int = 0) -> int | None:  # noqa: ANN401
        try:
            m = self.get_max_cache_shape(layer_idx)  # type: ignore[misc]
        except TypeError:
            # Older signatures might not accept a layer index.
            m = self.get_max_cache_shape()  # type: ignore[misc]
        except Exception:
            return None

        try:
            mi = int(m)
        except Exception:
            return None
        # Transformers uses negative to mean "unbounded".
        if mi < 0:
            return None
        return mi

    setattr(_get_max_length, "_heretic_patched", True)
    Cache.get_max_length = _get_max_length  # type: ignore[attr-defined]
    logger("* transformers: patched Cache.get_max_length() compatibility alias")

def ensure_remote_code_generation_mixin(logger: Callable[[str], None]) -> None:
    """
    Make remote-code generative models compatible with Transformers >=4.50.

    In Transformers 4.57.6, `PreTrainedModel.__init__` populates `generation_config` only if
    `cls.can_generate()` is True, which now requires *explicit* inheritance from `GenerationMixin`.

    Some remote-code models (e.g. Kimi/DeepSeek) implement `prepare_inputs_for_generation` but
    do not inherit `GenerationMixin`, so `generation_config` becomes None and `.generate()` fails.

    This shim patches `transformers.dynamic_module_utils.get_class_from_dynamic_module()` so that
    eligible remote-code `PreTrainedModel` classes are wrapped to inherit `GenerationMixin` at
    import time, *before* model instantiation.
    """
    try:
        import transformers.dynamic_module_utils as dmu  # type: ignore
        from transformers.generation.utils import GenerationMixin
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return

    current = getattr(dmu, "get_class_from_dynamic_module", None)
    if current is None:
        return
    if getattr(current, "_heretic_patched", False):
        return

    # Best-effort: identify HF's dynamic remote-code cache root so we only patch those modules.
    hf_modules_cache = None
    with suppress(Exception):
        import transformers.utils as tutils  # type: ignore

        hf_modules_cache = getattr(tutils, "HF_MODULES_CACHE", None)
        if isinstance(hf_modules_cache, str) and hf_modules_cache:
            hf_modules_cache = os.path.abspath(hf_modules_cache)
        else:
            hf_modules_cache = None

    original = current

    def _is_hf_dynamic_remote_module(cls: type) -> bool:
        """
        True iff the class comes from HF's dynamic module cache.
        """
        try:
            mod = sys.modules.get(getattr(cls, "__module__", ""), None)
            mod_file = getattr(mod, "__file__", None)
            if not isinstance(mod_file, str) or not mod_file:
                return False
            mod_file = os.path.abspath(mod_file)
            if isinstance(hf_modules_cache, str) and hf_modules_cache:
                return mod_file.startswith(hf_modules_cache + os.sep) or mod_file == hf_modules_cache
        except Exception:
            return False
        return False

    def _maybe_wrap_generation_mixin(cls: type) -> type:
        # Cache on the original class to keep identity stable across calls.
        cached = getattr(cls, "_heretic_generation_mixin_wrapped_cls", None)
        if isinstance(cached, type):
            return cached

        # Only wrap HF remote-code models.
        if not _is_hf_dynamic_remote_module(cls):
            return cls

        # Only wrap actual models, and only those that look generative but are missing GenerationMixin.
        if not issubclass(cls, PreTrainedModel):
            return cls
        if GenerationMixin in getattr(cls, "__mro__", ()):
            return cls
        if not callable(getattr(cls, "prepare_inputs_for_generation", None)):
            return cls

        try:
            Patched = type(
                f"{cls.__name__}WithGenerationMixin",
                (cls, GenerationMixin),
                {
                    "__module__": getattr(cls, "__module__", cls.__module__),
                    "_heretic_generation_mixin_patched": True,
                },
            )
            setattr(cls, "_heretic_generation_mixin_wrapped_cls", Patched)
            return Patched
        except Exception:
            return cls

    def _patched(
        class_reference: str,
        pretrained_model_name_or_path: Any,
        *args: Any,
        **kwargs: Any,
    ) -> type:
        cls = original(class_reference, pretrained_model_name_or_path, *args, **kwargs)
        wrapped = _maybe_wrap_generation_mixin(cls)
        # Log only when we actually change the class.
        if wrapped is not cls and not getattr(cls, "_heretic_generation_mixin_logged", False):
            with suppress(Exception):
                setattr(cls, "_heretic_generation_mixin_logged", True)
            logger(
                f"* remote-code: patched `{getattr(cls, '__name__', cls)}` to inherit GenerationMixin "
                "(enables generation_config loading)"
            )
        return wrapped

    setattr(_patched, "_heretic_patched", True)
    setattr(_patched, "_heretic_original", original)
    dmu.get_class_from_dynamic_module = _patched  # type: ignore[assignment]


def _fixed_check_model_inputs(original: Any):
    """
    Some broken builds return a decorator with the wrong signature. We force correct
    decorator application by going through the decorator-factory path.
    """

    @wraps(original)
    def fixed(func=None, *, tie_last_hidden_states: bool = True):
        if func is None:
            return original(None, tie_last_hidden_states=tie_last_hidden_states)
        decorator = original(None, tie_last_hidden_states=tie_last_hidden_states)
        return decorator(func)

    return fixed


def _check_model_inputs_selftest(candidate: Any) -> bool:
    """
    Returns True if `candidate` behaves as a correct decorator for keyword inputs.
    This must be cheap and must not import any model code.
    """

    class _Dummy:
        config = SimpleNamespace(
            return_dict=True,
            output_attentions=False,
        )

        def named_modules(self):
            return []

        training = False
        gradient_checkpointing = False

    def _forward(self, input_ids=None, **kwargs):  # noqa: ANN001
        return {"ok": True}

    try:
        decorated = candidate(_forward)
        _ = decorated(_Dummy(), input_ids=1)
        return True
    except Exception:
        return False


def ensure_transformers_compat(logger: Callable[[str], None]) -> None:
    """
    Ensure transformers is compatible with models using `@check_model_inputs`.

    This MUST run before `from_pretrained(..., trust_remote_code=True)` imports model code,
    because remote code does `from transformers.utils.generic import check_model_inputs`.
    """
    from transformers.utils import generic as generic_mod

    current = getattr(generic_mod, "check_model_inputs", None)
    if current is None:
        return

    if _check_model_inputs_selftest(current):
        return

    generic_mod.check_model_inputs = _fixed_check_model_inputs(current)  # type: ignore[attr-defined]

    if not _check_model_inputs_selftest(generic_mod.check_model_inputs):
        raise RuntimeError(
            "Failed to patch transformers check_model_inputs; compatibility shim self-test failed."
        )

    logger(
        "[yellow]Patched[/] transformers `check_model_inputs` for compatibility (preflight self-test failed)."
    )


def ensure_peft_compat(logger: Callable[[str], None]) -> None:
    """
    Ensure PEFT is compatible with quantized/compressed linear layers that may not expose
    a `.weight` attribute at LoRA injection time (e.g. `compressed_tensors` CompressedLinear,
    which deletes `.weight` until first forward).

    PEFT's `peft.tuners.tuners_utils._get_in_out_features` currently touches `module.weight`
    for `nn.Linear` (DTensor check). We patch it to use `in_features/out_features` when
    `.weight` is absent.
    """
    try:
        import torch.nn as nn
        from peft.tuners import tuners_utils as tu  # type: ignore
    except Exception:
        return

    current = getattr(tu, "_get_in_out_features", None)
    if current is None:
        return
    if getattr(current, "_heretic_patched", False):
        return

    original = current

    def _safe_get_in_out_features(module: nn.Module):  # type: ignore[override]
        if isinstance(module, nn.Linear) and not hasattr(module, "weight"):
            in_features = getattr(module, "in_features", None)
            out_features = getattr(module, "out_features", None)
            return in_features, out_features

        # Delegate to upstream for all other cases (including Linear when weight exists).
        return original(module)

    setattr(_safe_get_in_out_features, "_heretic_patched", True)
    setattr(_safe_get_in_out_features, "_heretic_original", original)
    tu._get_in_out_features = _safe_get_in_out_features  # type: ignore[assignment]

    # PEFT also re-exports `_get_in_out_features` into tuner modules (e.g. LoRA layer code)
    # via `from peft.tuners.tuners_utils import _get_in_out_features`. Patch those aliases too.
    try:
        from peft.tuners.lora import layer as lora_layer  # type: ignore

        if hasattr(lora_layer, "_get_in_out_features"):
            lora_layer._get_in_out_features = _safe_get_in_out_features  # type: ignore[assignment]
    except Exception:
        pass

    logger("[yellow]Patched[/] PEFT `_get_in_out_features` for compressed Linear compatibility.")


def ensure_generation_compat(logger: Callable[[str], None], model: Any) -> Any:
    """
    Some remote-code models define `prepare_inputs_for_generation` but do not inherit from
    `GenerationMixin`, and some Transformers builds no longer provide `.generate` via
    `PreTrainedModel`.

    This shim adds `GenerationMixin` to the instance's class MRO dynamically when needed.
    It is best-effort and only activates when `.generate` is missing.
    """
    try:
        from transformers.generation.utils import GenerationMixin
        from transformers.generation.configuration_utils import GenerationConfig
    except Exception:
        return model

    def _ensure_generation_config(obj: Any) -> None:
        # Transformers generation assumes `generation_config` is a GenerationConfig instance,
        # not `None`. Some remote-code models (or wrapper unwrapping) can leave it unset/None.
        if obj is None or not hasattr(obj, "config"):
            return
        if getattr(obj, "generation_config", None) is not None:
            return
        try:
            obj.generation_config = GenerationConfig.from_model_config(obj.config)
        except Exception:
            # Fall back to a default config. This avoids crashes in `generate()` when
            # downstream assumes `generation_config` is always an object.
            with suppress(Exception):
                obj.generation_config = GenerationConfig()

    if model is None or hasattr(model, "generate"):
        _ensure_generation_config(model)
        return model

    if not hasattr(model, "prepare_inputs_for_generation"):
        return model

    cls = model.__class__
    # Avoid repeated patching.
    if GenerationMixin in getattr(cls, "__mro__", ()):
        return model

    try:
        Patched = type(f"{cls.__name__}WithGenerationMixin", (cls, GenerationMixin), {})
        model.__class__ = Patched
        _ensure_generation_config(model)
        logger("[yellow]Patched[/] model class to include GenerationMixin (restore .generate).")
    except Exception as exc:
        logger(f"[yellow]GenerationMixin patch skipped[/] ({exc})")
    return model


def ensure_compressed_tensors_fast_load(
    logger: Callable[[str], None], *, enabled: bool = False
) -> None:
    """
    Experimental: speed up loading for pre-compressed `compressed-tensors` checkpoints.

    The `compressed_tensors.apply_quantization_config(..., run_compressed=True)` path does an
    expensive sweep over the entire model to mark/replace target modules. For composite
    multimodal wrappers (like Kimi K2.5), we only need quantization applied to the
    `language_model` subtree for Heretic's language-only use.

    This shim monkey-patches `compressed_tensors.quantization.lifecycle.apply.apply_quantization_config`
    so that, when the root model has a `.language_model` attribute, quantization is applied
    to that subtree only (leaving vision/projector modules inert and unmodified).
    """
    if not enabled:
        return

    try:
        from compressed_tensors.quantization.lifecycle import apply as apply_mod  # type: ignore
    except Exception:
        return

    fn = getattr(apply_mod, "apply_quantization_config", None)
    if fn is None:
        return
    if getattr(fn, "_heretic_patched", False):
        return

    original = fn

    def _apply_quantization_config_fast(model, config, run_compressed: bool = False):  # noqa: ANN001
        try:
            lm = getattr(model, "language_model", None)
            if lm is not None:
                # Apply quantization to language model only.
                logger(
                    "* compressed-tensors: applying quantization to `language_model` subtree only"
                )
                return original(lm, config, run_compressed=run_compressed)
        except Exception:
            # Fall back to original behavior.
            pass
        return original(model, config, run_compressed=run_compressed)

    setattr(_apply_quantization_config_fast, "_heretic_patched", True)
    setattr(_apply_quantization_config_fast, "_heretic_original", original)
    apply_mod.apply_quantization_config = _apply_quantization_config_fast  # type: ignore[assignment]
    logger("Enabled compressed-tensors fast-load shim (experimental).")


def ensure_compressed_tensors_ephemeral_decompression(logger: Callable[[str], None]) -> None:
    """
    Prevent `compressed-tensors` from "freezing" dense expert weights into VRAM.

    Why this is needed:
    - `compressed_tensors.linear.CompressedLinear.forward()` currently decompresses a compressed
      weight once, registers it as a real `Parameter` (`weight`), and flips the module's
      `quantization_status` to `FROZEN`.
    - For MoE models (like Kimi K2.5), visiting many experts causes *monotonic* VRAM growth:
      each visited expert permanently materializes multiple large dense matrices.

    What we do:
    - Monkey-patch `CompressedLinear.forward` to keep decompression *ephemeral*:
      decompress -> run `F.linear` -> drop the temporary tensor.
    - This trades throughput for bounded VRAM.

    Opt-out:
    - Set `HERETIC_CT_ALLOW_FREEZE=1` to keep upstream freezing behavior.
    """
    if os.environ.get("HERETIC_CT_ALLOW_FREEZE", "").strip() == "1":
        return

    try:
        from torch.nn.functional import linear as _linear

        from compressed_tensors.linear.compressed_linear import (  # type: ignore
            CompressedLinear,
        )
        from compressed_tensors.quantization import QuantizationStatus  # type: ignore
    except Exception:
        return

    current = getattr(CompressedLinear, "forward", None)
    if current is None:
        return
    if getattr(current, "_heretic_patched", False):
        return

    def _forward(self: Any, input: Any) -> Any:  # noqa: ANN401
        # If the layer is still compressed, decompress to a temporary tensor and do NOT
        # register it as a persistent parameter (which would "freeze" it into VRAM).
        try:
            if getattr(self, "quantization_status", None) == QuantizationStatus.COMPRESSED:
                w = self.compressor.decompress_module(self)
                if w is None:
                    # Best-effort fallback if compressor refuses to decompress.
                    return _linear(input, self.weight, self.bias)
                if getattr(w, "device", None) != getattr(input, "device", None):
                    w = w.to(input.device)
                return _linear(input, w, self.bias)
        except Exception:
            # Fall back to the original implementation if anything unexpected happens.
            return current(self, input)

        # Already frozen (or non-standard status): behave like a normal Linear.
        return _linear(input, self.weight, self.bias)

    setattr(_forward, "_heretic_patched", True)
    setattr(_forward, "_heretic_original", current)
    CompressedLinear.forward = _forward  # type: ignore[assignment]
    logger("* compressed-tensors: patched CompressedLinear to avoid frozen dense weights (ephemeral decompression)")


def ensure_compressed_tensors_skip_recompress(logger: Callable[[str], None]) -> None:
    """
    Avoid unnecessary `compressed-tensors` in-memory recompression during model load.

    Transformers' `CompressedTensorsHfQuantizer` calls:
      1) apply_quantization_config(model, ...)
      2) compressor.compress_model(model)  # emits `Compressing model: ...`

    On large pre-compressed checkpoints (e.g. Kimi K2.5), step (2) can be extremely slow and
    redundant if the checkpoint already stores compressed parameters.

    This shim monkey-patches the quantizer to *auto-detect* a pre-compressed checkpoint from
    `checkpoint_files` and skip `compress_model()` when safe.
    """
    try:
        from transformers.quantizers.quantizer_compressed_tensors import (  # type: ignore
            CompressedTensorsHfQuantizer,
        )
    except Exception:
        return

    current = getattr(CompressedTensorsHfQuantizer, "_process_model_before_weight_loading", None)
    if current is None:
        return
    if getattr(current, "_heretic_patched", False):
        return

    original = current

    def _targets_reference_language_model(ct_cfg: object) -> bool:
        # If targets explicitly include "language_model", applying to the subtree would drop the prefix
        # and potentially fail to match. In that case, don't subtree-restrict.
        with suppress(Exception):
            config_groups = getattr(ct_cfg, "config_groups", None)
            if isinstance(config_groups, dict):
                for scheme in config_groups.values():
                    targets = getattr(scheme, "targets", None)
                    if isinstance(targets, list) and any(
                        isinstance(t, str) and "language_model" in t for t in targets
                    ):
                        return True
        return False

    def _patched(self, model: Any, **kwargs: Any):  # noqa: ANN001
        """
        Principled loader:
        - Inspect safetensors index to classify checkpoint deterministically.
        - Apply structure initialization only.
        - Never fall back to `compress_model()` for unknown/mismatched cases; fail loudly.
        """
        from compressed_tensors.quantization import apply_quantization_config  # type: ignore

        checkpoint_files = kwargs.get("checkpoint_files")

        # Best-effort extraction of format (prefer nested compressed-tensors config object).
        expected_format = None
        with suppress(Exception):
            inner = getattr(self.quantization_config, "quantization_config", None)
            expected_format = getattr(inner, "format", None)
        if expected_format is None:
            with suppress(Exception):
                expected_format = getattr(self.quantization_config, "format", None)

        expect_precompressed = bool(
            getattr(self.quantization_config, "is_quantization_compressed", False)
            or getattr(self.quantization_config, "is_sparsification_compressed", False)
        )

        inspected = None
        index_path = None
        # Preferred path: shard files provided.
        if isinstance(checkpoint_files, (list, tuple)) and all(
            isinstance(x, str) for x in checkpoint_files
        ):
            inspected = inspect_checkpoint(
                checkpoint_files=checkpoint_files,  # type: ignore[arg-type]
                expected_format=expected_format,
                expect_precompressed=expect_precompressed,
            )
            index_path = inspected.index_path
        # Fallback: some Transformers builds pass `checkpoint_files=None` for sharded checkpoints,
        # and download/load happens later. We can still resolve the index deterministically.
        elif checkpoint_files is None:
            model_id = None
            with suppress(Exception):
                cfg = getattr(model, "config", None)
                model_id = getattr(cfg, "name_or_path", None) or getattr(cfg, "_name_or_path", None)
            if not isinstance(model_id, str) or not model_id:
                raise RuntimeError(
                    "compressed-tensors: `checkpoint_files` was None and model config did not provide `name_or_path`; "
                    "cannot resolve safetensors index."
                )

            # Resolve index path either from local dir or HF cache.
            if os.path.isdir(model_id):
                candidate = os.path.join(model_id, "model.safetensors.index.json")
                index_path = candidate if os.path.exists(candidate) else None
            if index_path is None:
                try:
                    from transformers.utils.hub import cached_file  # type: ignore

                    index_path = cached_file(
                        model_id,
                        "model.safetensors.index.json",
                        _raise_exceptions_for_missing_entries=True,
                        _raise_exceptions_for_gated_repo=True,
                        _raise_exceptions_for_connection_errors=True,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "compressed-tensors: `checkpoint_files` was None and safetensors index could not be resolved.\n"
                        f"- model_id={model_id}\n"
                        f"- expected_format={expected_format}\n"
                        f"- error={exc}\n"
                        "Remediation: ensure model weights and `model.safetensors.index.json` are accessible/cached."
                    ) from exc

            if not isinstance(index_path, str) or not index_path:
                raise RuntimeError(
                    "compressed-tensors: failed to resolve `model.safetensors.index.json` path."
                )

            index_path = Path(index_path)
            index_keys = load_weight_map_keys(index_path)
            inferred = infer_checkpoint_kind(
                keys=index_keys,
                expected_format=expected_format,
                expect_precompressed=expect_precompressed,
            )
            inspected = inferred
            inspected = type(inferred)(
                kind=inferred.kind,
                index_path=index_path,
                expected_format=inferred.expected_format,
                n_total_keys=inferred.n_total_keys,
                n_dense_weight=inferred.n_dense_weight,
                n_expected_artifacts=inferred.n_expected_artifacts,
                artifacts_suffixes=inferred.artifacts_suffixes,
                dense_suffix=inferred.dense_suffix,
            )
        else:
            raise RuntimeError(
                "compressed-tensors: expected `checkpoint_files` list[str] or None in quantizer preprocess hook; "
                f"got {type(checkpoint_files).__name__}."
            )

        if inspected is None or index_path is None:
            raise RuntimeError(
                "compressed-tensors: could not locate a `*.safetensors.index.json` next to shard files; "
                "refusing to run expensive `compress_model()` fallback.\n"
                f"- first_shard={checkpoint_files[0] if isinstance(checkpoint_files, (list, tuple)) and checkpoint_files else None}\n"
                f"- expected_format={expected_format}\n"
                "Remediation: ensure the sharded safetensors index file is present in the same directory as the shards."
            )

        # Decide whether we can scope to language_model subtree based on index keys.
        index_keys = load_weight_map_keys(index_path)
        ct_quantization_config = self.compressor.quantization_config
        target_model = model
        with suppress(Exception):
            lm = getattr(model, "language_model", None)
            if lm is not None:
                safe_scope = should_scope_to_language_model(
                    index_keys=index_keys,
                    artifact_suffixes=inspected.artifacts_suffixes,
                )
                if safe_scope and not _targets_reference_language_model(ct_quantization_config):
                    target_model = lm

        if inspected.kind == CheckpointKind.PRECOMPRESSED:
            self.run_compressed = True
            with suppress(Exception):
                if hasattr(self.quantization_config, "run_compressed"):
                    self.quantization_config.run_compressed = True
            apply_quantization_config(target_model, ct_quantization_config, run_compressed=True)
            logger(
                "ct_loader: kind=PRECOMPRESSED "
                f"format={inspected.expected_format} run_compressed=True recompress=skip "
                f"index={index_path.name}"
            )
            return

        if inspected.kind == CheckpointKind.DENSE:
            self.run_compressed = False
            with suppress(Exception):
                if hasattr(self.quantization_config, "run_compressed"):
                    self.quantization_config.run_compressed = False
            setattr(self, "_heretic_force_dense", True)
            apply_quantization_config(target_model, ct_quantization_config, run_compressed=False)
            logger(
                "ct_loader: kind=DENSE "
                f"format={inspected.expected_format} run_compressed=False recompress=skip "
                f"index={index_path.name}"
            )
            return

        # INCONSISTENT/UNKNOWN: fail fast with actionable context.
        raise RuntimeError(
            "compressed-tensors: checkpoint format is inconsistent with configuration; refusing to run `compress_model()`.\n"
            f"- kind={inspected.kind}\n"
            f"- expected_format={inspected.expected_format}\n"
            f"- index={index_path}\n"
            f"- n_total_keys={inspected.n_total_keys}\n"
            f"- n_dense_weight_keys={inspected.n_dense_weight}\n"
            f"- n_expected_artifact_keys={inspected.n_expected_artifacts}\n"
            f"- expected_artifact_suffixes={list(inspected.artifacts_suffixes)}\n"
            "Remediation: verify you are loading the correct revision and that the shard index matches the downloaded shards."
        )

    setattr(_patched, "_heretic_patched", True)
    setattr(_patched, "_heretic_original", original)
    CompressedTensorsHfQuantizer._process_model_before_weight_loading = _patched  # type: ignore[assignment]
    logger("Enabled principled compressed-tensors loader shim (index-based).")

    # Also prevent a costly/incorrect decompress sweep if we forced dense loading.
    after = getattr(CompressedTensorsHfQuantizer, "_process_model_after_weight_loading", None)
    if callable(after) and not getattr(after, "_heretic_patched", False):
        original_after = after

        def _patched_after(self, model: Any, **kwargs: Any):  # noqa: ANN001
            if getattr(self, "_heretic_force_dense", False):
                logger("* compressed-tensors: forced dense mode; skipping `decompress_model()` post-load.")
                return
            return original_after(self, model, **kwargs)

        setattr(_patched_after, "_heretic_patched", True)
        setattr(_patched_after, "_heretic_original", original_after)
        CompressedTensorsHfQuantizer._process_model_after_weight_loading = _patched_after  # type: ignore[assignment]

