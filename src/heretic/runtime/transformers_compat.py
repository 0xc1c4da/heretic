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

from functools import wraps
from types import SimpleNamespace
from typing import Any, Callable


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

    if model is None or hasattr(model, "generate"):
        # Still ensure generation_config exists (some remote-code models rely on it).
        if model is not None and not hasattr(model, "generation_config") and hasattr(model, "config"):
            try:
                model.generation_config = GenerationConfig.from_model_config(model.config)
            except Exception:
                pass
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
        if not hasattr(model, "generation_config") and hasattr(model, "config"):
            try:
                model.generation_config = GenerationConfig.from_model_config(model.config)
            except Exception:
                pass
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

