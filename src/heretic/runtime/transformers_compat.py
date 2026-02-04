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
from types import SimpleNamespace
from typing import Any, Callable
import os


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

    # Heuristic: presence of any of these parameter suffixes strongly indicates that the checkpoint
    # already contains compressed-tensors artifacts (qparams and/or packed weights).
    # If present, we should never run the expensive in-memory `compress_model()` sweep.
    #
    # Notes:
    # - Quantization params may be stored as <base>_{scale,zero_point,g_idx} where base is one of:
    #   {weight,input,output}.
    # - Some compressors store packed weights as `weight_packed` plus auxiliary metadata.
    # - Sparse formats store `compressed/bitmask/...` style tensors.
    compressed_suffixes = {
        ".weight_scale",
        ".weight_zero_point",
        ".weight_g_idx",
        ".weight_packed",
        ".weight_shape",
        ".weight_global_scale",
        ".input_scale",
        ".input_zero_point",
        ".input_g_idx",
        ".output_scale",
        ".output_zero_point",
        ".output_g_idx",
        ".scale_packed",
        ".meta",
        ".compressed",
        ".bitmask",
        ".row_offsets",
    }

    def _collect_checkpoint_keys(checkpoint_files: object, *, max_shards: int = 8) -> list[str]:
        """
        Collect checkpoint parameter names cheaply.

        Preference order:
        1) Read `*.safetensors.index.json` next to the shard files (fast, complete, no tensor IO)
        2) Fallback: open a few smallest shards and list keys (fast, partial)
        """
        if checkpoint_files is None:
            return []
        if isinstance(checkpoint_files, (str, bytes)):
            files = [checkpoint_files]
        elif isinstance(checkpoint_files, (list, tuple)):
            files = list(checkpoint_files)
        else:
            return []

        safetensors_files = [f for f in files if isinstance(f, str) and f.endswith(".safetensors")]
        if not safetensors_files:
            return []

        # 1) Index-based (best).
        with suppress(Exception):
            d = os.path.dirname(safetensors_files[0])
            for name in os.listdir(d):
                if name.endswith(".safetensors.index.json"):
                    import json

                    idx = os.path.join(d, name)
                    data = json.loads(open(idx, "r", encoding="utf-8").read())
                    wm = data.get("weight_map")
                    if isinstance(wm, dict):
                        keys = [k for k in wm.keys() if isinstance(k, str)]
                        if keys:
                            return keys

        # 2) Fallback: shard sampling.
        try:
            from safetensors import safe_open  # type: ignore
        except Exception:
            return []

        with suppress(Exception):
            safetensors_files.sort(key=lambda p: os.path.getsize(p))

        keys: list[str] = []
        for path in safetensors_files[: max(1, int(max_shards))]:
            try:
                with safe_open(path, framework="pt", device="cpu") as sf:
                    for k in sf.keys():
                        if isinstance(k, str):
                            keys.append(k)
            except Exception:
                continue
        return keys

    def _checkpoint_looks_precompressed(checkpoint_files: object) -> bool:
        keys = _collect_checkpoint_keys(checkpoint_files)
        return any(isinstance(k, str) and any(k.endswith(sfx) for sfx in compressed_suffixes) for k in keys)

    def _checkpoint_looks_dense_only(checkpoint_files: object) -> bool:
        """
        Best-effort detection that the checkpoint does *not* contain compressed artifacts.

        If true, we can avoid `compress_model()` by forcing `run_compressed=False` so weights
        load into standard dense `nn.Linear.weight` parameters.
        """
        keys = _collect_checkpoint_keys(checkpoint_files)
        if not keys:
            return False
        if any(isinstance(k, str) and any(k.endswith(sfx) for sfx in compressed_suffixes) for k in keys):
            return False
        return any(isinstance(k, str) and k.endswith(".weight") for k in keys)

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
        # Keep behavior identical to upstream, except for skipping the expensive recompress sweep.
        from compressed_tensors.quantization import apply_quantization_config  # type: ignore

        ct_quantization_config = self.compressor.quantization_config

        # Auto-restrict to language model subtree for composite multimodal wrappers when safe.
        target_model = model
        with suppress(Exception):
            lm = getattr(model, "language_model", None)
            if lm is not None and not _targets_reference_language_model(ct_quantization_config):
                target_model = lm

        checkpoint_files = kwargs.get("checkpoint_files")
        # Case 1: checkpoint already contains compressed artifacts -> run compressed wrappers, skip recompress.
        if _checkpoint_looks_precompressed(checkpoint_files):
            self.run_compressed = True
            with suppress(Exception):
                if hasattr(self.quantization_config, "run_compressed"):
                    self.quantization_config.run_compressed = True
            apply_quantization_config(target_model, ct_quantization_config, self.run_compressed)
            logger("* compressed-tensors: checkpoint appears pre-compressed; skipping `compress_model()`.")
            return

        # Case 2: checkpoint looks dense-only -> force dense execution to avoid recompress.
        if _checkpoint_looks_dense_only(checkpoint_files):
            self.run_compressed = False
            with suppress(Exception):
                if hasattr(self.quantization_config, "run_compressed"):
                    self.quantization_config.run_compressed = False
            setattr(self, "_heretic_force_dense", True)
            apply_quantization_config(target_model, ct_quantization_config, self.run_compressed)
            logger("* compressed-tensors: checkpoint looks dense; forcing run_compressed=False and skipping `compress_model()`.")
            return

        # Unknown: default to upstream behavior (may be slow, but safest).
        apply_quantization_config(target_model, ct_quantization_config, self.run_compressed)
        logger("* compressed-tensors: checkpoint format uncertain; falling back to `compress_model()`.")
        self.compressor.compress_model(model=target_model)

    setattr(_patched, "_heretic_patched", True)
    setattr(_patched, "_heretic_original", original)
    CompressedTensorsHfQuantizer._process_model_before_weight_loading = _patched  # type: ignore[assignment]
    logger("Enabled compressed-tensors skip-recompress shim (auto-detect).")

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

