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

