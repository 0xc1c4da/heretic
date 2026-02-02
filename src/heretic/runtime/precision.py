# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, Linear, Module


@dataclass(frozen=True)
class PrecisionSettings:
    # "auto" | "strict" | "off"
    policy: str
    # "auto" | "bfloat16" | "float16" | "float32"
    fallback_dtype: str
    debug: bool


class KernelCapabilityProbe:
    """
    Runtime probe of op+dtype support.

    This intentionally treats FP8 as unsupported for standard torch ops:
    - FP8 base layers may use custom kernels (e.g., transformers FP8Linear)
    - But generic PyTorch ops (including LoRA matmuls) will fail on float8 tensors
    """

    def __init__(self):
        self._cache: dict[tuple[str, torch.dtype, str, int | None], bool] = {}

    def is_supported(self, op: str, dtype: torch.dtype, device: torch.device) -> bool:
        if device.type == "cpu":
            return "float8" not in str(dtype)

        key = (op, dtype, device.type, device.index)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        ok = self._probe(op, dtype, device)
        self._cache[key] = ok
        return ok

    def fallback_dtype(self, requested: torch.dtype) -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if requested == torch.float32:
            return torch.float32
        return torch.float16

    def _probe(self, op: str, dtype: torch.dtype, device: torch.device) -> bool:
        if "float8" in str(dtype):
            return False
        try:
            if op == "linear":
                x = torch.randn(2, 4, device=device, dtype=dtype)
                w = torch.randn(3, 4, device=device, dtype=dtype)
                b = torch.randn(3, device=device, dtype=dtype)
                _ = F.linear(x, w, b)
                return True
            if op == "conv2d":
                x = torch.randn(1, 3, 8, 8, device=device, dtype=dtype)
                w = torch.randn(4, 3, 3, 3, device=device, dtype=dtype)
                b = torch.randn(4, device=device, dtype=dtype)
                _ = F.conv2d(x, w, b, stride=1, padding=1)
                return True
            if op == "matmul":
                x = torch.randn(4, 4, device=device, dtype=dtype)
                y = torch.randn(4, 4, device=device, dtype=dtype)
                _ = torch.matmul(x, y)
                return True
        except Exception:
            return False
        return False


class PrecisionPolicy:
    def __init__(self, settings: PrecisionSettings):
        self.settings = settings
        self.probe = KernelCapabilityProbe()
        self._warned: set[tuple[str, torch.dtype]] = set()

    @classmethod
    def from_settings(cls, settings: Any) -> PrecisionPolicy:
        policy = getattr(settings, "precision_policy", "auto")
        fallback_dtype = getattr(settings, "precision_fallback_dtype", "auto")
        debug = getattr(settings, "precision_debug", False)
        return cls(
            PrecisionSettings(
                policy=str(policy).strip().lower(),
                fallback_dtype=str(fallback_dtype).strip().lower(),
                debug=bool(debug),
            )
        )

    def resolve_model_dtype(self, requested: str) -> torch.dtype | str:
        if requested == "auto":
            return "auto"
        return getattr(torch, requested)

    def is_op_dtype_supported(self, op: str, dtype: torch.dtype, device: torch.device) -> bool:
        if self.settings.policy == "off":
            return True
        return self.probe.is_supported(op, dtype, device)

    def op_fallback_dtype(self, requested: torch.dtype) -> torch.dtype:
        if self.settings.fallback_dtype == "auto":
            return self.probe.fallback_dtype(requested)
        return getattr(torch, self.settings.fallback_dtype)

    def should_apply_hooks(self) -> bool:
        return self.settings.policy != "off"

    def warn_once(self, op: str, dtype: torch.dtype) -> bool:
        key = (op, dtype)
        if key in self._warned:
            return False
        self._warned.add(key)
        return True

    def log_probe_matrix(self, logger: Callable[[str], None]) -> None:
        if not self.settings.debug:
            return
        ops = ["linear", "conv2d", "matmul"]
        dtypes: list[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]
        for name in ("float8_e4m3fn", "float8_e5m2"):
            dt = getattr(torch, name, None)
            if dt is not None:
                dtypes.append(dt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger("[bold]Precision probe matrix (runtime capability)[/]")
        for op in ops:
            results = []
            for dtype in dtypes:
                supported = self.probe.is_supported(op, dtype, device)
                results.append(f"{dtype}: {'ok' if supported else 'fail'}")
            logger(f"  * {op}: {', '.join(results)}")


class PrecisionApplier:
    """
    Applies precision-related hooks:
    - LoRA wrapper input casting (pre-hook): enforce compute dtype boundary
    - FP8Linear output casting (post-hook): prevent float8 tensors from leaking downstream
    - General fallback for Linear/Conv2d: auto-cast when kernels unsupported
    """

    def __init__(
        self,
        policy: PrecisionPolicy,
        logger: Callable[[str], None],
        *,
        compute_dtype: torch.dtype | None = None,
    ):
        self.policy = policy
        self.logger = logger
        self.compute_dtype = compute_dtype
        # IMPORTANT: caches are keyed by module identity, not name.
        # Names from `named_modules()` are not globally unique when we traverse detached
        # subgraphs (e.g. a non-registered `base_layer`), so name-keyed caches can collide.
        self._module_fallbacks: weakref.WeakKeyDictionary[Module, torch.dtype] = (
            weakref.WeakKeyDictionary()
        )
        self._module_no_fallback: weakref.WeakSet[Module] = weakref.WeakSet()
        self._warned_by_module: weakref.WeakKeyDictionary[Module, set[torch.dtype]] = (
            weakref.WeakKeyDictionary()
        )

    def apply(self, model: Module) -> None:
        if not self.policy.should_apply_hooks():
            return

        linear_count = 0
        conv_count = 0
        fp8_count = 0
        lora_count = 0

        FP8Linear = None
        try:
            from transformers.integrations.finegrained_fp8 import FP8Linear
        except ImportError:
            pass

        # Graph-aware traversal:
        # - Walks registered submodules (`named_modules`)
        # - Also walks detached `base_layer` modules even if not registered as submodules
        seen_ids: set[int] = set()
        q: deque[tuple[str, Module]] = deque(model.named_modules())

        while q:
            name, module = q.popleft()
            mid = id(module)
            if mid in seen_ids:
                continue
            seen_ids.add(mid)

            # Always traverse base_layer edge if present.
            base_layer = getattr(module, "base_layer", None)
            if isinstance(base_layer, Module):
                child_name = f"{name}.base_layer" if name else "base_layer"
                q.append((child_name, base_layer))

            # Skip if already hooked.
            if getattr(module, "_precision_policy_applied", False):
                continue

            kind = self._apply_to_one_module(module, name, FP8Linear)
            if kind == "lora":
                lora_count += 1
            elif kind == "fp8":
                fp8_count += 1
            elif kind == "linear":
                linear_count += 1
            elif kind == "conv2d":
                conv_count += 1

        if self.policy.settings.debug:
            parts = [
                f"linear={linear_count}" if linear_count else None,
                f"conv2d={conv_count}" if conv_count else None,
                f"fp8_linear={fp8_count}" if fp8_count else None,
                f"lora_wrappers={lora_count}" if lora_count else None,
            ]
            msg = ", ".join([p for p in parts if p])
            self.logger(f"[bold]Precision hooks applied[/]: {msg}")

    def _apply_to_one_module(self, module: Module, name: str, FP8Linear: Any | None) -> str | None:
        # LoRA wrappers: force input dtype to compute dtype (or adapter dtype).
        if self._looks_like_lora_wrapper(module):
            self._attach_lora_input_hook(module, name)
            return "lora"

        is_fp8 = FP8Linear is not None and isinstance(module, FP8Linear)
        if is_fp8:
            self._attach_fp8_output_hook(module, name)
            return "fp8"

        if isinstance(module, Linear):
            self._attach_fallback_hooks(module, name, op="linear")
            return "linear"

        if isinstance(module, Conv2d):
            self._attach_fallback_hooks(module, name, op="conv2d")
            return "conv2d"

        return None

    def _attach_fp8_output_hook(self, module: Module, name: str) -> None:
        module._precision_policy_name = name
        module._precision_policy_applied = True

        def post_hook(mod: Module, args: Any, output: Any) -> Any:
            if isinstance(output, Tensor) and "float8" in str(output.dtype):
                target = self.compute_dtype or self.policy.op_fallback_dtype(output.dtype)
                return output.to(target)
            return output

        module.register_forward_hook(post_hook)

    def _looks_like_lora_wrapper(self, module: Module) -> bool:
        # Duck-typed to avoid PEFT internals dependency.
        return (
            hasattr(module, "base_layer")
            and hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
        )

    def _get_lora_compute_dtype(self, module: Module) -> torch.dtype | None:
        # Prefer dtype of adapter weights.
        for attr_name in ("lora_A", "lora_B"):
            d = getattr(module, attr_name, None)
            values = []
            if isinstance(d, dict):
                values = list(d.values())
            else:
                values = list(getattr(d, "values", lambda: [])())
            for sub in values:
                w = getattr(sub, "weight", None)
                if isinstance(w, Tensor):
                    return w.dtype
        return None

    def _attach_lora_input_hook(self, module: Module, name: str) -> None:
        module._precision_policy_name = name
        module._precision_policy_applied = True

        def pre_hook(mod: Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            # NOTE: We currently cast only positional args. Some architectures may route
            # activations via kwargs; if encountered, consider a guarded kwargs-cast path.
            x = self._first_tensor(args)
            if x is None:
                return None
            target_dtype = self.compute_dtype or self._get_lora_compute_dtype(mod)
            if target_dtype is None or x.dtype == target_dtype:
                return None
            return self._cast_tensors(args, target_dtype)

        module.register_forward_pre_hook(pre_hook)

    def _attach_fallback_hooks(self, module: Module, name: str, *, op: str) -> None:
        module._precision_policy_name = name
        module._precision_policy_applied = True
        module._precision_policy_op = op

        def pre_hook(mod: Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            # Only positional args to avoid keyword conflicts with transformers decorators.
            # NOTE: Some models may pass tensors via kwargs; if encountered, consider a guarded
            # kwargs-cast path that only targets known activation keys.
            input_tensor = self._first_tensor(args)
            if input_tensor is None:
                return None

            if mod in self._module_no_fallback:
                return None

            cached = self._module_fallbacks.get(mod)
            if cached is not None:
                if cached == input_tensor.dtype:
                    return None
                cast_args = self._cast_tensors(args, cached)
                restore = _prepare_cast(mod, cached)
                _push_restore(mod, restore)
                return cast_args

            if self.policy.is_op_dtype_supported(op, input_tensor.dtype, input_tensor.device):
                self._module_no_fallback.add(mod)
                return None

            fallback = self._choose_fallback(op, input_tensor)
            if self._warn_module_once(mod, input_tensor.dtype):
                self.logger(
                    f"[yellow]Precision fallback for {op}: {input_tensor.dtype} -> {fallback}[/]"
                )

            self._module_fallbacks[mod] = fallback

            cast_args = self._cast_tensors(args, fallback)
            restore = _prepare_cast(mod, fallback)
            _push_restore(mod, restore)
            return cast_args

        def post_hook(mod: Module, args: Any, output: Any) -> None:
            restore = _pop_restore(mod)
            if restore is not None:
                _restore_cast(restore)

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook, always_call=True)

    def _warn_module_once(self, module: Module, dtype: torch.dtype) -> bool:
        warned = self._warned_by_module.get(module)
        if warned is None:
            warned = set()
            self._warned_by_module[module] = warned
        if dtype in warned:
            return False
        warned.add(dtype)
        return True

    def _choose_fallback(self, op: str, input_tensor: Tensor) -> torch.dtype:
        fallback = self.policy.op_fallback_dtype(input_tensor.dtype)
        if self.policy.is_op_dtype_supported(op, fallback, input_tensor.device):
            return fallback
        return torch.float32

    def _first_tensor(self, args: tuple[Any, ...]) -> Tensor | None:
        for value in args:
            if isinstance(value, Tensor):
                return value
        return None

    def _cast_tensors(self, args: tuple[Any, ...], dtype: torch.dtype) -> tuple[Any, ...]:
        return tuple(self._cast_tree(v, dtype) for v in args)

    def _cast_tree(self, value: Any, dtype: torch.dtype) -> Any:
        if isinstance(value, Tensor):
            return value if value.dtype == dtype else value.to(dtype)
        if isinstance(value, (list, tuple)):
            return type(value)(self._cast_tree(v, dtype) for v in value)
        if isinstance(value, dict):
            return {k: self._cast_tree(v, dtype) for k, v in value.items()}
        return value


def _prepare_cast(
    module: Module,
    dtype: torch.dtype,
) -> tuple[list[tuple[Tensor, Tensor]], list[tuple[Module, str, Tensor]]]:
    original_data: list[tuple[Tensor, Tensor]] = []
    original_attrs: list[tuple[Module, str, Tensor]] = []

    for param in module.parameters(recurse=False):
        if param.dtype != dtype:
            original_data.append((param, param.data))
            param.data = param.data.to(dtype)

    _cast_tensor_attrs(module, dtype, original_attrs)
    return original_data, original_attrs


def _restore_cast(
    restore_info: tuple[list[tuple[Tensor, Tensor]], list[tuple[Module, str, Tensor]]],
) -> None:
    original_data, original_attrs = restore_info
    for param, data in original_data:
        param.data = data
    for mod, name, value in original_attrs:
        setattr(mod, name, value)


def _push_restore(
    module: Module,
    restore_info: tuple[list[tuple[Tensor, Tensor]], list[tuple[Module, str, Tensor]]],
) -> None:
    stack = getattr(module, "_precision_policy_restore_stack", None)
    if stack is None:
        stack = []
        module._precision_policy_restore_stack = stack
    stack.append(restore_info)


def _pop_restore(
    module: Module,
) -> tuple[list[tuple[Tensor, Tensor]], list[tuple[Module, str, Tensor]]] | None:
    stack = getattr(module, "_precision_policy_restore_stack", None)
    if not stack:
        return None
    return stack.pop()


def _cast_tensor_attrs(
    module: Module,
    dtype: torch.dtype,
    original_attrs: list[tuple[Module, str, Tensor]],
) -> None:
    for name in ("weight", "bias"):
        attr = getattr(module, name, None)
        if isinstance(attr, Tensor) and attr.dtype != dtype:
            # Don't cast quantized tensors (bitsandbytes, etc).
            if hasattr(attr, "quant_state"):
                continue
            original_attrs.append((module, name, attr))
            setattr(module, name, attr.to(dtype))

