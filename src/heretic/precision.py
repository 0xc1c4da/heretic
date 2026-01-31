# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, Linear, Module


@dataclass(frozen=True)
class PrecisionSettings:
    policy: str
    fallback_dtype: str
    debug: bool


class KernelCapabilityProbe:
    """Probes runtime support for various operations and dtypes."""

    def __init__(self):
        self._cache: dict[tuple[str, torch.dtype, str, int | None], bool] = {}

    def is_supported(self, op: str, dtype: torch.dtype, device: torch.device) -> bool:
        """Checks if a specific operation is supported on the given device and dtype."""
        if device.type == "cpu":
            # Assume CPU support for standard dtypes, but fail for FP8
            if "float8" in str(dtype):
                return False
            return True

        key = (op, dtype, device.type, device.index)
        if key in self._cache:
            return self._cache[key]

        result = self._probe(op, dtype, device)
        self._cache[key] = result
        return result

    def fallback_dtype(self, requested: torch.dtype) -> torch.dtype:
        """Returns the best available fallback dtype."""
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if requested == torch.float32:
            return torch.float32
        return torch.float16

    def _probe(self, op: str, dtype: torch.dtype, device: torch.device) -> bool:
        # Standard PyTorch addmm/linear kernels do not support FP8 yet.
        # Transformers uses custom Triton kernels for FP8Linear, but
        # standard torch operations (like LoRA's matmul) will fail.
        if "float8" in str(dtype):
            return False

        try:
            if op == "linear":
                return self._probe_linear(dtype, device)
            if op == "conv2d":
                return self._probe_conv2d(dtype, device)
            if op == "matmul":
                return self._probe_matmul(dtype, device)
        except Exception:
            return False
        return False

    def _probe_linear(self, dtype: torch.dtype, device: torch.device) -> bool:
        x = torch.randn(2, 4, device=device, dtype=dtype)
        w = torch.randn(3, 4, device=device, dtype=dtype)
        b = torch.randn(3, device=device, dtype=dtype)
        _ = F.linear(x, w, b)
        return True

    def _probe_conv2d(self, dtype: torch.dtype, device: torch.device) -> bool:
        x = torch.randn(1, 3, 8, 8, device=device, dtype=dtype)
        w = torch.randn(4, 3, 3, 3, device=device, dtype=dtype)
        b = torch.randn(4, device=device, dtype=dtype)
        _ = F.conv2d(x, w, b, stride=1, padding=1)
        return True

    def _probe_matmul(self, dtype: torch.dtype, device: torch.device) -> bool:
        x = torch.randn(4, 4, device=device, dtype=dtype)
        y = torch.randn(4, 4, device=device, dtype=dtype)
        _ = torch.matmul(x, y)
        return True


class PrecisionPolicy:
    """Manages precision handling and dtypes resolution."""

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
                policy=policy,
                fallback_dtype=fallback_dtype,
                debug=debug,
            )
        )

    def resolve_model_dtype(
        self,
        requested: str,
        device: torch.device,
    ) -> torch.dtype | str:
        """Resolves the requested dtype string to a torch.dtype or 'auto'."""
        if requested == "auto":
            return "auto"
        return getattr(torch, requested)

    def is_op_dtype_supported(
        self,
        op: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        """Checks if a specific operation is supported on the given device and dtype."""
        if self.settings.policy == "off":
            return True
        return self.probe.is_supported(op, dtype, device)

    def op_fallback_dtype(self, requested: torch.dtype) -> torch.dtype:
        """Returns the fallback dtype for a given requested dtype."""
        if self.settings.fallback_dtype == "auto":
            return self.probe.fallback_dtype(requested)
        return getattr(torch, self.settings.fallback_dtype)

    def should_apply_hooks(self, model: Module) -> bool:
        """Checks if precision hooks should be applied to the model."""
        return self.settings.policy != "off"

    def warn_once(self, op: str, dtype: torch.dtype) -> bool:
        """Warns once per operation and dtype."""
        key = (op, dtype)
        if key in self._warned:
            return False
        self._warned.add(key)
        return True

    def log_probe_matrix(self, logger: Callable[[str], None]) -> None:
        """Logs the precision probe matrix if debug mode is enabled."""
        if not self.settings.debug:
            return
        ops = ["linear", "conv2d", "matmul"]
        dtypes = self._probe_dtypes()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger("[bold]Precision probe matrix (runtime capability)[/]")
        for op in ops:
            results = []
            for dtype in dtypes:
                supported = self.probe.is_supported(op, dtype, device)
                results.append(f"{dtype}: {'ok' if supported else 'fail'}")
            logger(f"  * {op}: {', '.join(results)}")

    def _probe_dtypes(self) -> list[torch.dtype]:
        dtypes: list[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]
        for name in ("float8_e4m3fn", "float8_e5m2"):
            dtype = getattr(torch, name, None)
            if dtype is not None:
                dtypes.append(dtype)
        return dtypes


class PolicyApplier:
    """Applies precision hooks to a model's modules."""

    def __init__(self, policy: PrecisionPolicy, logger: Callable[[str], None]):
        self.policy = policy
        self.logger = logger
        self._logged_modules: set[str] = set()
        self._module_fallbacks: dict[str, torch.dtype] = {}
        self._module_no_fallback: set[str] = set()
        self._failure_logged: set[str] = set()

    def apply(self, model: Module) -> None:
        """Applies precision hooks to all relevant modules in the model."""
        if not self.policy.should_apply_hooks(model):
            return

        linear_count = 0
        conv_count = 0
        fp8_count = 0

        # Attempt to import FP8Linear if available
        FP8Linear = None
        try:
            from transformers.integrations.finegrained_fp8 import FP8Linear
        except ImportError:
            pass

        for name, module in model.named_modules():
            # Skip if already applied to this specific module instance
            if getattr(module, "_precision_policy_applied", False):
                continue

            # Identify module type
            is_linear = isinstance(module, Linear)
            is_conv = isinstance(module, Conv2d)
            is_fp8 = FP8Linear is not None and isinstance(module, FP8Linear)

            if is_fp8:
                self._attach_fp8_output_hook(module, name)
                fp8_count += 1
            elif is_linear:
                self._attach_fallback_hooks(module, name, "linear")
                linear_count += 1
            elif is_conv:
                self._attach_fallback_hooks(module, name, "conv2d")
                conv_count += 1
            else:
                # Handle wrapped modules (like PEFT/LoRA base layers)
                base_layer = getattr(module, "base_layer", None)
                if base_layer is not None and not getattr(base_layer, "_precision_policy_applied", False):
                    self.apply_to_module(base_layer, f"{name}.base_layer", FP8Linear)

        if self.policy.settings.debug:
            counts = [
                f"linear={linear_count}" if linear_count else None,
                f"conv2d={conv_count}" if conv_count else None,
                f"fp8_linear={fp8_count}" if fp8_count else None,
            ]
            msg = ", ".join(filter(None, counts))
            self.logger(f"[bold]Precision hooks applied[/]: {msg}")

    def apply_to_module(self, module: Module, name: str, FP8Linear: Any = None) -> None:
        """Helper to apply hooks to a specific module."""
        if FP8Linear is not None and isinstance(module, FP8Linear):
            self._attach_fp8_output_hook(module, name)
        elif isinstance(module, Linear):
            self._attach_fallback_hooks(module, name, "linear")
        elif isinstance(module, Conv2d):
            self._attach_fallback_hooks(module, name, "conv2d")

    def _attach_fp8_output_hook(self, module: Module, name: str) -> None:
        """Specifically for FP8Linear: ensures outputs are cast back to compute dtype
        so they don't break downstream LoRA/standard operations."""
        module._precision_policy_name = name
        module._precision_policy_applied = True

        def post_hook(mod: Module, args: Any, output: Any) -> Any:
            if isinstance(output, Tensor) and "float8" in str(output.dtype):
                fallback = self.policy.op_fallback_dtype(output.dtype)
                return output.to(fallback)
            return output

        module.register_forward_hook(post_hook)

    def _attach_fallback_hooks(self, module: Module, name: str, op: str) -> None:
        """Attaches pre/post hooks to handle dtype fallback for standard ops."""
        module._precision_policy_name = name
        module._precision_policy_applied = True
        module._precision_policy_op = op

        def pre_hook(mod: Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            # IMPORTANT: We only operate on positional args to avoid keyword conflicts
            # with decorators like @check_model_inputs.
            input_tensor = self._first_tensor(args)
            if input_tensor is None:
                return None

            # Skip if this operation is known to be supported
            name = getattr(mod, "_precision_policy_name", None)
            if name and name in self._module_no_fallback:
                return None

            # Use cached fallback if available
            cached_dtype = self._module_fallbacks.get(name) if name else None
            if cached_dtype is not None:
                if cached_dtype == input_tensor.dtype:
                    return None
                cast_args = self._cast_tensors(args, cached_dtype)
                restore_info = _prepare_cast(mod, cached_dtype)
                _push_restore(mod, restore_info)
                return cast_args

            # Check support for current dtype
            if self.policy.is_op_dtype_supported(op, input_tensor.dtype, input_tensor.device):
                if name:
                    self._module_no_fallback.add(name)
                return None

            # Determine fallback
            fallback = self._choose_fallback(op, input_tensor)
            if self._warn_module_once(mod, input_tensor.dtype):
                self.logger(
                    f"[yellow]Precision fallback for {op}: {input_tensor.dtype} -> {fallback}[/]"
                )

            if name:
                self._module_fallbacks[name] = fallback

            cast_args = self._cast_tensors(args, fallback)
            restore_info = _prepare_cast(mod, fallback)
            _push_restore(mod, restore_info)
            return cast_args

        def post_hook(mod: Module, args: Any, output: Any) -> None:
            restore_info = _pop_restore(mod)
            if restore_info is not None:
                _restore_cast(restore_info)

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook, always_call=True)

    def _first_tensor(self, args: tuple[Any, ...]) -> Tensor | None:
        for value in args:
            if isinstance(value, Tensor):
                return value
        return None

    def _choose_fallback(self, op: str, input_tensor: Tensor) -> torch.dtype:
        fallback = self.policy.op_fallback_dtype(input_tensor.dtype)
        if self.policy.is_op_dtype_supported(op, fallback, input_tensor.device):
            return fallback
        return torch.float32

    def _warn_module_once(self, module: Module, dtype: torch.dtype) -> bool:
        name = getattr(module, "_precision_policy_name", None)
        if name is None:
            return self.policy.warn_once("op", dtype)
        key = f"{name}:{dtype}"
        if key in self._logged_modules:
            return False
        self._logged_modules.add(key)
        return True

    def _cast_tensors(self, args: tuple[Any, ...], dtype: torch.dtype) -> tuple[Any, ...]:
        return tuple(self._cast_tree(value, dtype) for value in args)

    def _cast_tree(self, value: Any, dtype: torch.dtype) -> Any:
        if isinstance(value, Tensor):
            if value.dtype == dtype:
                return value
            return value.to(dtype)
        if isinstance(value, (list, tuple)):
            return type(value)(self._cast_tree(v, dtype) for v in value)
        if isinstance(value, dict):
            return {k: self._cast_tree(v, dtype) for k, v in value.items()}
        return value


def _prepare_cast(
    module: Module,
    dtype: torch.dtype,
) -> tuple[list[tuple[Tensor, Tensor]], list[tuple[Module, str, Tensor]]]:
    """Prepares and applies temporary casting of parameters/buffers for a module."""
    original_data: list[tuple[Tensor, Tensor]] = []
    original_attrs: list[tuple[Module, str, Tensor]] = []

    # Cast parameters directly on the module
    for param in module.parameters(recurse=False):
        if param.dtype != dtype:
            original_data.append((param, param.data))
            param.data = param.data.to(dtype)

    # Cast specific attributes (weight, bias)
    _cast_tensor_attrs(module, dtype, original_attrs)

    return original_data, original_attrs


def _restore_cast(
    restore_info: tuple[list[tuple[Tensor, Tensor]], list[tuple[Module, str, Tensor]]],
) -> None:
    """Restores original dtypes after a forward pass."""
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
            # Don't cast if it's already quantized (bitsandbytes, etc)
            if hasattr(attr, "quant_state"):
                continue
            original_attrs.append((module, name, attr))
            setattr(module, name, attr.to(dtype))
