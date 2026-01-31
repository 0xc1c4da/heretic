# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from __future__ import annotations

from contextlib import contextmanager
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
    def __init__(self):
        self._cache: dict[tuple[str, torch.dtype, str, int | None], bool] = {}

    def is_supported(self, op: str, dtype: torch.dtype, device: torch.device) -> bool:
        key = (op, dtype, device.type, device.index)
        if key in self._cache:
            return self._cache[key]
        result = self._probe(op, dtype, device)
        self._cache[key] = result
        return result

    def fallback_dtype(self, requested: torch.dtype) -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if requested == torch.float32:
            return torch.float32
        return torch.float16

    def _probe(self, op: str, dtype: torch.dtype, device: torch.device) -> bool:
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
        if requested == "auto":
            return "auto"
        return getattr(torch, requested)

    def is_op_dtype_supported(
        self,
        op: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        if self.settings.policy == "off":
            return True
        return self.probe.is_supported(op, dtype, device)

    def op_fallback_dtype(self, requested: torch.dtype) -> torch.dtype:
        if self.settings.fallback_dtype == "auto":
            return self.probe.fallback_dtype(requested)
        return getattr(torch, self.settings.fallback_dtype)

    def should_apply_hooks(self, model: Module) -> bool:
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
    def __init__(self, policy: PrecisionPolicy, logger: Callable[[str], None]):
        self.policy = policy
        self.logger = logger
        self._logged_modules: set[str] = set()
        self._module_fallbacks: dict[str, torch.dtype] = {}
        self._module_no_fallback: set[str] = set()

    def apply(self, model: Module) -> None:
        if not self.policy.should_apply_hooks(model):
            return
        linear_count = 0
        conv_count = 0
        for name, module in model.named_modules():
            if getattr(module, "_precision_policy_applied", False):
                continue
            if isinstance(module, Linear):
                self._attach_hooks(module, name, "linear")
                linear_count += 1
            elif isinstance(module, Conv2d):
                self._attach_hooks(module, name, "conv2d")
                conv_count += 1
            else:
                base_layer = getattr(module, "base_layer", None)
                if isinstance(base_layer, Linear) and not getattr(
                    base_layer, "_precision_policy_applied", False
                ):
                    self._attach_hooks(base_layer, f"{name}.base_layer", "linear")
                    linear_count += 1
                elif isinstance(base_layer, Conv2d) and not getattr(
                    base_layer, "_precision_policy_applied", False
                ):
                    self._attach_hooks(base_layer, f"{name}.base_layer", "conv2d")
                    conv_count += 1
        if self.policy.settings.debug:
            self.logger(
                f"[bold]Precision hooks applied[/]: linear={linear_count}, conv2d={conv_count}"
            )

    def _attach_hooks(self, module: Module, name: str, op: str) -> None:
        module._precision_policy_name = name  # type: ignore[attr-defined]
        module._precision_policy_applied = True  # type: ignore[attr-defined]
        module._precision_policy_op = op  # type: ignore[attr-defined]

        def pre_hook(
            mod: Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            input_tensor = self._first_tensor(args, kwargs)
            if input_tensor is None:
                return args, kwargs
            name = getattr(mod, "_precision_policy_name", None)
            if name and name in self._module_no_fallback:
                return args, kwargs
            cached = self._module_fallbacks.get(name) if name else None
            if cached is not None:
                if cached == input_tensor.dtype:
                    return args, kwargs
                cast_args, cast_kwargs = self._cast_tensors(args, kwargs, cached)
                restore = _prepare_cast(mod, cached)
                _push_restore(mod, restore)
                return cast_args, cast_kwargs
            if self.policy.is_op_dtype_supported(op, input_tensor.dtype, input_tensor.device):
                if name:
                    self._module_no_fallback.add(name)
                return args, kwargs
            fallback = self._choose_fallback(op, input_tensor)
            if self._warn_module_once(mod, input_tensor.dtype):
                self.logger(
                    f"[yellow]Precision fallback for {op}: {input_tensor.dtype} -> {fallback}[/]"
                )
            if name:
                self._module_fallbacks[name] = fallback
            cast_args, cast_kwargs = self._cast_tensors(args, kwargs, fallback)
            restore = _prepare_cast(mod, fallback)
            _push_restore(mod, restore)
            return cast_args, cast_kwargs

        def post_hook(mod: Module, _args: tuple[Any, ...], _output: Any) -> None:
            restore = _pop_restore(mod)
            if restore is not None:
                _restore_cast(restore)

        module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        module.register_forward_hook(post_hook, with_kwargs=True, always_call=True)

    def _first_tensor(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Tensor | None:
        for value in args:
            if isinstance(value, Tensor):
                return value
        for value in kwargs.values():
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

    def _log_module_dtypes(
        self,
        module: Module,
        input_tensor: Tensor,
        error: Exception,
    ) -> None:
        name = getattr(module, "_precision_policy_name", "<unknown>")
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        base_layer = getattr(module, "base_layer", None)
        base_weight = getattr(base_layer, "weight", None) if base_layer is not None else None
        self.logger(
            "[yellow]Precision op failed[/]: "
            f"name={name}, input={input_tensor.dtype}, "
            f"weight={getattr(weight, 'dtype', None)}, "
            f"bias={getattr(bias, 'dtype', None)}, "
            f"base_weight={getattr(base_weight, 'dtype', None)}, "
            f"error={type(error).__name__}: {error}"
        )

    def _cast_tensors(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        dtype: torch.dtype,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        cast_args = tuple(self._cast_tree(value, dtype) for value in args)
        cast_kwargs = {key: self._cast_tree(value, dtype) for key, value in kwargs.items()}
        return cast_args, cast_kwargs

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
    original_data: list[tuple[Tensor, Tensor]] = []
    original_attrs: list[tuple[Module, str, Tensor]] = []
    for param in module.parameters(recurse=True):
        if param.dtype != dtype:
            original_data.append((param, param.data))
            param.data = param.data.to(dtype)
    _cast_tensor_attrs(module, dtype, original_attrs)
    base_layer = getattr(module, "base_layer", None)
    if isinstance(base_layer, Module):
        for param in base_layer.parameters(recurse=True):
            if param.dtype != dtype:
                original_data.append((param, param.data))
                param.data = param.data.to(dtype)
        _cast_tensor_attrs(base_layer, dtype, original_attrs)
    return original_data, original_attrs


def _restore_cast(
    restore: tuple[list[tuple[Tensor, Tensor]], list[tuple[Module, str, Tensor]]],
) -> None:
    original_data, original_attrs = restore
    for param, data in original_data:
        param.data = data
    for mod, name, value in original_attrs:
        setattr(mod, name, value)


def _push_restore(
    module: Module,
    restore: tuple[list[tuple[Tensor, Tensor]], list[tuple[Module, str, Tensor]]],
) -> None:
    stack = getattr(module, "_precision_policy_restore_stack", None)
    if stack is None:
        stack = []
        module._precision_policy_restore_stack = stack  # type: ignore[attr-defined]
    stack.append(restore)


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
            original_attrs.append((module, name, attr))
            setattr(module, name, attr.to(dtype))
