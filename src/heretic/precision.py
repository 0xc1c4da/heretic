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


class PolicyApplier:
    def __init__(self, policy: PrecisionPolicy, logger: Callable[[str], None]):
        self.policy = policy
        self.logger = logger

    def apply(self, model: Module) -> None:
        if not self.policy.should_apply_hooks(model):
            return
        for module in model.modules():
            if getattr(module, "_precision_policy_applied", False):
                continue
            if isinstance(module, Linear):
                self._wrap_linear(module)
                module._precision_policy_applied = True  # type: ignore[attr-defined]
            elif isinstance(module, Conv2d):
                self._wrap_conv2d(module)
                module._precision_policy_applied = True  # type: ignore[attr-defined]

    def _wrap_linear(self, module: Linear) -> None:
        original_forward = module.forward

        def forward(*args: Any, **kwargs: Any) -> Tensor:
            return self._dispatch_linear(module, original_forward, args, kwargs)

        module.forward = forward  # type: ignore[assignment]

    def _wrap_conv2d(self, module: Conv2d) -> None:
        original_forward = module.forward

        def forward(*args: Any, **kwargs: Any) -> Tensor:
            return self._dispatch_conv2d(module, original_forward, args, kwargs)

        module.forward = forward  # type: ignore[assignment]

    def _dispatch_linear(
        self,
        module: Linear,
        original_forward: Callable[..., Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Tensor:
        input_tensor = self._first_tensor(args)
        if input_tensor is None:
            return original_forward(*args, **kwargs)
        if self.policy.is_op_dtype_supported("linear", input_tensor.dtype, input_tensor.device):
            return original_forward(*args, **kwargs)
        fallback = self._choose_fallback("linear", input_tensor)
        if self.policy.warn_once("linear", input_tensor.dtype):
            self.logger(
                f"[yellow]Precision fallback for linear: {input_tensor.dtype} -> {fallback}[/]"
            )
        cast_args, cast_kwargs = self._cast_tensors(args, kwargs, fallback)
        with _temporary_parameter_cast(module, fallback):
            return original_forward(*cast_args, **cast_kwargs)

    def _dispatch_conv2d(
        self,
        module: Conv2d,
        original_forward: Callable[..., Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Tensor:
        input_tensor = self._first_tensor(args)
        if input_tensor is None:
            return original_forward(*args, **kwargs)
        if self.policy.is_op_dtype_supported("conv2d", input_tensor.dtype, input_tensor.device):
            return original_forward(*args, **kwargs)
        fallback = self._choose_fallback("conv2d", input_tensor)
        if self.policy.warn_once("conv2d", input_tensor.dtype):
            self.logger(
                f"[yellow]Precision fallback for conv2d: {input_tensor.dtype} -> {fallback}[/]"
            )
        cast_args, cast_kwargs = self._cast_tensors(args, kwargs, fallback)
        with _temporary_parameter_cast(module, fallback):
            return original_forward(*cast_args, **cast_kwargs)

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


@contextmanager
def _temporary_parameter_cast(module: Module, dtype: torch.dtype) -> Iterator[None]:
    original_data: list[tuple[Tensor, Tensor]] = []
    try:
        for param in module.parameters(recurse=True):
            if param.dtype != dtype:
                original_data.append((param, param.data))
                param.data = param.data.to(dtype)
        yield
    finally:
        for param, data in original_data:
            param.data = data
