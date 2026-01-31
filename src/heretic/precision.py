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

    def apply(self, model: Module) -> None:
        if not self.policy.should_apply_hooks(model):
            return
        linear_count = 0
        conv_count = 0
        for name, module in model.named_modules():
            if getattr(module, "_precision_policy_applied", False):
                continue
            if isinstance(module, Linear):
                self._wrap_linear(module, name)
                module._precision_policy_applied = True  # type: ignore[attr-defined]
                linear_count += 1
            elif isinstance(module, Conv2d):
                self._wrap_conv2d(module, name)
                module._precision_policy_applied = True  # type: ignore[attr-defined]
                conv_count += 1
        if self.policy.settings.debug:
            self.logger(
                f"[bold]Precision hooks applied[/]: linear={linear_count}, conv2d={conv_count}"
            )

    def _wrap_linear(self, module: Linear, name: str) -> None:
        original_forward = module.forward
        module._precision_policy_name = name  # type: ignore[attr-defined]

        def forward(*args: Any, **kwargs: Any) -> Tensor:
            return self._dispatch_linear(module, original_forward, args, kwargs)

        module.forward = forward  # type: ignore[assignment]

    def _wrap_conv2d(self, module: Conv2d, name: str) -> None:
        original_forward = module.forward
        module._precision_policy_name = name  # type: ignore[attr-defined]

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
            try:
                return original_forward(*args, **kwargs)
            except Exception as error:
                if not self._should_retry(error):
                    raise
                if self.policy.settings.debug:
                    self._log_module_dtypes(module, input_tensor, error)
        fallback = self._choose_fallback("linear", input_tensor)
        if self._warn_module_once(module, input_tensor.dtype):
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
            try:
                return original_forward(*args, **kwargs)
            except Exception as error:
                if not self._should_retry(error):
                    raise
                if self.policy.settings.debug:
                    self._log_module_dtypes(module, input_tensor, error)
        fallback = self._choose_fallback("conv2d", input_tensor)
        if self._warn_module_once(module, input_tensor.dtype):
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

    def _warn_module_once(self, module: Module, dtype: torch.dtype) -> bool:
        name = getattr(module, "_precision_policy_name", None)
        if name is None:
            return self.policy.warn_once("op", dtype)
        key = f"{name}:{dtype}"
        if key in self._logged_modules:
            return False
        self._logged_modules.add(key)
        return True

    def _should_retry(self, error: Exception) -> bool:
        message = str(error).lower()
        return "addmm_cuda" in message or "same dtype" in message

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


@contextmanager
def _temporary_parameter_cast(module: Module, dtype: torch.dtype) -> Iterator[None]:
    original_data: list[tuple[Tensor, Tensor]] = []
    original_attrs: list[tuple[Module, str, Tensor]] = []
    try:
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
        yield
    finally:
        for param, data in original_data:
            param.data = data
        for mod, name, value in original_attrs:
            setattr(mod, name, value)


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
