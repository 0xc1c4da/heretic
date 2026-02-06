from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class BackendMetadata:
    backend_name: str
    backend_version: str | None
    model_id: str
    tokenizer_id: str | None
    max_context_len: int | None
    supports: dict[str, bool]


@dataclass(frozen=True)
class ModuleRef:
    """Canonical reference to a weight-bearing module/parameter in a backend."""

    module_path: str
    kind: Literal["parameter", "linear", "moe_expert", "unknown"] = "unknown"
    layer: int | None = None
    expert_id: int | None = None
    proj: str | None = None


@dataclass(frozen=True)
class ScoreResult:
    """Scoring output needed for KL + refusal metrics."""

    # Next-token logprobs for each prompt in batch.
    # Shape: (batch, vocab) when available; otherwise None (backend may only support top-k).
    logprobs_full: torch.Tensor | None

    # Optional top-k logprobs for each prompt (backend-dependent).
    logprobs_topk: list[dict[str, Any]] | None = None

    # Arbitrary metadata (prompt hashes, timings, etc).
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResidualCaptureResult:
    """Captured residual vectors for direction computation."""

    # Residuals for last prompt token.
    # Shape: (batch, layers, d_model)
    residuals: torch.Tensor
    captured_layers: list[int]
    capture_point: str
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class VTWResult:
    """Result of v^T W compute (vector) for a module."""

    target: ModuleRef
    vtw: torch.Tensor
    implementation: str | None = None
    meta: dict[str, Any] | None = None


class HereticBackend(ABC):
    @abstractmethod
    def get_metadata(self) -> BackendMetadata:
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        input_ids_batch: list[list[int]],
        *,
        adapter: str | None = None,
    ) -> ScoreResult:
        raise NotImplementedError

    @abstractmethod
    def capture_residuals(
        self,
        input_ids_batch: list[list[int]],
        *,
        capture_layers: list[int],
        capture_point: str = "block_input_last_token",
        adapter: str | None = None,
    ) -> ResidualCaptureResult:
        raise NotImplementedError

    @abstractmethod
    def compute_vtw(
        self,
        v: torch.Tensor,
        *,
        target: ModuleRef,
        adapter: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> VTWResult:
        raise NotImplementedError

    @abstractmethod
    def load_adapter(
        self, *, name: str, tensors: dict[str, torch.Tensor], config: dict
    ) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def unload_adapter(self, *, name: str) -> None:
        raise NotImplementedError

