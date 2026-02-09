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
    num_layers: int | None = None
    hidden_size: int | None = None
    vocab_size: int | None = None


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


@dataclass(frozen=True)
class TokenizeChatResult:
    """Canonical prompt token IDs produced by a backend (optional)."""

    token_ids: list[list[int]]
    prompt_ids_sha256: list[str] | None = None
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

    def score_full_vocab(
        self,
        input_ids_batch: list[list[int]],
        *,
        adapter: str | None = None,
    ) -> torch.Tensor:
        """Return next-token full-vocab logprobs (batch, vocab)."""
        result = self.score(input_ids_batch, adapter=adapter)
        if result.logprobs_full is None:
            raise NotImplementedError("Backend does not provide full-vocab logprobs.")
        return result.logprobs_full

    def score_full_vocab_paired(
        self,
        input_ids_batch: list[list[int]],
        *,
        adapter: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (base, adapted) full-vocab logprobs for KL computation.

        Semantics (source of truth: local/HF backend):
        - base:   log P_theta(· | x)
        - adapted: log P_{theta+adapter}(· | x)

        Default implementation falls back to two separate calls. Backends with
        cross-call drift (e.g. some SGLang stacks) MUST override this method and
        advertise support via `BackendMetadata.supports["score_full_vocab_paired"]=True`,
        so callers can rely on one-call paired semantics.
        """
        base = self.score_full_vocab(input_ids_batch, adapter=None)
        adapted = self.score_full_vocab(input_ids_batch, adapter=adapter)
        return base, adapted

    @abstractmethod
    def generate_text(
        self,
        input_ids_batch: list[list[int]],
        *,
        max_new_tokens: int,
        adapter: str | None = None,
        temperature: float = 0.0,
    ) -> list[str]:
        raise NotImplementedError

    def tokenize_chat(
        self,
        chats: list[list[dict[str, Any]]],
        *,
        continue_final_message: bool = False,
    ) -> TokenizeChatResult:
        """Tokenize chats using the backend's canonical template/tokenizer.

        Not all backends implement this; Heretic may tokenize locally instead.
        """
        raise NotImplementedError

    def module_map(
        self,
        *,
        include_projs: list[str] | None = None,
        include_layers: list[int] | None = None,
        include_experts: list[int] | None = None,
        max_experts_per_layer: int | None = None,
        expert_strategy: str = "first",
    ) -> list[dict[str, Any]]:
        """Return canonical module descriptors for ablation targeting.

        Implemented by SGLang backend; local backends may not provide a stable map.
        """
        raise NotImplementedError

    def compute_vtw_batch(
        self,
        *,
        items: list[dict[str, Any]],
        timeout_s: float = 300.0,
    ) -> list[dict[str, Any]]:
        """Optional batched v^T W endpoint (SGLang-only)."""
        raise NotImplementedError

    def build_full_rownorm_lora(
        self,
        *,
        name: str,
        v: torch.Tensor,
        weight: float,
        rank: int,
        out_dtype: str = "float16",
        svd_q: int | None = None,
        svd_niter: int = 6,
        timeout_s: float = 600.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional server-side FULL row-norm LoRA builder (SGLang-only)."""
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

