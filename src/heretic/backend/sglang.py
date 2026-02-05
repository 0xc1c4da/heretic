from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from multiprocessing.reduction import ForkingPickler
import io
import base64

import torch

from .base import (
    BackendMetadata,
    HereticBackend,
    ModuleRef,
    ResidualCaptureResult,
    ScoreResult,
    VTWResult,
)


@dataclass(frozen=True)
class _HTTPResponse:
    status: int
    data: Any


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float = 60.0) -> _HTTPResponse:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return _HTTPResponse(status=getattr(resp, "status", 200), data=json.loads(raw))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SGLang HTTP {e.code} error for {url}: {raw}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"SGLang request failed for {url}: {e}") from e


def _serialize_for_sglang(obj: Any) -> str:
    """Match SGLang MultiprocessingSerializer.serialize(..., output_str=True)."""
    buf = io.BytesIO()
    ForkingPickler(buf).dump(obj)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


class SGLangBackend(HereticBackend):
    """HTTP client for an SGLang server (OpenAI-compatible + admin endpoints)."""

    def __init__(self, *, base_url: str, admin_url: str | None = None, model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.admin_url = (admin_url or self.base_url).rstrip("/")
        self.model = model  # Optional override; else server default

    def get_metadata(self) -> BackendMetadata:
        # Keep lightweight. (We can add a dedicated /metadata endpoint later.)
        return BackendMetadata(
            backend_name="sglang",
            backend_version=None,
            model_id=self.model or "<server-default>",
            tokenizer_id=None,
            max_context_len=None,
            supports={
                "input_ids": True,
                "prompt_ids_sha256": True,
                "capture_layers": True,
                "compute_vtw": True,
                "lora_hot_swap": True,
            },
        )

    def score(self, input_ids_batch: list[list[int]], *, adapter: str | None = None) -> ScoreResult:
        model_name = self.model or "default"
        if adapter:
            model_name = f"{model_name}:{adapter}"

        # OpenAI completions supports token IDs in `prompt`.
        # NOTE: SGLang OpenAI logprobs are top-k; full-vocab KL may need a different path.
        resp = _post_json(
            f"{self.base_url}/v1/completions",
            {
                "model": model_name,
                "prompt": input_ids_batch,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 0,
                "stream": False,
            },
        )

        return ScoreResult(
            logprobs_full=None,
            logprobs_topk=None,
            meta={"raw": resp.data},
        )

    def capture_residuals(
        self,
        input_ids_batch: list[list[int]],
        *,
        capture_layers: list[int],
        capture_point: str = "block_input_last_token",
        adapter: str | None = None,
    ) -> ResidualCaptureResult:
        model_name = self.model or "default"
        if adapter:
            model_name = f"{model_name}:{adapter}"

        resp = _post_json(
            f"{self.base_url}/v1/completions",
            {
                "model": model_name,
                "prompt": input_ids_batch,
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
                "return_hidden_states": True,
                "capture_layers": capture_layers,
            },
        )

        # Extract hidden states from choices.
        choices = resp.data.get("choices", [])
        hs = []
        for c in choices:
            # SGLang places this on the choice, mirroring existing `hidden_states` handling.
            vec = c.get("hidden_states")
            if vec is None:
                raise RuntimeError("SGLang response missing hidden_states (return_hidden_states=True).")
            hs.append(vec)

        # Shape heuristics: SGLang concatenates captured layers along feature dim.
        t = torch.tensor(hs, dtype=torch.float32)
        if t.ndim == 2 and len(capture_layers) > 0 and t.shape[1] % len(capture_layers) == 0:
            d = t.shape[1] // len(capture_layers)
            t = t.view(t.shape[0], len(capture_layers), d)
        elif t.ndim == 1:
            t = t.view(1, 1, -1)
        elif t.ndim == 2:
            t = t.unsqueeze(1)

        return ResidualCaptureResult(
            residuals=t,
            captured_layers=capture_layers,
            capture_point=capture_point,
            meta={"raw": resp.data},
        )

    def compute_vtw(
        self,
        v: torch.Tensor,
        *,
        target: ModuleRef,
        adapter: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> VTWResult:
        if adapter is not None:
            # Server-side support could be added later; keep explicit.
            raise NotImplementedError("compute_vtw(adapter=...) is not supported yet.")

        payload = {
            "name": target.module_path,
            "v": v.detach().to(torch.float32).cpu().tolist(),
            "dtype": "float32",
        }
        resp = _post_json(f"{self.admin_url}/compute_vtw", payload)
        if "vtw" not in resp.data:
            raise RuntimeError(f"Unexpected /compute_vtw response: {resp.data}")
        vtw = torch.tensor(resp.data["vtw"], dtype=torch.float32)
        return VTWResult(
            target=target,
            vtw=vtw,
            implementation=resp.data.get("implementation"),
            meta={"raw": resp.data},
        )

    def load_adapter(self, *, name: str, tensors: dict[str, torch.Tensor], config: dict) -> None:
        # SGLang expects a dict of tensors serialized with ForkingPickler + base64.
        cpu_tensors = {k: v.detach().cpu() for k, v in tensors.items()}
        payload = {
            "lora_name": name,
            "config_dict": config,
            "serialized_tensors": _serialize_for_sglang(cpu_tensors),
            "pinned": False,
            "added_tokens_config": None,
            "lora_id": None,
        }
        _post_json(f"{self.admin_url}/load_lora_adapter_from_tensors", payload, timeout_s=300.0)

    def unload_adapter(self, *, name: str) -> None:
        _post_json(
            f"{self.admin_url}/unload_lora_adapter",
            {"lora_name": name, "lora_id": None},
            timeout_s=60.0,
        )

