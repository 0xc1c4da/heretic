from __future__ import annotations

import base64
import json
import pickle
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np

from .base import (
    BackendMetadata,
    HereticBackend,
    ModuleRef,
    ResidualCaptureResult,
    ScoreResult,
    TokenizeChatResult,
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
    """Serialize tensors safely for SGLang's SafeUnpickler over HTTP.

    Important: do NOT use `multiprocessing.reduction.ForkingPickler` here, because it can
    encode tensor storages using file descriptors (resource_sharer), which breaks across
    an HTTP boundary (authkey mismatch).
    """
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(payload).decode("utf-8")


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
                # Only true once the server echoes hashes (e.g., via sgl_ext.prompt_ids_sha256).
                "prompt_ids_sha256": False,
                "capture_layers": True,
                "logprobs_full": True,
                "compute_vtw": True,
                "lora_hot_swap": True,
                "tokenize_chat": True,
                "generate_text": True,
            },
        )

    def tokenize_chat(
        self,
        chats: list[list[dict[str, Any]]],
        *,
        continue_final_message: bool = False,
    ) -> TokenizeChatResult:
        resp = _post_json(
            f"{self.base_url}/heretic/tokenize_chat",
            {
                "chats": chats,
                "continue_final_message": continue_final_message,
            },
            timeout_s=120.0,
        )
        token_ids = resp.data.get("token_ids")
        prompt_ids_sha256 = resp.data.get("prompt_ids_sha256")
        if not isinstance(token_ids, list):
            raise RuntimeError(f"Unexpected /heretic/tokenize_chat response: {resp.data}")
        return TokenizeChatResult(
            token_ids=token_ids,
            prompt_ids_sha256=prompt_ids_sha256 if isinstance(prompt_ids_sha256, list) else None,
            meta={"raw": resp.data},
        )

    def score(self, input_ids_batch: list[list[int]], *, adapter: str | None = None) -> ScoreResult:
        resp = _post_json(
            f"{self.base_url}/heretic/score_full_vocab",
            {"input_ids": input_ids_batch, "lora_id": adapter},
            timeout_s=300.0,
        )

        data = resp.data
        b64_list = data.get("logprobs_full_fp16_b64")
        shapes = data.get("shape")
        dtype = data.get("dtype")
        if not isinstance(b64_list, list) or not isinstance(shapes, list) or dtype != "float16":
            raise RuntimeError(f"Unexpected /heretic/score_full_vocab response: {data}")
        if len(b64_list) != len(input_ids_batch) or len(shapes) != len(input_ids_batch):
            raise RuntimeError(
                f"Batch size mismatch in /heretic/score_full_vocab: {len(b64_list)=} {len(shapes)=} {len(input_ids_batch)=}"
            )

        rows: list[torch.Tensor] = []
        for b64, shape in zip(b64_list, shapes):
            if not isinstance(b64, str) or not isinstance(shape, list) or len(shape) != 1:
                raise RuntimeError(f"Unexpected row encoding in /heretic/score_full_vocab: {shape=}")
            vocab = int(shape[0])
            raw = base64.b64decode(b64.encode("ascii"))
            arr = np.frombuffer(raw, dtype=np.float16)
            if arr.size != vocab:
                raise RuntimeError(
                    f"Decoded fp16 size mismatch in /heretic/score_full_vocab: {arr.size=} {vocab=}"
                )
            rows.append(torch.from_numpy(arr.astype(np.float32, copy=False)))

        logprobs_full = torch.stack(rows, dim=0)
        return ScoreResult(
            logprobs_full=logprobs_full,
            logprobs_topk=None,
            meta={"raw": data},
        )

    def generate_text(
        self,
        input_ids_batch: list[list[int]],
        *,
        max_new_tokens: int,
        adapter: str | None = None,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate completions for token-id prompts via SGLang /generate endpoint.

        We prefer /generate over /v1/completions here because it supports `lora_id`
        directly and has a simpler, more stable schema for batched token-id prompts.
        """
        resp = _post_json(
            f"{self.base_url}/generate",
            {
                "input_ids": input_ids_batch,
                "sampling_params": {
                    "max_new_tokens": int(max_new_tokens),
                    "temperature": float(temperature),
                },
                "stream": False,
                "return_logprob": False,
                "lora_id": adapter,
            },
            timeout_s=300.0,
        )
        outputs = resp.data
        if not isinstance(outputs, list):
            raise RuntimeError(f"Unexpected /generate response: {outputs}")

        texts: list[str] = []
        for out in outputs:
            if not isinstance(out, dict):
                raise RuntimeError(f"Unexpected /generate item: {out}")
            t = out.get("text")
            if not isinstance(t, str):
                t = ""
            texts.append(t)
        return texts

    def module_map(self, *, include_projs: list[str] | None = None) -> list[dict[str, Any]]:
        resp = _post_json(
            f"{self.base_url}/heretic/module_map",
            {"include_projs": include_projs},
            timeout_s=300.0,
        )
        data = resp.data
        modules = data.get("modules")
        if not isinstance(modules, list):
            raise RuntimeError(f"Unexpected /heretic/module_map response: {data}")
        return modules

    def compute_vtw_batch(
        self,
        *,
        items: list[dict[str, Any]],
        timeout_s: float = 300.0,
    ) -> list[dict[str, Any]]:
        """Call SGLang /compute_vtw_batch admin endpoint."""
        resp = _post_json(
            f"{self.admin_url}/compute_vtw_batch",
            {"items": items},
            timeout_s=timeout_s,
        )
        data = resp.data
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected /compute_vtw_batch response: {data}")
        return data

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
        """Call SGLang /heretic/build_full_rownorm_lora and decode factors."""
        resp = _post_json(
            f"{self.base_url}/heretic/build_full_rownorm_lora",
            {
                "name": name,
                "v": v.detach().to(torch.float32).cpu().tolist(),
                "weight": float(weight),
                "rank": int(rank),
                "svd_q": svd_q,
                "svd_niter": int(svd_niter),
                "out_dtype": out_dtype,
            },
            timeout_s=timeout_s,
        )
        data = resp.data
        if data.get("dtype") not in ("float16", "bfloat16"):
            raise RuntimeError(f"Unexpected /heretic/build_full_rownorm_lora response: {data}")
        a_b64 = data.get("lora_A_b64")
        b_b64 = data.get("lora_B_b64")
        a_shape = data.get("lora_A_shape")
        b_shape = data.get("lora_B_shape")
        if (
            not isinstance(a_b64, str)
            or not isinstance(b_b64, str)
            or not isinstance(a_shape, list)
            or not isinstance(b_shape, list)
            or len(a_shape) != 2
            or len(b_shape) != 2
        ):
            raise RuntimeError(f"Malformed /heretic/build_full_rownorm_lora response: {data}")

        raw_a = base64.b64decode(a_b64.encode("ascii"))
        raw_b = base64.b64decode(b_b64.encode("ascii"))
        dt = np.float16 if data["dtype"] == "float16" else np.dtype("bfloat16")
        # Note: numpy doesn't have native bfloat16 everywhere; keep float16 for transport.
        if data["dtype"] != "float16":
            raise RuntimeError("bfloat16 transport is not supported by this client yet.")

        arr_a = np.frombuffer(raw_a, dtype=np.float16).reshape((int(a_shape[0]), int(a_shape[1])))
        arr_b = np.frombuffer(raw_b, dtype=np.float16).reshape((int(b_shape[0]), int(b_shape[1])))
        A = torch.from_numpy(arr_a.astype(np.float32, copy=False))
        B = torch.from_numpy(arr_b.astype(np.float32, copy=False))
        return A, B

    def capture_residuals(
        self,
        input_ids_batch: list[list[int]],
        *,
        capture_layers: list[int],
        capture_point: str = "block_input_last_token",
        adapter: str | None = None,
    ) -> ResidualCaptureResult:
        # Prefer native /generate: stable schema, supports batched input_ids + lora_id.
        # Fallback to OpenAI /v1/completions for older servers.
        try:
            resp = _post_json(
                f"{self.base_url}/generate",
                {
                    "input_ids": input_ids_batch,
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                    "stream": False,
                    "return_hidden_states": True,
                    "capture_layers": capture_layers,
                    "lora_id": adapter,
                },
                timeout_s=300.0,
            )
            outputs = resp.data
            if not isinstance(outputs, list):
                raise RuntimeError(f"Unexpected /generate response: {outputs}")

            hs: list[list[float]] = []
            for out in outputs:
                if not isinstance(out, dict):
                    raise RuntimeError(f"Unexpected /generate item: {out}")
                meta = out.get("meta_info") or {}
                hs_steps = meta.get("hidden_states")
                if not isinstance(hs_steps, list) or len(hs_steps) == 0:
                    raise RuntimeError(
                        "SGLang /generate response missing meta_info.hidden_states (return_hidden_states=True)."
                    )
                vec = hs_steps[-1]
                if not isinstance(vec, list):
                    raise RuntimeError(
                        f"Unexpected hidden_states step type in /generate: {type(vec)}"
                    )
                hs.append(vec)

        except Exception:
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
                vec = c.get("hidden_states")
                if vec is None:
                    raise RuntimeError(
                        "SGLang response missing hidden_states (return_hidden_states=True)."
                    )
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

    def load_adapter(self, *, name: str, tensors: dict[str, torch.Tensor], config: dict) -> str:
        # SGLang expects a dict of CPU tensors serialized + base64.
        cpu_tensors = {k: v.detach().cpu() for k, v in tensors.items()}
        payload = {
            "lora_name": name,
            "config_dict": config,
            "serialized_tensors": _serialize_for_sglang(cpu_tensors),
            "pinned": False,
            "added_tokens_config": None,
            "lora_id": None,
        }
        resp = _post_json(
            f"{self.admin_url}/load_lora_adapter_from_tensors",
            payload,
            timeout_s=300.0,
        )
        data = resp.data
        if not isinstance(data, dict) or not data.get("success"):
            raise RuntimeError(f"Unexpected load_lora_adapter_from_tensors response: {data}")
        loaded = data.get("loaded_adapters") or {}
        if not isinstance(loaded, dict) or name not in loaded:
            raise RuntimeError(f"Missing adapter ref in load_lora response: keys={list(loaded) if isinstance(loaded, dict) else loaded}")
        ref = loaded[name]
        if not isinstance(ref, dict) or not isinstance(ref.get("lora_id"), str):
            raise RuntimeError(f"Malformed adapter ref in load_lora response: {ref}")
        return ref["lora_id"]

    def unload_adapter(self, *, name: str) -> None:
        _post_json(
            f"{self.admin_url}/unload_lora_adapter",
            {"lora_name": name, "lora_id": None},
            timeout_s=60.0,
        )

