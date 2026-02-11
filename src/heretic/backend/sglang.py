from __future__ import annotations

import base64
import json
import pickle
import urllib.error
import urllib.request
import uuid
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


def _get_json(url: str, *, timeout_s: float = 60.0) -> _HTTPResponse:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return _HTTPResponse(status=getattr(resp, "status", 200), data=json.loads(raw))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SGLang HTTP {e.code} error for {url}: {raw}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"SGLang request failed for {url}: {e}") from e


def _post_bytes(
    url: str, payload: dict[str, Any], *, timeout_s: float = 60.0
) -> tuple[int, bytes, str | None]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type")
            return getattr(resp, "status", 200), raw, ctype
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
        self._adapter_ids_by_name: dict[str, str] = {}

    def get_metadata(self) -> BackendMetadata:
        # Prefer server-reported metadata when available.
        try:
            resp = _get_json(f"{self.base_url}/heretic/metadata", timeout_s=30.0)
            data = resp.data if isinstance(resp.data, dict) else {}
        except Exception:
            data = {}

        supports = data.get("supports") if isinstance(data, dict) else None
        if not isinstance(supports, dict):
            supports = {}

        return BackendMetadata(
            backend_name="sglang",
            backend_version=data.get("version") if isinstance(data, dict) else None,
            model_id=(data.get("served_model_name") if isinstance(data, dict) else None)
            or (self.model or "<server-default>"),
            tokenizer_id=None,
            max_context_len=None,
            supports={
                "input_ids": True,
                "prompt_ids_sha256": False,
                "capture_layers": True,
                "logprobs_full": True,
                # This client can implement within-call noise measurement by tripling a batched
                # full-vocab score request; no new server endpoint is required. If the server
                # supports paired scoring, we treat noise measurement as supported as well.
                "score_full_vocab_paired_with_noise": bool(
                    supports.get("score_full_vocab_paired", False)
                ),
                "compute_vtw": True,
                "lora_hot_swap": True,
                "tokenize_chat": True,
                "generate_text": True,
                **{k: bool(v) for k, v in supports.items()},
            },
            num_layers=(
                int(data.get("num_layers"))
                if isinstance(data, dict) and isinstance(data.get("num_layers"), int)
                else None
            ),
            hidden_size=(
                int(data.get("hidden_size"))
                if isinstance(data, dict) and isinstance(data.get("hidden_size"), int)
                else None
            ),
            vocab_size=(
                int(data.get("vocab_size"))
                if isinstance(data, dict) and isinstance(data.get("vocab_size"), int)
                else None
            ),
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
        logprobs_full, meta = self._score_full_vocab_with_lora_ids(input_ids_batch, lora_id=adapter)
        return ScoreResult(logprobs_full=logprobs_full, logprobs_topk=None, meta=meta)

    def _score_full_vocab_with_lora_ids(
        self,
        input_ids_batch: list[list[int]],
        *,
        lora_id: str | list[str | None] | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """One-call full-vocab scoring with scalar or per-item LoRA ids."""
        # Prefer compact binary transport when available; fall back to JSON.
        try:
            status, raw, ctype = _post_bytes(
                f"{self.base_url}/heretic/score_full_vocab_bin",
                {"input_ids": input_ids_batch, "lora_id": lora_id},
                timeout_s=300.0,
            )
            if status == 200 and (ctype is None or "application/octet-stream" in ctype):
                if len(raw) < 12 or raw[:4] != b"HSF1":
                    raise RuntimeError("Malformed binary score payload (missing magic).")
                bs = int.from_bytes(raw[4:8], "little", signed=False)
                vocab = int.from_bytes(raw[8:12], "little", signed=False)
                if bs != len(input_ids_batch):
                    raise RuntimeError(
                        f"Binary score batch mismatch: {bs=} {len(input_ids_batch)=}"
                    )
                payload = raw[12:]
                expected = bs * vocab * 2  # fp16 bytes
                if len(payload) != expected:
                    raise RuntimeError(
                        f"Binary score payload size mismatch: {len(payload)=} {expected=}"
                    )
                arr = np.frombuffer(payload, dtype=np.float16).reshape(bs, vocab)
                logprobs_full = torch.from_numpy(arr.astype(np.float32, copy=False))
                return logprobs_full, {"transport": "bin"}
        except Exception as e:
            # Fall back only when the binary endpoint is missing.
            msg = str(e)
            if "404" not in msg and "Not Found" not in msg:
                raise

        resp = _post_json(
            f"{self.base_url}/heretic/score_full_vocab",
            {"input_ids": input_ids_batch, "lora_id": lora_id},
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
                raise RuntimeError(
                    f"Unexpected row encoding in /heretic/score_full_vocab: {shape=}"
                )
            vocab = int(shape[0])
            raw_row = base64.b64decode(b64.encode("ascii"))
            arr = np.frombuffer(raw_row, dtype=np.float16)
            if arr.size != vocab:
                raise RuntimeError(
                    f"Decoded fp16 size mismatch in /heretic/score_full_vocab: {arr.size=} {vocab=}"
                )
            rows.append(torch.from_numpy(arr.astype(np.float32, copy=False)))

        logprobs_full = torch.stack(rows, dim=0)
        return logprobs_full, {"transport": "json", "raw": data}

    def score_full_vocab_paired(
        self,
        input_ids_batch: list[list[int]],
        *,
        adapter: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute base and adapted distributions within one server request/batch."""
        if not input_ids_batch:
            raise ValueError("input_ids_batch must be non-empty")
        b = len(input_ids_batch)
        paired_ids = list(input_ids_batch) + list(input_ids_batch)
        paired_loras: list[str | None] = ([None] * b) + ([str(adapter)] * b)
        both, _meta = self._score_full_vocab_with_lora_ids(paired_ids, lora_id=paired_loras)
        if both.ndim != 2 or both.shape[0] != 2 * b:
            raise RuntimeError(
                f"Unexpected paired score shape: {tuple(both.shape)} for batch {b}"
            )
        base = both[:b]
        adapted = both[b:]
        return base, adapted

    def score_full_vocab_paired_with_noise(
        self,
        input_ids_batch: list[list[int]],
        *,
        adapter: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (base1, adapted, base2) within one server request/batch."""
        if not input_ids_batch:
            raise ValueError("input_ids_batch must be non-empty")
        b = len(input_ids_batch)
        tripled_ids = list(input_ids_batch) + list(input_ids_batch) + list(input_ids_batch)
        tripled_loras: list[str | None] = ([None] * b) + ([str(adapter)] * b) + ([None] * b)
        all3, _meta = self._score_full_vocab_with_lora_ids(tripled_ids, lora_id=tripled_loras)
        if all3.ndim != 2 or all3.shape[0] != 3 * b:
            raise RuntimeError(
                f"Unexpected tripled score shape: {tuple(all3.shape)} for batch {b}"
            )
        base1 = all3[:b]
        adapted = all3[b : 2 * b]
        base2 = all3[2 * b :]
        return base1, adapted, base2

    def score_continuation_nll_paired(
        self,
        *,
        prompt_ids_batch: list[list[int]],
        continuation_ids_batch: list[list[int]],
        adapter: str,
    ) -> tuple[list[float], list[float]]:
        base1, adapted, _base2 = self.score_continuation_nll_paired_with_noise(
            prompt_ids_batch=prompt_ids_batch,
            continuation_ids_batch=continuation_ids_batch,
            adapter=adapter,
        )
        return base1, adapted

    def score_continuation_nll_paired_with_noise(
        self,
        *,
        prompt_ids_batch: list[list[int]],
        continuation_ids_batch: list[list[int]],
        adapter: str,
    ) -> tuple[list[float], list[float], list[float]]:
        """Paired teacher-forced continuation NLL, within one server /generate call."""
        if len(prompt_ids_batch) != len(continuation_ids_batch):
            raise ValueError("prompt_ids_batch and continuation_ids_batch must have same length")
        b = len(prompt_ids_batch)
        if b == 0:
            return ([], [], [])
        if not adapter:
            raise ValueError("adapter must be a non-empty string")

        full_ids: list[list[int]] = []
        start_lens: list[int] = []
        cont_lens: list[int] = []
        for p, c in zip(prompt_ids_batch, continuation_ids_batch, strict=True):
            if len(c) <= 0:
                raise ValueError("continuation must be non-empty")
            full_ids.append(list(p) + list(c))
            start_lens.append(max(0, int(len(p) - 1)))
            cont_lens.append(int(len(c)))

        tripled_ids = list(full_ids) + list(full_ids) + list(full_ids)
        tripled_starts = list(start_lens) + list(start_lens) + list(start_lens)
        tripled_loras: list[str | None] = ([None] * b) + ([str(adapter)] * b) + ([None] * b)

        # Force cache isolation per row.
        extra_key = [uuid.uuid4().hex for _ in range(3 * b)]

        resp = _post_json(
            f"{self.base_url}/generate",
            {
                "input_ids": tripled_ids,
                "sampling_params": {"max_new_tokens": 0, "temperature": 0.0, "top_k": 1},
                "stream": False,
                "return_logprob": True,
                "logprob_start_len": tripled_starts,
                "top_logprobs_num": 0,
                "token_ids_logprob": None,
                "return_text_in_logprobs": False,
                "lora_id": tripled_loras,
                "extra_key": extra_key,
            },
            timeout_s=300.0,
        )
        outs = resp.data
        if not isinstance(outs, list) or len(outs) != 3 * b:
            raise RuntimeError(f"Unexpected /generate logprob response: type={type(outs).__name__} len={getattr(outs,'__len__',lambda:None)()}")

        def _mean_nll(out: dict[str, Any], *, cont_len: int, item: int) -> float:
            if not isinstance(out, dict):
                raise RuntimeError(f"Unexpected /generate item: {out}")
            meta = out.get("meta_info") or {}
            itlp = meta.get("input_token_logprobs")
            if not isinstance(itlp, list) or not itlp:
                raise RuntimeError(
                    "Missing input_token_logprobs in /generate response meta_info. "
                    f"keys={list(meta.keys())}"
                )
            vals: list[float] = []
            for tup in itlp:
                if (
                    isinstance(tup, (list, tuple))
                    and len(tup) >= 1
                    and isinstance(tup[0], (float, int))
                ):
                    vals.append(float(tup[0]))
            if len(vals) < cont_len:
                raise RuntimeError(
                    f"input_token_logprobs too short for continuation: got {len(vals)} need {cont_len} (item={item})"
                )
            vals_suf = vals[-cont_len:]
            return float((-torch.tensor(vals_suf, dtype=torch.float32)).mean().item())

        all_mean_nll: list[float] = []
        for i, out in enumerate(outs):
            cont_len = cont_lens[i % b]
            all_mean_nll.append(_mean_nll(out, cont_len=cont_len, item=i))

        base1 = all_mean_nll[:b]
        adapted = all_mean_nll[b : 2 * b]
        base2 = all_mean_nll[2 * b :]
        return base1, adapted, base2

    def score_continuation_topk_paired_with_noise(
        self,
        *,
        prompt_ids_batch: list[list[int]],
        continuation_ids_batch: list[list[int]],
        adapter: str,
        top_k: int,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Paired top-k logprob distributions for continuation positions (within one /generate call)."""
        if len(prompt_ids_batch) != len(continuation_ids_batch):
            raise ValueError("prompt_ids_batch and continuation_ids_batch must have same length")
        b = len(prompt_ids_batch)
        if b == 0:
            return ([], [], [])
        if not adapter:
            raise ValueError("adapter must be a non-empty string")
        if int(top_k) <= 0:
            raise ValueError("top_k must be positive")

        full_ids: list[list[int]] = []
        start_lens: list[int] = []
        cont_lens: list[int] = []
        for p, c in zip(prompt_ids_batch, continuation_ids_batch, strict=True):
            if len(c) <= 0:
                raise ValueError("continuation must be non-empty")
            full_ids.append(list(p) + list(c))
            start_lens.append(max(0, int(len(p) - 1)))
            cont_lens.append(int(len(c)))

        tripled_ids = list(full_ids) + list(full_ids) + list(full_ids)
        tripled_starts = list(start_lens) + list(start_lens) + list(start_lens)
        tripled_loras: list[str | None] = ([None] * b) + ([str(adapter)] * b) + ([None] * b)
        extra_key = [uuid.uuid4().hex for _ in range(3 * b)]

        resp = _post_json(
            f"{self.base_url}/generate",
            {
                "input_ids": tripled_ids,
                "sampling_params": {"max_new_tokens": 0, "temperature": 0.0, "top_k": 1},
                "stream": False,
                "return_logprob": True,
                "logprob_start_len": tripled_starts,
                "top_logprobs_num": int(top_k),
                "token_ids_logprob": None,
                "return_text_in_logprobs": False,
                "lora_id": tripled_loras,
                "extra_key": extra_key,
            },
            timeout_s=300.0,
        )
        outs = resp.data
        if not isinstance(outs, list) or len(outs) != 3 * b:
            raise RuntimeError(f"Unexpected /generate topk response: {type(outs).__name__}")

        def _parse_topk(out: dict[str, Any], *, cont_len: int, item: int) -> list[dict[int, float]]:
            if not isinstance(out, dict):
                raise RuntimeError(f"Unexpected /generate item: {out}")
            meta = out.get("meta_info") or {}
            xtop = meta.get("input_top_logprobs")
            if not isinstance(xtop, list) or not xtop:
                raise RuntimeError(
                    "Missing input_top_logprobs in /generate response meta_info. "
                    f"keys={list(meta.keys())}"
                )
            pos_dicts: list[dict[int, float]] = []
            for pos in xtop:
                if not isinstance(pos, list):
                    continue
                d: dict[int, float] = {}
                for tup in pos:
                    if (
                        isinstance(tup, (list, tuple))
                        and len(tup) >= 2
                        and isinstance(tup[0], (float, int))
                        and isinstance(tup[1], int)
                    ):
                        d[int(tup[1])] = float(tup[0])
                pos_dicts.append(d)
            if len(pos_dicts) < cont_len:
                raise RuntimeError(
                    f"input_top_logprobs too short for continuation: got {len(pos_dicts)} need {cont_len} (item={item})"
                )
            return pos_dicts[-cont_len:]

        all_topk: list[list[dict[int, float]]] = []
        for i, out in enumerate(outs):
            cont_len = cont_lens[i % b]
            all_topk.append(_parse_topk(out, cont_len=cont_len, item=i))

        base1 = all_topk[:b]
        adapted = all_topk[b : 2 * b]
        base2 = all_topk[2 * b :]
        return base1, adapted, base2

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

    def generate_token_ids(
        self,
        input_ids_batch: list[list[int]],
        *,
        max_new_tokens: int,
        adapter: str | None = None,
        temperature: float = 0.0,
        top_k: int | None = 1,
    ) -> list[list[int]]:
        """Generate token IDs (no detokenization) via /generate.

        This is used to cache base continuations for paired damage metrics.
        """
        sp: dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "temperature": float(temperature)}
        if top_k is not None:
            sp["top_k"] = int(top_k)
        extra_key = [uuid.uuid4().hex for _ in range(len(input_ids_batch))]
        resp = _post_json(
            f"{self.base_url}/generate",
            {
                "input_ids": input_ids_batch,
                "sampling_params": sp,
                "stream": False,
                "return_logprob": False,
                "lora_id": adapter,
                "extra_key": extra_key,
            },
            timeout_s=300.0,
        )
        outputs = resp.data
        if not isinstance(outputs, list):
            raise RuntimeError(f"Unexpected /generate response: {outputs}")
        outs: list[list[int]] = []
        for out in outputs:
            if not isinstance(out, dict):
                raise RuntimeError(f"Unexpected /generate item: {out}")
            ids = out.get("output_ids")
            if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
                raise RuntimeError(
                    f"Unexpected /generate token-id schema: keys={list(out.keys())}"
                )
            outs.append(list(ids))
        return outs

    def module_map(
        self,
        *,
        include_projs: list[str] | None = None,
        include_layers: list[int] | None = None,
        include_experts: list[int] | None = None,
        max_experts_per_layer: int | None = None,
        expert_strategy: str = "first",
    ) -> list[dict[str, Any]]:
        resp = _post_json(
            f"{self.base_url}/heretic/module_map",
            {
                "include_projs": include_projs,
                "include_layers": include_layers,
                "include_experts": include_experts,
                "max_experts_per_layer": max_experts_per_layer,
                "expert_strategy": expert_strategy,
            },
            timeout_s=300.0,
        )
        data = resp.data
        modules = data.get("modules")
        if not isinstance(modules, list):
            raise RuntimeError(f"Unexpected /heretic/module_map response: {data}")
        # When SGLang runs with dp_size > 1, it returns one module list per DP rank:
        # {"modules": [rank0_modules, rank1_modules, ...]}.
        # Those lists should be identical; normalize to a single list for clients.
        if modules and isinstance(modules[0], list):
            modules = modules[0]
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

    def build_packed_w2_full_rownorm(
        self,
        *,
        lora_id: str,
        name: str,
        v: torch.Tensor,
        weight: float,
        rank: int,
        out_dtype: str = "float16",
        svd_q: int | None = None,
        svd_niter: int = 6,
        clear_existing: bool = True,
        timeout_s: float = 1800.0,
    ) -> dict[str, Any]:
        """Build+register packed MoE w2 FULL factors via HTTP endpoint."""
        payload = {
            "lora_id": str(lora_id),
            "name": str(name),
            "v": [float(x) for x in v.detach().to(torch.float32).cpu().tolist()],
            "weight": float(weight),
            "rank": int(rank),
            "svd_q": int(svd_q) if svd_q is not None else None,
            "svd_niter": int(svd_niter),
            "out_dtype": str(out_dtype),
            "clear_existing": bool(clear_existing),
        }
        resp = _post_json(
            f"{self.base_url}/heretic/build_packed_w2_full_rownorm",
            payload,
            timeout_s=timeout_s,
        )
        data = resp.data
        if not isinstance(data, dict) or not bool(data.get("success", False)):
            raise RuntimeError(f"Unexpected /heretic/build_packed_w2_full_rownorm response: {data}")
        return data

    def capture_residuals(
        self,
        input_ids_batch: list[list[int]],
        *,
        capture_layers: list[int],
        capture_point: str = "block_input_last_token",
        adapter: str | None = None,
    ) -> ResidualCaptureResult:
        # Hidden-state capture support can vary by server/model and may effectively only
        # work reliably for single-item requests. To keep the contract stable for Heretic
        # (batch, layers, d_model), we chunk larger batches into per-item requests.
        if len(input_ids_batch) > 1:
            parts: list[torch.Tensor] = []
            raws: list[Any] = []
            for ids in input_ids_batch:
                one = self.capture_residuals(
                    [ids],
                    capture_layers=capture_layers,
                    capture_point=capture_point,
                    adapter=adapter,
                )
                # one.residuals: (1, layers, d_model)
                parts.append(one.residuals[0])
                raws.append((one.meta or {}).get("raw"))
            t = torch.stack(parts, dim=0)
            return ResidualCaptureResult(
                residuals=t,
                captured_layers=capture_layers,
                capture_point=capture_point,
                meta={"raw": raws},
            )

        # Prefer native /generate: stable schema, supports batched input_ids + lora_id.
        # Fallback to OpenAI /v1/completions for older servers.
        try:
            resp = _post_json(
                f"{self.base_url}/generate",
                {
                    "input_ids": input_ids_batch,
                    # Residual capture is a pure prefill operation: we only need the prompt's
                    # hidden states (specifically, the last prompt token). Avoiding decode keeps
                    # server work and payload size lower and also avoids schema ambiguity
                    # across prefill vs decode steps.
                    "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
                    "stream": False,
                    "return_hidden_states": True,
                    "capture_layers": capture_layers,
                    "lora_id": adapter,
                },
                timeout_s=300.0,
            )
        except Exception:
            resp = None

        if resp is not None:
            outputs = resp.data
            if not isinstance(outputs, list):
                raise RuntimeError(f"Unexpected /generate response: {outputs}")

            def _is_num(x: Any) -> bool:
                return isinstance(x, (int, float))

            def _is_list_of_nums(x: Any) -> bool:
                return isinstance(x, list) and (len(x) == 0 or all(_is_num(v) for v in x))

            def _parse_hidden_states(raw_hs: Any) -> torch.Tensor:
                """Parse SGLang meta_info.hidden_states into (layers, d_model) float32.

                SGLang stores `req.hidden_states` as a list of *steps*. For `return_hidden_states=True`
                it uses `CaptureHiddenMode.FULL`, so the prefill step contains per-token vectors:

                - hidden_states: [ step0, step1, ... ]
                  - prefill step:  step0 = [token_vec0, token_vec1, ...]         (tokens x features)
                  - decode step(s): stepk = token_vec                            (features)

                For per-request `capture_layers`, SGLang (when properly configured server-side)
                concatenates the selected layer vectors along the feature dimension, so:
                features = len(capture_layers) * d_model

                Heretic wants the **last prompt token** and the **layer axis** explicitly, so we:
                - pick the last token vector from the prefill step when present
                - reshape features -> (layers, d_model)
                """
                if not isinstance(raw_hs, list) or len(raw_hs) == 0:
                    raise RuntimeError("hidden_states is empty or not a list.")

                vec: list[float] | None = None

                # Prefer the prefill step (tokens x features) and take last token of the prompt.
                for step in raw_hs:
                    if (
                        isinstance(step, list)
                        and len(step) > 0
                        and isinstance(step[-1], list)
                        and _is_list_of_nums(step[-1])
                    ):
                        vec = step[-1]
                        break

                # Fall back to the last step when it's already a vector (decode-style).
                if vec is None:
                    last = raw_hs[-1]
                    if _is_list_of_nums(last):
                        vec = last
                    elif (
                        isinstance(last, list)
                        and len(last) > 0
                        and isinstance(last[-1], list)
                        and _is_list_of_nums(last[-1])
                    ):
                        vec = last[-1]

                if vec is None:
                    raise RuntimeError("Unsupported hidden_states schema (expected numeric vectors).")

                t1 = torch.tensor(vec, dtype=torch.float32)
                if t1.ndim != 1:
                    raise RuntimeError(f"Unexpected hidden_states vector ndim: {t1.ndim}")

                if len(capture_layers) <= 0:
                    return t1.view(1, -1)

                if t1.numel() % len(capture_layers) != 0:
                    # If the server isn't actually returning layer-concatenated vectors,
                    # treat this as a hard error: downstream code assumes per-layer residuals.
                    raise RuntimeError(
                        f"Hidden-state feature dim {t1.numel()} is not divisible by requested layers {len(capture_layers)}. "
                        "This usually means the SGLang server returned only a single layer (e.g., final layer) and did not "
                        "apply per-request capture_layers."
                    )

                d_model = t1.numel() // len(capture_layers)
                return t1.view(len(capture_layers), d_model)

            per_item: list[torch.Tensor] = []
            for out in outputs:
                if not isinstance(out, dict):
                    raise RuntimeError(f"Unexpected /generate item: {out}")
                meta = out.get("meta_info") or {}
                applied = meta.get("capture_layers_applied")
                if applied is not None:
                    if not isinstance(applied, list) or any(
                        not isinstance(x, int) for x in applied
                    ):
                        raise RuntimeError(
                            f"Unexpected capture_layers_applied schema: {type(applied).__name__}"
                        )
                    if list(applied) != list(capture_layers):
                        raise RuntimeError(
                            f"SGLang did not apply requested capture_layers. requested={capture_layers} applied={applied}"
                        )
                hs_steps = meta.get("hidden_states")
                if hs_steps is None:
                    raise RuntimeError("SGLang /generate missing meta_info.hidden_states.")
                per_item.append(_parse_hidden_states(hs_steps))
        else:
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

        if "per_item" in locals():
            # /generate path: already parsed to (layers, d_model) per item.
            t = torch.stack(per_item, dim=0)
        else:
            # Fallback path: older servers may return a flat vector per choice.
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
        # Ensure minimal PEFT-style config keys required by SGLang.
        config = dict(config)
        config.setdefault("peft_type", "LORA")
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
        lora_id = ref["lora_id"]
        self._adapter_ids_by_name[name] = lora_id
        return lora_id

    def unload_adapter(self, *, name: str) -> None:
        lora_id = self._adapter_ids_by_name.get(name)
        _post_json(
            f"{self.admin_url}/unload_lora_adapter",
            {"lora_name": name, "lora_id": lora_id},
            timeout_s=60.0,
        )
        if lora_id is not None:
            # Best-effort cleanup for packed-MoE payload associated with this adapter id.
            try:
                _post_json(
                    f"{self.base_url}/heretic/unload_packed_moe_adapter",
                    {"lora_id": str(lora_id)},
                    timeout_s=60.0,
                )
            except Exception:
                pass
        if lora_id is not None:
            self._adapter_ids_by_name.pop(name, None)

