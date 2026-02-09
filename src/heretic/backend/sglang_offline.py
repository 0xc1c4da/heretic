from __future__ import annotations

import base64
import json
import logging
import pickle
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from .base import (
    BackendMetadata,
    HereticBackend,
    ModuleRef,
    ResidualCaptureResult,
    ScoreResult,
    TokenizeChatResult,
    VTWResult,
)

logger = logging.getLogger(__name__)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float))


def _is_list_of_nums(x: Any) -> bool:
    # Empty vectors are not valid hidden states; they usually indicate an empty
    # prompt (no tokens) or an upstream capture failure.
    return isinstance(x, list) and len(x) > 0 and all(_is_num(v) for v in x)


def _parse_hidden_states_last_prompt_token(
    raw_hs: Any,
    *,
    meta: dict[str, Any] | None,
    capture_layers: list[int],
    batch_index: int | None = None,
) -> torch.Tensor:
    """
    Parse SGLang `meta_info.hidden_states` into a (layers, d_model) tensor for the last prompt token.

    Expected schema (preferred):
    - meta_info.hidden_states_schema_version == "v0_steps"
    - meta_info.hidden_states is a list of "steps" where each step is:
      - a tokens×features tensor/list (prefill chunk), or
      - a features vector (decode step)

    When meta_info.prompt_tokens is present, this function indexes into the concatenated token stream
    to select the last prompt token, even under chunked/mixed prefill.
    """
    if not isinstance(raw_hs, list) or len(raw_hs) == 0:
        raise RuntimeError("hidden_states is empty or not a list.")

    schema = meta.get("hidden_states_schema_version") if isinstance(meta, dict) else None
    prompt_tokens = meta.get("prompt_tokens") if isinstance(meta, dict) else None
    prompt_tokens_i = int(prompt_tokens) if isinstance(prompt_tokens, int) else None

    vec_list: list[float] | None = None
    vec_arr: np.ndarray | None = None

    def _step_token_count(step: Any) -> int | None:
        if isinstance(step, np.ndarray):
            if step.ndim == 2:
                return int(step.shape[0])
            if step.ndim == 1:
                return 1
            return None
        if _is_list_of_nums(step):
            return 1
        if isinstance(step, list):
            # list-of-vectors (tokens x features)
            return len(step)
        return None

    def _extract_token_vec(step: Any, token_index: int) -> tuple[np.ndarray | None, list[float] | None]:
        # token_index is within this step (0-based)
        if isinstance(step, np.ndarray):
            if step.ndim == 2:
                if 0 <= token_index < int(step.shape[0]):
                    return step[token_index], None
                return None, None
            if step.ndim == 1:
                return (step if token_index == 0 else None), None
            return None, None
        if _is_list_of_nums(step):
            return None, (step if token_index == 0 else None)
        if isinstance(step, list):
            if 0 <= token_index < len(step) and _is_list_of_nums(step[token_index]):
                return None, step[token_index]
        return None, None

    # Schema-driven parsing:
    # - `v0_steps`: a list of "steps" (typically prefill chunks and/or decode steps).
    #   Select the *last prompt token* by indexing into the concatenated token stream
    #   using `prompt_tokens` when available.
    if schema == "v0_steps" and prompt_tokens_i is not None and prompt_tokens_i > 0:
        seen = 0
        for step in raw_hs:
            n = _step_token_count(step)
            if n is None or n <= 0:
                continue
            prev = seen
            seen += n
            if seen >= prompt_tokens_i:
                local_idx = (prompt_tokens_i - 1) - prev
                vec_arr, vec_list = _extract_token_vec(step, int(local_idx))
                break

    # Fallback: use the last non-empty step's last token/vector.
    if vec_arr is None and vec_list is None:
        for step in reversed(raw_hs):
            n = _step_token_count(step)
            if n is None or n <= 0:
                continue
            vec_arr, vec_list = _extract_token_vec(step, n - 1)
            if vec_arr is not None or vec_list is not None:
                break

    if vec_arr is None and vec_list is None:
        meta_keys = sorted(list(meta.keys())) if isinstance(meta, dict) else []
        raise RuntimeError(
            "Unsupported hidden_states schema (no usable token vector found). "
            f"{schema=} {prompt_tokens_i=} {batch_index=} meta_keys={meta_keys}"
        )

    if vec_arr is not None:
        if vec_arr.ndim != 1:
            raise RuntimeError(f"Unexpected hidden_states vector ndim: {vec_arr.ndim}")
        t1 = torch.from_numpy(vec_arr.astype(np.float32, copy=False))
    else:
        if len(vec_list) == 0:
            raise RuntimeError("hidden_states vector is empty.")
        t1 = torch.tensor(vec_list, dtype=torch.float32)
    if t1.ndim != 1:
        raise RuntimeError(f"Unexpected hidden_states vector ndim: {t1.ndim}")

    if len(capture_layers) <= 0:
        return t1.view(1, -1)

    # Sanity check: if SGLang reports which layers were captured, require it to match our request.
    if isinstance(meta, dict):
        applied = meta.get("capture_layers_applied")
        if isinstance(applied, list) and all(isinstance(x, int) for x in applied):
            if [int(x) for x in applied] != [int(x) for x in capture_layers]:
                raise RuntimeError(
                    "capture_layers mismatch: "
                    f"requested={list(capture_layers)} applied={applied}"
                )

    d_model_meta = None
    if isinstance(meta, dict) and isinstance(meta.get("hidden_states_d_model"), int):
        d_model_meta = int(meta["hidden_states_d_model"])

    if d_model_meta is not None:
        expected = len(capture_layers) * d_model_meta
        if t1.numel() != expected:
            raise RuntimeError(
                "Hidden-state feature dim mismatch vs meta_info.hidden_states_d_model: "
                f"{t1.numel()} != {expected} (layers={len(capture_layers)}, d_model={d_model_meta})."
            )
        d_model = d_model_meta
    else:
        if t1.numel() % len(capture_layers) != 0:
            raise RuntimeError(
                f"Hidden-state feature dim {t1.numel()} is not divisible by requested layers {len(capture_layers)}."
            )
        d_model = t1.numel() // len(capture_layers)

    return t1.view(len(capture_layers), d_model)


def _serialize_for_sglang(obj: Any) -> str:
    # Same format as the HTTP backend: SGLang expects a base64-encoded pickle payload.
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(payload).decode("utf-8")


@dataclass(frozen=True)
class _OfflineGen:
    outputs: list[dict[str, Any]]


class SGLangOfflineBackend(HereticBackend):
    """Embedded (offline) SGLang backend using `sglang.srt.entrypoints.engine.Engine`."""

    def __init__(
        self,
        *,
        model_path: str,
        trust_remote_code: bool = False,
        engine_args: dict[str, Any] | None = None,
        hidden_states_dump_path: str | None = None,
    ):
        try:
            from sglang.version import __version__ as sglang_version
            from sglang.srt.entrypoints.engine import Engine
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to import SGLang Engine.\n\n"
                "If you're working from this repo, make sure you've initialized and installed the vendored submodule:\n"
                "  git submodule update --init --recursive\n"
                "  uv pip install -e vendor/sglang/python\n\n"
                f"Original import error: {e}"
            ) from e

        self._sglang_version = sglang_version
        self._adapter_ids_by_name: dict[str, str] = {}
        self._hidden_states_dump_path = str(hidden_states_dump_path) if hidden_states_dump_path else None

        args = dict(engine_args or {})
        args.setdefault("model_path", model_path)
        args.setdefault("trust_remote_code", bool(trust_remote_code))

        # Heretic requires residual capture + LoRA hot swap.
        args.setdefault("enable_return_hidden_states", True)
        args.setdefault("enable_lora", True)
        # Reasonable defaults for Heretic's typical usage; users can override via engine_args.
        args.setdefault("max_lora_rank", 64)
        args.setdefault("lora_target_modules", ["all"])

        self._engine = Engine(**args)

    def _run(self, coro):
        return self._engine.loop.run_until_complete(coro)

    def _generate_req(self, obj) -> _OfflineGen:
        gen = self._engine.tokenizer_manager.generate_request(obj, None)

        async def _drain_last():
            last = None
            async for item in gen:
                last = item
            return last

        outputs = self._run(_drain_last())
        if outputs is None:
            raise RuntimeError("Unexpected SGLang offline generate output: no yields.")
        if not isinstance(outputs, list) or any(not isinstance(x, dict) for x in outputs):
            raise RuntimeError(
                f"Unexpected SGLang offline generate output: {type(outputs).__name__}"
            )
        return _OfflineGen(outputs=outputs)

    def get_metadata(self) -> BackendMetadata:
        tm = self._engine.tokenizer_manager
        hf_cfg = getattr(tm.model_config, "hf_config", None)

        num_layers = getattr(hf_cfg, "num_hidden_layers", None) if hf_cfg is not None else None
        hidden_size = getattr(hf_cfg, "hidden_size", None) if hf_cfg is not None else None
        vocab_size = getattr(hf_cfg, "vocab_size", None) if hf_cfg is not None else None

        served_model_name = getattr(tm, "served_model_name", None) or tm.server_args.model_path

        return BackendMetadata(
            backend_name="sglang_offline",
            backend_version=self._sglang_version,
            model_id=str(served_model_name),
            tokenizer_id=str(getattr(tm.server_args, "tokenizer_path", None) or tm.server_args.model_path),
            max_context_len=getattr(tm.model_config, "context_len", None),
            supports={
                "input_ids": True,
                "prompt_ids_sha256": False,
                "capture_layers": True,
                "logprobs_full": True,
                # Paired base-vs-adapted scoring in one engine call/batch.
                "score_full_vocab_paired": True,
                # Paired scoring plus within-call base/base noise measurement.
                "score_full_vocab_paired_with_noise": True,
                "compute_vtw": True,
                "lora_hot_swap": True,
                "tokenize_chat": True,
                "generate_text": True,
            },
            num_layers=int(num_layers) if isinstance(num_layers, int) else None,
            hidden_size=int(hidden_size) if isinstance(hidden_size, int) else None,
            vocab_size=int(vocab_size) if isinstance(vocab_size, int) else None,
        )

    def _score_full_vocab_with_lora_ids(
        self,
        input_ids_batch: list[list[int]],
        *,
        lora_id: str | list[str | None] | None,
    ) -> torch.Tensor:
        """One-call full-vocab scoring with scalar or per-item LoRA ids."""
        t, _meta = self._score_full_vocab_with_lora_ids_and_meta(
            input_ids_batch, lora_id=lora_id
        )
        return t

    def _score_full_vocab_with_lora_ids_and_meta(
        self,
        input_ids_batch: list[list[int]],
        *,
        lora_id: str | list[str | None] | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """One-call full-vocab scoring plus per-row meta.

        Returns:
          - logprobs_full: (batch, vocab) float32 tensor
          - meta: dict with per-row fields (e.g. prompt hash) aligned to rows
        """
        from sglang.srt.managers.io_struct import GenerateReqInput

        # IMPORTANT: use per-item extra_key values.
        #
        # Some SGLang cache paths can cause within-batch interactions for identical prompts if they
        # share the same (token_ids, extra_key) namespace. Using unique per-item keys keeps each
        # request cache-isolated even within a single batched call (critical for paired scoring).
        extra_key = [uuid.uuid4().hex for _ in range(len(input_ids_batch))]

        obj = GenerateReqInput(
            input_ids=input_ids_batch,
            # IMPORTANT: force greedy sampling semantics for scoring.
            #
            # Heretic expects full-vocab logprobs derived from next-token *logits* at the prompt boundary.
            # SGLang's non-greedy sampling path can mutate next_token_logits in-place (e.g. softmax),
            # which would make log_softmax(logits) incorrect and non-repeatable under chunked/multi-pass prefill.
            #
            # IMPORTANT: force greedy sampling (top_k=1) for scoring.
            #
            # If top_k is left at its default (often TOP_K_ALL / -1), SGLang will enter the non-greedy
            # sampling path and softmax logits in-place, which would make our log_softmax(logits)
            # incorrect and can introduce within-call non-repeatability on large vocabs.
            #
            # IMPORTANT: use prefill-only scoring (max_new_tokens=0). The prompt-boundary next-token
            # distribution exists at the end of EXTEND/prefill. Decode is not guaranteed to run.
            sampling_params={"max_new_tokens": 0, "temperature": 0.0, "top_k": 1},
            stream=False,
            # For Heretic full-vocab scoring, we do NOT need input logprobs and should avoid the
            # return_logprob=True path (it changes pruning/padding behavior in logits processing).
            #
            # Mixed-chunk batching is disabled separately via the `is_heretic_scoring` scheduler guard.
            return_logprob=False,
            return_next_token_logprobs_full=True,
            lora_id=lora_id,
            # Ensure scoring does not hit/poison prefix cache (cache namespace salt).
            extra_key=extra_key,
        )
        gen = self._generate_req(obj)

        rows: list[torch.Tensor] = []
        prompt_sha256: list[str | None] = []
        for out in gen.outputs:
            meta = out.get("meta_info") or {}
            b64_steps = meta.get("heretic_next_token_logprobs_full_fp16_b64")
            shape_steps = meta.get("heretic_next_token_logprobs_full_shape")
            dtype_steps = meta.get("heretic_next_token_logprobs_full_dtype")
            if not b64_steps or not shape_steps or not dtype_steps:
                raise RuntimeError(
                    f"Missing full-vocab logprobs in offline response meta_info: keys={list(meta.keys())}"
                )
            # Optional prompt identity (stable hash at scoring boundary).
            sha_steps = meta.get("heretic_input_ids_sha256")
            if isinstance(sha_steps, list) and sha_steps and isinstance(sha_steps[-1], str):
                prompt_sha256.append(sha_steps[-1])
            else:
                prompt_sha256.append(None)
            # These fields are list-of-steps. Always take the last step to represent the
            # distribution after consuming the full prompt, even under multi-pass execution.
            b64 = b64_steps[-1]
            shape = shape_steps[-1]
            dtype = dtype_steps[-1]
            if dtype != "float16" or not isinstance(shape, list) or len(shape) != 1:
                raise RuntimeError(
                    f"Unexpected full-vocab dtype/shape in offline response: {dtype=} {shape=}"
                )
            vocab = int(shape[0])
            raw = base64.b64decode(b64.encode("ascii"))
            arr = np.frombuffer(raw, dtype=np.float16)
            if arr.size != vocab:
                raise RuntimeError(f"Decoded fp16 size mismatch: {arr.size=} {vocab=}")
            rows.append(torch.from_numpy(arr.astype(np.float32, copy=False)))

        out_t = torch.stack(rows, dim=0)
        out_meta: dict[str, Any] = {
            "heretic_input_ids_sha256": prompt_sha256,
        }
        # Side-channel for internal callers (tools) that need per-row meta without changing APIs.
        try:
            setattr(self, "_last_score_prompt_sha256", list(prompt_sha256))
        except Exception:
            pass
        return out_t, out_meta

    def lora_status(self) -> list[dict[str, Any]]:
        """Offline equivalent of `GET /heretic/lora_status` (observability)."""
        tm = self._engine.tokenizer_manager
        if not getattr(tm.server_args, "enable_lora", False):
            return []
        refs = tm.lora_registry.get_all_adapters()
        out: list[dict[str, Any]] = []
        for _, ref in (refs or {}).items():
            # LoRARef is a dataclass-like object.
            out.append(
                {
                    "lora_id": getattr(ref, "lora_id", None),
                    "lora_name": getattr(ref, "lora_name", None),
                    "lora_path": getattr(ref, "lora_path", None),
                    "pinned": bool(getattr(ref, "pinned", False)),
                }
            )
        return out

    def tokenize_chat(
        self,
        chats: list[list[dict[str, Any]]],
        *,
        continue_final_message: bool = False,
    ) -> TokenizeChatResult:
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sglang.srt.utils.prompt_identity import sha256_token_ids_le_u32

        serving_chat = OpenAIServingChat(self._engine.tokenizer_manager, self._engine.template_manager)

        token_ids_batch: list[list[int]] = []
        sha_batch: list[str] = []

        for chat in chats:
            messages: list[dict[str, Any]] = []
            for m in chat:
                role = str(m.get("role"))
                content = m.get("content")
                if content is None:
                    content = ""
                msg: dict[str, Any] = {"role": role, "content": content}
                name = m.get("name")
                if name is not None:
                    msg["name"] = name
                messages.append(msg)

            chat_req = ChatCompletionRequest(
                model=self._engine.tokenizer_manager.served_model_name,
                messages=messages,
                stream=False,
                temperature=0.0,
                max_tokens=1,
                continue_final_message=continue_final_message,
            )

            processed = serving_chat._process_messages(chat_req, is_multimodal=False)  # noqa: SLF001
            prompt_ids = processed.prompt_ids
            if isinstance(prompt_ids, str):
                prompt_ids = self._engine.tokenizer_manager.tokenizer.encode(prompt_ids)
            token_ids = [int(x) for x in prompt_ids]
            token_ids_batch.append(token_ids)
            sha_batch.append(sha256_token_ids_le_u32(token_ids))

        return TokenizeChatResult(token_ids=token_ids_batch, prompt_ids_sha256=sha_batch)

    def module_map(
        self,
        *,
        include_projs: list[str] | None = None,
        include_layers: list[int] | None = None,
        include_experts: list[int] | None = None,
        max_experts_per_layer: int | None = None,
        expert_strategy: str = "first",
    ) -> list[dict[str, Any]]:
        from sglang.srt.managers.io_struct import HereticModuleMapReqInput

        obj = HereticModuleMapReqInput(
            include_projs=include_projs,
            include_layers=include_layers,
            include_experts=include_experts,
            max_experts_per_layer=max_experts_per_layer,
            expert_strategy=expert_strategy,
        )
        data = self._run(self._engine.tokenizer_manager.heretic_module_map(obj, None))
        modules = data.get("modules") if isinstance(data, dict) else None
        if not isinstance(modules, list):
            raise RuntimeError(f"Unexpected heretic_module_map output: {data}")
        return modules

    def compute_vtw_batch(
        self,
        *,
        items: list[dict[str, Any]],
        timeout_s: float = 300.0,
    ) -> list[dict[str, Any]]:
        from sglang.srt.managers.io_struct import ComputeVTWBatchItem, ComputeVTWBatchReqInput

        batch_items: list[ComputeVTWBatchItem] = []
        for it in items:
            batch_items.append(
                ComputeVTWBatchItem(
                    name=str(it["name"]),
                    v=[float(x) for x in it["v"]],
                    dtype=str(it.get("dtype") or "float32"),
                )
            )
        obj = ComputeVTWBatchReqInput(items=batch_items)
        # timeout is currently handled at the HTTP layer; offline calls are in-process.
        _ = timeout_s
        res = self._run(self._engine.tokenizer_manager.compute_vtw_batch(obj, None))
        if not isinstance(res, list):
            raise RuntimeError(f"Unexpected compute_vtw_batch output: {res}")
        return res

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
        from sglang.srt.managers.io_struct import HereticBuildFullRownormLoraReqInput

        obj = HereticBuildFullRownormLoraReqInput(
            name=name,
            v=[float(x) for x in v.detach().to(torch.float32).cpu().tolist()],
            weight=float(weight),
            rank=int(rank),
            svd_q=int(svd_q) if svd_q is not None else None,
            svd_niter=int(svd_niter),
            out_dtype=str(out_dtype),
        )
        _ = timeout_s
        data = self._run(self._engine.tokenizer_manager.heretic_build_full_rownorm_lora(obj, None))
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected build_full_rownorm_lora output: {data}")
        if data.get("dtype") != "float16":
            raise RuntimeError(f"Unexpected dtype in build_full_rownorm_lora: {data.get('dtype')}")

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
            raise RuntimeError(f"Malformed build_full_rownorm_lora output: {data}")

        raw_a = base64.b64decode(a_b64.encode("ascii"))
        raw_b = base64.b64decode(b_b64.encode("ascii"))
        arr_a = np.frombuffer(raw_a, dtype=np.float16).reshape((int(a_shape[0]), int(a_shape[1])))
        arr_b = np.frombuffer(raw_b, dtype=np.float16).reshape((int(b_shape[0]), int(b_shape[1])))
        A = torch.from_numpy(arr_a.astype(np.float32, copy=False))
        B = torch.from_numpy(arr_b.astype(np.float32, copy=False))
        return A, B

    def score(self, input_ids_batch: list[list[int]], *, adapter: str | None = None) -> ScoreResult:
        logprobs_full, per_row_meta = self._score_full_vocab_with_lora_ids_and_meta(
            input_ids_batch, lora_id=adapter
        )
        meta = {"transport": "offline"}
        meta.update(per_row_meta)
        return ScoreResult(logprobs_full=logprobs_full, meta=meta)

    def score_full_vocab_paired(
        self,
        input_ids_batch: list[list[int]],
        *,
        adapter: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute base and adapted distributions within one engine call/batch."""
        if not input_ids_batch:
            raise ValueError("input_ids_batch must be non-empty")
        b = len(input_ids_batch)
        paired_ids = list(input_ids_batch) + list(input_ids_batch)
        paired_loras: list[str | None] = ([None] * b) + ([str(adapter)] * b)
        both = self._score_full_vocab_with_lora_ids(paired_ids, lora_id=paired_loras)
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
        """Compute (base1, adapted, base2) within one engine call/batch.

        This enables a within-call diagnostic:
          KL_noise = KL(base1 || base2)
        which should be ~0 when the backend capture is stable.
        """
        if not input_ids_batch:
            raise ValueError("input_ids_batch must be non-empty")
        b = len(input_ids_batch)
        tripled_ids = list(input_ids_batch) + list(input_ids_batch) + list(input_ids_batch)
        tripled_loras: list[str | None] = ([None] * b) + ([str(adapter)] * b) + ([None] * b)
        all3 = self._score_full_vocab_with_lora_ids(tripled_ids, lora_id=tripled_loras)
        if all3.ndim != 2 or all3.shape[0] != 3 * b:
            raise RuntimeError(
                f"Unexpected tripled score shape: {tuple(all3.shape)} for batch {b}"
            )
        base1 = all3[:b]
        adapted = all3[b : 2 * b]
        base2 = all3[2 * b :]
        return base1, adapted, base2

    def generate_text(
        self,
        input_ids_batch: list[list[int]],
        *,
        max_new_tokens: int,
        adapter: str | None = None,
        temperature: float = 0.0,
    ) -> list[str]:
        from sglang.srt.managers.io_struct import GenerateReqInput

        obj = GenerateReqInput(
            input_ids=input_ids_batch,
            sampling_params={"max_new_tokens": int(max_new_tokens), "temperature": float(temperature)},
            stream=False,
            return_logprob=False,
            lora_id=adapter,
        )
        gen = self._generate_req(obj)
        texts: list[str] = []
        for out in gen.outputs:
            t = out.get("text")
            if not isinstance(t, str):
                raise RuntimeError(f"Unexpected offline generate text schema: {out}")
            texts.append(t)
        return texts

    def capture_residuals(
        self,
        input_ids_batch: list[list[int]],
        *,
        capture_layers: list[int],
        capture_point: str = "block_input_last_token",
        adapter: str | None = None,
    ) -> ResidualCaptureResult:
        # Keep contract identical to HTTP backend: (batch, layers, d_model) for last prompt token.
        from sglang.srt.managers.io_struct import GenerateReqInput

        _ = capture_point  # currently only one capture point is supported in SGLang integration.

        def _summarize_hidden_states_steps(raw_hs: Any, *, max_steps: int = 8) -> dict[str, Any]:
            if not isinstance(raw_hs, list):
                return {"type": type(raw_hs).__name__}
            out: dict[str, Any] = {"type": "list", "num_steps": len(raw_hs), "steps": []}
            for step in raw_hs[:max_steps]:
                if isinstance(step, np.ndarray):
                    out["steps"].append(
                        {
                            "type": "np.ndarray",
                            "dtype": str(step.dtype),
                            "shape": [int(x) for x in step.shape],
                        }
                    )
                elif isinstance(step, list):
                    if len(step) == 0:
                        out["steps"].append({"type": "list", "len": 0})
                    else:
                        last = step[-1]
                        out["steps"].append(
                            {
                                "type": "list",
                                "len": len(step),
                                "last_type": type(last).__name__,
                                "last_len": (len(last) if isinstance(last, list) else None),
                            }
                        )
                else:
                    out["steps"].append({"type": type(step).__name__})
            if len(raw_hs) > max_steps:
                out["truncated"] = True
            return out

        def _dump_hidden_states_debug(
            *,
            batch_index: int,
            meta: dict[str, Any],
            raw_hs: Any,
            err: str | None,
        ) -> None:
            # One-shot per backend instance (avoid spamming logs on retries).
            if getattr(self, "_hs_debug_dumped", False):
                return
            setattr(self, "_hs_debug_dumped", True)

            payload = {
                "where": "SGLangOfflineBackend.capture_residuals",
                "batch_index": int(batch_index),
                "error": err,
                "meta_keys": sorted(list(meta.keys())),
                "hidden_states_schema_version": meta.get("hidden_states_schema_version"),
                "capture_layers_applied": meta.get("capture_layers_applied"),
                "hidden_states_d_model": meta.get("hidden_states_d_model"),
                "prompt_tokens": meta.get("prompt_tokens"),
                "requested_capture_layers": [int(x) for x in capture_layers],
                "hidden_states_summary": _summarize_hidden_states_steps(raw_hs),
            }

            dump_path = self._hidden_states_dump_path
            if dump_path:
                try:
                    with open(dump_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "Failed writing hidden_states debug dump to %r: %s",
                        dump_path,
                        e,
                    )
            logger.warning("Hidden-states debug dump: %s", json.dumps(payload, ensure_ascii=False))

        obj = GenerateReqInput(
            input_ids=input_ids_batch,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            stream=False,
            return_hidden_states=True,
            capture_layers=capture_layers,
            lora_id=adapter,
        )
        gen = self._generate_req(obj)

        per_item: list[torch.Tensor] = []
        for i, out in enumerate(gen.outputs):
            meta = out.get("meta_info") or {}
            hs_steps = meta.get("hidden_states")
            if hs_steps is None:
                raise RuntimeError("SGLang offline generate missing meta_info.hidden_states.")
            try:
                per_item.append(
                    _parse_hidden_states_last_prompt_token(
                        hs_steps,
                        meta=meta,
                        capture_layers=capture_layers,
                        batch_index=i,
                    )
                )
            except Exception as e:
                # Always emit a one-shot diagnostic dump on first failure.
                _dump_hidden_states_debug(
                    batch_index=i,
                    meta=meta,
                    raw_hs=hs_steps,
                    err=str(e),
                )
                raise

        t = torch.stack(per_item, dim=0)
        return ResidualCaptureResult(
            residuals=t,
            captured_layers=capture_layers,
            capture_point=capture_point,
            meta={"raw": gen.outputs},
        )

    def compute_vtw(
        self,
        v: torch.Tensor,
        *,
        target: ModuleRef,
        adapter: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> VTWResult:
        from sglang.srt.managers.io_struct import ComputeVTWReqInput

        if adapter is not None:
            raise NotImplementedError("compute_vtw(adapter=...) is not supported yet.")
        if dtype != torch.float32:
            raise NotImplementedError("Only float32 is supported for compute_vtw in SGLang backends.")

        obj = ComputeVTWReqInput(
            name=target.module_path,
            v=[float(x) for x in v.detach().to(torch.float32).cpu().tolist()],
            dtype="float32",
        )
        data = self._run(self._engine.tokenizer_manager.compute_vtw(obj, None))
        if not isinstance(data, dict) or "vtw" not in data:
            raise RuntimeError(f"Unexpected compute_vtw output: {data}")
        vtw = torch.tensor(data["vtw"], dtype=torch.float32)
        return VTWResult(
            target=target,
            vtw=vtw,
            implementation=data.get("implementation"),
            meta={"raw": data},
        )

    def load_adapter(self, *, name: str, tensors: dict[str, torch.Tensor], config: dict) -> str | None:
        from sglang.srt.managers.io_struct import LoadLoRAAdapterFromTensorsReqInput

        cpu_tensors = {k: v.detach().cpu() for k, v in tensors.items()}
        config = dict(config)
        config.setdefault("peft_type", "LORA")

        obj = LoadLoRAAdapterFromTensorsReqInput(
            lora_name=name,
            config_dict=config,
            serialized_tensors=_serialize_for_sglang(cpu_tensors),
            pinned=False,
            added_tokens_config=None,
            lora_id=None,
        )
        out = self._run(self._engine.tokenizer_manager.load_lora_adapter_from_tensors(obj, None))
        if not getattr(out, "success", False):
            raise RuntimeError(f"Unexpected load_lora_adapter_from_tensors output: {out}")
        loaded = getattr(out, "loaded_adapters", None) or {}
        if not isinstance(loaded, dict) or name not in loaded:
            raise RuntimeError(
                f"Missing adapter ref in load_lora response: keys={list(loaded) if isinstance(loaded, dict) else loaded}"
            )
        ref = loaded[name]
        if not isinstance(ref, dict) or not isinstance(ref.get("lora_id"), str):
            raise RuntimeError(f"Malformed adapter ref in load_lora response: {ref}")
        lora_id = ref["lora_id"]
        self._adapter_ids_by_name[name] = lora_id
        return lora_id

    def unload_adapter(self, *, name: str) -> None:
        from sglang.srt.managers.io_struct import UnloadLoRAAdapterReqInput

        lora_id = self._adapter_ids_by_name.get(name)
        obj = UnloadLoRAAdapterReqInput(lora_name=name, lora_id=lora_id)
        out = self._run(self._engine.tokenizer_manager.unload_lora_adapter(obj, None))
        if not getattr(out, "success", True):
            raise RuntimeError(f"Unexpected unload_lora_adapter output: {out}")
        if lora_id is not None:
            self._adapter_ids_by_name.pop(name, None)

