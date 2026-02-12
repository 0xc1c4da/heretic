from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, cast

import torch
import math

from ..utils import Prompt, sha256_token_ids
from .base import HereticBackend, ModuleRef


class BackendValidationError(RuntimeError):
    """Base exception for backend validation failures."""


class PromptMismatchError(BackendValidationError):
    pass


class ResidualValidationError(BackendValidationError):
    pass


class LoRASanityError(BackendValidationError):
    pass


class VTWValidationError(BackendValidationError):
    pass


@dataclass(frozen=True)
class StartupValidationConfig:
    prompt_sample_min: int = 10
    prompt_sample_max: int = 50
    residual_capture_layers: tuple[int, ...] = (0, 1, 2)
    residual_rtol: float = 1e-3
    residual_atol: float = 1e-3
    vtw_rtol: float = 1e-3
    vtw_atol: float = 1e-3
    lora_sanity_min_abs_delta: float = 1e-4


@dataclass(frozen=True)
class StartupValidationReport:
    prompt_equivalence_ok: bool
    residual_mapping_ok: bool
    lora_sanity_ok: bool
    compute_vtw_ok: bool
    damage_metric_ok: bool
    notes: list[str]


def _js_other_bucket(p_log: dict[int, float], q_log: dict[int, float]) -> float:
    """Top-k JS divergence with an OTHER bucket (prob-mass outside union top-k)."""
    keys = set(p_log.keys()) | set(q_log.keys())

    def _collect(d: dict[int, float]) -> tuple[dict[int, float], float]:
        probs: dict[int, float] = {}
        s = 0.0
        for k in keys:
            lp = d.get(k)
            if lp is None:
                continue
            pk = math.exp(float(lp))
            if pk <= 0.0:
                continue
            probs[int(k)] = pk
            s += pk
        other = max(0.0, 1.0 - s)
        return probs, other

    p_probs, p_other = _collect(p_log)
    q_probs, q_other = _collect(q_log)

    def _kl(a: dict[int, float], a_other: float, b: dict[int, float], b_other: float) -> float:
        out = 0.0
        for k, ap in a.items():
            if ap <= 0.0:
                continue
            bp = b.get(k, 0.0)
            if bp <= 0.0:
                return float("inf")
            out += ap * (math.log(ap) - math.log(bp))
        if a_other > 0.0:
            if b_other <= 0.0:
                return float("inf")
            out += a_other * (math.log(a_other) - math.log(b_other))
        return out

    m: dict[int, float] = {}
    for k in keys:
        m[k] = 0.5 * p_probs.get(k, 0.0) + 0.5 * q_probs.get(k, 0.0)
    m_other = 0.5 * p_other + 0.5 * q_other

    kl_pm = _kl(p_probs, p_other, m, m_other)
    kl_qm = _kl(q_probs, q_other, m, m_other)
    if not math.isfinite(kl_pm) or not math.isfinite(kl_qm):
        return float("inf")
    return 0.5 * kl_pm + 0.5 * kl_qm


def validate_damage_metric_repeatability(
    backend: HereticBackend,
    *,
    input_ids_batch: list[list[int]],
    damage_metric: str,
    damage_noise_threshold: float,
    damage_retry_count: int,
    delta_nll_continuation_tokens: int,
    topk_js_k: int,
    topk_js_positions: int,
    build_synthetic_adapter: Callable[[], tuple[dict[str, torch.Tensor], dict[str, Any]]] | None,
) -> None:
    """Validate the startup repeatability contract for the configured damage metric.

    Contract:
    - Build a synthetic *no-op* adapter (A=B=0) and load it.
    - Measure paired-within-call (base1, adapted, base2) on cached base continuations.
    - Require within-call noise <= threshold.
    - Require no-op adapter damage to be commensurate with noise (i.e. not a large systematic shift).
    """
    if damage_metric not in ("paired_delta_nll", "topk_js"):
        raise BackendValidationError(f"Unknown damage_metric={damage_metric!r} for validation.")

    if not input_ids_batch:
        raise BackendValidationError("Damage validation requires non-empty input_ids_batch.")

    gen_ids = getattr(backend, "generate_token_ids", None)
    if gen_ids is None:
        raise BackendValidationError("Backend missing generate_token_ids required for damage validation.")

    cont_len = int(delta_nll_continuation_tokens)
    if cont_len <= 0:
        raise BackendValidationError("delta_nll_continuation_tokens must be > 0 for damage validation.")

    cont_ids = gen_ids(  # type: ignore[misc]
        input_ids_batch,
        max_new_tokens=cont_len,
        adapter=None,
        temperature=0.0,
        top_k=1,
    )
    if not isinstance(cont_ids, list) or len(cont_ids) != len(input_ids_batch):
        raise BackendValidationError("generate_token_ids returned unexpected batch shape.")

    if build_synthetic_adapter is None:
        raise BackendValidationError("Damage validation requires build_synthetic_adapter (LoRA hot-swap supported).")

    tensors, cfg = build_synthetic_adapter()
    # Contract: this must be a *no-op* adapter (A=B=0), regardless of how the backend-specific
    # adapter ABI is produced. (The LoRA sanity check uses a non-zero adapter; this check must not.)
    tensors = {k: torch.zeros_like(v) for k, v in (tensors or {}).items()}
    load = getattr(backend, "load_adapter", None)
    unload = getattr(backend, "unload_adapter", None)
    if load is None or unload is None:
        raise BackendValidationError("Backend missing load_adapter/unload_adapter required for damage validation.")

    adapter_name = "startup_validation_noop"
    adapter_id = load(name=adapter_name, tensors=tensors, config=cfg)  # type: ignore[misc]
    try:
        last_damage = float("nan")
        last_noise = float("inf")
        attempts = max(0, int(damage_retry_count)) + 1

        for attempt in range(attempts):
            if damage_metric == "paired_delta_nll":
                scorer = getattr(backend, "score_continuation_nll_paired_with_noise", None)
                if scorer is None:
                    raise BackendValidationError(
                        "Backend missing score_continuation_nll_paired_with_noise."
                    )
                base1, adapted, base2 = scorer(  # type: ignore[misc]
                    prompt_ids_batch=input_ids_batch,
                    continuation_ids_batch=cont_ids,
                    adapter=str(adapter_id),
                )
                if len(base1) != len(adapted) or len(base1) != len(base2):
                    raise BackendValidationError(
                        "paired_delta_nll returned unexpected batch shape."
                    )
                deltas = [float(a) - float(b) for a, b in zip(adapted, base1, strict=True)]
                noises = [
                    abs(float(b2) - float(b1))
                    for b2, b1 in zip(base2, base1, strict=True)
                ]
                damage = float(sum(deltas) / max(1, len(deltas)))
                noise = float(sum(noises) / max(1, len(noises)))
            else:
                scorer = getattr(backend, "score_continuation_topk_paired_with_noise", None)
                if scorer is None:
                    raise BackendValidationError(
                        "Backend missing score_continuation_topk_paired_with_noise."
                    )
                b1_topk, ad_topk, b2_topk = scorer(  # type: ignore[misc]
                    prompt_ids_batch=input_ids_batch,
                    continuation_ids_batch=cont_ids,
                    adapter=str(adapter_id),
                    top_k=int(topk_js_k),
                )
                if not (
                    isinstance(b1_topk, list)
                    and isinstance(ad_topk, list)
                    and isinstance(b2_topk, list)
                ):
                    raise BackendValidationError("topk_js returned unexpected schema.")
                # Validate only the first item for startup sanity to keep cost low.
                b1_pos = b1_topk[0]
                ad_pos = ad_topk[0]
                b2_pos = b2_topk[0]
                if not (
                    isinstance(b1_pos, list)
                    and isinstance(ad_pos, list)
                    and isinstance(b2_pos, list)
                ):
                    raise BackendValidationError(
                        "topk_js returned unexpected per-item schema."
                    )
                npos = min(int(topk_js_positions), len(b1_pos), len(ad_pos), len(b2_pos))
                if npos <= 0:
                    raise BackendValidationError("topk_js returned no continuation positions.")
                dmg = 0.0
                noi = 0.0
                for t in range(npos):
                    dmg += float(_js_other_bucket(b1_pos[t], ad_pos[t]))
                    noi += float(_js_other_bucket(b1_pos[t], b2_pos[t]))
                damage = float(dmg / npos)
                noise = float(noi / npos)

            last_damage = float(damage)
            last_noise = float(noise)
            if not math.isfinite(last_damage) or not math.isfinite(last_noise):
                continue

            if last_noise > float(damage_noise_threshold):
                continue

            # No-op adapter should not introduce damage much larger than measurement noise.
            allowed = max(float(damage_noise_threshold), 3.0 * float(last_noise))
            if abs(float(last_damage)) > allowed:
                continue

            # Passed.
            return

        raise BackendValidationError(
            f"Damage metric noise too high: {last_noise:.6g} > {damage_noise_threshold:.6g} "
            f"(metric={damage_metric}, retries={attempts-1}, last_damage={last_damage:.6g})"
        )
    finally:
        try:
            unload(name=adapter_name)  # type: ignore[misc]
        except Exception:
            pass


def _take_prompts(prompts: list[Prompt], *, min_n: int, max_n: int) -> list[Prompt]:
    if not prompts:
        return []
    n = max(min_n, min(max_n, len(prompts)))
    return prompts[:n]


def _extract_prompt_hashes_from_raw(raw: Any) -> list[str | None]:
    """
    Best-effort extraction of prompt_ids_sha256 from an OpenAI-compatible response.

    Supports:
    - top-level: {"prompt_ids_sha256": "..."}
    - per-choice: {"choices": [{"prompt_ids_sha256": "..."}, ...]}
    - SGLang extension: {"sgl_ext": {"prompt_ids_sha256": "..."}}
    - per-choice SGLang extension: {"choices": [{"sgl_ext": {"prompt_ids_sha256": "..."}}, ...]}
    """
    if not isinstance(raw, dict):
        return []
    if "prompt_ids_sha256" in raw:
        v = raw.get("prompt_ids_sha256")
        return [v if isinstance(v, str) else None]
    sgl_ext = raw.get("sgl_ext")
    if isinstance(sgl_ext, dict) and isinstance(sgl_ext.get("prompt_ids_sha256"), str):
        return [cast(str, sgl_ext["prompt_ids_sha256"])]
    choices = raw.get("choices")
    if not isinstance(choices, list):
        return []
    out: list[str | None] = []
    for c in choices:
        if not isinstance(c, dict):
            out.append(None)
            continue
        if isinstance(c.get("prompt_ids_sha256"), str):
            out.append(c["prompt_ids_sha256"])
            continue
        ext = c.get("sgl_ext")
        if isinstance(ext, dict) and isinstance(ext.get("prompt_ids_sha256"), str):
            out.append(ext["prompt_ids_sha256"])
        else:
            out.append(None)
    return out


def validate_full_vocab_score(
    backend: HereticBackend,
    *,
    input_ids_batch: list[list[int]],
) -> None:
    """Ensure backend.score returns full-vocab logprobs with correct batch shape."""
    result = backend.score(input_ids_batch)
    t = result.logprobs_full
    if t is None:
        raise BackendValidationError("Backend score returned no logprobs_full (required for KL).")
    if not isinstance(t, torch.Tensor):
        raise BackendValidationError(f"Backend logprobs_full is not a torch.Tensor: {type(t)}")
    if t.ndim != 2:
        raise BackendValidationError(f"Expected logprobs_full.ndim==2, got {t.ndim}")
    if t.shape[0] != len(input_ids_batch):
        raise BackendValidationError(
            f"Expected logprobs_full batch {len(input_ids_batch)}, got {t.shape[0]}"
        )


def validate_full_vocab_score_repeatability(
    backend: HereticBackend,
    *,
    input_ids_batch: list[list[int]],
    kl_threshold: float = 1e-2,
) -> None:
    """Ensure full-vocab score is repeatable for identical inputs.

    For backends that support paired scoring (one-call semantics), we validate *within-call*
    repeatability by scoring a duplicated batch in a single request.

    For all other backends, we fall back to a stricter *cross-call* repeatability check.
    """
    import torch.nn.functional as F

    supports = backend.get_metadata().supports
    if supports.get("score_full_vocab_paired", False):
        if not input_ids_batch:
            raise BackendValidationError("Repeatability check requires non-empty input_ids_batch.")
        b = len(input_ids_batch)
        doubled = list(input_ids_batch) + list(input_ids_batch)
        r = backend.score(doubled, adapter=None)
        t = r.logprobs_full
        # Hard prompt-identity invariant (best-effort).
        #
        # If the backend provides per-row prompt hashes at the scoring boundary, assert that the
        # duplicated halves refer to identical prompt token ids. This catches batch/row mixups even
        # when KL might coincidentally be small.
        try:
            sha_list = (r.meta or {}).get("heretic_input_ids_sha256")
            if isinstance(sha_list, list) and len(sha_list) == 2 * b:
                a = sha_list[:b]
                c = sha_list[b:]
                if all(isinstance(x, str) for x in a) and all(isinstance(x, str) for x in c):
                    for i, (ha, hc) in enumerate(zip(a, c, strict=True)):
                        if ha != hc:
                            raise BackendValidationError(
                                f"Within-call prompt-hash mismatch at {i}: {ha} != {hc}"
                            )
        except BackendValidationError:
            raise
        except Exception:
            # If hashes are absent or malformed, do not fail repeatability purely on that basis.
            pass
        if t is None:
            raise BackendValidationError("Backend score returned no logprobs_full (required for KL).")
        if t.ndim != 2 or t.shape[0] != 2 * b:
            raise BackendValidationError(
                f"Within-call repeatability unexpected shape: {tuple(t.shape)} for batch {b}"
            )
        t1 = t[:b]
        t2 = t[b:]
    else:
        r1 = backend.score(input_ids_batch)
        r2 = backend.score(input_ids_batch)
        t1 = r1.logprobs_full
        t2 = r2.logprobs_full
        if t1 is None or t2 is None:
            raise BackendValidationError("Backend score returned no logprobs_full (required for KL).")
        if tuple(t1.shape) != tuple(t2.shape):
            raise BackendValidationError(
                f"Repeatability shape mismatch: {tuple(t1.shape)} != {tuple(t2.shape)}"
            )

    # KL(base||base2) using Heretic evaluator semantics: input=t2, target=t1, log_target=True
    kl = float(F.kl_div(t2, t1, reduction="batchmean", log_target=True).item())
    if not (kl == kl):  # NaN check without importing math
        raise BackendValidationError("Non-finite KL in repeatability check (NaN).")
    if kl > float(kl_threshold):
        raise BackendValidationError(
            f"Backend full-vocab scoring is not repeatable: KL(base||base2)={kl:.6g} > {kl_threshold}. "
            "If this is SGLang, it often indicates drift or multi-pass capture selecting inconsistent steps."
        )


def validate_prompt_equivalence(
    backend: HereticBackend,
    *,
    input_ids_batch: list[list[int]],
) -> None:
    """
    Validate the backend echoes `prompt_ids_sha256` matching local token IDs.

    Only runs when backend metadata declares support for prompt hash echo.
    """
    supports = backend.get_metadata().supports
    if not supports.get("prompt_ids_sha256", False):
        return

    expected = [sha256_token_ids(ids) for ids in input_ids_batch]
    result = backend.score(input_ids_batch)

    # Prefer explicit meta fields when the backend provides them.
    meta = result.meta or {}
    got: list[str | None] = []
    if isinstance(meta.get("prompt_ids_sha256"), list):
        got = [x if isinstance(x, str) else None for x in meta.get("prompt_ids_sha256") or []]
    elif isinstance(meta.get("heretic_input_ids_sha256"), list):
        got = [
            x if isinstance(x, str) else None
            for x in meta.get("heretic_input_ids_sha256") or []
        ]
    else:
        raw = meta.get("raw")
        got = _extract_prompt_hashes_from_raw(raw)

    # If the response only has one hash, treat it as ambiguous but still validate
    # when there's only one prompt in batch.
    if len(input_ids_batch) == 1 and len(got) == 1 and got[0] is not None:
        if got[0] != expected[0]:
            raise PromptMismatchError(f"Expected {expected[0]}, got {got[0]}")
        return

    # Otherwise require per-choice hashes.
    if len(got) != len(input_ids_batch):
        raise PromptMismatchError(
            f"Backend did not return per-prompt hashes (expected {len(input_ids_batch)}, got {len(got)})."
        )
    for i, (e, g) in enumerate(zip(expected, got, strict=True)):
        if g is None:
            raise PromptMismatchError(f"Missing prompt_ids_sha256 for batch item {i}.")
        if g != e:
            raise PromptMismatchError(f"Prompt hash mismatch at {i}: expected {e}, got {g}")


def validate_tokenize_chat_equivalence(
    backend: HereticBackend,
    *,
    prompts: list[Prompt],
    trust_remote_code: bool | None,
    model_id: str,
) -> None:
    """Best-effort check: HF apply_chat_template token IDs == backend.tokenize_chat token IDs.

    This specifically catches drift between the HF template/tokenizer and SGLang's internal
    chat processing. It only runs when backend supports tokenize_chat and when we can
    load an HF tokenizer.
    """
    supports = backend.get_metadata().supports
    if not bool(supports.get("tokenize_chat", False)):
        return

    try:
        from transformers import AutoTokenizer  # ty: ignore[unresolved-import]
    except Exception:
        # Transformers not available in minimal envs.
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    chats = [
        [
            {"role": "system", "content": p.system},
            {"role": "user", "content": p.user},
        ]
        for p in prompts
    ]

    # Backend canonical
    out = backend.tokenize_chat(chats, continue_final_message=False)
    backend_ids = out.token_ids

    # HF canonical
    hf_ids = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        chats,
        add_generation_prompt=True,
        tokenize=True,
    )

    if not isinstance(hf_ids, list) or not hf_ids or not all(isinstance(x, list) for x in hf_ids):
        raise PromptMismatchError("HF apply_chat_template returned unexpected token-id schema.")

    if len(hf_ids) != len(backend_ids):
        raise PromptMismatchError(
            f"tokenize_chat batch size mismatch: hf={len(hf_ids)} backend={len(backend_ids)}"
        )

    for i, (a, b) in enumerate(zip(hf_ids, backend_ids, strict=True)):
        if [int(x) for x in a] != [int(x) for x in b]:
            ha = sha256_token_ids([int(x) for x in a])
            hb = sha256_token_ids([int(x) for x in b])
            raise PromptMismatchError(
                "HF vs backend chat-tokenization mismatch.\n"
                f"- item={i}\n"
                f"- hf_sha256={ha}\n"
                f"- backend_sha256={hb}\n"
                f"- hf_len={len(a)} backend_len={len(b)}"
            )


def _iter_prompts_in_batches(prompts: list[Prompt], *, batch_size: int) -> Iterable[list[Prompt]]:
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]


def validate_residual_mapping(
    backend: HereticBackend,
    *,
    input_ids_batch: list[list[int]],
    capture_layers: list[int],
    baseline_residuals: torch.Tensor | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    """
    Validate residual capture shape/layout, and optionally numeric correctness.

    If `baseline_residuals` is provided, it must be shaped (batch, layers, d_model)
    aligned with `capture_layers`.
    """
    supports = backend.get_metadata().supports
    if not supports.get("capture_layers", False):
        return

    out = backend.capture_residuals(input_ids_batch, capture_layers=capture_layers)
    t = out.residuals
    if t.ndim != 3:
        raise ResidualValidationError(f"Expected residuals.ndim==3, got {t.ndim}")
    if t.shape[0] != len(input_ids_batch):
        raise ResidualValidationError(
            f"Expected residual batch {len(input_ids_batch)}, got {t.shape[0]}"
        )
    if t.shape[1] != len(capture_layers):
        raise ResidualValidationError(
            f"Expected {len(capture_layers)} layers, got {t.shape[1]}"
        )
    if t.shape[2] <= 0:
        raise ResidualValidationError("Residual d_model dimension is empty.")

    if baseline_residuals is None:
        return

    if baseline_residuals.shape != t.shape:
        raise ResidualValidationError(
            f"Baseline residuals shape {tuple(baseline_residuals.shape)} != backend {tuple(t.shape)}"
        )

    # Compare in float32 on CPU for stability.
    a = baseline_residuals.detach().to(torch.float32).cpu()
    b = t.detach().to(torch.float32).cpu()
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        max_abs = (a - b).abs().max().item()
        raise ResidualValidationError(
            f"Residuals mismatch (rtol={rtol}, atol={atol}); max_abs={max_abs}"
        )


def validate_lora_sanity(
    backend: HereticBackend,
    *,
    input_ids_batch: list[list[int]],
    build_synthetic_adapter: Callable[[], tuple[dict[str, torch.Tensor], dict[str, Any]]] | None,
    adapter_name: str = "_heretic_validation_synthetic",
    min_abs_delta: float = 1e-4,
) -> None:
    """
    Load a synthetic adapter and verify the scoring output changes measurably.

    This is intentionally backend-agnostic: the caller supplies a builder function
    that returns (tensors, config) matching the backend's adapter ABI.
    """
    supports = backend.get_metadata().supports
    if not supports.get("lora_hot_swap", False):
        return
    if build_synthetic_adapter is None:
        return

    tensors, config = build_synthetic_adapter()
    if not tensors:
        raise LoRASanityError("Synthetic adapter builder returned no tensors.")

    base = backend.score(input_ids_batch)
    adapter_ref = backend.load_adapter(name=adapter_name, tensors=tensors, config=config)
    try:
        # Some backends (e.g. SGLang) use an internal adapter id for the hot path.
        adapted = backend.score(input_ids_batch, adapter=adapter_ref or adapter_name)
    finally:
        backend.unload_adapter(name=adapter_name)

    # Prefer full-vocab when available; else fall back to a conservative raw-diff heuristic.
    if base.logprobs_full is not None and adapted.logprobs_full is not None:
        delta = (adapted.logprobs_full - base.logprobs_full).abs().max().item()
        if delta < min_abs_delta:
            raise LoRASanityError(f"Adapter had negligible effect on logprobs_full (max_abs_delta={delta}).")
        return

    # If we don't have structured logprobs, compare serialized raw payloads as a last resort.
    raw0 = (base.meta or {}).get("raw")
    raw1 = (adapted.meta or {}).get("raw")
    if raw0 == raw1:
        raise LoRASanityError("Adapter had no observable effect on backend score response.")


def validate_topk_js_not_saturated_with_synthetic_lora(
    backend: HereticBackend,
    *,
    input_ids_batch: list[list[int]],
    build_synthetic_adapter: Callable[[], tuple[dict[str, torch.Tensor], dict[str, Any]]] | None,
    cont_len: int = 16,
    top_k: int = 64,
    positions: int = 16,
    adapter_name: str = "_heretic_validation_synthetic_topk",
    saturated_threshold: float = 0.69,
) -> None:
    """Fail fast if a small synthetic LoRA causes top-k JS to saturate near ln(2).

    This is designed to catch catastrophic adapter application/construction failures (e.g. FP8
    scale bugs) that manifest as distributions becoming nearly disjoint (JS ≈ ln(2)).

    Notes:
    - Runs only when the backend supports continuation top-k scoring and LoRA hot-swap.
    - Uses a *non-zero* synthetic adapter (unlike damage repeatability validation, which must be no-op).
    """
    supports = backend.get_metadata().supports
    if not (supports.get("lora_hot_swap", False) and supports.get("score_continuation_topk_paired_with_noise", False)):
        return
    if build_synthetic_adapter is None:
        return
    if not input_ids_batch:
        raise BackendValidationError("topk_js saturation check requires non-empty input_ids_batch.")

    gen_ids = getattr(backend, "generate_token_ids", None)
    scorer = getattr(backend, "score_continuation_topk_paired_with_noise", None)
    if gen_ids is None or scorer is None:
        return

    cont_ids = gen_ids(  # type: ignore[misc]
        input_ids_batch[:1],
        max_new_tokens=int(cont_len),
        adapter=None,
        temperature=0.0,
        top_k=1,
    )
    if not isinstance(cont_ids, list) or not cont_ids or not isinstance(cont_ids[0], list) or not cont_ids[0]:
        raise BackendValidationError("generate_token_ids returned empty continuation in topk_js saturation check.")

    tensors, config = build_synthetic_adapter()
    if not tensors:
        raise BackendValidationError("Synthetic adapter builder returned no tensors.")

    adapter_ref = backend.load_adapter(name=adapter_name, tensors=tensors, config=config)
    try:
        b1_topk, ad_topk, b2_topk = scorer(  # type: ignore[misc]
            prompt_ids_batch=input_ids_batch[:1],
            continuation_ids_batch=cont_ids[:1],
            adapter=str(adapter_ref or adapter_name),
            top_k=int(top_k),
        )
        b1_pos = b1_topk[0] if isinstance(b1_topk, list) and b1_topk else None
        ad_pos = ad_topk[0] if isinstance(ad_topk, list) and ad_topk else None
        b2_pos = b2_topk[0] if isinstance(b2_topk, list) and b2_topk else None
        if not (isinstance(b1_pos, list) and isinstance(ad_pos, list) and isinstance(b2_pos, list)):
            raise BackendValidationError("topk_js returned unexpected per-item schema in saturation check.")
        npos = min(int(positions), len(b1_pos), len(ad_pos), len(b2_pos))
        if npos <= 0:
            raise BackendValidationError("topk_js returned no continuation positions in saturation check.")
        dmg = 0.0
        noi = 0.0
        for t in range(npos):
            dmg += float(_js_other_bucket(b1_pos[t], ad_pos[t]))
            noi += float(_js_other_bucket(b1_pos[t], b2_pos[t]))
        dmg = float(dmg / npos)
        noi = float(noi / npos)
        if not math.isfinite(dmg) or not math.isfinite(noi):
            raise BackendValidationError(f"Non-finite topk_js in saturation check: damage={dmg!r} noise={noi!r}")
        if dmg >= float(saturated_threshold):
            raise BackendValidationError(
                f"topk_js appears saturated (synthetic LoRA): damage={dmg:.6g} noise={noi:.6g} "
                f"(threshold={saturated_threshold:.6g}). This often indicates adapter/FP8 scale bugs."
            )
    finally:
        try:
            backend.unload_adapter(name=adapter_name)
        except Exception:
            pass


def validate_compute_vtw(
    backend: HereticBackend,
    *,
    v: torch.Tensor,
    target: ModuleRef,
    baseline_vtw: torch.Tensor | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    supports = backend.get_metadata().supports
    if not supports.get("compute_vtw", False):
        return

    out = backend.compute_vtw(v, target=target)
    got = out.vtw
    if got.ndim == 0:
        raise VTWValidationError("compute_vtw returned a scalar; expected a vector.")

    if baseline_vtw is None:
        return

    a = baseline_vtw.detach().to(torch.float32).cpu()
    b = got.detach().to(torch.float32).cpu()
    if a.shape != b.shape:
        raise VTWValidationError(f"VTW shape mismatch: baseline {tuple(a.shape)} vs backend {tuple(b.shape)}")
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        max_abs = (a - b).abs().max().item()
        raise VTWValidationError(f"VTW mismatch (rtol={rtol}, atol={atol}); max_abs={max_abs}")


@dataclass
class ContinuousMonitor:
    """
    Lightweight runtime monitor (pure Python, no extra deps).

    This is intentionally minimal: it records timings and RSS snapshots that
    the caller can print/log at whatever cadence makes sense.
    """

    score_calls: int = 0
    score_seconds: float = 0.0
    adapter_load_calls: int = 0
    adapter_load_seconds: float = 0.0
    adapter_unload_calls: int = 0
    adapter_unload_seconds: float = 0.0
    rss_kb_samples: list[int] | None = None

    def __post_init__(self) -> None:
        if self.rss_kb_samples is None:
            self.rss_kb_samples = []

    def sample_rss_kb(self) -> int:
        # ru_maxrss is monotonic (max RSS). Still useful to detect runaway growth.
        import resource

        rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert self.rss_kb_samples is not None
        self.rss_kb_samples.append(rss_kb)
        return rss_kb

    def record_score(self, seconds: float) -> None:
        self.score_calls += 1
        self.score_seconds += seconds

    def record_adapter_load(self, seconds: float) -> None:
        self.adapter_load_calls += 1
        self.adapter_load_seconds += seconds

    def record_adapter_unload(self, seconds: float) -> None:
        self.adapter_unload_calls += 1
        self.adapter_unload_seconds += seconds


def run_startup_validations(
    *,
    backend: HereticBackend,
    prompts: list[Prompt],
    encode_prompts: Callable[[list[Prompt]], list[list[int]]],
    settings: Any | None = None,
    baseline_residuals_fn: Callable[[list[Prompt]], torch.Tensor] | None = None,
    build_synthetic_adapter: Callable[[], tuple[dict[str, torch.Tensor], dict[str, Any]]] | None = None,
    vtw_case: tuple[torch.Tensor, ModuleRef, torch.Tensor] | None = None,
    config: StartupValidationConfig | None = None,
) -> StartupValidationReport:
    """
    Execute Phase-8 startup validations.

    This is designed to be called by `main.py` before starting trials.
    """
    cfg = config or StartupValidationConfig()
    notes: list[str] = []

    sample = _take_prompts(prompts, min_n=cfg.prompt_sample_min, max_n=cfg.prompt_sample_max)
    if not sample:
        return StartupValidationReport(
            prompt_equivalence_ok=True,
            residual_mapping_ok=True,
            lora_sanity_ok=True,
            compute_vtw_ok=True,
            damage_metric_ok=True,
            notes=["No prompts available for validation; skipped."],
        )

    input_ids_batch = encode_prompts(sample)
    # Remote backends can have stricter per-request batch limits than Heretic's dataset slices.
    # Keep validations robust by only shrinking batches where it's sufficient to validate the contract.
    try:
        backend_name = backend.get_metadata().backend_name
    except Exception:
        backend_name = ""

    # If the caller didn't provide a synthetic adapter builder, try a best-effort fallback for
    # SGLang-style backends that expose module_map shapes.
    if build_synthetic_adapter is None:
        supports = {}
        try:
            supports = backend.get_metadata().supports
        except Exception:
            supports = {}
        if supports.get("lora_hot_swap", False):
            try:
                module_descs = backend.module_map(
                    include_projs=["o_proj", "down_proj"],
                    include_experts=[],
                )
            except Exception:
                module_descs = []

            def _pick_weight_with_shape():
                for d in module_descs:
                    if not isinstance(d, dict):
                        continue
                    path = d.get("module_path") or d.get("name")
                    shape = d.get("shape")
                    if not isinstance(path, str) or not isinstance(shape, list) or len(shape) != 2:
                        continue
                    if not all(isinstance(x, int) for x in shape):
                        continue
                    out_f, in_f = int(shape[0]), int(shape[1])
                    if out_f > 0 and in_f > 0 and path.endswith(".weight"):
                        return path, out_f, in_f
                return None

            picked = _pick_weight_with_shape()
            if picked is not None:
                path, out_f, in_f = picked
                base = path[: -len(".weight")]
                key_a = f"{base}.lora_A.weight"
                key_b = f"{base}.lora_B.weight"
                target_modules = (
                    ["down_proj"]
                    if "down_proj" in path
                    else (["o_proj"] if "o_proj" in path else [base.split(".")[-1]])
                )

                def _builder():
                    r = 1
                    # Build a *non-zero* adapter for LoRA hot-swap sanity.
                    #
                    # We intentionally keep this deterministic and modest in magnitude so it:
                    # - reliably changes outputs (max_abs_delta > ~1e-4),
                    # - but does not destabilize the backend or saturate logits.
                    g = torch.Generator()
                    g.manual_seed(0)
                    scale = 0.05
                    tensors = {
                        key_a: (scale * torch.randn((r, in_f), generator=g)).to(torch.float16),
                        key_b: (scale * torch.randn((out_f, r), generator=g)).to(torch.float16),
                    }
                    config = {
                        "peft_type": "LORA",
                        "r": r,
                        # Amplify slightly to make the effect robust on very large models.
                        "lora_alpha": 16,
                        "target_modules": target_modules,
                    }
                    return tensors, config

                build_synthetic_adapter = _builder

    prompt_ok = True
    residual_ok = True
    lora_ok = True
    vtw_ok = True
    damage_ok = True

    # 1) Prompt equivalence
    try:
        validate_prompt_equivalence(backend, input_ids_batch=input_ids_batch)
        # Additional best-effort check for chat template/tokenizer drift.
        if settings is not None:
            model_id = getattr(settings, "model", None)
            if isinstance(model_id, str) and model_id:
                validate_tokenize_chat_equivalence(
                    backend,
                    prompts=sample[: min(3, len(sample))],
                    trust_remote_code=getattr(settings, "trust_remote_code", None),
                    model_id=str(model_id),
                )
    except Exception as e:
        prompt_ok = False
        notes.append(f"prompt_equivalence failed: {e}")

    # 2) Residual mapping
    try:
        residual_sample = sample
        residual_input_ids_batch = input_ids_batch
        if backend_name in ("sglang", "sglang_offline") and len(input_ids_batch) > 1:
            # Hidden-states payloads can be large; residual mapping only needs to validate the shape/layout contract.
            notes.append(
                f"startup validation: limiting SGLang residual_mapping batch from {len(input_ids_batch)} to 1 prompt"
            )
            residual_sample = sample[:1]
            residual_input_ids_batch = input_ids_batch[:1]

        baseline = None
        if baseline_residuals_fn is not None:
            # baseline is (batch, layers, d_model); select requested capture layers.
            full = baseline_residuals_fn(residual_sample)
            baseline = full[:, list(cfg.residual_capture_layers), :].contiguous()
        validate_residual_mapping(
            backend,
            input_ids_batch=residual_input_ids_batch,
            capture_layers=list(cfg.residual_capture_layers),
            baseline_residuals=baseline,
            rtol=cfg.residual_rtol,
            atol=cfg.residual_atol,
        )
    except Exception as e:
        residual_ok = False
        notes.append(f"residual_mapping failed: {e}")
        # Extra context for debugging SGLang hidden-states schema issues.
        # The offline backend emits a one-shot hidden-state schema dump on first failure.
        # To persist the dump, set:
        #   sglang_hidden_states_dump_path = "/path/to/hs_dump.jsonl"
        try:
            backend_name = backend.get_metadata().backend_name
        except Exception:
            backend_name = ""
        if backend_name in ("sglang", "sglang_offline"):
            notes.append(
                "tip: set sglang_hidden_states_dump_path to persist the hidden-states dump for offline debugging"
            )

    # 3) LoRA sanity
    try:
        validate_lora_sanity(
            backend,
            input_ids_batch=input_ids_batch[:1],
            build_synthetic_adapter=build_synthetic_adapter,
            min_abs_delta=cfg.lora_sanity_min_abs_delta,
        )
        # Extra guard: ensure a small non-zero synthetic adapter doesn't catastrophically break
        # continuation top-k scoring (JS ≈ ln(2) saturation).
        validate_topk_js_not_saturated_with_synthetic_lora(
            backend,
            input_ids_batch=input_ids_batch[:1],
            build_synthetic_adapter=build_synthetic_adapter,
        )
    except Exception as e:
        lora_ok = False
        notes.append(f"lora_sanity failed: {e}")

    # 4) compute_vtw correctness
    try:
        if vtw_case is not None:
            v, target, baseline = vtw_case
            validate_compute_vtw(
                backend,
                v=v,
                target=target,
                baseline_vtw=baseline,
                rtol=cfg.vtw_rtol,
                atol=cfg.vtw_atol,
            )
        else:
            # Still ensure endpoint is reachable if supported.
            supports = backend.get_metadata().supports
            if supports.get("compute_vtw", False):
                # Best-effort: pick a target from module_map if backend provides it.
                target: ModuleRef | None = None
                v: torch.Tensor | None = None
                try:
                    module_descs = backend.module_map(include_projs=["o_proj", "down_proj"])
                    for d in module_descs:
                        if not isinstance(d, dict):
                            continue
                        path = d.get("module_path") or d.get("name")
                        out_f = d.get("out_features")
                        shape = d.get("shape")
                        if not isinstance(path, str):
                            continue
                        if isinstance(out_f, int) and out_f > 0:
                            out_dim = int(out_f)
                        elif (
                            isinstance(shape, list)
                            and len(shape) == 2
                            and all(isinstance(x, int) for x in shape)
                            and int(shape[0]) > 0
                        ):
                            out_dim = int(shape[0])
                        else:
                            continue
                        target = ModuleRef(
                            module_path=path,
                            kind="parameter",
                            layer=d.get("layer") if isinstance(d.get("layer"), int) else None,
                            expert_id=d.get("expert_id") if isinstance(d.get("expert_id"), int) else None,
                            proj=d.get("proj") if isinstance(d.get("proj"), str) else None,
                        )
                        v = torch.zeros((out_dim,), dtype=torch.float32)
                        v[0] = 1.0
                        break
                except Exception:
                    target = None
                    v = None

                if target is not None and v is not None:
                    validate_compute_vtw(backend, v=v, target=target, baseline_vtw=None)
                else:
                    notes.append(
                        "compute_vtw supported but no vtw_case provided and could not auto-select a target; skipped numeric check."
                    )
    except Exception as e:
        vtw_ok = False
        notes.append(f"compute_vtw failed: {e}")

    # 5) Damage metric repeatability (startup contract)
    try:
        damage_metric = str(getattr(settings, "damage_metric", "paired_delta_nll") if settings is not None else "paired_delta_nll")
        validate_damage_metric_repeatability(
            backend,
            input_ids_batch=input_ids_batch[:1],
            damage_metric=damage_metric,
            damage_noise_threshold=float(getattr(settings, "damage_noise_threshold", 0.05) if settings is not None else 0.05),
            damage_retry_count=int(getattr(settings, "damage_retry_count", 0) if settings is not None else 0),
            delta_nll_continuation_tokens=int(getattr(settings, "delta_nll_continuation_tokens", 32) if settings is not None else 32),
            topk_js_k=int(getattr(settings, "topk_js_k", 128) if settings is not None else 128),
            topk_js_positions=int(getattr(settings, "topk_js_positions", 32) if settings is not None else 32),
            build_synthetic_adapter=build_synthetic_adapter,
        )
    except Exception as e:
        damage_ok = False
        notes.append(f"damage_metric failed: {e}")

    # Allow disabling failures in emergency debug sessions.
    if os.environ.get("HERETIC_VALIDATION_STRICT", "1") not in ("0", "false", "False"):
        if not (prompt_ok and residual_ok and lora_ok and vtw_ok and damage_ok):
            raise BackendValidationError("; ".join(notes) if notes else "Backend validation failed.")

    return StartupValidationReport(
        prompt_equivalence_ok=prompt_ok,
        residual_mapping_ok=residual_ok,
        lora_sanity_ok=lora_ok,
        compute_vtw_ok=vtw_ok,
        damage_metric_ok=damage_ok,
        notes=notes,
    )

