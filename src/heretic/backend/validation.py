from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, cast

import torch

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
    notes: list[str]


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
    raw = (result.meta or {}).get("raw")
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
                    tensors = {
                        key_a: torch.zeros((r, in_f), dtype=torch.float16),
                        key_b: torch.zeros((out_f, r), dtype=torch.float16),
                    }
                    config = {
                        "peft_type": "LORA",
                        "r": r,
                        "lora_alpha": r,
                        "target_modules": target_modules,
                    }
                    return tensors, config

                build_synthetic_adapter = _builder

    prompt_ok = True
    residual_ok = True
    lora_ok = True
    vtw_ok = True

    # 1) Prompt equivalence
    try:
        validate_prompt_equivalence(backend, input_ids_batch=input_ids_batch)
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

    # 5) full-vocab scoring (KL prerequisite)
    try:
        validate_full_vocab_score(backend, input_ids_batch=input_ids_batch[: min(4, len(input_ids_batch))])
        # Repeatability check: if base vs base is unstable, KL-based optimization is meaningless.
        validate_full_vocab_score_repeatability(
            backend,
            input_ids_batch=input_ids_batch[:1],
            kl_threshold=1e-2,
        )
    except Exception as e:
        notes.append(f"full_vocab_score failed: {e}")
        # Treat as a hard failure in strict mode.
        prompt_ok = False

    # Allow disabling failures in emergency debug sessions.
    if os.environ.get("HERETIC_VALIDATION_STRICT", "1") not in ("0", "false", "False"):
        if not (prompt_ok and residual_ok and lora_ok and vtw_ok):
            raise BackendValidationError("; ".join(notes) if notes else "Backend validation failed.")

    return StartupValidationReport(
        prompt_equivalence_ok=prompt_ok,
        residual_mapping_ok=residual_ok,
        lora_sanity_ok=lora_ok,
        compute_vtw_ok=vtw_ok,
        notes=notes,
    )

