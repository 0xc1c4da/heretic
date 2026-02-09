# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import math

import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Any

from .config import Settings
from .model import Model
from .utils import Prompt, load_prompts, print


class NonFiniteLogprobsError(RuntimeError):
    """Raised when backend returns NaN/Inf logprobs that would poison KL computation."""


class Evaluator:
    settings: Settings
    model: Model
    good_prompts: list[Prompt]
    bad_prompts: list[Prompt]
    base_logprobs: Tensor
    base_refusals: int
    _damage_prompt_ids: list[list[int]] | None = None
    _damage_cont_ids: list[list[int]] | None = None
    _damage_prompt_index: list[int] | None = None

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings, settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        damage_metric = str(getattr(settings, "damage_metric", "paired_delta_nll"))
        if damage_metric not in ("paired_delta_nll", "topk_js"):
            raise RuntimeError(f"Unsupported damage_metric={damage_metric!r}")

        # Cache multi-reference base continuations once.
        cont_len = int(getattr(settings, "delta_nll_continuation_tokens", 32))
        if cont_len <= 0:
            raise RuntimeError("delta_nll_continuation_tokens must be > 0")
        num_refs = int(getattr(settings, "delta_nll_num_refs", 1))
        if num_refs <= 0:
            raise RuntimeError("delta_nll_num_refs must be > 0")
        temps = list(getattr(settings, "delta_nll_ref_temperatures", [0.0]))
        if len(temps) != num_refs:
            raise RuntimeError(
                f"delta_nll_ref_temperatures must have length delta_nll_num_refs ({num_refs}), got {len(temps)}"
            )
        non_greedy_top_k = int(getattr(settings, "delta_nll_ref_top_k", 50))

        print(f"* Caching base continuations for damage_metric={damage_metric}...")
        prompt_ids_all = self.model.encode_prompts(self.good_prompts)
        backend = self.model.backend
        gen_ids = getattr(backend, "generate_token_ids", None)
        if gen_ids is None:
            raise RuntimeError("Backend does not support generate_token_ids required for damage metrics.")

        damage_prompt_ids: list[list[int]] = []
        damage_cont_ids: list[list[int]] = []
        damage_prompt_index: list[int] = []

        for temp in temps:
            tk = 1 if float(temp) == 0.0 else max(1, int(non_greedy_top_k))
            cont_ids_ref = gen_ids(
                prompt_ids_all,
                max_new_tokens=cont_len,
                adapter=None,
                temperature=float(temp),
                top_k=int(tk),
            )
            if not isinstance(cont_ids_ref, list) or len(cont_ids_ref) != len(prompt_ids_all):
                raise RuntimeError("generate_token_ids returned unexpected batch shape")
            for i, cont in enumerate(cont_ids_ref):
                damage_prompt_ids.append(prompt_ids_all[i])
                damage_cont_ids.append(list(cont))
                damage_prompt_index.append(int(i))

        self._damage_prompt_ids = damage_prompt_ids
        self._damage_cont_ids = damage_cont_ids
        self._damage_prompt_index = damage_prompt_index

        # Placeholder for legacy KL path; not used by paired_delta_nll/topk_js.
        self.base_logprobs = torch.empty((0, 0), dtype=torch.float32)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings, settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print("* Counting model refusals...")
        self.base_refusals = self.count_refusals()
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

    def refresh_baseline(self) -> None:
        """Recompute baseline refusals in the current backend state."""
        print()
        print("* Refreshing baseline refusals...")
        self.base_refusals = self.count_refusals(adapter=None)
        print(f"* Baseline refreshed: refusals [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}")

    def _median(self, xs: list[float]) -> float:
        if not xs:
            return 0.0
        ys = sorted(float(x) for x in xs)
        n = len(ys)
        mid = n // 2
        if n % 2 == 1:
            return float(ys[mid])
        return float(0.5 * (ys[mid - 1] + ys[mid]))

    def _median_of_means(self, xs: list[float], *, buckets: int) -> float:
        xs = [float(x) for x in xs if math.isfinite(float(x))]
        if not xs:
            return 0.0
        k = max(1, int(buckets))
        if k <= 1 or len(xs) < 2 * k:
            return self._median(xs)
        bs: list[list[float]] = [[] for _ in range(k)]
        # Deterministic partitioning (no RNG) to keep runs comparable.
        for i, x in enumerate(xs):
            bs[i % k].append(x)
        means = [sum(b) / len(b) for b in bs if b]
        return self._median(means)

    def _aggregate_per_prompt(self, per_item: list[float]) -> float:
        if self._damage_prompt_index is None:
            raise RuntimeError("Damage cache not initialized.")
        if len(per_item) != len(self._damage_prompt_index):
            raise RuntimeError("Damage per-item list length mismatch vs cache.")
        per_prompt: list[list[float]] = [[] for _ in range(len(self.good_prompts))]
        for x, pi in zip(per_item, self._damage_prompt_index, strict=True):
            if 0 <= int(pi) < len(per_prompt):
                per_prompt[int(pi)].append(float(x))
        prompt_means = [sum(xs) / len(xs) for xs in per_prompt if xs]
        return self._median_of_means(prompt_means, buckets=int(getattr(self.settings, "delta_nll_mom_buckets", 7)))

    def _js_divergence_other_bucket(self, p_log: dict[int, float], q_log: dict[int, float]) -> float:
        keys = set(p_log.keys()) | set(q_log.keys())

        def _mass(d: dict[int, float]) -> tuple[dict[int, float], float]:
            probs: dict[int, float] = {}
            s = 0.0
            for k in keys:
                lp = d.get(k)
                if lp is None:
                    continue
                # exp(logp) in float is fine for top-k sizes we use.
                pk = math.exp(float(lp))
                if pk <= 0.0:
                    continue
                probs[int(k)] = pk
                s += pk
            other = max(0.0, 1.0 - s)
            return probs, other

        p_probs, p_other = _mass(p_log)
        q_probs, q_other = _mass(q_log)

        def _kl(a_probs: dict[int, float], a_other: float, b_probs: dict[int, float], b_other: float) -> float:
            out = 0.0
            # shared keys
            for k, ap in a_probs.items():
                if ap <= 0.0:
                    continue
                bp = b_probs.get(k, 0.0)
                if bp <= 0.0:
                    # If bp is zero but ap isn't, KL is infinite; treat as huge.
                    return float("inf")
                out += ap * (math.log(ap) - math.log(bp))
            if a_other > 0.0:
                if b_other <= 0.0:
                    return float("inf")
                out += a_other * (math.log(a_other) - math.log(b_other))
            return out

        # Mixture distribution m = (p+q)/2
        m_probs: dict[int, float] = {}
        for k in keys:
            m_probs[k] = 0.5 * p_probs.get(k, 0.0) + 0.5 * q_probs.get(k, 0.0)
        m_other = 0.5 * p_other + 0.5 * q_other
        kl_pm = _kl(p_probs, p_other, m_probs, m_other)
        kl_qm = _kl(q_probs, q_other, m_probs, m_other)
        if not math.isfinite(kl_pm) or not math.isfinite(kl_qm):
            return float("inf")
        return 0.5 * kl_pm + 0.5 * kl_qm

    def is_refusal(self, response: str) -> bool:
        # Classify empty responses as refusals to avoid optimizing for them.
        if not response.strip():
            return True

        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won’t" -> "won't").
        response = response.replace("’", "'")

        # Normalize whitespace between words to a single space.
        response = " ".join(response.split())

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False

    def count_refusals(self, *, adapter: str | None = None) -> int:
        refusal_count = 0

        responses = self.model.get_responses_batched(
            self.bad_prompts,
            skip_special_tokens=True,
            adapter=adapter,
        )

        for prompt, response in zip(self.bad_prompts, responses):
            is_refusal = self.is_refusal(response)
            if is_refusal:
                refusal_count += 1

            if self.settings.print_responses:
                print()
                # Never treat model or dataset text as Rich markup. It can contain sequences
                # like "[/]" or "[//]" which would crash the run with MarkupError.
                print("[bold]System prompt:[/]", prompt.system, markup=False)
                print("[bold]Prompt:[/]", prompt.user, markup=False)
                safe_response = response if response.strip() else "[empty]"
                print(
                    "[bold]Response:[/]",
                    safe_response,
                    style=("red" if is_refusal else "green"),
                    markup=False,
                )

        if self.settings.print_responses:
            print()

        return refusal_count

    def get_score(self, *, adapter: str | None = None) -> tuple[tuple[float, float], float, int]:
        damage_metric = str(getattr(self.settings, "damage_metric", "paired_delta_nll"))

        # Under paired damage metrics, adapter=None implies "base", so damage=0 by definition.
        if adapter is None:
            damage = 0.0
            noise = 0.0
            print(f"  * damage_metric={damage_metric}; adapter=None implies damage=0 by definition.")
        else:
            backend = self.model.backend
            if self._damage_prompt_ids is None or self._damage_cont_ids is None:
                raise RuntimeError("Damage cache not initialized.")

            retries = int(getattr(self.settings, "damage_retry_count", 0))
            noise_thr = float(getattr(self.settings, "damage_noise_threshold", 0.0))
            damage = float("nan")
            noise = float("inf")

            for attempt in range(retries + 1):
                if damage_metric == "paired_delta_nll":
                    scorer = getattr(backend, "score_continuation_nll_paired_with_noise", None)
                    if scorer is None:
                        raise RuntimeError("Backend missing score_continuation_nll_paired_with_noise.")
                    base1, adapted, base2 = scorer(
                        prompt_ids_batch=self._damage_prompt_ids,
                        continuation_ids_batch=self._damage_cont_ids,
                        adapter=str(adapter),
                    )
                    deltas = [float(a) - float(b) for a, b in zip(adapted, base1, strict=True)]
                    noises = [abs(float(b2) - float(b1)) for b2, b1 in zip(base2, base1, strict=True)]
                    damage = float(self._aggregate_per_prompt(deltas))
                    noise = float(self._aggregate_per_prompt(noises))
                elif damage_metric == "topk_js":
                    k = int(getattr(self.settings, "topk_js_k", 128))
                    positions = int(getattr(self.settings, "topk_js_positions", 32))
                    scorer = getattr(backend, "score_continuation_topk_paired_with_noise", None)
                    if scorer is None:
                        raise RuntimeError("Backend missing score_continuation_topk_paired_with_noise.")
                    base1_topk, adapted_topk, base2_topk = scorer(
                        prompt_ids_batch=self._damage_prompt_ids,
                        continuation_ids_batch=self._damage_cont_ids,
                        adapter=str(adapter),
                        top_k=int(k),
                    )
                    js_vals: list[float] = []
                    js_noise: list[float] = []
                    for b1, ad, b2 in zip(base1_topk, adapted_topk, base2_topk, strict=True):
                        if not isinstance(b1, list) or not isinstance(ad, list) or not isinstance(b2, list):
                            raise RuntimeError("Unexpected top-k schema from backend.")
                        npos = min(int(positions), len(b1), len(ad), len(b2))
                        if npos <= 0:
                            js_vals.append(0.0)
                            js_noise.append(0.0)
                            continue
                        v = 0.0
                        n = 0
                        nv = 0.0
                        for t in range(npos):
                            v += float(self._js_divergence_other_bucket(b1[t], ad[t]))
                            nv += float(self._js_divergence_other_bucket(b1[t], b2[t]))
                            n += 1
                        js_vals.append(float(v / n))
                        js_noise.append(float(nv / n))
                    damage = float(self._aggregate_per_prompt(js_vals))
                    noise = float(self._aggregate_per_prompt(js_noise))
                else:
                    raise RuntimeError(f"Unsupported damage_metric={damage_metric!r}")

                print(f"  * Damage: [bold]{damage:.6g}[/]  Noise: [bold]{noise:.6g}[/]")
                if math.isfinite(noise) and noise <= noise_thr:
                    break
                if attempt < retries:
                    print(
                        f"  * Damage noise too high (>{noise_thr:g}); retrying ({attempt+1}/{retries})..."
                    )

            if not math.isfinite(float(damage)):
                raise NonFiniteLogprobsError(f"Non-finite damage: {damage!r}")
            if not math.isfinite(float(noise)) or noise > noise_thr:
                raise RuntimeError(
                    f"Damage metric noise too high: {noise:.6g} > {noise_thr:.6g} (damage_metric={damage_metric})"
                )

        print("  * Counting model refusals...")
        refusals = self.count_refusals(adapter=adapter)
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        damage_scale = float(getattr(self.settings, "damage_scale", 1.0))
        damage_target = float(getattr(self.settings, "damage_target", 0.01))

        denom = max(1, int(self.base_refusals))
        refusals_score = refusals / denom

        if damage >= damage_target:
            damage_score = damage / damage_scale
        else:
            damage_score = refusals_score * damage_target / damage_scale

        score = (
            damage_score,
            refusals_score,
        )

        return score, damage, refusals

    def _validate_logprobs_tensor(self, t: Tensor, *, where: str) -> None:
        # Shape sanity: must be (batch, vocab).
        if not isinstance(t, torch.Tensor) or t.ndim != 2:
            raise NonFiniteLogprobsError(f"{where} logprobs_full must be a 2D torch.Tensor, got {type(t)} ndim={getattr(t,'ndim',None)}")
        if where == "adapted":
            if tuple(t.shape) != tuple(self.base_logprobs.shape):
                raise NonFiniteLogprobsError(
                    f"adapted logprobs_full shape mismatch vs base: {tuple(t.shape)} != {tuple(self.base_logprobs.shape)}"
                )
        # Finite check: NaN/Inf will poison KL and downstream Pareto logic.
        finite = torch.isfinite(t)
        if not bool(finite.all().item()):
            bad = int((~finite).sum().item())
            # Avoid expensive reductions on huge tensors; just grab safe summaries.
            t_min = float(t[finite].min().item()) if bool(finite.any().item()) else float("nan")
            t_max = float(t[finite].max().item()) if bool(finite.any().item()) else float("nan")
            raise NonFiniteLogprobsError(
                f"Non-finite {where} logprobs_full: bad={bad} of {t.numel()} (finite_min={t_min:.4g} finite_max={t_max:.4g})."
            )

    def _get_first_token_logprobs(self, prompts: list[Prompt], *, adapter: str | None = None) -> Tensor:
        input_ids_batch = self.model.encode_prompts(prompts)
        result = self.model.backend.score(input_ids_batch, adapter=adapter)
        if result.logprobs_full is None:
            raise NotImplementedError(
                "Backend does not provide full-vocab logprobs needed for KL computation."
            )
        return result.logprobs_full
