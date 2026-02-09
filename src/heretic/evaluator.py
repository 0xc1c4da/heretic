# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math

import torch.nn.functional as F
import torch
from torch import Tensor

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

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings, settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = self._get_first_token_logprobs(self.good_prompts)
        self._validate_logprobs_tensor(self.base_logprobs, where="base")

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
        """Recompute baseline logprobs + refusals in the current backend state.

        Some backends (notably SGLang offline/HTTP) can change effective execution state after
        warmup operations like hidden-state capture or adapter loads. Heretic's KL metric is only
        meaningful if the baseline and adapted distributions are captured in the same state.
        """
        print()
        print("* Refreshing baseline distributions/refusals...")
        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = self._get_first_token_logprobs(self.good_prompts, adapter=None)
        self._validate_logprobs_tensor(self.base_logprobs, where="base")
        print("* Counting model refusals...")
        self.base_refusals = self.count_refusals(adapter=None)
        print(
            f"* Baseline refreshed: refusals [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

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
        # For backends with cross-call drift (notably some SGLang stacks), KL is only meaningful
        # if base and adapted distributions are captured within the same backend call/batch.
        supports = self.model.backend.get_metadata().supports
        use_paired = bool(supports.get("score_full_vocab_paired", False))

        if use_paired and adapter is not None:
            print("  * Obtaining paired base/adapted distributions (one-call)...")
            input_ids_batch = self.model.encode_prompts(self.good_prompts)
            supports_noise = bool(supports.get("score_full_vocab_paired_with_noise", False))
            if supports_noise:
                base_lp, adapted_lp, base2_lp = self.model.backend.score_full_vocab_paired_with_noise(
                    input_ids_batch,
                    adapter=str(adapter),
                )
                self._validate_logprobs_tensor(base2_lp, where="base")
                kl_noise = float(
                    F.kl_div(base2_lp, base_lp, reduction="batchmean", log_target=True).item()
                )
                print(f"  * Within-call KL_noise (base||base2): [bold]{kl_noise:.6g}[/]")
                if kl_noise > float(self.settings.paired_kl_noise_threshold):
                    raise RuntimeError(
                        f"Within-call KL_noise too high: {kl_noise:.6g} > {self.settings.paired_kl_noise_threshold}. "
                        "Paired KL metric is not stable enough to optimize."
                    )
            else:
                base_lp, adapted_lp = self.model.backend.score_full_vocab_paired(
                    input_ids_batch,
                    adapter=str(adapter),
                )
            self._validate_logprobs_tensor(base_lp, where="base")
            self._validate_logprobs_tensor(adapted_lp, where="adapted")
            kl_divergence = F.kl_div(
                adapted_lp,
                base_lp,
                reduction="batchmean",
                log_target=True,
            ).item()
        elif use_paired and adapter is None:
            # Under drift, "base vs stored base" is not a stable diagnostic. Define KL(base||base)=0 here.
            print("  * Using paired-scoring backend; adapter=None implies KL=0 by definition.")
            kl_divergence = 0.0
        else:
            print("  * Obtaining first-token probability distributions...")
            logprobs = self._get_first_token_logprobs(self.good_prompts, adapter=adapter)
            self._validate_logprobs_tensor(logprobs, where="adapted")
            kl_divergence = F.kl_div(
                logprobs,
                self.base_logprobs,
                reduction="batchmean",
                log_target=True,
            ).item()

        if not math.isfinite(float(kl_divergence)):
            raise NonFiniteLogprobsError(f"Non-finite KL divergence: {kl_divergence!r}")
        print(f"  * KL divergence: [bold]{float(kl_divergence):.4f}[/]")

        print("  * Counting model refusals...")
        refusals = self.count_refusals(adapter=adapter)
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        kl_divergence_scale = self.settings.kl_divergence_scale
        kl_divergence_target = self.settings.kl_divergence_target

        refusals_score = refusals / self.base_refusals

        if kl_divergence >= kl_divergence_target:
            kld_score = kl_divergence / kl_divergence_scale
        else:
            kld_score = refusals_score * kl_divergence_target / kl_divergence_scale

        score = (
            kld_score,
            refusals_score,
        )

        return score, kl_divergence, refusals

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
