# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math
import os
import sys
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from os.path import commonprefix
from pathlib import Path

import huggingface_hub
import optuna
import torch
import torch.nn.functional as F
import transformers
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from huggingface_hub import ModelCard, ModelCardData
from optuna import Trial, TrialPruned
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import BaseSampler, GPSampler, TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import StudyDirection
from optuna.trial import TrialState
from pydantic import ValidationError
from questionary import Choice
from rich.traceback import install

from .analyzer import Analyzer
from .config import QuantizationMethod, SamplerType, Settings
from .evaluator import Evaluator
from .model import AbliterationParameters, Model, get_model_class
from .utils import (
    empty_cache,
    format_duration,
    get_readme_intro,
    get_trial_parameters,
    load_prompts,
    print,
    prompt_password,
    prompt_path,
    prompt_select,
    prompt_text,
)


def create_sampler(settings: Settings) -> BaseSampler:
    if settings.sampler == SamplerType.GP:
        print(
            f"Using [bold]GPSampler[/] "
            f"(deterministic={settings.gp_deterministic_objective})"
        )
        return GPSampler(
            n_startup_trials=settings.n_startup_trials,
            deterministic_objective=settings.gp_deterministic_objective,
            seed=settings.sampler_seed,
        )

    print(
        f"Using [bold]TPESampler[/] "
        f"(multivariate={settings.tpe_multivariate})"
    )
    return TPESampler(
        n_startup_trials=settings.n_startup_trials,
        n_ei_candidates=settings.tpe_n_ei_candidates,
        multivariate=settings.tpe_multivariate,
        seed=settings.sampler_seed,
    )


def obtain_merge_strategy(settings: Settings) -> str | None:
    """
    Prompts the user for how to proceed with saving the model.
    Provides info to the user if the model is quantized on memory use.
    Returns "merge", "adapter", or None (if cancelled/invalid).
    """

    # Prompt for all PEFT models to ensure user is aware of merge implications
    if settings.quantization == QuantizationMethod.BNB_4BIT:
        # Quantized models need special handling - we must reload the base model
        # in full precision to merge the LoRA adapters
        print()
        print(
            "[yellow]Model was loaded with quantization. Merging requires reloading the base model.[/]"
        )
        print(
            "[red](!) WARNING: CPU Merging requires dequantizing the entire model to System RAM.[/]"
        )
        print("[red]    This can lead to SYSTEM FREEZES if you run out of memory.[/]")
        print(
            "[yellow]    Rule of thumb: You need approx. 3x the parameter count in GB.[/]"
        )

        try:
            # Estimate memory requirements by loading the model structure on the "meta" device.
            # This doesn't consume actual RAM but allows us to inspect the parameter count/dtype.
            #
            # Suppress warnings during meta device loading (e.g., "Some weights were not initialized").
            # These are expected and harmless since we're only inspecting model structure, not running inference.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta_model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    device_map="meta",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                footprint_bytes = meta_model.get_memory_footprint()
                footprint_gb = footprint_bytes / (1024**3)
                print(
                    f"[yellow]    Estimated net RAM required for model weights (excluding overhead): [bold]~{footprint_gb:.1f} GB[/][/]"
                )
        except Exception:
            # Fallback if meta loading fails (e.g. owing to custom model code
            # or `bitsandbytes` quantization config issues on the meta device)
            print(
                "[yellow]    Example: A 27B model requires ~80GB RAM. A 70B model requires ~200GB RAM.[/]"
            )
        print()

    merge_choice = prompt_select(
        "How do you want to proceed?",
        choices=[
            Choice(
                title="Merge full model"
                + (
                    ""
                    if settings.quantization == QuantizationMethod.NONE
                    else " (reload base model on CPU - requires high RAM)"
                ),
                value="merge",
            ),
            Choice(
                title="Save LoRA adapter only (can be merged later with llama.cpp or more RAM)",
                value="adapter",
            ),
        ],
    )
    return merge_choice


def save_model(
    model: Model,
    save_directory: str,
    settings: Settings,
    strategy: str | None = None,
) -> None:
    print("Saving model...")
    if strategy is None:
        strategy = obtain_merge_strategy(settings)
        if strategy is None:
            print("[yellow]Action cancelled.[/]")
            return

    if strategy == "adapter":
        model.model.save_pretrained(save_directory)
    else:
        merged_model = model.get_merged_model()
        merged_model.save_pretrained(save_directory)
        del merged_model
        empty_cache()

    model.tokenizer.save_pretrained(save_directory)

    print(f"Model saved to [bold]{save_directory}[/].")


def run():
    # Enable expandable segments to reduce memory fragmentation on multi-GPU setups.
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print(f"[cyan]█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀[/]  v{version('heretic-llm')}")
    print("[cyan]█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░[/]")
    print(
        "[cyan]▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀[/]  [blue underline]https://github.com/p-e-w/heretic[/]"
    )
    print()

    if (
        # There is at least one argument (argv[0] is the program name).
        len(sys.argv) > 1
        # No model has been explicitly provided.
        and "--model" not in sys.argv
        # The last argument is a parameter value rather than a flag (such as "--help").
        and not sys.argv[-1].startswith("-")
    ):
        # Assume the last argument is the model.
        sys.argv.insert(-1, "--model")

    try:
        # The required argument "model" must be provided by the user,
        # either on the command line or in the configuration file.
        settings = Settings()  # ty:ignore[missing-argument]
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")

        for error in error.errors():
            print(f"[bold]{error['loc'][0]}[/]: [yellow]{error['msg']}[/]")

        print()
        print(
            "Run [bold]heretic --help[/] or see [bold]config.default.toml[/] for details about configuration parameters."
        )
        return

    # Apply filesystem/cache settings early (before downloading/loading anything).
    os.makedirs(settings.tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = settings.tmpdir

    os.makedirs(settings.hf_home, exist_ok=True)
    os.environ["HF_HOME"] = settings.hf_home

    # Adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/env.py
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"Detected [bold]{count}[/] CUDA device(s):")
        for i in range(count):
            print(f"* GPU {i}: [bold]{torch.cuda.get_device_name(i)}[/]")
    elif is_xpu_available():
        count = torch.xpu.device_count()
        print(f"Detected [bold]{count}[/] XPU device(s):")
        for i in range(count):
            print(f"* XPU {i}: [bold]{torch.xpu.get_device_name(i)}[/]")
    elif is_mlu_available():
        count = torch.mlu.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MLU device(s):")
        for i in range(count):
            print(f"* MLU {i}: [bold]{torch.mlu.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_sdaa_available():
        count = torch.sdaa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] SDAA device(s):")
        for i in range(count):
            print(f"* SDAA {i}: [bold]{torch.sdaa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_musa_available():
        count = torch.musa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MUSA device(s):")
        for i in range(count):
            print(f"* MUSA {i}: [bold]{torch.musa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_npu_available():
        print(f"NPU detected (CANN version: [bold]{torch.version.cann}[/])")  # ty:ignore[unresolved-attribute]
    elif torch.backends.mps.is_available():
        print("Detected [bold]1[/] MPS device (Apple Metal)")
    else:
        print(
            "[bold yellow]No GPU or other accelerator detected. Operations will be slow.[/]"
        )

    # We don't need gradients as we only do inference.
    torch.set_grad_enabled(False)

    # While determining the optimal batch size, we will try many different batch sizes,
    # resulting in many computation graphs being compiled. Raising the limit (default = 8)
    # avoids errors from TorchDynamo assuming that something is wrong because we
    # recompile too often.
    torch._dynamo.config.cache_size_limit = 64

    # Silence warning spam from Transformers.
    # In my entire career I've never seen a useful warning from that library.
    transformers.logging.set_verbosity_error()

    # We do our own trial logging, so we don't need the INFO messages
    # about parameters and results.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence the warning about multivariate TPE being experimental.
    if settings.sampler == SamplerType.TPE and settings.tpe_multivariate:
        warnings.filterwarnings("ignore", category=ExperimentalWarning)

    study_checkpoint_file = os.path.join(
        settings.study_checkpoint_dir,
        "".join(
            [(c if (c.isalnum() or c in ["_", "-"]) else "--") for c in settings.model]
        )
        + ".jsonl",
    )

    os.makedirs(settings.study_checkpoint_dir, exist_ok=True)
    lock_obj = JournalFileOpenLock(study_checkpoint_file)
    backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
    storage = JournalStorage(backend)

    try:
        existing_study = storage.get_all_studies()[0]
    except IndexError:
        existing_study = None

    if existing_study is not None:
        # A study is in here. Check if it's finished.
        choices = []
        if existing_study.user_attrs["finished"]:
            print(
                "[green]You have already processed this model. How would you like to proceed?[/]"
            )
            choices.append(
                Choice(
                    title="Show the results from the previous run, allowing you to export models, or to run additional trials.",
                    value="continue",
                )
            )
        else:
            print(
                "[yellow]You have already processed this model, but the run was interrupted. How would you like to proceed?[/]",
            )
            choices.append(
                Choice(
                    title="Continue the previous run from where it stopped (will override all specified settings).",
                    value="continue",
                )
            )
        choices.append(
            Choice(
                title="Ignore the previous run and start from scratch. This will delete the checkpoint file and all results from the previous run.",
                value="restart",
            )
        )
        choice = prompt_select("", choices)

        if choice == "continue":
            settings = Settings.model_validate_json(
                existing_study.user_attrs["settings"]
            )
        elif choice == "restart":
            os.unlink(study_checkpoint_file)
            backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
            storage = JournalStorage(backend)
        else:
            print("Cancelled; exiting.")
            return

    model = Model(settings)

    print()
    print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

    if settings.batch_size == 0:
        print()
        print("Determining optimal batch size...")

        batch_size = 1
        best_batch_size = -1
        best_performance = -1

        while batch_size <= settings.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")

            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]

            try:
                # Warmup run to build the computation graph so that part isn't benchmarked.
                model.get_responses(prompts)

                start_time = time.perf_counter()
                responses = model.get_responses(prompts)
                end_time = time.perf_counter()
            except Exception as error:
                if batch_size == 1:
                    # Even a batch size of 1 already fails.
                    # We cannot recover from this.
                    raise

                print(f"[red]Failed[/] ({error})")
                break

            response_lengths = [
                len(model.tokenizer.encode(response)) for response in responses
            ]
            performance = sum(response_lengths) / (end_time - start_time)

            print(f"[green]Ok[/] ([bold]{performance:.0f}[/] tokens/s)")

            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance

            batch_size *= 2

        settings.batch_size = best_batch_size
        print(f"* Chosen batch size: [bold]{settings.batch_size}[/]")

    print()
    print("Checking for common response prefix...")
    responses = model.get_responses_batched(good_prompts[:100] + bad_prompts[:100])

    # Despite being located in os.path, commonprefix actually performs
    # a naive string operation without any path-specific logic,
    # which is exactly what we need here. Trailing spaces are removed
    # to avoid issues where multiple different tokens that all start
    # with a space character lead to the common prefix ending with
    # a space, which would result in an uncommon tokenization.
    model.response_prefix = commonprefix(responses).rstrip(" ")

    # Suppress CoT output.
    if model.response_prefix.startswith("<think>"):
        # Most thinking models.
        model.response_prefix = "<think></think>"
    elif model.response_prefix.startswith("<|channel|>analysis<|message|>"):
        # gpt-oss.
        model.response_prefix = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
    elif model.response_prefix.startswith("<thought>"):
        # Unknown, suggested by user.
        model.response_prefix = "<thought></thought>"
    elif model.response_prefix.startswith("[THINK]"):
        # Unknown, suggested by user.
        model.response_prefix = "[THINK][/THINK]"

    if model.response_prefix:
        print(f"* Prefix found: [bold]{model.response_prefix!r}[/]")
    else:
        print("* None found")

    evaluator = Evaluator(settings, model)

    if settings.evaluate_model is not None:
        print()
        print(f"Loading model [bold]{settings.evaluate_model}[/]...")
        settings.model = settings.evaluate_model
        model.reset_model()
        print("* Evaluating...")
        evaluator.get_score()
        return

    print()
    print("Calculating per-layer refusal directions...")
    print("* Obtaining residuals for good prompts...")
    good_residuals = model.get_residuals_batched(good_prompts)
    print("* Obtaining residuals for bad prompts...")
    bad_residuals = model.get_residuals_batched(bad_prompts)

    harmless_means = good_residuals.mean(dim=0)
    harmful_means = bad_residuals.mean(dim=0)

    def compute_refusal_directions(orthogonalize: bool) -> torch.Tensor:
        directions = F.normalize(harmful_means - harmless_means, p=2, dim=1)

        if orthogonalize:
            # Implements https://huggingface.co/blog/grimjim/projected-abliteration
            # Adjust the refusal directions so that only the component that is
            # orthogonal to the harmless direction is subtracted during abliteration.
            harmless_directions = F.normalize(harmless_means, p=2, dim=1)
            projection_vector = torch.sum(directions * harmless_directions, dim=1)
            directions = directions - projection_vector.unsqueeze(1) * harmless_directions
            directions = F.normalize(directions, p=2, dim=1)

        return directions

    orthogonalize_setting = settings.orthogonalize_direction
    if isinstance(orthogonalize_setting, bool):
        refusal_directions = compute_refusal_directions(orthogonalize_setting)
        refusal_directions_raw = None
        refusal_directions_ortho = None
    else:
        # A/B gating mode: precompute both and decide later.
        refusal_directions_raw = compute_refusal_directions(False)
        refusal_directions_ortho = compute_refusal_directions(True)
        refusal_directions = refusal_directions_raw

    analyzer = Analyzer(settings, model, good_residuals, bad_residuals)

    if settings.print_residual_geometry:
        analyzer.print_residual_geometry()

    if settings.plot_residuals:
        analyzer.plot_residuals()

    # We don't need the residuals after computing refusal directions.
    del good_residuals, bad_residuals, analyzer
    empty_cache()

    trial_index = 0
    start_index = 0
    start_time = time.perf_counter()

    # A/B gating state for orthogonalization.
    locked_orthogonalize_direction: bool | None = None
    gating_trials_total = 0
    gating_mode = False
    gating_index = 0

    def objective(trial: Trial) -> tuple[float, float]:
        nonlocal trial_index, gating_index
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        # Decide which refusal directions to use for this trial.
        if locked_orthogonalize_direction is not None:
            ortho_choice = locked_orthogonalize_direction
            trial.set_user_attr("gating_phase", False)
        elif gating_mode:
            # Alternate to keep the split roughly even.
            ortho_choice = (gating_index % 2) == 1
            gating_index += 1
            trial.set_user_attr("gating_phase", True)
        else:
            # No gating: fall back to the (already validated) setting.
            setting = settings.orthogonalize_direction
            ortho_choice = setting if isinstance(setting, bool) else False
            trial.set_user_attr("gating_phase", False)

        trial.set_user_attr("orthogonalize_direction", ortho_choice)

        current_refusal_directions = refusal_directions
        if refusal_directions_raw is not None and refusal_directions_ortho is not None:
            current_refusal_directions = (
                refusal_directions_ortho if ortho_choice else refusal_directions_raw
            )

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        last_layer_index = len(model.get_layers()) - 1

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}

        for component in model.get_abliterable_components():
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            max_weight = trial.suggest_float(
                f"{component}.max_weight",
                settings.max_weight_min,
                settings.max_weight_max,
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            # For sampling purposes, min_weight is expressed as a fraction of max_weight,
            # again because multivariate TPE doesn't support variable-range parameters.
            # The value is transformed into the actual min_weight value below.
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                0.6 * last_layer_index,
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})

        print()
        print(
            f"Running trial [bold]{trial_index}[/] of [bold]{settings.n_trials}[/]..."
        )
        print("* Parameters:")
        print(f"  * orthogonalize_direction = [bold]{ortho_choice}[/]")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print("* Resetting model...")
        model.reset_model()
        print("* Abliterating...")
        model.abliterate(current_refusal_directions, direction_index, parameters)
        print("* Evaluating...")
        score, kl_divergence, refusals = evaluator.get_score()

        elapsed_time = time.perf_counter() - start_time
        remaining_time = (elapsed_time / (trial_index - start_index)) * (
            settings.n_trials - trial_index
        )
        print()
        print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed_time)}[/][/]")
        if trial_index < settings.n_trials:
            print(
                f"[grey50]Estimated remaining time: [bold]{format_duration(remaining_time)}[/][/]"
            )

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)

        return score

    def objective_wrapper(trial: Trial) -> tuple[float, float]:
        try:
            return objective(trial)
        except KeyboardInterrupt:
            # Stop the study gracefully on Ctrl+C.
            trial.study.stop()
            raise TrialPruned()

    sampler = create_sampler(settings)
    study = optuna.create_study(
        study_name="heretic",
        sampler=sampler,
        storage=storage,
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
        load_if_exists=True,
    )

    study.set_user_attr("settings", settings.model_dump_json())
    study.set_user_attr("finished", False)

    def count_completed_trials() -> int:
        # Count number of complete trials to compute trials to run.
        return sum([(1 if t.state == TrialState.COMPLETE else 0) for t in study.trials])

    def count_completed_gating_trials() -> int:
        return sum(
            [
                1
                for t in study.trials
                if t.state == TrialState.COMPLETE and t.user_attrs.get("gating_phase") is True
            ]
        )

    def run_trials(n: int) -> None:
        nonlocal start_index, start_time
        if n <= 0:
            return
        start_index = trial_index
        start_time = time.perf_counter()
        study.optimize(objective_wrapper, n_trials=n)

    start_index = trial_index = count_completed_trials()
    if start_index > 0:
        print("Resuming existing study.")

    try:
        # Determine orthogonalization mode (including resume support).
        existing_lock = study.user_attrs.get("locked_orthogonalize_direction")
        if isinstance(existing_lock, bool):
            locked_orthogonalize_direction = existing_lock

        orthogonalize_setting = settings.orthogonalize_direction
        if locked_orthogonalize_direction is None:
            if isinstance(orthogonalize_setting, bool):
                locked_orthogonalize_direction = orthogonalize_setting
            else:
                gating_trials_total = int(orthogonalize_setting)
                if gating_trials_total == 0:
                    locked_orthogonalize_direction = False

        # If already locked, ensure the active directions match and discard extras.
        if (
            locked_orthogonalize_direction is not None
            and refusal_directions_raw is not None
            and refusal_directions_ortho is not None
        ):
            refusal_directions = (
                refusal_directions_ortho
                if locked_orthogonalize_direction
                else refusal_directions_raw
            )
            refusal_directions_raw = None
            refusal_directions_ortho = None
            empty_cache()

        # Run early A/B gating trials if requested and not yet locked.
        if (
            locked_orthogonalize_direction is None
            and gating_trials_total > 0
            and refusal_directions_raw is not None
            and refusal_directions_ortho is not None
        ):
            completed_gating = count_completed_gating_trials()
            gating_index = completed_gating
            to_run = min(
                gating_trials_total - completed_gating,
                settings.n_trials - count_completed_trials(),
            )
            if to_run > 0:
                print()
                print(
                    f"Running [bold]{to_run}[/] early A/B trials to choose [bold]orthogonalize_direction[/]..."
                )
                gating_mode = True
                run_trials(to_run)
                gating_mode = False

            # If gating finished (and we have both branches), pick the winner and lock it.
            completed_gating = count_completed_gating_trials()
            if completed_gating >= gating_trials_total:
                gating_trials = [
                    t
                    for t in study.trials
                    if t.state == TrialState.COMPLETE and t.user_attrs.get("gating_phase") is True
                ]
                trials_by_choice: dict[bool, list[Trial]] = {False: [], True: []}
                for t in gating_trials:
                    choice = bool(t.user_attrs.get("orthogonalize_direction", False))
                    trials_by_choice[choice].append(t)

                def best_pair(trials: list[Trial]) -> tuple[int, float] | None:
                    if not trials:
                        return None
                    # Per-branch Pareto preference:
                    # primary: min refusals, tie-break: min KL.
                    sorted_trials = sorted(
                        trials,
                        key=lambda tr: (
                            tr.user_attrs.get("refusals", math.inf),
                            tr.user_attrs.get("kl_divergence", math.inf),
                        ),
                    )
                    best = sorted_trials[0]
                    return (
                        int(best.user_attrs.get("refusals", math.inf)),
                        float(best.user_attrs.get("kl_divergence", math.inf)),
                    )

                best_false = best_pair(trials_by_choice[False])
                best_true = best_pair(trials_by_choice[True])

                if best_false is not None and best_true is not None:
                    locked_orthogonalize_direction = best_true < best_false
                    study.set_user_attr(
                        "locked_orthogonalize_direction", locked_orthogonalize_direction
                    )

                    refusal_directions = (
                        refusal_directions_ortho
                        if locked_orthogonalize_direction
                        else refusal_directions_raw
                    )
                    refusal_directions_raw = None
                    refusal_directions_ortho = None
                    empty_cache()

                    print()
                    print(
                        f"Locked [bold]orthogonalize_direction[/] to [bold]{locked_orthogonalize_direction}[/] "
                        f"based on early A/B trials (best false={best_false}, best true={best_true})."
                    )

        # Continue (or start) main optimization.
        run_trials(settings.n_trials - count_completed_trials())

    except KeyboardInterrupt:
        # This additional handler takes care of the small chance that KeyboardInterrupt
        # is raised just between trials, which wouldn't be caught by the handler
        # defined in objective_wrapper above.
        pass

    if count_completed_trials() == settings.n_trials:
        study.set_user_attr("finished", True)

    def refusal_directions_for_trial(trial: Trial) -> torch.Tensor:
        default_choice = settings.orthogonalize_direction
        default_choice = default_choice if isinstance(default_choice, bool) else False
        choice = bool(
            trial.user_attrs.get(
                "orthogonalize_direction", default_choice
            )
        )
        if refusal_directions_raw is not None and refusal_directions_ortho is not None:
            return refusal_directions_ortho if choice else refusal_directions_raw
        return refusal_directions

    while True:
        # If no trials at all have been evaluated, the study must have been stopped
        # by pressing Ctrl+C while the first trial was running. In this case, we just
        # re-raise the interrupt to invoke the standard handler defined below.
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if locked_orthogonalize_direction is None:
            lock_from_study = study.user_attrs.get("locked_orthogonalize_direction")
            if isinstance(lock_from_study, bool):
                locked_orthogonalize_direction = lock_from_study

        if locked_orthogonalize_direction is not None:
            # After locking, hide trials from the losing branch so they can't be selected
            # (we discard its precomputed directions to save memory).
            completed_trials = [
                t
                for t in completed_trials
                if bool(t.user_attrs.get("orthogonalize_direction", locked_orthogonalize_direction))
                == locked_orthogonalize_direction
            ]
        if not completed_trials:
            raise KeyboardInterrupt

        # Get the Pareto front of trials. We can't use study.best_trials directly
        # as get_score() doesn't return the pure KL divergence and refusal count.
        # Note: Unlike study.best_trials, this does not handle objective constraints.
        sorted_trials = sorted(
            completed_trials,
            key=lambda trial: (
                trial.user_attrs["refusals"],
                trial.user_attrs["kl_divergence"],
            ),
        )
        min_divergence = math.inf
        best_trials = []
        for trial in sorted_trials:
            kl_divergence = trial.user_attrs["kl_divergence"]
            if kl_divergence < min_divergence:
                min_divergence = kl_divergence
                best_trials.append(trial)

        choices = [
            Choice(
                title=(
                    f"[Trial {trial.user_attrs['index']:>3}] "
                    f"Refusals: {trial.user_attrs['refusals']:>2}/{len(evaluator.bad_prompts)}, "
                    f"KL divergence: {trial.user_attrs['kl_divergence']:.4f}"
                ),
                value=trial,
            )
            for trial in best_trials
        ]

        choices.append(
            Choice(
                title="Continue optimization (run more trials)",
                value="continue",
            )
        )

        choices.append(
            Choice(
                title="None (exit program)",
                value="",
            )
        )

        print()
        print("[bold green]Optimization finished![/]")
        print()
        print(
            (
                "The following trials resulted in Pareto optimal combinations of refusals and KL divergence. "
                "After selecting a trial, you will be able to save the model, upload it to Hugging Face, "
                "or chat with it to test how well it works. You can return to this menu later to select a different trial. "
                "[yellow]Note that KL divergence values above 1 usually indicate significant damage to the original model's capabilities.[/]"
            )
        )

        while True:
            print()
            trial = prompt_select("Which trial do you want to use?", choices)

            if trial == "continue":
                while True:
                    try:
                        n_more_trials = int(
                            prompt_text("How many more trials do you want to run?")
                        )
                        if n_more_trials > 0:
                            break
                        print("[red]Please enter a number greater than 0.[/]")
                    except ValueError:
                        print("[red]Invalid input. Please enter a number.[/]")

                settings.n_trials += n_more_trials
                study.set_user_attr("settings", settings.model_dump_json())
                study.set_user_attr("finished", False)
                try:
                    run_trials(settings.n_trials - count_completed_trials())
                except KeyboardInterrupt:
                    pass
                if count_completed_trials() == settings.n_trials:
                    study.set_user_attr("finished", True)
                break

            elif trial is None or trial == "":
                return

            print()
            print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]...")
            print("* Parameters:")
            for name, value in get_trial_parameters(trial).items():
                print(f"  * {name} = [bold]{value}[/]")
            print("* Resetting model...")
            model.reset_model()
            print("* Abliterating...")
            model.abliterate(
                refusal_directions_for_trial(trial),
                trial.user_attrs["direction_index"],
                {
                    k: AbliterationParameters(**v)
                    for k, v in trial.user_attrs["parameters"].items()
                },
            )

            while True:
                print()
                action = prompt_select(
                    "What do you want to do with the decensored model?",
                    [
                        "Save the model to a local folder",
                        "Upload the model to Hugging Face",
                        "Chat with the model",
                        "Nothing (return to trial selection menu)",
                    ],
                )

                if (
                    action is None
                    or action == "Nothing (return to trial selection menu)"
                ):
                    break

                # All actions are wrapped in a try/except block so that if an error occurs,
                # another action can be tried, instead of the program crashing and losing
                # the optimized model.
                try:
                    match action:
                        case "Save the model to a local folder":
                            save_directory = prompt_path("Path to the folder:")
                            if not save_directory:
                                continue

                            save_model(
                                model,
                                save_directory,
                                settings,
                            )

                        case "Upload the model to Hugging Face":
                            # We don't use huggingface_hub.login() because that stores the token on disk,
                            # and since this program will often be run on rented or shared GPU servers,
                            # it's better to not persist credentials.
                            token = huggingface_hub.get_token()
                            if not token:
                                token = prompt_password("Hugging Face access token:")
                            if not token:
                                continue

                            user = huggingface_hub.whoami(token)
                            fullname = user.get(
                                "fullname",
                                user.get("name", "unknown user"),
                            )
                            email = user.get("email", "no email found")
                            print(f"Logged in as [bold]{fullname} ({email})[/]")

                            repo_id = prompt_text(
                                "Name of repository:",
                                default=f"{user['name']}/{Path(settings.model).name}-heretic",
                            )

                            visibility = prompt_select(
                                "Should the repository be public or private?",
                                [
                                    "Public",
                                    "Private",
                                ],
                            )
                            private = visibility == "Private"

                            strategy = obtain_merge_strategy(settings)
                            if strategy is None:
                                print("[yellow]Action cancelled.[/]")
                                continue

                            if strategy == "adapter":
                                print("Uploading LoRA adapter...")
                                model.model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                            else:
                                print("Uploading merged model...")
                                merged_model = model.get_merged_model()
                                merged_model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                                del merged_model
                                empty_cache()

                            model.tokenizer.push_to_hub(
                                repo_id,
                                private=private,
                                token=token,
                            )

                            # If the model path doesn't exist locally, it can be assumed
                            # to be a model hosted on the Hugging Face Hub, in which case
                            # we can retrieve the model card.
                            if not Path(settings.model).exists():
                                card = ModelCard.load(settings.model)
                                if card.data is None:
                                    card.data = ModelCardData()
                                if card.data.tags is None:
                                    card.data.tags = []
                                card.data.tags.append("heretic")
                                card.data.tags.append("uncensored")
                                card.data.tags.append("decensored")
                                card.data.tags.append("abliterated")
                                card.text = (
                                    get_readme_intro(
                                        settings,
                                        trial,
                                        evaluator.base_refusals,
                                        evaluator.bad_prompts,
                                    )
                                    + card.text
                                )
                                card.push_to_hub(repo_id, token=token)

                            print(f"Model uploaded to [bold]{repo_id}[/].")

                        case "Chat with the model":
                            print()
                            print(
                                "[cyan]Press Ctrl+C at any time to return to the menu.[/]"
                            )

                            chat = [
                                {"role": "system", "content": settings.system_prompt},
                            ]

                            while True:
                                try:
                                    message = prompt_text(
                                        "User:",
                                        qmark=">",
                                        unsafe=True,
                                    )
                                    if not message:
                                        break
                                    chat.append({"role": "user", "content": message})

                                    print("[bold]Assistant:[/] ", end="")
                                    response = model.stream_chat_response(chat)
                                    chat.append(
                                        {"role": "assistant", "content": response}
                                    )
                                except (KeyboardInterrupt, EOFError):
                                    # Ctrl+C/Ctrl+D
                                    break

                except Exception as error:
                    print(f"[red]Error: {error}[/]")


def main():
    # Install Rich traceback handler.
    install()

    try:
        run()
    except BaseException as error:
        # Transformers appears to handle KeyboardInterrupt (or BaseException)
        # internally in some places, which can re-raise a different error in the handler,
        # masking the root cause. We therefore check both the error itself and its context.
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise
