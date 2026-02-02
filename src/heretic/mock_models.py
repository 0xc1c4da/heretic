# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Callable

import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module


@dataclass(frozen=True)
class TinyMiniMaxM2Spec:
    hidden_size: int = 64
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 2048
    sliding_window: int | None = 256
    num_experts_per_tok: int = 2
    num_local_experts: int = 2
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    seed: int = 0


def _infer_vocab_size(source_dir: str) -> int:
    """
    Infer vocab size from `vocab.json` (preferred) or fall back to a conservative default.

    We intentionally match the tokenizer's vocab so random prompts can't produce token IDs
    outside the embedding/lm_head range.
    """
    size = 0

    vocab_path = os.path.join(source_dir, "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # Values are token IDs.
        max_id = max(vocab.values()) if vocab else 0
        size = max(size, int(max_id) + 1)

    # MiniMax uses "added_tokens_decoder" with IDs above the base vocab.
    tok_cfg_path = os.path.join(source_dir, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        try:
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            added = tok_cfg.get("added_tokens_decoder") or {}
            if isinstance(added, dict) and added:
                max_added = max(int(k) for k in added.keys())
                size = max(size, max_added + 1)
        except Exception:
            pass

    # Conservative fallback.
    if size <= 0:
        return 65536
    return size


def _copy_if_exists(src: str, dst: str) -> None:
    if os.path.exists(src):
        shutil.copy2(src, dst)


def looks_like_minimax_m2_source_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "configuration_minimax_m2.py"))
        and os.path.exists(os.path.join(path, "modeling_minimax_m2.py"))
    )


def materialize_tiny_minimax_m2_repo(
    *,
    source_dir: str,
    out_dir: str,
    spec: TinyMiniMaxM2Spec,
    logger: Callable[[str], None],
) -> str:
    """
    Create a tiny-but-real MiniMax M2 checkpoint directory suitable for fast iteration.

    - Keeps the *same* model code (remote-code style) so module names/structure match.
    - Produces real weight files (tiny) so `from_pretrained()` works normally.
    - Uses the source tokenizer files, but sets config.vocab_size to match them.
    """
    if not looks_like_minimax_m2_source_dir(source_dir):
        raise ValueError(f"Not a MiniMax M2 source dir: {source_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # Always copy model code + tokenizer artifacts (cheap) so updates to the source dir
    # (e.g. adding a missing `chat_template.jinja`) propagate even when weights already exist.
    _copy_if_exists(
        os.path.join(source_dir, "configuration_minimax_m2.py"),
        os.path.join(out_dir, "configuration_minimax_m2.py"),
    )
    _copy_if_exists(
        os.path.join(source_dir, "modeling_minimax_m2.py"),
        os.path.join(out_dir, "modeling_minimax_m2.py"),
    )

    # Copy tokenizer + generation config artifacts (so heretic can load tokenizer from the same dir).
    for fname in (
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "generation_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
    ):
        _copy_if_exists(os.path.join(source_dir, fname), os.path.join(out_dir, fname))

    # If already materialized, reuse weights/config.
    if os.path.exists(os.path.join(out_dir, "pytorch_model.bin")) and os.path.exists(
        os.path.join(out_dir, "config.json")
    ):
        # Validate that the existing checkpoint matches what we'd generate now.
        # If it doesn't, we regenerate in-place to avoid subtle runtime failures.
        try:
            with open(os.path.join(out_dir, "config.json"), "r", encoding="utf-8") as f:
                existing_cfg = json.load(f)
            expected_vocab = _infer_vocab_size(source_dir)
            if int(existing_cfg.get("vocab_size", -1)) == int(expected_vocab):
                return out_dir
        except Exception:
            pass

        try:
            os.unlink(os.path.join(out_dir, "pytorch_model.bin"))
        except Exception:
            pass

    # Instantiate tiny config + model and save weights.
    MiniMaxM2Config = get_class_from_dynamic_module(
        "configuration_minimax_m2.MiniMaxM2Config",
        source_dir,
    )
    MiniMaxM2ForCausalLM = get_class_from_dynamic_module(
        "modeling_minimax_m2.MiniMaxM2ForCausalLM",
        source_dir,
    )

    vocab_size = _infer_vocab_size(source_dir)

    # Deterministic init helps with reproducibility and debugging.
    torch.manual_seed(spec.seed)

    config = MiniMaxM2Config(
        vocab_size=vocab_size,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        num_key_value_heads=spec.num_key_value_heads,
        max_position_embeddings=spec.max_position_embeddings,
        sliding_window=spec.sliding_window,
        num_experts_per_tok=spec.num_experts_per_tok,
        num_local_experts=spec.num_local_experts,
        router_aux_loss_coef=spec.router_aux_loss_coef,
        router_jitter_noise=spec.router_jitter_noise,
    )

    # This vendored MiniMax code expects ROPE_INIT_FUNCTIONS["default"], but some transformers
    # builds only expose explicit keys like "linear"/"dynamic"/"llama3"/...
    # Setting rope_scaling forces a compatible rope_type.
    config.rope_scaling = {"rope_type": "linear", "factor": 1.0}  # type: ignore[attr-defined]

    # Remote-code glue so AutoModel/AutoConfig can import from this directory.
    config.architectures = ["MiniMaxM2ForCausalLM"]
    config.auto_map = {
        "AutoConfig": "configuration_minimax_m2.MiniMaxM2Config",
        "AutoModelForCausalLM": "modeling_minimax_m2.MiniMaxM2ForCausalLM",
    }

    model = MiniMaxM2ForCausalLM(config)
    model.eval()

    # Write config + weights (avoid safetensors to keep it simple).
    # We avoid `model.save_pretrained` because vendored remote-code can disagree with
    # transformers metadata (`_tied_weights_keys`) and fail at save time.
    config.save_pretrained(out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))

    logger(
        f"[green]Materialized tiny MiniMax M2 checkpoint[/] at [bold]{out_dir}[/] "
        f"(layers={spec.num_hidden_layers}, hidden={spec.hidden_size}, vocab={vocab_size})."
    )
    return out_dir
