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


@dataclass(frozen=True)
class TinyKimiK25Spec:
    # ---- Text (DeepseekV3) ----
    hidden_size: int = 64
    intermediate_size: int = 256
    moe_intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    max_position_embeddings: int = 512

    n_routed_experts: int = 2
    n_shared_experts: int = 1
    num_experts_per_tok: int = 1

    # Keep MLA ranks small but non-zero.
    kv_lora_rank: int = 8
    q_lora_rank: int = 16

    # Choose dims so that:
    # - q_head_dim = qk_nope + qk_rope
    # - v_head_dim matches q_head_dim
    # - hidden_size == num_attention_heads * v_head_dim (default 64 == 4 * 16)
    qk_rope_head_dim: int = 8
    qk_nope_head_dim: int = 8
    v_head_dim: int = 16

    # Ensure we exercise MoE in tiny configs.
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1

    # ---- Vision tower (still instantiated by the wrapper, even for text-only) ----
    patch_size: int = 14
    vt_hidden_size: int = 64
    vt_intermediate_size: int = 256
    vt_num_hidden_layers: int = 2
    vt_num_attention_heads: int = 4
    merge_kernel_size: tuple[int, int] = (2, 2)
    video_attn_type: str = "spatial_temporal"
    merge_type: str = "sd2_tpool"

    # ---- MM projector ----
    mm_projector_type: str = "mlp"
    mm_hidden_size: int = 64

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


def looks_like_kimi_k25_source_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "configuration_kimi_k25.py"))
        and os.path.exists(os.path.join(path, "configuration_deepseek.py"))
        and os.path.exists(os.path.join(path, "modeling_kimi_k25.py"))
        and os.path.exists(os.path.join(path, "modeling_deepseek.py"))
        and os.path.exists(os.path.join(path, "tokenizer_config.json"))
        and os.path.exists(os.path.join(path, "tiktoken.model"))
    )


def materialize_tiny_kimi_k25_repo(
    *,
    source_dir: str,
    out_dir: str,
    spec: TinyKimiK25Spec,
    logger: Callable[[str], None],
) -> str:
    """
    Create a tiny-but-real Kimi K2.5 checkpoint directory suitable for fast iteration.

    Notes:
    - Kimi K2.5 is a composite wrapper that instantiates vision + projector + LM even if we
      only test text-only generation. So the tiny config keeps the vision tower small and forces
      eager attention to avoid FlashAttention dependencies.
    - We keep vocab size >= the tokenizer's highest special-token id (PAD is 163839).
    """
    if not looks_like_kimi_k25_source_dir(source_dir):
        raise ValueError(f"Not a Kimi K2.5 source dir: {source_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # Copy remote-code model + tokenizer artifacts (cheap).
    for fname in (
        # model code
        "configuration_kimi_k25.py",
        "configuration_deepseek.py",
        "modeling_kimi_k25.py",
        "modeling_deepseek.py",
        # tokenizer remote code + deps
        "tokenization_kimi.py",
        "tool_declaration_ts.py",
        "tiktoken.model",
        "tokenizer_config.json",
        "chat_template.jinja",
        # optional niceties (if present)
        "generation_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
    ):
        _copy_if_exists(os.path.join(source_dir, fname), os.path.join(out_dir, fname))

    # Kimi remote-code bug/workaround: `MoonViT3dEncoder.__init__` references
    # `self.use_deterministic_attn` without defining it first.
    kimi_model_path = os.path.join(out_dir, "modeling_kimi_k25.py")
    kimi_cfg_path = os.path.join(out_dir, "configuration_kimi_k25.py")
    try:
        if os.path.exists(kimi_model_path):
            with open(kimi_model_path, "r", encoding="utf-8") as f:
                kimi_src = f.read()
            anchor = "\n        self.video_attn_type = video_attn_type\n        self.rope_2d ="
            if anchor in kimi_src:
                kimi_src = kimi_src.replace(
                    anchor,
                    "\n".join(
                        [
                            "",
                            "        self.video_attn_type = video_attn_type",
                            "        # Default to non-deterministic attention for toy checkpoints.",
                            "        self.use_deterministic_attn = False",
                            "        self.rope_2d =",
                        ]
                    ),
                )
                with open(kimi_model_path, "w", encoding="utf-8") as f:
                    f.write(kimi_src)
    except Exception:
        pass

    # Ensure the tiny checkpoint can reload without FlashAttention installed.
    # The serialized `vision_config` drops private keys like `_attn_implementation`, so we
    # patch the default in the copied config to `"eager"`.
    try:
        if os.path.exists(kimi_cfg_path):
            with open(kimi_cfg_path, "r", encoding="utf-8") as f:
                cfg_src = f.read()
            cfg_src2 = cfg_src.replace(
                "_attn_implementation: str = 'flash_attention_2'",
                "_attn_implementation: str = 'eager'",
            )
            if cfg_src2 != cfg_src:
                with open(kimi_cfg_path, "w", encoding="utf-8") as f:
                    f.write(cfg_src2)
    except Exception:
        pass

    # Transformers-side API drift: newer `PreTrainedModel.init_weights()` calls
    # `tie_weights(recompute_mapping=...)`. The upstream Kimi remote-code defines
    # `tie_weights(self)` without kwargs. Make it signature-compatible.
    try:
        if os.path.exists(kimi_model_path):
            with open(kimi_model_path, "r", encoding="utf-8") as f:
                kimi_src = f.read()
            needle = "def tie_weights(self):"
            if needle in kimi_src:
                kimi_src = kimi_src.replace(
                    needle,
                    "def tie_weights(self, *args, **kwargs):",
                )
                with open(kimi_model_path, "w", encoding="utf-8") as f:
                    f.write(kimi_src)
    except Exception:
        pass

    # The upstream Kimi remote-code is slightly ahead of our pinned transformers:
    # it imports `is_torch_fx_available`, which does not exist in this transformers rev.
    # Patch the copied file (not the source dir) to keep the toy checkpoint loadable.
    deepseek_path = os.path.join(out_dir, "modeling_deepseek.py")
    try:
        if os.path.exists(deepseek_path):
            with open(deepseek_path, "r", encoding="utf-8") as f:
                deepseek_src = f.read()
            changed = False
            needle = "from transformers.utils.import_utils import is_torch_fx_available"
            if needle in deepseek_src:
                deepseek_src = deepseek_src.replace(
                    needle,
                    "\n".join(
                        [
                            "try:",
                            "    from transformers.utils.import_utils import is_torch_fx_available",
                            "except Exception:  # pragma: no cover",
                            "    # Transformers 5 removed/moved `is_torch_fx_available` in some builds.",
                            "    # For our toy models we can safely disable FX-specific branches.",
                            "    def is_torch_fx_available() -> bool:",
                            "        return False",
                        ]
                    ),
                )
                changed = True

            # Transformers cache API drift: some builds do not provide
            # `DynamicCache.from_legacy_cache`. The upstream code calls it even when
            # `past_key_values is None` on the first generation step.
            legacy_snippet = "\n".join(
                [
                    "                past_key_values = DynamicCache.from_legacy_cache(",
                    "                    past_key_values)",
                ]
            )
            if legacy_snippet in deepseek_src:
                deepseek_src = deepseek_src.replace(
                    legacy_snippet,
                    "\n".join(
                        [
                            "                if past_key_values is None:",
                            "                    past_key_values = DynamicCache()",
                            "                elif hasattr(DynamicCache, \"from_legacy_cache\"):",
                            "                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)",
                            "                else:",
                            "                    past_key_values = DynamicCache()",
                        ]
                    ),
                )
                changed = True

            legacy_out_snippet = "\n".join(
                [
                    "            next_cache = (next_decoder_cache.to_legacy_cache()",
                    "                          if use_legacy_cache else next_decoder_cache)",
                ]
            )
            if legacy_out_snippet in deepseek_src:
                deepseek_src = deepseek_src.replace(
                    legacy_out_snippet,
                    "\n".join(
                        [
                            "            if use_legacy_cache and hasattr(next_decoder_cache, \"to_legacy_cache\"):",
                            "                next_cache = next_decoder_cache.to_legacy_cache()",
                            "            else:",
                            "                next_cache = next_decoder_cache",
                        ]
                    ),
                )
                changed = True

            if changed:
                with open(deepseek_path, "w", encoding="utf-8") as f:
                    f.write(deepseek_src)
    except Exception:
        # Best-effort patch; if it fails, the subsequent load will surface the error.
        pass

    # If already materialized, reuse weights/config if it matches the expected vocab size.
    if os.path.exists(os.path.join(out_dir, "pytorch_model.bin")) and os.path.exists(
        os.path.join(out_dir, "config.json")
    ):
        try:
            with open(os.path.join(out_dir, "config.json"), "r", encoding="utf-8") as f:
                existing_cfg = json.load(f)
            expected_vocab = _infer_vocab_size(source_dir)
            # Kimi stores vocab size under the nested text config.
            existing_text_cfg = existing_cfg.get("text_config", {}) or {}
            existing_vocab = existing_text_cfg.get(
                "vocab_size", existing_cfg.get("vocab_size")
            )

            # If we change the tiny-spec defaults, we should regenerate rather than
            # silently reusing stale weights/config.
            expected_n_group = 1
            expected_topk_group = 1
            existing_n_group = existing_text_cfg.get("n_group", None)
            existing_topk_group = existing_text_cfg.get("topk_group", None)

            if (
                int(existing_vocab) == int(expected_vocab)
                and int(existing_n_group) == expected_n_group
                and int(existing_topk_group) == expected_topk_group
            ):
                return out_dir
        except Exception:
            pass

        try:
            os.unlink(os.path.join(out_dir, "pytorch_model.bin"))
        except Exception:
            pass
        try:
            os.unlink(os.path.join(out_dir, "config.json"))
        except Exception:
            pass

    # Instantiate tiny config + model and save weights.
    KimiK25Config = get_class_from_dynamic_module(
        "configuration_kimi_k25.KimiK25Config",
        out_dir,
        force_download=True,
    )
    DeepseekV3Config = get_class_from_dynamic_module(
        "configuration_deepseek.DeepseekV3Config",
        out_dir,
        force_download=True,
    )
    KimiK25ForConditionalGeneration = get_class_from_dynamic_module(
        "modeling_kimi_k25.KimiK25ForConditionalGeneration",
        out_dir,
        force_download=True,
    )

    vocab_size = _infer_vocab_size(source_dir)

    # These ids are fixed by the released tokenizer_config for Kimi K2.5.
    bos_token_id = 163584
    eos_token_id = 163585
    pad_token_id = 163839
    media_placeholder_token_id = 163605

    torch.manual_seed(spec.seed)

    text_config = DeepseekV3Config(
        vocab_size=vocab_size,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        moe_intermediate_size=spec.moe_intermediate_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        num_key_value_heads=spec.num_key_value_heads,
        n_routed_experts=spec.n_routed_experts,
        n_shared_experts=spec.n_shared_experts,
        num_experts_per_tok=spec.num_experts_per_tok,
        # Tiny configs must keep the routing-group math consistent.
        n_group=1,
        topk_group=1,
        moe_layer_freq=spec.moe_layer_freq,
        first_k_dense_replace=spec.first_k_dense_replace,
        kv_lora_rank=spec.kv_lora_rank,
        q_lora_rank=spec.q_lora_rank,
        qk_rope_head_dim=spec.qk_rope_head_dim,
        qk_nope_head_dim=spec.qk_nope_head_dim,
        v_head_dim=spec.v_head_dim,
        max_position_embeddings=spec.max_position_embeddings,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        # DeepSeek-V3 attention expects either `None` or a dict with a `"type"` key.
        # Use a minimal identity scaling to avoid key errors in remote-code.
        rope_scaling={"type": "linear", "factor": 1.0},
        _attn_implementation="eager",
    )

    vision_config = {
        "patch_size": spec.patch_size,
        "vt_hidden_size": spec.vt_hidden_size,
        "vt_intermediate_size": spec.vt_intermediate_size,
        "vt_num_hidden_layers": spec.vt_num_hidden_layers,
        "vt_num_attention_heads": spec.vt_num_attention_heads,
        "merge_kernel_size": spec.merge_kernel_size,
        "video_attn_type": spec.video_attn_type,
        "merge_type": spec.merge_type,
        "_attn_implementation": "eager",
        # projector bits
        "mm_projector_type": spec.mm_projector_type,
        "mm_hidden_size": spec.mm_hidden_size,
        "text_hidden_size": spec.hidden_size,
        # special tokens
        "media_placeholder_token_id": media_placeholder_token_id,
        "pad_token_id": pad_token_id,
    }

    config = KimiK25Config(
        text_config=text_config,
        vision_config=vision_config,
        media_placeholder_token_id=media_placeholder_token_id,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )

    # Remote-code glue so AutoModel/AutoConfig can import from this directory.
    config.architectures = ["KimiK25ForConditionalGeneration"]
    config.auto_map = {
        "AutoConfig": "configuration_kimi_k25.KimiK25Config",
        "AutoModel": "modeling_kimi_k25.KimiK25ForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_kimi_k25.KimiK25ForConditionalGeneration",
    }

    model = KimiK25ForConditionalGeneration(config)
    model.eval()

    config.save_pretrained(out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))

    logger(
        f"[green]Materialized tiny Kimi K2.5 checkpoint[/] at [bold]{out_dir}[/] "
        f"(layers={spec.num_hidden_layers}, hidden={spec.hidden_size}, vocab={vocab_size})."
    )
    return out_dir
