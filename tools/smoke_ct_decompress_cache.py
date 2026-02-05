#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

import torch

from heretic.model import Model
from heretic.utils import Prompt


def _default_max_memory(per_gpu: str) -> dict[int | str, str]:
    if not torch.cuda.is_available():
        return {"cpu": "1200GiB"}
    n = torch.cuda.device_count()
    mm: dict[int | str, str] = {i: per_gpu for i in range(n)}
    mm["cpu"] = "1200GiB"
    return mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="moonshotai/Kimi-K2.5")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--max-memory-per-gpu", default="110GiB")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args()

    # Enable periodic cache stats logging.
    os.environ.setdefault("HERETIC_CT_CACHE_STATS", "1")
    os.environ.setdefault("HERETIC_CT_CACHE_STATS_INTERVAL_S", "5")

    # Avoid persistent upstream freeze behavior; rely on bounded cache.
    os.environ.setdefault("HERETIC_CT_ALLOW_FREEZE", "0")

    settings = SimpleNamespace(
        model=str(args.model),
        evaluate_model=None,
        trust_remote_code=True,
        dtypes=["auto"],
        device_map=str(args.device_map),
        max_memory=_default_max_memory(str(args.max_memory_per_gpu)),
        quantization="none",
        quantization_config_type=None,
        quantization_kwargs=None,
        precision_fallback_dtype="auto",
        precision_debug=False,
        ct_fast_load=True,
        ct_loading_info=False,
        ct_allow_freeze=False,
        mock_tiny_model=False,
        mock_tiny_out_dir="~/.cache/heretic/mock_models",
        mock_tiny_hidden_size=64,
        mock_tiny_intermediate_size=256,
        mock_tiny_num_hidden_layers=2,
        mock_tiny_num_attention_heads=4,
        mock_tiny_num_key_value_heads=4,
        mock_tiny_max_position_embeddings=512,
        mock_tiny_sliding_window=256,
        mock_tiny_num_experts_per_tok=1,
        mock_tiny_num_local_experts=2,
        mock_tiny_seed=0,
        system_prompt="You are a helpful assistant.",
        batch_size=1,
        max_batch_size=1,
        max_response_length=int(args.max_new_tokens),
    )

    m = Model(settings)

    prompts = [
        Prompt(
            system=settings.system_prompt,
            user=f"Write a one-line summary of the number {i}.",
        )
        for i in range(int(args.n_prompts))
    ]

    for r in range(int(args.repeat)):
        t0 = time.time()
        _ = m.get_responses_batched(
            prompts,
            max_new_tokens=int(args.max_new_tokens),
            use_cache=False,
            batch_size=1,
        )
        dt = time.time() - t0
        print(f"[smoke] pass={r+1}/{args.repeat} elapsed_s={dt:.2f}")


if __name__ == "__main__":
    main()

