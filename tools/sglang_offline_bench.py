#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from typing import Any


def _make_token_ids(token_id: int, n: int) -> list[int]:
    return [int(token_id)] * int(n)


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark embedded SGLang Engine throughput (offline).")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tp-size", type=int, default=8)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--with-lora", action="store_true")
    ap.add_argument("--with-hidden-states", action="store_true")
    ap.add_argument("--engine-args-json", default="{}")
    args = ap.parse_args()

    import json
    import torch

    from heretic.backend.sglang_offline import SGLangOfflineBackend
    from heretic.hf_resolve import resolve_model_dir

    extra = json.loads(args.engine_args_json)
    if not isinstance(extra, dict):
        raise SystemExit("--engine-args-json must decode to an object/dict")

    resolved = resolve_model_dir(args.model_path)
    backend = SGLangOfflineBackend(
        model_path=resolved.resolved_dir,
        trust_remote_code=True,
        engine_args={
            "tp_size": int(args.tp_size),
            "attention_backend": "flashinfer",
            "enable_return_hidden_states": True,
            "enable_lora": True,
            "max_lora_rank": 64,
            "lora_target_modules": ["o_proj", "down_proj"],
            "max_loras_per_batch": 2,
            "max_loaded_loras": 8,
            "lora_backend": "csgmv",
            "max_lora_chunk_size": 64,
            "enable_lora_overlap_loading": True,
            **extra,
        },
    )

    # Build synthetic prompts as token IDs. Use a “safe” token id (1) to avoid special tokens.
    prompt_id = 1
    input_ids_batch = [
        _make_token_ids(prompt_id, args.prompt_tokens) for _ in range(args.num_prompts)
    ]

    adapter_ref: str | None = None
    adapter_name = "_bench_adapter"
    if args.with_lora:
        # Best-effort: create a tiny zero adapter (should still exercise load/unload path).
        modules = backend.module_map(include_projs=["o_proj", "down_proj"], include_experts=[])
        picked: dict[str, Any] | None = None
        for m in modules:
            if not isinstance(m, dict):
                continue
            path = m.get("module_path") or m.get("name")
            shape = m.get("shape")
            if isinstance(path, str) and isinstance(shape, list) and len(shape) == 2:
                if all(isinstance(x, int) for x in shape) and path.endswith(".weight"):
                    picked = m
                    break
        if picked is None:
            raise RuntimeError("Could not pick a module with a 2D shape from module_map.")

        path = str(picked["module_path"])
        out_f, in_f = int(picked["shape"][0]), int(picked["shape"][1])
        base = path[: -len(".weight")]
        r = 1
        tensors = {
            f"{base}.lora_A.weight": torch.zeros((r, in_f), dtype=torch.float16),
            f"{base}.lora_B.weight": torch.zeros((out_f, r), dtype=torch.float16),
        }
        target_modules = ["down_proj"] if "down_proj" in path else ["o_proj"]
        adapter_ref = backend.load_adapter(
            name=adapter_name,
            tensors=tensors,
            config={
                "peft_type": "LORA",
                "r": r,
                "lora_alpha": r,
                "target_modules": target_modules,
            },
        )

    try:
        # 1) Prefill-only residual capture benchmark (hidden states path).
        if args.with_hidden_states:
            t0 = time.perf_counter()
            out = backend.capture_residuals(
                input_ids_batch,
                capture_layers=[0, 1, 2],
                adapter=adapter_ref,
            )
            t1 = time.perf_counter()
            # tokens processed in prefill ~ batch*prompt_len (rough proxy)
            toks = args.num_prompts * args.prompt_tokens
            print(
                f"prefill_residual_capture: {toks/(t1-t0):.2f} tok/s (proxy), "
                f"residuals_shape={tuple(out.residuals.shape)}"
            )

        # 2) Decode benchmark (generate_text). Hidden states are enabled server-side but not requested.
        t0 = time.perf_counter()
        texts = backend.generate_text(
            input_ids_batch,
            max_new_tokens=int(args.max_new_tokens),
            adapter=adapter_ref,
            temperature=0.0,
        )
        t1 = time.perf_counter()
        # Count generated tokens approximately by tokenizing generated suffixes with backend tokenizer is non-trivial;
        # use a conservative approximation: batch * max_new_tokens.
        gen_toks = args.num_prompts * args.max_new_tokens
        print(
            f"decode_generate_text: {gen_toks/(t1-t0):.2f} tok/s (approx), "
            f"batch={args.num_prompts} max_new_tokens={args.max_new_tokens} got_texts={len(texts)}"
        )
    finally:
        if adapter_ref is not None:
            backend.unload_adapter(name=adapter_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

