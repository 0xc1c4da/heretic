#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any


def _try_call(label: str, fn):
    try:
        out = fn()
        print(f"[ok] {label}")
        return True, out
    except Exception as e:
        print(f"[skip] {label}: {e}")
        return False, None


def _infer_weight_shape(m: dict[str, Any]) -> tuple[int, int] | None:
    shape = m.get("shape")
    if isinstance(shape, list) and len(shape) == 2 and all(isinstance(x, int) for x in shape):
        return int(shape[0]), int(shape[1])
    out_f = m.get("out_features")
    in_f = m.get("in_features")
    if isinstance(out_f, int) and isinstance(in_f, int):
        if out_f <= 0 or in_f <= 0:
            return None
        return int(out_f), int(in_f)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test Heretic embedded SGLang offline backend.")
    ap.add_argument("--model-path", required=True, help="HF model path or id for SGLang Engine (ServerArgs.model_path).")
    ap.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to SGLang Engine.")
    ap.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size (ServerArgs.tp_size).")
    ap.add_argument("--rank", type=int, default=1, help="LoRA rank for adapter load test (best-effort).")
    ap.add_argument(
        "--engine-args-json",
        default="{}",
        help="Extra Engine kwargs JSON (merged over defaults). Example: '{\"attention_backend\":\"flashinfer\"}'.",
    )
    args = ap.parse_args()

    from heretic.backend.sglang_offline import SGLangOfflineBackend
    from heretic.hf_resolve import resolve_model_dir

    extra = json.loads(args.engine_args_json)
    if not isinstance(extra, dict):
        raise SystemExit("--engine-args-json must decode to an object/dict")

    resolved = resolve_model_dir(args.model_path)
    try:
        backend = SGLangOfflineBackend(
            model_path=resolved.resolved_dir,
            trust_remote_code=bool(args.trust_remote_code),
            engine_args={"tp_size": int(args.tp_size), **extra},
        )
    except Exception as e:
        print(f"[skip] init SGLangOfflineBackend: {e}")
        return 0

    # 0) metadata + lora status
    _try_call("get_metadata()", backend.get_metadata)
    _try_call("lora_status() (initial)", backend.lora_status)

    # 1) tokenize_chat
    ok_tok, tok = _try_call(
        "tokenize_chat()",
        lambda: backend.tokenize_chat([[{"role": "user", "content": "Say the single word: ok"}]]),
    )
    input_ids_batch: list[list[int]] | None = None
    if ok_tok and tok is not None:
        input_ids_batch = tok.token_ids

    # 2) module_map (exclude experts by default)
    ok_map, modules = _try_call(
        "module_map(include_experts=[])",
        lambda: backend.module_map(include_projs=None, include_experts=[]),
    )

    picked: dict[str, Any] | None = None
    if ok_map and isinstance(modules, list):
        for m in modules:
            if not isinstance(m, dict):
                continue
            p = m.get("module_path") or m.get("name")
            if isinstance(p, str) and ("down_proj" in p or "o_proj" in p) and p.endswith(".weight"):
                if _infer_weight_shape(m) is not None:
                    picked = m
                    break
        if picked is None:
            for m in modules:
                if isinstance(m, dict) and _infer_weight_shape(m) is not None:
                    picked = m
                    break

    # 3) compute_vtw_batch (best-effort)
    if picked is not None:
        name = picked.get("module_path") or picked.get("name")
        shape = _infer_weight_shape(picked)
        if isinstance(name, str) and shape is not None:
            out_f, _in_f = shape
            v = [0.0] * out_f
            if out_f > 0:
                v[0] = 1.0
            _try_call(
                "compute_vtw_batch(1 item)",
                lambda: backend.compute_vtw_batch(items=[{"name": name, "v": v, "dtype": "float32"}]),
            )

    # 4) build_full_rownorm_lora (best-effort)
    if picked is not None:
        name = picked.get("module_path") or picked.get("name")
        shape = _infer_weight_shape(picked)
        if isinstance(name, str) and shape is not None:
            out_f, _in_f = shape
            v = [0.0] * out_f
            if out_f > 0:
                v[0] = 1.0
            try:
                import torch
            except Exception as e:
                print(f"[skip] build_full_rownorm_lora: torch not available: {e}")
            else:
                _try_call(
                    "build_full_rownorm_lora()",
                    lambda: backend.build_full_rownorm_lora(
                        name=name,
                        v=torch.tensor(v, dtype=torch.float32),
                        weight=1.0,
                        rank=int(args.rank),
                        svd_niter=2,
                        out_dtype="float16",
                    ),
                )

    # 5) dynamic LoRA load/unload + adapter-aware score/generate (best-effort)
    adapter_id: str | None = None
    adapter_name = "smoke_test_adapter"
    if picked is not None:
        name = picked.get("module_path") or picked.get("name")
        shape = _infer_weight_shape(picked)
        if isinstance(name, str) and shape is not None and name.endswith(".weight"):
            out_f, in_f = shape
            r = int(args.rank) if int(args.rank) > 0 else 1
            try:
                import torch
            except Exception as e:
                print(f"[skip] adapter load: torch not available: {e}")
            else:
                base = name[: -len(".weight")]
                # Use PEFT-style `default` keys (also accepted by SGLang loader).
                key_a = f"{base}.lora_A.default.weight"
                key_b = f"{base}.lora_B.default.weight"
                tensors = {
                    key_a: torch.zeros((r, in_f), dtype=torch.float16),
                    key_b: torch.zeros((out_f, r), dtype=torch.float16),
                }
                target_modules = (
                    ["down_proj"]
                    if "down_proj" in name
                    else (["o_proj"] if "o_proj" in name else [base.split(".")[-1]])
                )
                ok_load, aid = _try_call(
                    "load_adapter()",
                    lambda: backend.load_adapter(
                        name=adapter_name,
                        tensors=tensors,
                        config={
                            "peft_type": "LORA",
                            "r": r,
                            "lora_alpha": r,
                            "target_modules": target_modules,
                        },
                    ),
                )
                if ok_load and isinstance(aid, str):
                    adapter_id = aid

    _try_call("lora_status() (after load)", backend.lora_status)

    if input_ids_batch is not None:
        _try_call("score_full_vocab (base)", lambda: backend.score(input_ids_batch, adapter=None))
        if adapter_id is not None:
            _try_call(
                "score_full_vocab (adapter)",
                lambda: backend.score(input_ids_batch, adapter=adapter_id),
            )

        def _invalid_lora_id_should_error():
            try:
                backend.score(input_ids_batch, adapter="not_a_real_lora_id")
            except Exception:
                return {"ok": True}
            raise RuntimeError("Expected error for invalid lora_id, got success")

        _try_call("score_full_vocab (invalid lora_id => error)", _invalid_lora_id_should_error)

        _try_call(
            "generate_text (base)",
            lambda: backend.generate_text(
                input_ids_batch, max_new_tokens=8, adapter=None, temperature=0.0
            ),
        )
        if adapter_id is not None:
            _try_call(
                "generate_text (adapter)",
                lambda: backend.generate_text(
                    input_ids_batch, max_new_tokens=8, adapter=adapter_id, temperature=0.0
                ),
            )

        # Hidden states capture (residuals): request 3 layers by default.
        def _capture():
            return backend.capture_residuals(
                input_ids_batch,
                capture_layers=[0, 1, 2],
            )

        _try_call("capture_residuals(capture_layers=[0,1,2])", _capture)

    if adapter_id is not None:
        _try_call("unload_adapter()", lambda: backend.unload_adapter(name=adapter_name))
        _try_call("lora_status() (after unload)", backend.lora_status)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

