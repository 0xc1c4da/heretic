#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
import urllib.error
import urllib.request
from typing import Any


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float = 60.0) -> Any:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {url}: {raw}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed for {url}: {e}") from e


def _get_json(url: str, *, timeout_s: float = 30.0) -> Any:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {url}: {raw}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed for {url}: {e}") from e


def _serialize_for_sglang(obj: Any) -> str:
    """Serialize tensors safely for SGLang's SafeUnpickler over HTTP.

    Do NOT use `multiprocessing.reduction.ForkingPickler` here: it can encode tensor
    storages via `multiprocessing.resource_sharer` (FD passing), which fails across
    an HTTP boundary (authkey mismatch).
    """
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(payload).decode("utf-8")


def _try_call(label: str, fn):
    try:
        out = fn()
        print(f"[ok] {label}")
        return True, out
    except Exception as e:
        print(f"[skip] {label}: {e}")
        return False, None


def _infer_weight_shape(m: dict[str, Any]) -> tuple[int, int] | None:
    # Best-effort: accept either "shape": [out,in] or explicit keys.
    shape = m.get("shape")
    if isinstance(shape, list) and len(shape) == 2 and all(isinstance(x, int) for x in shape):
        return int(shape[0]), int(shape[1])
    out_f = m.get("out_features")
    in_f = m.get("in_features")
    if isinstance(out_f, int) and isinstance(in_f, int):
        return int(out_f), int(in_f)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test Heretic SGLang endpoints.")
    ap.add_argument("--base-url", required=True, help="SGLang base URL, e.g. http://127.0.0.1:30000")
    ap.add_argument("--admin-url", default=None, help="SGLang admin URL (defaults to base-url)")
    ap.add_argument("--model", default=None, help="Optional served model name")
    ap.add_argument("--rank", type=int, default=1, help="LoRA rank for adapter load test (best-effort)")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    admin_url = (args.admin_url or args.base_url).rstrip("/")

    # 0) Basic server liveness
    _try_call("GET /get_model_info", lambda: _get_json(f"{base_url}/get_model_info"))

    # 1) Tokenize chat (preferred way to obtain token ids)
    ok_tok, tok = _try_call(
        "POST /heretic/tokenize_chat",
        lambda: _post_json(
            f"{base_url}/heretic/tokenize_chat",
            {
                "chats": [
                    [
                        {"role": "user", "content": "Say the single word: ok"},
                    ]
                ],
                "add_generation_prompt": True,
                "response_prefix": None,
            },
        ),
    )

    input_ids_batch: list[list[int]] | None = None
    if ok_tok and isinstance(tok, dict):
        ids = tok.get("token_ids_batch") or tok.get("input_ids_batch")
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            input_ids_batch = ids  # type: ignore[assignment]

    # 2) Module map (used to select a target weight)
    ok_map, modmap = _try_call(
        "POST /heretic/module_map",
        lambda: _post_json(
            f"{base_url}/heretic/module_map",
            {"include_projs": None},
        ),
    )

    modules: list[dict[str, Any]] = []
    if ok_map:
        if isinstance(modmap, dict) and isinstance(modmap.get("modules"), list):
            modules = [m for m in modmap["modules"] if isinstance(m, dict)]
        elif isinstance(modmap, list):
            modules = [m for m in modmap if isinstance(m, dict)]

    picked: dict[str, Any] | None = None
    for m in modules:
        p = m.get("module_path") or m.get("name")
        if isinstance(p, str) and ("down_proj" in p or "o_proj" in p) and p.endswith(".weight"):
            if _infer_weight_shape(m) is not None:
                picked = m
                break
    if picked is None:
        for m in modules:
            if _infer_weight_shape(m) is not None:
                picked = m
                break

    # 3) Batched v^T W (best-effort)
    if picked is not None:
        name = picked.get("module_path") or picked.get("name")
        shape = _infer_weight_shape(picked)
        if isinstance(name, str) and shape is not None:
            out_f, _in_f = shape
            v = [0.0] * out_f
            if out_f > 0:
                v[0] = 1.0

            def _do_vtw_batch():
                return _post_json(
                    f"{admin_url}/compute_vtw_batch",
                    {"items": [{"name": name, "v": v, "dtype": "float32"}]},
                    timeout_s=300.0,
                )

            ok_vtwb, vtwb = _try_call("POST /compute_vtw_batch (1 item)", _do_vtw_batch)
            if not ok_vtwb:
                _try_call(
                    "POST /compute_vtw (fallback)",
                    lambda: _post_json(
                        f"{admin_url}/compute_vtw",
                        {"name": name, "v": v, "dtype": "float32"},
                        timeout_s=300.0,
                    ),
                )

    # 4) FULL rownorm LoRA primitive (best-effort)
    if picked is not None:
        name = picked.get("module_path") or picked.get("name")
        shape = _infer_weight_shape(picked)
        if isinstance(name, str) and shape is not None:
            out_f, _in_f = shape
            v = [0.0] * out_f
            if out_f > 0:
                v[0] = 1.0

            def _do_full():
                out = _post_json(
                    f"{base_url}/heretic/build_full_rownorm_lora",
                    {
                        "name": name,
                        "v": v,
                        "weight": 1.0,
                        "rank": int(args.rank),
                        "svd_q": None,
                        "svd_niter": 2,
                        "out_dtype": "float16",
                    },
                    timeout_s=600.0,
                )
                if not isinstance(out, dict):
                    raise RuntimeError(f"Unexpected response type: {type(out)}")
                for key in ("lora_A_b64", "lora_B_b64", "lora_A_shape", "lora_B_shape", "dtype"):
                    if key not in out:
                        raise RuntimeError(f"Missing key {key} in response: keys={list(out)}")
                if out["dtype"] != "float16":
                    raise RuntimeError(f"Unexpected dtype: {out['dtype']}")
                raw_a = base64.b64decode(out["lora_A_b64"])
                raw_b = base64.b64decode(out["lora_B_b64"])
                a_shape = out["lora_A_shape"]
                b_shape = out["lora_B_shape"]
                if (
                    not isinstance(a_shape, list)
                    or not isinstance(b_shape, list)
                    or len(a_shape) != 2
                    or len(b_shape) != 2
                ):
                    raise RuntimeError(f"Bad shapes: {a_shape=} {b_shape=}")
                a_elems = int(a_shape[0]) * int(a_shape[1])
                b_elems = int(b_shape[0]) * int(b_shape[1])
                if len(raw_a) != a_elems * 2 or len(raw_b) != b_elems * 2:
                    raise RuntimeError(
                        f"Bad buffer sizes: lenA={len(raw_a)} vs {a_elems*2}, lenB={len(raw_b)} vs {b_elems*2}"
                    )
                return out

            _try_call("POST /heretic/build_full_rownorm_lora", _do_full)

    # 5) Dynamic LoRA load/unload + adapter-aware score/generate (best-effort)
    adapter_id: str | None = None
    adapter_name = "smoke_test_adapter"
    if picked is not None:
        name = picked.get("module_path") or picked.get("name")
        shape = _infer_weight_shape(picked)
        if isinstance(name, str) and shape is not None and name.endswith(".weight"):
            out_f, in_f = shape
            r = int(args.rank)
            if r <= 0:
                r = 1

            try:
                import torch
            except Exception as e:
                print(f"[skip] adapter load: torch not available: {e}")
            else:
                base = name[: -len(".weight")]
                # PEFT-like keys; SGLang only requires that names include "layers.<n>.", the target module,
                # and "lora_A"/"lora_B" substrings.
                key_a = f"{base}.lora_A.weight"
                key_b = f"{base}.lora_B.weight"
                tensors = {
                    key_a: torch.zeros((r, in_f), dtype=torch.float16),
                    key_b: torch.zeros((out_f, r), dtype=torch.float16),
                }
                target_modules = ["down_proj"] if "down_proj" in name else (["o_proj"] if "o_proj" in name else [])
                if not target_modules:
                    # Fallback: last module name segment before ".weight"
                    target_modules = [base.split(".")[-1]]

                def _do_load():
                    out = _post_json(
                        f"{admin_url}/load_lora_adapter_from_tensors",
                        {
                            "lora_name": adapter_name,
                            "config_dict": {
                                "r": r,
                                "lora_alpha": r,
                                "target_modules": target_modules,
                            },
                            "serialized_tensors": _serialize_for_sglang(tensors),
                            "pinned": False,
                            "added_tokens_config": None,
                            "lora_id": None,
                        },
                        timeout_s=300.0,
                    )
                    if not isinstance(out, dict) or not out.get("success"):
                        raise RuntimeError(f"Load failed: {out}")
                    loaded = out.get("loaded_adapters") or {}
                    if not isinstance(loaded, dict) or adapter_name not in loaded:
                        raise RuntimeError(f"Missing loaded adapter ref: {loaded}")
                    ref = loaded[adapter_name]
                    if not isinstance(ref, dict) or not isinstance(ref.get("lora_id"), str):
                        raise RuntimeError(f"Bad adapter ref: {ref}")
                    return ref["lora_id"]

                ok_load, aid = _try_call("POST /load_lora_adapter_from_tensors", _do_load)
                if ok_load and isinstance(aid, str):
                    adapter_id = aid

    if input_ids_batch is not None:
        _try_call(
            "POST /heretic/score_full_vocab (base)",
            lambda: _post_json(
                f"{base_url}/heretic/score_full_vocab",
                {"input_ids": input_ids_batch, "lora_id": None},
                timeout_s=300.0,
            ),
        )
        if adapter_id is not None:
            _try_call(
                "POST /heretic/score_full_vocab (adapter)",
                lambda: _post_json(
                    f"{base_url}/heretic/score_full_vocab",
                    {"input_ids": input_ids_batch, "lora_id": adapter_id},
                    timeout_s=300.0,
                ),
            )

        _try_call(
            "POST /generate (base)",
            lambda: _post_json(
                f"{base_url}/generate",
                {
                    "input_ids": input_ids_batch,
                    "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
                    "stream": False,
                    "return_logprob": False,
                    "lora_id": None,
                },
                timeout_s=300.0,
            ),
        )
        if adapter_id is not None:
            _try_call(
                "POST /generate (adapter)",
                lambda: _post_json(
                    f"{base_url}/generate",
                    {
                        "input_ids": input_ids_batch,
                        "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
                        "stream": False,
                        "return_logprob": False,
                        "lora_id": adapter_id,
                    },
                    timeout_s=300.0,
                ),
            )

    if adapter_id is not None:
        _try_call(
            "POST /unload_lora_adapter",
            lambda: _post_json(
                f"{admin_url}/unload_lora_adapter",
                {"lora_name": adapter_name, "lora_id": None},
                timeout_s=60.0,
            ),
        )

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

