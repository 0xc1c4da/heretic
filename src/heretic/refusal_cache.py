from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor

from .backend.base import BackendMetadata
from .config import Settings
from .utils import Prompt, sha256_token_ids

_FORMAT_VERSION = 1


def _to_jsonable(x: Any) -> Any:
    # Best-effort conversion to JSON-serializable objects for identity hashing.
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, set):
        return sorted([_to_jsonable(v) for v in x])
    if isinstance(x, dict):
        # Stable order comes from json.dumps(sort_keys=True).
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    # Fallback: preserve information but avoid non-serializable structures.
    return repr(x)


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hash_token_ids_batch(input_ids_batch: list[list[int]]) -> dict[str, Any]:
    """Hash a batch of token id sequences without storing the full IDs.

    We hash each prompt sequence using `sha256_token_ids()` (little-endian uint32),
    then combine those digests in-order (plus per-sequence length) to produce a
    batch-level digest.
    """
    h = hashlib.sha256()
    total_tokens = 0
    for ids in input_ids_batch:
        total_tokens += int(len(ids))
        d = sha256_token_ids(ids)
        # Mix in length to make the composition robust to any hypothetical digest concatenation ambiguity.
        h.update(struct.pack("<I", int(len(ids))))
        h.update(bytes.fromhex(d))
    return {
        "count": int(len(input_ids_batch)),
        "total_tokens": int(total_tokens),
        "sha256": h.hexdigest(),
    }


def get_cache_dir(settings: Settings) -> str:
    if settings.refusal_cache_dir:
        return str(settings.refusal_cache_dir)
    return os.path.join(str(settings.study_checkpoint_dir), "refusal_cache")


def _backend_metadata_dict(meta: BackendMetadata) -> dict[str, Any]:
    # Drop the large/unstable supports dict? Keep it: it is small and can change behavior.
    return _to_jsonable(asdict(meta))


def _sglang_offline_effective_args(settings: Settings, model: Any) -> dict[str, Any] | None:
    """Capture relevant offline engine args for identity.

    We prefer pulling values from the live Engine's `server_args` when available,
    because it includes any defaults injected by Heretic and SGLang.
    """
    try:
        backend_setting = getattr(settings, "backend", None)
        # BackendType is an Enum; be robust to string-y configs/tests.
        if str(backend_setting) not in ("BackendType.SGLANG_OFFLINE", "sglang_offline"):
            return None
        backend = getattr(model, "backend", None)
        eng = getattr(backend, "_engine", None)
        server_args = getattr(eng, "server_args", None)
        if server_args is None:
            return None

        # Keys: everything the user specified, plus the defaults Heretic injects/relies on.
        user_args = dict(getattr(settings, "sglang_offline_args", None) or {})
        keys = set(str(k) for k in user_args.keys())
        keys |= {
            "model_path",
            "tokenizer_path",
            "kt_weight_path",
            "trust_remote_code",
            "enable_return_hidden_states",
            "enable_lora",
        }

        out: dict[str, Any] = {}
        for k in sorted(keys):
            if hasattr(server_args, k):
                out[k] = _to_jsonable(getattr(server_args, k))
            elif k in user_args:
                out[k] = _to_jsonable(user_args[k])
        return out
    except Exception:
        return None


def compute_identity(
    settings: Settings,
    model: Any,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
) -> dict[str, Any]:
    meta = model.backend.get_metadata()
    num_layers = meta.num_layers
    capture_layers = list(range(int(num_layers))) if isinstance(num_layers, int) and num_layers > 0 else None

    good_ids = model.encode_prompts(good_prompts)
    bad_ids = model.encode_prompts(bad_prompts)

    identity: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "backend_metadata": _backend_metadata_dict(meta),
        "settings": {
            "model": str(settings.model),
            "hf_revision": getattr(settings, "hf_revision", None),
            "trust_remote_code": bool(getattr(settings, "trust_remote_code", False)),
            "winsorization_quantile": float(settings.winsorization_quantile),
            "orthogonalize_direction": bool(settings.orthogonalize_direction),
            "detect_response_prefix": bool(settings.detect_response_prefix),
        },
        "capture": {
            "capture_point": "block_input_last_token",
            "capture_layers": capture_layers,
        },
        "prompts": {
            "good": _hash_token_ids_batch(good_ids),
            "bad": _hash_token_ids_batch(bad_ids),
        },
    }

    # Include the actual response prefix string (when present) for debuggability; token IDs already encode it.
    rp = getattr(model, "response_prefix", None)
    if isinstance(rp, str):
        identity["settings"]["response_prefix"] = rp

    # Backend-specific knobs.
    offline_args = _sglang_offline_effective_args(settings, model)
    if offline_args is not None:
        identity["sglang_offline_args_effective"] = offline_args
    else:
        # Still include user-specified args (if any) for non-offline backends, as a no-op field.
        if getattr(settings, "sglang_offline_args", None):
            identity["sglang_offline_args_effective"] = _to_jsonable(settings.sglang_offline_args)

    return identity


def identity_hash(identity: dict[str, Any]) -> str:
    return _sha256_hex(_stable_json_dumps(identity))


def _cache_paths(cache_dir: str, identity: dict[str, Any]) -> tuple[str, str]:
    h = identity_hash(identity)
    pt_path = os.path.join(cache_dir, f"{h}.pt")
    json_path = os.path.join(cache_dir, f"{h}.json")
    return pt_path, json_path


def try_load_refusal_directions(
    cache_dir: str,
    identity: dict[str, Any],
) -> Tensor | None:
    t, _reason = probe_refusal_cache(cache_dir, identity)
    return t


def probe_refusal_cache(
    cache_dir: str,
    identity: dict[str, Any],
) -> tuple[Tensor | None, str]:
    """Load refusal directions and also return a human-readable status string."""
    pt_path, _ = _cache_paths(cache_dir, identity)
    if not os.path.exists(pt_path):
        return None, "not_found"

    try:
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return None, f"load_error:{type(e).__name__}"

    if not isinstance(obj, dict):
        return None, "invalid:payload_not_dict"
    if obj.get("format_version") != _FORMAT_VERSION:
        return None, "invalid:format_version"
    if obj.get("identity_hash") != identity_hash(identity):
        return None, "invalid:identity_hash_mismatch"

    t = obj.get("refusal_directions")
    if not isinstance(t, torch.Tensor):
        return None, "invalid:missing_tensor"
    if t.ndim != 2:
        return None, "invalid:ndim"

    # Shape validation when possible (use identity's metadata).
    meta = identity.get("backend_metadata") or {}
    num_layers = meta.get("num_layers")
    hidden_size = meta.get("hidden_size")
    if isinstance(num_layers, int) and num_layers > 0 and int(t.shape[0]) != int(num_layers):
        return None, "invalid:layers_shape"
    if isinstance(hidden_size, int) and hidden_size > 0 and int(t.shape[1]) != int(hidden_size):
        return None, "invalid:hidden_shape"

    return t.to(torch.float32), "hit"


def save_refusal_directions(
    cache_dir: str,
    identity: dict[str, Any],
    refusal_directions: Tensor,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    pt_path, json_path = _cache_paths(cache_dir, identity)

    payload = {
        "format_version": _FORMAT_VERSION,
        "identity_hash": identity_hash(identity),
        "identity": identity,
        "refusal_directions": refusal_directions.detach().to(torch.float32).cpu(),
    }

    # Atomic write (temp file in same directory, then replace).
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_refusal_", suffix=".pt", dir=cache_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
        os.replace(tmp_path, pt_path)
    finally:
        # Best-effort cleanup if something went wrong before replace.
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

    # Optional sidecar for inspection/debugging (best-effort).
    try:
        fd2, tmp2 = tempfile.mkstemp(prefix=".tmp_refusal_", suffix=".json", dir=cache_dir)
        try:
            with os.fdopen(fd2, "w", encoding="utf-8") as f2:
                f2.write(_stable_json_dumps(identity))
            os.replace(tmp2, json_path)
        finally:
            try:
                if os.path.exists(tmp2):
                    os.unlink(tmp2)
            except Exception:
                pass
    except Exception:
        pass

