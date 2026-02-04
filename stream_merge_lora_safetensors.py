#!/usr/bin/env python3
"""
Streaming / shard-wise merger for a vanilla PEFT LoRA adapter into a base HF safetensors checkpoint.

Design goals:
- Do NOT instantiate a full Transformers model (avoids Accelerate hooks, device_map surprises).
- Bound peak CPU RAM to ~one base shard worth of tensors (plus small LoRA A/B and a working chunk).
- Write an HF-compatible merged checkpoint: same shard layout + `model.safetensors.index.json`.

Scope (v1):
- Vanilla PEFT LoRA only (no DoRA/aLoRA/Arrow/etc.)
- Base checkpoint must be safetensors (sharded or single-file).
"""

from __future__ import annotations

import argparse
import errno
import gc
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from safetensors.torch import save_file
from safetensors.torch import safe_open


BASE_MODEL_PREFIXES = (
    "base_model.model.",
    "base_model.",  # conservative fallback
)


def _die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    raise SystemExit(msg)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _strip_known_prefixes(k: str) -> str:
    # PEFT keys commonly start with one or more wrapper prefixes like:
    #   base_model.model.<base_key>
    # Some stacks end up with repeated "base_model.model." segments, so strip repeatedly.
    changed = True
    while changed:
        changed = False
        for p in BASE_MODEL_PREFIXES:
            if k.startswith(p):
                k = k[len(p) :]
                changed = True
    return k


def _is_probably_regex(s: str) -> bool:
    # Heuristic: rank/alpha patterns frequently include anchors or regex operators.
    return any(ch in s for ch in "^$.*+?[](){}|\\")


@dataclass(frozen=True)
class AdapterSpec:
    r_default: int
    lora_alpha_default: float
    fan_in_fan_out: bool
    use_rslora: bool
    bias: str
    modules_to_save: Optional[list[str]]
    rank_pattern: Optional[dict[str, int]]
    alpha_pattern: Optional[dict[str, float]]


@dataclass(frozen=True)
class LoraForWeight:
    base_weight_key: str
    a_key: str
    b_key: str
    # Optional LoRA-B bias (lora_bias=True); applied to base bias key.
    b_bias_key: Optional[str]
    module_key: str  # base module name without ".weight"


@dataclass(frozen=True)
class BaseLayout:
    base_dir: Path
    shard_files: list[Path]  # absolute paths to base shard files
    weight_map: dict[str, str]  # param -> shard filename (relative)
    index_json: Optional[dict[str, Any]]  # original index json, if any


def _parse_adapter_spec(adapter_dir: Path) -> AdapterSpec:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        _die(f"Missing adapter config: {cfg_path}")
    cfg = _load_json(cfg_path)

    peft_type = str(cfg.get("peft_type", "")).upper()
    if peft_type not in {"LORA", "PeftType.LORA".upper()}:
        _die(f"Unsupported adapter peft_type={cfg.get('peft_type')!r}. Only vanilla LoRA is supported in v1.")

    # Reject known variant knobs early.
    for k in ("use_dora", "use_alora", "arrow_config", "alora_invocation_tokens"):
        if cfg.get(k):
            _die(f"Unsupported LoRA variant setting {k}={cfg.get(k)!r}. v1 supports vanilla LoRA only.")

    bias = str(cfg.get("bias", "none"))
    if bias not in {"none", "all", "lora_only"}:
        _die(f"Unsupported LoRA bias mode: {bias!r}")

    r_default = int(cfg.get("r", 0) or 0)
    lora_alpha_default = float(cfg.get("lora_alpha", 0) or 0)
    if r_default <= 0:
        _die(f"Invalid adapter rank r={r_default}.")
    if lora_alpha_default <= 0:
        # Alpha can be 0 in some edge cases but would make a no-op adapter.
        _die(f"Invalid adapter lora_alpha={lora_alpha_default}.")

    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    use_rslora = bool(cfg.get("use_rslora", False))
    modules_to_save = cfg.get("modules_to_save", None)
    if modules_to_save is not None and not isinstance(modules_to_save, list):
        _die("adapter_config.json: modules_to_save must be a list or null.")

    rank_pattern = cfg.get("rank_pattern", None)
    alpha_pattern = cfg.get("alpha_pattern", None)
    if rank_pattern is not None and not isinstance(rank_pattern, dict):
        _die("adapter_config.json: rank_pattern must be a dict or null.")
    if alpha_pattern is not None and not isinstance(alpha_pattern, dict):
        _die("adapter_config.json: alpha_pattern must be a dict or null.")

    # Note: rank_pattern affects adapter shapes at train time; at merge time we trust A/B shapes.
    return AdapterSpec(
        r_default=r_default,
        lora_alpha_default=lora_alpha_default,
        fan_in_fan_out=fan_in_fan_out,
        use_rslora=use_rslora,
        bias=bias,
        modules_to_save=modules_to_save,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )


def _adapter_weights_path(adapter_dir: Path) -> Path:
    st = adapter_dir / "adapter_model.safetensors"
    if st.exists():
        return st
    binp = adapter_dir / "adapter_model.bin"
    if binp.exists():
        _die("adapter_model.bin is not supported in v1; re-save adapter with safe_serialization=True.")
    _die(f"Missing adapter weights: {st}")


def _detect_base_layout(base_dir: Path) -> BaseLayout:
    idx = base_dir / "model.safetensors.index.json"
    if idx.exists():
        j = _load_json(idx)
        weight_map = j.get("weight_map", None)
        if not isinstance(weight_map, dict) or not weight_map:
            _die(f"Invalid index json (missing weight_map): {idx}")
        shard_names = sorted({str(v) for v in weight_map.values()})
        shard_files = [base_dir / n for n in shard_names]
        missing = [p for p in shard_files if not p.exists()]
        if missing:
            _die(f"Base checkpoint is missing shard files: {missing[:3]}{'...' if len(missing) > 3 else ''}")
        return BaseLayout(
            base_dir=base_dir,
            shard_files=shard_files,
            weight_map={str(k): str(v) for k, v in weight_map.items()},
            index_json=j,
        )

    # Single-file layout: choose a single *.safetensors in base_dir
    st_files = sorted(base_dir.glob("*.safetensors"))
    if not st_files:
        _die(f"No safetensors found in base dir: {base_dir}")
    if len(st_files) > 1:
        _die(
            "Base dir has multiple .safetensors files but no model.safetensors.index.json. "
            "Provide a proper HF index or consolidate."
        )
    shard = st_files[0]
    # We will synthesize a weight_map from the keys in this single file.
    with safe_open(shard.as_posix(), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    weight_map = {k: shard.name for k in keys}
    return BaseLayout(
        base_dir=base_dir,
        shard_files=[shard],
        weight_map=weight_map,
        index_json=None,
    )


def _compile_patterns(d: Optional[dict[str, Any]]) -> list[tuple[re.Pattern[str], Any]]:
    if not d:
        return []
    compiled: list[tuple[re.Pattern[str], Any]] = []
    for pat, val in d.items():
        try:
            compiled.append((re.compile(pat), val))
        except re.error:
            # Some users provide literal keys that are not valid regex; treat as suffix match.
            esc = re.escape(pat)
            compiled.append((re.compile(esc), val))
    return compiled


def _resolve_alpha_for_module(
    *,
    adapter_spec: AdapterSpec,
    module_key: str,
    alpha_patterns: Sequence[tuple[re.Pattern[str], float]],
) -> float:
    # Match against both stripped and unstripped variants (some configs include base_model.model prefixes).
    candidates = (module_key, "base_model.model." + module_key)
    for c in candidates:
        for pat, alpha in alpha_patterns:
            if pat.search(c):
                return float(alpha)
    return float(adapter_spec.lora_alpha_default)


def _scaling(alpha: float, r: int, use_rslora: bool) -> float:
    if use_rslora:
        return alpha / math.sqrt(r)
    return alpha / r


def _build_lora_maps(
    *,
    adapter_dir: Path,
    adapter_spec: AdapterSpec,
    adapter_name: str,
) -> tuple[dict[str, LoraForWeight], dict[str, str], dict[str, tuple[str, float]]]:
    """
    Returns:
    - lora_by_base_weight: base_weight_key -> LoraForWeight (A/B keys in adapter ckpt)
    - overrides: base_key -> adapter_key (overwrite base tensor)
    - bias_updates: base_bias_key -> (adapter_b_bias_key, scaling)
    """
    adapter_weights = _adapter_weights_path(adapter_dir)
    alpha_patterns = _compile_patterns(adapter_spec.alpha_pattern)

    # In PEFT adapter checkpoints, adapter_name is typically stripped from keys.
    # Support both formats:
    # - "...lora_A.weight" (most common)
    # - "...lora_A.<adapter_name>.weight" (older / raw state_dict)
    suffixes = {
        "a": (f".lora_A.weight", f".lora_A.{adapter_name}.weight"),
        "b": (f".lora_B.weight", f".lora_B.{adapter_name}.weight"),
        "b_bias": (f".lora_B.bias", f".lora_B.{adapter_name}.bias"),
        "emb_a": (f".lora_embedding_A", f".lora_embedding_A.{adapter_name}"),
        "emb_b": (f".lora_embedding_B", f".lora_embedding_B.{adapter_name}"),
    }

    # Discover keys.
    with safe_open(adapter_weights.as_posix(), framework="pt", device="cpu") as f:
        keys = list(f.keys())

    # Overrides include anything that is not a LoRA component tensor (e.g. modules_to_save, saved embeddings, biases).
    overrides: dict[str, str] = {}

    a_keys: dict[str, str] = {}  # module_key -> adapter_key
    b_keys: dict[str, str] = {}
    b_bias_keys: dict[str, str] = {}
    emb_a_keys: dict[str, str] = {}
    emb_b_keys: dict[str, str] = {}

    def add_first_matching(mapping: MutableMapping[str, str], k: str, kind: str) -> bool:
        for suf in suffixes[kind]:
            if k.endswith(suf):
                module = k[: -len(suf)]
                module = _strip_known_prefixes(module)
                mapping[module] = k
                return True
        return False

    for k in keys:
        if add_first_matching(a_keys, k, "a"):
            continue
        if add_first_matching(b_keys, k, "b"):
            continue
        if add_first_matching(b_bias_keys, k, "b_bias"):
            continue
        if add_first_matching(emb_a_keys, k, "emb_a"):
            continue
        if add_first_matching(emb_b_keys, k, "emb_b"):
            continue

        # Not a LoRA component; treat as a direct override.
        base_k = _strip_known_prefixes(k)
        overrides[base_k] = k

    # Build LoRA targets for linear/conv: module_key + ".weight"
    lora_by_base_weight: dict[str, LoraForWeight] = {}
    bias_updates: dict[str, tuple[str, float]] = {}

    # Linear/conv
    all_modules = sorted(set(a_keys.keys()) | set(b_keys.keys()))
    for module_key in all_modules:
        a_k = a_keys.get(module_key)
        b_k = b_keys.get(module_key)
        if not a_k or not b_k:
            _die(
                "Adapter is missing LoRA components for module "
                f"{module_key!r}: has_A={bool(a_k)} has_B={bool(b_k)}"
            )

        base_weight_key = f"{module_key}.weight"
        lora_by_base_weight[base_weight_key] = LoraForWeight(
            base_weight_key=base_weight_key,
            a_key=a_k,
            b_key=b_k,
            b_bias_key=b_bias_keys.get(module_key),
            module_key=module_key,
        )

        if module_key in b_bias_keys:
            # LoRA bias updates add into the base module bias.
            alpha = _resolve_alpha_for_module(
                adapter_spec=adapter_spec,
                module_key=module_key,
                alpha_patterns=[(p, float(v)) for p, v in alpha_patterns],
            )
            # r from A shape; load just A header tensor to read shape.
            with safe_open(adapter_weights.as_posix(), framework="pt", device="cpu") as f:
                r = int(f.get_tensor(a_k).shape[0])
            scale = _scaling(alpha, r, adapter_spec.use_rslora)
            bias_key = f"{module_key}.bias"
            bias_updates[bias_key] = (b_bias_keys[module_key], scale)

    # Embeddings (optional)
    if emb_a_keys or emb_b_keys:
        # PEFT stores embedding A/B as Parameters. Map module_key -> base weight module_key+".weight".
        all_emb = sorted(set(emb_a_keys.keys()) | set(emb_b_keys.keys()))
        for module_key in all_emb:
            a_k = emb_a_keys.get(module_key)
            b_k = emb_b_keys.get(module_key)
            if not a_k or not b_k:
                _die(
                    "Adapter is missing embedding LoRA components for module "
                    f"{module_key!r}: has_A={bool(a_k)} has_B={bool(b_k)}"
                )
            base_weight_key = f"{module_key}.weight"
            lora_by_base_weight[base_weight_key] = LoraForWeight(
                base_weight_key=base_weight_key,
                a_key=a_k,
                b_key=b_k,
                b_bias_key=None,
                module_key=module_key,
            )

    return lora_by_base_weight, overrides, bias_updates


def _copy_support_files(base_dir: Path, out_dir: Path) -> None:
    # Best-effort copy of common HF repo artifacts needed for loading.
    candidates = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.json",
        "preprocessor_config.json",
        "processor_config.json",
        "spiece.model",
        "modeling_rope_utils.py",  # some repos ship helpers; harmless if missing
    ]
    for name in candidates:
        src = base_dir / name
        if src.exists() and src.is_file():
            shutil.copy2(src, out_dir / name)


def _parse_device(s: str) -> torch.device:
    s = s.strip().lower()
    if s == "cpu":
        return torch.device("cpu")
    if s.startswith("cuda"):
        if not torch.cuda.is_available():
            _die("CUDA requested but torch.cuda.is_available() is false.")
        if s == "cuda":
            return torch.device("cuda:0")
        return torch.device(s)
    _die(f"Unsupported device: {s}")


def _parse_dtype(s: str) -> torch.dtype:
    s = s.strip().lower()
    if s in {"float32", "fp32"}:
        return torch.float32
    if s in {"float16", "fp16"}:
        return torch.float16
    if s in {"bfloat16", "bf16"}:
        return torch.bfloat16
    _die(f"Unsupported dtype: {s}")


def _is_subpath(child: Path, parent: Path) -> bool:
    """Return True if child is inside parent (or equal) after resolving."""
    c = child.resolve()
    p = parent.resolve()
    try:
        common = Path(os.path.commonpath([c.as_posix(), p.as_posix()]))
    except Exception:
        return False
    return common == p


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return
    # Refuse non-empty output dir to avoid mixed linked/rewritten shards on rerun.
    if any(out_dir.iterdir()):
        _die(
            f"Refusing to write into non-empty out_dir={out_dir} without --overwrite. "
            "Delete the directory or pass --overwrite."
        )


def _link_unchanged_shard(
    *,
    src: Path,
    dst: Path,
    mode: str,
    overwrite: bool,
) -> None:
    """
    Link src -> dst without duplicating blocks where possible.

    mode:
      - auto: hardlink then symlink
      - hardlink: hardlink only
      - symlink: symlink only
      - off: do not link (caller should rewrite/copy)
    """
    if mode == "off":
        _die("Internal error: _link_unchanged_shard called with mode=off.")

    if dst.exists() or dst.is_symlink():
        if not overwrite:
            _die(f"Destination already exists: {dst}")
        dst.unlink()

    if mode in {"auto", "hardlink"}:
        try:
            os.link(src.as_posix(), dst.as_posix())
            return
        except OSError as e:
            if mode == "hardlink":
                _die(f"Hardlink failed for {src} -> {dst}: {e}")
            # auto: fall through to symlink on common hardlink failures.
            if e.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK, errno.ENOENT}:
                pass

    if mode in {"auto", "symlink"}:
        rel = os.path.relpath(src.as_posix(), start=dst.parent.as_posix())
        try:
            os.symlink(rel, dst.as_posix())
            return
        except OSError as e:
            _die(f"Symlink failed for {src} -> {dst}: {e}")

    _die(f"Unsupported link mode: {mode}")


def _chunk_rows_for_delta(
    *,
    in_features: int,
    compute_dtype: torch.dtype,
    chunk_mib: float,
) -> int:
    bytes_per = torch.tensor([], dtype=compute_dtype).element_size()
    target_bytes = int(chunk_mib * 1024 * 1024)
    if in_features <= 0:
        return 1
    rows = max(1, target_bytes // (in_features * bytes_per))
    return int(rows)


def _apply_lora_blockwise(
    *,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float,
    fan_in_fan_out: bool,
    device: torch.device,
    compute_dtype: torch.dtype,
    chunk_mib: float,
    verify: bool,
) -> torch.Tensor:
    """
    Returns merged weight tensor on CPU (same dtype as W).
    """
    if W.ndim != 2:
        _die(f"LoRA merge expects 2D weights, got shape={tuple(W.shape)}")
    if A.ndim != 2 or B.ndim != 2:
        _die(f"LoRA A/B must be 2D, got A={tuple(A.shape)} B={tuple(B.shape)}")

    r = int(A.shape[0])
    in_features = int(A.shape[1])
    out_features = int(B.shape[0])
    if int(B.shape[1]) != r:
        _die(f"LoRA shape mismatch: A is (r,in)=({r},{in_features}) but B is {tuple(B.shape)}")

    if not fan_in_fan_out:
        # W is (out,in)
        if tuple(W.shape) != (out_features, in_features):
            _die(
                "Base weight shape does not match LoRA components for fan_in_fan_out=False: "
                f"W={tuple(W.shape)} expected={(out_features, in_features)}"
            )
        row_chunk = _chunk_rows_for_delta(in_features=in_features, compute_dtype=compute_dtype, chunk_mib=chunk_mib)
    else:
        # W is (in,out)
        if tuple(W.shape) != (in_features, out_features):
            _die(
                "Base weight shape does not match LoRA components for fan_in_fan_out=True: "
                f"W={tuple(W.shape)} expected={(in_features, out_features)}"
            )
        row_chunk = _chunk_rows_for_delta(in_features=in_features, compute_dtype=compute_dtype, chunk_mib=chunk_mib)

    W_out = W.clone()

    # CPU matmul in bf16/fp16 can be slow; PEFT casts to fp32 in that case. Mirror that behavior by default.
    cpu_cast_fp32 = device.type == "cpu" and W.dtype in (torch.float16, torch.bfloat16)
    eff_compute_dtype = torch.float32 if cpu_cast_fp32 else compute_dtype

    A_dev = A.to(device=device, dtype=eff_compute_dtype)
    B_dev = B.to(device=device, dtype=eff_compute_dtype)

    # Optional verification: compare against full delta for a small random slice.
    verify_slice: Optional[Tuple[slice, slice]] = None
    if verify:
        if not fan_in_fan_out:
            i0 = random.randrange(0, out_features)
            i1 = min(out_features, i0 + min(8, out_features - i0))
            verify_slice = (slice(i0, i1), slice(0, min(16, in_features)))
        else:
            j0 = random.randrange(0, out_features)
            j1 = min(out_features, j0 + min(8, out_features - j0))
            verify_slice = (slice(0, min(16, in_features)), slice(j0, j1))

    for i in range(0, out_features, row_chunk):
        m = min(row_chunk, out_features - i)
        B_chunk = B_dev[i : i + m, :]  # (m,r)
        delta = B_chunk @ A_dev  # (m,in)
        if fan_in_fan_out:
            delta = delta.T  # (in,m)
            # apply into W_out[:, i:i+m]
            delta_cpu = delta.to(device="cpu", dtype=W_out.dtype)
            W_out[:, i : i + m].add_(delta_cpu, alpha=float(scaling))
        else:
            delta_cpu = delta.to(device="cpu", dtype=W_out.dtype)
            W_out[i : i + m, :].add_(delta_cpu, alpha=float(scaling))

    if verify and verify_slice is not None:
        # Recompute full delta for the slice only (still uses full A, full B) but slices output.
        with torch.no_grad():
            delta_full = (B_dev @ A_dev)
            if fan_in_fan_out:
                delta_full = delta_full.T
            delta_full = delta_full * float(scaling)
            delta_full_cpu = delta_full.to(device="cpu", dtype=W_out.dtype)
        if fan_in_fan_out:
            got = (W_out - W)[:, :]
        else:
            got = (W_out - W)[:, :]
        ss0, ss1 = verify_slice
        err = (got[ss0, ss1] - delta_full_cpu[ss0, ss1]).abs().max().item()
        if not math.isfinite(err) or err > 5e-2:
            # Loose tolerance for bf16; this is a smoke check, not a proof.
            _die(f"Verification failed: max_abs_err={err} on slice {verify_slice}")

    return W_out


def merge_lora_streaming(
    *,
    base_dir: Path,
    adapter_dir: Path,
    out_dir: Path,
    adapter_name: str,
    device: torch.device,
    compute_dtype: torch.dtype,
    chunk_mib: float,
    verify: bool,
    link_unchanged: str = "auto",
    extras_mode: str = "extra_shard",
    overwrite: bool = False,
) -> None:
    adapter_spec = _parse_adapter_spec(adapter_dir)
    base_layout = _detect_base_layout(base_dir)

    lora_by_base_weight, overrides, bias_updates = _build_lora_maps(
        adapter_dir=adapter_dir, adapter_spec=adapter_spec, adapter_name=adapter_name
    )

    # Pre-flight: ensure base contains all targeted weights / overrides.
    base_keys = set(base_layout.weight_map.keys())
    missing_weights = [k for k in lora_by_base_weight.keys() if k not in base_keys]
    if missing_weights:
        ex = missing_weights[:5]
        _die(
            "Base checkpoint is missing LoRA-targeted weights. Example missing keys:\n"
            + "\n".join(f"- {k}" for k in ex)
        )
    extra_override_keys = [k for k in overrides.keys() if k not in base_keys]
    missing_bias_targets = [k for k in bias_updates.keys() if k not in base_keys]
    if missing_bias_targets:
        ex = missing_bias_targets[:5]
        _die(
            "Adapter requires updating base bias tensors but they were not found in base checkpoint. Example:\n"
            + "\n".join(f"- {k}" for k in ex)
        )

    if link_unchanged != "off":
        if _is_subpath(out_dir, base_dir) or _is_subpath(base_dir, out_dir):
            if out_dir.resolve() == base_dir.resolve():
                _die("out_dir must differ from base_dir when linking unchanged shards (use in-place mode separately).")
            _die(
                "Refusing to link into a directory that is the same as or nested within the base directory. "
                f"base_dir={base_dir} out_dir={out_dir}"
            )

    _prepare_out_dir(out_dir, overwrite=overwrite)

    adapter_weights_path = _adapter_weights_path(adapter_dir)
    # Keep adapter weights open across shards for faster access.
    adapter_f = safe_open(adapter_weights_path.as_posix(), framework="pt", device="cpu")

    try:
        # We preserve base shard layout. If the adapter contains additional tensors not present in base (e.g.
        # modules_to_save), we can write them to a dedicated extras shard and extend the index weight_map accordingly.
        weight_map_out = dict(base_layout.weight_map)
        extras_to_write: dict[str, torch.Tensor] = {}
        extras_shard_name: Optional[str] = None
        if extra_override_keys:
            if extras_mode not in {"extra_shard", "attach_last_shard"}:
                _die(f"Unsupported extras_mode: {extras_mode}")
            if extras_mode == "extra_shard":
                extras_shard_name = "model-extras.safetensors"
                for base_k in extra_override_keys:
                    src_k = overrides[base_k]
                    t = adapter_f.get_tensor(src_k)
                    extras_to_write[base_k] = t.to(device="cpu").contiguous()
                    weight_map_out[base_k] = extras_shard_name
            else:
                last_shard_name = base_layout.shard_files[-1].name
                print(f"[merge] attaching {len(extra_override_keys)} extra adapter tensor(s) to {last_shard_name}")
                for base_k in extra_override_keys:
                    src_k = overrides[base_k]
                    t = adapter_f.get_tensor(src_k)
                    extras_to_write[base_k] = t.to(device="cpu").contiguous()
                    weight_map_out[base_k] = last_shard_name

        if link_unchanged not in {"auto", "hardlink", "symlink", "off"}:
            _die(f"Unsupported link_unchanged mode: {link_unchanged}")

        base_override_keys = set(overrides.keys()) & base_keys
        affected_keys = (set(lora_by_base_weight.keys()) | base_override_keys | set(bias_updates.keys())) & base_keys
        changed_shards = {base_layout.weight_map[k] for k in affected_keys}
        if extras_mode == "attach_last_shard" and extras_to_write:
            changed_shards.add(base_layout.shard_files[-1].name)

        print(
            f"[merge] shards_total={len(base_layout.shard_files)} "
            f"shards_changed={len(changed_shards)} "
            f"shards_linked={len(base_layout.shard_files) - len(changed_shards)} "
            f"link_mode={link_unchanged} extras_mode={extras_mode}"
        )

        for shard_path in base_layout.shard_files:
            shard_name = shard_path.name
            out_shard_path = out_dir / shard_name
            if link_unchanged != "off" and shard_name not in changed_shards:
                _link_unchanged_shard(
                    src=shard_path,
                    dst=out_shard_path,
                    mode=link_unchanged,
                    overwrite=overwrite,
                )
                print(f"[merge] shard {shard_name} -> {out_shard_path} (linked)")
                continue

            print(f"[merge] shard {shard_name} -> {out_shard_path} (rewritten)")

            with safe_open(shard_path.as_posix(), framework="pt", device="cpu") as base_f:
                out_tensors: dict[str, torch.Tensor] = {}

                for k in base_f.keys():
                    W = base_f.get_tensor(k)

                    # Direct override wins.
                    if k in overrides:
                        src_k = overrides[k]
                        t = adapter_f.get_tensor(src_k)
                        out_tensors[k] = t.to(device="cpu", dtype=W.dtype).contiguous()
                        continue

                    # Bias update for lora_bias=True.
                    if k in bias_updates:
                        b_bias_key, scale = bias_updates[k]
                        bias = W.clone()
                        delta_bias = adapter_f.get_tensor(b_bias_key).to(device="cpu", dtype=bias.dtype)
                        bias.add_(delta_bias, alpha=float(scale))
                        out_tensors[k] = bias.contiguous()
                        continue

                    # Weight merge.
                    lora = lora_by_base_weight.get(k)
                    if lora is not None:
                        A = adapter_f.get_tensor(lora.a_key)
                        B = adapter_f.get_tensor(lora.b_key)
                        # Compute scaling per-module (supports alpha_pattern; r from A).
                        alpha_patterns = _compile_patterns(adapter_spec.alpha_pattern)
                        alpha = _resolve_alpha_for_module(
                            adapter_spec=adapter_spec,
                            module_key=lora.module_key,
                            alpha_patterns=[(p, float(v)) for p, v in alpha_patterns],
                        )
                        r = int(A.shape[0])
                        scale = _scaling(alpha, r, adapter_spec.use_rslora)
                        merged = _apply_lora_blockwise(
                            W=W,
                            A=A,
                            B=B,
                            scaling=scale,
                            fan_in_fan_out=adapter_spec.fan_in_fan_out,
                            device=device,
                            compute_dtype=compute_dtype,
                            chunk_mib=chunk_mib,
                            verify=verify,
                        )
                        out_tensors[k] = merged.contiguous()
                        continue

                    # Default: clone so we don’t keep references to mmap-backed buffers past file close.
                    out_tensors[k] = W.clone().contiguous()

                # If using attach_last_shard mode, inject any extra tensors into the final shard.
                if extras_mode == "attach_last_shard" and extras_to_write and shard_name == base_layout.shard_files[-1].name:
                    for k_extra, t_extra in extras_to_write.items():
                        if k_extra in out_tensors:
                            # Should not happen (extras are defined as missing from base), but avoid silent overwrite.
                            _die(f"Internal error: extra tensor key already exists in base shard: {k_extra}")
                        out_tensors[k_extra] = t_extra

                # Write shard.
                save_file(out_tensors, out_shard_path.as_posix(), metadata={"format": "pt"})

            # Free per-shard memory.
            del out_tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # If using extra_shard mode, write it now (small) and avoid rewriting any base shard purely for extras.
        if extras_mode == "extra_shard" and extras_to_write:
            assert extras_shard_name is not None
            extras_path = out_dir / extras_shard_name
            print(f"[merge] writing extras shard -> {extras_path}")
            if (extras_path.exists() or extras_path.is_symlink()) and not overwrite:
                _die(f"Extras shard already exists: {extras_path}")
            if extras_path.exists() or extras_path.is_symlink():
                extras_path.unlink()
            save_file(extras_to_write, extras_path.as_posix(), metadata={"format": "pt"})

        # Index json
        if base_layout.index_json is not None:
            out_index = dict(base_layout.index_json)
            out_index["weight_map"] = dict(weight_map_out)
            _dump_json(out_dir / "model.safetensors.index.json", out_index)
        else:
            # For single-file case, create a minimal index (still acceptable to HF tooling).
            out_index = {
                "metadata": {"format": "pt"},
                "weight_map": dict(weight_map_out),
            }
            _dump_json(out_dir / "model.safetensors.index.json", out_index)

        _copy_support_files(base_dir, out_dir)

    finally:
        try:
            adapter_f.close()
        except Exception:
            pass


def _self_test() -> None:
    """
    Offline self-test:
    - Create a tiny GPT2 model, save base safetensors
    - Create a LoRA adapter via PEFT, save adapter safetensors
    - Merge via this script
    - Compare merged weights to PEFT merge_and_unload output
    """
    import tempfile

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    random.seed(0)

    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        n_positions=64,
        n_ctx=64,
        vocab_size=128,
    )
    base = GPT2LMHeadModel(cfg)
    base.eval()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base_dir = td / "base"
        adapter_dir = td / "adapter"
        out_dir = td / "merged"
        out_dir_linked = td / "merged_linked"
        base_dir.mkdir()
        adapter_dir.mkdir()

        # Force *many* shards so the linking path is guaranteed to be exercised.
        base.save_pretrained(base_dir.as_posix(), safe_serialization=True, max_shard_size="10KB")

        lcfg = LoraConfig(
            r=4,
            lora_alpha=4,
            # Keep targets narrow so not all shards are affected.
            target_modules=["c_attn"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(GPT2LMHeadModel(cfg), lcfg)
        # Randomize adapter weights to make merge non-trivial.
        for n, p in peft_model.named_parameters():
            if "lora_" in n:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
        peft_model.save_pretrained(adapter_dir.as_posix(), safe_serialization=True)

        # Add an extra tensor to exercise extras_shard handling.
        from safetensors.torch import load_file as safe_load_file

        adapter_path = adapter_dir / "adapter_model.safetensors"
        st = dict(safe_load_file(adapter_path.as_posix()))
        st["extra.test_tensor"] = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        save_file(st, adapter_path.as_posix(), metadata={"format": "pt"})

        # Expected: PEFT merge (canonical path)
        exp_base = GPT2LMHeadModel.from_pretrained(base_dir.as_posix())
        exp = PeftModel.from_pretrained(exp_base, adapter_dir.as_posix(), adapter_name="default")
        exp_merged = exp.merge_and_unload(safe_merge=False)
        exp_sd = exp_merged.state_dict()

        # Our merge
        merge_lora_streaming(
            base_dir=base_dir,
            adapter_dir=adapter_dir,
            out_dir=out_dir,
            adapter_name="default",
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            chunk_mib=8.0,
            verify=True,
            link_unchanged="off",
            extras_mode="extra_shard",
            overwrite=False,
        )
        got = GPT2LMHeadModel.from_pretrained(out_dir.as_posix())
        got_sd = got.state_dict()

        # Compare a few keys
        keys = sorted(k for k in exp_sd.keys() if k in got_sd and exp_sd[k].dtype.is_floating_point)
        sample = random.sample(keys, k=min(10, len(keys)))
        for k in sample:
            a = exp_sd[k].cpu()
            b = got_sd[k].cpu()
            err = (a - b).abs().max().item()
            if err > 5e-2:
                _die(f"self-test mismatch key={k} max_abs_err={err}")

        # Confirm extras shard exists and index references it.
        idx = _load_json(out_dir / "model.safetensors.index.json")
        if idx["weight_map"].get("extra.test_tensor") != "model-extras.safetensors":
            _die("self-test expected extra.test_tensor to be mapped to model-extras.safetensors")
        if not (out_dir / "model-extras.safetensors").exists():
            _die("self-test expected model-extras.safetensors to exist")

        # Second run: linking enabled, ensure at least one shard is hardlinked.
        merge_lora_streaming(
            base_dir=base_dir,
            adapter_dir=adapter_dir,
            out_dir=out_dir_linked,
            adapter_name="default",
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            chunk_mib=8.0,
            verify=False,
            link_unchanged="auto",
            extras_mode="extra_shard",
            overwrite=False,
        )
        linked = 0
        for p in out_dir_linked.glob("*.safetensors"):
            if p.name == "model-extras.safetensors":
                continue
            st_out = os.stat(p.as_posix())
            if st_out.st_nlink >= 2:
                linked += 1
        if linked == 0:
            _die("self-test expected at least one hardlinked shard in linked merge output")

        print("[self-test] ok")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, required=False, help="Base model directory (safetensors shards)")
    p.add_argument("--adapter-dir", type=str, required=False, help="PEFT adapter directory (adapter_config.json + safetensors)")
    p.add_argument("--out-dir", type=str, required=False, help="Output directory for merged checkpoint")
    p.add_argument("--adapter-name", type=str, default="default")
    p.add_argument("--device", type=str, default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--compute-dtype", type=str, default="float32", help="float32|bfloat16|float16")
    p.add_argument("--chunk-mib", type=float, default=128.0, help="Delta chunk size in MiB (per update block)")
    p.add_argument("--verify", action="store_true", help="Spot-verify a few merges with full matmul slices")
    p.add_argument(
        "--link-unchanged",
        type=str,
        default="auto",
        help="How to handle unchanged shards: auto|hardlink|symlink|off (default: auto)",
    )
    p.add_argument(
        "--extras-mode",
        type=str,
        default="extra_shard",
        help="Where to write adapter extra tensors: extra_shard|attach_last_shard (default: extra_shard)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory (dangerous).",
    )
    p.add_argument("--self-test", action="store_true", help="Run offline self-test and exit")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.self_test:
        _self_test()
        return

    if not args.base_dir or not args.adapter_dir or not args.out_dir:
        p.print_help()
        _die("Missing required arguments: --base-dir, --adapter-dir, --out-dir")

    merge_lora_streaming(
        base_dir=Path(args.base_dir),
        adapter_dir=Path(args.adapter_dir),
        out_dir=Path(args.out_dir),
        adapter_name=str(args.adapter_name),
        device=_parse_device(args.device),
        compute_dtype=_parse_dtype(args.compute_dtype),
        chunk_mib=float(args.chunk_mib),
        verify=bool(args.verify),
        link_unchanged=str(args.link_unchanged),
        extras_mode=str(args.extras_mode),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()

