# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class CheckpointKind(str, Enum):
    PRECOMPRESSED = "precompressed"
    DENSE = "dense"
    INCONSISTENT = "inconsistent"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class InspectionResult:
    kind: CheckpointKind
    index_path: Path | None
    expected_format: str | None
    # counts of important suffix families (for logs/errors)
    n_total_keys: int
    n_dense_weight: int
    n_expected_artifacts: int
    artifacts_suffixes: tuple[str, ...]
    dense_suffix: str = ".weight"


def _norm_format(fmt: str | None) -> str | None:
    if not isinstance(fmt, str):
        return None
    s = fmt.strip().lower()
    return s or None


def expected_artifact_suffixes_for_format(fmt: str | None) -> tuple[str, ...]:
    """
    Return the minimal set of key suffixes that indicates a checkpoint stores compressed artifacts.

    This is intentionally format-aware, so it stays stable and avoids generic suffix grab-bags.
    """
    f = _norm_format(fmt)
    if f in {"pack-quantized"}:
        # Pack-quantized INT4 typically stores packed weights + per-group scales + shape.
        return (".weight_packed", ".weight_scale", ".weight_shape")
    if f in {"nvfp4-pack-quantized", "mxfp4-pack-quantized"}:
        # FP4 pack formats include packed weights and a global scale.
        return (".weight_packed", ".weight_global_scale", ".weight_scale")
    if f in {"marlin-24"}:
        return (".weight_packed", ".scale_packed", ".meta")
    if f in {"sparse-bitmask", "sparse-24-bitmask"}:
        return (".compressed", ".bitmask", ".shape")
    if f in {"dense", None}:
        return tuple()
    # Unknown format: fall back to the most universal artifact identifiers.
    return (".weight_packed", ".weight_scale", ".weight_shape")


def resolve_index_path(checkpoint_files: Sequence[str]) -> Path | None:
    """
    Resolve `*.safetensors.index.json` path from a list of resolved shard paths.

    We intentionally do not attempt network IO here; this runs inside `from_pretrained`.
    """
    if not checkpoint_files:
        return None
    # `checkpoint_files` is typically a list of resolved `.safetensors` shard paths.
    first = checkpoint_files[0]
    if not isinstance(first, str):
        return None
    d = os.path.dirname(first)
    if not d:
        return None

    try:
        names = os.listdir(d)
    except Exception:
        return None

    # Prefer canonical name when present.
    if "model.safetensors.index.json" in names:
        return Path(d) / "model.safetensors.index.json"

    # Otherwise take a unique `*.safetensors.index.json`.
    idxs = [n for n in names if n.endswith(".safetensors.index.json")]
    if len(idxs) == 1:
        return Path(d) / idxs[0]
    return None


def load_weight_map_keys(index_path: Path) -> set[str]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    wm = data.get("weight_map")
    if not isinstance(wm, Mapping):
        return set()
    return {k for k in wm.keys() if isinstance(k, str)}


def _count_suffix(keys: Iterable[str], suffix: str) -> int:
    return sum(1 for k in keys if k.endswith(suffix))


def _count_any_suffix(keys: Iterable[str], suffixes: Sequence[str]) -> int:
    suf = tuple(suffixes)
    if not suf:
        return 0
    return sum(1 for k in keys if k.endswith(suf))


def infer_checkpoint_kind(
    *,
    keys: Iterable[str],
    expected_format: str | None,
    expect_precompressed: bool,
) -> InspectionResult:
    key_list = [k for k in keys if isinstance(k, str)]
    n_total = len(key_list)

    artifacts = expected_artifact_suffixes_for_format(expected_format)
    n_artifacts = _count_any_suffix(key_list, artifacts)
    n_dense_weight = _count_suffix(key_list, ".weight")

    if n_total == 0:
        return InspectionResult(
            kind=CheckpointKind.UNKNOWN,
            index_path=None,
            expected_format=expected_format,
            n_total_keys=0,
            n_dense_weight=0,
            n_expected_artifacts=0,
            artifacts_suffixes=artifacts,
        )

    # If we see expected artifacts, it is precompressed regardless of dense leftovers.
    if n_artifacts > 0:
        return InspectionResult(
            kind=CheckpointKind.PRECOMPRESSED,
            index_path=None,
            expected_format=expected_format,
            n_total_keys=n_total,
            n_dense_weight=n_dense_weight,
            n_expected_artifacts=n_artifacts,
            artifacts_suffixes=artifacts,
        )

    # If we expected precompressed but see none, that is inconsistent.
    if expect_precompressed:
        return InspectionResult(
            kind=CheckpointKind.INCONSISTENT,
            index_path=None,
            expected_format=expected_format,
            n_total_keys=n_total,
            n_dense_weight=n_dense_weight,
            n_expected_artifacts=0,
            artifacts_suffixes=artifacts,
        )

    # Otherwise, if we see dense weights and no artifacts, treat as dense.
    if n_dense_weight > 0:
        return InspectionResult(
            kind=CheckpointKind.DENSE,
            index_path=None,
            expected_format=expected_format,
            n_total_keys=n_total,
            n_dense_weight=n_dense_weight,
            n_expected_artifacts=0,
            artifacts_suffixes=artifacts,
        )

    return InspectionResult(
        kind=CheckpointKind.UNKNOWN,
        index_path=None,
        expected_format=expected_format,
        n_total_keys=n_total,
        n_dense_weight=n_dense_weight,
        n_expected_artifacts=0,
        artifacts_suffixes=artifacts,
    )


def inspect_checkpoint(
    *,
    checkpoint_files: Sequence[str],
    expected_format: str | None,
    expect_precompressed: bool,
) -> InspectionResult:
    idx = resolve_index_path(checkpoint_files)
    if idx is None:
        # No safe fallback; treat as unknown and let caller raise.
        return InspectionResult(
            kind=CheckpointKind.UNKNOWN,
            index_path=None,
            expected_format=expected_format,
            n_total_keys=0,
            n_dense_weight=0,
            n_expected_artifacts=0,
            artifacts_suffixes=expected_artifact_suffixes_for_format(expected_format),
        )

    keys = load_weight_map_keys(idx)
    res = infer_checkpoint_kind(
        keys=keys,
        expected_format=expected_format,
        expect_precompressed=expect_precompressed,
    )
    # Re-wrap with index path for logs/errors.
    return InspectionResult(
        kind=res.kind,
        index_path=idx,
        expected_format=res.expected_format,
        n_total_keys=res.n_total_keys,
        n_dense_weight=res.n_dense_weight,
        n_expected_artifacts=res.n_expected_artifacts,
        artifacts_suffixes=res.artifacts_suffixes,
        dense_suffix=res.dense_suffix,
    )


def should_scope_to_language_model(
    *,
    index_keys: Iterable[str],
    artifact_suffixes: Sequence[str],
    threshold: float = 0.95,
) -> bool:
    """
    Decide if it is safe to apply compressed-tensors initialization to `model.language_model` subtree.

    We base this on index keys (facts): if the overwhelming majority of artifact keys begin with
    `language_model.`, then the checkpoint is essentially language-only and scoping is safe.
    """
    suf = tuple(artifact_suffixes)
    if not suf:
        return False
    total = 0
    lm = 0
    for k in index_keys:
        if not isinstance(k, str) or not k.endswith(suf):
            continue
        total += 1
        if k.startswith("language_model."):
            lm += 1
    if total == 0:
        return False
    return (lm / total) >= float(threshold)

