from __future__ import annotations

import os
from dataclasses import dataclass

import huggingface_hub


@dataclass(frozen=True)
class ResolvedModelDir:
    """Resolved local directory for an HF model repo or a local path."""

    input: str
    resolved_dir: str
    source: str  # "path" | "hf"
    revision: str | None = None


def resolve_model_dir(
    model: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> ResolvedModelDir:
    """Resolve `model` to a local on-disk directory.

    - If `model` is an existing local path, returns it unchanged.
    - Otherwise treats `model` as a Hugging Face repo id and resolves/downloads it via HF cache.

    This is intentionally thin glue around `huggingface_hub.snapshot_download`, so we avoid
    inventing a parallel download/cache mechanism.
    """
    if os.path.exists(model):
        return ResolvedModelDir(
            input=model,
            resolved_dir=model,
            source="path",
            revision=revision,
        )

    resolved = huggingface_hub.snapshot_download(
        repo_id=model,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        # Important: download the full snapshot so KTransformers can find any extra files.
        allow_patterns=None,
        ignore_patterns=None,
    )
    return ResolvedModelDir(
        input=model,
        resolved_dir=resolved,
        source="hf",
        revision=revision,
    )

