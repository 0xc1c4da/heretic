#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Interactive chat REPL for a *local* model directory using Heretic's embedded SGLang backend.

This is intended to test merged checkpoints (e.g. output of tools/merge_lora.py) by loading
the model from disk and running inference via `SGLangOfflineBackend`.

Why this is a separate tool:
- SGLang offline engine often uses multiprocessing ("spawn"), which is brittle under stdin / `python -c`.
- A real .py file with a proper __main__ entrypoint is more reliable.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _truncate_chat_response(text: str, *, max_chars: int) -> str:
    """Defensive cap for interactive chat output (print + stored context)."""
    try:
        max_chars_i = int(max_chars)
    except Exception:
        max_chars_i = 0
    if max_chars_i <= 0:
        return text
    if len(text) <= max_chars_i:
        return text
    return text[:max_chars_i] + "\n[... truncated ...]"


def _parse_engine_args_json(s: str) -> dict[str, Any]:
    if not s:
        return {}
    try:
        data = json.loads(s)
    except Exception as e:
        raise SystemExit(f"--engine-args-json must be valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise SystemExit("--engine-args-json must decode to an object/dict.")
    return dict(data)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Chat with a local model directory via embedded SGLang (offline).",
    )
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Local model directory to load (merged checkpoint output dir).",
    )
    ap.add_argument(
        "--engine-args-json",
        default="{}",
        help=(
            "JSON object of additional SGLang Engine kwargs (ServerArgs fields). "
            "These are merged over tool defaults. Example: '{\"tp_size\":8,\"attention_backend\":\"triton\"}'."
        ),
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to the SGLang Engine.",
    )
    ap.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt for the chat session.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=4000,
        help="Maximum number of tokens to generate per assistant response.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy).",
    )
    ap.add_argument(
        "--max-response-chars",
        type=int,
        default=20000,
        help="Hard cap (in characters) applied to responses before printing/storing.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from heretic.backend.sglang_offline import SGLangOfflineBackend
    from heretic.hf_resolve import resolve_model_dir

    engine_args = _parse_engine_args_json(str(args.engine_args_json))

    # Resolve local directory (also supports HF ids, but this tool is intended for local paths).
    resolved = resolve_model_dir(str(args.model_dir))
    model_path = resolved.resolved_dir

    # Match Heretic's offline initialization defaults: set tokenizer_path unless overridden.
    engine_args = dict(engine_args)
    engine_args.setdefault("tokenizer_path", model_path)

    backend = SGLangOfflineBackend(
        model_path=model_path,
        trust_remote_code=bool(args.trust_remote_code),
        engine_args=engine_args,
    )

    # Print a small bit of startup info (helps confirm we loaded the expected checkpoint).
    try:
        meta = backend.get_metadata()
        print(
            f"[sglang_offline_chat] loaded model_id={meta.model_id!r} "
            f"tokenizer_id={meta.tokenizer_id!r} backend_version={meta.backend_version!r}"
        )
        if meta.num_layers is not None and meta.hidden_size is not None:
            print(
                f"[sglang_offline_chat] num_layers={meta.num_layers} hidden_size={meta.hidden_size}"
            )
    except Exception as e:
        print(f"[sglang_offline_chat] warning: failed to read metadata: {e}")

    print()
    print("Enter a message and press Enter.")
    print("Press Ctrl+C or Ctrl+D to exit.")
    print()

    chat: list[dict[str, Any]] = [{"role": "system", "content": str(args.system)}]

    while True:
        try:
            user = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return 0
        if not user:
            # Match Heretic’s UI behavior: empty input returns to menu; here it exits.
            return 0

        chat.append({"role": "user", "content": user})

        try:
            tok = backend.tokenize_chat([chat], continue_final_message=False)
            texts = backend.generate_text(
                tok.token_ids,
                max_new_tokens=int(args.max_new_tokens),
                adapter=None,
                temperature=float(args.temperature),
            )
            resp = texts[0] if texts else ""
        except KeyboardInterrupt:
            print()
            return 0
        except Exception as e:
            print(f"[error] generation failed: {e}")
            # Keep the conversation history; user can retry.
            continue

        resp = _truncate_chat_response(resp, max_chars=int(args.max_response_chars))
        print()
        print("Assistant:")
        print(resp)
        print()
        chat.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Allow piping output to head/tee without stacktraces.
        raise SystemExit(0)
