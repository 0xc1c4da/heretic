#!/usr/bin/env python3
from __future__ import annotations

"""
CPU-only smoke tests for Heretic's SGLang hidden-states parser.

This is intentionally lightweight (no pytest dependency) and validates:
- v0_steps concatenation semantics using meta_info.prompt_tokens
- skipping empty steps
- reshaping via meta_info.hidden_states_d_model
"""

import sys

import torch


def main() -> int:
    from heretic.backend.sglang_offline import _parse_hidden_states_last_prompt_token

    capture_layers = [0, 1, 2]
    d_model = 4
    feat = len(capture_layers) * d_model

    # Steps: [empty], [2 tokens], [1 token] => prompt_tokens=3 (last prompt token is step3 token0)
    raw_hs = [
        [],
        [
            [1.0] * feat,  # prompt token 0
            [2.0] * feat,  # prompt token 1
        ],
        [
            [3.0] * feat,  # prompt token 2 (last)
        ],
    ]
    meta = {
        "hidden_states_schema_version": "v0_steps",
        "prompt_tokens": 3,
        "hidden_states_d_model": d_model,
    }

    t = _parse_hidden_states_last_prompt_token(
        raw_hs,
        meta=meta,
        capture_layers=capture_layers,
        batch_index=0,
    )
    assert isinstance(t, torch.Tensor)
    assert tuple(int(x) for x in t.shape) == (len(capture_layers), d_model)
    # All entries should be 3.0 after reshape.
    if not torch.allclose(t, torch.full((len(capture_layers), d_model), 3.0, dtype=torch.float32)):
        raise AssertionError(f"Unexpected parsed tensor values: min={t.min().item()} max={t.max().item()}")

    # Fallback behavior: if prompt_tokens is missing, we should pick last non-empty step's last token.
    meta2 = {"hidden_states_schema_version": "v0_steps", "hidden_states_d_model": d_model}
    t2 = _parse_hidden_states_last_prompt_token(
        raw_hs,
        meta=meta2,
        capture_layers=capture_layers,
        batch_index=0,
    )
    assert torch.allclose(t2, t)

    print("[ok] hidden_states parser smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

