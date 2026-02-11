from __future__ import annotations

from typing import Any


def hf_greedy_sampling_params(*, max_new_tokens: int) -> dict[str, Any]:
    """Sampling params that emulate HF `generate(do_sample=False)` (argmax).

    SGLang defaults can vary with server args / preferred params; setting these explicitly
    keeps refusal counting and cached continuations closer to HF-local semantics.
    """

    return {
        "max_new_tokens": int(max_new_tokens),
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

