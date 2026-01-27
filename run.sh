#!/usr/bin/env bash

if [ -z "$TMUX" ]; then
    echo "Error: not in tmux" >&2
    exit 1
fi

if [ -z "$1" ]; then
    echo "Error: model argument required" >&2
    echo "Usage: $0 <model>" >&2
    exit 1
fi

uv sync
uv run pip install git+https://github.com/huggingface/transformers.git
uv run heretic "$1"