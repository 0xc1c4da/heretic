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

# Ensure temp/cache locations are set before Python starts.
if [ -z "$TMPDIR" ]; then
    export TMPDIR="/workspace/tmp"
fi
if [ -z "$HF_HOME" ]; then
    export HF_HOME="/workspace/hf"
fi
if [ -z "$HUGGINGFACE_HUB_CACHE" ]; then
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
fi
if [ -z "$HF_DATASETS_CACHE" ]; then
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
fi

mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"

# uv add tiktoken
# uv add git+https://github.com/huggingface/transformers.git
uv sync
uv run heretic "$1"