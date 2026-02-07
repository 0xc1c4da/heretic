# Runbook: Kimi‑K2.5 on 8×H200 (embedded SGLang Engine + KTransformers)

This runbook is for the **embedded/offline** backend:

- Heretic embeds `sglang.srt.entrypoints.engine.Engine` (no HTTP server).
- SGLang loads model weights and runs inference in-process.
- Heretic orchestrates: prompt building, residual capture (hidden states), LoRA hot-swap, scoring.

## Baseline expectations

### Throughput target
- **Decode**: aim for **>20 tok/s** (single stream) on 8×H200 once you’re on the fused-kernel fast path.

### Highest performance levers (in order)
- **KT-Kernel enabled (`kt_*`)** for MoE expert execution (RAW‑INT4 weights).
- **FlashInfer attention** (`attention_backend = "flashinfer"`).
- **TP=8** (`tp_size = 8`) on 8×H200.
- **LoRA overhead minimization**: restrict `lora_target_modules`, keep `max_loras_per_batch` small.
- **Hidden-states overhead minimization**: keep `enable_return_hidden_states=true` (required), but only request residual capture where needed; avoid returning hidden states during long decode.

## Prerequisites (H200)

- **CUDA / driver**: a recent CUDA 12.x + Hopper-capable driver (SM90).
- **PyTorch**: a CUDA build matching your CUDA/driver.
- **Fused-kernel deps**:
  - FlashInfer (attention backend)
  - KT-Kernel / KTransformers kernels for Kimi‑K2.5 (RAW‑INT4)
- **Model path**: use a fully explicit directory (recommended): `/models/kimi-k2.5`.

## Get model weights

```bash
huggingface-cli download moonshotai/Kimi-K2.5 \
  --local-dir /models/kimi-k2.5 \
  --local-dir-use-symlinks False
```

## Repo setup (Heretic + vendored SGLang)

```bash
git submodule update --init --recursive
uv sync
```

Why `uv sync` is enough:
- `pyproject.toml` declares `sglang` as an editable `tool.uv.sources` dependency at `vendor/sglang/python`.
- `tool.uv.default-groups` includes the `inference` group so `uv sync` installs it automatically.

## Configure Heretic for embedded SGLang

```bash
cp config.kimi_k25_h200_offline.toml config.toml
```

Key config fields:
- `backend = "sglang_offline"`
- `sglang_offline_args = { ... }` contains SGLang `ServerArgs` kwargs
- `enable_lora=true` and `enable_return_hidden_states=true` are required for Heretic
- `kt_*` keys enable KT-Kernel and are the main MoE performance lever
- `model` can be either a local directory or an HF id (e.g. `moonshotai/Kimi-K2.5`); when it’s an HF id, Heretic resolves it to an HF cache snapshot directory and uses that same resolved path for SGLang+KT.

## First-run validation checklist (fast path)

### KT-Kernel is active
In `sglang_offline_args`, ensure:
- `kt_weight_path` points to the Kimi folder
- `kt_method = "RAWINT4"`
- `kt_num_gpu_experts = 384` (all experts on GPU on 8×H200)

### Attention backend is FlashInfer
- `attention_backend = "flashinfer"`

### TP is correct
- `tp_size = 8`

### LoRA configured for throughput
For Heretic’s common pattern (base + one adapter):
- `max_loras_per_batch = 2`
- `lora_target_modules = ["o_proj", "down_proj"]` (avoid `"all"` unless required)
- `enable_lora_overlap_loading = true` if you frequently load adapters

### Hidden states used only where needed
SGLang must have `enable_return_hidden_states=true`, but the **decode** path should not be returning hidden states continuously. Heretic’s residual capture should be prefill-only (`max_new_tokens=0`) and request only needed layers.

## Run Heretic

```bash
uv run heretic
```

## Benchmark tok/s (embedded)

```bash
uv run python tools/sglang_offline_bench.py \
  --model-path /models/kimi-k2.5 \
  --tp-size 8 \
  --num-prompts 8 \
  --prompt-tokens 64 \
  --max-new-tokens 128 \
  --with-lora \
  --with-hidden-states
```

Notes:
- `--with-hidden-states` benchmarks **prefill residual capture** (not per-step decode hidden states).
- `--with-lora` loads a tiny synthetic adapter to exercise the hot-swap path.

## Common failure modes

### OOM on startup
- Reduce `max_total_tokens`.
- Reduce `mem_fraction_static`.
- Temporarily set `kt_num_gpu_experts` lower (e.g. 256) to verify boot, then increase.

### Slow (single digits tok/s)
Most likely one of:
- KT-Kernel not active (wrong `kt_*` values, missing kernel build, wrong weight path).
- Attention backend not FlashInfer.
- Hidden states being returned on the decode path rather than prefill-only residual capture.
- LoRA targets too broad (e.g. `lora_target_modules = ["all"]`) increasing overhead.

