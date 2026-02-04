"""
Merge a PEFT LoRA adapter into a (float) HF base model.

Why your previous script OOM'd:
- `device_map="auto"` can place some parameters on CPU (Accelerate offload hooks).
- During `PeftModel.merge_and_unload()`, PEFT calls `onload_layer()` which triggers
  Accelerate's hook to move the target module's tensors to the hook's execution device.
- For CPU-offloaded modules the execution device is typically GPU 0.
- If GPU 0 is already nearly full with weights, the first extra allocation (even ~30MiB)
  will fail with CUDA OOM.

This script reserves headroom per GPU via `max_memory` so merges have space to materialize
the per-layer LoRA delta, and optionally uses disk offload for overflow.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_dtype(s: str) -> torch.dtype | str:
    s = s.lower().strip()
    if s in {"auto"}:
        return "auto"
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise SystemExit(f"Unsupported dtype: {s}")


def _auto_max_memory(headroom_gib: float, cpu_gib: float | None) -> dict[Any, str]:
    """
    Build `max_memory` dict leaving `headroom_gib` free on each CUDA device.
    Uses GiB (base-2) to match what Accelerate/Transformers expect in strings.
    """
    max_mem: dict[Any, str] = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_bytes = torch.cuda.get_device_properties(i).total_memory
            total_gib = total_bytes / (1024**3)
            allow_gib = max(1.0, total_gib - headroom_gib)
            # Keep 1 decimal to avoid overly optimistic rounding.
            max_mem[i] = f"{allow_gib:.1f}GiB"
    if cpu_gib is not None:
        max_mem["cpu"] = f"{cpu_gib:.1f}GiB"
    return max_mem


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base model HF id or local path")
    p.add_argument("--lora", required=True, help="LoRA adapter folder (PEFT)")
    p.add_argument("--out", required=True, help="Output directory for merged model")

    p.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype: auto|bf16|fp16|fp32 (default: auto)",
    )
    p.add_argument(
        "--device-map",
        default="balanced_low_0",
        help="Transformers device_map (default: balanced_low_0). "
        "If you use 'auto' and any modules offload to CPU, merges tend to concentrate on GPU 0.",
    )
    p.add_argument(
        "--headroom-gib",
        type=float,
        default=20.0,
        help="GiB to leave free per GPU for merge temporaries (default: 20).",
    )
    p.add_argument(
        "--cpu-max-gib",
        type=float,
        default=None,
        help="Optional CPU RAM cap for Accelerate max_memory (GiB). "
        "If omitted, Transformers may use as much CPU RAM as it wants.",
    )
    p.add_argument(
        "--offload-folder",
        default=None,
        help="Optional folder for disk offload (recommended when the model can't fully fit in GPU+CPU caps).",
    )
    p.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Shard size for saving (default: 5GB). Smaller reduces peak CPU RAM during save.",
    )
    args = p.parse_args()

    # Reduce fragmentation on multi-GPU and very large tensor workloads.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    dtype = _parse_dtype(args.dtype)
    max_memory = _auto_max_memory(headroom_gib=args.headroom_gib, cpu_gib=args.cpu_max_gib)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, Any] = dict(
        dtype=dtype,
        device_map=args.device_map,
        max_memory=max_memory if max_memory else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if args.offload_folder:
        load_kwargs["offload_folder"] = args.offload_folder
        # When disk offload is active, keeping the (CPU) state dict on disk avoids big RAM spikes.
        load_kwargs["offload_state_dict"] = True

    print("loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print("loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.lora, is_trainable=False)
    model.eval()

    print("merging (this can take a while)...")
    with torch.no_grad():
        merged_model = model.merge_and_unload(safe_merge=False)
    merged_model.eval()

    print("saving merged model...")
    merged_model.save_pretrained(
        out_dir.as_posix(),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(out_dir.as_posix())

    print("done.")


if __name__ == "__main__":
    main()