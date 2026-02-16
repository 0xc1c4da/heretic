#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if _TOOLS_DIR.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS_DIR.as_posix())

from merge_lora import _detect_base_layout, _validate_structural


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Smoke-test merged checkpoint structure against base checkpoint."
    )
    ap.add_argument("--base-dir", required=True, help="Base checkpoint directory")
    ap.add_argument("--merged-dir", required=True, help="Merged checkpoint directory")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    merged_dir = Path(args.merged_dir)
    base_layout = _detect_base_layout(base_dir)

    _validate_structural(
        base_layout=base_layout,
        out_dir=merged_dir,
        weight_map_out=base_layout.weight_map,
    )
    print("[smoke-test] structural validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
