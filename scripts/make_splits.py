#!/usr/bin/env python3
"""Build the 500-problem probe-val + 2000-problem probe-train split from MATH train."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.config import ARTIFACTS_DIR
from tokenaware.data import load_math_train, save_split, stratified_split


def main() -> None:
    rows = load_math_train()
    split = stratified_split(rows)
    out = ARTIFACTS_DIR / "splits" / "math_probe_split.json"
    save_split(split, out)
    print(f"source MATH train: {split['n_source']}")
    print(f"val:   {len(split['val'])}  levels={split['level_counts']['val']}")
    print(f"train: {len(split['train'])}  levels={split['level_counts']['train']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
