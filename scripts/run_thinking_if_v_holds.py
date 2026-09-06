#!/usr/bin/env python3
"""Continue thinking labels if the old V head still works on the n=20 smoke.

Retrain **T only** (k=2). Mix levels so P(T≤B) sees short and long remainders.
Resumes the same JSONL as the running 20 L5 job — those 20 count toward L5:100.

Wait until 20/20 is done, then:

  python scripts/run_thinking_if_v_holds.py --dtype bfloat16

~160 new problems at k=2, cap 16384, width 2. ~1–1.5 GPU-days after the 20.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "generate_thinking_siblings.py"

# L5:100 includes the smoke 20. L1–L2: tight-B / overthinking calibration.
QUOTAS = ["5:100", "4:60", "3:20", "2:15", "1:15"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--complete-tokens", type=int, default=16384)
    args = p.parse_args()
    cmd = [
        sys.executable,
        str(GEN),
        "--k",
        "2",
        "--width",
        "2",
        "--parent-tokens",
        "1024",
        "--peek-tokens",
        "256",
        "--complete-tokens",
        str(args.complete_tokens),
        "--level-counts",
        *QUOTAS,
        "--dtype",
        args.dtype,
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.out:
        cmd += ["--out", args.out]
    print("V holds → T corpus k=2 mix", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
