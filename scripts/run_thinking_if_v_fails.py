#!/usr/bin/env python3
"""Continue thinking labels if the old V head is dead on thinking h_26.

Do **not** k=8 the whole mix. Two stages, same JSONL as the n=20 smoke
(those 20 stay k=2; they are not re-rolled):

  1. 50 **new** L5 at k=8 (quota 5:70 so the 20 already done are skipped)
  2. L1–L4 at k=2 for T / tight-B calibration

Wait until 20/20 is done, then:

  python scripts/run_thinking_if_v_fails.py --dtype bfloat16

k=8 on ~50 L5 is ~2 GPU-days; the k=2 mix is ~half a day more.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "generate_thinking_siblings.py"


def run(cmd: list[str]) -> int:
    print("V fails →", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--complete-tokens", type=int, default=16384)
    args = p.parse_args()

    def extra() -> list[str]:
        bits = [
            "--width",
            "2",
            "--parent-tokens",
            "1024",
            "--peek-tokens",
            "256",
            "--complete-tokens",
            str(args.complete_tokens),
            "--dtype",
            args.dtype,
        ]
        if args.model:
            bits += ["--model", args.model]
        if args.out:
            bits += ["--out", args.out]
        return bits

    # 20 smoke L5 already in jsonl → skipped. 70−20 = 50 new at k=8.
    rc = run(
        [
            sys.executable,
            str(GEN),
            "--k",
            "8",
            "--level-counts",
            "5:70",
            *extra(),
        ]
    )
    if rc != 0:
        return rc
    return run(
        [
            sys.executable,
            str(GEN),
            "--k",
            "2",
            "--level-counts",
            "4:60",
            "3:20",
            "2:15",
            "1:15",
            *extra(),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
