#!/usr/bin/env python3
"""Turn thinking sibling JSONL + hidden_L26.npy into a probe cache.

  python scripts/build_thinking_cache.py \
      --src $TOKENAWARE_ARTIFACTS/thinking/siblings \
      --out $TOKENAWARE_ARTIFACTS/cache/thinking_pilot

Then follow ``thinking-t-train.md`` (T-only, max_tokens 16384). Do not train V.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.config import ARTIFACTS_DIR
from tokenaware.probes.cache import SOURCE_BRANCH, SPLIT_TRAIN, SPLIT_VAL, SUBJECTS, CacheWriter


def holdout(pid: str, frac: float = 0.2) -> bool:
    digest = hashlib.sha1(f"think-val:{pid}".encode()).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF < frac


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ARTIFACTS_DIR / "thinking" / "siblings"))
    p.add_argument("--out", default=None)
    p.add_argument("--layer", type=int, default=26)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    args = p.parse_args()

    src = Path(args.src)
    jsonl = src / "states.jsonl"
    hidden_path = src / f"hidden_L{args.layer}.npy"
    if not jsonl.is_file() or not hidden_path.is_file():
        print(f"missing {jsonl} or {hidden_path}", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    states = [r for r in rows if not r.get("skipped")]
    if not states:
        print("no labelled sibling states", file=sys.stderr)
        return 2
    hidden = np.load(hidden_path)
    max_row = max(int(r["hidden_row"]) for r in states)
    if hidden.shape[0] <= max_row:
        print(f"hidden rows {hidden.shape[0]} <= max hidden_row {max_row}", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else ARTIFACTS_DIR / "cache" / "thinking_pilot"
    writer = CacheWriter(out, layers=[args.layer], hidden_dim=hidden.shape[1], n_states=len(states))
    subject_index = {s: i for i, s in enumerate(SUBJECTS)}
    problems: dict[str, int] = {}
    groups: dict[str, int] = {}
    max_k = max(int(r["mc_k"]) for r in states)

    for rec in states:
        pid = rec["problem_id"]
        if pid not in problems:
            problems[pid] = len(problems)
        gid = rec["group_id"]
        if gid not in groups:
            groups[gid] = len(groups)
        split_id = SPLIT_VAL if holdout(pid, args.holdout_frac) else SPLIT_TRAIN
        h = hidden[int(rec["hidden_row"]) : int(rec["hidden_row"]) + 1]
        state_row = {
            "problem_ord": problems[pid],
            "problem_idx": problems[pid],
            "sample_id": int(rec["branch"]),
            "step_index": 0,
            "n_steps": 1,
            "row_in_trace": 0,
            "hist_start": writer.cursor,
            "tokens_so_far": int(rec["tokens_so_far"]),
            "trace_len": int(rec["tokens_so_far"] + rec["t_mc_mean"]),
            "t_same": int(round(rec["t_mc_mean"])),
            "y_same": 1 if rec["v_mc"] >= 0.5 else 0,
            "trace_truncated": 1 if rec["n_truncated"] == rec["mc_k"] else 0,
            "level": int(rec["level"]),
            "subject_idx": subject_index.get(rec.get("subject", "algebra"), 0),
            "problem_chars": int(rec["problem_chars"]),
            "split": split_id,
            "source": SOURCE_BRANCH,
            "group_id": groups[gid],
            "mc_row": -1,
        }
        start = writer.add_states([state_row], {args.layer: h})
        draws = rec["continuations"]
        mc_idx = writer.add_mc(
            {
                "state_row": start,
                "fraction": 0.25,
                "v_mc": float(rec["v_mc"]),
                "t_mc_mean": float(rec["t_mc_mean"]),
                "mc_k": int(rec["mc_k"]),
                "n_correct": int(rec["n_correct"]),
                "n_truncated": int(rec["n_truncated"]),
                "source_tokens_remaining": int(round(rec["t_mc_mean"])),
            },
            draws,
            max_k,
        )
        writer.set_state_mc_row(start, mc_idx)

    writer.finalize(
        {
            "source": "thinking_siblings",
            "layer": args.layer,
            "n_problems": len(problems),
            "n_groups": len(groups),
            "max_k": max_k,
        }
    )
    print(f"wrote {len(states)} states, {len(problems)} problems -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
