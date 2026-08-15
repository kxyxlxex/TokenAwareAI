#!/usr/bin/env python3
"""Step 2a: k=8 root rollouts per probe-train / probe-val problem.

Writes one JSONL per problem under artifacts/rollouts/root/{split}/.
Hidden states are stored in a sidecar .pt to keep JSONL readable.

Usage:
  python scripts/generate_root_rollouts.py --split train --limit 4
  python scripts/generate_root_rollouts.py --split train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd() / "src"))
try:
    from tokenaware.paths import ensure_src_on_path
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tokenaware.paths import ensure_src_on_path

ensure_src_on_path()

from tokenaware.config import ARTIFACTS_DIR, ROOT_K
from tokenaware.data import load_split
from tokenaware.generate import generate_rollout, load_model


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def elapsed(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def main() -> None:
    run_started = time.monotonic()
    print(f"[{timestamp()}] root rollout job started", flush=True)
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--k", type=int, default=ROOT_K)
    p.add_argument("--limit", type=int, default=0, help="0 = all problems in split")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--model", default=None, help="HF id or local dir")
    p.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16"))
    p.add_argument("--load-in-4bit", action="store_true")
    args = p.parse_args()

    split = load_split(ARTIFACTS_DIR / "splits" / "math_probe_split.json")
    problems = split[args.split][args.offset :]
    if args.limit:
        problems = problems[: args.limit]

    out_dir = ARTIFACTS_DIR / "rollouts" / "root" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    load_started = time.monotonic()
    model, tokenizer = load_model(
        model_id=args.model, dtype=args.dtype, load_in_4bit=args.load_in_4bit
    )
    print(
        f"[{timestamp()}] model ready after {elapsed(time.monotonic() - load_started)}",
        flush=True,
    )
    for i, problem in enumerate(problems):
        problem_started = time.monotonic()
        dest = out_dir / f"{problem['problem_id'].replace(':', '_')}.jsonl"
        hid_dest = dest.with_suffix(".pt")
        if dest.exists() and hid_dest.exists():
            print(f"skip {problem['problem_id']}")
            continue
        records = []
        hidden_pack = []
        for s in range(args.k):
            r = generate_rollout(
                model,
                tokenizer,
                problem=problem["problem"],
                gold=problem["gold"],
                problem_id=problem["problem_id"],
                sample_id=s,
            )
            hidden_pack.append(r.hidden)
            rec = {
                "problem_id": r.problem_id,
                "sample_id": r.sample_id,
                "level": problem["level"],
                "subject": problem["subject"],
                "gold": problem["gold"],
                "text": r.text,
                "n_tokens": r.n_tokens,
                "truncated": r.truncated,
                "boxed": r.boxed,
                "correct": r.correct,
                "steps": r.steps,
            }
            records.append(rec)
            print(
                f"[{timestamp()}] [{i+1}/{len(problems)}] {problem['problem_id']} "
                f"s={s} tok={r.n_tokens} steps={len(r.steps)} ok={r.correct}",
                flush=True,
            )
        dest.write_text("\n".join(json.dumps(x) for x in records) + "\n")
        torch.save(hidden_pack, hid_dest)
        print(
            f"[{timestamp()}] finished {problem['problem_id']} in "
            f"{elapsed(time.monotonic() - problem_started)}",
            flush=True,
        )
    print(
        f"[{timestamp()}] root rollout job finished in "
        f"{elapsed(time.monotonic() - run_started)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
