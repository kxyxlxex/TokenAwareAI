#!/usr/bin/env python3
"""Step 2b: k=8 root rollouts per probe-train / probe-val problem.

Writes one JSONL per problem under artifacts/rollouts/root/{split}/.
Hidden states are stored in a sidecar .pt to keep JSONL readable.

Usage:
  python scripts/generate_root_rollouts.py --split train --limit 4 --batch-size 4
  python scripts/generate_root_rollouts.py --split train --batch-size 8 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.artifacts import root_paths
from tokenaware.config import ARTIFACTS_DIR, MODEL_ID, ROOT_K
from tokenaware.data import load_split
from tokenaware.generate import generate_rollouts_batch, load_model


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def elapsed(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def output_matches(path: Path, *, k: int, model_id: str, dtype: str) -> bool:
    if not path.exists():
        return False
    try:
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    return len(records) == k and all(
        record.get("run_config")
        == {"k": k, "model": model_id, "dtype": dtype}
        for record in records
    )


def main() -> None:
    run_started = time.monotonic()
    print(f"[{timestamp()}] root rollout job started", flush=True)
    print(f"[{timestamp()}] ARTIFACTS_DIR={ARTIFACTS_DIR}", flush=True)
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--k", type=int, default=ROOT_K)
    p.add_argument("--batch-size", type=int, default=ROOT_K)
    p.add_argument("--limit", type=int, default=0, help="0 = all problems in split")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--model", default=None, help="HF id or local dir")
    p.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16"))
    args = p.parse_args()
    if args.k < 1 or args.batch_size < 1:
        p.error("--k and --batch-size must be positive")

    split = load_split(ARTIFACTS_DIR / "splits" / "math_probe_split.json")
    numbered_problems = list(enumerate(split[args.split], start=1))[args.offset :]
    if args.limit:
        numbered_problems = numbered_problems[: args.limit]

    out_dir = ARTIFACTS_DIR / "rollouts" / "root" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = args.model or MODEL_ID
    pending = []
    for problem_number, problem in numbered_problems:
        dest, hidden_dest = root_paths(
            out_dir, problem_number, problem["problem_id"]
        )
        if hidden_dest.exists() and output_matches(
            dest, k=args.k, model_id=model_id, dtype=args.dtype
        ):
            print(f"skip complete p={problem_number:04d} {problem['problem_id']}")
        else:
            pending.append((problem_number, problem, dest, hidden_dest))
    if not pending:
        print("nothing to do; all requested root outputs match this configuration")
        return

    load_started = time.monotonic()
    model, tokenizer = load_model(model_id=args.model, dtype=args.dtype)
    print(
        f"[{timestamp()}] model ready after {elapsed(time.monotonic() - load_started)}",
        flush=True,
    )
    for i, (problem_number, problem, dest, hid_dest) in enumerate(pending):
        problem_started = time.monotonic()
        records = []
        hidden_pack = []
        requests = [
            {
                "problem": problem["problem"],
                "gold": problem["gold"],
                "problem_id": problem["problem_id"],
                "sample_id": sample_id,
            }
            for sample_id in range(args.k)
        ]
        for batch_start in range(0, len(requests), args.batch_size):
            batch = requests[batch_start : batch_start + args.batch_size]
            rollouts = generate_rollouts_batch(
                model, tokenizer, batch, capture_hidden=True
            )
            for r in rollouts:
                hidden_pack.append(r.hidden)
                records.append(
                    {
                        "problem_id": r.problem_id,
                        "problem_number": problem_number,
                        "run_config": {
                            "k": args.k,
                            "model": model_id,
                            "dtype": args.dtype,
                        },
                        "sample_id": r.sample_id,
                        "level": problem["level"],
                        "subject": problem["subject"],
                        "gold": problem["gold"],
                        "text": r.text,
                        "gen_ids": r.gen_ids,
                        "n_tokens": r.n_tokens,
                        "truncated": r.truncated,
                        "boxed": r.boxed,
                        "correct": r.correct,
                        "steps": r.steps,
                    }
                )
                print(
                    f"[{timestamp()}] [{i+1}/{len(pending)}] "
                    f"p={problem_number:04d} "
                    f"{problem['problem_id']} s={r.sample_id} tok={r.n_tokens} "
                    f"steps={len(r.steps)} ok={r.correct}",
                    flush=True,
                )
        json_temp = dest.with_suffix(".jsonl.tmp")
        hidden_temp = hid_dest.with_suffix(".pt.tmp")
        json_temp.write_text("\n".join(json.dumps(x) for x in records) + "\n")
        torch.save(hidden_pack, hidden_temp)
        hidden_temp.replace(hid_dest)
        json_temp.replace(dest)
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
