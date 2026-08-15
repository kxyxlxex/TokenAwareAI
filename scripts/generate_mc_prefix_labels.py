#!/usr/bin/env python3
"""Step 2c: sample fresh continuations from selected root-rollout prefixes.

Requires Step 2b root files under artifacts/rollouts/root/{split}/.
Writes one resumable JSONL per problem under artifacts/labels/mc/{split}/.

Usage:
  python scripts/generate_mc_prefix_labels.py --split train --limit 1 --k 2 --load-in-4bit
  python scripts/generate_mc_prefix_labels.py --split train --load-in-4bit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
try:
    from tokenaware.paths import ensure_src_on_path
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tokenaware.paths import ensure_src_on_path

ensure_src_on_path()

from tokenaware.config import ARTIFACTS_DIR, MC_K
from tokenaware.data import load_split
from tokenaware.generate import generate_continuation, load_model
from tokenaware.mc import mc_label, read_jsonl, select_prefix_states


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--k", type=int, default=MC_K)
    p.add_argument("--limit", type=int, default=0, help="0 = all problems in split")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--traces-per-problem", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--model", default=None, help="HF id or local dir")
    p.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16"))
    p.add_argument("--load-in-4bit", action="store_true")
    args = p.parse_args()
    if args.k < 1 or args.traces_per_problem < 1 or args.max_new_tokens < 1:
        p.error("--k, --traces-per-problem, and --max-new-tokens must be positive")

    split = load_split(ARTIFACTS_DIR / "splits" / "math_probe_split.json")
    problems = split[args.split][args.offset :]
    if args.limit:
        problems = problems[: args.limit]

    root_dir = ARTIFACTS_DIR / "rollouts" / "root" / args.split
    out_dir = ARTIFACTS_DIR / "labels" / "mc" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[dict, Path, Path]] = []
    for problem in problems:
        stem = problem["problem_id"].replace(":", "_")
        source = root_dir / f"{stem}.jsonl"
        dest = out_dir / f"{stem}.jsonl"
        if dest.exists():
            print(f"skip complete {problem['problem_id']}")
        elif not source.exists():
            print(f"skip missing root rollouts {problem['problem_id']}")
        else:
            pending.append((problem, source, dest))

    if not pending:
        print("nothing to do; generate root rollouts first or outputs already exist")
        return

    model, tokenizer = load_model(
        model_id=args.model, dtype=args.dtype, load_in_4bit=args.load_in_4bit
    )
    for problem_i, (problem, source, dest) in enumerate(pending, start=1):
        states = select_prefix_states(
            read_jsonl(source), traces_per_problem=args.traces_per_problem
        )
        if not states:
            print(f"skip no parseable prefixes {problem['problem_id']}")
            continue

        records = []
        for state_i, state in enumerate(states):
            continuations = []
            for sample_id in range(args.k):
                result = generate_continuation(
                    model,
                    tokenizer,
                    problem=problem["problem"],
                    prefix=state["prefix"],
                    gold=problem["gold"],
                    max_new_tokens=args.max_new_tokens,
                )
                result["sample_id"] = sample_id
                continuations.append(result)
                print(
                    f"[{problem_i}/{len(pending)}] {problem['problem_id']} "
                    f"state={state_i + 1}/{len(states)} mc={sample_id + 1}/{args.k} "
                    f"tok={result['n_tokens']} ok={result['correct']}"
                )

            records.append(
                {
                    "problem_id": problem["problem_id"],
                    "level": problem["level"],
                    "subject": problem["subject"],
                    "gold": problem["gold"],
                    "state_id": (
                        f"{problem['problem_id']}:r{state['source_sample_id']}:"
                        f"s{state['step_index']}"
                    ),
                    **state,
                    **mc_label(continuations),
                    "continuations": continuations,
                }
            )

        temp = dest.with_suffix(".jsonl.tmp")
        temp.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        temp.replace(dest)
        print(f"wrote {dest} ({len(records)} prefix states)")


if __name__ == "__main__":
    main()
