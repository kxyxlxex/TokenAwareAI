#!/usr/bin/env python3
"""Step 2c: sample fresh continuations from selected root-rollout prefixes.

Requires Step 2b root files under artifacts/rollouts/root/{split}/.
Writes one resumable JSONL per problem under artifacts/labels/mc/{split}/.

Usage:
  python scripts/generate_mc_prefix_labels.py --split train --limit 1 --k 2 --batch-size 2
  python scripts/generate_mc_prefix_labels.py --split train --batch-size 8 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.artifacts import mc_path, root_paths
from tokenaware.config import ARTIFACTS_DIR, MC_K, MODEL_ID
from tokenaware.data import load_split
from tokenaware.generate import generate_continuations_batch, load_model
from tokenaware.mc import mc_label, read_jsonl, select_prefix_states


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def elapsed(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return []


def output_matches(path: Path, run_config: dict) -> bool:
    records = read_records(path)
    return bool(records) and all(
        record.get("run_config") == run_config for record in records
    )


def main() -> None:
    run_started = time.monotonic()
    print(f"[{timestamp()}] MC prefix job started", flush=True)
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--k", type=int, default=MC_K)
    p.add_argument("--batch-size", type=int, default=MC_K)
    p.add_argument("--limit", type=int, default=0, help="0 = all problems in split")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--traces-per-problem", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--model", default=None, help="HF id or local dir")
    p.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16"))
    args = p.parse_args()
    if (
        args.k < 1
        or args.batch_size < 1
        or args.traces_per_problem < 1
        or args.max_new_tokens < 1
    ):
        p.error(
            "--k, --batch-size, --traces-per-problem, and --max-new-tokens "
            "must be positive"
        )

    split = load_split(ARTIFACTS_DIR / "splits" / "math_probe_split.json")
    numbered_problems = list(enumerate(split[args.split], start=1))[args.offset :]
    if args.limit:
        numbered_problems = numbered_problems[: args.limit]

    root_dir = ARTIFACTS_DIR / "rollouts" / "root" / args.split
    out_dir = ARTIFACTS_DIR / "labels" / "mc" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "k": args.k,
        "model": args.model or MODEL_ID,
        "dtype": args.dtype,
        "traces_per_problem": args.traces_per_problem,
        "max_new_tokens": args.max_new_tokens,
    }

    pending: list[tuple[int, dict, Path, Path]] = []
    for problem_number, problem in numbered_problems:
        source, _ = root_paths(root_dir, problem_number, problem["problem_id"])
        dest = mc_path(out_dir, problem_number, problem["problem_id"])
        if output_matches(dest, run_config):
            print(f"skip complete {problem['problem_id']}")
        elif not source.exists():
            print(f"skip missing root rollouts {problem['problem_id']}")
        else:
            pending.append((problem_number, problem, source, dest))

    if not pending:
        print("nothing to do; generate root rollouts first or outputs already exist")
        return

    load_started = time.monotonic()
    model, tokenizer = load_model(model_id=args.model, dtype=args.dtype)
    print(
        f"[{timestamp()}] model ready after {elapsed(time.monotonic() - load_started)}",
        flush=True,
    )
    for problem_i, (problem_number, problem, source, dest) in enumerate(
        pending, start=1
    ):
        problem_started = time.monotonic()
        states = select_prefix_states(
            read_jsonl(source), traces_per_problem=args.traces_per_problem
        )
        if not states:
            print(f"skip no parseable prefixes {problem['problem_id']}")
            continue

        records = []
        temp = dest.with_suffix(".jsonl.tmp")
        partial_records = read_records(temp)
        if temp.exists() and not partial_records:
            temp.unlink()
        if partial_records and not all(
            record.get("run_config") == run_config for record in partial_records
        ):
            temp.unlink()
            partial_records = []
        completed_states = {record["state_id"] for record in partial_records}
        for state_i, state in enumerate(states):
            state_id = (
                f"{problem['problem_id']}:r{state['source_sample_id']}:"
                f"s{state['step_index']}"
            )
            if state_id in completed_states:
                continue
            continuations = []
            requests = [
                {
                    "problem": problem["problem"],
                    "prefix": state["prefix"],
                    "prefix_token_ids": state["prefix_token_ids"],
                    "gold": problem["gold"],
                    "sample_id": sample_id,
                }
                for sample_id in range(args.k)
            ]
            for batch_start in range(0, len(requests), args.batch_size):
                batch = requests[batch_start : batch_start + args.batch_size]
                results = generate_continuations_batch(
                    model,
                    tokenizer,
                    batch,
                    max_new_tokens=args.max_new_tokens,
                )
                for request, result in zip(batch, results):
                    result["sample_id"] = request["sample_id"]
                    continuations.append(result)
                    print(
                        f"[{timestamp()}] [{problem_i}/{len(pending)}] "
                        f"p={problem_number:04d} {problem['problem_id']} "
                        f"state={state_i + 1}/{len(states)} "
                        f"mc={request['sample_id'] + 1}/{args.k} "
                        f"tok={result['n_tokens']} ok={result['correct']}",
                        flush=True,
                    )

            record = {
                "problem_id": problem["problem_id"],
                "problem_number": problem_number,
                "run_config": run_config,
                "level": problem["level"],
                "subject": problem["subject"],
                "gold": problem["gold"],
                "state_id": state_id,
                **state,
                **mc_label(continuations),
                "continuations": continuations,
            }
            records.append(record)
            with temp.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

        records = partial_records + records
        if not records:
            print(f"skip no MC records {problem['problem_id']}")
            continue
        temp.replace(dest)
        print(
            f"[{timestamp()}] wrote {dest} ({len(records)} prefix states) in "
            f"{elapsed(time.monotonic() - problem_started)}",
            flush=True,
        )
    print(
        f"[{timestamp()}] MC prefix job finished in "
        f"{elapsed(time.monotonic() - run_started)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
