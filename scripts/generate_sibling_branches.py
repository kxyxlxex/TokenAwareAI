#!/usr/bin/env python3
"""Generate true sibling states and Monte-Carlo label them.

Why this exists. The kill criterion is *within-problem sibling* ranking, but the
root/MC corpus has no real siblings: its states come from two independent traces
at 25/50/75% depth, so a "sibling pair" differs in both content and absolute
position. Tree search compares candidates that share a parent prefix and differ
by one step. This script builds exactly that:

  parent prefix  ->  n sampled next steps (deduplicated)  ->  k continuations each

Each sibling state gets a hidden vector captured at its own last token, so the
probe is evaluated on the states it would actually score during search.

  python scripts/generate_sibling_branches.py --split train --limit 25   # pilot
  python scripts/generate_sibling_branches.py --split train --problems 500

Cost per problem at defaults (1 parent, 3 branches, k=8): 3 short branch
generations plus 24 continuations, roughly 2.6K output tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from tokenaware.artifacts import problem_stem, root_paths
from tokenaware.config import ARTIFACTS_DIR, MC_K, MODEL_ID
from tokenaware.data import load_split
from tokenaware.generate import (
    capture_state_vectors,
    generate_continuation_ids_batch,
    generate_continuations_batch,
    load_model,
)
from tokenaware.mc import mc_label, read_jsonl
from tokenaware.steps import attach_token_offsets, parse_steps


def stamp() -> str:
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


def pick_parents(
    rollouts: list[dict], fractions: tuple[float, ...], source_sample_id: int
) -> list[dict]:
    """Parent prefixes on one trace, at the requested depth fractions."""
    trace = next(
        (
            r
            for r in rollouts
            if r.get("sample_id") == source_sample_id
            and r.get("steps")
            and r.get("gen_ids")
        ),
        None,
    )
    if trace is None:
        return []
    steps = trace["steps"]
    parents, used = [], set()
    for fraction in fractions:
        step_index = min(len(steps) - 1, max(0, round(fraction * len(steps)) - 1))
        if step_index in used:
            continue
        step = steps[step_index]
        offset = step.get("last_token_offset")
        if offset is None:
            continue
        used.add(step_index)
        parents.append(
            {
                "source_sample_id": trace["sample_id"],
                "fraction": fraction,
                "step_index": step_index,
                "prefix_char_end": step["char_end"],
                "prefix": trace["text"][: step["char_end"]],
                "prefix_token_ids": trace["gen_ids"][: offset + 1],
            }
        )
    return parents


def first_step_slice(text: str, gen_ids: list[int], tokenizer) -> tuple[str, list[int]] | None:
    """Cut a continuation at its first reasoning-step boundary, in token space."""
    steps = attach_token_offsets(parse_steps(text), text, gen_ids, tokenizer)
    if not steps or steps[0].last_token_offset is None:
        return None
    offset = int(steps[0].last_token_offset)
    return text[: steps[0].char_end], gen_ids[: offset + 1]


def main() -> int:
    run_started = time.monotonic()
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--k", type=int, default=MC_K, help="continuations per sibling")
    p.add_argument("--branches", type=int, default=3, help="sampled next steps")
    p.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.5],
        help="parent depth fractions; one sibling group per fraction",
    )
    p.add_argument("--source-sample-id", type=int, default=0)
    p.add_argument("--branch-max-tokens", type=int, default=160)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--problems", type=int, default=0, help="0 = every problem with root rollouts"
    )
    p.add_argument("--limit", type=int, default=0, help="alias for --problems")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--model", default=None)
    p.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16"))
    args = p.parse_args()

    n_problems = args.problems or args.limit
    split = load_split(ARTIFACTS_DIR / "splits" / "math_probe_split.json")
    numbered = list(enumerate(split[args.split], start=1))[args.offset :]

    root_dir = ARTIFACTS_DIR / "rollouts" / "root" / args.split
    out_dir = ARTIFACTS_DIR / "branches" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "k": args.k,
        "branches": args.branches,
        "fractions": list(args.fractions),
        "source_sample_id": args.source_sample_id,
        "branch_max_tokens": args.branch_max_tokens,
        "max_new_tokens": args.max_new_tokens,
        "model": args.model or MODEL_ID,
        "dtype": args.dtype,
    }

    pending = []
    for problem_number, problem in numbered:
        source, _ = root_paths(root_dir, problem_number, problem["problem_id"])
        stem = problem_stem(problem_number, problem["problem_id"])
        dest = out_dir / f"{stem}.jsonl"
        dest_pt = out_dir / f"{stem}.pt"
        existing = read_records(dest)
        if (
            existing
            and dest_pt.exists()
            and all(r.get("run_config") == run_config for r in existing)
        ):
            continue
        if not source.exists():
            continue
        pending.append((problem_number, problem, source, dest, dest_pt))
        if n_problems and len(pending) >= n_problems:
            break

    if not pending:
        print("nothing to do; root rollouts missing or outputs already complete")
        return 0

    print(f"[{stamp()}] sibling-branch job: {len(pending)} problems", flush=True)
    model, tokenizer = load_model(model_id=args.model, dtype=args.dtype)

    totals = {"groups": 0, "siblings": 0, "collapsed_groups": 0, "duplicates": 0}
    for problem_i, (problem_number, problem, source, dest, dest_pt) in enumerate(
        pending, start=1
    ):
        problem_started = time.monotonic()
        rollouts = read_jsonl(source)
        parents = pick_parents(
            rollouts, tuple(args.fractions), args.source_sample_id
        )
        if not parents:
            print(f"skip no usable parent prefix {problem['problem_id']}")
            continue

        records: list[dict] = []
        hidden_pack: list[dict] = []
        for parent in parents:
            group_id = (
                f"{problem['problem_id']}:r{parent['source_sample_id']}"
                f":s{parent['step_index']}:f{parent['fraction']}"
            )
            branch_requests = [
                {
                    "problem": problem["problem"],
                    "prefix": parent["prefix"],
                    "prefix_token_ids": parent["prefix_token_ids"],
                }
                for _ in range(args.branches)
            ]
            raw = []
            for start in range(0, len(branch_requests), args.batch_size):
                raw += generate_continuation_ids_batch(
                    model,
                    tokenizer,
                    branch_requests[start : start + args.batch_size],
                    max_new_tokens=args.branch_max_tokens,
                )

            siblings = []
            seen_steps: set[str] = set()
            for item in raw:
                cut = first_step_slice(item["text"], item["gen_ids"], tokenizer)
                if cut is None:
                    continue
                branch_text, branch_ids = cut
                key = branch_text.strip()
                if not key:
                    continue
                if key in seen_steps:
                    totals["duplicates"] += 1
                    continue
                seen_steps.add(key)
                siblings.append(
                    {
                        "branch_text": branch_text,
                        "state_text": parent["prefix"] + branch_text,
                        "state_token_ids": list(parent["prefix_token_ids"])
                        + list(branch_ids),
                    }
                )
            totals["groups"] += 1
            if len(siblings) < 2:
                totals["collapsed_groups"] += 1
                print(
                    f"[{stamp()}] group collapsed to {len(siblings)} unique step(s) "
                    f"{group_id}",
                    flush=True,
                )
                if not siblings:
                    continue

            vectors = capture_state_vectors(
                model,
                tokenizer,
                [
                    {
                        "problem": problem["problem"],
                        "prefix_token_ids": sibling["state_token_ids"],
                    }
                    for sibling in siblings
                ],
            )

            for branch_index, (sibling, hidden) in enumerate(zip(siblings, vectors)):
                continuations = []
                requests = [
                    {
                        "problem": problem["problem"],
                        "prefix": sibling["state_text"],
                        "prefix_token_ids": sibling["state_token_ids"],
                        "gold": problem["gold"],
                        "sample_id": sample_id,
                    }
                    for sample_id in range(args.k)
                ]
                for start in range(0, len(requests), args.batch_size):
                    batch = requests[start : start + args.batch_size]
                    results = generate_continuations_batch(
                        model,
                        tokenizer,
                        batch,
                        max_new_tokens=args.max_new_tokens,
                    )
                    for request, result in zip(batch, results):
                        result["sample_id"] = request["sample_id"]
                        continuations.append(result)
                label = mc_label(continuations)
                records.append(
                    {
                        "problem_id": problem["problem_id"],
                        "problem_number": problem_number,
                        "run_config": run_config,
                        "level": problem["level"],
                        "subject": problem["subject"],
                        "gold": problem["gold"],
                        "group_id": group_id,
                        "group_size": len(siblings),
                        "branch_index": branch_index,
                        "state_id": f"{group_id}:b{branch_index}",
                        "parent_source_sample_id": parent["source_sample_id"],
                        "parent_step_index": parent["step_index"],
                        "parent_fraction": parent["fraction"],
                        "parent_prefix_char_end": parent["prefix_char_end"],
                        "branch_text": sibling["branch_text"],
                        "tokens_so_far": len(sibling["state_token_ids"]),
                        **label,
                        "continuations": continuations,
                    }
                )
                hidden_pack.append(hidden)
                totals["siblings"] += 1
                print(
                    f"[{stamp()}] [{problem_i}/{len(pending)}] "
                    f"p={problem_number:04d} {group_id} b{branch_index} "
                    f"v_mc={label['v_mc']:.3f} t_mc={label['t_mc_mean']:.1f}",
                    flush=True,
                )

        if not records:
            continue
        tmp_jsonl = dest.with_suffix(".jsonl.tmp")
        tmp_pt = dest_pt.with_suffix(".pt.tmp")
        tmp_jsonl.write_text("".join(json.dumps(r) + "\n" for r in records))
        torch.save(hidden_pack, tmp_pt)
        tmp_jsonl.replace(dest)
        tmp_pt.replace(dest_pt)
        print(
            f"[{stamp()}] wrote {dest.name} ({len(records)} sibling states) in "
            f"{elapsed(time.monotonic() - problem_started)}",
            flush=True,
        )

    print(
        f"[{stamp()}] finished in {elapsed(time.monotonic() - run_started)} "
        f"{json.dumps(totals)}",
        flush=True,
    )
    if totals["groups"] and totals["collapsed_groups"] / totals["groups"] > 0.3:
        print(
            "WARNING: over 30% of sibling groups collapsed to fewer than two unique "
            "next steps. At temperature 0.7 the policy is near-deterministic at these "
            "prefixes; raise --branches or sample parents deeper before trusting the "
            "sibling metric.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
