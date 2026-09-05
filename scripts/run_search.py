#!/usr/bin/env python3
"""Budget-matched search evaluation: V-only vs V+T (and baselines) on MATH-500.

  # smoke: 20 problems, two arms, one budget
  python scripts/run_search.py --probe $TOKENAWARE_ARTIFACTS/sweeps/phase0/<tag>/probe.pt \
      --arms v_only vt --budgets 256 --seeds 0 --limit 20 --dtype bfloat16

  # the paper grid
  python scripts/run_search.py --probe .../probe.pt \
      --arms single_chain majority_vote random v_only vt bang_per_buck feasibility_gate \
      --budgets 256 512 1024 --seeds 0 1 2 --dtype bfloat16

  # negative control
  python scripts/run_search.py --dataset gsm8k --probe .../probe.pt --arms v_only vt \
      --budgets 256 512 --seeds 0

One JSONL per (arm, budget, seed) under ``<out>/<dataset>/``; problems already
present are skipped, so the job is resumable. ``summary.json`` is rewritten at
the end of every cell with accuracy and token spend, pooled and per level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from tokenaware.config import ARTIFACTS_DIR, MODEL_ID
from tokenaware.data import load_eval_set
from tokenaware.search import (
    ALL_ARMS,
    CHAIN_ARMS,
    PROBE_ARMS,
    ChainRun,
    HFGenerator,
    OnlineProbe,
    SearchConfig,
    SearchRun,
    read_records,
    run_chain_baselines,
    run_tree_searches,
    summarize_records,
)


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def stable_hash(text: str) -> int:
    """Process-independent 32-bit hash (``hash()`` is salted per interpreter)."""
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


def elapsed(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def cell_path(out_dir: Path, arm: str, budget: int, seed: int) -> Path:
    return out_dir / f"{arm}_B{budget}_s{seed}.jsonl"


def write_summary(out_dir: Path) -> dict:
    summary: dict = {}
    for path in sorted(out_dir.glob("*_B*_s*.jsonl")):
        records = read_records(path)
        if not records:
            continue
        arm, budget, seed = records[0]["arm"], records[0]["budget"], records[0]["seed"]
        summary.setdefault(arm, {}).setdefault(str(budget), {})[str(seed)] = (
            summarize_records(records)
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def print_table(summary: dict) -> None:
    print(f"\n{'arm':<18}{'B0':>6}{'seeds':>6}{'n':>6}{'acc':>8}{'acc<=B':>8}{'tokens':>9}")
    for arm in sorted(summary):
        for budget in sorted(summary[arm], key=int):
            cells = list(summary[arm][budget].values())
            n = int(np.mean([c["n"] for c in cells]))
            acc = np.mean([c["accuracy"] for c in cells])
            acc_b = np.mean([c["accuracy_within_budget"] for c in cells])
            tok = np.mean([c["tokens_mean"] for c in cells])
            print(
                f"{arm:<18}{budget:>6}{len(cells):>6}{n:>6}{acc:>8.3f}{acc_b:>8.3f}{tok:>9.1f}"
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("math500", "gsm8k"), default="math500")
    p.add_argument("--arms", nargs="+", default=["single_chain", "majority_vote", "v_only", "vt"])
    p.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--probe", default=None, help="probe.pt for the probe-scored arms")
    p.add_argument("--width", type=int, default=3, help="candidate next steps per expansion")
    p.add_argument("--branch-max-tokens", type=int, default=160)
    p.add_argument("--eta", type=float, default=0.2, help="stop branching below this remaining fraction")
    p.add_argument("--alpha-max", type=float, default=8.0)
    p.add_argument("--greedy", action="store_true", help="argmax selection instead of sampling")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--limit", type=int, default=0, help="first N problems (0 = all)")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--levels", type=int, nargs="+", default=None)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--model", default=None)
    p.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="default <artifacts>/search")
    p.add_argument("--summary-only", action="store_true", help="rebuild summary.json and exit")
    args = p.parse_args()

    for arm in args.arms:
        if arm not in ALL_ARMS:
            print(f"unknown arm {arm!r}; choose from {ALL_ARMS}", file=sys.stderr)
            return 2
    needs_probe = any(arm in PROBE_ARMS for arm in args.arms)
    if needs_probe and not args.probe:
        print("--probe is required for probe-scored arms", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else ARTIFACTS_DIR / "search"
    out_dir = out_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        print_table(write_summary(out_dir))
        return 0

    problems = load_eval_set(args.dataset)
    if args.levels:
        problems = [q for q in problems if q.get("level") in set(args.levels)]
    problems = problems[args.offset :]
    if args.limit:
        problems = problems[: args.limit]
    print(f"[{stamp()}] {args.dataset}: {len(problems)} problems, arms={args.arms}, "
          f"budgets={args.budgets}, seeds={args.seeds}")

    cells = [
        (arm, budget, seed)
        for seed in args.seeds
        for budget in args.budgets
        for arm in args.arms
    ]
    todo = []
    for arm, budget, seed in cells:
        done_ids = {r["problem_id"] for r in read_records(cell_path(out_dir, arm, budget, seed))}
        pending = [q for q in problems if q["problem_id"] not in done_ids]
        if pending:
            todo.append((arm, budget, seed, pending))
    if not todo:
        print("every cell is complete")
        print_table(write_summary(out_dir))
        return 0
    print(f"[{stamp()}] {len(todo)} cells with pending problems")

    from tokenaware.generate import load_model

    model, tokenizer = load_model(model_id=args.model, dtype=args.dtype)
    probe = OnlineProbe(args.probe, device=args.device) if needs_probe else None
    # Hook only the layers the probe reads; the replay pass is cheaper for it.
    generator = HFGenerator(model, tokenizer, layers=tuple(probe.layers) if probe else ())
    if probe is not None:
        print(f"probe {probe.tag}: layers={probe.layers} params={probe.n_params / 1e6:.2f}M")

    run_meta = {
        "model": args.model or MODEL_ID,
        "dtype": args.dtype,
        "probe": args.probe,
        "probe_tag": probe.tag if probe else None,
        "width": args.width,
        "branch_max_tokens": args.branch_max_tokens,
        "eta": args.eta,
        "alpha_max": args.alpha_max,
        "greedy": args.greedy,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    job_started = time.monotonic()
    for arm, budget, seed, pending in todo:
        cell_started = time.monotonic()
        path = cell_path(out_dir, arm, budget, seed)
        print(f"\n[{stamp()}] cell arm={arm} B0={budget} seed={seed}: {len(pending)} problems")
        torch.manual_seed(seed * 1_000_003 + budget)
        random.seed(seed)

        def progress(round_index: int, n_active: int) -> None:
            if round_index % 5 == 0:
                print(
                    f"  [{stamp()}] round {round_index}: {n_active} active "
                    f"({elapsed(time.monotonic() - cell_started)})",
                    flush=True,
                )

        if arm in CHAIN_ARMS:
            runs = [ChainRun(problem=q, arm=arm, budget=budget) for q in pending]
            run_chain_baselines(runs, generator, batch_size=args.batch_size, progress=progress)
        else:
            cfg = SearchConfig(
                arm=arm,
                budget=budget,
                width=args.width,
                branch_max_tokens=args.branch_max_tokens,
                eta=args.eta,
                alpha_max=args.alpha_max,
                greedy=args.greedy,
                max_steps=args.max_steps,
            )
            runs = [
                SearchRun(
                    problem=q,
                    cfg=cfg,
                    rng=np.random.default_rng([seed, budget, stable_hash(q["problem_id"])]),
                )
                for q in pending
            ]
            run_tree_searches(
                runs, generator, probe, batch_size=args.batch_size, progress=progress
            )

        with path.open("a") as fh:
            for run in runs:
                fh.write(json.dumps(run.record(seed)) + "\n")
        cell = summarize_records(read_records(path))
        print(
            f"[{stamp()}] done arm={arm} B0={budget} seed={seed} in "
            f"{elapsed(time.monotonic() - cell_started)}: acc={cell['accuracy']} "
            f"acc<=B={cell['accuracy_within_budget']} tokens={cell['tokens_mean']} "
            f"reasons={cell['reasons']}",
            flush=True,
        )
        write_summary(out_dir)

    print(f"\n[{stamp()}] finished in {elapsed(time.monotonic() - job_started)}")
    print_table(write_summary(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
