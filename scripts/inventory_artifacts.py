#!/usr/bin/env python3
"""Audit the artifact corpus before spending GPU time on probes.

Answers, per split: how many problems have complete root pairs, how many have MC
labels, what the correctness / length / truncation / step-count distributions
look like, and how many probe training rows the corpus can yield.

  python scripts/inventory_artifacts.py
  python scripts/inventory_artifacts.py --split train --deep   # also opens .pt headers

Writes artifacts/reports/inventory.json and prints a summary. Reads JSONL only
(cheap) unless --deep, which validates that every .pt row count matches the
number of step boundaries in its JSONL sibling.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.artifacts import problem_stem
from tokenaware.config import ARTIFACTS_DIR, N_PROBE_TRAIN, N_PROBE_VAL
from tokenaware.data import load_split
from tokenaware.probes.cache import pt_row_counts


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def q(p: float) -> float:
        if len(ordered) == 1:
            return round(float(ordered[0]), 2)
        idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return round(float(ordered[idx]), 2)

    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 2),
        "p10": q(0.10),
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": round(float(ordered[-1]), 2),
    }


def read_jsonl(path: Path) -> list[dict]:
    try:
        return [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return []


def contiguous_ranges(numbers: list[int]) -> list[str]:
    if not numbers:
        return []
    numbers = sorted(numbers)
    out, start, prev = [], numbers[0], numbers[0]
    for n in numbers[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = n
    out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return out


def audit_split(artifacts: Path, split_name: str, problems: list[dict], deep: bool) -> dict:
    root_dir = artifacts / "rollouts" / "root" / split_name
    mc_dir = artifacts / "labels" / "mc" / split_name
    branch_dir = artifacts / "branches" / split_name

    report: dict = {
        "split": split_name,
        "n_problems_in_split": len(problems),
        "root": {},
        "mc": {},
        "branches": {},
    }

    have_pair, only_jsonl, missing = [], [], []
    trace_tokens, trace_steps = [], []
    n_traces = n_correct = n_truncated = 0
    correct_by_level: dict[int, list[int]] = defaultdict(list)
    steps_with_offset = 0
    steps_total = 0
    run_configs: Counter = Counter()
    pt_mismatches: list[str] = []

    for ordinal, problem in enumerate(problems, start=1):
        stem = problem_stem(ordinal, problem["problem_id"])
        jsonl = root_dir / f"{stem}.jsonl"
        pt = root_dir / f"{stem}.pt"
        if not jsonl.exists():
            missing.append(ordinal)
            continue
        if not pt.exists():
            only_jsonl.append(ordinal)
            continue
        have_pair.append(ordinal)

        rows = read_jsonl(jsonl)
        if not rows:
            pt_mismatches.append(f"{stem}: unreadable jsonl")
            continue
        run_configs[json.dumps(rows[0].get("run_config"), sort_keys=True)] += 1
        for row in rows:
            n_traces += 1
            n_correct += bool(row.get("correct"))
            n_truncated += bool(row.get("truncated"))
            trace_tokens.append(row.get("n_tokens", 0))
            steps = row.get("steps") or []
            trace_steps.append(len(steps))
            steps_total += len(steps)
            steps_with_offset += sum(
                1 for s in steps if s.get("last_token_offset") is not None
            )
            correct_by_level[int(row.get("level", 0))].append(bool(row.get("correct")))

        if deep:
            try:
                counts = pt_row_counts(pt)
            except (OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
                pt_mismatches.append(f"{stem}: unreadable .pt ({exc})")
                continue
            if len(counts) != len(rows):
                pt_mismatches.append(
                    f"{stem}: .pt has {len(counts)} samples, jsonl has {len(rows)}"
                )
                continue
            for row, count in zip(rows, counts):
                expected = sum(
                    1
                    for s in (row.get("steps") or [])
                    if s.get("last_token_offset") is not None
                )
                if expected != count:
                    pt_mismatches.append(
                        f"{stem} sample {row.get('sample_id')}: "
                        f".pt rows={count} expected={expected}"
                    )

    report["root"] = {
        "complete_pairs": len(have_pair),
        "coverage_pct": pct(len(have_pair), len(problems)),
        "jsonl_without_pt": len(only_jsonl),
        "missing_problem_ordinals": contiguous_ranges(missing)[:20],
        "n_missing": len(missing),
        "n_traces": n_traces,
        "correct_rate": pct(n_correct, n_traces),
        "truncation_rate": pct(n_truncated, n_traces),
        "tokens_per_trace": quantiles([float(v) for v in trace_tokens]),
        "steps_per_trace": quantiles([float(v) for v in trace_steps]),
        "step_states_total": steps_total,
        "step_states_with_hidden": steps_with_offset,
        "step_offset_loss_pct": pct(steps_total - steps_with_offset, steps_total),
        "correct_rate_by_level": {
            str(lvl): pct(sum(vals), len(vals))
            for lvl, vals in sorted(correct_by_level.items())
        },
        "run_configs": {k: v for k, v in run_configs.most_common()},
        "pt_alignment_problems": pt_mismatches[:20],
        "n_pt_alignment_problems": len(pt_mismatches),
        "deep_checked": deep,
    }

    mc_files = sorted(mc_dir.glob("*.jsonl")) if mc_dir.is_dir() else []
    mc_states = 0
    v_values, t_values, mc_ks = [], [], Counter()
    mc_by_fraction: dict[str, int] = Counter()
    mc_trunc = 0
    mc_continuations = 0
    v_degenerate = 0
    joinable = 0
    for path in mc_files:
        rows = read_jsonl(path)
        for row in rows:
            mc_states += 1
            mc_ks[row.get("mc_k")] += 1
            v_values.append(float(row.get("v_mc", 0.0)))
            t_values.append(float(row.get("t_mc_mean", 0.0)))
            mc_by_fraction[str(row.get("fraction"))] += 1
            mc_trunc += int(row.get("n_truncated", 0))
            mc_continuations += len(row.get("continuations") or [])
            if float(row.get("v_mc", 0.0)) in (0.0, 1.0):
                v_degenerate += 1
            if row.get("source_sample_id") is not None and row.get("step_index") is not None:
                joinable += 1

    report["mc"] = {
        "problem_files": len(mc_files),
        "coverage_pct": pct(len(mc_files), len(problems)),
        "labelled_states": mc_states,
        "states_per_problem": round(mc_states / len(mc_files), 2) if mc_files else 0,
        "continuations": mc_continuations,
        "mc_k_histogram": {str(k): v for k, v in mc_ks.most_common()},
        "joinable_to_root": joinable,
        "v_mc": quantiles(v_values),
        "v_mc_saturated_pct": pct(v_degenerate, mc_states),
        "t_mc_mean": quantiles(t_values),
        "continuation_truncation_rate": pct(mc_trunc, mc_continuations),
        "by_fraction": dict(sorted(mc_by_fraction.items())),
    }

    branch_files = sorted(branch_dir.glob("*.jsonl")) if branch_dir.is_dir() else []
    branch_states = 0
    branch_groups = 0
    for path in branch_files:
        rows = read_jsonl(path)
        branch_states += len(rows)
        branch_groups += len({r.get("group_id") for r in rows})
    report["branches"] = {
        "problem_files": len(branch_files),
        "sibling_states": branch_states,
        "sibling_groups": branch_groups,
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default=str(ARTIFACTS_DIR))
    p.add_argument("--split", action="append", choices=("train", "val"), default=None)
    p.add_argument(
        "--deep",
        action="store_true",
        help="also verify every .pt row count against its JSONL (slower, no torch needed)",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    artifacts = Path(args.artifacts).expanduser().resolve()
    split_path = artifacts / "splits" / "math_probe_split.json"
    if not split_path.is_file():
        print(
            f"missing {split_path}. Fetch artifacts first "
            f"(scripts/fetch_artifacts_hf.py) or run scripts/make_splits.py.",
            file=sys.stderr,
        )
        return 2
    split = load_split(split_path)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "artifacts_dir": str(artifacts),
        "split_file": {
            "seed": split.get("seed"),
            "n_source": split.get("n_source"),
            "n_train": len(split.get("train") or []),
            "n_val": len(split.get("val") or []),
            "expected_train": N_PROBE_TRAIN,
            "expected_val": N_PROBE_VAL,
            "level_counts": split.get("level_counts"),
        },
        "splits": {},
    }
    for name in args.split or ("train", "val"):
        report["splits"][name] = audit_split(
            artifacts, name, split.get(name) or [], deep=args.deep
        )

    train = report["splits"].get("train", {})
    root = train.get("root", {})
    mc = train.get("mc", {})
    report["probe_row_estimate"] = {
        "same_trace_states": root.get("step_states_with_hidden", 0),
        "mc_states": mc.get("labelled_states", 0),
        "mc_draws": mc.get("continuations", 0),
        "total_outcome_draws": root.get("step_states_with_hidden", 0)
        + mc.get("continuations", 0),
        "hidden_cache_gib_per_layer": round(
            root.get("step_states_with_hidden", 0) * 4096 * 2 / (1 << 30), 2
        ),
    }

    out = Path(args.out) if args.out else artifacts / "reports" / "inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")

    warnings = []
    for name, section in report["splits"].items():
        r = section["root"]
        if r["complete_pairs"] == 0:
            warnings.append(f"{name}: no root rollouts at all")
        elif r["coverage_pct"] < 99:
            warnings.append(
                f"{name}: root coverage {r['coverage_pct']}% "
                f"({r['n_missing']} problems missing)"
            )
        if r["jsonl_without_pt"]:
            warnings.append(
                f"{name}: {r['jsonl_without_pt']} problems have JSONL but no .pt "
                "(hidden states missing — those problems are unusable)"
            )
        if r["n_pt_alignment_problems"]:
            warnings.append(
                f"{name}: {r['n_pt_alignment_problems']} .pt/JSONL alignment failures"
            )
        levels = set(r.get("correct_rate_by_level", {}))
        if r["complete_pairs"] and len(levels) < 5:
            warnings.append(
                f"{name}: only levels {sorted(levels)} are present. The split list is "
                "ordered by level, so an --offset/--limit prefix of the corpus is a "
                "difficulty-biased subset and must not be used for the go/no-go."
            )
        if r.get("truncation_rate", 0) > 5:
            warnings.append(
                f"{name}: truncation rate {r['truncation_rate']}% > 5% — "
                "T labels are censored more than the plan assumed"
            )
        m = section["mc"]
        if m["coverage_pct"] < 99:
            warnings.append(
                f"{name}: MC coverage {m['coverage_pct']}% "
                f"({m['problem_files']} problem files)"
            )
        if m["labelled_states"] and m["v_mc_saturated_pct"] > 80:
            warnings.append(
                f"{name}: {m['v_mc_saturated_pct']}% of MC states have v_mc in "
                "{0,1} — V has little headroom, expect weak V ranking"
            )
    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("\nno warnings — corpus looks complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
