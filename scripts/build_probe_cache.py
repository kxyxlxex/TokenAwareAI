#!/usr/bin/env python3
"""Join root rollouts (.jsonl + .pt), MC labels, and sibling branches into one cache.

This is the ETL that makes probe training fast and reproducible. Run once per
corpus version; every later script reads only the cache.

  python scripts/build_probe_cache.py --out artifacts/cache/phase0
  python scripts/build_probe_cache.py --layers 8 35 --out artifacts/cache/l8l35

Probe-val selection:
  * ``--val-source auto`` (default) uses the real 500-problem val corpus when it
    has at least ``--min-val-problems`` MC-labelled problems, and otherwise
    carves a problem-disjoint holdout out of the train corpus by hashing
    problem_id. Either way, no problem appears on both sides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.artifacts import problem_stem
from tokenaware.config import ARTIFACTS_DIR, PROBE_LAYER_INDICES
from tokenaware.data import load_split
from tokenaware.probes.cache import (
    SOURCE_BRANCH,
    SOURCE_ROOT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SUBJECTS,
    CacheWriter,
)


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return []


def holdout_bucket(problem_id: str, salt: str = "probe-val") -> float:
    digest = hashlib.sha1(f"{salt}:{problem_id}".encode()).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF


def offset_steps(row: dict) -> list[tuple[int, dict]]:
    """Steps that have a token offset, paired with their index in ``steps``."""
    return [
        (i, s)
        for i, s in enumerate(row.get("steps") or [])
        if s.get("last_token_offset") is not None
    ]


def discover(artifacts: Path, split_name: str, problems: list[dict]) -> list[dict]:
    """Problems in ``split_name`` that have a complete root pair on disk."""
    root_dir = artifacts / "rollouts" / "root" / split_name
    mc_dir = artifacts / "labels" / "mc" / split_name
    branch_dir = artifacts / "branches" / split_name
    found = []
    for ordinal, problem in enumerate(problems, start=1):
        stem = problem_stem(ordinal, problem["problem_id"])
        jsonl = root_dir / f"{stem}.jsonl"
        pt = root_dir / f"{stem}.pt"
        if not (jsonl.is_file() and pt.is_file()):
            continue
        found.append(
            {
                "ordinal": ordinal,
                "problem": problem,
                "split_name": split_name,
                "root_jsonl": jsonl,
                "root_pt": pt,
                "mc_jsonl": mc_dir / f"{stem}.jsonl",
                "branch_jsonl": branch_dir / f"{stem}.jsonl",
                "branch_pt": branch_dir / f"{stem}.pt",
            }
        )
    return found


def count_rows(entries: list[dict]) -> tuple[int, int, int]:
    """(n_states, n_mc_states, max_mc_k) from JSONL only — no .pt reads."""
    n_states = n_mc = max_k = 0
    for entry in entries:
        for row in read_jsonl(entry["root_jsonl"]):
            n_states += len(offset_steps(row))
        mc_rows = read_jsonl(entry["mc_jsonl"])
        for row in mc_rows:
            n_mc += 1
            max_k = max(max_k, len(row.get("continuations") or []))
        branch_rows = read_jsonl(entry["branch_jsonl"])
        if branch_rows and entry["branch_pt"].is_file():
            n_states += len(branch_rows)
            n_mc += len(branch_rows)
            for row in branch_rows:
                max_k = max(max_k, len(row.get("continuations") or []))
    return n_states, n_mc, max(max_k, 1)


def main() -> int:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default=str(ARTIFACTS_DIR))
    p.add_argument("--out", default=None, help="default <artifacts>/cache/phase0")
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(PROBE_LAYER_INDICES),
        help="HuggingFace layer indices (plan layer - 1). Default 8 17 26 35.",
    )
    p.add_argument("--limit", type=int, default=0, help="cap problems per split")
    p.add_argument(
        "--val-source", choices=("auto", "split", "holdout"), default="auto"
    )
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--min-val-problems", type=int, default=150)
    args = p.parse_args()

    artifacts = Path(args.artifacts).expanduser().resolve()
    out = Path(args.out) if args.out else artifacts / "cache" / "phase0"
    split = load_split(artifacts / "splits" / "math_probe_split.json")

    entries: list[dict] = []
    for split_name in ("train", "val"):
        found = discover(artifacts, split_name, split.get(split_name) or [])
        if args.limit:
            found = found[: args.limit]
        print(f"[{stamp()}] {split_name}: {len(found)} complete root pairs", flush=True)
        entries.extend(found)
    if not entries:
        print("no complete root pairs found; fetch artifacts first", file=sys.stderr)
        return 2

    val_entries = [e for e in entries if e["split_name"] == "val"]
    val_with_mc = sum(1 for e in val_entries if e["mc_jsonl"].is_file())
    if args.val_source == "auto":
        use_real_val = val_with_mc >= args.min_val_problems
    else:
        use_real_val = args.val_source == "split"
    holdout_frac = 0.0 if use_real_val else args.holdout_frac
    print(
        f"[{stamp()}] val corpus: {len(val_entries)} root pairs, "
        f"{val_with_mc} with MC labels -> "
        f"{'using real val split' if use_real_val else f'hashing {holdout_frac:.0%} of train as probe-val'}",
        flush=True,
    )

    n_states, n_mc, max_k = count_rows(entries)
    print(
        f"[{stamp()}] sizing: {n_states} states, {n_mc} MC states, max_k={max_k}, "
        f"{len(args.layers)} layers -> "
        f"{n_states * 4096 * 2 * len(args.layers) / (1 << 30):.2f} GiB hidden",
        flush=True,
    )

    writer = CacheWriter(out, layers=args.layers, hidden_dim=4096, n_states=n_states)
    problems_meta: list[dict] = []
    problem_index: dict[str, int] = {}
    groups: list[str] = []
    group_index: dict[str, int] = {}
    subject_index = {s: i for i, s in enumerate(SUBJECTS)}

    stats = {
        "skipped_pt_unreadable": 0,
        "skipped_pt_misaligned": 0,
        "mc_unjoinable": 0,
        "branch_files_used": 0,
        "states_root": 0,
        "states_branch": 0,
    }
    started = time.monotonic()

    for i, entry in enumerate(entries, start=1):
        problem = entry["problem"]
        pid = problem["problem_id"]
        if pid not in problem_index:
            problem_index[pid] = len(problems_meta)
            problems_meta.append(
                {
                    "problem_id": pid,
                    "subject": problem["subject"],
                    "level": int(problem["level"]),
                    "ordinal": entry["ordinal"],
                    "split_name": entry["split_name"],
                }
            )
        pidx = problem_index[pid]

        if entry["split_name"] == "val" and use_real_val:
            split_id = SPLIT_VAL
        elif entry["split_name"] == "val":
            # Real val corpus is too thin to evaluate on; do not train on it either.
            continue
        elif holdout_frac > 0 and holdout_bucket(pid) < holdout_frac:
            split_id = SPLIT_VAL
        else:
            split_id = SPLIT_TRAIN

        rows = read_jsonl(entry["root_jsonl"])
        try:
            pack = torch.load(entry["root_pt"], map_location="cpu")
        except Exception:  # noqa: BLE001 - a corrupt sidecar must not stop the ETL
            stats["skipped_pt_unreadable"] += 1
            continue
        if not isinstance(pack, list) or len(pack) != len(rows):
            stats["skipped_pt_misaligned"] += 1
            continue

        # (sample_id, step_index) -> global cache row, for the MC join.
        row_lookup: dict[tuple[int, int], int] = {}
        misaligned = False
        for sample_i, row in enumerate(rows):
            steps = offset_steps(row)
            tensors = pack[sample_i]
            if not steps:
                continue
            expected = len(steps)
            if any(
                str(layer) not in tensors
                or tensors[str(layer)].shape[0] != expected
                for layer in args.layers
            ):
                misaligned = True
                break
            hidden = {
                layer: tensors[str(layer)].to(torch.float16).numpy()
                for layer in args.layers
            }
            state_rows = []
            for local_i, (step_index, step) in enumerate(steps):
                offset = int(step["last_token_offset"])
                state_rows.append(
                    {
                        "problem_ord": entry["ordinal"],
                        "problem_idx": pidx,
                        "sample_id": int(row.get("sample_id", sample_i)),
                        "step_index": step_index,
                        "n_steps": len(row.get("steps") or []),
                        "row_in_trace": local_i,
                        "hist_start": 0,  # patched after we know the base row
                        "tokens_so_far": offset + 1,
                        "trace_len": int(row.get("n_tokens", 0)),
                        "t_same": int(step.get("tokens_remaining_this_trace", 0)),
                        "y_same": 1 if row.get("correct") else 0,
                        "trace_truncated": 1 if row.get("truncated") else 0,
                        "level": int(problem["level"]),
                        "subject_idx": subject_index.get(problem["subject"], 0),
                        "problem_chars": len(problem.get("problem", "")),
                        "split": split_id,
                        "source": SOURCE_ROOT,
                        "group_id": -1,
                        "mc_row": -1,
                    }
                )
            base = writer.add_states(state_rows, hidden)
            for local_i, sr in enumerate(state_rows):
                writer.set_state_field(base + local_i, "hist_start", base)
                row_lookup[(sr["sample_id"], sr["step_index"])] = base + local_i
            stats["states_root"] += len(state_rows)
        if misaligned:
            stats["skipped_pt_misaligned"] += 1

        for mc_row in read_jsonl(entry["mc_jsonl"]):
            key = (
                int(mc_row.get("source_sample_id", -1)),
                int(mc_row.get("step_index", -1)),
            )
            state_row = row_lookup.get(key)
            if state_row is None:
                stats["mc_unjoinable"] += 1
                continue
            continuations = mc_row.get("continuations") or []
            mc_index = writer.add_mc(
                {
                    "state_row": state_row,
                    "fraction": float(mc_row.get("fraction", 0.0)),
                    "v_mc": float(mc_row.get("v_mc", 0.0)),
                    "t_mc_mean": float(mc_row.get("t_mc_mean", 0.0)),
                    "mc_k": len(continuations),
                    "n_correct": int(mc_row.get("n_correct", 0)),
                    "n_truncated": int(mc_row.get("n_truncated", 0)),
                    "source_tokens_remaining": int(
                        mc_row.get("source_tokens_remaining", 0)
                    ),
                },
                continuations,
                max_k,
            )
            writer.set_state_mc_row(state_row, mc_index)

        branch_rows = read_jsonl(entry["branch_jsonl"])
        if branch_rows and entry["branch_pt"].is_file():
            try:
                branch_pack = torch.load(entry["branch_pt"], map_location="cpu")
            except Exception:  # noqa: BLE001
                branch_pack = None
            if branch_pack is not None and len(branch_pack) == len(branch_rows):
                stats["branch_files_used"] += 1
                state_rows = []
                hidden_blocks = {layer: [] for layer in args.layers}
                kept = []
                for b_i, b_row in enumerate(branch_rows):
                    tensors = branch_pack[b_i]
                    if any(str(layer) not in tensors for layer in args.layers):
                        continue
                    group_key = str(b_row.get("group_id", f"{pid}:g{b_i}"))
                    if group_key not in group_index:
                        group_index[group_key] = len(groups)
                        groups.append(group_key)
                    for layer in args.layers:
                        vec = tensors[str(layer)].reshape(-1)[:4096]
                        hidden_blocks[layer].append(
                            vec.to(torch.float16).numpy()
                        )
                    state_rows.append(
                        {
                            "problem_ord": entry["ordinal"],
                            "problem_idx": pidx,
                            "sample_id": 100 + int(b_row.get("branch_index", b_i)),
                            "step_index": int(b_row.get("parent_step_index", 0)) + 1,
                            "n_steps": 0,
                            "row_in_trace": 0,
                            "hist_start": 0,
                            "tokens_so_far": int(b_row.get("tokens_so_far", 0)),
                            "trace_len": 0,
                            "t_same": 0,
                            "y_same": 0,
                            "trace_truncated": 0,
                            "level": int(problem["level"]),
                            "subject_idx": subject_index.get(problem["subject"], 0),
                            "problem_chars": len(problem.get("problem", "")),
                            "split": split_id,
                            "source": SOURCE_BRANCH,
                            "group_id": group_index[group_key],
                            "mc_row": -1,
                        }
                    )
                    kept.append(b_row)
                if state_rows:
                    hidden = {
                        layer: np.stack(hidden_blocks[layer]) for layer in args.layers
                    }
                    base = writer.add_states(state_rows, hidden)
                    for local_i in range(len(state_rows)):
                        writer.set_state_field(
                            base + local_i, "hist_start", base + local_i
                        )
                    stats["states_branch"] += len(state_rows)
                    for local_i, b_row in enumerate(kept):
                        continuations = b_row.get("continuations") or []
                        mc_index = writer.add_mc(
                            {
                                "state_row": base + local_i,
                                "fraction": float(
                                    b_row.get("parent_fraction", 0.0)
                                ),
                                "v_mc": float(b_row.get("v_mc", 0.0)),
                                "t_mc_mean": float(b_row.get("t_mc_mean", 0.0)),
                                "mc_k": len(continuations),
                                "n_correct": int(b_row.get("n_correct", 0)),
                                "n_truncated": int(b_row.get("n_truncated", 0)),
                                "source_tokens_remaining": 0,
                            },
                            continuations,
                            max_k,
                        )
                        writer.set_state_mc_row(base + local_i, mc_index)

        if i % 100 == 0 or i == len(entries):
            print(
                f"[{stamp()}] {i}/{len(entries)} problems, "
                f"{writer.cursor} states written "
                f"({time.monotonic() - started:.0f}s)",
                flush=True,
            )

    writer.finalize(
        {
            "built_at": stamp(),
            "artifacts_dir": str(artifacts),
            "subjects": list(SUBJECTS),
            "problems": problems_meta,
            "groups": groups,
            "max_mc_k": max_k,
            "val_source": "split" if use_real_val else "holdout",
            "holdout_frac": holdout_frac,
            "stats": stats,
        }
    )
    print(f"[{stamp()}] wrote cache to {out}")
    print(json.dumps(stats, indent=2))
    if stats["skipped_pt_misaligned"] or stats["skipped_pt_unreadable"]:
        print(
            "WARNING: some .pt sidecars were skipped; run "
            "scripts/inventory_artifacts.py --deep to see which",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
