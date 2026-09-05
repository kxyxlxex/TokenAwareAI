#!/usr/bin/env python3
"""Re-evaluate saved probes and print the Phase 0 go/no-go verdict.

  python scripts/evaluate_probes.py --cache artifacts/cache/phase0 \
      --probe artifacts/probes/joint_star_dist_L26-pos/probe.pt

  # every checkpoint under a directory, on a wider budget grid
  python scripts/evaluate_probes.py --cache artifacts/cache/phase0 \
      --probe-dir artifacts/probes --budgets 32 64 128 256 512 1024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.config import ARTIFACTS_DIR
from tokenaware.probes.cache import SPLIT_TRAIN, SPLIT_VAL, load_cache
from tokenaware.probes.evaluate import DEFAULT_BUDGETS, evaluate_report
from tokenaware.probes.train import load_probe


def summarize(report: dict) -> str:
    lines = []
    sibling = report.get("sibling", {})
    primary = sibling.get("primary", {})
    lines.append(
        f"grouping={sibling.get('primary_grouping')} "
        f"groups={primary.get('n_groups')} states={report.get('n_eval_states')} "
        f"problems={report.get('n_problems')}"
    )
    t_pair = primary.get("t_pairwise", {})
    ceiling = primary.get("t_noise_ceiling", {})
    pos = primary.get("t_pairwise_position_baseline", {})
    lines.append(
        f"sibling T pairwise = {t_pair.get('accuracy')} "
        f"(CI {t_pair.get('group_ci95')}, n={t_pair.get('n_pairs')} pairs) | "
        f"label-noise ceiling {ceiling.get('accuracy')} | "
        f"position-only baseline {pos.get('accuracy')}"
    )
    v_pair = primary.get("v_pairwise", {})
    lines.append(
        f"sibling V pairwise = {v_pair.get('accuracy')} | "
        f"ceiling {primary.get('v_noise_ceiling', {}).get('accuracy')}"
    )
    g = report.get("global", {})
    if "v" in g:
        lines.append(
            f"global V: AUROC {g['v'].get('auroc')} PR-AUC {g['v'].get('pr_auc')} "
            f"ECE {g['v'].get('ece')} (position-only AUROC "
            f"{g.get('v_position_baseline', {}).get('auroc')})"
        )
    if "t" in g:
        lines.append(
            f"global T: MAE {g['t'].get('mae')} vs median baseline "
            f"{g['t'].get('mae_median_baseline')} "
            f"({g['t'].get('mae_reduction_pct')}% cut), rho {g['t'].get('spearman')}, "
            f"q90 coverage {g['t'].get('coverage_q0.9')}"
        )
    for budget, section in (report.get("budget_utility") or {}).items():
        selectors = section.get("selectors", {})
        vt = selectors.get("v_times_feasibility", {})
        v_only = selectors.get("v_only", {})
        lines.append(
            f"B={budget}: u(V only)={v_only.get('utility')} "
            f"u(V*P(T<=B))={vt.get('utility')} "
            f"delta={vt.get('delta_vs_v_only')} "
            f"CI {vt.get('delta_ci95_vs_v_only')} "
            f"win-rate {vt.get('win_rate')} | oracle {section.get('oracle_utility')}"
        )
    verdict = report.get("verdict", {})
    lines.append(f"DECISION: {verdict.get('decision')} — {verdict.get('reason')}")
    for check in verdict.get("checks", []):
        flag = {True: "pass", False: "FAIL", None: "n/a"}[check["passed"]]
        lines.append(
            f"  [{check['threshold_kind']:>4}] {flag:<4} {check['name']} = "
            f"{check['value']}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(ARTIFACTS_DIR / "cache" / "phase0"))
    p.add_argument("--probe", action="append", default=None)
    p.add_argument("--probe-dir", default=None)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    p.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="restrict evaluation to these MATH levels, e.g. --levels 4 5",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="directory for report JSON files")
    p.add_argument("--no-preload", dest="preload", action="store_false")
    p.set_defaults(preload=True)
    args = p.parse_args()
    levels = tuple(sorted(set(args.levels))) if args.levels else None
    suffix = f"_L{''.join(str(x) for x in levels)}" if levels else ""

    paths = [Path(x) for x in (args.probe or [])]
    if args.probe_dir:
        paths += sorted(Path(args.probe_dir).rglob("probe.pt"))
    if not paths:
        print("pass --probe or --probe-dir", file=sys.stderr)
        return 2

    cache = load_cache(args.cache)
    split = SPLIT_VAL if args.split == "val" else SPLIT_TRAIN
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for path in paths:
        print(f"\n=== {path} ===")
        probe, store, cfg = load_probe(
            path, cache, device=args.device, preload=args.preload
        )
        report = evaluate_report(
            cache,
            probe,
            store,
            cfg,
            split=split,
            budgets=tuple(args.budgets),
            seed=cfg.seed,
            levels=levels,
        )
        if "error" in report:
            print(report["error"])
            del probe, store
            continue
        if levels:
            print(f"levels={list(levels)} counts={report.get('level_counts')}")
        print(summarize(report))
        target = (out_dir / f"{path.parent.name}{suffix}.json") if out_dir else (
            path.parent / f"report_{args.split}{suffix}.json"
        )
        target.write_text(json.dumps(report, indent=2, default=str))
        summary.append(
            {
                "probe": str(path),
                "tag": cfg.tag(),
                "levels": list(levels) if levels else None,
                "decision": report.get("verdict", {}).get("decision"),
                "sibling_t_pairwise": report.get("sibling", {})
                .get("primary", {})
                .get("t_pairwise", {})
                .get("accuracy"),
                "headline_vt_vs_v": report.get("headline_vt_vs_v"),
            }
        )
        del probe, store

    if out_dir:
        (out_dir / f"summary{suffix}.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
