#!/usr/bin/env python3
"""Layer / architecture / feature sweep, selected on sibling metrics.

The plan's rule is that the layer is picked by *within-problem* sibling ranking,
not global MAE, so this ranks runs by the primary sibling T pairwise accuracy and
by the V+T budget-utility gain, and prints both next to the global numbers so a
mismatch is visible.

  python scripts/sweep_probes.py --cache artifacts/cache/phase0 --preset full
  python scripts/sweep_probes.py --cache artifacts/cache/phase0 --preset layers
  python scripts/sweep_probes.py --cache artifacts/cache/phase0 --preset controls

Writes <out>/sweep.json, <out>/sweep.csv and one subdirectory per run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.config import ARTIFACTS_DIR, PROBE_LAYER_INDICES
from tokenaware.probes.cache import SPLIT_VAL, load_cache
from tokenaware.probes.evaluate import DEFAULT_BUDGETS, evaluate_report
from tokenaware.probes.train import TrainConfig, dump_history, save_probe, train_probe


def preset_configs(name: str, base: TrainConfig, layers: list[int]) -> list[TrainConfig]:
    runs: list[TrainConfig] = []
    if name in ("layers", "full"):
        for layer in layers:
            runs.append(replace(base, layers=[layer]))
        runs.append(replace(base, layers=list(layers)))
    if name in ("arch", "full"):
        best_layer = [layers[len(layers) // 2]]
        runs += [
            replace(base, layers=best_layer, trunk="mlp"),
            replace(base, layers=best_layer, trunk="linear", t_head="point", lr=2e-4),
            replace(base, layers=best_layer, t_head="quantile"),
            replace(base, layers=best_layer, t_head="point"),
            replace(base, layers=best_layer, use_delta=True),
            replace(base, layers=best_layer, trunk="attn", history=6),
        ]
    if name in ("controls", "full"):
        best_layer = [layers[len(layers) // 2]]
        runs += [
            # No hidden state at all: the position-only control.
            replace(base, layers=best_layer, use_hidden=False, use_meta=True),
            # No position features: does the hidden state alone carry T?
            replace(base, layers=best_layer, use_pos=False),
            # MC labels only: is the abundant same-trace corpus helping?
            replace(base, layers=best_layer, use_same_trace=False, epochs=10),
            # Same-trace only: how far can free labels get us?
            replace(base, layers=best_layer, use_mc=False, mc_finetune_epochs=0),
            # Single-task heads, to test whether the joint trunk hurts either task.
            replace(base, layers=best_layer, task="v"),
            replace(base, layers=best_layer, task="t"),
        ]
    if name == "seeds":
        best_layer = [layers[len(layers) // 2]]
        runs += [replace(base, layers=best_layer, seed=s) for s in (0, 1, 2)]
    if not runs:
        raise SystemExit(f"unknown preset {name!r}")
    # De-duplicate while preserving order.
    seen, unique = set(), []
    for cfg in runs:
        key = json.dumps(cfg.to_dict(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cfg)
    return unique


def flatten(cfg: TrainConfig, report: dict, seconds: float) -> dict:
    sibling = report.get("sibling", {})
    primary = sibling.get("primary", {})
    global_metrics = report.get("global", {})
    headline = report.get("headline_vt_vs_v") or {}
    return {
        "tag": cfg.tag(),
        "layers": "+".join(str(x) for x in cfg.layers),
        "trunk": cfg.trunk,
        "t_head": cfg.t_head,
        "task": cfg.task,
        "use_hidden": cfg.use_hidden,
        "use_pos": cfg.use_pos,
        "use_delta": cfg.use_delta,
        "use_same_trace": cfg.use_same_trace,
        "use_mc": cfg.use_mc,
        "seed": cfg.seed,
        "grouping": sibling.get("primary_grouping"),
        "sibling_t_pairwise": primary.get("t_pairwise", {}).get("accuracy"),
        "sibling_t_ceiling": primary.get("t_noise_ceiling", {}).get("accuracy"),
        "sibling_t_position_baseline": primary.get(
            "t_pairwise_position_baseline", {}
        ).get("accuracy"),
        "sibling_v_pairwise": primary.get("v_pairwise", {}).get("accuracy"),
        "sibling_v_ceiling": primary.get("v_noise_ceiling", {}).get("accuracy"),
        "v_auroc": global_metrics.get("v", {}).get("auroc"),
        "v_pr_auc": global_metrics.get("v", {}).get("pr_auc"),
        "v_ece": global_metrics.get("v", {}).get("ece"),
        "t_mae": global_metrics.get("t", {}).get("mae"),
        "t_mae_baseline": global_metrics.get("t", {}).get("mae_median_baseline"),
        "t_mae_reduction_pct": global_metrics.get("t", {}).get("mae_reduction_pct"),
        "t_spearman": global_metrics.get("t", {}).get("spearman"),
        "t_coverage_q0.9": global_metrics.get("t", {}).get("coverage_q0.9"),
        "vt_gain_budget": headline.get("budget"),
        "vt_gain_delta": headline.get("delta"),
        "vt_gain_ci_low": headline.get("ci_low"),
        "vt_win_rate": headline.get("win_rate"),
        "partial_t_utility_given_v": sibling.get(
            "partial_corr_t_utility_given_v_best"
        ),
        "decision": report.get("verdict", {}).get("decision"),
        "params_m": report.get("probe_params_m"),
        "seconds": round(seconds, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(ARTIFACTS_DIR / "cache" / "phase0"))
    p.add_argument("--out", default=str(ARTIFACTS_DIR / "sweeps" / "phase0"))
    p.add_argument(
        "--preset",
        default="full",
        choices=("layers", "arch", "controls", "seeds", "full"),
    )
    p.add_argument(
        "--layers", type=int, nargs="+", default=list(PROBE_LAYER_INDICES)
    )
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--mc-finetune-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    p.add_argument("--no-preload", dest="preload", action="store_false")
    p.add_argument("--save-checkpoints", action="store_true")
    p.set_defaults(preload=True)
    args = p.parse_args()

    base = TrainConfig(
        epochs=args.epochs,
        mc_finetune_epochs=args.mc_finetune_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        preload=args.preload,
    )
    runs = preset_configs(args.preset, base, list(args.layers))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} runs -> {out}")

    # One cache load covers every run; layers are selected per FeatureStore.
    cache = load_cache(args.cache)
    rows: list[dict] = []
    for i, cfg in enumerate(runs, start=1):
        print(f"\n=== [{i}/{len(runs)}] {cfg.tag()} ===", flush=True)
        started = time.monotonic()
        try:
            result = train_probe(cache, cfg)
            report = evaluate_report(
                cache,
                result["probe"],
                result["store"],
                cfg,
                split=SPLIT_VAL,
                budgets=tuple(args.budgets),
                seed=cfg.seed,
            )
        except Exception:  # noqa: BLE001 - one bad config must not kill the sweep
            print(traceback.format_exc(), file=sys.stderr)
            rows.append(
                {
                    "tag": cfg.tag(),
                    "layers": "+".join(str(x) for x in cfg.layers),
                    "decision": "ERROR",
                }
            )
            continue
        run_dir = out / f"{i:02d}_{cfg.tag()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
        dump_history(run_dir / "history.json", result)
        if args.save_checkpoints:
            save_probe(run_dir / "probe.pt", result["probe"], result["store"], cfg)
        row = flatten(cfg, report, time.monotonic() - started)
        rows.append(row)
        print(
            f"  sibling_T={row['sibling_t_pairwise']} "
            f"(ceiling {row['sibling_t_ceiling']}, "
            f"pos-baseline {row['sibling_t_position_baseline']}) "
            f"V_auroc={row['v_auroc']} T_MAE={row['t_mae']} "
            f"VT_gain={row['vt_gain_delta']} decision={row['decision']}",
            flush=True,
        )
        del result

    (out / "sweep.json").write_text(json.dumps(rows, indent=2, default=str))
    fieldnames = sorted({key for row in rows for key in row})
    with (out / "sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def sort_key(row: dict):
        value = row.get("sibling_t_pairwise")
        return value if isinstance(value, (int, float)) else -1.0

    print("\n=== ranked by primary sibling T pairwise accuracy ===")
    for row in sorted(rows, key=sort_key, reverse=True):
        print(
            f"  {str(row.get('sibling_t_pairwise')):>7}  "
            f"ceil={str(row.get('sibling_t_ceiling')):>7}  "
            f"pos={str(row.get('sibling_t_position_baseline')):>7}  "
            f"Vauc={str(row.get('v_auroc')):>7}  "
            f"MAE={str(row.get('t_mae')):>8}  "
            f"VTgain={str(row.get('vt_gain_delta')):>8}  "
            f"{row.get('tag')}"
        )
    print(f"\nwrote {out / 'sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
