#!/usr/bin/env python3
"""Train one V/T probe on a built cache and write its evaluation report.

  python scripts/train_probe.py --cache artifacts/cache/phase0 --task joint \
      --trunk star --t-head dist --layers 26 --out artifacts/probes/joint_l26

  # the linear T diagnostic from *How Much is Left?*
  python scripts/train_probe.py --cache artifacts/cache/phase0 --task t \
      --trunk linear --t-head point --no-pos --layers 35 --lr 2e-4

  # position-only control: does the hidden state add anything at all?
  python scripts/train_probe.py --cache artifacts/cache/phase0 --task joint \
      --no-hidden --use-meta
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.config import ARTIFACTS_DIR
from tokenaware.probes.cache import SPLIT_VAL, load_cache
from tokenaware.probes.evaluate import DEFAULT_BUDGETS, evaluate_report
from tokenaware.probes.heads import TRUNK_NAMES
from tokenaware.probes.train import (
    TrainConfig,
    dump_history,
    save_probe,
    train_probe,
)


def add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--layers", type=int, nargs="+", default=[26])
    p.add_argument("--no-hidden", dest="use_hidden", action="store_false")
    p.add_argument("--delta", dest="use_delta", action="store_true")
    p.add_argument("--no-pos", dest="use_pos", action="store_false")
    p.add_argument("--use-meta", dest="use_meta", action="store_true")
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--task", choices=("v", "t", "joint"), default="joint")
    p.add_argument("--trunk", choices=TRUNK_NAMES, default="star")
    p.add_argument("--t-head", choices=("dist", "quantile", "point"), default="dist")
    p.add_argument("--n-bins", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--mc-finetune-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--pos-weight", type=float, default=3.0)
    p.add_argument("--aux-l1-weight", type=float, default=0.0)
    p.add_argument("--no-same-trace", dest="use_same_trace", action="store_false")
    p.add_argument("--no-mc", dest="use_mc", action="store_false")
    p.add_argument("--mc-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-preload", dest="preload", action="store_false")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--patience", type=int, default=3)
    p.set_defaults(
        use_hidden=True,
        use_delta=False,
        use_pos=True,
        use_meta=False,
        use_same_trace=True,
        use_mc=True,
        preload=True,
        amp=True,
    )


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        layers=list(args.layers),
        use_hidden=args.use_hidden,
        use_delta=args.use_delta,
        use_pos=args.use_pos,
        use_meta=args.use_meta,
        history=args.history,
        task=args.task,
        trunk=args.trunk,
        t_head=args.t_head,
        n_bins=args.n_bins,
        max_tokens=args.max_tokens,
        dropout=args.dropout,
        epochs=args.epochs,
        mc_finetune_epochs=args.mc_finetune_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        aux_l1_weight=args.aux_l1_weight,
        use_same_trace=args.use_same_trace,
        use_mc=args.use_mc,
        mc_weight=args.mc_weight,
        seed=args.seed,
        device=args.device,
        preload=args.preload,
        amp=args.amp,
        patience=args.patience,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(ARTIFACTS_DIR / "cache" / "phase0"))
    p.add_argument("--out", default=None, help="default <artifacts>/probes/<tag>")
    p.add_argument(
        "--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS)
    )
    add_config_args(p)
    args = p.parse_args()

    cfg = config_from_args(args)
    cache = load_cache(args.cache, layers=cfg.layers if cfg.use_hidden else [])
    out = Path(args.out) if args.out else ARTIFACTS_DIR / "probes" / cfg.tag()
    out.mkdir(parents=True, exist_ok=True)

    print(f"cache={args.cache} states={cache.n_states} mc={cache.n_mc}")
    print(f"config={cfg.tag()}")
    result = train_probe(cache, cfg)
    save_probe(out / "probe.pt", result["probe"], result["store"], cfg)
    dump_history(out / "history.json", result)

    report = evaluate_report(
        cache,
        result["probe"],
        result["store"],
        cfg,
        split=SPLIT_VAL,
        budgets=tuple(args.budgets),
        seed=cfg.seed,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    verdict = report.get("verdict", {})
    print(f"\nwrote {out}")
    print(f"decision: {verdict.get('decision')} — {verdict.get('reason')}")
    if "global" in report:
        print(json.dumps(report["global"], indent=2, default=str))
    if report.get("headline_vt_vs_v"):
        print("headline V+T vs V-only:")
        print(json.dumps(report["headline_vt_vs_v"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
