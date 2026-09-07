#!/usr/bin/env python3
"""Score the Phase 0 V head on thinking sibling states. CPU is enough.

  python scripts/build_thinking_cache.py
  python scripts/score_old_v_on_thinking.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.config import ARTIFACTS_DIR
from tokenaware.probes.cache import load_cache
from tokenaware.probes.evaluate import predict_states
from tokenaware.probes.metrics import (
    group_indices,
    pairwise_ranking_accuracy,
    spearman,
    v_global_metrics,
)
from tokenaware.probes.train import load_probe

DEFAULT_PROBES = [
    ARTIFACTS_DIR / "sweeps" / "phase0_best" / "01_joint_mlp_dist_L26-pos" / "probe.pt",
    ARTIFACTS_DIR / "probes" / "phase0_best" / "01_joint_mlp_dist_L26-pos" / "probe.pt",
]


def find_probe(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"no probe at {path}")
        return path
    for path in DEFAULT_PROBES:
        if path.is_file():
            return path
    matches = sorted(ARTIFACTS_DIR.glob("**/01_joint_mlp_dist_L26-pos/probe.pt"))
    if matches:
        return matches[0]
    raise SystemExit(
        f"no Phase 0 L26 probe under {ARTIFACTS_DIR}; pass --probe"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache",
        default=str(ARTIFACTS_DIR / "cache" / "thinking_pilot"),
    )
    p.add_argument("--probe", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cache_dir = Path(args.cache)
    if not (cache_dir / "states.npz").is_file():
        print(f"missing cache {cache_dir}; run build_thinking_cache.py", file=sys.stderr)
        return 2

    probe_path = find_probe(args.probe)
    cache = load_cache(cache_dir, layers=[26])
    rows = np.arange(cache.n_states, dtype=np.int64)
    probe, store, _ = load_probe(probe_path, cache, device=args.device, preload=True)
    pred = predict_states(probe, store, rows)
    v_hat = pred["v_prob"]
    if v_hat is None:
        print("probe has no V head", file=sys.stderr)
        return 2

    mc_idx = cache.states["mc_row"]
    n_ok = cache.mc["n_correct"][mc_idx]
    k = cache.mc["mc_k"][mc_idx]
    v_mc = cache.mc["v_mc"][mc_idx]
    t_mc = cache.mc["t_mc_mean"][mc_idx]
    groups = group_indices(cache.states["group_id"])
    report = {
        "probe": str(probe_path),
        "n_states": int(cache.n_states),
        "n_groups": len(groups),
        "global_v": v_global_metrics(v_hat, n_ok, k),
        "spearman_vhat_vs_t": round(spearman(v_hat, t_mc), 4),
        "spearman_vmc_vs_t": round(spearman(v_mc, t_mc), 4),
        "sibling_v_pairwise": pairwise_ranking_accuracy(groups, v_hat, v_mc),
        "mean_vhat_by_vmc": {
            f"v_mc={x:.2f}": round(float(v_hat[np.isclose(v_mc, x)].mean()), 4)
            for x in sorted(set(np.round(v_mc, 2)))
            if np.isclose(v_mc, x).sum()
        },
    }
    print(json.dumps(report, indent=2))
    auroc = report["global_v"]["auroc"]
    print(
        f"\nAUROC={auroc}  (pooled, leaky MATH-train ids, n={cache.n_states}). "
        ">=0.70 is a problem-level gate, not sibling ranking.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
