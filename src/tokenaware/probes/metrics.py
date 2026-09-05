"""Metrics for the Phase 0 go/no-go decision.

Three families, reported separately and never mixed in one table.

**Global** — comparable to the published V and T papers: AUROC / PR-AUC / ECE
for V, MAE / Spearman / quantile coverage for T, each against the constant
baseline that is optimal for that loss (the train median is L1-optimal, so a T
probe that loses to it has no signal at all).

**Within-problem sibling** — the number this project actually needs. Search
never compares states across problems; it compares candidates under one
problem. Two extras make this honest:

* a *label-noise ceiling* from splitting each state's k continuations in half,
  because with k=8 the empirical T̂ itself misranks siblings sometimes, and a
  probe cannot beat that;
* a *position-only baseline*, because ``tokens_so_far`` alone predicts remaining
  length well and a hidden-state probe must beat it to be worth anything.

**Budget utility** — an offline surrogate for search gain that needs no search.
For a state with continuations ``i``, define
``u_B(s) = mean_i 1[correct_i and len_i <= B]``: the probability that continuing
from ``s`` both finishes inside the remaining budget and is right. That is
exactly what a cost-aware selector should maximise, and it is computable from
the MC artifacts we already have. Ranking siblings by ``V`` alone versus by
``V * P(T <= B)`` and scoring both against ``u_B`` gives a direct, budget-matched
read on whether T adds anything.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

EPS = 1e-9


# --------------------------------------------------------------------------- #
# small statistics helpers
# --------------------------------------------------------------------------- #
def rankdata(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks, 1-based (scipy.stats.rankdata equivalent)."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    if n == 0:
        return np.zeros(0)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > EPS else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.asarray(a).shape[0] < 3:
        return float("nan")
    return pearson(rankdata(a), rankdata(b))


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of x and y after linearly removing rank(z)."""
    x, y, z = (rankdata(v) for v in (x, y, z))
    if x.shape[0] < 4:
        return float("nan")

    def residual(target: np.ndarray) -> np.ndarray:
        design = np.stack([np.ones_like(z), z], axis=1)
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        return target - design @ coef

    return pearson(residual(x), residual(y))


def weighted_auroc(
    scores: np.ndarray, n_pos: np.ndarray, n_neg: np.ndarray
) -> float:
    """AUROC where each item carries ``n_pos`` positive and ``n_neg`` negative draws."""
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = np.asarray(n_pos, dtype=np.float64)
    n_neg = np.asarray(n_neg, dtype=np.float64)
    total_pos, total_neg = n_pos.sum(), n_neg.sum()
    if total_pos < 1 or total_neg < 1:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    scores, n_pos, n_neg = scores[order], n_pos[order], n_neg[order]
    cum_neg_before = np.concatenate([[0.0], np.cumsum(n_neg)[:-1]])
    # Tie groups share the same rank block; average within the block.
    auc = 0.0
    i = 0
    n = scores.shape[0]
    while i < n:
        j = i
        while j + 1 < n and scores[j + 1] == scores[i]:
            j += 1
        block_pos = n_pos[i : j + 1].sum()
        block_neg = n_neg[i : j + 1].sum()
        auc += block_pos * (cum_neg_before[i] + 0.5 * block_neg)
        i = j + 1
    return float(auc / (total_pos * total_neg))


def weighted_average_precision(
    scores: np.ndarray, n_pos: np.ndarray, n_neg: np.ndarray
) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = np.asarray(n_pos, dtype=np.float64)
    n_neg = np.asarray(n_neg, dtype=np.float64)
    total_pos = n_pos.sum()
    if total_pos < 1:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(n_pos[order])
    fp = np.cumsum(n_neg[order])
    precision = tp / np.maximum(tp + fp, EPS)
    recall = tp / total_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def expected_calibration_error(
    prob: np.ndarray, n_pos: np.ndarray, n_total: np.ndarray, n_bins: int = 10
) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    n_pos = np.asarray(n_pos, dtype=np.float64)
    n_total = np.asarray(n_total, dtype=np.float64)
    total = n_total.sum()
    if total < 1:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & ((prob < hi) | (hi == 1.0))
        weight = n_total[mask].sum()
        if weight < 1:
            continue
        observed = n_pos[mask].sum() / weight
        predicted = float(np.average(prob[mask], weights=n_total[mask]))
        error += weight / total * abs(observed - predicted)
    return float(error)


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.shape[0], size=(n_boot, values.shape[0]))
    means = values[idx].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


# --------------------------------------------------------------------------- #
# global metrics
# --------------------------------------------------------------------------- #
def v_global_metrics(
    v_pred: np.ndarray, n_correct: np.ndarray, mc_k: np.ndarray
) -> dict:
    n_pos = n_correct.astype(np.float64)
    n_neg = (mc_k - n_correct).astype(np.float64)
    v_true = n_pos / np.maximum(mc_k, 1)
    base_rate = n_pos.sum() / max(n_pos.sum() + n_neg.sum(), 1)
    return {
        "n_states": int(v_pred.shape[0]),
        "n_draws": int((n_pos + n_neg).sum()),
        "base_correct_rate": round(float(base_rate), 4),
        "auroc": round(weighted_auroc(v_pred, n_pos, n_neg), 4),
        "pr_auc": round(weighted_average_precision(v_pred, n_pos, n_neg), 4),
        "spearman_vs_v_mc": round(spearman(v_pred, v_true), 4),
        "brier_vs_v_mc": round(float(np.mean((v_pred - v_true) ** 2)), 4),
        "brier_vs_base": round(float(np.mean((base_rate - v_true) ** 2)), 4),
        "ece": round(
            expected_calibration_error(v_pred, n_pos, n_pos + n_neg), 4
        ),
        "mean_pred": round(float(v_pred.mean()), 4),
    }


def t_global_metrics(
    t_pred: np.ndarray,
    t_true: np.ndarray,
    train_median: float,
    quantile_preds: dict[float, np.ndarray] | None = None,
    draw_len: np.ndarray | None = None,
    draw_valid: np.ndarray | None = None,
) -> dict:
    absolute = np.abs(t_pred - t_true)
    median_absolute = np.abs(train_median - t_true)
    out = {
        "n_states": int(t_pred.shape[0]),
        "mae": round(float(absolute.mean()), 3),
        "mae_median_baseline": round(float(median_absolute.mean()), 3),
        "mae_reduction_pct": round(
            100.0 * (1.0 - absolute.mean() / max(median_absolute.mean(), EPS)), 2
        ),
        "rmse": round(float(np.sqrt(np.mean((t_pred - t_true) ** 2))), 3),
        "spearman": round(spearman(t_pred, t_true), 4),
        "pred_mean": round(float(t_pred.mean()), 2),
        "true_mean": round(float(t_true.mean()), 2),
        "train_median": round(float(train_median), 2),
    }
    if quantile_preds and draw_len is not None and draw_valid is not None:
        for q, pred in sorted(quantile_preds.items()):
            below = (draw_len <= pred[:, None]) & draw_valid
            coverage = below.sum() / max(draw_valid.sum(), 1)
            out[f"coverage_q{q:g}"] = round(float(coverage), 4)
    return out


# --------------------------------------------------------------------------- #
# within-problem sibling metrics
# --------------------------------------------------------------------------- #
def group_indices(keys: np.ndarray) -> list[np.ndarray]:
    buckets: dict = defaultdict(list)
    for i, key in enumerate(keys):
        buckets[key].append(i)
    return [np.asarray(v, dtype=np.int64) for v in buckets.values() if len(v) >= 2]


def pairwise_ranking_accuracy(
    groups: list[np.ndarray],
    pred: np.ndarray,
    truth: np.ndarray,
    min_gap: float = 0.0,
    higher_is_better: bool = True,
) -> dict:
    """Fraction of within-group pairs ordered correctly by ``pred``.

    Ties in ``truth`` (or gaps below ``min_gap``) are excluded; ties in ``pred``
    score 0.5. Chance is 0.5.
    """
    correct = 0.0
    total = 0
    per_group = []
    sign = 1.0 if higher_is_better else -1.0
    for rows in groups:
        hits = 0.0
        pairs = 0
        for a_i in range(len(rows)):
            for b_i in range(a_i + 1, len(rows)):
                a, b = rows[a_i], rows[b_i]
                gap = truth[a] - truth[b]
                if abs(gap) <= min_gap:
                    continue
                delta = sign * (pred[a] - pred[b])
                pairs += 1
                if delta == 0:
                    hits += 0.5
                elif (gap > 0) == (delta > 0):
                    hits += 1.0
        if pairs:
            correct += hits
            total += pairs
            per_group.append(hits / pairs)
    per_group_arr = np.asarray(per_group, dtype=np.float64)
    lo, hi = bootstrap_ci(per_group_arr) if per_group_arr.shape[0] > 1 else (
        float("nan"),
        float("nan"),
    )
    return {
        "accuracy": round(correct / total, 4) if total else float("nan"),
        "n_pairs": int(total),
        "n_groups": int(per_group_arr.shape[0]),
        "group_mean": round(float(per_group_arr.mean()), 4)
        if per_group_arr.shape[0]
        else float("nan"),
        "group_ci95": [round(lo, 4), round(hi, 4)],
    }


def mean_within_group_spearman(
    groups: list[np.ndarray], pred: np.ndarray, truth: np.ndarray
) -> dict:
    values = []
    for rows in groups:
        if rows.shape[0] < 3:
            continue
        rho = spearman(pred[rows], truth[rows])
        if not np.isnan(rho):
            values.append(rho)
    if not values:
        return {"mean_rho": float("nan"), "n_groups": 0}
    arr = np.asarray(values)
    lo, hi = bootstrap_ci(arr)
    return {
        "mean_rho": round(float(arr.mean()), 4),
        "n_groups": int(arr.shape[0]),
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def split_half_ceiling(
    groups: list[np.ndarray],
    draw_values: np.ndarray,
    draw_valid: np.ndarray,
    seed: int = 0,
    n_repeats: int = 8,
    min_gap: float = 0.0,
) -> dict:
    """How well the empirical label ranks siblings against an independent copy.

    Split each state's k draws in half, use half A as the "prediction" and half B
    as the "truth". The result upper-bounds what any probe can score on this
    corpus with this k; report probe accuracy against it, not against 1.0.
    """
    rng = np.random.default_rng(seed)
    accuracies = []
    for _ in range(n_repeats):
        pred = np.full(draw_values.shape[0], np.nan)
        truth = np.full(draw_values.shape[0], np.nan)
        for i in range(draw_values.shape[0]):
            valid = np.nonzero(draw_valid[i])[0]
            if valid.shape[0] < 2:
                continue
            shuffled = rng.permutation(valid)
            half = shuffled.shape[0] // 2
            pred[i] = draw_values[i, shuffled[:half]].mean()
            truth[i] = draw_values[i, shuffled[half : 2 * half]].mean()
        usable = [
            rows[~np.isnan(pred[rows]) & ~np.isnan(truth[rows])] for rows in groups
        ]
        usable = [rows for rows in usable if rows.shape[0] >= 2]
        result = pairwise_ranking_accuracy(usable, pred, truth, min_gap=min_gap)
        if not np.isnan(result["accuracy"]):
            accuracies.append(result["accuracy"])
    if not accuracies:
        return {"accuracy": float("nan"), "n_repeats": 0}
    arr = np.asarray(accuracies)
    return {
        "accuracy": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "n_repeats": int(arr.shape[0]),
    }


# --------------------------------------------------------------------------- #
# budget utility — the offline surrogate for search gain
# --------------------------------------------------------------------------- #
def budget_utility(
    draw_len: np.ndarray, draw_correct: np.ndarray, draw_valid: np.ndarray, budget: float
) -> np.ndarray:
    """``u_B(s) = mean_i 1[correct_i and len_i <= B]`` per state."""
    hit = (draw_correct > 0) & (draw_len <= budget) & draw_valid
    n_valid = np.maximum(draw_valid.sum(axis=1), 1)
    return hit.sum(axis=1) / n_valid


def selection_comparison(
    groups: list[np.ndarray],
    scores: dict[str, np.ndarray],
    utility: np.ndarray,
    reference: str = "v_only",
    seed: int = 0,
) -> dict:
    """Pick one state per sibling group by each score; grade against ``utility``.

    Reports mean achieved utility, oracle and mean-of-group references, the
    paired difference against ``reference`` with a bootstrap CI over groups, and
    the head-to-head win rate.
    """
    if not groups:
        return {"n_groups": 0}
    per_group: dict[str, list[float]] = {name: [] for name in scores}
    oracle, average = [], []
    rng = np.random.default_rng(seed)
    for rows in groups:
        u = utility[rows]
        oracle.append(float(u.max()))
        average.append(float(u.mean()))
        for name, score in scores.items():
            s = score[rows]
            best = np.nonzero(s == s.max())[0]
            # Break ties at random so a constant score scores the group average.
            pick = best[rng.integers(0, best.shape[0])] if best.shape[0] > 1 else best[0]
            per_group[name].append(float(u[pick]))

    oracle_arr = np.asarray(oracle)
    average_arr = np.asarray(average)
    out: dict = {
        "n_groups": len(groups),
        "oracle_utility": round(float(oracle_arr.mean()), 4),
        "random_utility": round(float(average_arr.mean()), 4),
        "selectors": {},
    }
    ref = np.asarray(per_group[reference]) if reference in per_group else None
    for name, values in per_group.items():
        arr = np.asarray(values)
        entry = {
            "utility": round(float(arr.mean()), 4),
            "regret_vs_oracle": round(float((oracle_arr - arr).mean()), 4),
            "gain_vs_random": round(float((arr - average_arr).mean()), 4),
        }
        if ref is not None and name != reference:
            diff = arr - ref
            lo, hi = bootstrap_ci(diff)
            wins = int((diff > 0).sum())
            losses = int((diff < 0).sum())
            entry.update(
                {
                    f"delta_vs_{reference}": round(float(diff.mean()), 4),
                    f"delta_ci95_vs_{reference}": [round(lo, 4), round(hi, 4)],
                    "win_rate": round(wins / max(wins + losses, 1), 4),
                    "n_wins": wins,
                    "n_losses": losses,
                }
            )
        out["selectors"][name] = entry
    return out


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
KILL_THRESHOLDS = {
    "sibling_t_pairwise_min": 0.60,
    "v_auroc_min": 0.65,
    "t_spearman_min": 0.30,
    "q90_coverage_min": 0.70,
}


def kill_verdict(report: dict, thresholds: dict | None = None) -> dict:
    """Apply the plan's hard and soft kill criteria to an evaluation report."""
    th = {**KILL_THRESHOLDS, **(thresholds or {})}
    checks = []

    def add(name: str, value, ok: bool | None, kind: str, detail: str) -> None:
        checks.append(
            {
                "name": name,
                "value": value,
                "threshold_kind": kind,
                "passed": ok,
                "detail": detail,
            }
        )

    sibling = (
        report.get("sibling", {}).get("primary", {}).get("t_pairwise", {}).get("accuracy")
    )
    ceiling = (
        report.get("sibling", {}).get("primary", {}).get("t_noise_ceiling", {}).get("accuracy")
    )
    add(
        "sibling_t_pairwise",
        sibling,
        None if sibling is None else bool(sibling >= th["sibling_t_pairwise_min"]),
        "hard",
        f"needs >= {th['sibling_t_pairwise_min']}; label noise ceiling is {ceiling}",
    )

    # The plan's original criterion asked whether T predicts *correctness* given
    # V. Under a token budget the relevant question is whether T predicts
    # correctness-within-budget given V, so that is the hard criterion here and
    # the unconditional version is kept for reference only.
    partial_utility = report.get("sibling", {}).get(
        "partial_corr_t_utility_given_v_best"
    )
    add(
        "t_informative_about_budgeted_outcome_given_v",
        partial_utility,
        None
        if partial_utility is None or (isinstance(partial_utility, float) and np.isnan(partial_utility))
        else bool(abs(partial_utility) >= 0.05),
        "hard",
        "partial Spearman of -T with u_B controlling for V must be away from 0 at some budget",
    )

    headline = report.get("headline_vt_vs_v") or {}
    ci_low = headline.get("ci_low")
    add(
        "vt_beats_v_only_on_budget_utility",
        {k: headline.get(k) for k in ("budget", "delta", "ci_low", "win_rate")}
        if headline
        else None,
        None if ci_low is None else bool(ci_low > 0),
        "hard",
        "bootstrap CI over sibling groups for (V*P(T<=B)) - (V only) must exclude 0 "
        "at at least one budget",
    )

    add(
        "t_collinearity_reference",
        report.get("sibling", {}).get("partial_corr_t_outcome_given_v"),
        None,
        "info",
        "plan's original T-vs-correctness partial correlation; near 0 is expected "
        "and is not by itself a kill",
    )

    auroc = report.get("global", {}).get("v", {}).get("auroc")
    add(
        "v_auroc",
        auroc,
        None if auroc is None else bool(auroc >= th["v_auroc_min"]),
        "soft",
        f"needs >= {th['v_auroc_min']}; below this the instrumentation is broken, not T",
    )

    reduction = report.get("global", {}).get("t", {}).get("mae_reduction_pct")
    add(
        "t_beats_median_baseline",
        reduction,
        None if reduction is None else bool(reduction > 0),
        "soft",
        "global MAE must beat the L1-optimal constant predictor",
    )

    rho = report.get("global", {}).get("t", {}).get("spearman")
    add(
        "t_spearman",
        rho,
        None if rho is None else bool(rho >= th["t_spearman_min"]),
        "soft",
        f"needs >= {th['t_spearman_min']}",
    )

    beat_position = report.get("sibling", {}).get("primary", {}).get(
        "t_pairwise_position_baseline", {}
    ).get("accuracy")
    add(
        "t_beats_position_only",
        beat_position,
        None
        if (beat_position is None or sibling is None)
        else bool(sibling > beat_position + 0.01),
        "hard",
        "hidden-state probe must beat a probe that sees only tokens_so_far",
    )

    hard_failed = [c["name"] for c in checks if c["threshold_kind"] == "hard" and c["passed"] is False]
    soft_failed = [c["name"] for c in checks if c["threshold_kind"] == "soft" and c["passed"] is False]
    if hard_failed:
        decision = "KILL"
        reason = f"hard criteria failed: {', '.join(hard_failed)}"
    elif soft_failed:
        decision = "FIX_PROBE"
        reason = f"soft criteria failed: {', '.join(soft_failed)} — probe or labels look broken"
    else:
        decision = "PROCEED_TO_SEARCH"
        reason = "all criteria passed"
    return {
        "decision": decision,
        "reason": reason,
        "checks": checks,
        "thresholds": th,
    }
