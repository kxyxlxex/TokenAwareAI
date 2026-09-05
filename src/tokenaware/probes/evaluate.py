"""Turn a trained probe into the Phase 0 decision report."""

from __future__ import annotations

import numpy as np
import torch

from .cache import SOURCE_BRANCH, SOURCE_ROOT, SPLIT_TRAIN, SPLIT_VAL, ProbeCache
from .features import EvalStates, FeatureStore, build_eval_states
from .heads import Probe
from .metrics import (
    budget_utility,
    group_indices,
    kill_verdict,
    mean_within_group_spearman,
    pairwise_ranking_accuracy,
    partial_spearman,
    selection_comparison,
    split_half_ceiling,
    t_global_metrics,
    v_global_metrics,
)
from .train import TrainConfig, forward_batch

DEFAULT_BUDGETS = (64, 128, 256, 512)
# Minimum |ΔT̂| in tokens for a sibling pair to count as decidable.
T_GAP_THRESHOLDS = (20, 50)


@torch.no_grad()
def predict_states(
    probe: Probe,
    store: FeatureStore,
    rows: np.ndarray,
    batch_size: int = 4096,
) -> dict:
    """Batched probe outputs for cache rows, as numpy arrays."""
    device = store.device
    v_prob, t_mean, t_pmf, t_quantiles = [], [], [], []
    for start in range(0, rows.shape[0], batch_size):
        chunk = torch.from_numpy(rows[start : start + batch_size].astype(np.int64)).to(
            device
        )
        out = forward_batch(probe, store, chunk)
        if "v_logit" in out:
            v_prob.append(torch.sigmoid(out["v_logit"]).float().cpu().numpy())
        if probe.t_head is not None:
            t_mean.append(probe.t_mean(out).float().cpu().numpy())
            if probe.config.t_head == "dist":
                t_pmf.append(probe.t_pmf(out).float().cpu().numpy())
            elif probe.config.t_head == "quantile":
                q = out["t_quantiles"]
                if probe.config.log_target:
                    q = torch.expm1(q)
                t_quantiles.append(q.float().cpu().numpy())
    result: dict = {"n": int(rows.shape[0])}
    result["v_prob"] = np.concatenate(v_prob) if v_prob else None
    result["t_mean"] = np.concatenate(t_mean) if t_mean else None
    result["t_pmf"] = np.concatenate(t_pmf) if t_pmf else None
    result["t_quantiles"] = np.concatenate(t_quantiles) if t_quantiles else None
    result["bin_edges"] = probe.bin_edges.detach().cpu().numpy()
    result["quantiles"] = list(probe.config.quantiles)
    result["t_head"] = probe.config.t_head
    return result


def cdf_at(pred: dict, budget: float) -> np.ndarray:
    """``P(T <= budget)`` per state, from whichever T head is present."""
    if pred.get("t_pmf") is not None:
        included = pred["bin_edges"] <= budget
        return pred["t_pmf"][:, included].sum(axis=1)
    if pred.get("t_quantiles") is not None:
        q = pred["t_quantiles"]
        lo, hi = q[:, 0], q[:, -1]
        spread = np.maximum(np.abs(hi - lo), 1.0)
        mid = q[:, q.shape[1] // 2]
        return 1.0 / (1.0 + np.exp(-(budget - mid) / (spread / 2.197)))
    if pred.get("t_mean") is not None:
        mean = pred["t_mean"]
        spread = np.maximum(np.abs(mean) * 0.5, 1.0)
        return 1.0 / (1.0 + np.exp(-(budget - mean) / (spread / 2.197)))
    return np.ones(pred["n"], dtype=np.float64)


def quantile_from_pmf(pred: dict, q: float) -> np.ndarray:
    """Interpolate the ``q`` quantile inside the bin that crosses it.

    Returning the bin's upper edge instead systematically over-covers, which
    would make the quantile-calibration check look better than it is.
    """
    edges = pred["bin_edges"].copy()
    edges[-1] = edges[-2] * 2 if edges.shape[0] > 1 else 1.0
    lower = np.concatenate([[0.0], edges[:-1]])
    cdf = np.cumsum(pred["t_pmf"], axis=1)
    n_bins = edges.shape[0]
    idx = np.where(
        (cdf >= q).any(axis=1), np.argmax(cdf >= q, axis=1), n_bins - 1
    )
    rows = np.arange(idx.shape[0])
    cdf_below = np.where(idx > 0, cdf[rows, np.maximum(idx - 1, 0)], 0.0)
    mass = np.maximum(cdf[rows, idx] - cdf_below, 1e-12)
    within = np.clip((q - cdf_below) / mass, 0.0, 1.0)
    return lower[idx] + within * (edges[idx] - lower[idx])


# --------------------------------------------------------------------------- #
# baselines that the probe has to beat
# --------------------------------------------------------------------------- #
class PositionBaseline:
    """Predicts V and T from prefix position alone: no hidden state.

    ``tokens_so_far`` is strongly predictive of remaining length, so a probe that
    only matches this has learned position, not reasoning state. Fitted as a
    binned mean on the train split, which is nonparametric and cannot be beaten
    by a smarter functional form.
    """

    def __init__(self, n_bins: int = 24) -> None:
        self.n_bins = n_bins
        self.edges: np.ndarray | None = None
        self.t_by_bin: np.ndarray | None = None
        self.v_by_bin: np.ndarray | None = None
        self.t_global = 0.0
        self.v_global = 0.0

    def fit(self, tokens_so_far: np.ndarray, t_true: np.ndarray, v_true: np.ndarray):
        if tokens_so_far.shape[0] == 0:
            return self
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        self.edges = np.unique(np.quantile(tokens_so_far, quantiles))
        index = np.clip(
            np.searchsorted(self.edges, tokens_so_far, side="right") - 1,
            0,
            self.edges.shape[0] - 1,
        )
        n_slots = self.edges.shape[0]
        self.t_global = float(np.median(t_true))
        self.v_global = float(v_true.mean())
        self.t_by_bin = np.full(n_slots, self.t_global)
        self.v_by_bin = np.full(n_slots, self.v_global)
        for b in range(n_slots):
            mask = index == b
            if mask.sum() >= 10:
                self.t_by_bin[b] = float(np.mean(t_true[mask]))
                self.v_by_bin[b] = float(np.mean(v_true[mask]))
        return self

    def predict(self, tokens_so_far: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.edges is None:
            n = tokens_so_far.shape[0]
            return np.full(n, self.t_global), np.full(n, self.v_global)
        index = np.clip(
            np.searchsorted(self.edges, tokens_so_far, side="right") - 1,
            0,
            self.edges.shape[0] - 1,
        )
        return self.t_by_bin[index], self.v_by_bin[index]


def fit_position_baseline(cache: ProbeCache) -> PositionBaseline:
    """Fit on same-trace train states: one draw each, but ~8x more of them.

    Same-trace realised outcomes are unbiased samples of the same quantities the
    MC labels estimate, and there are tens of thousands of them, so the binned
    means are far better resolved than they would be on the MC subset alone.
    A weak baseline would flatter the probe.
    """
    states = cache.states
    rows = np.nonzero(
        cache.split_mask(SPLIT_TRAIN) & (states["source"] == SOURCE_ROOT)
    )[0]
    if rows.shape[0] >= 200:
        return PositionBaseline().fit(
            states["tokens_so_far"][rows].astype(np.float64),
            states["t_same"][rows].astype(np.float64),
            states["y_same"][rows].astype(np.float64),
        )
    train = build_eval_states(cache, SPLIT_TRAIN)
    if len(train) == 0:
        return PositionBaseline()
    return PositionBaseline().fit(
        train.tokens_so_far.astype(np.float64), train.t_mc_mean, train.v_mc
    )


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #
def sibling_groupings(eval_states: EvalStates) -> dict[str, list[np.ndarray]]:
    """Candidate definitions of "sibling", best first.

    ``branch`` groups share a literal parent prefix and are the only definition
    that matches what tree search actually does. ``problem_fraction`` pairs the
    two independent traces at the same depth fraction and is the best proxy
    available from the root/MC corpus alone. ``problem`` pools all depths and is
    the loosest.
    """
    groupings: dict[str, list[np.ndarray]] = {}
    branch_mask = eval_states.source == SOURCE_BRANCH
    if branch_mask.sum() >= 4:
        keys = np.where(
            branch_mask, eval_states.group_id, -1 - eval_states.problem_idx
        )
        branch_rows = np.nonzero(branch_mask)[0]
        sub_keys = keys[branch_rows]
        groups = []
        for rows in group_indices(sub_keys):
            groups.append(branch_rows[rows])
        if groups:
            groupings["branch"] = groups
    fraction_keys = np.stack(
        [eval_states.problem_idx, np.round(eval_states.fraction * 100).astype(np.int64)],
        axis=1,
    )
    groupings["problem_fraction"] = group_indices(
        [tuple(row) for row in fraction_keys]
    )
    groupings["problem"] = group_indices(eval_states.problem_idx)
    return groupings


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def evaluate_report(
    cache: ProbeCache,
    probe: Probe,
    store: FeatureStore,
    cfg: TrainConfig,
    split: int = SPLIT_VAL,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    seed: int = 0,
) -> dict:
    eval_states = build_eval_states(cache, split)
    if len(eval_states) == 0:
        return {"error": "no Monte-Carlo-labelled states in this split"}

    pred = predict_states(probe, store, eval_states.rows)
    v_pred = (
        pred["v_prob"]
        if pred["v_prob"] is not None
        else np.full(len(eval_states), 0.5)
    )
    t_pred = (
        pred["t_mean"]
        if pred["t_mean"] is not None
        else np.full(len(eval_states), np.nan)
    )

    baseline = fit_position_baseline(cache)
    t_base, v_base = baseline.predict(eval_states.tokens_so_far.astype(np.float64))

    train_states = build_eval_states(cache, SPLIT_TRAIN)
    train_median = (
        float(np.median(train_states.t_mc_mean)) if len(train_states) else 0.0
    )

    n_correct = (eval_states.draw_correct * eval_states.draw_valid).sum(axis=1)
    mc_k = eval_states.draw_valid.sum(axis=1)

    report: dict = {
        "split": "val" if split == SPLIT_VAL else "train",
        "config": cfg.to_dict(),
        "probe_params_m": round(probe.n_params / 1e6, 3),
        "n_eval_states": len(eval_states),
        "n_problems": int(np.unique(eval_states.problem_idx).shape[0]),
        "budgets": list(budgets),
        "global": {},
        "sibling": {},
        "budget_utility": {},
    }

    report["global"]["v"] = v_global_metrics(v_pred, n_correct, mc_k)
    report["global"]["v_position_baseline"] = v_global_metrics(
        v_base, n_correct, mc_k
    )

    quantile_preds = None
    if pred["t_pmf"] is not None:
        quantile_preds = {q: quantile_from_pmf(pred, q) for q in (0.5, 0.9)}
    elif pred["t_quantiles"] is not None:
        quantile_preds = {
            q: pred["t_quantiles"][:, i] for i, q in enumerate(pred["quantiles"])
        }
    if not np.all(np.isnan(t_pred)):
        report["global"]["t"] = t_global_metrics(
            t_pred,
            eval_states.t_mc_mean,
            train_median,
            quantile_preds,
            eval_states.draw_len,
            eval_states.draw_valid,
        )
        report["global"]["t_position_baseline"] = t_global_metrics(
            t_base, eval_states.t_mc_mean, train_median
        )

    groupings = sibling_groupings(eval_states)
    primary_name = "branch" if "branch" in groupings else "problem_fraction"
    report["sibling"]["primary_grouping"] = primary_name
    report["sibling"]["grouping_sizes"] = {
        name: len(groups) for name, groups in groupings.items()
    }

    for name, groups in groupings.items():
        section: dict = {"n_groups": len(groups)}
        if not np.all(np.isnan(t_pred)):
            section["t_pairwise"] = pairwise_ranking_accuracy(
                groups, -t_pred, -eval_states.t_mc_mean
            )
            section["t_pairwise_position_baseline"] = pairwise_ranking_accuracy(
                groups, -t_base, -eval_states.t_mc_mean
            )
            section["t_noise_ceiling"] = split_half_ceiling(
                groups,
                eval_states.draw_len.astype(np.float64),
                eval_states.draw_valid,
                seed=seed,
            )
            # Pairs whose empirical remaining lengths barely differ are decided
            # by k=8 label noise, not by the probe. Restricting to clearly
            # separated pairs says whether the probe gets the *decidable* cases
            # right, which is what a selector needs.
            for gap in T_GAP_THRESHOLDS:
                section[f"t_pairwise_gap{gap}"] = pairwise_ranking_accuracy(
                    groups, -t_pred, -eval_states.t_mc_mean, min_gap=float(gap)
                )
                section[f"t_noise_ceiling_gap{gap}"] = split_half_ceiling(
                    groups,
                    eval_states.draw_len.astype(np.float64),
                    eval_states.draw_valid,
                    seed=seed,
                    min_gap=float(gap),
                )
                section[f"t_pairwise_position_baseline_gap{gap}"] = (
                    pairwise_ranking_accuracy(
                        groups,
                        -t_base,
                        -eval_states.t_mc_mean,
                        min_gap=float(gap),
                    )
                )
            section["t_within_group_spearman"] = mean_within_group_spearman(
                groups, t_pred, eval_states.t_mc_mean
            )
        section["v_pairwise"] = pairwise_ranking_accuracy(
            groups, v_pred, eval_states.v_mc
        )
        section["v_noise_ceiling"] = split_half_ceiling(
            groups,
            eval_states.draw_correct.astype(np.float64),
            eval_states.draw_valid,
            seed=seed,
        )
        report["sibling"][name] = section
    report["sibling"]["primary"] = report["sibling"][primary_name]

    report["sibling"]["partial_corr_t_outcome_given_v"] = round(
        partial_spearman(-t_pred, eval_states.v_mc, v_pred), 4
    )

    primary_groups = groupings[primary_name]
    utility_partials = {}
    eps = 1e-6
    for budget in budgets:
        utility = budget_utility(
            eval_states.draw_len,
            eval_states.draw_correct,
            eval_states.draw_valid,
            float(budget),
        )
        feasibility = cdf_at(pred, float(budget))
        scores = {
            "v_only": v_pred,
            "v_times_feasibility": v_pred * feasibility,
            "bang_per_buck": v_pred / (np.nan_to_num(t_pred, nan=1e6) + 1.0),
            "feasibility_only": feasibility,
            "position_baseline": v_base
            * (
                1.0
                / (
                    1.0
                    + np.exp(-(budget - t_base) / np.maximum(t_base * 0.5, 1.0) * 2.197)
                )
            ),
            "oracle_v_mc": eval_states.v_mc,
        }
        comparison = selection_comparison(
            primary_groups, scores, utility, reference="v_only", seed=seed
        )
        comparison["mean_utility_all_states"] = round(float(utility.mean()), 4)
        report["budget_utility"][str(budget)] = comparison
        utility_partials[str(budget)] = round(
            partial_spearman(-t_pred, utility, v_pred), 4
        )

    report["sibling"]["partial_corr_t_utility_given_v"] = utility_partials
    finite = [v for v in utility_partials.values() if not np.isnan(v)]
    report["sibling"]["partial_corr_t_utility_given_v_best"] = (
        round(float(max(finite, key=abs)), 4) if finite else float("nan")
    )

    best_delta = None
    for budget, section in report["budget_utility"].items():
        entry = section.get("selectors", {}).get("v_times_feasibility", {})
        ci = entry.get("delta_ci95_vs_v_only")
        if not ci or np.isnan(ci[0]):
            continue
        if best_delta is None or ci[0] > best_delta["ci_low"]:
            best_delta = {
                "budget": budget,
                "delta": entry.get("delta_vs_v_only"),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "win_rate": entry.get("win_rate"),
            }
    report["headline_vt_vs_v"] = best_delta
    report["verdict"] = kill_verdict(report)
    return report
