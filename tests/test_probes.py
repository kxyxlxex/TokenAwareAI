"""Unit tests for the probe cache, metrics, and training loop.

The cache / metrics tests are pure numpy. The training tests need torch and are
skipped when it is missing, so this file runs on a laptop without CUDA wheels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.probes import metrics
from tokenaware.probes.cache import (
    SOURCE_BRANCH,
    SOURCE_ROOT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    CacheWriter,
    load_cache,
)
from tokenaware.probes.features import build_draws, build_eval_states


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #
def test_rankdata_averages_ties():
    ranks = metrics.rankdata(np.array([10.0, 20.0, 20.0, 40.0]))
    assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_is_one_for_monotone_transform():
    x = np.arange(1, 21, dtype=float)
    assert metrics.spearman(x, np.exp(x / 5.0)) == pytest.approx(1.0, abs=1e-9)
    assert metrics.spearman(x, -x) == pytest.approx(-1.0, abs=1e-9)


def test_weighted_auroc_perfect_and_chance():
    scores = np.array([0.1, 0.9])
    assert metrics.weighted_auroc(scores, np.array([0, 4]), np.array([4, 0])) == 1.0
    assert metrics.weighted_auroc(scores, np.array([4, 0]), np.array([0, 4])) == 0.0
    tied = np.array([0.5, 0.5])
    assert metrics.weighted_auroc(tied, np.array([2, 2]), np.array([2, 2])) == 0.5


def test_weighted_auroc_matches_unweighted_definition():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=40)
    scores = rng.normal(size=40) + labels
    n_pos = labels.astype(float)
    n_neg = 1.0 - n_pos
    manual = 0.0
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    for a in pos_scores:
        for b in neg_scores:
            manual += 1.0 if a > b else (0.5 if a == b else 0.0)
    manual /= pos_scores.size * neg_scores.size
    assert metrics.weighted_auroc(scores, n_pos, n_neg) == pytest.approx(manual)


def test_average_precision_perfect_ranking():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    n_pos = np.array([1, 1, 0, 0], dtype=float)
    n_neg = np.array([0, 0, 1, 1], dtype=float)
    assert metrics.weighted_average_precision(scores, n_pos, n_neg) == pytest.approx(
        1.0
    )


def test_ece_is_zero_when_calibrated():
    prob = np.array([0.0, 0.5, 1.0])
    n_total = np.array([10.0, 10.0, 10.0])
    n_pos = np.array([0.0, 5.0, 10.0])
    assert metrics.expected_calibration_error(prob, n_pos, n_total) == pytest.approx(
        0.0, abs=1e-9
    )


# --------------------------------------------------------------------------- #
# sibling metrics
# --------------------------------------------------------------------------- #
def test_pairwise_ranking_accuracy_directions():
    groups = [np.array([0, 1]), np.array([2, 3])]
    truth = np.array([1.0, 2.0, 5.0, 3.0])
    perfect = np.array([1.0, 2.0, 5.0, 3.0])
    inverted = -perfect
    assert metrics.pairwise_ranking_accuracy(groups, perfect, truth)["accuracy"] == 1.0
    assert metrics.pairwise_ranking_accuracy(groups, inverted, truth)["accuracy"] == 0.0
    constant = np.zeros(4)
    assert metrics.pairwise_ranking_accuracy(groups, constant, truth)["accuracy"] == 0.5


def test_pairwise_ranking_excludes_tied_truth():
    groups = [np.array([0, 1])]
    result = metrics.pairwise_ranking_accuracy(
        groups, np.array([0.0, 1.0]), np.array([3.0, 3.0])
    )
    assert result["n_pairs"] == 0
    assert np.isnan(result["accuracy"])


def test_group_indices_drops_singletons():
    groups = metrics.group_indices(np.array([1, 1, 2, 3, 3, 3]))
    sizes = sorted(g.shape[0] for g in groups)
    assert sizes == [2, 3]


def test_split_half_ceiling_is_high_for_separable_states_and_chance_for_identical():
    valid = np.ones((4, 8), dtype=bool)
    separable = np.stack(
        [np.full(8, 10.0), np.full(8, 500.0), np.full(8, 20.0), np.full(8, 400.0)]
    )
    groups = [np.array([0, 1]), np.array([2, 3])]
    result = metrics.split_half_ceiling(groups, separable, valid, n_repeats=4)
    assert result["accuracy"] == pytest.approx(1.0)

    rng = np.random.default_rng(0)
    identical = rng.normal(100.0, 50.0, size=(40, 8))
    many_groups = [np.array([i, i + 1]) for i in range(0, 40, 2)]
    noisy = metrics.split_half_ceiling(many_groups, identical, valid.repeat(10, axis=0), n_repeats=8)
    assert 0.3 < noisy["accuracy"] < 0.7


def test_partial_spearman_removes_the_confound():
    rng = np.random.default_rng(1)
    z = rng.normal(size=200)
    x = z + 0.01 * rng.normal(size=200)
    y = z + 0.01 * rng.normal(size=200)
    assert metrics.spearman(x, y) > 0.9
    assert abs(metrics.partial_spearman(x, y, z)) < 0.4


# --------------------------------------------------------------------------- #
# budget utility
# --------------------------------------------------------------------------- #
def test_budget_utility_counts_only_correct_and_within_budget():
    draw_len = np.array([[10, 200, 10, 200]])
    draw_correct = np.array([[1, 1, 0, 0]])
    draw_valid = np.ones((1, 4), dtype=bool)
    assert metrics.budget_utility(draw_len, draw_correct, draw_valid, 100)[
        0
    ] == pytest.approx(0.25)
    assert metrics.budget_utility(draw_len, draw_correct, draw_valid, 1000)[
        0
    ] == pytest.approx(0.5)


def test_selection_comparison_rewards_the_cost_aware_score():
    # Two siblings: A is more likely correct but always overruns a tight budget;
    # B is a bit less likely correct but always finishes cheaply.
    draw_len = np.array([[400, 400, 400, 400], [50, 50, 50, 50]])
    draw_correct = np.array([[1, 1, 1, 1], [1, 1, 1, 0]])
    draw_valid = np.ones((2, 4), dtype=bool)
    utility = metrics.budget_utility(draw_len, draw_correct, draw_valid, 100.0)
    assert utility.tolist() == [0.0, 0.75]

    v_pred = np.array([1.0, 0.75])
    feasibility = np.array([0.0, 1.0])
    result = metrics.selection_comparison(
        [np.array([0, 1])],
        {"v_only": v_pred, "v_times_feasibility": v_pred * feasibility},
        utility,
    )
    assert result["selectors"]["v_only"]["utility"] == 0.0
    assert result["selectors"]["v_times_feasibility"]["utility"] == 0.75
    assert result["selectors"]["v_times_feasibility"]["delta_vs_v_only"] == 0.75
    assert result["oracle_utility"] == 0.75


def test_kill_verdict_paths():
    passing = {
        "sibling": {
            "primary": {
                "t_pairwise": {"accuracy": 0.68},
                "t_noise_ceiling": {"accuracy": 0.82},
                "t_pairwise_position_baseline": {"accuracy": 0.58},
            },
            "partial_corr_t_utility_given_v_best": 0.2,
        },
        "global": {
            "v": {"auroc": 0.72},
            "t": {"mae_reduction_pct": 21.0, "spearman": 0.45},
        },
        "headline_vt_vs_v": {"budget": "128", "delta": 0.03, "ci_low": 0.01},
    }
    assert metrics.kill_verdict(passing)["decision"] == "PROCEED_TO_SEARCH"

    killed = json.loads(json.dumps(passing))
    killed["sibling"]["primary"]["t_pairwise"]["accuracy"] = 0.52
    assert metrics.kill_verdict(killed)["decision"] == "KILL"

    soft = json.loads(json.dumps(passing))
    soft["global"]["v"]["auroc"] = 0.51
    assert metrics.kill_verdict(soft)["decision"] == "FIX_PROBE"

    position = json.loads(json.dumps(passing))
    position["sibling"]["primary"]["t_pairwise_position_baseline"]["accuracy"] = 0.70
    assert metrics.kill_verdict(position)["decision"] == "KILL"


# --------------------------------------------------------------------------- #
# cache round-trip
# --------------------------------------------------------------------------- #
def synthetic_cache(
    tmp_path: Path,
    n_problems: int = 12,
    n_traces: int = 4,
    n_steps: int = 3,
    hidden_dim: int = 16,
    layers: tuple[int, ...] = (8, 35),
    seed: int = 0,
) -> Path:
    """A tiny cache whose hidden states genuinely encode V and T.

    Dimension 0 carries the remaining-length signal and dimension 1 carries the
    correctness signal, so a working probe must recover both.
    """
    rng = np.random.default_rng(seed)
    n_states = n_problems * n_traces * n_steps
    out = tmp_path / "cache"
    writer = CacheWriter(out, list(layers), hidden_dim, n_states)
    problems = []
    max_k = 4
    for problem in range(n_problems):
        problems.append(
            {
                "problem_id": f"algebra:{problem:04d}",
                "subject": "algebra",
                "level": 1 + problem % 5,
                "ordinal": problem + 1,
                "split_name": "train",
            }
        )
        split = SPLIT_VAL if problem % 3 == 0 else SPLIT_TRAIN
        for trace in range(n_traces):
            trace_len = 60 + 40 * trace
            correct = int((problem + trace) % 2 == 0)
            rows, blocks = [], {layer: [] for layer in layers}
            for step in range(n_steps):
                tokens_so_far = 10 + step * 15
                remaining = max(trace_len - tokens_so_far, 1)
                vector = rng.normal(0.0, 0.2, size=hidden_dim).astype(np.float32)
                vector[0] = np.log1p(remaining)
                vector[1] = 2.0 * correct - 1.0
                for layer in layers:
                    blocks[layer].append(vector)
                rows.append(
                    {
                        "problem_ord": problem + 1,
                        "problem_idx": problem,
                        "sample_id": trace,
                        "step_index": step,
                        "n_steps": n_steps,
                        "row_in_trace": step,
                        "hist_start": 0,
                        "tokens_so_far": tokens_so_far,
                        "trace_len": trace_len,
                        "t_same": remaining,
                        "y_same": correct,
                        "trace_truncated": 0,
                        "level": 1 + problem % 5,
                        "subject_idx": 0,
                        "problem_chars": 120,
                        "split": split,
                        "source": SOURCE_ROOT,
                        "group_id": -1,
                        "mc_row": -1,
                    }
                )
            base = writer.add_states(
                rows, {layer: np.stack(blocks[layer]) for layer in layers}
            )
            for offset in range(len(rows)):
                writer.set_state_field(base + offset, "hist_start", base)
            # MC-label the middle step of every trace.
            middle = base + n_steps // 2
            remaining = rows[n_steps // 2]["t_same"]
            continuations = [
                {
                    "n_tokens": int(max(remaining + rng.integers(-5, 6), 1)),
                    "correct": correct,
                    "truncated": False,
                }
                for _ in range(max_k)
            ]
            mc_index = writer.add_mc(
                {
                    "state_row": middle,
                    "fraction": 0.5,
                    "v_mc": float(correct),
                    "t_mc_mean": float(
                        np.mean([c["n_tokens"] for c in continuations])
                    ),
                    "mc_k": max_k,
                    "n_correct": max_k * correct,
                    "n_truncated": 0,
                    "source_tokens_remaining": int(remaining),
                },
                continuations,
                max_k,
            )
            writer.set_state_mc_row(middle, mc_index)
    writer.finalize(
        {
            "built_at": "test",
            "subjects": ["algebra"],
            "problems": problems,
            "groups": [],
            "max_mc_k": max_k,
            "val_source": "holdout",
        }
    )
    return out


def test_cache_round_trip(tmp_path):
    path = synthetic_cache(tmp_path)
    cache = load_cache(path)
    assert cache.n_states == 12 * 4 * 3
    assert cache.n_mc == 12 * 4
    assert cache.hidden_dim == 16
    assert cache.layers == [8, 35]
    assert cache.hidden[8].shape == (cache.n_states, 16)
    assert set(np.unique(cache.states["split"])) == {SPLIT_TRAIN, SPLIT_VAL}

    train_problems = set(
        cache.states["problem_idx"][cache.split_mask(SPLIT_TRAIN)].tolist()
    )
    val_problems = set(
        cache.states["problem_idx"][cache.split_mask(SPLIT_VAL)].tolist()
    )
    assert train_problems.isdisjoint(val_problems)


def test_build_draws_expands_mc_into_one_row_per_continuation(tmp_path):
    cache = load_cache(synthetic_cache(tmp_path))
    same_only = build_draws(cache, SPLIT_TRAIN, use_same_trace=True, use_mc=False)
    mc_only = build_draws(cache, SPLIT_TRAIN, use_same_trace=False, use_mc=True)
    both = build_draws(cache, SPLIT_TRAIN, use_same_trace=True, use_mc=True)

    n_train_states = int(cache.split_mask(SPLIT_TRAIN).sum())
    n_train_mc = cache.mc_rows_for_split(SPLIT_TRAIN).shape[0]
    assert len(same_only) == n_train_states
    assert len(mc_only) == n_train_mc * 4
    assert len(both) == len(same_only) + len(mc_only)
    assert set(np.unique(both.origin).tolist()) == {0, 1}
    assert (both.length > 0).all()


def test_eval_states_only_contain_the_requested_split(tmp_path):
    cache = load_cache(synthetic_cache(tmp_path))
    val = build_eval_states(cache, SPLIT_VAL)
    assert len(val) > 0
    assert (cache.states["split"][val.rows] == SPLIT_VAL).all()
    assert val.draw_len.shape[1] == cache.meta["max_mc_k"]


def test_writer_rejects_overflow(tmp_path):
    writer = CacheWriter(tmp_path / "small", [8], 4, n_states=1)
    row = {key: 0 for key in ("problem_ord",)}
    with pytest.raises(ValueError):
        writer.add_states([row, row], {8: np.zeros((2, 4), dtype=np.float16)})


# --------------------------------------------------------------------------- #
# torch-dependent
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch", reason="torch is not installed")


def test_bin_edges_and_indexing_are_monotone():
    from tokenaware.probes.heads import bin_of, make_bin_edges

    edges = make_bin_edges(1024, 16)
    assert edges.shape[0] == 16
    assert torch.isinf(edges[-1])
    assert float(edges[-2]) == 1024.0
    assert bool((edges[:-1].diff() > 0).all())
    lengths = torch.tensor([0.0, 5.0, 100.0, 1023.0, 5000.0])
    bins = bin_of(lengths, edges)
    assert bool((bins.diff() >= 0).all())
    # A trace at the cap lands in the last finite bin; only impossible lengths
    # reach the overflow bin.
    assert int(bin_of(torch.tensor([1024.0]), edges)[0]) == 14
    assert int(bins[-1]) == 15


def test_censored_nll_prefers_the_tail_for_censored_draws():
    from tokenaware.probes.losses import t_censored_nll

    # Mass concentrated on the last bin: cheap for a censored draw, costly for
    # an observed draw in bin 0.
    logits = torch.tensor([[0.0, 0.0, 10.0]])
    bins = torch.tensor([2])
    censored_loss = t_censored_nll(logits, bins, torch.tensor([1.0]))
    observed_loss = t_censored_nll(logits, torch.tensor([0]), torch.tensor([0.0]))
    assert float(censored_loss) < float(observed_loss)


def test_quantile_head_outputs_are_monotone_in_q():
    from tokenaware.probes.heads import Probe, ProbeConfig

    probe = Probe(ProbeConfig(task="t", trunk="mlp", d_in=8, t_head="quantile"))
    out = probe(torch.randn(16, 8))
    q = out["t_quantiles"]
    assert bool((q.diff(dim=1) >= 0).all())


def test_dist_head_cdf_is_monotone_and_capped_by_termination_mass():
    from tokenaware.probes.heads import Probe, ProbeConfig

    probe = Probe(
        ProbeConfig(task="t", trunk="mlp", d_in=8, t_head="dist", n_bins=12, max_tokens=1024)
    )
    out = probe(torch.randn(4, 8))
    previous = torch.zeros(4)
    for budget in (16.0, 64.0, 256.0, 1024.0, 4096.0):
        current = probe.t_cdf_at(out, torch.full((4,), budget))
        assert bool((current >= previous - 1e-6).all())
        assert bool((current <= 1.0 + 1e-6).all())
        previous = current
    # Beyond the cap the CDF equals the probability of terminating at all, i.e.
    # everything except the overflow bin.
    finite_mass = probe.t_pmf(out)[:, :-1].sum(dim=1)
    assert torch.allclose(previous, finite_mass, atol=1e-6)


def test_train_probe_recovers_planted_v_and_t_signal(tmp_path):
    from tokenaware.probes.evaluate import evaluate_report, predict_states
    from tokenaware.probes.train import TrainConfig, train_probe

    cache = load_cache(synthetic_cache(tmp_path, n_problems=40, seed=3))
    cfg = TrainConfig(
        layers=[8],
        trunk="mlp",
        task="joint",
        t_head="dist",
        n_bins=16,
        epochs=12,
        mc_finetune_epochs=2,
        batch_size=128,
        lr=3e-3,
        device="cpu",
        amp=False,
        patience=0,
        use_pos=False,
    )
    result = train_probe(cache, cfg, verbose=False)
    eval_states = build_eval_states(cache, SPLIT_VAL)
    pred = predict_states(result["probe"], result["store"], eval_states.rows)

    assert metrics.spearman(pred["t_mean"], eval_states.t_mc_mean) > 0.5
    correct = eval_states.v_mc > 0.5
    assert pred["v_prob"][correct].mean() > pred["v_prob"][~correct].mean()

    report = evaluate_report(
        cache, result["probe"], result["store"], cfg, split=SPLIT_VAL
    )
    assert report["global"]["t"]["mae"] < report["global"]["t"]["mae_median_baseline"]
    assert "decision" in report["verdict"]


def test_probe_save_load_round_trip(tmp_path):
    from tokenaware.probes.evaluate import predict_states
    from tokenaware.probes.train import TrainConfig, load_probe, save_probe, train_probe

    cache = load_cache(synthetic_cache(tmp_path, n_problems=8, seed=5))
    cfg = TrainConfig(
        layers=[8],
        trunk="mlp",
        epochs=2,
        mc_finetune_epochs=0,
        batch_size=64,
        device="cpu",
        amp=False,
    )
    result = train_probe(cache, cfg, verbose=False)
    rows = np.arange(min(20, cache.n_states))
    before = predict_states(result["probe"], result["store"], rows)

    path = save_probe(tmp_path / "probe.pt", result["probe"], result["store"], cfg)
    probe, store, loaded_cfg = load_probe(path, cache, device="cpu")
    after = predict_states(probe, store, rows)

    assert loaded_cfg.trunk == "mlp"
    assert np.allclose(before["v_prob"], after["v_prob"], atol=1e-5)
    assert np.allclose(before["t_mean"], after["t_mean"], atol=1e-4)


def test_attn_trunk_consumes_step_history(tmp_path):
    from tokenaware.probes.features import FeatureSpec, FeatureStore

    cache = load_cache(synthetic_cache(tmp_path, n_problems=6))
    spec = FeatureSpec(layers=[8], history=3, use_pos=True)
    store = FeatureStore(cache, spec, device="cpu")
    rows = torch.arange(6)
    x, mask = store.history(rows)
    assert x.shape == (6, 3, store.d_in)
    # The first step of a trace has no history, so its slots are all padding.
    assert bool(mask[0, 0])
    assert not bool(mask[0, -1])


def test_delta_feature_doubles_hidden_width(tmp_path):
    from tokenaware.probes.features import FeatureSpec, FeatureStore

    cache = load_cache(synthetic_cache(tmp_path, n_problems=6))
    plain = FeatureStore(cache, FeatureSpec(layers=[8], use_pos=False), device="cpu")
    delta = FeatureStore(
        cache, FeatureSpec(layers=[8], use_delta=True, use_pos=False), device="cpu"
    )
    assert delta.d_in == 2 * plain.d_in
    x = delta.raw(torch.tensor([0]))
    # Step 0 has no predecessor, so the delta block must be exactly zero.
    assert torch.allclose(x[0, plain.d_in :], torch.zeros(plain.d_in))
