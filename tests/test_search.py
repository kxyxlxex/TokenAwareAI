"""Search state machine, policies and baselines against a stub generator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from tokenaware.search import (  # noqa: E402
    MIN_CHAIN_TOKENS,
    ChainRun,
    SearchConfig,
    SearchRun,
    annealing_alpha,
    majority_answer,
    policy_scores,
    run_chain_baselines,
    run_tree_searches,
    selection_probabilities,
    summarize_records,
)


# --------------------------------------------------------------------------- #
# stubs
# --------------------------------------------------------------------------- #
class StubGenerator:
    """Deterministic policy: ``width`` distinct steps, boxed answer at ``answer_step``.

    Candidate ``j`` of step ``k`` is ``"- Step k: opt j\\n"`` costing ``5 + j``
    tokens. Candidate 0 is the "good" one; the stub probe knows that.
    """

    def __init__(self, answer_step: int = 3, gold: str = "42", wrong_from: int | None = None):
        self.answer_step = answer_step
        self.gold = gold
        self.wrong_from = wrong_from  # candidate index that leads to a wrong answer
        self.calls = {"steps": 0, "vectors": 0, "full": 0}

    def _step_index(self, prefix_ids):
        return sum(1 for t in prefix_ids if t == 0)  # one 0-token per accepted step

    def sample_steps(self, requests):
        self.calls["steps"] += 1
        out = []
        counters: dict[int, int] = {}
        for req in requests:
            key = id(req["prefix_ids"]) if False else len(req["prefix_ids"])
            j = counters.get(key, 0)
            counters[key] = j + 1
            k = self._step_index(req["prefix_ids"])
            cap = req["max_new_tokens"]
            if k >= self.answer_step:
                n = min(cap, 4)
                text = f"\\boxed{{{self.gold}}}\n" if not self._went_wrong(req["prefix_ids"]) else "\\boxed{0}\n"
                ids = [0] + [1] * (n - 1)
                out.append({"step_ids": ids, "text": text, "finished": True, "n_tokens": n})
            else:
                n = min(cap, 5 + j)
                ids = [0] + [10 + j] * (n - 1)
                out.append(
                    {
                        "step_ids": ids,
                        "text": f"- Step {k}: opt {j}\n",
                        "finished": False,
                        "n_tokens": n,
                    }
                )
        return out

    def _went_wrong(self, prefix_ids):
        return self.wrong_from is not None and (10 + self.wrong_from) in prefix_ids

    def state_vectors(self, requests):
        self.calls["vectors"] += 1
        out = []
        for req in requests:
            last = req["prefix_ids"][-1] if req["prefix_ids"] else 0
            vec = torch.full((1, 8), float(last), dtype=torch.float16)
            out.append({"26": vec})
        return out

    def sample_full(self, requests):
        self.calls["full"] += 1
        out = []
        for i, req in enumerate(requests):
            n = min(req["max_new_tokens"], 50)
            answer = self.gold if (i % 3) != 2 else "7"
            out.append(
                {
                    "gen_ids": [1] * n,
                    "text": f"- Step 1: x\n\\boxed{{{answer}}}",
                    "finished": n < req["max_new_tokens"],
                    "n_tokens": n,
                }
            )
        return out

    def decode(self, ids):
        return ""


class StubProbe:
    """Candidate whose last id is 10 (opt 0) gets high V; longer opts look longer."""

    layers = [26]
    tag = "stub"

    def scalars(self, tokens_so_far, step_index, problem_chars, level, subject):
        return np.array([tokens_so_far, step_index], dtype=np.float32)

    def predict(self, hidden, scalars, budgets):
        last = np.array([float(h["26"].reshape(-1)[-1]) for h in hidden])
        opt = np.clip(last - 10, 0, 9)
        v = np.where(opt == 0, 0.9, 0.6 - 0.1 * opt)
        t = 30 + 40 * opt
        feas = (t <= budgets).astype(np.float64) * 0.9 + 0.05
        return {"v": v, "feasibility": feas, "t_mean": t}


def make_run(arm="vt", budget=200, **kw):
    cfg = SearchConfig(arm=arm, budget=budget, width=3, branch_max_tokens=20, **kw)
    return SearchRun(
        problem={"problem_id": "p", "problem": "1+1?", "gold": "42", "level": 3, "subject": "algebra"},
        cfg=cfg,
        rng=np.random.default_rng(0),
    )


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
def test_policy_scores_definitions():
    v = np.array([0.8, 0.5])
    feas = np.array([0.5, 1.0])
    t = np.array([100.0, 20.0])
    fin = np.array([False, False])
    assert np.allclose(policy_scores("v_only", v, feas, t, fin), v)
    assert np.allclose(policy_scores("vt", v, feas, t, fin), v * feas)
    assert np.allclose(policy_scores("bang_per_buck", v, feas, t, fin), v / (t + 1))
    assert np.allclose(policy_scores("shortest", v, feas, t, fin), -t)
    assert np.allclose(policy_scores("random", v, feas, t, fin), 0)


def test_finished_candidate_has_zero_remaining_length():
    v = np.array([0.7, 0.7])
    feas = np.array([0.1, 0.1])
    t = np.array([500.0, 500.0])
    fin = np.array([True, False])
    vt = policy_scores("vt", v, feas, t, fin)
    assert vt[0] == pytest.approx(0.7) and vt[1] == pytest.approx(0.07)
    bpb = policy_scores("bang_per_buck", v, feas, t, fin)
    assert bpb[0] > bpb[1]


def test_feasibility_gate_ranks_feasible_first():
    v = np.array([0.9, 0.3, 0.6])
    feas = np.array([0.2, 0.8, 0.7])
    t = np.array([300.0, 50.0, 80.0])
    fin = np.zeros(3, bool)
    s = policy_scores("feasibility_gate", v, feas, t, fin)
    assert np.argmax(s) == 2  # feasible with the higher V
    assert s[0] < s[1] < s[2]
    # nothing feasible -> fall back to V * feasibility
    s2 = policy_scores("feasibility_gate", v, np.array([0.1, 0.2, 0.3]), t, fin)
    assert np.allclose(s2, v * np.array([0.1, 0.2, 0.3]))


def test_selection_probabilities():
    s = np.array([0.1, 0.5, 0.9])
    p = selection_probabilities(s, alpha=1.0)
    assert p.sum() == pytest.approx(1.0) and np.argmax(p) == 2
    sharp = selection_probabilities(s, alpha=8.0)
    assert sharp[2] > p[2]
    assert np.allclose(selection_probabilities(s, 1.0, greedy=True), [0, 0, 1])
    assert np.allclose(selection_probabilities(np.zeros(3), 1.0), 1 / 3)
    assert np.allclose(selection_probabilities(np.array([0.4]), 1.0), [1.0])


def test_annealing_alpha_bounds():
    assert annealing_alpha(512, 512) == 1.0
    assert annealing_alpha(512, 128) == pytest.approx(4.0)
    assert annealing_alpha(512, 1) == 8.0
    assert annealing_alpha(512, 0) == 8.0


# --------------------------------------------------------------------------- #
# search run
# --------------------------------------------------------------------------- #
def test_receive_charges_every_sample_and_dedups():
    run = make_run()
    results = [
        {"step_ids": [0, 1, 1], "text": "- a\n", "finished": False, "n_tokens": 3},
        {"step_ids": [0, 1, 1], "text": "- a\n", "finished": False, "n_tokens": 3},
        {"step_ids": [0, 2], "text": "- b\n", "finished": False, "n_tokens": 2},
    ]
    cands = run.receive(results, decode=lambda ids: "")
    assert run.spent == 8  # duplicate is still paid for
    assert [c.text for c in cands] == ["- a\n", "- b\n"]
    assert run.needs_scoring()


def test_boxed_in_full_text_marks_finished():
    run = make_run()
    run.text = "- Step 0: x\n"
    cands = run.receive(
        [{"step_ids": [0], "text": "\\boxed{42}\n", "finished": False, "n_tokens": 1}],
        decode=lambda ids: "",
    )
    assert cands[0].finished
    chosen = run.select()
    assert chosen is not None and run.done and run.reason == "answered"
    rec = run.record(seed=0)
    assert rec["correct"] and rec["answer"] == "42" and rec["tokens_spent"] == 1


def test_committing_narrows_width_and_budget_caps_step():
    run = make_run(budget=100)
    assert run.current_width() == 3
    run.spent = 85  # 15 remaining <= eta * 100
    assert run.committing() and run.current_width() == 1
    assert run.step_cap() == 15
    run.spent = 100
    assert run.step_cap() == 1


def test_tree_search_vt_picks_good_candidate_and_answers():
    gen = StubGenerator(answer_step=3)
    run = make_run(arm="vt", budget=400, greedy=True)
    run_tree_searches([run], gen, StubProbe(), batch_size=8)
    rec = run.record(0)
    assert rec["reason"] == "answered" and rec["correct"]
    assert rec["n_steps"] == 4
    # every expansion before the answer chose opt 0 (the high-V candidate)
    for step in rec["steps"][:3]:
        assert step["chosen"] == 0
        assert len(step["candidates"]) == 3
    # accounting: 3 steps x (5+6+7) tokens + final expansion (4 x 3 candidates)
    assert rec["tokens_spent"] == 3 * 18 + 3 * 4
    assert rec["n_probe_calls"] == 3  # final round: all candidates identical -> no scoring
    assert gen.calls["vectors"] == 3


def test_random_arm_never_scores():
    gen = StubGenerator(answer_step=2)
    run = make_run(arm="random", budget=400)
    run_tree_searches([run], gen, probe=None, batch_size=8)
    assert run.done and run.n_probe_calls == 0 and gen.calls["vectors"] == 0


def test_probe_arm_requires_probe():
    with pytest.raises(ValueError):
        run_tree_searches([make_run(arm="vt")], StubGenerator(), probe=None)


def test_budget_exhaustion_terminates_and_is_recorded():
    gen = StubGenerator(answer_step=50)  # never answers
    run = make_run(arm="v_only", budget=60, greedy=True)
    run_tree_searches([run], gen, StubProbe(), batch_size=8)
    rec = run.record(0)
    assert rec["reason"] == "budget"
    assert rec["tokens_spent"] >= 60
    assert not rec["correct"] and rec["answer"] is None
    assert not rec["within_budget"] or rec["tokens_spent"] == 60


def test_lockstep_many_runs_share_batches():
    gen = StubGenerator(answer_step=2)
    runs = [make_run(arm="vt", budget=300, greedy=True) for _ in range(5)]
    for i, r in enumerate(runs):
        r.problem = dict(r.problem, problem_id=f"p{i}")
    run_tree_searches(runs, gen, StubProbe(), batch_size=64)
    assert all(r.done and r.record(0)["correct"] for r in runs)
    # 3 rounds (2 reasoning steps + answer), each one generation call
    assert gen.calls["steps"] == 3


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def test_majority_answer_normalises_and_breaks_ties_early():
    assert majority_answer(["1/2", " 1/2", "3"]) == "1/2"
    assert majority_answer(["a", "b"]) == "a"
    assert majority_answer([None, None]) is None
    assert majority_answer(["\\frac{1}{2}", "\\frac{1}{2}", "0.5", "0.5", "0.5"]) == "0.5"


def test_chain_baselines_budget_accounting():
    gen = StubGenerator()
    problem = {"problem_id": "p", "problem": "?", "gold": "42", "level": 1}
    single = ChainRun(problem=problem, arm="single_chain", budget=300)
    vote = ChainRun(problem=problem, arm="majority_vote", budget=300)
    run_chain_baselines([single, vote], gen, batch_size=8)
    s, v = single.record(0), vote.record(0)
    assert s["n_expansions"] == 1 and s["tokens_spent"] == 50 and s["correct"]
    assert v["n_expansions"] == 6  # 6 x 50 = 300, then remaining < MIN_CHAIN_TOKENS
    assert v["tokens_spent"] == 300 and v["within_budget"]
    assert v["correct"]  # 4 votes for 42 vs 2 for 7
    assert 300 - v["tokens_spent"] < MIN_CHAIN_TOKENS


def test_summarize_records_per_level():
    recs = [
        {"correct": True, "correct_within_budget": True, "tokens_spent": 100, "budget": 256, "level": 1, "reason": "answered"},
        {"correct": False, "correct_within_budget": False, "tokens_spent": 300, "budget": 256, "level": 5, "reason": "budget"},
    ]
    s = summarize_records(recs)
    assert s["n"] == 2 and s["accuracy"] == 0.5 and s["over_budget_rate"] == 0.5
    assert s["by_level"]["5"]["accuracy"] == 0.0
    assert summarize_records([]) == {"n": 0}
