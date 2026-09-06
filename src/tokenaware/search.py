"""Budgeted step-level tree search with probe-scored node selection.

This is the online counterpart of the offline sibling metrics. For one problem
and an output-token budget ``B_0``:

1. sample ``width`` candidate next steps from the current prefix (each stops at
   its first newline, i.e. one reasoning line);
2. read the hidden state at every candidate's last token, run the probe, and
   score each candidate with the arm's policy;
3. pick one (stochastically, sharpening toward greedy as the budget drains);
4. repeat until the chosen step contains a boxed answer or the budget is gone.

**Accounting.** Every generated token is charged, including the discarded
siblings, so all tree arms pay the same expansion overhead and differ only in
which sibling they keep. The probe reads hidden states from a forward pass over
the chosen prefix and adds no output tokens.

**Arms.**

======================  =====================================================
``single_chain``        one sample, hard cut at ``B_0`` (floor)
``majority_vote``       as many independent chains as fit in ``B_0``, then vote
``random``              tree, uniform pick (tree overhead without a probe)
``v_only``              tree, ``Score = V``  (ReProbe-style control)
``vt``                  tree, ``Score = V * P(T <= B_rem)``  (the claim)
``bang_per_buck``       tree, ``Score = V / (T + 1)``
``feasibility_gate``    tree, keep ``P(T <= B_rem) >= 0.5`` then rank by ``V``
``shortest``            tree, ``Score = -T``  (T driving alone; expected to hurt)
======================  =====================================================

The search state machine is model-agnostic: ``Generator`` is a protocol, so the
lockstep driver runs against a stub in unit tests and against Qwen3-8B on a GPU.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
import torch

from .config import PROBE_LAYER_INDICES, TEMPERATURE, TOP_K, TOP_P
from .probes.cache import SUBJECTS
from .probes.features import N_LEVELS, FeatureSpec, Normalizer
from .probes.heads import Probe, ProbeConfig
from .scoring import _normalize, answers_equal
from .steps import extract_boxed

TREE_ARMS = (
    "random",
    "v_only",
    "vt",
    "bang_per_buck",
    "feasibility_gate",
    "shortest",
)
CHAIN_ARMS = ("single_chain", "majority_vote")
ALL_ARMS = CHAIN_ARMS + TREE_ARMS
PROBE_ARMS = tuple(a for a in TREE_ARMS if a != "random")

STOP_STRINGS = ("\n",)
MIN_CHAIN_TOKENS = 32  # majority vote stops when less than this remains


# --------------------------------------------------------------------------- #
# generator protocol
# --------------------------------------------------------------------------- #
class Generator(Protocol):
    """Batched access to the frozen policy model."""

    def sample_steps(self, requests: list[dict]) -> list[dict]:
        """One reasoning line per request.

        request: ``{"problem", "prefix_ids", "max_new_tokens"}``
        result:  ``{"step_ids", "text", "finished", "n_tokens"}`` where
        ``finished`` means EOS was produced (the caller also checks for a boxed
        answer in the full text).
        """

    def state_vectors(self, requests: list[dict]) -> list[dict[str, torch.Tensor]]:
        """Hidden vectors at the last token of ``prompt + prefix_ids``.

        request: ``{"problem", "prefix_ids"}``; result maps ``str(layer)`` to a
        ``[1, hidden]`` fp16 CPU tensor, as ``generate.capture_state_vectors``.
        """

    def sample_full(self, requests: list[dict]) -> list[dict]:
        """A full continuation per request.

        request: ``{"problem", "prefix_ids", "max_new_tokens"}``
        result:  ``{"gen_ids", "text", "finished", "n_tokens"}``
        """

    def decode(self, ids: Sequence[int]) -> str:
        ...


def batched(fn, requests: list[dict], batch_size: int) -> list:
    out: list = []
    for start in range(0, len(requests), batch_size):
        out.extend(fn(requests[start : start + batch_size]))
    return out


# --------------------------------------------------------------------------- #
# online probe
# --------------------------------------------------------------------------- #
class OnlineProbe:
    """A trained probe applied to fresh hidden states, no cache required.

    Rebuilds exactly the feature vector ``FeatureStore`` produced at training
    time: hidden vectors for ``spec.layers`` in order, then the causal scalars
    (``log1p(tokens)``, ``tokens/1024``, ``step/16``, ``log1p(chars)/8``) and
    optional level/subject one-hots, normalised with the saved statistics.
    """

    def __init__(self, path: str | Path, device: str = "cuda") -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.path = str(path)
        self.device = torch.device(
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.spec = FeatureSpec(**payload["feature_spec"])
        if self.spec.history != 1 or self.spec.use_delta:
            raise NotImplementedError(
                "online scoring supports history=1 without delta features; "
                f"got history={self.spec.history} delta={self.spec.use_delta}"
            )
        self.normalizer = (
            Normalizer.from_state_dict(payload["normalizer"]).to(self.device)
            if payload.get("normalizer")
            else None
        )
        self.probe = Probe(ProbeConfig.from_dict(payload["probe_config"])).to(
            self.device
        )
        self.probe.load_state_dict(payload["probe_state"])
        self.probe.eval()
        self.train_config = payload.get("train_config", {})
        self.tag = self.train_config.get("tag") or Path(path).parent.name
        self.layers = [int(x) for x in self.spec.layers]
        self.has_v = self.probe.v_head is not None
        self.has_t = self.probe.t_head is not None

    @property
    def n_params(self) -> int:
        return self.probe.n_params

    def scalars(
        self,
        tokens_so_far: int,
        step_index: int,
        problem_chars: int,
        level: int,
        subject: str,
    ) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.spec.use_pos:
            blocks.append(
                np.array(
                    [
                        math.log1p(float(tokens_so_far)),
                        tokens_so_far / 1024.0,
                        step_index / 16.0,
                        math.log1p(float(problem_chars)) / 8.0,
                    ],
                    dtype=np.float32,
                )
            )
        if self.spec.use_meta:
            level_oh = np.zeros(N_LEVELS, dtype=np.float32)
            level_oh[int(np.clip(level - 1, 0, N_LEVELS - 1))] = 1.0
            subject_oh = np.zeros(len(SUBJECTS), dtype=np.float32)
            idx = SUBJECTS.index(subject) if subject in SUBJECTS else 0
            subject_oh[idx] = 1.0
            blocks += [level_oh, subject_oh]
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocks)

    @torch.no_grad()
    def predict(
        self,
        hidden: list[dict[str, torch.Tensor]],
        scalars: np.ndarray,
        budgets: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """``v``, ``feasibility`` = P(T <= budget), and ``t_mean`` per row."""
        n = len(hidden)
        parts: list[torch.Tensor] = []
        if self.spec.use_hidden:
            rows = []
            for h in hidden:
                vec = torch.cat(
                    [h[str(layer)].reshape(-1).float() for layer in self.layers]
                )
                rows.append(vec)
            parts.append(torch.stack(rows).to(self.device))
        if self.spec.n_scalar:
            parts.append(
                torch.from_numpy(np.asarray(scalars, dtype=np.float32)).to(self.device)
            )
        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        if self.normalizer is not None:
            x = (x - self.normalizer.mean) / self.normalizer.std
        out = self.probe(x)
        result: dict[str, np.ndarray] = {}
        result["v"] = (
            torch.sigmoid(out["v_logit"]).float().cpu().numpy()
            if self.has_v
            else np.full(n, 0.5, dtype=np.float32)
        )
        if self.has_t:
            budget_t = torch.from_numpy(np.asarray(budgets, dtype=np.float32)).to(
                self.device
            )
            result["feasibility"] = (
                self.probe.t_cdf_at(out, budget_t).float().cpu().numpy()
            )
            result["t_mean"] = self.probe.t_mean(out).float().cpu().numpy()
        else:
            result["feasibility"] = np.ones(n, dtype=np.float32)
            result["t_mean"] = np.full(n, np.nan, dtype=np.float32)
        return result


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
def policy_scores(
    arm: str,
    v: np.ndarray,
    feasibility: np.ndarray,
    t_mean: np.ndarray,
    finished: np.ndarray,
) -> np.ndarray:
    """Per-candidate desirability under ``arm``. Higher is better.

    A candidate that already contains the final answer has ``T = 0``: its
    feasibility is 1 and its remaining length is 0, whatever the probe says.
    """
    feas = np.where(finished, 1.0, feasibility).astype(np.float64)
    t = np.where(finished, 0.0, np.nan_to_num(t_mean, nan=1024.0)).astype(np.float64)
    v = np.asarray(v, dtype=np.float64)
    if arm == "v_only":
        return v
    if arm == "vt":
        return v * feas
    if arm == "bang_per_buck":
        return v / (t + 1.0)
    if arm == "feasibility_gate":
        gate = feas >= 0.5
        if gate.any():
            # Infeasible candidates are ranked below every feasible one.
            return np.where(gate, 1.0 + v, feas * 1e-3)
        return v * feas
    if arm == "shortest":
        return -t
    if arm == "random":
        return np.zeros_like(v)
    raise ValueError(f"unknown tree arm {arm!r}")


def selection_probabilities(
    scores: np.ndarray, alpha: float, greedy: bool = False
) -> np.ndarray:
    """``p_i ∝ (score_i - min + eps)^alpha``; greedy puts all mass on the max."""
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n == 1:
        return np.ones(1)
    if greedy:
        p = np.zeros(n)
        p[int(np.argmax(scores))] = 1.0
        return p
    shifted = scores - scores.min()
    if not np.isfinite(shifted).all() or shifted.max() <= 0:
        return np.full(n, 1.0 / n)
    shifted = shifted / shifted.max() + 1e-3
    weights = shifted**alpha
    return weights / weights.sum()


def annealing_alpha(budget: int, remaining: int, alpha_max: float = 8.0) -> float:
    """BAVT-style sharpening: ``alpha = 1 / r`` with ``r = B_rem / B_0``."""
    if budget <= 0:
        return alpha_max
    r = max(remaining, 1) / float(budget)
    return float(min(alpha_max, max(1.0, 1.0 / r)))


# --------------------------------------------------------------------------- #
# search state
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    step_ids: list[int]
    text: str
    n_tokens: int
    finished: bool  # EOS or boxed answer present in the full text
    hidden: dict[str, torch.Tensor] | None = None
    v: float = float("nan")
    feasibility: float = float("nan")
    t_mean: float = float("nan")
    score: float = float("nan")


@dataclass
class SearchConfig:
    arm: str = "vt"
    budget: int = 512
    width: int = 3
    branch_max_tokens: int = 48
    eta: float = 0.2  # below this remaining fraction, stop branching
    alpha_max: float = 8.0
    greedy: bool = False
    max_steps: int = 40
    max_empty_steps: int = 3

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SearchRun:
    problem: dict
    cfg: SearchConfig
    rng: np.random.Generator
    prefix_ids: list[int] = field(default_factory=list)
    text: str = ""
    spent: int = 0
    step_index: int = 0
    done: bool = False
    reason: str = ""
    n_expansions: int = 0
    n_probe_calls: int = 0
    empty_streak: int = 0
    log: list[dict] = field(default_factory=list)
    pending: list[Candidate] = field(default_factory=list)
    spent_before_round: int = 0

    # -- budget --------------------------------------------------------------
    @property
    def remaining(self) -> int:
        return self.cfg.budget - self.spent

    def committing(self) -> bool:
        """True once the remaining fraction is at or below ``eta``."""
        return self.remaining <= self.cfg.eta * self.cfg.budget

    def current_width(self) -> int:
        if self.committing():
            return 1
        return max(1, self.cfg.width)

    def step_cap(self) -> int:
        return max(1, min(self.cfg.branch_max_tokens, self.remaining))

    # -- lifecycle -----------------------------------------------------------
    def receive(self, results: list[dict], decode) -> list[Candidate]:
        """Charge every sampled step and keep the unique ones."""
        candidates: list[Candidate] = []
        seen: set[str] = set()
        self.spent_before_round = self.spent
        for res in results:
            ids = list(res["step_ids"])
            n = int(res.get("n_tokens", len(ids)))
            self.spent += n
            text = res.get("text")
            if text is None:
                text = decode(ids)
            full = self.text + text
            finished = bool(res.get("finished")) or extract_boxed(full) is not None
            key = text.strip()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(Candidate(ids, text, n, finished))
        self.n_expansions += 1
        self.pending = candidates
        return candidates

    def needs_scoring(self) -> bool:
        return (
            len(self.pending) > 1
            and self.cfg.arm in PROBE_ARMS
        )

    def select(self) -> Candidate | None:
        cands = self.pending
        self.pending = []
        if not cands:
            self.empty_streak += 1
            if self.empty_streak >= self.cfg.max_empty_steps:
                self.finish("no_step")
            elif self.remaining <= 0:
                self.finish("budget")
            return None

        finished = np.array([c.finished for c in cands])
        if len(cands) == 1 or self.cfg.arm not in PROBE_ARMS:
            scores = np.zeros(len(cands))
        else:
            scores = policy_scores(
                self.cfg.arm,
                np.array([c.v for c in cands]),
                np.array([c.feasibility for c in cands]),
                np.array([c.t_mean for c in cands]),
                finished,
            )
        for c, s in zip(cands, scores):
            c.score = float(s)
        alpha = annealing_alpha(self.cfg.budget, self.remaining, self.cfg.alpha_max)
        probs = selection_probabilities(scores, alpha, greedy=self.cfg.greedy)
        pick = int(self.rng.choice(len(cands), p=probs))
        chosen = cands[pick]

        self.log.append(
            {
                "step_index": self.step_index,
                "spent_before": self.spent_before_round,
                "remaining_after": self.remaining,
                "alpha": round(alpha, 3),
                "chosen": pick,
                "candidates": [
                    {
                        "n_tokens": c.n_tokens,
                        "finished": c.finished,
                        "v": _r(c.v),
                        "feasibility": _r(c.feasibility),
                        "t_mean": _r(c.t_mean),
                        "score": _r(c.score),
                        "text": c.text[:200],
                    }
                    for c in cands
                ],
            }
        )

        self.prefix_ids.extend(chosen.step_ids)
        self.text += chosen.text
        self.step_index += 1
        if chosen.text.strip():
            self.empty_streak = 0
        else:
            self.empty_streak += 1

        if chosen.finished:
            self.finish("answered")
        elif self.remaining <= 0:
            self.finish("budget")
        elif self.step_index >= self.cfg.max_steps:
            self.finish("max_steps")
        elif self.empty_streak >= self.cfg.max_empty_steps:
            self.finish("no_step")
        return chosen

    def finish(self, reason: str) -> None:
        self.done = True
        self.reason = reason

    def record(self, seed: int) -> dict:
        answer = extract_boxed(self.text)
        correct = answers_equal(answer, self.problem["gold"])
        return {
            "problem_id": self.problem["problem_id"],
            "level": self.problem.get("level"),
            "subject": self.problem.get("subject"),
            "arm": self.cfg.arm,
            "budget": self.cfg.budget,
            "seed": seed,
            "correct": bool(correct),
            "answer": answer,
            "tokens_spent": int(self.spent),
            "within_budget": bool(self.spent <= self.cfg.budget),
            "correct_within_budget": bool(correct and self.spent <= self.cfg.budget),
            "n_steps": self.step_index,
            "n_expansions": self.n_expansions,
            "n_probe_calls": self.n_probe_calls,
            "reason": self.reason,
            "steps": self.log,
            "text": self.text,
        }


def _r(x: float) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), 4)


# --------------------------------------------------------------------------- #
# lockstep drivers
# --------------------------------------------------------------------------- #
def run_tree_searches(
    runs: list[SearchRun],
    generator: Generator,
    probe: OnlineProbe | None,
    batch_size: int = 24,
    progress=None,
) -> None:
    """Advance every run one step per round until all are done.

    Rounds batch the generation requests of all active runs, then the hidden
    state requests of all runs that need scoring, then score them in one probe
    call. ``progress(round_index, n_active)`` is called once per round.
    """
    for run in runs:
        if run.cfg.arm in PROBE_ARMS and probe is None:
            raise ValueError(f"arm {run.cfg.arm!r} needs a probe")

    round_index = 0
    while True:
        active = [r for r in runs if not r.done]
        if not active:
            return
        if progress:
            progress(round_index, len(active))
        round_index += 1

        gen_requests: list[dict] = []
        owners: list[SearchRun] = []
        for run in active:
            cap = run.step_cap()
            for _ in range(run.current_width()):
                gen_requests.append(
                    {
                        "problem": run.problem["problem"],
                        "prefix_ids": list(run.prefix_ids),
                        "max_new_tokens": cap,
                    }
                )
                owners.append(run)
        results = batched(generator.sample_steps, gen_requests, batch_size)

        per_run: dict[int, list[dict]] = {}
        for owner, res in zip(owners, results):
            per_run.setdefault(id(owner), []).append(res)
        for run in active:
            run.receive(per_run.get(id(run), []), generator.decode)

        score_requests: list[dict] = []
        score_owners: list[tuple[SearchRun, Candidate]] = []
        for run in active:
            if not run.needs_scoring():
                continue
            for cand in run.pending:
                score_requests.append(
                    {
                        "problem": run.problem["problem"],
                        "prefix_ids": list(run.prefix_ids) + list(cand.step_ids),
                    }
                )
                score_owners.append((run, cand))
        if score_requests:
            vectors = batched(generator.state_vectors, score_requests, batch_size)
            scalars = []
            budgets = []
            for (run, cand), vec in zip(score_owners, vectors):
                cand.hidden = vec
                scalars.append(
                    probe.scalars(
                        tokens_so_far=len(run.prefix_ids) + len(cand.step_ids),
                        step_index=run.step_index,
                        problem_chars=len(run.problem["problem"]),
                        level=int(run.problem.get("level") or 0),
                        subject=str(run.problem.get("subject") or ""),
                    )
                )
                budgets.append(max(0, run.remaining))
            pred = probe.predict(
                [c.hidden for _, c in score_owners],
                np.stack(scalars),
                np.asarray(budgets, dtype=np.float32),
            )
            for i, (run, cand) in enumerate(score_owners):
                cand.v = float(pred["v"][i])
                cand.feasibility = float(pred["feasibility"][i])
                cand.t_mean = float(pred["t_mean"][i])
                cand.hidden = None  # free memory; the vector is not logged
            for run in {id(r): r for r, _ in score_owners}.values():
                run.n_probe_calls += 1

        for run in active:
            run.select()


@dataclass
class ChainRun:
    """Single-chain and majority-vote baselines under the same budget."""

    problem: dict
    arm: str
    budget: int
    spent: int = 0
    answers: list[str | None] = field(default_factory=list)
    chains: list[dict] = field(default_factory=list)
    done: bool = False
    reason: str = ""

    @property
    def remaining(self) -> int:
        return self.budget - self.spent

    def wants_chain(self) -> bool:
        if self.done:
            return False
        if self.arm == "single_chain":
            return not self.chains
        return self.remaining >= MIN_CHAIN_TOKENS

    def receive(self, res: dict) -> None:
        self.spent += int(res["n_tokens"])
        answer = extract_boxed(res["text"])
        self.answers.append(answer)
        self.chains.append(
            {
                "n_tokens": int(res["n_tokens"]),
                "finished": bool(res.get("finished")),
                "answer": answer,
            }
        )
        if self.arm == "single_chain" or not self.wants_chain():
            self.done = True
            self.reason = "answered" if self.arm == "single_chain" else "budget"

    def final_answer(self) -> str | None:
        if self.arm == "single_chain":
            return self.answers[0] if self.answers else None
        return majority_answer(self.answers)

    def record(self, seed: int) -> dict:
        answer = self.final_answer()
        correct = answers_equal(answer, self.problem["gold"])
        return {
            "problem_id": self.problem["problem_id"],
            "level": self.problem.get("level"),
            "subject": self.problem.get("subject"),
            "arm": self.arm,
            "budget": self.budget,
            "seed": seed,
            "correct": bool(correct),
            "answer": answer,
            "tokens_spent": int(self.spent),
            "within_budget": bool(self.spent <= self.budget),
            "correct_within_budget": bool(correct and self.spent <= self.budget),
            "n_steps": 0,
            "n_expansions": len(self.chains),
            "n_probe_calls": 0,
            "reason": self.reason,
            "chains": self.chains,
        }


def majority_answer(answers: Iterable[str | None]) -> str | None:
    """Most common normalised boxed answer; ties resolve to the earliest."""
    votes: Counter = Counter()
    first_form: dict[str, str] = {}
    order: dict[str, int] = {}
    for i, a in enumerate(answers):
        if a is None:
            continue
        key = _normalize(a)
        if not key:
            continue
        votes[key] += 1
        first_form.setdefault(key, a)
        order.setdefault(key, i)
    if not votes:
        return None
    best = max(votes, key=lambda k: (votes[k], -order[k]))
    return first_form[best]


def run_chain_baselines(
    runs: list[ChainRun],
    generator: Generator,
    batch_size: int = 24,
    progress=None,
) -> None:
    round_index = 0
    while True:
        active = [r for r in runs if r.wants_chain()]
        if not active:
            for r in runs:
                if not r.done:
                    r.done = True
                    r.reason = "budget"
            return
        if progress:
            progress(round_index, len(active))
        round_index += 1
        requests = [
            {
                "problem": r.problem["problem"],
                "prefix_ids": [],
                "max_new_tokens": max(1, r.remaining),
            }
            for r in active
        ]
        results = batched(generator.sample_full, requests, batch_size)
        for run, res in zip(active, results):
            run.receive(res)


# --------------------------------------------------------------------------- #
# Hugging Face generator
# --------------------------------------------------------------------------- #
class HFGenerator:
    """``Generator`` backed by the frozen Qwen3-8B loaded via ``generate.load_model``."""

    def __init__(self, model, tokenizer, layers: tuple[int, ...] = PROBE_LAYER_INDICES):
        from .generate import _prepare_tokenizer_for_batch

        self.model = model
        self.tokenizer = tokenizer
        self.layers = tuple(layers)
        _prepare_tokenizer_for_batch(tokenizer)
        eos = tokenizer.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        self._prompt_cache: dict[str, list[int]] = {}

    # -- helpers -------------------------------------------------------------
    def _prompt_ids(self, problem: str) -> list[int]:
        from .generate import build_prompt

        ids = self._prompt_cache.get(problem)
        if ids is None:
            # Force the format header in the prompt so the first sampled
            # step is "- Step 1: ...", not a wasted "Reasoning Steps:\n" node.
            prompt = build_prompt(self.tokenizer, problem) + "Reasoning Steps:\n"
            ids = self.tokenizer(prompt)["input_ids"]
            self._prompt_cache[problem] = ids
        return ids

    def _sequences(self, requests: list[dict]) -> list[list[int]]:
        seqs = []
        for req in requests:
            prefix = list(req.get("prefix_ids") or [])
            if prefix and prefix[-1] in self.eos_ids:
                prefix = prefix[:-1]
            seqs.append(self._prompt_ids(req["problem"]) + prefix)
        return seqs

    def _pad(self, seqs: list[list[int]]):
        inputs = self.tokenizer.pad(
            {
                "input_ids": seqs,
                "attention_mask": [[1] * len(s) for s in seqs],
            },
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}

    def _trim(self, ids: list[int]) -> tuple[list[int], bool]:
        pad = self.tokenizer.pad_token_id
        for i, t in enumerate(ids):
            if t in self.eos_ids:
                return ids[: i + 1], True
            if pad is not None and t == pad:
                return ids[:i], False
        return ids, False

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    @torch.inference_mode()
    def _generate(self, requests: list[dict], stop_at_newline: bool):
        seqs = self._sequences(requests)
        inputs = self._pad(seqs)
        width = inputs["input_ids"].shape[1]
        cap = max(int(r["max_new_tokens"]) for r in requests)
        kwargs = dict(
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            max_new_tokens=cap,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if stop_at_newline:
            kwargs["stop_strings"] = list(STOP_STRINGS)
            kwargs["tokenizer"] = self.tokenizer
        generated = self.model.generate(**inputs, **kwargs)
        out = []
        for i, req in enumerate(requests):
            raw = generated[i, width:].tolist()
            ids, eos = self._trim(raw)
            own_cap = int(req["max_new_tokens"])
            if len(ids) > own_cap:
                ids, eos = ids[:own_cap], False
            out.append((ids, eos))
        return out

    def sample_steps(self, requests: list[dict]) -> list[dict]:
        results = []
        for ids, eos in self._generate(requests, stop_at_newline=True):
            results.append(
                {
                    "step_ids": ids,
                    "text": self.decode(ids),
                    "finished": eos,
                    "n_tokens": len(ids),
                }
            )
        return results

    def sample_full(self, requests: list[dict]) -> list[dict]:
        results = []
        for ids, eos in self._generate(requests, stop_at_newline=False):
            results.append(
                {
                    "gen_ids": ids,
                    "text": self.decode(ids),
                    "finished": eos,
                    "n_tokens": len(ids),
                }
            )
        return results

    @torch.inference_mode()
    def state_vectors(self, requests: list[dict]) -> list[dict[str, torch.Tensor]]:
        from .hooks import hidden_state_hooks, last_token_vectors

        seqs = self._sequences(requests)
        device = next(self.model.parameters()).device
        max_len = max(len(s) for s in seqs)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((len(seqs), max_len), pad, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention[i, : len(s)] = 1
        # Run the backbone only. CausalLM.forward materialises full-seq logits
        # through lm_head (~vocab × hidden × batch × length), which OOMs a 20 GiB
        # card; the hooks already sit on model.model.layers.
        backbone = getattr(self.model, "model", self.model)
        with hidden_state_hooks(self.model, self.layers) as cache:
            backbone(
                input_ids.to(device),
                attention_mask=attention.to(device),
                use_cache=False,
            )
            cpu = {k: t.to(dtype=torch.float16, device="cpu") for k, t in cache.items()}
        if device.type == "cuda":
            torch.cuda.empty_cache()
        out = []
        for i, s in enumerate(seqs):
            vecs = last_token_vectors(cpu, len(s) - 1, batch=i)
            out.append({str(k): v.reshape(1, -1) for k, v in vecs.items()})
        return out


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def summarize_records(records: list[dict]) -> dict:
    """Accuracy and spend for one (arm, budget) cell, pooled and per level."""
    if not records:
        return {"n": 0}
    n = len(records)
    acc = float(np.mean([r["correct"] for r in records]))
    acc_within = float(np.mean([r["correct_within_budget"] for r in records]))
    spent = np.array([r["tokens_spent"] for r in records], dtype=np.float64)
    summary = {
        "n": n,
        "accuracy": round(acc, 4),
        "accuracy_within_budget": round(acc_within, 4),
        "tokens_mean": round(float(spent.mean()), 1),
        "tokens_p50": round(float(np.median(spent)), 1),
        "tokens_p90": round(float(np.quantile(spent, 0.9)), 1),
        "over_budget_rate": round(float(np.mean(spent > np.array([r["budget"] for r in records]))), 4),
        "reasons": dict(Counter(r.get("reason") for r in records)),
        "by_level": {},
    }
    levels = sorted({r.get("level") for r in records if r.get("level") is not None})
    for level in levels:
        sub = [r for r in records if r.get("level") == level]
        summary["by_level"][str(level)] = {
            "n": len(sub),
            "accuracy": round(float(np.mean([r["correct"] for r in sub])), 4),
            "tokens_mean": round(float(np.mean([r["tokens_spent"] for r in sub])), 1),
        }
    return summary
