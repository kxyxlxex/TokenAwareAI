"""Select root-rollout prefixes and aggregate Monte Carlo V/T labels."""

from __future__ import annotations

import json
from pathlib import Path

PREFIX_FRACTIONS = (0.25, 0.50, 0.75)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_prefix_states(
    rollouts: list[dict],
    traces_per_problem: int = 2,
    fractions: tuple[float, ...] = PREFIX_FRACTIONS,
) -> list[dict]:
    """Select nearest unique step boundaries at 25/50/75% on two traces."""
    usable = [r for r in rollouts if r.get("steps") and r.get("gen_ids")]
    selected: list[dict] = []
    for trace in usable[:traces_per_problem]:
        steps = trace["steps"]
        used: set[int] = set()
        for fraction in fractions:
            step_index = min(len(steps) - 1, max(0, round(fraction * len(steps)) - 1))
            if step_index in used:
                continue
            used.add(step_index)
            step = steps[step_index]
            token_end = step.get("last_token_offset")
            if token_end is None:
                continue
            selected.append(
                {
                    "source_sample_id": trace["sample_id"],
                    "fraction": fraction,
                    "step_index": step_index,
                    "prefix_char_end": step["char_end"],
                    "prefix": trace["text"][: step["char_end"]],
                    "prefix_token_ids": trace["gen_ids"][: token_end + 1],
                    "source_trace_correct": trace["correct"],
                    "source_tokens_remaining": step["tokens_remaining_this_trace"],
                }
            )
    return selected


def mc_label(continuations: list[dict]) -> dict:
    """Empirical V and T for one prefix from its fresh continuations."""
    if not continuations:
        raise ValueError("At least one continuation is required")
    n = len(continuations)
    lengths = [c["n_tokens"] for c in continuations]
    return {
        "mc_k": n,
        "v_mc": sum(bool(c["correct"]) for c in continuations) / n,
        "t_mc_mean": sum(lengths) / n,
        "t_mc_lengths": lengths,
        "n_correct": sum(bool(c["correct"]) for c in continuations),
        "n_truncated": sum(bool(c["truncated"]) for c in continuations),
    }
