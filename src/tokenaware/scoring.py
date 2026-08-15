"""Gold-answer match. Prefer math-verify; fall back to normalized string match."""

from __future__ import annotations

from .steps import extract_boxed


def _normalize(s: str) -> str:
    s = s.strip().replace(" ", "").replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\;", "")
    if s.startswith("\\text{") and s.endswith("}"):
        s = s[6:-1]
    return s.lower()


def answers_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        from math_verify import parse, verify

        gold_p = parse(gold if "\\boxed" in gold else f"\\boxed{{{gold}}}")
        pred_p = parse(pred if "\\boxed" in pred else f"\\boxed{{{pred}}}")
        if gold_p is not None and pred_p is not None and verify(gold_p, pred_p):
            return True
    except Exception:
        pass
    return _normalize(pred) == _normalize(gold)


def gold_from_solution(solution: str) -> str:
    boxed = extract_boxed(solution)
    if boxed is not None:
        return boxed
    return solution.strip().splitlines()[-1].strip()


def rollout_correct(generated: str, gold_answer: str) -> bool:
    return answers_equal(extract_boxed(generated), gold_answer)
