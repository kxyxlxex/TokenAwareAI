"""Parse ReProbe-style one-line-per-step CoT and map steps onto token ids."""

from __future__ import annotations

import re
from dataclasses import dataclass


STEP_LINE_RE = re.compile(r"^\s*(?:-\s*)?(?:Step\s+\d+\s*:\s*)?(.*\S)\s*$", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


@dataclass
class Step:
    index: int
    text: str
    # Character span in the *generated* text (not including the prompt).
    char_start: int
    char_end: int
    # Token index of the last generated token belonging to this step
    # (offset into the generated-token sequence, 0-based).
    last_token_offset: int | None = None


def extract_boxed(text: str) -> str | None:
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def parse_steps(generated: str) -> list[Step]:
    """Split generated text on newlines; skip headers / empty / answer lines."""
    skip_prefixes = (
        "reasoning steps:",
        "<start of response>",
        "<end of response>",
        "<think>",
        "</think>",
    )
    steps: list[Step] = []
    cursor = 0
    for raw_line in generated.splitlines(keepends=True):
        line_start = cursor
        line_end = cursor + len(raw_line)
        cursor = line_end
        stripped = raw_line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(lower.startswith(p) for p in skip_prefixes):
            continue
        if lower.startswith("<answer>") or lower.startswith("answer:"):
            continue
        if stripped.startswith("\\boxed") or lower.startswith("final answer"):
            continue
        m = STEP_LINE_RE.match(stripped)
        body = (m.group(1) if m else stripped).strip()
        if not body:
            continue
        steps.append(
            Step(
                index=len(steps),
                text=body,
                char_start=line_start,
                char_end=line_end,
            )
        )
    return steps


def attach_token_offsets(
    steps: list[Step],
    generated: str,
    generated_ids: list[int],
    tokenizer,
) -> list[Step]:
    """Map each step's last character to a generated-token offset.

    Binary-searches decoded prefixes of the actual generated IDs, avoiding
    re-tokenization and the previous quadratic full-prefix scan.
    """
    if not generated_ids:
        return steps
    for step in steps:
        target = step.char_end
        low, high = 0, len(generated_ids) - 1
        if len(tokenizer.decode(generated_ids, skip_special_tokens=True)) < target:
            step.last_token_offset = None
            continue
        while low < high:
            mid = (low + high) // 2
            decoded_len = len(
                tokenizer.decode(
                    generated_ids[: mid + 1], skip_special_tokens=True
                )
            )
            if decoded_len >= target:
                high = mid
            else:
                low = mid + 1
        step.last_token_offset = low
    return steps
