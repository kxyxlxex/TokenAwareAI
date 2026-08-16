"""Stable, sortable artifact names tied to split ordinals."""

from __future__ import annotations

import re
from pathlib import Path


def problem_stem(problem_number: int, problem_id: str) -> str:
    """Return a 1-based split ordinal plus filesystem-safe problem id."""
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", problem_id)
    return f"p{problem_number:04d}_{safe_id}"


def root_paths(
    root_dir: Path, problem_number: int, problem_id: str
) -> tuple[Path, Path]:
    jsonl = root_dir / f"{problem_stem(problem_number, problem_id)}.jsonl"
    return jsonl, jsonl.with_suffix(".pt")


def mc_path(root_dir: Path, problem_number: int, problem_id: str) -> Path:
    return root_dir / f"{problem_stem(problem_number, problem_id)}.jsonl"
