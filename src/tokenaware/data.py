"""Load Hendrycks MATH and build the Phase 0 probe splits."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import DATASETS_DIR, MATH_DATASET_ID, N_PROBE_TRAIN, N_PROBE_VAL, SPLIT_SEED
from .scoring import gold_from_solution

SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def _problem_id(subject: str, problem: str) -> str:
    h = hashlib.sha1(f"{subject}\n{problem}".encode()).hexdigest()[:12]
    return f"{subject}:{h}"


def _row_from_example(subject: str, ex, train_index: int) -> dict | None:
    raw_level = str(ex["level"]).replace("Level ", "").strip()
    if not raw_level.isdigit():
        return None  # 2 geometry rows are "Level ?"
    problem = ex["problem"]
    return {
        "problem_id": _problem_id(subject, problem),
        "subject": subject,
        "level": int(raw_level),
        "problem": problem,
        "solution": ex["solution"],
        "gold": gold_from_solution(ex["solution"]),
        "train_index": int(train_index),
    }


def load_math_train(local_dir: Path | None = None) -> list[dict]:
    """Local parquet if present, otherwise Hugging Face `EleutherAI/hendrycks_math`."""
    root = local_dir or (DATASETS_DIR / "hendrycks_math")
    rows: list[dict] = []
    local_ok = all((root / s / "train-00000-of-00001.parquet").is_file() for s in SUBJECTS)
    if local_ok:
        for subject in SUBJECTS:
            df = pd.read_parquet(root / subject / "train-00000-of-00001.parquet")
            for i, ex in df.iterrows():
                row = _row_from_example(subject, ex, i)
                if row:
                    rows.append(row)
        return rows

    from datasets import load_dataset

    for subject in SUBJECTS:
        ds = load_dataset(MATH_DATASET_ID, subject, split="train")
        for i, ex in enumerate(ds):
            row = _row_from_example(subject, ex, i)
            if row:
                rows.append(row)
    return rows


def stratified_split(
    rows: list[dict],
    n_val: int = N_PROBE_VAL,
    n_train: int = N_PROBE_TRAIN,
    seed: int = SPLIT_SEED,
) -> dict:
    """Hold out n_val from MATH train, then sample n_train from the rest.

    Stratify both draws by official Level 1–5.
    """
    by_level: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)
    for lvl in by_level:
        by_level[lvl].sort(key=lambda r: r["problem_id"])
        # Deterministic shuffle from problem_id + seed
        by_level[lvl].sort(
            key=lambda r: hashlib.sha1(f"{seed}:{r['problem_id']}".encode()).hexdigest()
        )

    total = sum(len(v) for v in by_level.values())
    val: list[dict] = []
    remain: list[dict] = []
    for lvl in sorted(by_level):
        bucket = by_level[lvl]
        n = max(1, round(n_val * len(bucket) / total))
        n = min(n, len(bucket) - 1)  # leave at least one for train pool
        val.extend(bucket[:n])
        remain.extend(bucket[n:])

    # Trim / top-up val to exactly n_val
    remain.sort(key=lambda r: hashlib.sha1(f"{seed}:r:{r['problem_id']}".encode()).hexdigest())
    if len(val) > n_val:
        extra = val[n_val:]
        val = val[:n_val]
        remain = extra + remain
    elif len(val) < n_val:
        need = n_val - len(val)
        val.extend(remain[:need])
        remain = remain[need:]

    remain_by_level: dict[int, list[dict]] = defaultdict(list)
    for r in remain:
        remain_by_level[r["level"]].append(r)
    remain_total = len(remain)
    train: list[dict] = []
    leftover: list[dict] = []
    for lvl in sorted(remain_by_level):
        bucket = remain_by_level[lvl]
        n = max(1, round(n_train * len(bucket) / remain_total))
        n = min(n, len(bucket))
        train.extend(bucket[:n])
        leftover.extend(bucket[n:])
    leftover.sort(key=lambda r: hashlib.sha1(f"{seed}:t:{r['problem_id']}".encode()).hexdigest())
    if len(train) > n_train:
        train = train[:n_train]
    elif len(train) < n_train:
        train.extend(leftover[: n_train - len(train)])

    val_ids = {r["problem_id"] for r in val}
    train_ids = {r["problem_id"] for r in train}
    assert val_ids.isdisjoint(train_ids)
    return {
        "seed": seed,
        "n_source": total,
        "val": val,
        "train": train,
        "level_counts": {
            "source": {str(k): len(v) for k, v in sorted(by_level.items())},
            "val": _count_levels(val),
            "train": _count_levels(train),
        },
    }


def _count_levels(rows: list[dict]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for r in rows:
        c[str(r["level"])] += 1
    return dict(sorted(c.items()))


def save_split(split: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = {
        "seed": split["seed"],
        "n_source": split["n_source"],
        "level_counts": split["level_counts"],
        "val": split["val"],
        "train": split["train"],
    }
    path.write_text(json.dumps(slim, indent=2))


def load_split(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# search evaluation sets (never used for probe training)
# --------------------------------------------------------------------------- #
MATH500_DATASET_ID = "HuggingFaceH4/MATH-500"
GSM8K_DATASET_ID = "openai/gsm8k"

_SUBJECT_ALIASES = {
    "algebra": "algebra",
    "counting & probability": "counting_and_probability",
    "counting and probability": "counting_and_probability",
    "geometry": "geometry",
    "intermediate algebra": "intermediate_algebra",
    "number theory": "number_theory",
    "prealgebra": "prealgebra",
    "precalculus": "precalculus",
}


def _normalize_subject(name: str) -> str:
    key = str(name).strip().lower()
    return _SUBJECT_ALIASES.get(key, key.replace(" ", "_"))


def load_math500(local_dir: Path | None = None) -> list[dict]:
    """MATH-500 (Lightman et al. uniform subset of MATH test), cached as JSON.

    Disjoint from MATH train, hence from every probe training problem.
    """
    root = local_dir or DATASETS_DIR
    cache = root / "math500.json"
    if cache.is_file():
        return json.loads(cache.read_text())

    from datasets import load_dataset

    ds = load_dataset(MATH500_DATASET_ID, split="test")
    rows = []
    for ex in ds:
        level = ex.get("level")
        try:
            level = int(str(level).replace("Level", "").strip())
        except (TypeError, ValueError):
            level = 0
        rows.append(
            {
                "problem_id": f"math500:{ex.get('unique_id') or _problem_id('math500', ex['problem'])}",
                "subject": _normalize_subject(ex.get("subject", "")),
                "level": level,
                "problem": ex["problem"],
                "gold": str(ex["answer"]).strip(),
            }
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=1))
    return rows


def load_gsm8k_test(local_dir: Path | None = None) -> list[dict]:
    """GSM8K test (1,319). Negative control: V+T should not help here."""
    root = local_dir or DATASETS_DIR
    cache = root / "gsm8k_test.json"
    if cache.is_file():
        return json.loads(cache.read_text())

    from datasets import load_dataset

    ds = load_dataset(GSM8K_DATASET_ID, "main", split="test")
    rows = []
    for i, ex in enumerate(ds):
        gold = str(ex["answer"]).split("####")[-1].strip().replace(",", "")
        rows.append(
            {
                "problem_id": f"gsm8k:{i:04d}",
                "subject": "gsm8k",
                "level": 0,
                "problem": ex["question"],
                "gold": gold,
            }
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=1))
    return rows


def load_eval_set(name: str) -> list[dict]:
    if name == "math500":
        return load_math500()
    if name == "gsm8k":
        return load_gsm8k_test()
    raise ValueError(f"unknown eval set {name!r}")
