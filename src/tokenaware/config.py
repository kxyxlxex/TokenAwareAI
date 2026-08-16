from __future__ import annotations

import os
from pathlib import Path


def resolve_repo_root() -> Path:
    """Use TOKENAWARE_ROOT when set, otherwise infer the checkout root."""
    env = os.environ.get("TOKENAWARE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def resolve_artifacts_dir(repo_root: Path) -> Path:
    """Use persistent TOKENAWARE_ARTIFACTS when set, otherwise repo/artifacts."""
    env = os.environ.get("TOKENAWARE_ARTIFACTS")
    if env:
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = repo_root / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


REPO_ROOT = resolve_repo_root()
ARTIFACTS_DIR = resolve_artifacts_dir(REPO_ROOT)
DATASETS_DIR = REPO_ROOT / "datasets"

# A finished local snapshot is optional; otherwise load from the Hub.
MODEL_ID = os.environ.get("TOKENAWARE_MODEL", "Qwen/Qwen3-8B")
MODEL_DIR = Path(os.environ.get("TOKENAWARE_MODEL_DIR", REPO_ROOT / "models" / "Qwen3-8B"))
MATH_DATASET_ID = "EleutherAI/hendrycks_math"

# Qwen3-8B: 36 layers. Plan uses 1-indexed depths at 0.25/0.50/0.75/1.0 L.
# HuggingFace layer index = plan_layer - 1.
PROBE_LAYERS_1INDEXED = (9, 18, 27, 36)
PROBE_LAYER_INDICES = tuple(i - 1 for i in PROBE_LAYERS_1INDEXED)

TEMPERATURE = 0.7
TOP_K = 20
TOP_P = 0.95
MAX_NEW_TOKENS = 1024

ROOT_K = 8
MC_K = 8
VAL_MC_K = 32

N_PROBE_VAL = 500
N_PROBE_TRAIN = 2000
SPLIT_SEED = 42

SYSTEM_PROMPT = (
    "You will be presented with a math problem. Think step by step carefully.\n"
    "Output format (follow STRICTLY):\n"
    "Reasoning Steps:\n"
    "- Step 1: <one logical operation on this line only>\n"
    "- Step 2: <next operation>\n"
    "- Step N: <final reasoning>\n"
    "Put the final answer in \\boxed{}.\n"
    "Each reasoning step MUST be a single line. No line breaks inside a step."
)
