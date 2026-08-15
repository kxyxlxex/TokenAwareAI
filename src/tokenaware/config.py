from __future__ import annotations

import os
from pathlib import Path


def _detect_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


IN_COLAB = _detect_colab()


def resolve_repo_root() -> Path:
    """Repo root: TOKENAWARE_ROOT, then this file's parents, then /content/TokenAwareAI."""
    env = os.environ.get("TOKENAWARE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parents[2]
    if (here / "src" / "tokenaware").is_dir():
        return here
    colab = Path("/content/TokenAwareAI")
    if (colab / "src" / "tokenaware").is_dir():
        return colab
    return here


def resolve_artifacts_dir(repo_root: Path) -> Path:
    """Prefer Drive on Colab so rollouts survive runtime disconnects."""
    env = os.environ.get("TOKENAWARE_ARTIFACTS")
    if env:
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    if IN_COLAB:
        drive = Path("/content/drive/MyDrive/TokenAwareAI/artifacts")
        if Path("/content/drive/MyDrive").is_dir():
            drive.mkdir(parents=True, exist_ok=True)
            return drive
    p = repo_root / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


REPO_ROOT = resolve_repo_root()
ARTIFACTS_DIR = resolve_artifacts_dir(REPO_ROOT)
DATASETS_DIR = REPO_ROOT / "datasets"

# Load from the Hub on Colab. A finished local snapshot is optional.
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
