# Current state — TokenAwareAI

Last updated 15 Aug 2026.

## Grand scheme

Build **cost-aware tree search** for math reasoning: a frozen LLM expands a CoT tree; two tiny probes on its hidden states score each node by **V** (P(correct)) and **T** (tokens left). Selection uses remaining budget. The paper lives on **V-only vs V+T**.

Plan: `plan-cost-aware-tree-search.md`. Idea review: `idea-cost-aware-tree-search.md`. Literature: `budget-aware-ai-literature.md`.

## Stage we are in

**Phase 0, Step 1 smoke-tested successfully on a Colab T4; Step 2b/2c code is ready.**

| Step | Status |
|---|---|
| 0. Downloads / Colab Hub load | Done. Do **not** keep 8B weights in git. Colab pulls `Qwen/Qwen3-8B` from Hugging Face. |
| 1. Instrumentation (hooks, step parse, generate) | **Passed Colab T4 smoke:** correct 144-token solution, 6 steps, four layers × 4096 dimensions. |
| 2a. Probe split (500 val + 2000 train from MATH train) | **Done locally:** `artifacts/splits/math_probe_split.json`. Rebuild on Colab with `make_splits.py` (same seed → same IDs). |
| 2b. Root rollouts `k=8` | **Not started.** Next GPU job. |
| 2c. Monte Carlo labels on prefixes | **Code ready; not run yet.** Requires root rollouts. |
| 3. Train V / T heads | Not started. |
| 4. Sibling-ranking kill criteria | Not started. |
| 5–8. Search + paper eval | Blocked on Phase 0. |

**Next action (Colab, GPU):** pull the latest commit, run the small root+MC smoke below, inspect its labels, then scale root rollouts before MC labels.

```python
# In the already-running Colab session:
%cd /content/TokenAwareAI
!git pull

# Tiny end-to-end test: 1 problem × 2 roots, then up to 6 prefixes × 2 continuations.
!python scripts/generate_root_rollouts.py --split train --limit 1 --k 2 --load-in-4bit
!python scripts/generate_mc_prefix_labels.py --split train --limit 1 --k 2 --load-in-4bit
```

The root script skips a problem if existing root artifacts are present. Remove that one
problem's `.jsonl` and `.pt` only if you intentionally need to regenerate it with another
`--k`. After the smoke passes, use the plan defaults:

```python
!python scripts/generate_root_rollouts.py --split train --load-in-4bit
!python scripts/generate_mc_prefix_labels.py --split train --load-in-4bit
```

Copy-paste bootstrap (after Runtime → GPU):

**Why clone failed:** `kxyxlxex/TokenAwareAI` is **private**. Plain `git clone https://github.com/...` cannot prompt for a password in Colab, so you get `could not read Username` and `/content/TokenAwareAI` never exists. Pick **one** of the three ways below.

### Option A — GitHub token (keep repo private)

1. GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained** (or classic with `repo` scope).
2. Grant read access to `TokenAwareAI`.
3. In Colab, paste the token into the clone URL (do **not** commit or share this cell):

```python
from google.colab import drive
drive.mount("/content/drive")

TOKEN = "ghp_xxxxxxxx"  # your PAT — delete this cell after clone
!git clone https://{TOKEN}@github.com/kxyxlxex/TokenAwareAI.git /content/TokenAwareAI
%cd /content/TokenAwareAI
!pip install -q -r requirements.txt bitsandbytes

import os
os.environ["TOKENAWARE_ROOT"] = "/content/TokenAwareAI"
os.environ["TOKENAWARE_ARTIFACTS"] = "/content/drive/MyDrive/TokenAwareAI/artifacts"

!python scripts/make_splits.py
!python scripts/smoke_instrument.py --load-in-4bit
```

### Option B — Make the repo public (simplest)

GitHub repo → Settings → Danger zone → **Change visibility → Public**. Then the original clone works with no token:

```python
!git clone https://github.com/kxyxlxex/TokenAwareAI.git /content/TokenAwareAI
```

### Option C — Zip upload (no GitHub auth on Colab)

On your laptop (repo root):

```bash
cd /Users/kylexu/TokenAwareAI
git archive -o TokenAwareAI.zip HEAD
```

Upload `TokenAwareAI.zip` to Google Drive (`MyDrive/`), then in Colab:

```python
from google.colab import drive
drive.mount("/content/drive")

!unzip -q /content/drive/MyDrive/TokenAwareAI.zip -d /content/TokenAwareAI
%cd /content/TokenAwareAI
!pip install -q -r requirements.txt bitsandbytes

import os
os.environ["TOKENAWARE_ROOT"] = "/content/TokenAwareAI"
os.environ["TOKENAWARE_ARTIFACTS"] = "/content/drive/MyDrive/TokenAwareAI/artifacts"

!python scripts/make_splits.py
!python scripts/smoke_instrument.py --load-in-4bit
```

Push from your laptop before cloning/uploading, or Colab will get an old tree.

---

## What each file is

### Docs (keep)

| File | Why |
|---|---|
| `current.md` | This file. Stage + inventory. Update as we go. |
| `plan-cost-aware-tree-search.md` | Numbered execution plan with paper-backed sizes. |
| `idea-cost-aware-tree-search.md` | V/T definitions, score, why sibling ranking is the kill test. |
| `budget-aware-ai-literature.md` | Annotated bibliography. |

### Package `src/tokenaware/`

| File | What it does |
|---|---|
| `__init__.py` | Package marker. |
| `paths.py` | Find repo root if run as a script, imported, or pasted after `cd` to the repo. Puts `src/` on `sys.path`. |
| `config.py` | Constants: model id, layers `{9,18,27,36}`, decode (`T=0.7`, top-k 20, top-p 0.95, 1024 cap), split sizes, CoT system prompt. Artifacts → Drive on Colab if mounted. |
| `data.py` | Load MATH train (local parquet **or** Hub `EleutherAI/hendrycks_math`). Stratified 500/2000 split, seed 42. Drops 2 `Level ?` geometry rows. |
| `steps.py` | Parse one-line-per-step CoT; extract `\\boxed{}`; map step ends to token offsets. |
| `scoring.py` | Gold match via `math-verify`, string fallback. |
| `hooks.py` | Forward hooks on residual-stream layers; read `h` at a token index. |
| `generate.py` | Load Qwen3-8B from Hub (or a complete local snapshot). Non-thinking chat template. Generate + cache step-boundary hidden states. `--load-in-4bit` for T4. |
| `mc.py` | Select 25/50/75% prefix states from two root traces and aggregate fresh continuations into empirical V/T labels. |

### Scripts `scripts/`

| File | What it does |
|---|---|
| `make_splits.py` | Writes `artifacts/splits/math_probe_split.json`. No GPU. |
| `smoke_instrument.py` | One train problem: generate, print steps, check 4-layer `h` shapes. **First Colab check.** |
| `generate_root_rollouts.py` | Step 2a: `k=8` root samples per problem. JSONL + `.pt` hidden sidecars. `--limit` / `--offset` for chunks. |
| `generate_mc_prefix_labels.py` | Step 2c: continue six selected prefixes/problem `k=8` times and write empirical `v_mc` and `t_mc_mean`. |

### Tests

| File | What it does |
|---|---|
| `tests/test_steps_and_split.py` | Parse boxed answers; split is 500/2000, no ID leak, all 5 levels. |

### Repo plumbing

| File | Why |
|---|---|
| `requirements.txt` | torch, transformers, datasets, math-verify, … Colab T4 also needs `bitsandbytes`. |
| `.gitignore` | Ignores `models/*`, `datasets/*`, `artifacts/*` (except `.gitkeep`), weights, `.venv`. |

### Local-only (not in git, not required)

| Path | Notes |
|---|---|
| `datasets/` | Optional local MATH/GSM8K. Colab uses the Hub. |
| `artifacts/splits/math_probe_split.json` | Already built on this machine. Rebuild on Colab. |
| `.venv/` | Laptop Python env. Ignore on Colab. |
| `models/` | Empty on purpose. Do not download 8B here. |

---

## Deleted on purpose

- `notebooks/colab_phase0.ipynb` — you paste cells, not open an ipynb.
- `scripts/colab_setup.py` — bootstrap lives in this file.
- `third_party/ReProbe/` — reference clone; protocol is already in `config.py` / `steps.py`.
- Local Qwen download / incomplete shards — Colab fetches the Hub.

---

## Not in the repo yet (later stages)

V-head / T-head training, sibling metrics, tree search, MATH-500 eval.
