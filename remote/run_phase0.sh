#!/usr/bin/env bash
# Phase 0 end to end: fetch corpus -> audit -> build cache -> sweep probes ->
# evaluate -> push results. Every stage is skippable and resumable.
#
#   source ~/tokenaware-data/env.sh
#   bash remote/run_phase0.sh
#   bash remote/run_phase0.sh --stages cache,sweep
#   bash remote/run_phase0.sh --branches 500        # add true-sibling data first
#
# Long stages should run under tmux:
#   tmux new -s phase0 'bash remote/run_phase0.sh 2>&1 | tee -a $TOKENAWARE_ARTIFACTS/logs/phase0.log'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="${TOKENAWARE_ARTIFACTS:-$REPO_ROOT/artifacts}"
CACHE_DIR="${TOKENAWARE_CACHE:-$ARTIFACTS/cache/phase0}"
SWEEP_DIR="${TOKENAWARE_SWEEP:-$ARTIFACTS/sweeps/phase0}"
STAGES="fetch,inventory,cache,sweep,evaluate,push"
PRESET="full"
BRANCH_PROBLEMS=0
DEVICE="cuda"
EXTRA_FETCH=""

usage() {
  cat <<'EOF'
usage: run_phase0.sh [options]

  --stages LIST      comma-separated subset of:
                     fetch,inventory,branches,cache,sweep,evaluate,push
                     (default: fetch,inventory,cache,sweep,evaluate,push)
  --preset NAME      sweep preset: layers|arch|controls|seeds|full (default full)
  --branches N       generate true sibling states for N problems
                     (implies the `branches` stage; needs the model on the box)
  --device DEV       cuda|cpu (default cuda)
  --fetch-args STR   extra args forwarded to fetch_artifacts_hf.py
  -h, --help         this message
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --stages) STAGES="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --branches) BRANCH_PROBLEMS="$2"; STAGES="${STAGES/cache/branches,cache}"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --fetch-args) EXTRA_FETCH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option $1" >&2; usage; exit 2 ;;
  esac
done

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
banner() { printf '\n===== %s (%s) =====\n' "$1" "$(date -Is)"; }

cd "$REPO_ROOT"
mkdir -p "$ARTIFACTS/logs" "$ARTIFACTS/reports"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

if has_stage fetch; then
  banner "fetch corpus from Hugging Face"
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set; the artifact repo is private" >&2
    exit 2
  fi
  # shellcheck disable=SC2086
  python scripts/fetch_artifacts_hf.py --dest "$ARTIFACTS" $EXTRA_FETCH
fi

if has_stage inventory; then
  banner "audit corpus"
  python scripts/inventory_artifacts.py --artifacts "$ARTIFACTS" --deep \
    | tee "$ARTIFACTS/logs/inventory.txt"
fi

if has_stage branches && [ "$BRANCH_PROBLEMS" -gt 0 ]; then
  banner "generate true sibling branches ($BRANCH_PROBLEMS problems)"
  python scripts/generate_sibling_branches.py \
    --split train --problems "$BRANCH_PROBLEMS" --dtype bfloat16 \
    2>&1 | tee -a "$ARTIFACTS/logs/branches.log"
fi

if has_stage cache; then
  banner "build probe cache"
  python scripts/build_probe_cache.py --artifacts "$ARTIFACTS" --out "$CACHE_DIR" \
    2>&1 | tee -a "$ARTIFACTS/logs/cache.log"
fi

if has_stage sweep; then
  banner "probe sweep (preset=$PRESET)"
  python scripts/sweep_probes.py --cache "$CACHE_DIR" --out "$SWEEP_DIR" \
    --preset "$PRESET" --device "$DEVICE" --save-checkpoints \
    2>&1 | tee -a "$ARTIFACTS/logs/sweep.log"
fi

if has_stage evaluate; then
  banner "evaluate saved probes"
  if compgen -G "$SWEEP_DIR/*/probe.pt" > /dev/null; then
    python scripts/evaluate_probes.py --cache "$CACHE_DIR" \
      --probe-dir "$SWEEP_DIR" --out "$ARTIFACTS/reports/phase0" \
      --budgets 32 64 128 256 512 1024 --device "$DEVICE" \
      2>&1 | tee "$ARTIFACTS/logs/evaluate.log"
  else
    echo "no probe.pt files under $SWEEP_DIR; run the sweep with --save-checkpoints"
  fi
fi

if has_stage push; then
  banner "push results to Hugging Face"
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN not set; skipping push" >&2
  else
    python scripts/push_artifacts_hf.py --artifacts "$ARTIFACTS" --preset results
    if [ "$BRANCH_PROBLEMS" -gt 0 ]; then
      python scripts/push_artifacts_hf.py --artifacts "$ARTIFACTS" --preset branches
    fi
  fi
fi

banner "done"
echo "verdicts:"
grep -h '"decision"' "$ARTIFACTS/reports/phase0/summary.json" 2>/dev/null || \
  echo "  (see $SWEEP_DIR/sweep.csv)"
