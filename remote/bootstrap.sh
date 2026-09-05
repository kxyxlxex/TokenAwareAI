#!/usr/bin/env bash
# Provision a fresh CUDA box for TokenAwareAI probe training.
#
#   ssh ubuntu@<ip>
#   git clone <repo> tokenaware && cd tokenaware
#   export HF_TOKEN=hf_...
#   bash remote/bootstrap.sh
#
# Idempotent: re-running only installs what is missing. Everything heavy lands
# under $TOKENAWARE_DATA (default ~/tokenaware-data) so the repo checkout stays
# small and the artifacts survive a `git clean`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${TOKENAWARE_DATA:-$HOME/tokenaware-data}"
VENV="${TOKENAWARE_VENV:-$DATA_ROOT/venv}"
CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "== TokenAwareAI bootstrap =="
echo "repo:   $REPO_ROOT"
echo "data:   $DATA_ROOT"
echo "venv:   $VENV"

mkdir -p "$DATA_ROOT/artifacts" "$DATA_ROOT/hf-cache"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi
PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "python: $PY_VERSION"
python3 - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit("need Python >= 3.10 for Qwen3 / transformers")
EOF

if [ ! -d "$VENV" ]; then
  echo "-- creating venv"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null

if ! python -c "import torch" 2>/dev/null; then
  echo "-- installing torch from $CUDA_INDEX"
  pip install torch --index-url "$CUDA_INDEX"
fi
echo "-- installing project requirements"
pip install -r "$REPO_ROOT/requirements.txt"

ENV_FILE="$DATA_ROOT/env.sh"
cat > "$ENV_FILE" <<EOF
# source this before running any TokenAwareAI script
export TOKENAWARE_ROOT="$REPO_ROOT"
export TOKENAWARE_ARTIFACTS="$DATA_ROOT/artifacts"
export HF_HOME="$DATA_ROOT/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH="$REPO_ROOT/src:\${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
source "$VENV/bin/activate"
EOF
echo "-- wrote $ENV_FILE"

# shellcheck disable=SC1090
source "$ENV_FILE"

python - <<'EOF'
import shutil
import torch

print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {props.name} {props.total_memory / (1 << 30):.0f} GiB")
total, used, free = shutil.disk_usage("/")
print(f"disk: {free / (1 << 30):.0f} GiB free of {total / (1 << 30):.0f} GiB")
EOF

if [ -z "${HF_TOKEN:-}" ]; then
  cat <<'EOF'

!! HF_TOKEN is not set. The artifact repo is private, so set it before fetching:
     export HF_TOKEN=hf_...
EOF
else
  echo "HF_TOKEN is set (${#HF_TOKEN} chars)"
fi

cat <<EOF

== ready ==
Every shell from now on:
  source $ENV_FILE

Then:
  bash remote/run_phase0.sh              # fetch -> inventory -> cache -> sweep
  bash remote/run_phase0.sh --help
EOF
