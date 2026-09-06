#!/usr/bin/env bash
# Free enough disk to download DeepSeek-R1-Distill-Qwen-7B (~15 GiB).
# You cannot keep Qwen3-8B (~16 GiB) and the 7B reasoner under a ~20 GiB budget.
#
# Safe to delete: HF weight snapshots (re-downloadable).
# Do not delete rollouts/labels/branches until they are on the Hub.
set -euo pipefail
source "${TOKENAWARE_DATA:-$HOME/tokenaware-data}/env.sh" 2>/dev/null || true

echo "== disk =="
df -h "$HOME" "${TOKENAWARE_ARTIFACTS:-$HOME/tokenaware-data/artifacts}" "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null || df -h "$HOME"
echo
echo "== largest HF snapshots =="
du -sh "${HF_HOME:-$HOME/.cache/huggingface}/hub"/models--* 2>/dev/null | sort -h || true
echo
echo "== artifacts =="
du -sh "${TOKENAWARE_ARTIFACTS:-$HOME/tokenaware-data/artifacts}"/* 2>/dev/null | sort -h || true

QWEN="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-8B"
if [ -d "$QWEN" ]; then
  echo
  echo "-- removing Qwen3-8B snapshot (needed to make room for R1-Distill-7B)"
  du -sh "$QWEN"
  rm -rf "$QWEN"
  echo "deleted $QWEN"
fi

echo
echo "== after =="
df -h "$HOME" | tail -1
echo "Need ~16 GiB free before the 7B download. If still short, delete"
echo "  \$TOKENAWARE_ARTIFACTS/cache/pilot  (rebuildable)"
echo "and only then cache/phase0 (rebuildable from rollouts)."
