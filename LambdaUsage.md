# Lambda Cloud usage — full V/T corpus

Last updated 16 Aug 2026.

This run happens entirely on a remote Lambda instance. Code, model weights, MATH split,
rollouts, MC labels, and logs all live on that machine. Your laptop may disconnect, sleep,
or shut down after the job is running in `tmux`. Use `pull_artifacts.py` only when you
want a local backup copy. Lambda continues charging until you **terminate** the instance.

## Where things live

| Item | During the job | After `pull_artifacts` |
|---|---|---|
| Repo / scripts | Lambda `~/TokenAwareAI` | Already on your Mac clone |
| Qwen3-8B weights | Lambda `~/tokenaware-data/huggingface` (~16.4 GB) | Stay on Lambda; do not pull |
| Split JSON | Lambda `$TOKENAWARE_ARTIFACTS/splits/` | Optional local copy |
| Root + MC outputs | Lambda `$TOKENAWARE_ARTIFACTS/` | Local `artifacts/` backup |
| Logs | Lambda `$TOKENAWARE_ARTIFACTS/logs/` | Optional local copy |

Generation scripts never write to your Mac. They only use `$TOKENAWARE_ARTIFACTS` and
`$HF_HOME` on the instance.

## Recommended machine

Use one **H100 PCIe 80 GB** instance when available. It is the practical time/cost choice
for Qwen3-8B generation:

- run Qwen3-8B in BF16; do not use 4-bit on H100;
- use batch 8 for both jobs;
- use one GPU and one process—no distributed setup is required.

Batch 8 is not a guess: the protocol requests exactly `k=8` root samples per problem and
exactly `k=8` continuations per prefix. The scripts keep problem/prefix outputs atomic, so
there are only eight homogeneous requests available at once. A larger `--batch-size` cannot
increase the actual batch. Combining unrelated problems would add prompt-padding waste and
make recovery less granular. Qwen3-8B's BF16 weights are about 16.4 GB; its BF16 KV cache is
147,456 bytes per sequence token, so eight sequences at the conservative ~2K-token
prompt-plus-output bound use about 2.25 GiB of KV cache—well within an 80 GB H100.

Lambda's public price on 16 Aug 2026 is $3.29/GPU-hour for H100 PCIe. Pricing and
availability change, so verify [Lambda pricing](https://lambda.ai/pricing) before launch.
Billing is by the minute while the instance is running, including idle time.

## 1. Storage choice

Use the H100 PCIe instance's included **1 TB root volume**. Expected generated corpus is
about **4.59 GB** (reserve 6 GB). This avoids separate filesystem charges.

Everything for this phase stays on that volume. Your laptop is offline-safe during
generation. Before you terminate the instance, pull a backup with `pull_artifacts.py`
(§7). Paid Lambda filesystem storage is unnecessary for this corpus.

## 2. Connect and put the project on the instance

Push the latest repo from your Mac first, so Lambda clones the current scripts:

```bash
# on your Mac, after committing locally
git push origin main
```

Copy the SSH command from the instance page and run it on your laptop:

```bash
ssh ubuntu@INSTANCE_IP
```

On the instance:

```bash
set -e
export WORK="$HOME/tokenaware-data"
mkdir -p "$WORK/artifacts" "$WORK/huggingface"

git clone https://github.com/kxyxlxex/TokenAwareAI.git "$HOME/TokenAwareAI"
cd "$HOME/TokenAwareAI"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a private repository, use an SSH deploy key or a fine-grained GitHub token. Do not put
a token in a checked-in script or shell history.

Force all outputs and the model cache onto the instance disk:

```bash
export TOKENAWARE_ROOT="$HOME/TokenAwareAI"
export TOKENAWARE_ARTIFACTS="$HOME/tokenaware-data/artifacts"
export HF_HOME="$HOME/tokenaware-data/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
```

Save them for future SSH sessions:

```bash
cat >> "$HOME/.bashrc" <<'EOF'
export TOKENAWARE_ROOT="$HOME/TokenAwareAI"
export TOKENAWARE_ARTIFACTS="$HOME/tokenaware-data/artifacts"
export HF_HOME="$HOME/tokenaware-data/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
EOF
```

Confirm the scripts will write on the cloud, not under a Mac path:

```bash
cd "$HOME/TokenAwareAI"
source .venv/bin/activate
python - <<'PY'
from tokenaware.config import ARTIFACTS_DIR, MODEL_ID, REPO_ROOT
import os
print("REPO_ROOT       =", REPO_ROOT)
print("ARTIFACTS_DIR   =", ARTIFACTS_DIR)
print("HF_HOME         =", os.environ.get("HF_HOME"))
print("MODEL_ID        =", MODEL_ID)
assert str(ARTIFACTS_DIR).startswith(str(os.path.expanduser("~/tokenaware-data"))), ARTIFACTS_DIR
print("cloud paths OK")
PY
```

Optional Hugging Face authentication avoids unauthenticated rate limits:

```bash
huggingface-cli login
```

## 3. Check the GPU and build the deterministic split

```bash
cd "$HOME/TokenAwareAI"
source .venv/bin/activate
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name()); print(torch.cuda.is_bf16_supported())"
python scripts/make_splits.py
ls "$TOKENAWARE_ARTIFACTS/splits/math_probe_split.json"
```

`make_splits.py` writes the train/val problem list into
`$TOKENAWARE_ARTIFACTS/splits/math_probe_split.json` on the instance. Every later script
loads that same file.

## 4. Benchmark before the full bill

Run one 25-problem pilot in `tmux`. `tee` records logs on the instance.

```bash
tmux new -s tokenaware
cd "$HOME/TokenAwareAI"
source .venv/bin/activate
mkdir -p "$TOKENAWARE_ARTIFACTS/logs"

python scripts/generate_root_rollouts.py \
  --split train --limit 25 --k 8 --batch-size 8 --dtype bfloat16 \
  2>&1 | tee "$TOKENAWARE_ARTIFACTS/logs/root-pilot.log"

python scripts/generate_mc_prefix_labels.py \
  --split train --limit 25 --k 8 --batch-size 8 --dtype bfloat16 \
  2>&1 | tee "$TOKENAWARE_ARTIFACTS/logs/mc-pilot.log"
```

These scripts load:

- split: `$TOKENAWARE_ARTIFACTS/splits/math_probe_split.json`
- model: Hub `Qwen/Qwen3-8B` into `$HF_HOME`
- outputs: `$TOKENAWARE_ARTIFACTS/rollouts/...` and `$TOKENAWARE_ARTIFACTS/labels/...`

Detach without stopping it: press `Ctrl-B`, release, then press `D`.

Reconnect later:

```bash
ssh ubuntu@INSTANCE_IP
tmux attach -t tokenaware
```

Your laptop does not need to stay awake or charged after detaching.

Check utilization from a second SSH session:

```bash
watch -n 2 nvidia-smi
```

Use batch 8. A larger value has no effect while `k=8`; a smaller value leaves avoidable
parallelism unused. The pilot is for validating correctness and projecting elapsed time,
not searching batch sizes.

## 5. Run the full training corpus

After the pilot output and labels look correct:

```bash
tmux new -s tokenaware-full
cd "$HOME/TokenAwareAI"
source .venv/bin/activate
mkdir -p "$TOKENAWARE_ARTIFACTS/logs"

python scripts/generate_root_rollouts.py \
  --split train --k 8 --batch-size 8 --dtype bfloat16 \
  2>&1 | tee "$TOKENAWARE_ARTIFACTS/logs/root-full.log"

python scripts/generate_mc_prefix_labels.py \
  --split train --k 8 --batch-size 8 --dtype bfloat16 \
  2>&1 | tee "$TOKENAWARE_ARTIFACTS/logs/mc-full.log"
```

This targets:

- 2,000 problems × 8 root samples = 16,000 root rollouts;
- up to 6 selected prefix states/problem;
- 8 continuations/state = up to 96,000 MC continuations.

The scripts embed their run configuration in each record and skip only compatible outputs.
Root files use temporary writes before publication. MC temporary files publish one prefix
state at a time, so a restarted job resumes within the interrupted problem. Start the same
commands again after any interruption.

## 6. Monitor without keeping the laptop on

After reconnecting:

```bash
tmux attach -t tokenaware-full
```

Or inspect the saved logs:

```bash
tail -f "$TOKENAWARE_ARTIFACTS/logs/root-full.log"
tail -f "$TOKENAWARE_ARTIFACTS/logs/mc-full.log"
```

Count completed problems:

```bash
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["TOKENAWARE_ARTIFACTS"])
print("root:", len(list((root / "rollouts/root/train").glob("*.jsonl"))))
print("mc:", len(list((root / "labels/mc/train").glob("*.jsonl"))))
PY
```

Expected final counts are 2,000 root JSONL files and up to 2,000 MC JSONL files. Some MC
problems may be absent if neither selected root trace contained parseable steps; inspect
the MC log for `skip no parseable prefixes`.

Check available instance space:

```bash
df -h "$HOME"
du -sh "$TOKENAWARE_ARTIFACTS" "$HF_HOME"
```

## 7. When and how to call `pull_artifacts`

`pull_artifacts.py` runs on your **Mac**. It does not run on Lambda. It only mirrors
completed files from the instance to a local backup.

### When

| Situation | Call it? |
|---|---|
| Job still running; you want a mid-run backup | Optional |
| Job finished; before terminating the instance | **Required** |
| Laptop asleep during generation | No — wait until you want a backup |
| After terminating the instance | Too late — data on the instance disk is gone |

### How

On the Mac, with the instance still up and SSH reachable:

```bash
cd /Users/kylexu/TokenAwareAI
python3 scripts/pull_artifacts.py \
  --remote ubuntu@INSTANCE_IP:~/tokenaware-data/artifacts \
  --local /Users/kylexu/TokenAwareAI/artifacts
```

Replace `INSTANCE_IP` with the Lambda public IP. The Mac clone must contain
`scripts/pull_artifacts.py` (push/pull git if needed).

What it does:

- copies **all completed** remote artifacts in one rsync;
- never pulls `.tmp` in-progress files;
- pulls a root problem only when **both** its `.jsonl` and `.pt` exist;
- uses `rsync --checksum`, so already-identical local files are skipped by content;
- does not delete local files that are absent remotely;
- never transfers `~/tokenaware-data/huggingface`.

After the full job, validate on the instance, pull once more, then terminate:

```bash
# on Lambda
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["TOKENAWARE_ARTIFACTS"])
print("root JSONL:", len(list((root / "rollouts/root/train").glob("*.jsonl"))))
print("root PT:", len(list((root / "rollouts/root/train").glob("*.pt"))))
print("MC JSONL:", len(list((root / "labels/mc/train").glob("*.jsonl"))))
PY
du -sh "$TOKENAWARE_ARTIFACTS"
```

```bash
# on Mac
python3 scripts/pull_artifacts.py \
  --remote ubuntu@INSTANCE_IP:~/tokenaware-data/artifacts \
  --local /Users/kylexu/TokenAwareAI/artifacts
```

Verify local counts, then terminate the GPU instance in the Lambda console/API. Closing
SSH, detaching `tmux`, or shutting down your laptop does not stop GPU billing.

## Expected storage

Hidden tensors are exact once average parsed step count `S` is known:

```text
16,000 rollouts × S steps × 4 layers × 4,096 × 2 bytes
= 524,288,000 × S bytes
```

The smoke run had 6 steps / 144 tokens = 24 tokens/step. Applying that to the plan's
204-token MATH mean gives `S≈8.5`:

| Component | Size |
|---|---:|
| Hidden `.pt` tensors at `S=8.5` | **4.456 GB** |
| `.pt` pickle/file overhead | 0.018 GB |
| Root JSONL | 0.061 GB |
| MC JSONL | 0.055 GB |
| Split + logs | 0.004 GB |
| **Total to copy home** | **≈4.59 GB** |

At `S=6` / `10` / `12`, totals are about **3.28 / 5.38 / 6.43 GB**. Reserve **6 GB** on
the Mac for the backup. The 16.4 GB model cache stays on Lambda and is not copied.

## Operational cautions

- Use BF16 on H100; this Lambda pipeline intentionally has no 4-bit path.
- Do not launch root and MC scripts concurrently on the same GPU.
- Do not run multiple Python worker processes on one GPU. Use `--batch-size`.
- Do not terminate the instance until artifacts have been copied home and verified.
- The H100 still needs a measured pilot. Price estimates made from T4 throughput are not a
  reliable bill forecast.
