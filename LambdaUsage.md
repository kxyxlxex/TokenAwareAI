# Lambda Cloud usage — full V/T corpus

Last updated 16 Aug 2026.

This run happens on a remote Lambda instance. Your laptop may disconnect, sleep, or shut
down after the job is running in `tmux`. Lambda continues charging until you **terminate**
the instance.

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

## 1. Create persistent storage first

In the Lambda Cloud console:

1. Open **Filesystems**.
2. Create a filesystem in the same region where an H100 PCIe is available.
3. Name it `tokenaware-data`.
4. Allocate **50 GB** or Lambda's larger minimum. The expected persistent footprint is
   under 25 GB; 50 GB leaves safe headroom.
5. Launch the H100 in that same region and attach `tokenaware-data`.

The exact mount path is shown in the instance/Filesystem UI. Confirm it over SSH:

```bash
df -h
ls /lambda
```

In this guide the attached mount is called `/lambda/nfs`. If Lambda shows another path,
replace `/lambda/nfs` everywhere below. Never assume a mount exists—verify with `df -h`.

Persistent filesystems continue to incur storage charges after the GPU instance is
terminated. Delete the filesystem only after downloading or otherwise backing up results.

## 2. Connect and prepare directories

Copy the SSH command from the instance page and run it on your laptop:

```bash
ssh ubuntu@INSTANCE_IP
```

On the instance:

```bash
set -e
export PERSIST=/lambda/nfs
test -d "$PERSIST"
mkdir -p "$PERSIST/TokenAwareAI/artifacts" "$PERSIST/huggingface"

git clone https://github.com/kxyxlxex/TokenAwareAI.git "$HOME/TokenAwareAI"
cd "$HOME/TokenAwareAI"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a private repository, use an SSH deploy key or a fine-grained GitHub token. Do not put
a token in a checked-in script or shell history.

Set persistent paths:

```bash
export TOKENAWARE_ROOT="$HOME/TokenAwareAI"
export TOKENAWARE_ARTIFACTS="$PERSIST/TokenAwareAI/artifacts"
export HF_HOME="$PERSIST/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
```

Save them for future SSH sessions:

```bash
cat >> "$HOME/.bashrc" <<'EOF'
export PERSIST=/lambda/nfs
export TOKENAWARE_ROOT="$HOME/TokenAwareAI"
export TOKENAWARE_ARTIFACTS="$PERSIST/TokenAwareAI/artifacts"
export HF_HOME="$PERSIST/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
EOF
```

If your mount differs, edit `PERSIST` before appending this block.

Optional Hugging Face authentication avoids unauthenticated rate limits:

```bash
source "$HOME/TokenAwareAI/.venv/bin/activate"
huggingface-cli login
```

## 3. Check the GPU and build the deterministic split

```bash
cd "$HOME/TokenAwareAI"
source .venv/bin/activate
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name()); print(torch.cuda.is_bf16_supported())"
python scripts/make_splits.py
```

The split and every generated artifact now live on the persistent filesystem, not on the
instance's disposable root disk.

## 4. Benchmark before the full bill

Run one 25-problem pilot in `tmux`. `tee` records logs persistently.

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

Or inspect persistent logs:

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

Check available persistent space:

```bash
df -h "$PERSIST"
du -sh "$TOKENAWARE_ARTIFACTS" "$HF_HOME"
```

## 7. Validate and stop billing

When both commands finish:

```bash
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

Download or sync a backup if desired. Then use the Lambda console/API to **terminate the
GPU instance**. Closing SSH, detaching `tmux`, or shutting down your laptop does not stop
GPU billing.

Keep the filesystem for probe training or delete it after a verified backup. Filesystem
storage is billed separately while it exists.

## Expected storage

Using six parsed steps per root rollout as the current measured planning point:

- hidden tensors:
  `16,000 × 6 × 4 layers × 4,096 values × 2 bytes ≈ 3.15 GB`;
- root JSONL text and metadata: approximately 0.05–0.2 GB;
- MC JSONL prefixes, continuation text, and labels: approximately 0.1–0.5 GB;
- split and logs: under 0.1 GB.

At 4–10 average steps, hidden tensors span roughly 2.1–5.2 GB. Allow **4–7 GB for
generated artifacts** because step count and text length vary. The
Hugging Face BF16 model cache is approximately 16.4 GB, bringing the expected persistent
total to roughly **21–24 GB**. Python packages live on the instance root disk in this guide
and are not included. A 50 GB persistent filesystem is therefore sufficient with more than
2× expected headroom.

## Operational cautions

- Use BF16 on H100; this Lambda pipeline intentionally has no 4-bit path.
- Do not launch root and MC scripts concurrently on the same GPU.
- Do not run multiple Python worker processes on one GPU. Use `--batch-size`.
- Do not terminate the instance until persistent-path counts and logs are checked.
- The H100 still needs a measured pilot. Price estimates made from T4 throughput are not a
  reliable bill forecast.
