# Current state — TokenAwareAI

Last updated 17 Aug 2026.

## Goal

Build cost-aware tree search for math reasoning. Frozen Qwen3-8B generates reasoning;
small probes read hidden states to estimate:

- **V:** probability that continuing from a prefix produces a correct answer;
- **T:** expected remaining generation tokens.

The experiment compares V-only node selection against V+T selection under output-token
budgets.

## Current stage

Phase 0 data instrumentation is implemented and passed a one-problem Colab T4 smoke test.
The production target is now a Lambda A100 80 GB instance:

1. Build the deterministic 2,000-train/500-validation MATH split.
2. Generate 8 root rollouts for every train problem.
3. Select 25/50/75% states from two traces/problem.
4. Generate 8 fresh continuations from each selected state for MC V/T labels.
5. Train and evaluate probes; apply the kill criteria before implementing search.

Exact Lambda setup and commands are in `LambdaUsage.md`.

## Production configuration

- Model: `Qwen/Qwen3-8B`, non-thinking mode
- Precision: BF16 on A100
- Batch size: 8 (effective max with current per-problem/prefix atomic scripts)
- Root corpus: 2,000 × 8 = 16,000 rollouts
- MC corpus: up to 2,000 × 6 × 8 = 96,000 continuations
- Remote output: `$TOKENAWARE_ARTIFACTS`
- Default local output: `artifacts/`

Batch 8 is exact for the current protocol: each atomic problem/prefix has `k=8` samples,
and the scripts only batch those eight homogeneous requests. A larger `--batch-size` cannot
collect more than eight requests without a packing change. An A100 80 GB is not the
constraint: Qwen3-8B BF16 KV is 147,456 bytes/token, so batch 8 at 1024 tokens is ~1.1 GiB.

## Artifact layout

Generated files are ignored by `artifacts/.gitignore`. During the Lambda job, everything
lives on the instance (`TOKENAWARE_ARTIFACTS`, `HF_HOME`). `scripts/pull_artifacts.py` is
a Mac-only backup of completed files; it is required once before you terminate the
instance, optional mid-job. Do not attach a paid Lambda filesystem for this corpus.

```text
artifacts/
  splits/math_probe_split.json
  rollouts/root/train/
    p0001_algebra_<hash>.jsonl
    p0001_algebra_<hash>.pt
  labels/mc/train/
    p0001_algebra_<hash>.jsonl
  logs/
```

`p0001` is the stable 1-based ordinal in the selected split, making completed ranges
visible and auditable. Root JSONL contains text and metadata. Root `.pt` contains four
FP16 tensors per rollout, one `[steps, 4096]` tensor per probe layer. MC JSONL contains
exact generated prefix token IDs, continuation text, and empirical `v_mc`/`t_mc_mean`.
Run configuration is embedded in every record so incompatible `k`/model/dtype outputs are
regenerated instead of silently skipped. Root writes are staged through temporary files;
MC temporary files resume at prefix-state granularity.

## File inventory

### Execution

- `scripts/make_splits.py` — deterministic MATH train/validation split.
- `scripts/smoke_instrument.py` — one rollout validating generation, parsing, scoring,
  and hidden-state extraction.
- `scripts/generate_root_rollouts.py` — batched root generation and hidden-state saving.
- `scripts/generate_mc_prefix_labels.py` — batched fresh continuations and MC labels.
- `scripts/pull_artifacts.py` — Mac-only backup of completed Lambda artifacts (not used on the instance).

### Package

- `src/tokenaware/config.py` — model, decoding, layers, sizes, and artifact paths.
- `src/tokenaware/data.py` — MATH loading and stratified split.
- `src/tokenaware/generate.py` — model loading, batched generation, hidden-state replay.
- `src/tokenaware/hooks.py` — selected-layer residual-stream capture.
- `src/tokenaware/steps.py` — reasoning-step parsing and token alignment.
- `src/tokenaware/scoring.py` — mathematical answer verification.
- `src/tokenaware/mc.py` — prefix selection and MC aggregation.
- `src/tokenaware/artifacts.py` — stable numbered artifact names.

### Documentation

- `LambdaUsage.md` — production deployment and recovery procedure.
- `plan-cost-aware-tree-search.md` — experimental plan and kill criteria.
- `idea-cost-aware-tree-search.md` — research framing.
- `budget-aware-ai-literature.md` — supporting literature.

## Probe training (Phase 0 decision)

Implemented and tested. The corpus now lives in a private Hugging Face dataset repo
(`kxyxlxex/tokenaware-artifacts`) as archives, and training runs on a rented GPU that
fetches from the Hub; nothing large is kept locally. Full plan, rationale, and command
sequence: `RemoteTraining.md`.

### Execution

- `remote/bootstrap.sh` — provision a CUDA box and write `env.sh`.
- `remote/run_phase0.sh` — fetch, inventory, branches, cache, sweep, evaluate, push.
- `scripts/fetch_artifacts_hf.py` / `scripts/push_artifacts_hf.py` — Hub transfer.
- `scripts/inventory_artifacts.py` — corpus audit and warnings.
- `scripts/build_probe_cache.py` — join root `.jsonl`+`.pt`, MC labels, sibling branches.
- `scripts/train_probe.py`, `scripts/sweep_probes.py`, `scripts/evaluate_probes.py`.
- `scripts/generate_sibling_branches.py` — true sibling states (shared parent prefix).

### Package

- `src/tokenaware/hfio.py` — Hub transfer and safe archive extraction.
- `src/tokenaware/probes/cache.py` — memmap cache schema, writer, reader.
- `src/tokenaware/probes/features.py` — causal features and outcome-draw tables.
- `src/tokenaware/probes/heads.py` — trunks plus V and T heads.
- `src/tokenaware/probes/losses.py` — BCE, censored discretised NLL, pinball, L1.
- `src/tokenaware/probes/metrics.py` — global, sibling, budget-utility, kill verdict.
- `src/tokenaware/probes/train.py` — two-stage training loop.
- `src/tokenaware/probes/evaluate.py` — prediction, baselines, report assembly.

## Next action

Fetch the Hub corpus onto a GPU box and run `scripts/inventory_artifacts.py --deep`
first. The audit decides what else needs generating before the go/no-go is meaningful:

1. Confirm all five difficulty levels are present. The split list is level-ordered, so a
   prefix of the corpus is levels 1–3 only, and Level 4–5 is where allocation should
   matter.
2. Confirm MC coverage across the 2,000 train problems.
3. Generate sibling branches (~500 problems) so the sibling metric uses real siblings
   instead of the two-trace proxy.
4. Optionally add val MC labels at `k=32` to halve the V label noise.

Then sweep layers {9,18,27,36} and read the verdict. Known risk from the local audit:
81% trace correctness and 74% of MC states with `v_mc ∈ {0,1}` leave little headroom
above the V-only arm, so report per level and headline the tight-budget, Level 4–5 cells.
