# Current state — TokenAwareAI

Last updated 16 Aug 2026.

## Goal

Build cost-aware tree search for math reasoning. Frozen Qwen3-8B generates reasoning;
small probes read hidden states to estimate:

- **V:** probability that continuing from a prefix produces a correct answer;
- **T:** expected remaining generation tokens.

The experiment compares V-only node selection against V+T selection under output-token
budgets.

## Current stage

Phase 0 data instrumentation is implemented and passed a one-problem Colab T4 smoke test.
The production target is now a persistent Lambda H100 PCIe instance:

1. Build the deterministic 2,000-train/500-validation MATH split.
2. Generate 8 root rollouts for every train problem.
3. Select 25/50/75% states from two traces/problem.
4. Generate 8 fresh continuations from each selected state for MC V/T labels.
5. Train and evaluate probes; apply the kill criteria before implementing search.

Exact Lambda setup and commands are in `LambdaUsage.md`.

## Production configuration

- Model: `Qwen/Qwen3-8B`, non-thinking mode
- Precision: BF16 on H100
- Batch size: 8
- Root corpus: 2,000 × 8 = 16,000 rollouts
- MC corpus: up to 2,000 × 6 × 8 = 96,000 continuations
- Persistent output: `$TOKENAWARE_ARTIFACTS`
- Default local output: `artifacts/`

Batch 8 is exact for the current protocol: each atomic problem/prefix has `k=8` samples.
A larger configured batch cannot collect more than eight requests.

## Artifact layout

Generated files are ignored by `artifacts/.gitignore`. On Lambda, point
`TOKENAWARE_ARTIFACTS` at the attached persistent filesystem.

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

## Next action

Run the 25-problem Lambda H100 pilot in `LambdaUsage.md`. Validate numbered files,
correctness distribution, truncation, output size, and elapsed time. Then rerun the same
commands without `--limit`; completed files are skipped.
