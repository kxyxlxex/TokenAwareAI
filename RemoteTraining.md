# Remote V+T probe training — plan and runbook

Phase 0 decision run, executed entirely on a rented GPU. The corpus lives in a
private Hugging Face dataset repo; nothing is downloaded to a laptop. Companion
docs: `plan-cost-aware-tree-search.md` (the experiment), `LambdaUsage.md` (the
generation jobs that produced the corpus).

---

## 1. What decides whether V+T works

The claim is **cost-aware node selection**: `Score(s) = V̂(s) · P̂(T(s) ≤ B_rem)`
beats `Score(s) = V̂(s)` under a matched output-token budget. Six choices below
are what give that claim its best shot. Each is implemented, not aspirational.

### 1.1 Train on outcome draws, not on label means

Every training example is one realised continuation: `(h, correct, length,
censored)`. A same-trace state contributes one draw; a Monte-Carlo state
contributes `k`. Bernoulli and discretised-count likelihoods are unbiased on
single draws, so the ~135K "cheap, biased" same-trace labels stop being second
class and become the main training corpus — roughly **8× more supervision** than
the MC subset alone.

The corpus supports this. On the 25 MC-labelled problems available locally, the
source trace's realised remainder and the mean of 8 fresh continuations agree
closely:

| depth fraction | source trace remainder | mean of 8 fresh continuations |
|---|---|---|
| 0.25 | 126.1 tokens | 132.9 |
| 0.50 | 77.9 | 83.5 |
| 0.75 | 33.0 | 35.3 |

Median ratio 1.03. Same-trace correctness (0.70) also tracks `v_mc` (0.75).
Two label sources, one estimand.

Schedule: **stage A** on all draws, **stage B** fine-tunes on MC draws only at
`lr/10`, so the final weights sit on the low-noise label set while still having
seen the large one. `--no-same-trace` and `--no-mc` run the ablations.

### 1.2 A distributional T head, because the score needs a CDF

`P(T ≤ B_rem)` is what the score consumes. Predicting a point `T̂` and then
guessing a spread throws away exactly the quantity being used. The primary head
emits logits over log-spaced remaining-token bins, so `P(T ≤ B)` is a partial
sum — no parametric assumption, and the quantiles come free.

The `n_bins - 1` finite edges end exactly at the 1024-token generation cap and
the final bin is the overflow. So the finite bins mean precisely "terminated
within the cap", and `P(T ≤ B)` for `B ≥ 1024` tops out at the probability of
terminating at all. A trace that never finished did not finish inside any budget.

**Censoring is kept, not dropped.** *How Much is Left?* discards truncated
sequences; that biases V by survivorship and removes exactly the tail a budget
cares about. A censored draw here contributes `−log P(T ≥ its bin)`, which is all
the observation supports. Quantile and point heads are still available
(`--t-head quantile|point`) as the published-recipe comparisons.

### 1.3 The decision metric is budget utility, computed offline

For a state with continuations `i`, define

```text
u_B(s) = mean_i  1[ correct_i  and  len_i <= B ]
```

the probability that continuing from `s` both finishes inside the remaining
budget and is right. This is computable from the MC artifacts already on disk,
and it is exactly what a cost-aware selector should maximise. Ranking siblings by
`V` alone versus by `V · P(T ≤ B)` and grading both against `u_B` gives a
budget-matched read on whether T adds anything — **before** any search code
exists. Reported with a paired bootstrap CI over sibling groups, plus win/loss
counts, against oracle and random-choice references.

Selectors compared at every budget: `v_only`, `v_times_feasibility`,
`bang_per_buck` (`V/(T+ε)`), `feasibility_only`, `position_baseline`, and
`oracle_v_mc`.

### 1.4 One deviation from `plan-cost-aware-tree-search.md`

The plan lists as a *hard* kill: "partial correlation of T with outcome given V
consistent with 0". Under a token budget that criterion is mis-specified. T can
be conditionally independent of plain correctness given V and still be decisive,
because the budget makes length matter on its own. The hard criteria used here
are the budget-aware versions:

* partial Spearman of `−T̂` with `u_B` controlling for `V̂`, at some budget;
* bootstrap CI lower bound of `u_B(V·P(T≤B)) − u_B(V only)` above 0, at some budget.

The plan's original correlation is still computed and printed, labelled
`t_collinearity_reference` and marked informational.

### 1.5 Two baselines the probe must beat, or the number means nothing

**Position-only.** `tokens_so_far` predicts remaining length well by itself. A
nonparametric binned-mean predictor of V and T from prefix position alone is
fitted on the train split's same-trace states (tens of thousands of rows, so it
is well resolved and not a straw man) and reported beside every probe metric.
"Probe beats position-only on sibling T ranking" is a hard criterion.

Why this matters, from the local audit: pooling sibling pairs across depths gave
T pairwise accuracy **0.89**, while restricting to pairs at the same depth
fraction gave **0.54**. Depth was doing nearly all the work. A single "sibling
accuracy" number without this control is not interpretable.

**Label-noise ceiling.** With `k=8`, the empirical `T̂` itself misranks siblings.
Splitting each state's draws in half and ranking one half against the other
measures how well the label can do; on the local slice that ceiling was **0.87**
for same-depth pairs. Probe accuracy is reported against that, not against 1.0.
Gap-thresholded variants (`min |ΔT̂| > 20` and `> 50` tokens) separate "cannot
rank" from "coin-flipping on near-ties".

### 1.6 Real siblings, because the existing corpus has none

The kill criterion is *within-problem sibling* ranking, but the root/MC corpus
compares two independent traces at 25/50/75% depth: pairs that differ in content
*and* absolute position. Tree search compares candidates that share a parent
prefix and differ by one step.

`scripts/generate_sibling_branches.py` builds those: parent prefix → `n` sampled
next steps (deduplicated) → `k` continuations each, with the hidden vector
captured at each sibling's own last token. When branch data is present the
evaluator promotes `branch` to the primary grouping automatically; otherwise it
falls back to `problem_fraction` and says so in the report.

Cost at defaults (1 parent, 3 branches, `k=8`): ~2.6K output tokens per problem.
500 problems ≈ 1.3M tokens, well under an hour on an A100. **This is the single
highest-value GPU spend left**, because it is the only thing that makes the
headline number mean what the paper says it means.

---

## 2. Findings from the local corpus audit

Run against the 660 root problems and 25 MC problems currently on disk. Levels
1–3 only, so treat every number as indicative, not final.

| quantity | value | consequence |
|---|---|---|
| trace correctness | 81.2% | V headroom is thin |
| `v_mc ∈ {0,1}` | 74% of MC states | V ranking has few decidable pairs |
| tokens per trace | mean 240, p50 197, p90 409, p99 945 | budget grid 64/128/256/512 is well placed |
| steps per trace | mean 8.5 | ~135K same-trace states at 2,000 problems |
| truncation rate | 0.87% | below the 5% line; censoring is a tail effect, not a bias |
| step→hidden alignment | 44,766 / 44,766 | `.pt` sidecars are sound; deep check found 0 failures |
| MC joinable to root | 150 / 150 | the join key `(source_sample_id, step_index)` is reliable |

Two things to act on:

**The split list is ordered by level.** `stratified_split` appends level 1, then
2, and so on, so `--offset/--limit` prefixes are difficulty-biased. Ordinals
1–660 are levels 1–3; levels 4–5 sit in 661–2000. `inventory_artifacts.py` now
warns when fewer than five levels are present. If the Hub corpus is also a
prefix, the go/no-go cannot be run on it — Level 4–5 is where allocation is
supposed to matter, and it would be entirely missing.

**V saturation is the main threat to a positive result.** At 81% pass rate the
V-only arm is strong and there is little room above it. Mitigations, in order:
report per-level and headline Level 4–5; use tight budgets (64/128) where
feasibility binds hardest; and rely on `u_B` rather than raw correctness, since
`u_B` is far from saturated even when V is.

---

## 3. Deploy

### 3.1 Provision

```bash
ssh ubuntu@<gpu-ip>
git clone <this-repo> tokenaware && cd tokenaware
export HF_TOKEN=hf_...                 # the artifact repo is private
bash remote/bootstrap.sh
source ~/tokenaware-data/env.sh         # every new shell
```

`bootstrap.sh` creates a venv outside the checkout, installs CUDA torch plus
requirements, writes `env.sh` (sets `TOKENAWARE_ARTIFACTS`, `HF_HOME`,
`PYTHONPATH`), and prints GPU and disk facts. Re-running is safe.

Sizing: corpus ~4.6 GiB, probe cache ~4.4 GiB (135K states × 4096 × fp16 × 4
layers), model weights ~16 GiB. **40 GiB disk** is comfortable. Any 24 GiB+ GPU
trains the probes; the A100 is only needed if you also generate branches.

### 3.2 One command

```bash
tmux new -s phase0 'bash remote/run_phase0.sh 2>&1 | tee -a $TOKENAWARE_ARTIFACTS/logs/phase0.log'
```

Stages: `fetch → inventory → cache → sweep → evaluate → push`. Each is skippable
and resumable via `--stages`.

Recommended first pass, which adds real siblings before the sweep:

```bash
bash remote/run_phase0.sh --branches 500 --preset full
```

### 3.3 Or step by step

```bash
# 1. what is actually in the Hub repo
python scripts/fetch_artifacts_hf.py --list
python scripts/fetch_artifacts_hf.py                      # download + extract

# 2. audit before spending anything. Read the WARNINGS block.
python scripts/inventory_artifacts.py --deep

# 3. optional but recommended: true sibling states
python scripts/generate_sibling_branches.py --split train --problems 25 --dtype bfloat16   # pilot
python scripts/generate_sibling_branches.py --split train --problems 500 --dtype bfloat16

# 4. ETL (~30 s for 2,000 problems)
python scripts/build_probe_cache.py --out $TOKENAWARE_ARTIFACTS/cache/phase0

# 5. sweep: layers {9,18,27,36} x architectures x controls
python scripts/sweep_probes.py --cache $TOKENAWARE_ARTIFACTS/cache/phase0 \
    --preset full --save-checkpoints

# 6. full metric suite on a wide budget grid
python scripts/evaluate_probes.py --cache $TOKENAWARE_ARTIFACTS/cache/phase0 \
    --probe-dir $TOKENAWARE_ARTIFACTS/sweeps/phase0 \
    --budgets 32 64 128 256 512 1024 --out $TOKENAWARE_ARTIFACTS/reports/phase0

# 7. get everything off the box BEFORE terminating
python scripts/push_artifacts_hf.py --preset results
python scripts/push_artifacts_hf.py --preset branches
```

Missing corpus pieces are generated with the existing scripts, e.g. root
rollouts for the val split:

```bash
python scripts/generate_root_rollouts.py --split val --dtype bfloat16
python scripts/generate_mc_prefix_labels.py --split val --k 32 --dtype bfloat16
```

`k=32` on val cuts the V label SE from 0.177 to 0.088, which is the difference
between a noisy go/no-go and a clean one. `build_probe_cache.py --val-source
auto` uses the real val corpus once ≥150 of its problems have MC labels, and
otherwise hash-holds-out 20% of train problems; either way the split is
problem-disjoint and the choice is recorded in the cache metadata.

### 3.4 Probe-val selection

| situation | what happens |
|---|---|
| val corpus has ≥150 MC-labelled problems | that corpus is probe-val, all of train is probe-train |
| otherwise | 20% of train problems, chosen by hashing `problem_id`, become probe-val |

Never a leak either way: assignment is per problem, and states inherit it.

---

## 4. Reading the output

`sweep.csv` has one row per run. Ranked printout is by primary sibling T pairwise
accuracy. The columns that matter, in order:

1. `sibling_t_pairwise` against `sibling_t_ceiling` and
   `sibling_t_position_baseline` — the go/no-go, with its ceiling and its control.
2. `vt_gain_delta` / `vt_gain_ci_low` / `vt_win_rate` — the headline claim.
3. `v_auroc`, `t_mae_reduction_pct`, `t_spearman` — sanity, comparable to
   published numbers. If these are bad, the probe is broken; that is a different
   problem from "T does not help".
4. `grouping` — `branch` means true siblings, `problem_fraction` means the proxy.

`evaluate_probes.py` prints a decision per checkpoint:

| decision | meaning | next move |
|---|---|---|
| `PROCEED_TO_SEARCH` | all criteria passed | build the tree search (plan Step 6) |
| `FIX_PROBE` | soft criteria failed | instrumentation or labels are wrong, not the idea |
| `KILL` | a hard criterion failed | pivot per plan Step 5 |

Thresholds, all overridable in `metrics.KILL_THRESHOLDS`: sibling T pairwise
≥ 0.60, V AUROC ≥ 0.65, T Spearman ≥ 0.30, q90 coverage ≥ 0.70, T MAE better
than the median predictor, probe better than position-only, and the two
budget-aware criteria from §1.4.

### If the verdict is KILL

Check in this order before believing it.

1. **Is `grouping` = `problem_fraction`?** Then you measured the proxy. Generate
   branches and re-run; that alone can move the number in either direction.
2. **Is `sibling_t_ceiling` near 0.5?** The labels cannot rank these pairs at
   `k=8`. Raise `k` on val before concluding anything about the probe.
3. **Did `t_pairwise_gap50` pass while `t_pairwise` failed?** The probe ranks
   clearly-separated siblings and coin-flips on near-ties. That is a usable
   result: gate on feasibility instead of ranking on T.
4. **Are levels 4–5 present?** If not, you tested on the easy half.

Then pivot per plan Step 5: feasibility gating (`feasibility_only` is already a
reported selector), or the probe-cost accounting paper.

---

## 5. File map

| path | role |
|---|---|
| `remote/bootstrap.sh` | provision a CUDA box, write `env.sh` |
| `remote/run_phase0.sh` | staged orchestration of everything below |
| `scripts/fetch_artifacts_hf.py` | pull + extract corpus archives from the Hub |
| `scripts/push_artifacts_hf.py` | send reports / checkpoints / new data back |
| `scripts/inventory_artifacts.py` | corpus audit, coverage, distributions, warnings |
| `scripts/build_probe_cache.py` | join root `.jsonl`+`.pt`, MC labels, branches |
| `scripts/train_probe.py` | one probe, one report |
| `scripts/sweep_probes.py` | layer × architecture × control grid |
| `scripts/evaluate_probes.py` | full metric suite + verdict |
| `scripts/generate_sibling_branches.py` | true sibling states + MC labels |
| `src/tokenaware/hfio.py` | Hub transfer, safe archive extraction |
| `src/tokenaware/probes/cache.py` | cache schema, memmap writer/reader |
| `src/tokenaware/probes/features.py` | causal features, outcome-draw tables |
| `src/tokenaware/probes/heads.py` | trunks and V/T heads |
| `src/tokenaware/probes/losses.py` | BCE, censored discretised NLL, pinball, L1 |
| `src/tokenaware/probes/metrics.py` | global, sibling, budget-utility, verdict |
| `src/tokenaware/probes/train.py` | two-stage training loop, save/load |
| `src/tokenaware/probes/evaluate.py` | prediction, baselines, report assembly |
| `tests/test_probes.py` | 32 tests; numpy-only ones run without torch |

### Cache layout

```text
<cache>/meta.json          layers, hidden dim, problems, groups, provenance
<cache>/hidden_L{n}.npy    float16 [n_states, 4096], one file per layer
<cache>/states.npz         one row per step boundary that has a hidden vector
<cache>/mc.npz             MC labels + per-continuation length/correct/truncated
```

Rows of one `(problem, sample)` trace are contiguous and ordered by step, so a
state's history is `rows[hist_start : row + 1]` — that is what the `attn` trunk
and the `--delta` feature read.

### Feature legality

Allowed, because search can compute them from the prefix: `tokens_so_far`,
`step_index`, problem length, level, subject. **Forbidden:** `n_steps` and any
depth *fraction* within the finished trace — they encode the future. The cache
stores `n_steps` for auditing and `features.py` never reads it.

---

## 6. Budget

| stage | cost |
|---|---|
| fetch corpus | minutes, network-bound |
| inventory `--deep` | ~1 min for 2,000 problems |
| sibling branches, 500 problems | ~1.3M output tokens, well under an hour on A100 |
| build cache | ~30 s, ~4.4 GiB |
| one probe (9.5M params) | seconds per epoch on an A100; 4 s/epoch for 37K draws on a laptop CPU |
| `--preset full` sweep | ~20 runs, well under an hour |
| val MC at `k=32`, 500 problems | ~24K continuations, the largest optional job |

Phase 0 is a few GPU-hours, not GPU-days. Spend the time on branches and on val
`k=32`, not on bigger probes: 9.5M parameters already sits under ReProbe's
"<10M" banner and the data, not capacity, is the binding constraint.
