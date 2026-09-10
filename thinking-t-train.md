# Thinking T probe — train, metrics, pre-registered verdict

Written 9 Sep 2026. Companion: `thinking-regime.md` (why thinking), mix dump
(204 pairs, mean |ΔT| = 726 on both-V=1). This note is the **frozen protocol**
for the thinking **T** head. Do not retune after seeing val pairwise.

This stage is not search and not V. Success is: a tiny head on HF layer 26
orders same-parent thinking siblings by remaining length, especially when the
gap exceeds the 256-token peek tax. That is the ISEF number for this stage.
BAGEN showed *verbalized absolute intervals* fail (47% coverage). This run is
the easier question: **is A shorter than B?**

---

## 0. What this is not

- Not V. MATH thinking is V-tied (~27 V=0 / 408). Do not `--task joint`.
- Not the Phase 0 `report.json` `verdict`. That kill table requires V AUROC
  ≥ 0.65 and V+T vs V-only. On this corpus those checks are undefined or
  will `KILL` / `FIX_PROBE` for the wrong reason. **Ignore `decision:`.**
- Not k=8. Split-half ceiling with k=2 is one draw vs one draw. Report it;
  do not treat it as Phase 0’s 0.71.
- Not “MAE of 400 tokens is a fail.” Absolute T is BAGEN’s hard problem.
  Ordinal sibling rank is ours.
- Not Qwen in memory. Probe train reads `hidden_L26.npy` (408 × 4096 fp16).
  **Do not load Qwen3-8B. Do not download another model.** CPU is enough;
  GPU is faster. Do not hog a MIG slice for this.

---

## 1. Data (already collected)

| | |
|---|---|
| JSONL / hidden | `$TOKENAWARE_ARTIFACTS/thinking/siblings/{states.jsonl,hidden_L26.npy}` |
| Labelled states | 408 (204 complete pairs) + 6 skipped parents |
| Cut | parent 1024 + peek 256, width 2, k=2, cap **16384** |
| `h` | HF layer 26, last token of parent+peek |
| Holdout | `build_thinking_cache.py`: SHA1 `think-val:{pid}`, **20% of problems** |

Expected val size: ~40 problems, ~80 states, ~40 pairs. Binomial SE at
accuracy 0.65 is ~0.075; a 95% CI is about ±0.15. **If the CI includes 0.50,
the result is inconclusive, not a trophy and not a kill of the idea.**

Restore from Hub only if the JSONL is missing. Extract with `tar` into
`$TOKENAWARE_ARTIFACTS` so the `thinking/` prefix is kept (`fetch_artifacts_hf.py`
still strips unknown top-level dirs).

---

## 2. Frozen train command

On the Jupyter box, after `source $HOME/tokenaware-data/env.sh`:

```bash
python scripts/build_thinking_cache.py \
    --src $TOKENAWARE_ARTIFACTS/thinking/siblings \
    --out $TOKENAWARE_ARTIFACTS/cache/thinking_k2mix

python scripts/train_probe.py \
    --cache $TOKENAWARE_ARTIFACTS/cache/thinking_k2mix \
    --task t \
    --trunk mlp \
    --t-head dist \
    --layers 26 \
    --max-tokens 16384 \
    --n-bins 32 \
    --no-same-trace \
    --epochs 8 \
    --mc-finetune-epochs 0 \
    --batch-size 32 \
    --lr 5e-4 \
    --dropout 0.1 \
    --patience 3 \
    --seed 0 \
    --budgets 2048 4096 8192 16384 \
    --out $TOKENAWARE_ARTIFACTS/probes/thinking_t_l26_s0
```

Why these knobs (do not “improve” them after val):

| Knob | Why |
|---|---|
| `--task t` | V is not a class here |
| `--trunk mlp` | 408 states; `star` (~9.5M) overfits; Phase 0 mlp vs star was a wash |
| `--t-head dist` | score needs `P(T ≤ B)`; censored NLL keeps truncations |
| `--max-tokens 16384` | generation cap; bins end here; overflow = did not finish |
| `--no-same-trace` | this cache is MC siblings only |
| `--mc-finetune-epochs 0` | no cheap same-trace pretrain; stage B would be the same 326 rows |
| `--batch-size 32` | default 512 is one step per epoch |
| `--budgets 2048…16384` | default 64–512 is the non-thinking grid; meaningless here |

Truncations stay in the loss (`t_censored_nll`). Do not drop them.

### Required controls (same cache, seed 0)

Run **before** celebrating the primary. All three must exist in the write-up.

```bash
# Position/meta only: must be ~chance on this corpus (tokens_so_far ≈ 1280 for everyone)
python scripts/train_probe.py --cache $TOKENAWARE_ARTIFACTS/cache/thinking_k2mix \
    --task t --trunk mlp --t-head dist --layers 26 --max-tokens 16384 \
    --no-same-trace --no-hidden --use-meta --epochs 8 --mc-finetune-epochs 0 \
    --batch-size 32 --seed 0 --budgets 2048 4096 8192 16384 \
    --out $TOKENAWARE_ARTIFACTS/probes/thinking_t_nohidden_s0

# Linear T (*How Much is Left?* diagnostic): is the signal even linear?
python scripts/train_probe.py --cache $TOKENAWARE_ARTIFACTS/cache/thinking_k2mix \
    --task t --trunk linear --t-head point --layers 26 --max-tokens 16384 \
    --no-same-trace --epochs 8 --mc-finetune-epochs 0 --batch-size 32 --seed 0 \
    --budgets 2048 4096 8192 16384 \
    --out $TOKENAWARE_ARTIFACTS/probes/thinking_t_linear_s0
```

### Seeds (primary architecture only)

```bash
for s in 1 2; do
  python scripts/train_probe.py --cache $TOKENAWARE_ARTIFACTS/cache/thinking_k2mix \
      --task t --trunk mlp --t-head dist --layers 26 --max-tokens 16384 \
      --no-same-trace --epochs 8 --mc-finetune-epochs 0 --batch-size 32 \
      --seed $s --budgets 2048 4096 8192 16384 \
      --out $TOKENAWARE_ARTIFACTS/probes/thinking_t_l26_s$s
done
```

### Per-level eval (L5 is load-bearing)

```bash
python scripts/evaluate_probes.py \
    --cache $TOKENAWARE_ARTIFACTS/cache/thinking_k2mix \
    --probe $TOKENAWARE_ARTIFACTS/probes/thinking_t_l26_s0/probe.pt \
    --budgets 2048 4096 8192 16384 \
    --levels 5 \
    --out $TOKENAWARE_ARTIFACTS/probes/thinking_t_l26_s0/eval
```

Also `--levels 4 5`. Do not headline L1–L3 pairwise (n is tiny; L1 mean |ΔT| = 281
does not pay tax).

Optional, **after** the scorecard is filled, not to pick a model: `--n-bins 64`
as a one-line appendix. If it moves val pairwise by <0.03, keep 32.

---

## 3. Metrics — what to copy out of `report.json`

`evaluate.py` now emits gap **20 / 50 / 256 / 512**. Use the **val** split.
Always copy **n_pairs** and **group_ci95**. A point estimate without n and CI
does not go on a board.

### Primary (the ISEF panel)

From `sibling.primary`:

| Key | What it is | Chance |
|---|---|---|
| `t_pairwise.accuracy` | of two siblings, does `T̂` pick the empirically shorter? | 0.50 |
| `t_pairwise.group_ci95` | bootstrap over groups | — |
| `t_pairwise.n_pairs` | val pairs with |ΔT̂_label| > 0 | — |
| `t_pairwise_gap256` | same, only pairs with label \|ΔT\| > 256 (one peek) | 0.50 |
| `t_pairwise_gap512` | label \|ΔT\| > 512 (width-2 vs random chain break-even) | 0.50 |
| `t_pairwise_position_baseline` | `tokens_so_far` only | ~0.50 here |
| `t_noise_ceiling` | k=2 split-half; noisy, **not** Phase 0’s 0.71 | — |

Shorter-is-better is already encoded (`-t_pred` vs `-t_mc_mean`).

### Secondary (science, not the title)

From `global.t`:

| Key | Role |
|---|---|
| `spearman` | global rank; inflated by L1 vs L5. Soft check ≥ 0.30 |
| `mae` vs `mae_median_baseline` | must beat train median (L1-optimal constant) |
| `mae_reduction_pct` | report; do not headline. 20–40% cut would match BAGEN’s 28% midpoint error band, not refute it |
| `coverage_q0.9` | calibration of the dist head; Phase 0 wanted ≥ 0.70. Soft here |
| `pred_mean` vs `true_mean` | optimism check (BAGEN: models under-estimate remaining cost) |

From `history.json`: train vs val NLL. If val NLL rises after epoch 2 (Phase 0
pattern), the weights in `probe.pt` are last-patience, not magic. Say so.

### Do not headline

- `global.v` / V AUROC (constant 0.5 if `--task t`)
- `headline_vt_vs_v` as “V+T works” (V is flat; this collapses to T vs random)
- Phase 0 `verdict.decision`
- Train-split pairwise (overfit diagnostic only)
- Mix *label* gaps (726 tokens) as if they were probe accuracy

---

## 4. Pre-registered scorecard

Fill this once. Do not add rows after seeing numbers.

### Hard (all must pass to `PROCEED_TO_CONTROLLER`)

| # | Criterion | Pass if |
|---|---|---|
| H1 | Val pairwise CI | `group_ci95[0] > 0.50` on seed 0 |
| H2 | Point estimate | `t_pairwise ≥ 0.60` on seed 0 (same bar as Phase 0) |
| H3 | Hidden state does the work | primary pairwise > no-hidden pairwise + 0.01 |
| H4 | Not position | primary pairwise > position baseline + 0.01 |
| H5 | Tax-relevant pairs | `t_pairwise_gap256 ≥ 0.60` **or** (if `n_pairs < 15`) CI lower > 0.50 and point ≥ 0.58. Copy `n_pairs`. |
| H6 | Seeds | seeds 0,1,2: all pairwise > 0.50; **median ≥ 0.60** |
| H7 | L5 | `--levels 5` pairwise ≥ 0.55, or CI overlapping pooled. If L5 is chance and pooled is 0.62, **KILL** (easy levels carried it). |

### Soft (fail → `FIX_PROBE` / underpowered, not a story)

| # | Criterion |
|---|---|
| S1 | Global Spearman ≥ 0.30 |
| S2 | MAE < train-median MAE |
| S3 | Linear point-T pairwise > 0.55 (signal is at least partly linear) |
| S4 | `gap512` pairwise ≥ `t_pairwise` (large gaps should be easier; if not, the head is ranking noise) |

### Decisions

| Verdict | Meaning | Next |
|---|---|---|
| **PROCEED_TO_CONTROLLER** | H1–H7 pass | 3-action decoder (continue / peek / stop) vs chain, tax charged. Still no V. |
| **UNDERPOWERED** | Point ~0.58–0.65 but H1 fails (CI includes 0.50) | More labels or 5-fold **after** this card, not hparam fishing. Do not search. |
| **KILL_T** | Hidden ≈ no-hidden, or L5 chance, or gap256 chance while overall looks good | Tax-audit (MATH miss / thinking label gaps) remains the ISEF negative result. Do not invent V to rescue T. |

Phase 0 thinking analogue to hope for, not to require: pairwise **0.62–0.70**
recovering most of a noisy k=2 label. **0.85 is not a T number we have ever
earned.** Tele-Lens says early CoT is a hostile place to read global length;
clearing 0.60 on *true siblings* is already the result.

---

## 5. What “ISEF grand award level” requires from this run

A Grand Award board does not say “we trained a probe.” It says a **predicted
measurement**, with controls, that can fail.

**Sentence the judge repeats:** Remaining thinking length is in layer-26 hidden
states well enough to rank two forks of the same prefix, and the ranking holds
on pairs that would pay a 256-token peek.

**Figure (this stage only):**

1. Bars: chance 0.50 | position | no-hidden | linear | mlp T | (k=2 ceiling if finite)
2. Beside them: gap>256 and gap>512
3. Error bars = `group_ci95`
4. Caption: n val pairs, 204 train problems held out by hash, three seeds

**What would make a working scientist walk away:** n=40 with CI 0.48–0.74 sold
as 0.65. L1 mixed into the headline. Calling MAE the result. Calling V AUROC
0.85 a length result. Tuning mlp width after looking at val.

**What this stage cannot yet claim:** search beats a chain; T+V; AIME accuracy;
universality. Those are later, gated on `PROCEED_TO_CONTROLLER`.

---

## 6. Log (fill after the run)

Primary `thinking_t_l26_s0`

| Metric | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| val n_pairs | | | |
| t_pairwise (CI) | | | |
| gap256 (n) | | | |
| gap512 (n) | | | |
| position | | | |
| no-hidden | | | |
| linear | | | |
| L5 pairwise | | | |
| Spearman / MAE red. % | | | |

Verdict (circle one): `PROCEED_TO_CONTROLLER` / `UNDERPOWERED` / `KILL_T`

Reason (one sentence):

---

## 7. After this file is closed

If proceed: controller vs chain on thinking, matched budget, peek tax billed.
Eval later is AIME (pool years, several seeds). MATH thinking is the T corpus,
not the headline exam.

If underpowered: another k=2 dump, **same cut**, more L5, not k=8.

If kill: write the tax law with the MATH tree-loss and the label gaps; do not
spend GPU on a V head.
