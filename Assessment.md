# Preflight assessment — TokenAwareAI

Written 30 Aug 2026, before any probe has been trained.

Computed from the full corpus in the private Hub dataset `kxyxlxex/tokenaware-artifacts`:
16,000 root traces and 11,997 Monte Carlo labelled prefix states over 2,000 problems
spanning MATH levels 1–5, yielding 5,998 same-depth sibling pairs. This is the complete
set of label files in the archive: the tarball contains the train split only, and no
artifacts of any kind exist yet for the 500 validation problems.

## Verdict

Go. The corpus is complete and skewed toward the difficulty range where the idea should
work, both prediction targets clear their label noise by more than a factor of three,
neither target is reachable from token position alone, and the oracle token saving is
11.3% overall rising to 14.3% at level 5. One question remains genuinely open: whether a
small head on frozen hidden states can recover any of this. Nothing in the labels can
answer that, and a single GPU run can.

Estimated probability the headline claim survives: **about 60%**, up from 35% before the
corpus was inspected.

## Correction to earlier analysis

Earlier conclusions drawn from the local `artifacts/` directory were wrong, and the
existing note in `current.md` predicted exactly why: the split list is level-ordered, so
a partial local copy is a prefix of the easiest problems.

| | Local `artifacts/` | Hub corpus |
|---|---|---|
| Root traces | 5,280 over 660 problems | 16,000 over 2,000 problems |
| MC states | 150 over 25 problems | 11,997 over 2,000 problems |
| Levels present | 1–3 root, **level 1 only** for MC | 1–5 for both |
| Sibling pairs | 75 | 5,998 |

Three conclusions I drew from the local slice do not hold on the real distribution: that
V was hopelessly saturated, that labels were 1% complete, and that the budget only
separates siblings in an unusably narrow band. All three were artifacts of sampling only
level 1.

## Does V work?

The label carries strong, well-conditioned signal, and it improves monotonically with
difficulty.

| Level | States | v = 1 | v = 0 | Mean v | SD of v | τ (sibling) | SNR |
|---|---|---|---|---|---|---|---|
| 1 | 898 | 77.3% | 8.6% | 0.856 | 0.314 | 0.185 | 3.34× |
| 2 | 2,160 | 71.7% | 10.0% | 0.824 | 0.337 | 0.177 | 2.81× |
| 3 | 2,550 | 64.7% | 15.6% | 0.753 | 0.389 | 0.208 | 3.18× |
| 4 | 2,705 | 57.0% | 22.1% | 0.676 | 0.428 | 0.225 | 3.35× |
| 5 | 3,684 | 34.8% | 41.3% | 0.465 | 0.455 | 0.238 | 3.30× |
| **All** | **11,997** | **56.0%** | **23.4%** | **0.668** | **0.402** | **0.215** | **3.22×** |

I previously treated the high share of states with `v ∈ {0,1}` as fatal saturation. That
was a mistake in framing. Saturation only hurts when states pile up at *one* extreme,
which is the level-1 situation (77% all-correct). At level 5 the mass is split 34.8%
against 41.3%, which is close to ideal class balance for a binary probe, and the spread of
`v` across states rises to 0.455. What matters for a value probe is variance across
states, not uncertainty within a state, and that variance is largest exactly where the
project needs it.

## Does T work?

T is the better-conditioned of the two targets, and it scales up sharply with difficulty.

| Level | Mean T | SD T | τ (sibling) | Label SE at k=8 | SNR | Truncated |
|---|---|---|---|---|---|---|
| 1 | 84.4 | 58.3 | 23.3 | 12.8 | 1.81× | 0.2% |
| 2 | 118.3 | 122.9 | 57.7 | 17.0 | 3.40× | 1.0% |
| 3 | 148.1 | 156.5 | 85.8 | 23.1 | 3.72× | 1.4% |
| 4 | 203.9 | 216.4 | 101.1 | 28.4 | 3.55× | 3.8% |
| 5 | 310.8 | 276.9 | 145.5 | 40.4 | 3.61× | 8.5% |
| **All** | **—** | **—** | **105.0** | **29.3** | **3.58×** | **3.6%** |

The quantity that matters is τ, the true sibling-level spread in remaining tokens after
subtracting label noise. It grows from 23 tokens at level 1 to 146 at level 5. Two
siblings at the same depth of the same level-5 problem genuinely differ by well over a
hundred tokens of remaining work, and eight continuations are enough to see it at better
than 3.5 to 1.

Truncation at the 1024-token cap reaches 8.5% at level 5, so the censored discretised
likelihood already implemented in `probes/losses.py` is load-bearing rather than
defensive. A point regression on observed lengths would be biased low at exactly the
difficulty tier that matters.

## Do we really save tokens?

Yes, and more than I estimated from the local slice. The accounting below includes the
31.7 tokens it costs to generate the extra candidate, measured over all 16,000 root
traces.

| Level | Blind-commit accuracy | V-only tokens | V+T tokens | Saving |
|---|---|---|---|---|
| 1 | 0.856 | 117.0 | 110.6 | 5.5% |
| 3 | 0.753 | 177.1 | 160.1 | 9.6% |
| 5 | 0.465 | 339.7 | 291.1 | 14.3% |
| **All** | **0.668** | **229.5** | **203.6** | **11.3%** |

Selection accuracy is 0.742 for V-only against 0.736 for V+T at a single global
λ = 0.0015. That 0.006 gap is tuning debt, not an intrinsic cost — λ should be fitted per
level, since the right penalty scale obviously differs when mean remaining length ranges
from 84 to 311 tokens.

Two framing rules follow, and both matter more than the headline number:

1. **Compare search against search.** The V-only and V+T policies generate identical
   candidate sets and pay the same 31.7-token branching overhead. The probe itself is a
   head on hidden states the forward pass already produced, so T is close to free at
   inference. Against a no-search baseline the comparison is invalid, because branching
   costs 31.7 tokens while choosing the shorter sibling recovers roughly 8 — length
   savings alone cannot fund a search, and any framing that omits the accuracy gain
   collapses under that objection.
2. **Never let T drive.** On the local slice, selecting the strictly shortest sibling
   scored 0.713 accuracy against 0.748 for not branching at all. T works as a penalty
   inside a narrow λ band and is actively harmful outside it.

## Where the budget actually binds

Percent of sibling pairs whose feasibility differs by more than 0.1 at each remaining
budget. This is the fraction of decisions where T can change the outcome under a hard
budget.

| Level | B=64 | B=128 | B=256 | B=512 | B=1024 |
|---|---|---|---|---|---|
| 1 | 29% | 24% | 6% | 2% | 0% |
| 3 | 32% | 33% | 21% | 11% | 0% |
| 5 | 24% | 35% | 42% | 34% | 0% |
| **All** | **28%** | **33%** | **27%** | **18%** | **0%** |

At level 1 the window collapses by 256 tokens, which is what produced my earlier
"narrow window" objection. At level 5 separation holds between 24% and 42% across the
entire range from 64 to 512 tokens. The zero column at 1024 is the generation cap, not a
property of the data.

## What must be true, and the kill test

Position-only ranking — order siblings by how many tokens they have already spent — is at
or below chance on both targets:

| Level | T baseline | T ceiling | V baseline | V ceiling |
|---|---|---|---|---|
| 1 | 0.566 | 0.766 | 0.232 | 0.926 |
| 2 | 0.524 | 0.793 | 0.401 | 0.860 |
| 3 | 0.469 | 0.814 | 0.392 | 0.879 |
| 4 | 0.507 | 0.799 | 0.381 | 0.887 |
| 5 | 0.455 | 0.814 | 0.456 | 0.892 |
| **All** | **0.489** | **0.803** | **0.406** | **0.885** |

This is the single most useful result in the whole assessment, and it cuts both ways.
There is no positional confound available to explain away a positive probe result, which
removes the most obvious reviewer objection. It also means the probe gets no free lift: a
head that learns nothing will sit at 0.5, not at 0.6.

**Kill test, to run before anything else is built.** Train a T probe on levels 3–5 and
measure sibling pairwise ranking accuracy. If it fails to clear 0.55, the frozen hidden
states do not encode remaining length and the project stops. Ceilings near 0.80 mean the
realistic target band is 0.62–0.72; anything above 0.80 indicates a leak and should be
investigated rather than celebrated.

## What the claim should be

The competitor is self-consistency, and it is remarkably inefficient at the margin.
Measured on the full corpus:

| Policy | Accuracy | Tokens | Accuracy per 100 tokens |
|---|---|---|---|
| majority@1 | 0.6513 | 363.9 | 0.179 |
| majority@3 | 0.7023 | 1,091.5 | 0.064 |
| majority@5 | 0.7242 | 1,820.3 | 0.040 |
| majority@8 | 0.7358 | 2,910.4 | 0.025 |

Tripling the token spend buys 5.1 accuracy points, and return per token falls sevenfold
from one sample to eight. At levels 4–5 it is worse: majority@8 spends 3,663 tokens to
reach 0.6221, at 0.017 accuracy per 100 tokens.

So the claim to aim at is **not** "we beat greedy decoding" and **not** "we save tokens"
standing alone. It is that cost-aware selection reaches matched accuracy for 11–14% fewer
tokens, in a regime where the standard way of spending extra tokens returns almost
nothing. Report against per-node remaining budget rather than the initial budget, and
headline levels 4–5.

## What to do, in order

1. **Build the probe cache and train the first V and T probes.** Everything else is
   downstream. Levels 3–5 first.
2. **Run the kill test above** before writing any tree search code.
3. **Generate true sibling branches** from a shared parent prefix. The current 5,998 pairs
   are two-trace proxies whose absolute positions differ, which inflates τ by an unknown
   amount. `scripts/generate_sibling_branches.py` exists for this; roughly 500 problems is
   enough to check how much of τ survives.
4. **Fit λ per level.** A single global value already costs 0.006 accuracy.
5. **Weight evaluation toward levels 4–5**, which hold 53% of MC states and nearly all of
   the usable signal. Report per level throughout; pooled numbers are dominated by easy
   problems where nothing is at stake.
6. **Generate validation artifacts.** The archive has nothing for the 500 validation
   problems, so until they exist every reported number comes from a problem-disjoint slice
   of the train set. Labelling them at k=32 rather than k=8 would also halve V label noise
   and raise the measurement ceiling.

## Residual risks

In rough order of how likely they are to end the project:

- **The hidden states may not encode remaining length.** Untested, unknowable from labels,
  and the entire premise. This is what the kill test is for.
- **The 11.3% saving is an oracle figure** using true `v` and `t`. A probe recovers only
  part of it; a realised 3–5% is a reasonable expectation and would still be a result,
  but it is a smaller one.
- **Sibling pairs are proxies.** Two independent traces at the same depth fraction differ
  in absolute position as well as content, so some of τ is position rather than branch
  identity. Real branches will show a smaller τ.
- **λ sensitivity.** T is harmful when it drives selection. If the usable λ band turns out
  to be narrow and level-dependent, the method becomes fragile in a way reviewers will
  press on.
- **Truncation at level 5.** 8.5% censoring is handled in the loss, but it also caps the
  measurable dynamic range of T at exactly the tier that matters most.

## Method notes

All figures come from JSONL label files only. The `.pt` hidden-state tensors were not
downloaded; the tarball was streamed and filtered so that only label files landed on disk.
Sibling pairs are states sharing a problem and depth fraction. τ is computed by
noise-correcting the variance of the within-pair difference: τ² = Var(Δ)/2 − SE², where SE
is the standard error of the k=8 mean. Noise ceilings come from split-half resampling of
the eight continuations, so the true k=8 ceiling is marginally higher than reported.
Budget utility is the fraction of continuations both correct and finishing within B.

Two caveats on scope. The archive was streamed to completion, and it contains no
validation artifacts at all, so every number here is drawn from the 2,000 train problems
and none of it is held out. Probe evaluation therefore has to rely on the problem-disjoint
split that `build_probe_cache.py` carves out of the train set, until validation rollouts
are generated. And the level-1 subset is the one place where these numbers overlap the
earlier local analysis, so disagreements between this document and anything written before
today should be resolved in favour of this one.
