# Idea review — Greedy Tree Search with V/T regression heads

Reviewed 14 Aug 2026 against the literature in `budget-aware-ai-literature.md`.

**The proposal.** Replace BAVT's LLM-as-critic with a lightweight regression head on hidden states predicting two quantities — **V** (reasoning value) and **T** (expected tokens to finish) — and select nodes by:

```text
Score = V_n · exp(−λ · T_n / B)
```

---

## Strict definitions

Let:

- `x` = the problem
- `π` = the decoding policy of the base LLM (temperature and stopping rules included)
- `s` = a **partial trajectory** (a node): the prompt plus all tokens / tool observations generated so far
- `τ ~ π(· | s)` = a rollout that continues from `s` until termination and produces a final answer `ŷ(τ)`
- `L(τ)` = number of **tokens generated after `s`** until termination (not including tokens already in `s`)
- `1[ŷ(τ) = y*]` = 1 if the answer matches ground truth, else 0 (failure, timeout, or format error)

### V(s) — value

```text
V(s) = E_{τ ~ π(·|s)} [ 1[ŷ(τ) = y*] ]
     = Pr( eventual answer is correct | s, π )
```

- **Range:** `[0, 1]`
- A **state value under `π`**, not a process-step reward and not “how good this last step looked.”
- **Policy-dependent:** change temperature / search / stopping and `V` changes.
- **Empirical label (IVF-style):** draw `k` on-policy rollouts from `s`;

  ```text
  V̂(s) = (1/k) · Σ_{i=1..k} 1[ŷ(τ_i) = y*]
  ```

  Failed rollouts count as 0.

### T(s) — remaining token cost

```text
T(s) = E_{τ ~ π(·|s)} [ L(τ) ]
     = expected tokens still to be generated from s until termination
```

- **Range:** `[0, ∞)`. Units: **tokens after `s`**, under the same `π`.
- **Termination** = natural EOS / answer delimiter / forced budget cut, whichever `π` uses. Specify this once and keep it fixed.
- It is **not** “tokens used so far,” not “tokens in the official solution,” and not “tokens of the next step only.”
- **Empirical label:**

  ```text
  T̂(s) = (1/k) · Σ_{i=1..k} L(τ_i)
  ```

  Prefer quantiles `T̂_p(s)` (e.g. `p ∈ {0.5, 0.9}`) if the head is distributional.
- **Optional conditioned variants** (only if named explicitly):
  - `T⁺(s) = E[L(τ) | success]`
  - `T⁻(s) = E[L(τ) | failure]`
  - Default `T` is the **unconditional** expectation over all rollouts (success and failure).

### What the probes approximate

Given frozen hidden state `h(s)` at a chosen layer / token position:

```text
V̂_θ(h(s)) ≈ V(s)
T̂_φ(h(s)) ≈ T(s)
```

### What the score is allowed to mean

With remaining budget `B_rem` (tokens still allowed after `s`):

- **Heuristic (original sketch):**

  ```text
  Score(s) = V(s) · exp(−λ · T(s) / B_rem)
  ```

- **Principled default** (when `T` is distributional):

  ```text
  Score(s) = V(s) · Pr( T(s) ≤ B_rem )
  ```

  i.e. expected utility under a hard remaining-token constraint, if “over budget” is treated as zero reward.

Anything weaker (prompt-only length guess, reference-solution length, LLM verbalized “tokens left”) is **not** `T` under this definition.

---

## Bottom line

The instinct is right and the specific contribution is narrower than it looks. **Both heads have direct, recent prior art.** A probe-instead-of-PRM value head is published (ReProbe, ACL 2026; Internal Value Functions). Predicting remaining output length from hidden states is published several times over (STAR; *How Much is Left?*; *How Far Ahead Do LLMs Plan?*; Entropy-Guided Representations).

What I could **not** find is anyone using a predicted-remaining-cost signal as a **search selection criterion**. The length-prediction work uses it for serving-level batching and scheduling; the value-probe work selects on V alone. So the novel claim available to you is specific:

> Cost-aware node selection in reasoning search, where the cost term is a learned estimate of remaining tokens rather than a heuristic or a uniform assumption.

That is a real contribution, but it means **the entire paper rests on one ablation: V-only vs. V+T.** V-only is already published. Run that ablation before building anything else.

---

## Prior art you need to read first

**Kills the V-head as a standalone contribution:**

- **ReProbe** — *Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States* (ACL 2026, [paper](https://aclanthology.org/2026.acl-long.536.pdf), [site](https://reprobe.github.io/)). A &lt;10M-param probe on frozen-LLM hidden states, attention, and logits, doing step-level verification. Matches or beats PRMs **750–810× larger**, up to **25× faster**, with better OOD generalization, and it already steers Best-of-N and beam search. This is your V-head, done, published, with the exact cost argument you were planning to make.
- **Internal Value Functions (IVF)** ([OpenReview](https://openreview.net/pdf?id=KRYy2dFCeH)). Hidden states → state-value function approximating P(trajectory converges to correct answer), explicitly framed as avoiding separate PRM evaluations. **Read this one for its labeling methodology** — it generates labels by averaging multiple on-policy rollouts or early-stop rollouts, which is the fix to your data problem below.

**Kills the T-head as a standalone contribution:**

- **How Much is Left? LLMs Linearly Encode Their Remaining Output Length** ([arXiv:2607.05316](https://arxiv.org/abs/2607.05316), Jul 2026). Minimal-capacity **linear** probes on frozen hidden states of three 7–8B models across seven datasets. Finds total response length is linearly decodable **from the prompt's last hidden state before any output is emitted**, and that probe directions transfer across datasets. Note their own hedge: "decodable, not necessarily used causally."
- **How Far Ahead Do LLMs Plan?** ([arXiv:2602.02103](https://arxiv.org/abs/2602.02103)). Tele-Lens, a low-rank adapter probing CoT hidden states to predict subsequent tokens, final answers, **and reasoning lengths**, across 12 datasets. A single regression layer on the hidden state predicts thinking length — architecturally identical to your proposal.
- **STAR: Decode-Phase Rescheduling** ([arXiv:2510.13668](https://arxiv.org/abs/2510.13668)). Uses the final-layer hidden state of the last token to predict **remaining** output length, refined iteratively during decode. Cuts predictor parameters 93% and MAE 49% vs. baselines.
- **Predicting Output Length via Entropy Guided Representations** ([OpenReview](https://openreview.net/attachment?id=3loQDtveWI&name=pdf)). Progressive Length Prediction estimates remaining length at each decode step specifically to handle stochastic "one-to-many" sampling. Releases **ForeLen**, a benchmark with long-sequence, CoT, and RL data — likely directly reusable as your T-head training and eval set.

**Practical upside:** you do not need to invent or validate either head. Take ReProbe/IVF for V, take STAR/PLP for T, and spend your effort on the part nobody has done.

---

## Categorization

| Axis | Placement |
| --- | --- |
| Family | §2 test-time compute scaling — an allocation policy over search |
| Lineage | Direct descendant of BAVT (§7); sibling of BET and Conformal Thinking |
| V-head | Process reward model lineage — Let's Verify Step by Step → Math-Shepherd → Rewarding Progress → ReProbe/IVF |
| T-head | Serving/scheduling length-prediction lineage — Response Length Perception (NeurIPS 2023) → STAR → PLP |
| Novel seam | Joining the two: cost-aware selection. Nobody has crossed these lineages. |

The cross-lineage framing is your best pitch. The scheduling community predicts length to pack batches; the reasoning community predicts value to pick branches. Nobody predicts length **to decide where to think**.

---

## Technical problems, with fixes

### 1. `B` should be `B_remaining` — as written this is not budget-aware

```text
Score = V_n · exp(−λ · T_n / B)
```

If `B` is the *initial* budget it is a constant, so the score is a static length prior that never changes as you spend. All adaptivity is gone. BAVT's actual mechanism is that the *remaining* ratio drives annealing.

**Fix:** use `B_rem`. Then as `B_rem → 0`, the penalty on high-`T` nodes sharpens, producing sensible "wrap it up, take the cheapest viable node" behavior. Watch for numerical collapse: all scores → 0 together, which breaks softmax sampling. Normalize or work in log-space.

### 2. The score is a heuristic; there is a principled alternative that also deletes λ

`V · exp(−λ · T / B)` has no decision-theoretic reading, and `λ` is a free hyperparameter you must tune — which forfeits the "parameter-free" advantage BAVT explicitly claims. Three better options:

- **Bang-per-buck:** `Score = V_n / (T_n + ε)`. The classic greedy knapsack index, with a known approximation guarantee in the fractional case. One fewer hyperparameter. Should be a baseline regardless.
- **Deadline-feasibility (recommended):** `Score = V_n · P(T_n ≤ B_rem)`. This literally reads *"probability this node produces a correct answer within the remaining budget"* — expected utility, not a proxy. Removes `λ` entirely; get `P(T ≤ B_rem)` by having the T-head emit **quantiles** instead of a point estimate. Given that `T` is high-variance, a distributional head is better anyway.
- **Keep BAVT's annealing on top,** applying the exponent `1/τ` with `τ ∝ B_rem / B_0` to whichever base score you choose. Then your contribution is orthogonal to BAVT's and composes with it, rather than competing.

### 3. Predict T for subtree completion, not for the node

A node's cost is the expected tokens to reach a terminal answer *through* it under the current policy, not the length of the next step. Make sure your labels reflect that.

### 4. Drop "greedy"

Greedy best-first over a noisy learned value is brittle, and greedy is precisely what BAVT *anneals into* at low budget. Starting greedy discards the exploration that makes tree search beat a single chain, and hands a reviewer an easy "why not just anneal?" rebuttal. Keep stochastic selection; let greedy emerge as the low-budget limit.

### 5. Head architecture details

- **Logistic regression is wrong for T.** `T` is a positive count. Use linear regression on `log T`, a Poisson/negative-binomial head, or (best) **quantile regression**, which feeds option 2 above. Logistic is fine for `V` if `V = P(success)`.
- **Probe multiple layers, not the final one.** Middle layers frequently probe better than the last. Sweep `ℓ ∈ {0.25, 0.5, 0.75, 1.0} · L` — cheap and often worth several points.
- **Train both heads on a shared trunk, multi-task.** They read the same hidden state; a shared 2-layer MLP with two output heads costs almost nothing and `V` and `T` are correlated (hard states are both low-value and long).
- **White-box only.** This rules out frontier APIs. Fine for a paper, but state it — it is a real deployment limitation that BAVT (training-free, black-box-compatible) does not have.

---

## The flaw most likely to sink the experiment

> "Take step-by-step math solutions and mark each reasoning step with the number of tokens left before reaching the answer."

Three problems, in increasing severity:

1. **Off-policy labels.** If you label *reference* solutions (e.g. MATH's official write-ups), the probe learns reference-solution style, which has almost nothing to do with how long *your model* will ramble. Labels must come from your own policy's rollouts.
2. **Single-sample labels for a random variable.** One trajectory is one draw. The same state can terminate in 200 or 2000 tokens depending on whether the model backtracks. Regressing on one draw fits noise. **Fix (copy IVF):** sample `k ≥ 8` on-policy rollouts per state and regress on the empirical mean, or better, on quantiles.
3. **Survivorship bias — the worst one.** Training only on solutions that reached an answer means you never observe states destined to fail. But pruning bad branches *is the entire point of the V-head*. You must include failed rollouts as negatives. Related: `E[T | success] ≠ E[T | failure]`, and BET's finding is that unsolvable problems are the *expensive* ones. Consider predicting both conditionals, or predicting `T` jointly with success.

---

## Will it work? Honest predictions

**Will work (high confidence).** The V-probe recovers correctness signal — ReProbe and IVF already demonstrate this, matching PRMs 750× larger. Expect AUROC roughly 0.70–0.80 on math. The cost argument is also definitionally sound: a probe on activations you already computed is ~free next to BAVT's critic, which burns up to 512 output tokens per call. In a *token-budget* method, spending generation tokens on evaluation is self-defeating, and fixing that is a genuinely good catch.

**Uncertain — this is the crux, and it is not "can T be predicted."** `T` *is* predictable; that is settled. The risk is that the published numbers do not transfer to search, for a structural reason:

> **Every length-prediction paper evaluates T globally, across a dataset of different prompts. Search needs T locally, among siblings of one prompt.**

Global evaluation is dominated by *between-problem* variance — hard/long prompts vs. easy/short ones. A probe that only learns "this looks hard, expect many tokens" scores well, and for scheduling that is genuinely sufficient, because you are packing unrelated requests into batches. But when you rank sibling nodes you are comparing candidates **from the same problem at the same depth**, which factors out exactly the variance those probes are exploiting. What remains is within-problem, between-sibling discrimination — strictly harder, and unmeasured in the literature. A reported global ρ of 0.6 is fully consistent with near-zero usable signal for node selection.

**Compounding this: V and T are probably correlated.** States going badly tend to be both low-value and long. To the degree that holds, `T` is collinear with `V` and the product term adds nothing beyond what `V` already encodes. Measure this directly (partial correlation of `T` with outcome, controlling for `V`) — if it is near zero, no scoring function can rescue the idea.

My estimate for the *global* metrics is ρ ≈ 0.4–0.6, MAPE ≈ 25–40% (BAGEN's verbalized estimates hit 28% median relative error after SFT+RL, so a probe should beat that, but not enormously). My estimate for the *within-problem sibling* metrics is materially worse, and that is the number that decides the project. Gains, if any, will concentrate at **tight budgets** where feasibility binds, and be invisible at generous ones.

**Likely to disappoint.**

- Gains will be small and easy to lose in noise. Do not evaluate on AIME (30 problems). Use MATH-500 Level 4–5, or BAVT's multi-hop QA setup where you can reuse their compute-matched baseline.
- Easy benchmarks have nothing to allocate. GSM8K will show nothing.
- Greedy-on-noisy-V will underperform BAVT's annealing if you run it as specified.

---

## Suggested plan

### Phase 0 — offline probe feasibility (~1 day, decides everything)

Do not build a search until this passes. No tree, no `λ`, no scoring.

1. Open model (Qwen3-8B or R1-Distill-Qwen-7B), ~2000 MATH problems spanning Level 1–5.
2. Roll out **8 samples per problem** at temperature 0.7. At each reasoning-step boundary, cache the hidden state and label it with (a) did *this rollout* end correct, (b) tokens remaining in *this rollout*. Aggregate across the 8 to get per-state `V ≈ P(success)` and the empirical distribution of `T`. **Keep failed rollouts.**
3. Train a linear probe and a 2-layer MLP at four layer depths.
4. Report **both** of these, and do not confuse them:
   - *Global* (comparable to published work, and the optimistic number): V AUROC; T Spearman ρ, MAPE, quantile calibration, pooled across all problems.
   - *Within-problem sibling* (**the number that actually decides the project**): group states by problem and depth, then measure pairwise ranking accuracy — given two siblings, how often does the probe correctly order their true `T`? Also report ρ computed within each problem and then averaged. Expect this to be substantially worse than the global figure; if it is not, that is a strong positive result worth highlighting.
5. Measure **partial correlation of `T` with the outcome, controlling for `V`.** If `T` carries no information beyond `V`, no scoring function fixes that, and you should learn it here for the cost of one regression.

**Kill criteria.** Global MAPE &gt; 50% or ρ &lt; 0.3 — the probe is broken. More importantly: **sibling pairwise ranking accuracy &lt; ~0.60, or near-zero partial correlation — `T` cannot inform selection**, regardless of how good the global numbers look. Pivot to V-only-plus-feasibility-gating, or reframe (below).

### Phase 1 — search, only if Phase 0 passes

Baselines, all budget-matched, all multi-seed, with the probe's own FLOPs counted:

1. Single chain
2. Majority voting at identical budget (BAVT's baseline — reuse it)
3. Dynasor/Certaindex early exit (**the strongest efficiency baseline; algorithm-agnostic and training-free — you must beat this**)
4. BAVT with LLM critic
5. Yours, V-only ← *this is ReProbe; it is the paper's real control*
6. Yours, V+T

Headline metric: **area under the accuracy-vs-token-budget curve**, not accuracy at one budget. That is where the field is going and it is much harder to cherry-pick.

---

## Reframe if Phase 0 is marginal

Two fallbacks that survive a weak T signal:

- **Feasibility, not discounting.** Do not use `T` to rank. Use it only to *prune infeasible branches* (`P(T > B_rem) > threshold`). Feasibility is a much easier prediction problem than precise regression — recall BAGEN's finding that binary feasibility is a *calibration* problem that SFT fixes (25.5% → ~90%) while interval estimation is a *reasoning* problem that stays broken at 47% coverage. Aim at the tractable one.
- **Attack the accounting instead.** Reframe the paper as: *test-time search papers systematically exclude verifier cost, and once you charge for it, most reported gains shrink.* Then present the probe as the fix that makes verifier-guided search honestly cheap. This is a real gap — Snell et al.'s FLOP matching excludes PRM passes, and BAVT's own appendix shows search cost dominating tokens by 10×. That paper is publishable even if `T` contributes nothing.
