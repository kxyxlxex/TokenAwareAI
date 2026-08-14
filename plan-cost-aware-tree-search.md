# Execution plan — cost-aware tree search with V/T probes

This is the do-this-then-that plan for the idea in `idea-cost-aware-tree-search.md`. Every numeric claim below is taken from a cited paper or dataset card, not guessed. If a number is a *budget you will spend*, it is labeled as such and justified by a published precedent of the same size.

**Paper claim (the only novel seam).** Cost-aware node selection in reasoning search, where remaining cost is a learned estimate of tokens-to-termination rather than a heuristic. V-only search is already ReProbe / IVF. T-only prediction is already STAR / *How Much is Left?* / PLP. The paper lives or dies on **V-only vs V+T** under a matched token budget.

**Do not build a search until Step 6 (Phase 0) passes.** The idea review is right: published T metrics are *global*. Search needs *within-problem sibling* ranking.

---

## 0. Definitions (lock these; do not drift)

Copied from `idea-cost-aware-tree-search.md`. `π` is the frozen decoding policy of the base LLM (temperature, top-p/k, EOS / answer delimiter / hard cut). `s` is a partial trajectory. `τ ~ π(·|s)` is a continuation until termination. `L(τ)` is tokens generated *after* `s`. `1[ŷ(τ)=y*]` is exact-match correctness (failures, timeouts, and format errors are 0).

```text
V(s) = E_{τ~π(·|s)} [ 1[ŷ(τ)=y*] ]     ∈ [0,1]
T(s) = E_{τ~π(·|s)} [ L(τ) ]            ∈ [0, ∞)   tokens after s
```

Default `T` is **unconditional** (success and failure). Labels are on-policy Monte Carlo:

```text
V̂(s) = (1/k) Σ_i 1[ŷ(τ_i)=y*]
T̂(s) = (1/k) Σ_i L(τ_i)
```

and, for the T-head, also empirical quantiles `T̂_p` for `p ∈ {0.5, 0.9}`.

**Score (principled default, no λ):**

```text
Score(s) = V̂(s) · P̂( T(s) ≤ B_rem )
```

`B_rem` is remaining output-token budget *after* `s`, not the initial budget `B_0`. Keep `V · exp(−λ T / B_rem)` and `V / (T+ε)` as ablations only.

Selection is **stochastic** (BAVT-style), not greedy. BAVT’s remaining-budget ratio is `r_t = min(b_tool,t/B_tool, b_token,t/B_token)` and the sampling exponent is `α_t ∝ 1/r_t` ([BAVT, arXiv:2603.12634](https://arxiv.org/abs/2603.12634), eqs. 4–5). For a math-only first paper, drop the tool channel: `r_t = B_rem / B_0`, sample proportional to `Score^{α_t}`.

---

## 1. Read these six papers before writing code

In this order. The rest of the bibliography is context, not a blocker.

| # | Paper | Why it is load-bearing | Exact numbers you will reuse |
|---|---|---|---|
| 1 | ReProbe, ACL 2026, [arXiv:2511.06209](https://arxiv.org/abs/2511.06209), [anthology](https://aclanthology.org/2026.acl-long.536.pdf) | V-head + TTS already published. Your V-only arm **is this paper**. | Probe **&lt;10M** params; 1 transformer encoder layer, hidden **512**, **16** heads, LR **5e-4**, batch **128**, **5** epochs, positive-class weight **3**; **10.8K** PRM800K prompts × **3** trajectories ≈ **32K** traces; annotation **$200**, train **4 GH200 GPU-hours** (hidden-state variant); Qwen3-8B MATH step PR-AUC **0.534–0.558** vs Qwen2.5-Math-PRM-7B **0.531** (Table 1); BoN MATH **92.7–94.4** vs pass@1 **92.4** on their 200-problem slice (Table 3); runtime on 500 MATH samples **13–14 s** vs Math-Shepherd-7B **5 min 51 s** (Table 13); they quote **2.6×–25×** vs PRMs and **750–810×** fewer params. Training gens capped at **256** tokens; eval uncapped. |
| 2 | IVF, [OpenReview KRYy2dFCeH](https://openreview.net/forum?id=KRYy2dFCeH) | Labeling recipe: `V` is P(trajectory converges), estimated by averaging on-policy or early-stop rollouts from the partial thought. | Use this *method*, not ReProbe’s LLM-judge step labels. ReProbe labels *step correctness*; you need *state value*. |
| 3 | Math-Shepherd, [arXiv:2312.08935](https://arxiv.org/abs/2312.08935) | Concrete `k` for Monte Carlo process labels. | Completer decodes **N=8** continuations per step. They then get **~170k** GSM8K solutions and **~270k** MATH solutions from 15 samples × 7B/13B generators. |
| 4 | *How Much is Left?*, [arXiv:2607.05316](https://arxiv.org/abs/2607.05316) | Linear T-probe on frozen residual stream. Closest published T-head. | Llama-3.1-8B MATH: prompt-end MAE **115.29** vs median baseline **166.32**; per-token Remaining Count Probe MAE **109.87** vs baseline **151.20** (Tables 1–2). Probe = **one linear layer**, MSE, AdamW LR **2e-4**, **4000** steps, batch **8**, `max_new_tokens=1024`, `T=0.7`, top-p **0.8**, top-k **20**, 3 seeds. MATH split ≈ **7,500 / 5,000**. They exclude sequences that hit the length cap (survivorship bias — you will *not* copy that for V). |
| 5 | STAR / ARES length predictor, [arXiv:2510.13668](https://arxiv.org/abs/2510.13668) | MLP remaining-length head on last-layer last-token hidden state. Architecture you can copy. | DeepSeek-R1-Distill-Qwen-7B **d=3584**; 4-layer MLP **2048–512–64–1**; **100k** `(h_t, y_t)` samples, request-level **70/15/15** split, labels every **20** tokens; L1 loss, one H800, ≤**100** epochs, early-stop patience **10**. Average MAE **3873.21** vs TetriInfer **7658.14** (**49.42%** MAE cut; **93.28%** fewer predictor params). Prompt-only MAE on 30–32K outputs: **18,256**; after 8k generated tokens: **2,929**. |
| 6 | BAVT, [arXiv:2603.12634](https://arxiv.org/abs/2603.12634) | Search + remaining-budget annealing. Your selection skeleton. | Tool/token tiers: Low **5** calls / **2000** (reasoner) or **1000** (instruct) tokens; Middle **10** / **4000** or **2000**; High **20** / **8000** or **4000**. Max output tokens per call **512**. OSS-20B Low BAVT avg EM **0.338** vs High majority-vote **0.334**. Critic cost is charged against the token budget (their own limitation, §5). |

Also keep on the desk (do not block coding):

- **PLP / ForeLen**, [arXiv:2602.11812](https://arxiv.org/abs/2602.11812): remaining length at each decode step; ForeLen MATH split **4,500 / 1,500 / 1,500** unique prompts, grouped sampling **K=4**. EGTP MAE cut **29.16%** vs best baseline. HuggingFace `abinzzz/ForeLen`, **525,633** rows.
- **Tele-Lens**, [arXiv:2602.02103](https://arxiv.org/abs/2602.02103): LoRA rank **r=256**, up to **4000/100/500** problems per task, ~**5K** steps, best layers **not** last (Qwen3-32B layer **48/64**; their 28-layer model layer **21/28**). **Negative result for T:** early CoT hidden states do *not* reliably predict global reasoning length except on shortcut tasks (Parity/Subsum). This is why sibling ranking can fail even if global MAE looks fine.
- **BAGEN**, [arXiv:2606.00198](https://arxiv.org/abs/2606.00198): verbalized remaining-budget *interval* coverage **47%** after SFT+RL, **28%** median midpoint relative error; binary feasibility **25.5% → ~90%** with SFT alone on Qwen-7B. Feasibility is the fallback if T regression is weak.
- **Let’s Verify Step by Step**, [arXiv:2305.20050](https://arxiv.org/abs/2305.20050): PRM800K = **~800,000** step labels on **75,000** solutions to **12,000** problems (unfiltered **1,085,590** labels / **101,599** solutions). MATH-500 = **500** problems drawn uniformly from the MATH test set after moving **4,500** test problems into PRM training.
- **MATH**, Hendrycks et al. [arXiv:2103.03874](https://arxiv.org/abs/2103.03874): **12,500** problems, **7,500** train / **5,000** test, 7 subjects, difficulty **Level 1–5**.
- **GSM8K**, Cobbe et al. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168): **7,473** train / **1,319** test (also Table 7 of *How Much is Left?*).
- **Qwen3**, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) Table 1: Qwen3-8B = **36** layers, **32/8** GQA heads, **128K** context. Table 17/18: MATH-500 **97.4%** thinking / **87.4%** non-thinking. Do **not** use thinking-mode Qwen3-8B as the search model on MATH-500; it is saturated.

---

## 2. Choose the base model (do this once)

**Primary: Qwen3-8B, non-thinking / structured CoT.** Same backbone ReProbe used. ReProbe Table 14: on 200 MATH questions, mean output **204 tokens**, **6.2** steps/trace, **85.5%** of steps judged correct, pass@1 **92.4%** on their slice (Table 3). That is high but not 97%. Official non-thinking MATH-500 is **87.4%** (Qwen3 report Table 18) — enough headroom for search, unlike thinking-mode **97.4%**.

**Secondary (length-head transfer check only, not the search model in v1):** DeepSeek-R1-Distill-Qwen-7B. STAR already trained a remaining-length MLP on this exact model with **d=3584**. Long traces make T more interesting and more expensive; do not start here.

**Do not start with:** thinking-mode Qwen3-8B on MATH-500 (ceiling); AIME-24/25 (**30** problems; one item = **3.3** points); GSM8K as the *headline* (ReProbe pass@1 **95.6%** on their slice; official GSM8K is saturated).

Decoding to freeze as `π` (match ReProbe’s Qwen3 annotator / Qwen3 defaults, not temperature 1.0):

- temperature **0.7**, top-k **20**, top-p **0.95** (ReProbe Appendix B.1 for Qwen3-8B)
- termination = EOS or boxed-answer delimiter, whichever comes first
- hard cut: **1024** new tokens for Phase 0 non-thinking MATH (*How Much is Left?* Table 6). Raise later if the empirical truncation rate exceeds ~5% of rollouts; *How Much is Left?* **drops** truncated traces, which is exactly the survivorship bias you cannot afford for V.

---

## 3. Data partitions (no leakage)

| Split | Source | Size | Use |
|---|---|---|---|
| Probe train | MATH **train** | **7,500** problems (Hendrycks) | Sample a subset (Step 4). Never MATH-500. |
| Probe val | random **500** from MATH train | 500 | Early stop, layer pick, kill criteria. |
| Search / paper eval | MATH-500 | **500**, Lightman uniform subset of MATH test | Headline accuracy-vs-budget. Disjoint from MATH train. |
| Difficulty slice | MATH-500 ∩ Level 4–5 | report n from the public JSON; do not invent it | Where allocation can matter (*When More Thinking Hurts*, [arXiv:2604.10739](https://arxiv.org/abs/2604.10739): Level 1–2 overthinking threshold ~**2K** tokens vs ~**8K** for Level 5). |
| Negative control | GSM8K **test** | **1,319** | Expect ~zero V+T gain. If you “win” here, the eval is wrong. |
| Optional T extra | ForeLen `MATH` RL split | **4,500 / 1,500 / 1,500** prompts, **K=4** samples | Not a substitute for on-policy hidden states of *your* `π`. |
| Later paper | BAVT’s HotpotQA / 2Wiki / MuSiQue / Bamboogle | Bamboogle is **125** questions; subset sizes for the others are not foregrounded in BAVT | Only after math V+T works. |

**Do not train the probe on PRM800K traces** even though ReProbe did. PRM800K’s train set includes **4,500** MATH *test* problems, which overlaps the **4,500** complement of MATH-500. Using it would leak into any analysis that touches the rest of MATH test, and the traces are off-policy for Qwen3-8B.

---

## 4. Step-by-step work

### Step 1 — Instrumentation (half a day)

Hook one residual-stream layer per forward pass. You will sweep layers later; start with last-layer last-token (STAR) plus mid-network.

Qwen3-8B has **36** layers. Sweep

```text
ℓ ∈ {9, 18, 27, 36}     # 0.25, 0.50, 0.75, 1.0 × L
```

which is the grid the idea review asked for and is consistent with Tele-Lens (best ≠ last: layer **48/64** = 0.75 on Qwen3-32B).

Cache `h_ℓ(s)` at **step boundaries**, not every token, for Phase 0. ReProbe extracts steps as one CoT line in non-thinking mode (pattern match); native thinking uses sentences. Use the line protocol so you can compare to ReProbe.

Also log, per rollout: token count after each step, final exact-match, whether the hard cut fired.

### Step 2 — Phase 0 corpus (the expensive step)

**Problems.** Draw **2,000** from MATH train, stratified by official Level 1–5 (Hendrycks encodes levels 1–5 per subject). Hold out the 500-problem probe-val set first, then sample 2,000 from the remaining 7,000.

**Rollouts.** **k = 8** independent samples per problem from the **root** (empty reasoning), temperature 0.7. Precedent: Math-Shepherd **N=8** continuations per step. That is **2,000 × 8 = 16,000** full generations.

Token volume, non-thinking Qwen3-8B, using ReProbe’s measured mean **204** tokens on MATH: **16,000 × 204 ≈ 3.26M** output tokens. That is a small inference job. (If you later move to R1-Distill-7B, STAR’s regime is thousands of tokens; do not budget 204.)

**State labels.** You cannot afford 8 continuations from *every* intermediate step (that is Math-Shepherd’s full pipeline: 15 solutions × N=8 completions × every step → 270k MATH solutions). Do this instead, which is IVF’s cheaper branch:

1. From each of the 16,000 traces, take every step boundary `s`.
2. **Same-trace labels** (cheap, biased): `1[this trace correct]`, `tokens remaining on this trace`. Keep failures.
3. **Monte Carlo labels (the real V/T)** only on a prefix set: for each problem, pick **2** traces and, on each, the step boundaries at **25/50/75%** of that trace’s length (**3** states × **2** traces = **6** states/problem). From each such `s`, draw **k=8** *new* on-policy continuations.

MC continuation count: **2,000 × 6 × 8 = 96,000** partial rollouts. If a typical continuation from mid-trace is ~half of 204 tokens ≈ **100** tokens, that is ~**9.6M** additional output tokens. Still feasible. If truncation or cost blows up, cut to **1,000** problems for MC labels and keep the other 1,000 as same-trace-only (use them only for T quantile pretraining, not for kill criteria).

**Binomial noise on V.** For `k=8`, SE = `√(p(1-p)/8)` ≤ **0.177** at `p=0.5`. That is why kill criteria use ranking/AUROC, not calibrated Brier as the go/no-go. Optional: on the 500-problem val set only, use **k=32** (SE ≤ **0.088**) so the *evaluation* of the probe is not dominated by label noise.

**Keep failed rollouts.** Training only on traces that boxed an answer is the survivorship bug *How Much is Left?* acknowledges in §6.

### Step 3 — Train heads (hours, not days)

Freeze the LLM. Two architectures, both published, both small enough to train on one GPU.

**V-head (ReProbe-mini, logistic).** Copy ReProbe B.2: 1-layer transformer encoder, hidden **512**, **16** heads, then 2-layer MLP to a logit; BCE with positive-class weight **3**; LR **5e-4**; batch **128**; **5** epochs. Input = hidden states of tokens in the current step at the chosen layer (not LLM-judge text). Target = Monte Carlo `V̂(s)` for MC-labeled states, else same-trace `0/1` only in an ablation.

ReProbe trained on **32K trajectories** from **10.8K** prompts. You will have **16K** full traces plus **96K** MC continuations. That is in the same order of magnitude. Their hidden-state probe trained in **4 GH200 GPU-hours**. Budget **&lt;1 GPU-day**.

**T-head (STAR MLP + quantile).** Copy STAR’s 4-layer MLP on last-token hidden state. For Qwen3-8B residual width: released config is **4096** (32 Q heads × 128). Use `m1=2048, m2=512, m3=64` as STAR did for `d=3584`. Parameter count at `d=3584` is

```text
3584·2048 + 2048·512 + 512·64 + 64 = 8,421,440
```

(~**8.42M**, matching STAR’s **93.28%** cut vs OPT-125M). At `d=4096` the first layer is **4096·2048 = 8,388,608**, total **~9.47M**, still under ReProbe’s **&lt;10M** banner.

**Do not use a point MSE as the only T loss.** `T` is a positive, high-variance count. Train:

- pinball / quantile loss at **q = 0.5 and 0.9** (feeds `P(T ≤ B_rem)` via a parametric assumption or by interpolating the two quantiles)
- optional log-T L1 (STAR used L1 on raw remaining length)

Also train a **linear** T probe (*How Much is Left?* recipe: MSE, LR **2e-4**, 4000 steps) as the “is the signal even linear?” diagnostic.

**Shared trunk ablation (after single heads work):** 2-layer MLP, hidden 512, two output heads (V logit + T quantiles). Cheap; V and T are correlated.

**Layer sweep:** four depths × {linear T, MLP T, V probe}. Pick by **sibling** metrics on probe-val, not global MAE.

### Step 4 — Offline metrics (the decision)

Report **both** families. Do not mix them in a table without a label.

**A. Global (comparable to published T/V papers)**

| Metric | What “good” looks like in print | Your kill line |
|---|---|---|
| V AUROC | ReProbe reports **PR-AUC** not AUROC; Hidden-state Self-anno MATH **0.558** (Table 1). Unsupervised UQ on the same cell is **0.173–0.257**. | V AUROC **&lt; 0.65** on probe-val → instrumentation or labels are broken. ReProbe already got ~0.53 PR-AUC; you should not be near chance. |
| T MAE / Spearman ρ | Llama-3.1-8B MATH remaining-count MAE **109.87** vs median **151.20** (*How Much is Left?* Table 2). Prompt-end MAE **115.29** vs **166.32**. | Global MAE worse than the **train-split median predictor** (they prove this is the L1-optimal constant). Or Spearman **ρ &lt; 0.3**. |
| T quantile calibration | BAGEN interval coverage **47%** even after SFT+RL — that is verbalized, not a probe. A probe should beat that on in-policy hidden states. | 90% quantile empirical coverage **&lt; 0.70** on val. |

**B. Within-problem sibling (the number the project needs)**

Group states by `(problem_id, depth_bucket)`. Depth buckets: step index or 25/50/75% prefixes.

1. **Pairwise ranking accuracy for T:** among sibling pairs with distinct empirical `T̂`, how often does the probe order them correctly? Chance = **0.5**.
2. **Same for V.**
3. **Mean within-problem Spearman ρ**, then average over problems (not a pooled ρ).
4. **Partial correlation** of `T̂` with eventual correctness, controlling for `V̂` (and vice versa). If `ρ_{T, outcome | V} ≈ 0`, T cannot help selection.

**Kill criteria (hard):**

- Sibling T pairwise accuracy **&lt; 0.60**
- or partial correlation of T with outcome given V consistent with 0 on the 500-problem val set

**Soft kill (probe broken, not “T is collinear with V”):**

- global T MAE worse than the median baseline, or ρ &lt; 0.3
- V AUROC &lt; 0.65

Tele-Lens is the prior that makes a sibling kill *likely*: they already failed to decode global reasoning length from early CoT states on MATH/GSM8K-style tasks, with high correlation only when length is a prompt-visible shortcut.

### Step 5 — If Phase 0 fails, pivot before writing a search

Two publishable leftovers, both already sketched in the idea review:

1. **Feasibility gate, not ranking.** Predict `P(T > B_rem)` and prune. BAGEN: binary feasibility is a calibration problem SFT fixes (**25.5% → ~90%** on Qwen-7B); interval estimation stays broken (**47%** coverage). You are aiming at the tractable task.
2. **Accounting paper.** ReProbe Table 13: a 7B PRM can take **34 s–5 min 51 s** to score 500 MATH samples vs **13–14 s** for the probe; BAVT’s critic is an LLM call of up to **512** output tokens per node (Appendix hyperparameter table). Snell et al. ([arXiv:2408.03314](https://arxiv.org/abs/2408.03314)) FLOP-match **excludes verifier passes**. Charge the critic and show V-probe search vs PRM search vs majority vote on one plot.

Do **not** silently drop T and sell V-only as novel.

### Step 6 — Search, only if Step 4 passes

Tree: step-level beam / best-first with stochastic selection.

**Budgets.** MATH non-thinking mean is **204** tokens (ReProbe Table 14), so BAVT’s 2000–8000 token tiers are the wrong scale. Use a grid around the empirical length distribution of *your* 16k rollouts. A concrete starting grid, to be replaced by the 10th/50th/90th percentiles of those rollouts:

| Tier | `B_0` (output tokens, whole search) | Intent |
|---|---|---|
| Tight | **256** | ReProbe’s *training* cap; forces feasibility to bind |
| Mid | **512** | BAVT max *per call* |
| Loose | **1024** | *How Much is Left?* `max_new_tokens` |

If the 90th percentile of single-chain length is `L90`, also run `B_0 ∈ {0.5, 1.0, 2.0} × L90`. Headline = **area under accuracy vs tokens-consumed**, tokens = generator + probe (probe is ~free) + any critic.

**Expansion.** At each selected node, sample **n=3** next steps (ReProbe trained with 3 trajectories/problem; beam search is how they already steer). Width 3 is a starting point, not a result.

**Selection.** Sample node `i` with probability proportional to `Score_i^{α}` , `α = 1/r`, `r = B_rem/B_0`. Force an answer when `r ≤ 0.2` (BAVT backstop `η` in the hyperparameter table).

**Charge everything.** If you run a Dynasor Certaindex baseline, its extra forward passes count. If you run BAVT’s LLM critic, its up-to-**512**-token judgments count.

### Step 7 — Baselines (all budget-matched, ≥3 seeds)

| Arm | What it is | Why it must be in the table |
|---|---|---|
| Single chain | one sample, same `π`, stop at `B_0` | Floor |
| Majority vote | as many independent chains as fit in `B_0` | BAVT’s compute-matched baseline; Adaptive-Consistency ([arXiv:2305.11860](https://arxiv.org/abs/2305.11860)) cut sample budget up to **7.9×** with &lt;**0.1%** avg drop — vote is strong when you stop early |
| Dynasor / Certaindex | [arXiv:2412.20993](https://arxiv.org/abs/2412.20993) | Training-free early exit; up to **50%** compute / **3.3×** throughput. Algorithm-agnostic. Beat this or you have not beaten the cheap thing. |
| V-only probe search | ReProbe | **The paper’s real control.** Same tree, `Score = V`. |
| V+T probe search | yours | The claim |
| Bang-per-buck | `Score = V/(T+ε)` | Classic greedy index; one fewer knob |
| Discounting | `V · exp(−λ T/B_rem)` | Original sketch; sweep λ, do not headline it |
| Feasibility prune + V rank | T only as a gate | Fallback that can still show T is useful |
| BAVT critic (optional, expensive) | LLM residual value + annealing | Only if you have budget; their own Appendix prices search API at **$0.005/query** and **&gt;90%** of $ cost — different domain (tool-use QA) |

**Do not** compare against “unprompted long CoT” (checklist item 1 in `budget-aware-ai-literature.md`). **Do not** report a single budget.

Seeds: *How Much is Left?* used **{0,1,2}** and saw &lt;**1 token** MAE seed variance. For accuracy on MATH-500, 3 seeds is the minimum; AIME is forbidden as a headline because **n=30**.

### Step 8 — What to put in the paper if it works

1. **Sibling T ranking** as Figure 1, next to global MAE. If sibling accuracy is only slightly above 0.60, say so.
2. **Accuracy vs tokens** with V-only and V+T on the same axes. The area between those two curves *is* the contribution.
3. **Partial correlation** table: T ⊥ outcome | V or not.
4. **Where it helps:** tight `B_0`, Level 4–5. *When More Thinking Hurts* found negative flips dominating past ~**7K** tokens and Level 1–2 peaking near **2K**. Your non-thinking 204-token regime is a different scale; recompute the flip analysis on your traces rather than quoting 7K.
5. **Cost of the probe:** ReProbe **13–14 s / 500 MATH** vs PRM **34 s–5 min 51 s**. Reproduce a wall-clock table.

### Step 9 — Stretch (only after Step 8)

- Compose with BAVT annealing on HotpotQA-style tool search (Low/Mid/High **5/10/20** calls).
- Distributional T (ForeLen PLP progressive prediction) if point/quantile T is mediocre but sibling signal exists late in the trace (STAR: MAE **18256 → 2929** after 8k tokens — information arrives late).
- White-box limitation: state it. BAVT is black-box; you are not.

---

## 5. Compute and calendar (anchored, not vibes)

| Step | Precedent | Your scale |
|---|---|---|
| 16k root rollouts | ReProbe: 10.8K × 3 = 32K traces, 256-token cap, Qwen3-8B | 16k traces × ~204 tokens ≈ **3.3M** tokens |
| 96k MC continuations | Math-Shepherd N=8 per step, but they did every step of 270k MATH solutions — you will not | ~**10M** tokens if mid-trace ~100 tokens |
| Probe train | ReProbe **4 GH200-h** for 32K hidden-state traces; STAR **1×H800**, ≤100 epochs, 100k `(h,y)` pairs | **&lt;1 GPU-day** after caches exist |
| Search eval | MATH-500 × (6 methods) × (3 budgets) × (3 seeds) = **27,000** problem-runs | Dominant cost. Cap per-run at `B_0`. At `B_0=512`, upper bound **27k × 512 ≈ 14M** tokens (most methods use less). |

Phase 0 (Steps 1–4) is **a few GPU-days**, not a week, at Qwen3-8B non-thinking length. The search grid is the real bill. If Phase 0 kills T, you skip most of that bill.

---

## 6. Checklist of numbers reviewers can re-verify

- MATH **12,500 = 7,500 + 5,000** — Hendrycks et al. 2021.
- MATH-500 **500**, from **5,000 − 4,500** — Lightman et al., PRM800K.
- GSM8K **7,473 / 1,319** — Cobbe et al.; *How Much is Left?* Table 7.
- PRM800K **~800k** labels / **75k** solutions / **12k** problems — Lightman et al.
- ReProbe **10.8K × 3 ≈ 32K**, **&lt;10M** params, **4 GH200-h**, MATH PR-AUC **0.534–0.558**, mean MATH length **204** tokens — Ni et al. 2026 Tables 1, 14, B.2–B.3.
- Math-Shepherd **N=8**, **~270k** MATH solutions — Wang et al. 2023.
- *How Much is Left?* Llama-3.1-8B MATH prompt-end MAE **115.29** vs **166.32** — Tables 1–2.
- STAR MLP **8.42M** params at `d=3584`, MAE **3873.21**, **100k** samples, **70/15/15** — arXiv:2510.13668.
- ForeLen MATH prompts **4,500 / 1,500 / 1,500**, **K=4** — Xie et al. Table 4.
- BAVT **5 / 10 / 20** tool calls, tokens **2000/4000/8000** (reasoner), Low EM **0.338** vs High baseline **0.334** — arXiv:2603.12634.
- Qwen3-8B **36** layers, MATH-500 **97.4% / 87.4%** (think / not) — Yang et al. 2025 Tables 1, 17, 18.
- BAGEN feasibility **25.5% → ~90%**, interval coverage **47%**, median relative error **28%** — arXiv:2606.00198.
- AIME **n=30**; do not headline it.

---

## 7. Order of operations (one screen)

1. Freeze `π` (Qwen3-8B non-thinking, T=0.7, top-k=20, top-p=0.95, 1024 cap).
2. Hook layers {9,18,27,36}; step-boundary cache.
3. 2,000 MATH-train problems × 8 root samples; keep failures.
4. MC-label 6 prefixes/problem with k=8 (k=32 on 500 val).
5. Train linear T, MLP quantile T, ReProbe-style V; pick layer on sibling metrics.
6. **Stop if sibling T rank &lt; 0.60 or T ⊥ outcome | V.** Pivot to feasibility or accounting.
7. Else: MATH-500 accuracy-vs-budget, 3 seeds, 3 budgets, arms = single / vote / Dynasor / **V-only** / V+T / V/(T+ε).
8. Headline = area between V-only and V+T. Everything else is appendix.
