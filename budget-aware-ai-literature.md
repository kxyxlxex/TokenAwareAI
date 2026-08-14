# Budget-Aware AI — Annotated Literature Review

Work on minimizing, controlling, or customizing the tokens/compute an AI system spends while holding quality roughly constant. Covers general generation and reasoning models.

Curated, not exhaustive: superseded and obsolete papers are omitted. Last updated August 2026.

**Entry format** — Summary (6 sentences) · Data & access · Weak points (what the paper downplays).

---

## Read every claim against this checklist

Nearly every paper here claims *"−X% tokens, only −Y% quality."* Six failure modes recur; they are listed once instead of in all ~100 entries.

1. **Verbose strawman baseline.** Savings measured against unprompted rambling CoT. Much of the gain is often recoverable by appending "be concise."
2. **Uncounted overhead.** Budget estimators, routers, compressors, and PRMs need forward passes. Most papers count only target-model tokens.
3. **Tiny evals.** AIME has 30 problems — one problem is 3.3 points. Most 1–3 point deltas in this literature are noise, and seed variance is rarely reported.
4. **Not compute-matched.** The honest baseline for N agents or K samples is *one* run with the same total budget. Run properly, many gains shrink or invert.
5. **Stale prices.** 2023–24 cascade papers priced against GPT-4 at \$30/\$60 per M tokens. Frontier inference has fallen 1–2 orders of magnitude since; those cost ratios do not survive re-pricing.
6. **Averaged easy and hard.** Cutting tokens is safe on GSM8K and dangerous on AIME/GPQA. Degradation concentrates in the hardest split and hides in the mean. GSM8K/MATH are also heavily contaminated.

**And the objective confusion:** tokens, latency, dollars, and energy are four different targets, and papers optimize one while claiming victory on another. Parallel sampling is cheap in latency, expensive in dollars; long CoT is the reverse. Speculative decoding cuts latency while raising total FLOPs. KV eviction cuts memory, not billed tokens. Prompt compression cuts billed input tokens but is largely undercut by prefix caching (~90% discount on hits, now standard). A paper that does not name its objective is optimizing whichever one flatters it.

---

## Taxonomy

| § | Family | Lever | Typical claim |
|---|---|---|---|
| 1 | Length-controlled reasoning | Shorten the CoT | −70% output tokens, <5% accuracy |
| 2 | Test-time compute scaling | How many samples/steps | Compute-optimal beats uniform |
| 3 | Routing & cascades | Which model answers | Frontier quality at a fraction of cost |
| 4 | Input-side compression | Shrink prompt/context/KV | 20x compression, quality retained |
| 5 | Agents & multi-agent economics | How many calls | Fewer calls, same success |
| 6 | Surveys, benchmarks, economics | Measure the tradeoff | Here is the cost-quality frontier |
| 7 | 2026 frontier | Buy reasoning only when it pays | Solvability-aware allocation |

---

## §0 Foundations (pre-LLM, still load-bearing)

Three ideas the whole field rests on. Included because they are still cited and still correct, not for completeness.

### Adaptive Computation Time for RNNs
Graves (DeepMind) · [arXiv:1603.08983](https://arxiv.org/abs/1603.08983) · Mar 2016

**Summary.** Standard RNNs spend identical compute per input regardless of difficulty, which starves hard inputs and wastes compute on easy ones. ACT adds a learned halting unit letting the network choose how many computational steps to take between reading an input and emitting an output. The mechanism is deterministic and differentiable, adds no gradient noise, and requires minimal architectural change. On four synthetic tasks (parity, logic, addition, sorting) it dramatically improves performance by adapting step count to difficulty. On Hutter Prize Wikipedia character-level LM it yields little performance gain, but allocates visibly more computation to hard-to-predict transitions such as word boundaries and sentence ends. The paper suggests adaptive computation could serve as a generic method for inferring segment boundaries in sequences.

**Data.** Four synthetic tasks generated procedurally; Hutter Prize Wikipedia (enwik8), public. No official code release from the author.

**Weak points.**
- Real-data result is essentially negative — the honest reading is that ACT helped on synthetic tasks built to need it.
- The ponder-cost penalty is a hand-tuned hyperparameter that the halting behavior is very sensitive to; this is under-discussed and became the known reproducibility complaint.
- Interpretability claim ("more compute at word boundaries") is post-hoc pattern-reading, not a tested hypothesis.

**Why it still matters.** Every learned-budget method in §7 is ACT with a policy-gradient trainer and a token-count cost term.

### Confident Adaptive Language Modeling (CALM)
Schuster, Fisch, Gupta, Dehghani, Bahri, Tran, Tay, Metzler (Google) · [arXiv:2207.07061](https://arxiv.org/abs/2207.07061) · NeurIPS 2022

**Summary.** Transformer LMs spend full depth on every token even though most next-token predictions are trivial. CALM exits decoding early per token when an intermediate layer is confident enough, allocating compute per input *and* per timestep. It solves three specific problems: which confidence measure to use (softmax response, state propagation, or a trained early-exit classifier), how to connect sequence-level quality constraints to local per-token exit decisions, and how to attend back to hidden states that were never computed because of earlier exits. The key contribution is calibration: using distribution-free risk control on a held-out set, local exit thresholds are set so that a global sequence-level metric (ROUGE, BLEURT) is provably maintained with high probability, e.g. 95%. Training uses a weighted average of per-layer prediction losses so intermediate layers produce usable predictions without degrading the top layer. Across three text generation tasks it reaches up to 3x speedup while provably retaining quality.

**Data.** Three generation tasks — CNN/DailyMail summarization, WMT15 EN-FR translation, Open-domain QA (SQuAD-derived). All public. T5-based models; code released within Google's T5X.

**Weak points.**
- 3x is the best case and is *theoretical FLOP* speedup; realized wall-clock gain is much lower because early exit breaks batching — different sequences exit at different layers, so a served batch runs at the depth of its deepest member.
- The guarantee is conformal, so it holds only under exchangeability with the calibration set; it silently lapses under distribution shift.
- Requires a labeled in-domain calibration set per task, which is the expensive part for anything open-ended.
- Skipped-state attention is patched by copying the last computed hidden state — an approximation whose error compounds over long generations, examined only briefly.

**Why it still matters.** CALM is the direct ancestor of Conformal Thinking (§7), and it got the risk-control framing right four years earlier.

### Cascades and the cost-quality frontier
Viola & Jones (CVPR 2001) established the attentional cascade: run a cheap classifier first, escalate only the uncertain cases. Every §3 LLM cascade is this idea with an LLM in the slot, and its known failure mode transfers exactly — cascades win only when the cheap stage's confidence is well-calibrated *and* the cheap stage is genuinely cheap relative to the expensive one. Both conditions are eroding as small-model quality rises and frontier prices fall.

---

## §1 Length-controlled and token-budgeted reasoning

The largest and fastest-moving family. Ordered by mechanism: prompt-level → theory → RL-trained length control → learned think/no-think → latent CoT.

### 1a. Prompt-level budgeting

**TALE — Token-Budget-Aware LLM Reasoning.** Han et al. (Nanjing Univ. / Rutgers) · [arXiv:2412.18547](https://arxiv.org/abs/2412.18547) · Findings of ACL 2025 · [code](https://github.com/GeniusHTX/TALE)

CoT improves accuracy but inflates token cost, and the paper's finding is that current CoT is unnecessarily lengthy and compressible simply by naming a token budget in the prompt. The catch is that the budget value matters enormously, so TALE first *estimates* a per-question budget from reasoning complexity, then injects it into the prompt. Two variants: TALE-EP estimates via zero-shot prompting, TALE-PT internalizes budget-awareness by post-training so no explicit budget is needed at inference. TALE-EP cuts token usage ~67% with under 3% accuracy loss; TALE-PT cuts ~50% versus vanilla CoT at competitive accuracy. Experiments center on GPT-4o-mini with generalization checks across other LLMs. Its most-cited contribution is not the method but the diagnosis of **token elasticity**: when the budget is set too low, models *overshoot it badly* — a 10-token budget produced 157 output tokens where a 50-token budget produced 86.

*Data.* GSM8K, GSM8K-Zero, MathBench-College. All public; code released.

*Weak points.* Budget estimation costs an extra LLM call that is not netted out of the 67%. GSM8K-Zero is constructed so answers are nearly given in the question, which inflates the compression headline. Evaluated mainly on one closed model at one price point. Token elasticity means the method's own control knob is unreliable at exactly the aggressive settings where it would matter most — the paper reports this honestly but does not let it dent the headline.

**Token Complexity — How Well do LLMs Compress Their Own CoT?** Lee et al. (Columbia) · [arXiv:2503.01141](https://arxiv.org/abs/2503.01141)

The most theoretically useful paper in this family, and the right one to read first. It runs the first systematic study of reasoning length versus accuracy across many compression instructions ("use 10 words or less", "remove all punctuation"), rather than proposing another method. It finds a *universal* length–accuracy tradeoff curve that persists across very different reasoning chains. The mechanism is a sharp per-question threshold: every task has an intrinsic **token complexity**, a minimum token count below which the model fails, and compression works right up to that cliff and then falls off it. This lets the authors compute information-theoretic limits on the accuracy–compression tradeoff, against which they show **existing prompt-based compression operates far from optimal** — i.e. most of the field's reported gains are leaving large amounts on the table. It also formalizes why *adaptive* compression (short answers for easy questions) is the only way to approach the limit.

*Data.* Standard math reasoning benchmarks across multiple models and dozens of compression prompts.

*Weak points.* Token complexity is measured empirically per question per model, so it is a descriptive quantity, not a predictable one you could use at inference — which limits it to being a benchmark rather than a method. The "information-theoretic limit" is derived under assumptions about the accuracy-length curve that are fit, not proven. Math-only.

**Chain of Draft (CoD).** Xu et al. (Zoom) · [arXiv:2502.18600](https://arxiv.org/abs/2502.18600) · [code](https://github.com/sileix/chain-of-draft)

Standard CoT prompting mandates verbose step-by-step prose, whereas humans draft terse intermediate notes carrying only essential information. CoD is a prompting change: instruct the model to emit minimalistic but informative intermediate steps. There is no training, no auxiliary model, and no budget estimator — it is a one-line prompt swap, which is the whole appeal. It reports matching or surpassing CoT accuracy while using **as little as 7.6% of the tokens**, cutting both cost and latency. Evaluated across arithmetic, commonsense, and symbolic reasoning tasks on GPT-4o and Claude 3.5 Sonnet. It has become the standard cheap baseline that heavier methods must beat.

*Data.* GSM8K, date/sports understanding (BIG-bench), coin flip. Public; code released.

*Weak points.* "As little as 7.6%" is the single best task; the average is much weaker and small models degrade sharply because they cannot follow the terse format. CoD substantially hurts on tasks needing genuine multi-step search, and the benchmark selection favors tasks with short answer paths. Being prompt-only, it is at the mercy of model-specific instruction-following.

**Sketch-of-Thought (SoT).** Aytes et al. · [arXiv:2503.05179](https://arxiv.org/abs/2503.05179) · EMNLP 2025

CoT's verbosity is treated here as a failure to match reasoning *style* to task type. SoT is a modular prompting framework combining cognitively-inspired paradigms with hard linguistic constraints. It ships three paradigms — Conceptual Chaining, Chunked Symbolism, Expert Lexicons — and a lightweight router picks one per query at test time. Across **18 reasoning datasets** spanning domains, languages, and modalities it reports up to 84% token reduction with minimal accuracy loss, and on math and multi-hop reasoning it *improves* accuracy while shortening outputs. The 18-dataset breadth is unusual in this literature and makes it a more trustworthy result than single-benchmark competitors. It is the natural successor to CoD.

*Data.* 18 public datasets across domains/languages/modalities; code released.

*Weak points.* The router is an extra model whose cost is excluded from the reduction figures. "Up to 84%" again marks the best cell. Three hand-designed paradigms are a curated inductive bias that may not transfer to task types the authors did not anticipate.

**s1 — budget forcing.** Muennighoff et al. (Stanford/UW/AI2) · [arXiv:2501.19393](https://arxiv.org/abs/2501.19393) · EMNLP 2025 · [code](https://github.com/simplescaling/s1)

Primarily a test-time-scaling paper (§2), but it contributes the simplest budget *control* mechanism in use. **Budget forcing** manipulates the end-of-thinking token: to cap compute, append the delimiter and "Final Answer:" to force an early exit; to extend it, suppress the delimiter and append "Wait", which often makes the model double-check and fix errors. Combined with SFT on just 1,000 curated examples (s1K), s1-32B built on Qwen2.5-32B-Instruct exceeds o1-preview on competition math by up to 27%. Budget forcing lets it extrapolate past its own untuned ceiling, from 50% to 57% on AIME24. Model, data, and code are all open.

*Weak points.* The "1,000 examples is all you need" framing understates that the reasoning traces were **distilled from Gemini**, so the capability was bought from a stronger teacher, not created by the recipe. Scaling flattens quickly — the paper shows it, but the headline does not. AIME24's 30 problems mean 50%→57% is about two problems.

### 1b. RL-trained length control

**L1 / LCPO — Controlling How Long A Reasoning Model Thinks.** Aggarwal & Welleck (CMU) · [arXiv:2503.04697](https://arxiv.org/abs/2503.04697) · COLM 2025 · [code](https://www.cmu-l3.github.io/l1)

Reasoning models improve by thinking longer but their CoT length is uncontrollable, so you cannot dial test-time compute to a target. LCPO is an RL method optimizing jointly for accuracy *and* adherence to a length constraint given in the prompt, producing L1, a model that honors a requested length. This gives a smooth, user-facing cost/accuracy dial rather than a fixed operating point, and it outperforms s1's budget forcing at length control. The unexpected finding is that LCPO training yields **Short Reasoning Models** that retain full-length reasoning *patterns* at non-reasoning-model lengths — and at equal reasoning length, the 1.5B L1 model **surpasses GPT-4o**. Code and model weights are released. This is the canonical length-control-via-RL reference.

*Data.* Math reasoning benchmarks (GSM8K, MATH, AIME, plus OOD transfer). Public; models released.

*Weak points.* "1.5B surpasses GPT-4o at equal reasoning lengths" is a real result but a rigged framing — GPT-4o is not a reasoning model and was never optimized to use a short token budget well; the comparison flatters L1 by constraining the opponent to its worst regime. Length adherence degrades at the extremes of the requested range. Trained and evaluated on math, so the dial's behavior elsewhere is unestablished.

**Elastic Reasoning.** Xu et al. (Salesforce) · [arXiv:2505.05315](https://arxiv.org/abs/2505.05315) · [code](https://github.com/SalesforceAIResearch/Elastic-Reasoning)

Uncontrolled LRM output length breaks deployments with hard token, latency, or compute limits, and naive truncation is catastrophic because it usually cuts off before the answer exists. Elastic Reasoning **separates generation into thinking and solution phases with independently allocated budgets**, prioritizing completion of the solution segment so that a truncated run still returns an answer. To make the model robust to a cut-short thinking phase, it adds a lightweight budget-constrained rollout strategy inside GRPO, which generalizes to unseen budgets without retraining. It is evaluated on both math (AIME, MATH-500) and code (LiveCodeBench, Codeforces), which is broader than most of this family. It performs robustly under strict budgets at notably lower training cost than baselines, and produces more concise reasoning even when unconstrained. The phase-separation idea is the transferable contribution.

*Weak points.* Reserving a solution budget means that under a tight total budget you get *less* thinking than the competition, so the comparison is favorable mainly in the truncation regime it was designed for. Gains in unconstrained settings are a side effect and modest. Two hand-set budgets replace one, so the tuning burden moved rather than disappeared.

**ThinkPrune.** Hou et al. (UCSB) · [arXiv:2504.01296](https://arxiv.org/abs/2504.01296) · [code](https://github.com/UCSB-NLP-Chang/ThinkPrune)

Prior length-reduction work mostly forces early exit rather than teaching the model to consolidate its reasoning, which yields a sub-optimal length–performance tradeoff. ThinkPrune instead continues RL training with a hard token limit where anything unfinished gets **zero reward**, so the model learns to fit its reasoning inside the budget rather than being cut off. An iterative schedule applies successive RL rounds at increasingly strict limits, which preserves accuracy far better than jumping straight to a tight cap. On AIME24, R1-Distill-Qwen-1.5B's reasoning length is **halved for a 2% performance drop**. Inspection shows pruned models skip unnecessary steps while keeping the core chain intact. The iterative-tightening schedule is the reusable idea.

*Weak points.* AIME24 is 30 problems, so "2% drop" is well under one problem's resolution and needs seed variance to mean anything. Demonstrated at 1.5B, the scale where redundancy is highest. Multiple RL rounds make it considerably more expensive to train than single-stage competitors, which the length-performance plots do not reflect.

**O1-Pruner.** Luo et al. · [arXiv:2501.12570](https://arxiv.org/abs/2501.12570) · [code](https://github.com/StarDewXXX/O1-Pruner)

Establishes experimentally that long-thought models fail to allocate token budget according to problem difficulty and carry substantial redundancy. Length-Harmonizing Fine-Tuning first pre-samples to estimate the model's own baseline performance, then applies RL-style fine-tuning that rewards shorter reasoning subject to an accuracy constraint. Anchoring the reward to the model's *measured* baseline rather than an absolute target is the key design choice, since it makes the pressure adaptive per problem. Across math reasoning benchmarks it both reduces inference overhead substantially and reports *higher* accuracy. Evaluated on Marco-o1 and QwQ-style long-thought models.

*Weak points.* Pre-sampling to estimate baselines is a real and uncounted training-time cost. Accuracy *improving* while length falls usually indicates the baseline was under-tuned or that shorter chains avoid self-derailment — either way it signals the comparison point was weak. Math-only.

**Training Language Models to Reason Efficiently.** Arora & Zanette (CMU) · [arXiv:2502.04463](https://arxiv.org/abs/2502.04463) · NeurIPS 2025

Frames inference cost as an economic, UX, and sustainability problem rather than a benchmark nuisance. The method uses RL to train reasoning models to allocate inference compute dynamically by task complexity, penalizing unnecessary overhead while holding accuracy. Its practical contribution is that a **single hyperparameter** generates an entire family of models at different efficiency levels, so an operator picks a point on the frontier rather than retraining. Experiments on two open-weight large reasoning models show significant inference-cost reductions while preserving most accuracy. The framing — one knob, a family of operating points — is what later work (including §7) builds on.

*Weak points.* "Preserving most of the accuracy" is deliberately vague in the abstract; the per-benchmark losses are where the real story is. Two models is a thin base for a claimed general recipe. Requires full RL access, so it is unavailable to anyone consuming closed APIs.

**Concise Reasoning via RL.** Fatemi et al. (Wand AI) · [arXiv:2504.05185](https://arxiv.org/abs/2504.05185)

The most conceptually important paper in this subsection: it explains *why* reasoning models are verbose in the first place. The claim is that verbosity is not deeper reasoning but an **optimization artifact of RL loss minimization on incorrect answers** — and since unsolvable problems dominate training, the effect compounds into systematic lengthening. This is proven theoretically for PPO and GRPO, including the result that incorrect answers drive policies toward verbosity *even when γ=1*. Empirically, conciseness and correctness are positively correlated across both reasoning and non-reasoning models, inverting the usual assumption that length buys quality. The practical consequence is a two-phase procedure where a brief secondary RL stage on a small set of **solvable** problems sharply cuts length while preserving or improving accuracy. It also documents that GRPO has collapse modes that make it unreliable for concise reasoning.

*Weak points.* The theory holds under specific assumptions about reward structure that real RLHF pipelines violate in various ways. "Extensive experiments" in the abstract without headline numbers is a hedge. An ICLR 2026 submission of this work was withdrawn.

**DAST — Difficulty-Adaptive Slow-Thinking.** Shen et al. · [arXiv:2503.04472](https://arxiv.org/abs/2503.04472) · EMNLP 2025 Industry Track

Notes correctly that uniform token reduction is the wrong tool, because it degrades exactly the hard problems that needed the tokens. DAST defines a **Token Length Budget** metric to quantify difficulty, then uses budget-aware reward shaping and budget preference optimization so the model self-adjusts CoT length. It penalizes overlong responses on simple tasks while actively incentivizing sufficient reasoning on complex ones — a two-sided pressure most competitors lack. Across datasets and model scales it cuts token usage **over 30% on average** while preserving accuracy on complex problems. It is one of the few in this family evaluated with deployment framing.

*Weak points.* TLB is derived from observed lengths, so difficulty is defined by how long the model already took — partly circular. 30% is modest next to neighbors claiming 50–70%, which is likely a sign of a more honest baseline, but the paper does not make that argument. The linked code repo is an anonymized placeholder.

**CoT-Valve.** Ma et al. (NUS) · [arXiv:2502.09601](https://arxiv.org/abs/2502.09601) · ACL 2025 · [code](https://github.com/horseee/CoT-Valve)

Starts from the observation that reasoning paths compress easily on easy tasks and resist compression on hard ones, motivating one model with an elastic length dial instead of several fixed-length models. The mechanism is distinctive: identify a **direction in parameter space** that, when manipulated, controls generated CoT length. The authors build datasets of long-to-short chains for identical questions and add two refinements, precise length-compressible tuning and progressive chain-length compression. It beats prompt-based length control on both controllability and compressibility. On QwQ-32B-Preview it cuts GSM8K chains from 741 to 225 tokens with accuracy moving only 95.07%→94.92%, and AIME chains from 6827 to 4629 tokens at the cost of one additional wrong answer.

*Weak points.* "One additional incorrect answer" on AIME is honest phrasing that also reveals the eval's granularity — the entire hard-benchmark result rests on single-problem resolution. GSM8K at 95% accuracy is saturated and contaminated, so compressing it is close to free. Parameter-space steering needs weight access.

**TokenSkip.** Xia et al. (PolyU) · [arXiv:2502.12067](https://arxiv.org/abs/2502.12067) · EMNLP 2025 · [code](https://github.com/hemingkx/TokenSkip)

Motivated by decoding latency: autoregressive generation makes long CoT linearly slow, which is painful past 10,000 tokens. The paper analyzes the semantic importance of individual tokens within CoT outputs and finds their contributions to the final answer vary widely. TokenSkip trains models to selectively skip low-importance tokens, giving controllable compression via a ratio parameter. On Qwen2.5-14B-Instruct it cuts GSM8K reasoning tokens 40% (313→181) with under 0.4% accuracy loss. Code and checkpoints are released. It is the cleanest instance of treating CoT as a compressible token sequence rather than a semantic object.

*Weak points.* 313 tokens is already a short chain, so this is compressing an efficient baseline — the result does not obviously extend to the 10K-token chains the motivation invokes. Token-importance labels come from an LLM scorer whose cost is excluded. Compression ratios beyond ~40% degrade sharply.

**LightThinker.** Zhang et al. (ZJUNLP) · [arXiv:2502.15589](https://arxiv.org/abs/2502.15589) · EMNLP 2025 (oral) · [code](https://github.com/zjunlp/LightThinker)

Targets the memory and compute cost of holding long reasoning in context, not just the billed token count. LightThinker trains the model to dynamically compress intermediate thoughts into compact **gist tokens** mid-reasoning and discard the original chain, shrinking what stays in the context window. This requires three coordinated pieces: data construction teaching *when* and *how* to compress, a mapping from hidden states to gist tokens, and specialized attention masks. It introduces a **Dependency (Dep) metric** quantifying compression by measuring reliance on historical tokens. Across four datasets and two models it cuts peak memory and inference time at competitive accuracy. It is the bridge between §1 CoT compression and §4 context compression.

*Weak points.* Optimizes memory and latency, not billed output tokens — a different objective than most of this section, and easy to conflate. Discarded reasoning is unrecoverable, so errors cannot be traced or self-corrected against the original chain. Four datasets, two models.

### 1c. Learning when *not* to think

**Do NOT Think That Much for 2+3=? — On the Overthinking of o1-Like LLMs.** Chen et al. (Tencent AI Lab) · [arXiv:2412.21187](https://arxiv.org/abs/2412.21187)

The paper that named the problem, and still the standard citation for it. It presents the first comprehensive study of overthinking, where o1-like models pour compute into trivial problems for minimal benefit. It contributes efficiency metrics from both **outcome and process** perspectives, letting you separate "wasted tokens" from "tokens that changed the answer" — a distinction most later work elides. Mitigation uses a self-training paradigm to streamline reasoning without sacrificing accuracy. Results hold across GSM8K, MATH-500, GPQA, and AIME, i.e. across genuinely different difficulty levels rather than one easy set. The authors updated it for DeepSeek-R1 and report all conclusions still hold.

*Weak points.* The efficiency metrics require knowing the correct answer to identify wasted computation, so they are analysis tools, not runtime signals. "Overthinking" is operationalized partly through solution redundancy after the first correct answer appears, which conflates verification with waste — sometimes rechecking is what makes the answer reliable.

**Reasoning Models Can Be Effective Without Thinking (NoThinking).** Ma et al. (UC Berkeley) · [arXiv:2504.09858](https://arxiv.org/abs/2504.09858)

The most important negative result in this literature, and the baseline every efficient-reasoning paper should be forced to run. Using R1-Distill-Qwen, the authors bypass the thinking block entirely via simple prompting and find it works startlingly well. **Controlling for token count**, NoThinking beats Thinking across seven challenging datasets spanning math, formal theorem proving, and coding — most sharply in low-budget settings, e.g. **51.3 vs 28.9 on AMC-23 at 700 tokens**. NoThinking also scales better with pass@k as k grows. Building on this, generating N independent NoThinking outputs and aggregating (verifier or best-of-N confidence) beats comparable-latency Thinking baselines and matches Thinking runs that take up to **9x longer**. The implication is uncomfortable: much of what long CoT buys at a fixed budget is available from parallel short generations instead.

*Data.* Seven datasets including AMC-23, AIME, MiniF2F, LiveCodeBench. Public.

*Weak points.* Parallel scaling is cheap in *latency* and expensive in *dollars* — N independent generations cost N times as much, so the 9x latency win is not a cost win, and the framing leans on the flattering axis. Best-of-N aggregation needs a verifier, which is free for formal proving and math but unavailable for open-ended work. Demonstrated on distilled models, which may retain usable non-thinking behavior that natively-RL'd reasoners lack.

**AdaptThink.** Zhang et al. (Tsinghua KEG) · [arXiv:2505.13417](https://arxiv.org/abs/2505.13417) · [code](https://github.com/THU-KEG/AdaptThink)

Takes NoThinking's finding and makes it a learned decision instead of a fixed prompt. The paper first confirms NoThinking is better on relatively simple tasks in both performance and efficiency, then proposes an RL algorithm that picks the thinking mode per problem. Two components make it work: a constrained optimization objective that pushes toward NoThinking subject to maintaining overall performance, and an importance-sampling scheme balancing Thinking and NoThinking samples during on-policy training, which enables cold start and keeps both modes explored. On three math datasets it cuts R1-Distill-Qwen-1.5B's average response length **53% while improving accuracy 2.4%**. Code and models released.

*Weak points.* Simultaneous large length cuts and accuracy gains almost always mean the base model was badly mis-calibrated for the benchmark rather than that the method found free lunch. 1.5B only. Three math datasets, so "when to think" is learned entirely within one domain's difficulty structure.

**Thinkless.** Fang et al. (NUS) · [arXiv:2505.13379](https://arxiv.org/abs/2505.13379) · [code](https://github.com/VainF/Thinkless)

Same question as AdaptThink, different and arguably cleaner training solution. Thinkless gives the model two control tokens, `<short>` and `<think>`, and trains it to choose based on both task complexity and **its own ability** — the latter being the solvability intuition that §7 later formalizes. The core algorithm, Decoupled GRPO (DeGRPO), splits the objective into a control-token loss governing mode selection and a response loss governing answer accuracy. Decoupling matters because vanilla GRPO collapses to a single mode; separating the two losses gives fine-grained control over their relative contribution and stabilizes training. On Minerva Algebra, MATH-500, and GSM8K it reduces long-chain thinking usage by **50–90%**. The collapse diagnosis is the durable contribution.

*Weak points.* The 50–90% range is wide because the low end is on the hard benchmark and the high end on the easy one — the average across a realistic mixed workload is not reported. Accuracy retention is stated less prominently than the usage reduction. Math-only.

**AdaCoT.** Lou et al. (ByteDance Seed) · [arXiv:2505.11896](https://arxiv.org/abs/2505.11896)

The only entry in this family with **production traffic** evaluation, which makes it worth more than its citation count suggests. AdaCoT frames adaptive reasoning explicitly as Pareto optimization balancing performance against both the frequency and the compute cost of CoT invocation. It uses PPO to move the CoT-triggering decision boundary by adjusting penalty coefficients, letting the model infer CoT necessity from implicit query complexity rather than an explicit difficulty label. Its key technical fix is **Selective Loss Masking**, which prevents decision-boundary collapse during multi-stage RL — the same failure mode Thinkless attacks differently. On the authors' production traffic test set it drops CoT triggering to as low as 3.18% and cuts average response tokens **69.06%** while holding performance on complex tasks. The 3.18% figure is the most concrete evidence available that real-world query mixes are overwhelmingly easy.

*Weak points.* The headline rests on a proprietary production test set nobody can inspect or reproduce, and its difficulty mix is what generates the 3.18%. "Maintaining high performance on complex tasks" is asserted with much less specificity than the savings figure. Requires full RL control over a deployed model.

**Dynasor / Certaindex.** Fu et al. (UCSD Hao AI Lab) · [arXiv:2412.20993](https://arxiv.org/abs/2412.20993) · NeurIPS 2025 · [code](https://github.com/hao-ai-lab/Dynasor)

The systems-level entry, and the one closest to something you can deploy. It observes that CoT, self-consistency, and MCTS all exhibit **answer stabilization** — intermediate solutions stop changing past a point, after which more compute cannot change the outcome. Certaindex is an algorithm-agnostic metric measuring that evolving stability, signaling when further computation is wasted. Because it is lightweight and algorithm-agnostic, it supports early exit, dynamic token allocation, gang scheduling, and other serving-system integrations, rather than being a per-request trick. Built into the Dynasor serving system it delivers up to **50% compute savings and 3.3x higher throughput on real workloads with no accuracy drop**. Reporting throughput alongside token savings puts it ahead of most of this literature on measurement rigor.

*Weak points.* "No accuracy drop" holds at the operating point they chose; the savings/accuracy curve elsewhere is less prominent. Stabilization is a weaker signal on problems where the model is consistently and confidently wrong — it detects convergence, not correctness. Throughput gains partly come from scheduling rather than from Certaindex itself, and the two are not fully separated.

### 1d. Latent chain-of-thought

Replaces reasoning tokens with vectors, deleting the token cost entirely. The most theoretically attractive and least practically ready branch.

**Coconut — Reasoning in a Continuous Latent Space.** Hao et al. (Meta) · [arXiv:2412.06769](https://arxiv.org/abs/2412.06769) · COLM 2025

Language may simply be the wrong medium for reasoning: most word tokens exist for textual coherence, not computation, so paying for them is waste. Coconut feeds the LLM's **last hidden state back in as the next input embedding**, reasoning in continuous space instead of decoding to words. The consequence is more interesting than the savings: a continuous thought can encode **multiple alternative next steps simultaneously**, letting the model do something like breadth-first search rather than committing to one path as token-CoT must. It outperforms CoT on logical reasoning tasks requiring substantial planning search, with a better accuracy/efficiency tradeoff. Evaluated on GSM8K, ProntoQA, and ProsQA.

*Weak points.* Gains are concentrated on synthetic planning tasks (ProntoQA/ProsQA) built to reward search; on GSM8K it does not clearly beat CoT. Training requires a multi-stage curriculum that is finicky and reported as such only in detail. Latent reasoning is **uninspectable**, which forfeits the auditability that is a large part of why CoT is deployed. Small models only.

**CODI — Compressing CoT into Continuous Space via Self-Distillation.** Shen et al. (KCL) · [arXiv:2502.21074](https://arxiv.org/abs/2502.21074) · EMNLP 2025 · [code](https://github.com/zhenyi4/codi)

Honestly states the field's problem: prior implicit-CoT methods bypass language entirely and have **consistently underperformed explicit CoT**. CODI's fix is self-distillation — jointly train a teacher task (explicit CoT) and a student task (implicit CoT) in the same model, aligning the hidden states of one designated token so reasoning ability transfers from language into continuous space. It is the first implicit-CoT approach to *match* explicit CoT on GSM8K at GPT-2 scale, at a 3.1x compression rate, beating the prior state of the art by 28.2% in accuracy. It also generalizes to more complex datasets and offers some interpretability by decoding the continuous thoughts. Code released.

*Weak points.* "First to match explicit CoT" is qualified by **at GPT-2 scale** — a 2019 model on a benchmark modern models saturate. 3.1x compression is far below what token-level methods achieve. The comparison baseline is explicit CoT from the same small model, not a competent modern one.

*Lineage note.* Deng et al.'s **Stepwise Internalization** ([arXiv:2405.14838](https://arxiv.org/abs/2405.14838)) is the precursor — gradually delete CoT steps and finetune until reasoning is internalized, getting GPT-2 Small to 99% on 9-by-9 multiplication and Mistral 7B past 50% on GSM8K with no intermediate steps. Meta's **Token Assorted** ([arXiv:2502.03275](https://arxiv.org/abs/2502.03275), ICML 2025) abstracts early reasoning steps into VQ-VAE latent tokens mixed with text. Both are superseded by CODI on the headline metric but remain the clearest statements of the idea.

### 1e. Production thinking-budget controls

**Kimi k1.5** (Moonshot AI, [arXiv:2501.12599](https://arxiv.org/abs/2501.12599)) and **Qwen3** (Alibaba, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)) matter here not as methods but as evidence that length control shipped. Kimi k1.5 uses an explicit **length penalty** in its RL objective plus long-to-short distillation to move long-CoT ability into short-CoT models. Qwen3 unifies thinking and non-thinking modes in one model with a user-facing **thinking budget**, letting callers allocate compute adaptively at inference. Together with Anthropic's `budget_tokens`, Gemini's `thinkingBudget`, and OpenAI's `reasoning_effort`, they establish the budget knob as standard infrastructure — which is why §7's papers are about *policies over* the knob rather than the knob itself.

*Weak point common to both.* These are technical reports with strong incentives to present favorable frontiers, no independent replication, and no ablation isolating the length mechanism's contribution from everything else in the training pipeline. Read the accuracy-vs-budget curves as marketing-adjacent until reproduced. LLMThinkBench (§7) is the closest thing to an independent audit, and it found zero gain from effort scaling on basic math for the GPT-5/o-series.

---

## §2 Test-time compute scaling and adaptive sampling budgets

Where §1 shortens one chain, §2 decides how many chains to buy. The three foundational papers below all landed in mid-2024 and set the agenda.

**Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters.** Snell et al. (UC Berkeley / Google DeepMind) · [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)

The canonical reference for allocating test-time compute, and the origin of the term *compute-optimal* in this setting. It asks a specific question: given a fixed non-trivial inference budget, how much can an LLM improve on a hard prompt? It analyzes two mechanisms — searching against dense process-based verifier reward models, and adaptively updating the model's response distribution at test time. The central finding is that the *right* strategy depends critically on prompt difficulty, so no fixed allocation is good; a compute-optimal strategy that allocates adaptively per prompt improves test-time compute efficiency by **more than 4x over a best-of-N baseline**. In a FLOPs-matched comparison, on problems where a smaller base model already has non-trivial success rates, test-time compute lets it **outperform a 14x larger model**. The difficulty-conditioning insight is what every adaptive-budget paper since has built on.

*Data.* MATH, with PaLM 2-S* models. Difficulty bins derived from model pass rates.

*Weak points.* The "14x larger model" headline is heavily conditioned — it holds only where the small model already has non-trivial success, i.e. explicitly not on the hard problems where you wanted the big model. FLOPs matching **excludes the verifier's forward passes**, which for dense PRM search is a large fraction of real cost; this is the single biggest accounting gap in the paper. Difficulty bins are computed using the model's own pass rate, information unavailable at inference, so the "compute-optimal" oracle is not implementable as described. One model family, one benchmark.

**Inference Scaling Laws: Compute-Optimal Inference for Problem-Solving.** Wu et al. (Tsinghua / CMU) · [arXiv:2408.00724](https://arxiv.org/abs/2408.00724)

The complementary study, focused on the model-size versus extra-tokens tradeoff rather than on per-prompt allocation. It systematically maps cost–performance for greedy search, majority voting, best-of-N, weighted voting, and two tree-search algorithms across model sizes and budgets. The conclusion is that scaling inference compute can be more computationally efficient than scaling parameters, and that **small models plus advanced inference algorithms are Pareto-optimal**. The concrete demonstration: Llemma-7B with their novel tree search consistently outperforms Llemma-34B across all tested inference strategies on MATH. Together with Snell et al. this pair established that inference is a scaling axis, not a fixed cost.

*Weak points.* Llemma is a math-specialized model family on the math benchmark it was built for, so the small-beats-large result may not survive on general tasks. Tree search requires a verifier or reward model whose cost is again not fully charged. Wall-clock and memory are not modeled — a 7B running deep tree search may be worse than a 34B doing greedy decode on a real serving stack.

**Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.** Brown et al. (Stanford) · [arXiv:2407.21787](https://arxiv.org/abs/2407.21787)

The most-cited demonstration that brute-force sampling scales, and simultaneously the clearest statement of its limits. Sampling repeatedly from one model, **coverage** — the fraction of problems solved by *any* sample — scales log-linearly with sample count over four orders of magnitude, fitting an exponentiated power law. Where answers are automatically verifiable, coverage converts directly into performance: on SWE-bench Lite, DeepSeek-Coder-V2-Instruct goes from 15.9% with one sample to **56% with 250 samples**, beating the then-SOTA single-sample 43%. The crucial caveat is delivered in the same abstract: in domains **without** automatic verifiers, majority voting and reward models **plateau beyond several hundred samples** and fail to use the budget. This is the coverage-versus-selection gap, and it is the reason most pass@k headlines elsewhere are misleading.

*Data.* GSM8K, MATH, MiniF2F, CodeContests, SWE-bench Lite. Public.

*Weak points.* 250 samples is a ~250x cost multiplier, which the framing ("outperforming SOTA") does not foreground; on a cost-normalized axis this is a very expensive way to buy 13 points. The result depends on having a verifier — SWE-bench has tests, most work does not. Read this paper *before* any paper quoting pass@k as an achievement.

**Adaptive-Consistency.** Aggarwal et al. (CMU) · [arXiv:2305.11860](https://arxiv.org/abs/2305.11860) · EMNLP 2023 · [code](https://www.sample-step-by-step.info)

Self-consistency polls the model a fixed number of times regardless of how quickly the answers agree, which wastes most of the budget on questions that were settled after three samples. Adaptive-Consistency adds a lightweight stopping criterion that halts sampling once agreement is sufficient, distributing the budget non-uniformly across questions. It is model-agnostic and requires no training, which is why it remains the default recommendation. Across **17 reasoning and code-generation datasets and three LLMs**, it cuts the sample budget by **up to 7.9x with average accuracy drop under 0.1%**. That breadth of evaluation is rare and makes it more credible than most entries here. It is the highest value-per-effort item in this entire bibliography.

*Weak points.* 7.9x is the best case; the average reduction is materially lower. It only helps where self-consistency was already the strategy — it reduces waste rather than beating single-sample on cost. The stopping rule has a threshold that trades accuracy for savings and needs per-task tuning. Early stopping on agreement is systematically wrong when the model is *confidently and consistently* wrong, which is exactly the hard-problem regime.

*Also:* **ESC (Early-Stopping Self-Consistency)**, Li et al., [arXiv:2401.10480](https://arxiv.org/abs/2401.10480), ICLR 2024 — same idea via sequential sampling windows, with per-benchmark reductions of −33.8% (MATH), −80.1% (GSM8K), −76.8% (StrategyQA), −84.2% (Coin Flip) at comparable performance. The spread across benchmarks is the useful data point: savings track how easy the benchmark is, which is the honest general lesson of this subsection.

**Fast Best-of-N Decoding via Speculative Rejection.** Sun et al. · [arXiv:2410.20290](https://arxiv.org/abs/2410.20290) · NeurIPS 2024

Best-of-N is as effective as state-of-the-art post-training alignment while avoiding the post-training complexity, but it demands vastly more inference resources, which makes it non-viable in practice. Speculative Rejection makes it viable by starting many generations and **killing low-scoring ones early**, using a reward model's partial-sequence scores to prune before completion. It produces high-scoring responses like Best-of-N while being **16 to 32 times more computationally efficient**. The insight that reward-model scores on partial generations correlate well enough with final scores to prune on is the transferable part.

*Weak points.* Efficiency is measured against full Best-of-N, an intentionally wasteful baseline. It inherits Best-of-N's dependence on a reward model whose forward passes over many partial sequences are a real cost. Early rejection is biased against responses that start weakly and recover — precisely the long-reasoning pattern that §1's papers show is valuable.

**The Illusion of Thinking.** Shojaee et al. (Apple) · [arXiv:2506.06941](https://arxiv.org/abs/2506.06941) · NeurIPS 2025

The most important skeptical paper on test-time scaling, and essential context for every optimistic claim above. Standard evaluations are contaminated and only score final answers, so the authors use **controllable puzzle environments** where complexity can be dialed precisely while logical structure stays constant, allowing analysis of the reasoning traces themselves. They find LRMs suffer **complete accuracy collapse beyond a complexity threshold**, and — most damning for budget research — a counterintuitive scaling limit where **reasoning effort rises with complexity up to a point and then declines even though token budget remains**. Comparing LRMs against standard LLMs at matched inference compute yields three regimes: standard models win at low complexity, LRMs win at medium, and both collapse at high. They further show LRMs fail to execute explicit algorithms and reason inconsistently across scales. The practical implication for budget-aware systems is that spending more tokens has a hard ceiling that arrives earlier than the scaling curves suggest.

*Weak points.* Puzzle environments (Tower of Hanoi and similar) are far from the tasks people deploy on, and the paper was widely criticized on this point — several rebuttals argued the "collapse" partly reflects output-token limits and the impracticality of writing out exponentially long solutions, not a reasoning failure. Black-box frontier models with unknown served configurations. The compute-matched comparison is the strongest part of the paper and is less discussed than the headline.

**A Survey on Test-Time Scaling: What, How, Where, and How Well?** Zhang et al. · [arXiv:2503.24235](https://arxiv.org/abs/2503.24235) · [repo](https://github.com/testtimescaling/testtimescaling.github.io/)

The best-organized entry point to §2. It structures the field along four dimensions — what to scale, how to scale, where to scale, how well to scale — and reviews methods, applications, and evaluation within that frame. It includes hands-on deployment guidelines rather than stopping at taxonomy, and maintains a live repository and website. Use it as a map, not a source of numbers.

*Weak point.* Like all surveys here, reported figures are copied across papers with incompatible base models, decoding settings, and budgets, so cross-method comparisons in the tables are not valid. No unified re-evaluation.

---

## §3 Routing, cascades, and model selection

Choose a cheaper *model* rather than fewer tokens. Read §3 with checklist item 5 (stale prices) permanently in mind: this subfield's value proposition is the most exposed to market movement of anything in this document.

**FrugalGPT.** Chen, Zaharia, Zou (Stanford) · [arXiv:2305.05176](https://arxiv.org/abs/2305.05176)

The paper that created the subfield. It documents that LLM API pricing is heterogeneous by **two orders of magnitude**, then lays out three cost-reduction strategy families — prompt adaptation, LLM approximation, and LLM cascade. FrugalGPT instantiates the cascade: a learned policy over which combinations of LLMs to query for which queries, calling cheap models first and escalating on low scores. It reports matching the best individual LLM (GPT-4) with **up to 98% cost reduction**, or beating GPT-4's accuracy by 4% at equal cost. Evaluation is on HEADLINES, OVERRULED, and COQA. Its taxonomy is still the standard framing even though its numbers are not.

*Weak points.* **The 98% figure is a 2023 artifact and should not be quoted today.** It depends on GPT-4 at 2023 prices against much cheaper alternatives; frontier prices have since fallen by one to two orders of magnitude while cheap-model quality rose, which compresses the arbitrage the cascade was harvesting. The three benchmarks are classification-flavored tasks where a small model is nearly sufficient and a scorer is easy to train — the best possible case for cascading. The cascade scorer needs labeled training data per task, a cost excluded from the accounting, and serial escalation adds latency. Reproduction requires paid API access to models that no longer exist.

**AutoMix.** Aggarwal et al. (CMU / Google) · [arXiv:2310.12963](https://arxiv.org/abs/2310.12963) · NeurIPS 2024

Routes to larger models based on the *approximate correctness* of a smaller model's output, rather than on a difficulty estimate made before seeing any output. Two contributions carry it: a **few-shot self-verification** mechanism that estimates output reliability without extensive training, and — because self-verification is noisy — a **POMDP-based router** that selects model size under that noise. Modeling verification noise explicitly is what distinguishes it from confidence-threshold cascades. Across five language models and five challenging datasets it reduces computational cost by **over 50% at comparable performance**.

*Weak points.* Self-verification costs an extra small-model call per query, and the POMDP's own overhead is not charged. Few-shot self-verification is known to be poorly calibrated, and the POMDP mitigates rather than fixes this. 50% against a route-everything-to-large baseline, which nobody actually runs.

**RouteLLM.** Ong et al. (UC Berkeley / LMSYS) · [arXiv:2406.18665](https://arxiv.org/abs/2406.18665)

The most-used open routing work, largely because of what it trains on. Rather than task labels, it learns routers from **human preference data** (Chatbot Arena) plus data augmentation, selecting between one strong and one weak model per query. It reports cost reductions of **over 2x in some cases** without quality loss. The genuinely valuable finding is **transfer**: the routers keep working when the strong and weak models are swapped at test time, which means a router need not be retrained every model release — the single biggest practical objection to routing.

*Weak points.* Binary strong/weak routing is a simplification of a real multi-model fleet. "Over 2x in certain cases" is much weaker than FrugalGPT-era claims, which is a sign of more honest accounting rather than a worse method. Quality is measured by preference-model and benchmark proxies, so the router is partly optimizing the same signal it is evaluated on. Preference data at Arena scale is not something most teams can collect.

**Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing.** Ding et al. (UBC / Microsoft) · [arXiv:2404.14618](https://arxiv.org/abs/2404.14618) · ICLR 2024

Frames routing around the edge/cloud split: big models need expensive cloud servers, small models fit on cheap or edge devices but lag in quality. The router assigns queries by predicted difficulty **and a desired quality level**, and the quality target is a dial that can be **tuned dynamically at test time** to trade quality for cost per scenario. Exposing the operating point as a runtime parameter rather than a training-time choice is the contribution worth copying. It reports up to **40% fewer calls to the large model with no drop in response quality**.

*Weak points.* "No drop in quality" is measured by a quality metric the router was trained against, which is close to circular. 40% is call-count, not cost — and not latency, since the small model runs first on every query. Difficulty prediction from the prompt alone is the weakest link and degrades under distribution shift.

**Language Model Cascades: Token-Level Uncertainty and Beyond.** Gupta et al. (Google Research) · [arXiv:2404.10136](https://arxiv.org/abs/2404.10136)

The most technically careful paper in §3 and the one that explains why naive cascades underperform. Deferral by predicted uncertainty is well understood for *classification* but not for *generation*, and the authors show the natural extension — predicted sequence uncertainty — suffers a **length bias**, over- or under-weighting outputs by their length, because an LM emits one uncertainty value per token and token counts vary across examples. Their fix exploits the richer **token-level** uncertainty that simple aggregation throws away, via learned post-hoc deferral rules. These significantly outperform aggregation-based deferral across natural language benchmarks with FLAN-T5. Adding embeddings from the small model and intermediate layers of the large model improves the cost-quality tradeoff further. If you are building a cascade, this is the paper that tells you how to write the deferral rule.

*Weak points.* FLAN-T5 only, which is small and encoder-decoder — generalization to modern decoder-only chat models is assumed. Learned post-hoc deferral rules need labeled data per task, so the practical cost is higher than confidence thresholding. Requires logits, ruling out most closed APIs.

**When Does Confidence-Based Cascade Deferral Suffice?** Jitkrittum et al. (Google Research) · [arXiv:2307.02764](https://arxiv.org/abs/2307.02764) · NeurIPS 2023

The theory companion. Confidence-based deferral is oblivious to cascade structure — it never models downstream errors — yet works remarkably well, and this paper characterizes exactly when it fails. It gives a theoretical characterization of the optimal deferral rule, then identifies three concrete regimes where post-hoc deferral mechanisms beat confidence: when **downstream models are specialists** good on only a subset of inputs, when samples carry **label noise**, and when there is **distribution shift** between train and test. Those three conditions describe most real deployments, which is the practical takeaway.

*Weak points.* Primarily classification-framed, so applying it to generative cascades requires the bridge that Gupta et al. build. Post-hoc mechanisms need held-out data with downstream-model labels.

**UniRoute — Universal Model Routing for Efficient LLM Inference.** Jitkrittum et al. (Google Research) · [arXiv:2502.08773](https://arxiv.org/abs/2502.08773)

Addresses the objection that kills routing in practice: existing routers assume a **fixed pool**, but new models appear constantly and retraining per release is untenable. UniRoute handles **dynamic routing to LLMs unseen at training time** by representing each LLM as a feature vector derived from its predictions on a set of representative prompts. Two instantiations are given, cluster-based routing and a learned cluster map, both shown to be estimates of a theoretically optimal routing rule with an excess-risk bound quantifying the error. Experiments route among **more than 30 unseen LLMs** on public benchmarks. This is the most deployment-relevant routing paper of the group.

*Weak points.* Characterizing each new model still requires running it on the representative prompt set — cheaper than retraining but not free, and it must be redone when a provider silently updates a model behind a stable name. The excess-risk bound relies on assumptions about cluster structure that real model pools may violate. Benchmark-based evaluation does not test the shifting, messy traffic that motivates dynamic routing.

**RouterBench.** Hu et al. (Martian) · [arXiv:2403.12031](https://arxiv.org/abs/2403.12031) · [code](https://github.com/withmartian/routerbench)

The standard evaluation substrate for §3, and the reason routing papers became comparable at all. It provides an evaluation framework plus a dataset of **over 405k inference outcomes** from representative LLMs, so researchers can simulate routing decisions offline instead of spending money on APIs. It also offers a theoretical framework for routing and a comparative analysis of existing approaches. Being able to evaluate a router without paid inference is what makes this useful.

*Weak points.* Pre-computed outcomes freeze a model set that is now dated, and the cost model embeds the API prices of its era — the exact staleness problem in checklist item 5. Offline simulation cannot capture latency, rate limits, or provider variance. Authored by a company selling routing, which is worth noting when reading the framing.

---

## §7 The 2026 frontier: solvability-aware and risk-controlled budgeting

Placed early because it is the current state of the art. The 2026 shift is from *difficulty*-conditioned allocation (spend more on hard problems) to **solvability**-conditioned allocation (spend more only where the model can still win). The hardest problems are frequently where extra tokens are pure waste.

A caution that applies to BET, Conformal Thinking, and BAGEN alike: **much of the reported savings is abstention, not efficiency.** Giving up early on problems you would have failed is nearly free on an accuracy metric and expensive in product terms. Read every "we also improved accuracy" claim in this section with that in mind.

The two agent-budget papers here are complements and should be read together. **BAGEN** establishes that agents *cannot* reliably estimate their own remaining budget (all models optimistically biased, interval coverage only 47% after SFT+RL), which is precisely why **BAVT** does not ask the model to self-regulate against a budget hint and instead makes the budget an external annealing parameter on the search. BAGEN is the diagnosis; BAVT is one response to it.

### BET — Nice Fold or Hero Call: Learning Budget-Efficient Thinking
[arXiv:2605.11625](https://arxiv.org/abs/2605.11625) · May 2026 · no code found

**Summary.** Prior efficiency work conditions budget on *perceived difficulty* and ignores *solvability*, so models burn large budgets on problems beyond their capability while compressing hard-but-solvable ones that needed the depth. BET reframes adaptive reasoning as investment under uncertainty: budget should track the expected *return* of reasoning, not difficulty. It is two-stage — behavioral cold-start, then GRPO under an investment-cost-aware reward that aligns solve-or-fold decisions with rollout-derived solvability labels. Three behaviors emerge: *short solve* (easy queries, concise), *nice fold* (abstain when expected return is near zero), *hero call* (preserve depth for hard-but-solvable). Across seven benchmarks and three base models it cuts reasoning tokens ~55% on average while reporting overall accuracy gains, with >90% token reduction on unsolvable queries. It transfers zero-shot from math to scientific QA and logical reasoning.

**Data.** In-domain: Omni-Math, MATH-500, AMC-23, AIME-2025. OOD: GPQA-Diamond, MuSR, LSAT-AR. Models: Qwen3-4B, R1-Distill-Qwen-7B/14B. All benchmarks public on HF. **The reproduction bottleneck is the rollout-derived solvability labels** — many rollouts per training question — plus the GRPO pipeline, neither released.

**Weak points.**
- The headline saving is mostly abstention: >90% cut on "unsolvable" queries restates that the model stopped trying.
- "Overall performance improvements" is partly a selection artifact — folding on low-solvability items raises the mean without raising capability. The honest comparison is accuracy on *attempted* items vs. a compute-matched baseline.
- Solvability labels are policy-dependent and estimated on the same math distribution as the in-domain evals.
- AMC-23 (40 problems) and AIME-2025 (30) cannot resolve the per-benchmark deltas claimed.
- Multi-rollout labeling plus GRPO is a large one-time cost absent from the 55% figure.

### TAB — Not All Turns Are Equally Hard: Adaptive Thinking Budgets for Multi-Turn Reasoning
[arXiv:2604.05164](https://arxiv.org/abs/2604.05164) · Apr 2026 · no code found

**Summary.** Length regularization, routing, and difficulty-based allocation are all single-turn and cannot handle multi-turn sequential dependencies, where an underfunded early turn wrecks a later one. The paper models multi-turn reasoning as a scalarized multi-objective MDP over compute allocation. TAB is a GRPO-trained allocation policy that reads conversation history and distributes a *global per-problem* token budget across turns, spending less on easy turns to bank tokens for crucial ones. On math reasoning benchmarks it saves up to 35% of tokens while maintaining or improving accuracy over static budgets and off-the-shelf LLM-judge baselines. A variant, TAB All-SubQ, budgets using all past *and future* sub-questions when the full plan is known in advance, saving up to 40%. The framing — budget as a sequential allocation problem rather than a per-call cap — is the contribution that generalizes.

**Data.** Standard math benchmarks decomposed into multi-turn sub-question sequences. The decomposition is the actual novel dataset and its release status is unclear; the underlying problems are public.

**Weak points.**
- "Up to 35%/40%" are best cells; the average is what matters and is not the headline.
- TAB All-SubQ assumes an oracle plan of all future sub-questions — precisely the information that makes budgeting easy. Read 40% as an upper bound no online method can reach.
- Math-only, with clean pre-decomposition. Real multi-turn (dialogue, tool loops, code iteration) has far less predictable turn difficulty.
- The router runs *every turn*, so its overhead accumulates in exactly the setting being optimized.
- Static baselines under a global cap are easy to under-tune relative to a learned policy.

### Conformal Thinking: Risk Control for Reasoning on a Compute Budget
[arXiv:2602.03814](https://arxiv.org/abs/2602.03814) · Feb 2026 · no code found

**Summary.** Adaptive reasoning is well-motivated but nobody has a principled way to *set* the budget or the stopping threshold, both of which encode a risk–accuracy tradeoff. This paper recasts budget-setting as distribution-free risk control: given a target risk and a validation set, choose stopping thresholds that bound the error rate while minimizing compute. Two thresholds handle two distinct wastes — an *upper* threshold stops once the model has converged (risking a wrong answer), and a novel *parametric lower* threshold preemptively kills instances whose reasoning is not making progress (risking premature stoppage). When several stopping criteria are available, an efficiency loss picks the cheapest adequate one, including ensembles of rules. Experiments across reasoning tasks and models show compute savings from the lower threshold and from ensemble stopping while hitting the user-specified risk target. The lower threshold — measuring reasoning *progress* rather than confidence — is the genuinely new mechanism.

**Data.** Standard math/reasoning suites and models. The real dependency is a **labeled held-out calibration set from the deployment distribution**.

**Weak points.**
- Conformal guarantees require exchangeability; production traffic drifts, and the "provable" bound quietly stops holding when it does.
- Coverage is marginal, not conditional — the aggregate target can be met while error concentrates on hard subgroups, which is where you wanted the guarantee.
- Needs per-task labeled calibration data. Cheap for verifiable math; for open-ended generation, obtaining correctness labels *is* the problem.
- The lower threshold is abstention again.
- Confidence/progress scoring may need extra passes; not clearly netted out of savings.

### BAGEN: Are LLM Agents Budget-Aware?
[arXiv:2606.00198](https://arxiv.org/abs/2606.00198) · Jun 2026

**Summary.** A diagnostic, not a method: can agents estimate their own remaining budget and notice when a task has become infeasible — the prerequisite for any agent-level budget control. Twenty model–environment pairs are tested on binary feasibility judgment and on interval estimation of remaining budget. The clean finding is a decomposition: **feasibility is a calibration problem, interval estimation is a reasoning problem**, and every model is optimistically biased about remaining budget, with weaker models *more* optimistic. SFT alone lifts Qwen-7B feasibility accuracy from 25.5% to ~90%, so the capability was latent and merely unelicited; interval coverage reaches only 47% after SFT+RL, with 28% median midpoint relative error. Acting on the signal works — early-stopping on predicted-impossible saves 28–64% of tokens on failed trajectories for 1.6–4.2 points of success rate. Training is fragile: RL without SFT warm-start collapses entirely.

**Data.** Sokoban, Warehouse, SWE-bench, Search-R1 across twenty model–environment pairs including GPT-5.2 and Claude Opus. Environments public; the trajectories and budget annotations are the novel data, and reproduction needs substantial frontier-API spend.

**Weak points.**
- The savings denominator is *failed* trajectories only. Workload-level savings depend entirely on your failure rate; the framing invites over-generalization.
- False aborts are the real cost and are reported softly: GPT-5.2 has both the best savings (64%) and the worst false-abort rate (6.6%).
- Near-zero benefit on Search-R1 — rollouts end before infeasibility fires. It helps only where runs are long.
- Frontier-model results are black-box: unobservable served configs and hidden reasoning-token billing make the token accounting approximate.
- Agent runs are non-deterministic; twenty pairs with few repeats leaves wide error bars everywhere.

### BAVT — Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents
[arXiv:2603.12634](https://arxiv.org/abs/2603.12634) · Mar 2026 · no code found

**Summary.** Test-time scaling treats compute as abundant, letting agents burn token and tool budgets on redundant steps and dead-end trajectories, while existing budget-aware methods either need expensive fine-tuning or rely on coarse trajectory-level heuristics that cannot intervene mid-execution. BAVT is a **training-free** inference-time framework that models multi-hop reasoning as a dynamic search tree guided by step-level value estimation inside a single LLM backbone. Its distinctive mechanism is **budget-conditioned node selection**: the remaining resource ratio becomes a scaling exponent over node values, so search anneals from broad exploration to greedy exploitation automatically — near-uniform weighting when budget is plentiful, effectively winner-takes-all as it runs out. To counter the well-documented overconfidence of LLM self-evaluation, a **residual value predictor** scores *relative progress* against the parent node rather than absolute state quality, making pruning of uninformative or redundant tool calls more reliable. The paper adds a convergence guarantee that BAVT reaches a terminal answer with probability at least 1−ε under an explicit finite budget bound. Across four multi-hop QA benchmarks and two model families, BAVT at **Low budget (5 tool calls) rivals or surpasses the baseline at High budget (20 calls)**, a 4x resource advantage, supporting the claim that allocation beats brute-force scaling.

**Data.** HotpotQA, 2WikiMultihopQA, MuSiQue, and Bamboogle — chosen because they require sequential tool use and cannot be answered from parametric memory. Models: GPT-OSS-20B (reasoning) and Qwen3-30B-A3B-Instruct-2507 (instruct). Built in Inspect AI following the Search-R1 retrieval setup: 2018 Wikipedia dump, E5 dense retriever, exactly five passages per query, scored by Exact Match and F1. Benchmarks and retrieval stack are public, making this more reproducible than most of §7, but no code release was found.

**Credit where due.** The baseline is budget-constrained majority voting that **natively consumes the identical tool-call and token budget** — a genuinely compute-matched comparison, which checklist item 4 notes most of this literature skips.

**Weak points.**
- **The paper's own cost table undercuts its framing.** Appendix pricing puts the search API at \$0.005/query, **over 90% of cost per sample** at every budget tier: roughly \$0.001–0.004 of tokens against \$0.025–0.100 of search. BAVT is therefore a *tool-call* budget method whose dollar savings come from issuing fewer queries, not from token efficiency — despite being framed as test-time scaling and token budgeting. With a cheap or self-hosted retriever the savings largely evaporate.
- **The 4x claim compares across budget tiers, and the baseline may be degrading rather than BAVT improving.** Majority voting is non-monotonic in call count (Chen et al., §5): more calls help easy queries and hurt hard ones. If the high-budget baseline sits past its own peak, "low-budget BAVT beats the 4x-budget baseline" partly measures the baseline falling over.
- **The convergence guarantee is liveness, not quality.** Reaching a terminal answer with probability ≥1−ε means the agent stops and emits something, which any agent with a budget backstop does trivially. It says nothing about correctness, and reads as stronger than it is.
- **"Parameter-free" covers only the annealing exponent.** The hyperparameter table lists a 1–10 critic scale, residual deltas clipped to [−4,+4], values normalized to [0.1,1.0], a 0.8 terminal confidence threshold, a 0.2 budget backstop, and 512 max output tokens per call.
- **The critic is still LLM self-evaluation.** Scoring relative progress mitigates overconfidence rather than removing it, and no calibration study of the residual predictor is reported.
- **One baseline family.** Compute-matched parallel sampling is fair but weak; there is no comparison against other budget-aware agent methods (e.g. BATS, arXiv:2511.17006, cited in related work) or a well-tuned sequential ReAct agent with a sensible stopping rule.
- **A static, forgiving environment.** A frozen 2018 Wikipedia dump with a fixed retriever has no tool failures, latency, rate limits, or ambiguity, unlike the live browsing the motivation invokes. Bamboogle is 125 questions; subset sizes for the others are not foregrounded. Two mid-size open models, no frontier model, and EM/F1 without seed variance or significance testing.

### ARES: Adaptive Reasoning Effort Selection for Efficient LLM Agents
[arXiv:2603.07915](https://arxiv.org/abs/2603.07915) · Mar 2026

**Summary.** Agents on thinking LLMs buy accuracy with long CoT, and although most models now expose reasoning levels (high/medium/low), static policies fail — low everywhere degrades badly, random selection preserves neither accuracy nor cost. The insight is that effort should vary *per step within a trajectory*: high for navigating complex site structure, low for opening a known URL. ARES trains a lightweight router to predict the lowest adequate reasoning level per step from interaction history, supported by a pipeline that labels the minimum effort each step needed to still succeed. Because it toggles a model's *own* thinking levels rather than swapping models, ARES argues the cost–performance frontier is more monotonic and predictable than cross-model routing. The router is plug-and-play for any agent. On TAU-Bench, BrowseComp-Plus, and WebArena it cuts reasoning tokens by up to 52.7% versus fixed high effort with minimal success-rate loss.

**Data.** TAU-Bench, BrowseComp-Plus, WebArena — public, all requiring real API spend. The **per-step minimum-effort labels** are the novel asset, generated by expensive search over effort levels, and are the reproduction bottleneck.

**Weak points.**
- 52.7% is the best cell, and it is against *fixed high effort* — the most expensive baseline, not what a cost-conscious team runs. The number that matters is the margin over a tuned fixed medium, which is not the headline.
- Label generation requires re-running steps at multiple effort levels; this one-time cost can exceed years of savings for a small deployment.
- The router is trained against one model's effort behavior and goes stale whenever the provider updates the model — every few months for frontier APIs.
- WebArena and TAU-Bench success rates swing across runs; "minimal degradation" needs multi-seed error bars.
- A per-step router in an inner loop is not free in latency or tokens across a long trajectory.

### When More Thinking Hurts: Overthinking in Test-Time Compute Scaling
[arXiv:2604.10739](https://arxiv.org/abs/2604.10739) · Findings of ACL 2026

**Summary.** The scaling literature assumes longer thinking monotonically helps; this paper measures the *marginal utility* of additional reasoning tokens as budget grows and finds it does not. Returns diminish sharply at high budgets, and models overthink — extended reasoning is associated with **abandoning previously correct answers**. Answer-flip tracking makes the mechanism concrete: positive flips (wrong→right) dominate at low budgets, but past ~7K tokens negative flips become more frequent, with all flip ratios statistically significant at ≥7K. Optimal length varies strongly with difficulty — MATH Level 1–2 problems cross the overthinking threshold around 2K tokens versus ~8K for Level 5 — so uniform allocation is provably suboptimal. Under a cost-aware framework with weight λ, the optimal budget collapses as λ rises: λ=0.5 favors stopping near 6K tokens for ~50% compute reduction at ~6% accuracy loss, λ=1.0 favors ~2K. Indicator-based early stopping reaches 97% of peak accuracy at 60% of compute, and their overthinking indicators predict negative flips at 76.3% precision / 80% recall.

**Data.** MATH problems stratified by the dataset's own Level 1–5 labels, swept across token budgets. Reproduction means a full budget sweep per model; trace release unconfirmed.

**Weak points.**
- λ is a free parameter that manufactures the conclusion. "50% reduction for 6% loss" is one point on a curve; there is no principled way to choose λ.
- 6% accuracy loss is presented as an acceptable trade. In most production settings a 6-point regression is disqualifying.
- "Overthinking" partly measures sampling variance in longer stochastic generations, not a distinct cognitive failure.
- The 7K threshold is model- and dataset-specific but reads as a general finding.
- Math only, with benchmark-assigned difficulty labels standing in for actual model competence.

### LLMThinkBench: Do LLMs Overthink Basic Math Reasoning?
Srivastava, Hussain, Srinivasan, Wang (Virginia Tech) · [arXiv:2507.04023](https://arxiv.org/abs/2507.04023) · Findings of ACL 2026
[github](https://github.com/ctrl-gaurav/LLMThinkBench) · `pip install llmthinkbench` · [leaderboard](https://ctrl-gaurav.github.io/LLMThinkBench/)

**Summary.** Models ace hard math benchmarks yet still fail *basic* arithmetic while being wildly verbose, and this benchmark targets that accuracy–verbosity tradeoff directly. It formalizes the tradeoff, introduces an **Overthinking Score** (harmonic mean of accuracy and token efficiency, so a model must be both correct and concise), and defines a protocol over 14 basic math tasks with **dynamically generated** instances, which structurally defeats contamination. The study is unusually large: 53 LLMs including reasoning and quantized variants across reasoning budgets. Reasoning models generate **~18x more tokens** while sometimes scoring *lower*, and collapse by up to ~36% when tokens are constrained. The accuracy–verbosity curve is non-monotonic, and GPT-5/o-series models show **zero accuracy gain** from low→medium→high reasoning effort on these tasks. It ships as a pip package with a public leaderboard.

**Data.** 14 basic math tasks generated at runtime rather than distributed — nothing to download, nothing to contaminate. Fully reproducible via the package with vLLM and API backends. Best reproducibility story in this bibliography.

**Weak points.**
- Deliberately trivial tasks cut both ways: "18x tokens for no gain" on sorting eight numbers indicts default verbosity but says nothing about whether effort is wasted on the hard problems people actually buy reasoning models for. The title invites exactly that over-generalization.
- The Overthinking Score min-max normalizes token efficiency **across the evaluated pool**, so a model's score changes when other models are added or removed. The metric is not portable and the ranking is sensitive to pool composition.
- The 36% "collapse" under token limits is partly a formatting artifact — truncating mid-thought yields unparseable answers scored as wrong. Real deployment problem, but graceful-degradation failure ≠ reasoning failure, and the two are not separated.
- Closed-model claims depend on unauditable provider behavior behind the effort parameter.
- Dynamic generation trades contamination for construct validity: templated synthetic tasks are a different confound, not no confound.

### SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning
Li, Dong, Ma, Zhang, Jia, Sui (PKU et al.) · [arXiv:2505.11274](https://arxiv.org/abs/2505.11274) · Findings of ACL 2026

**Summary.** Reasoning models overspend on simple queries, wasting compute and user patience, and this paper argues the fix should be adaptive *and* user-controllable. SelfBudgeter trains the model to emit a self-estimated minimal token budget *before* reasoning, then to generate within either that estimate or a user-supplied budget. Training is two-phase: cold-start to produce budgets in a standard format, then **budget-guided GRPO** rewarding adherence without accuracy loss. The interaction design is the real contribution — an up-front budget lets users see expected wait time and interrupt, or hard-cap length by pre-filling the field. Reported compression is 61% average response length for the 1.5B model and 48% for the 7B, with near-undiminished accuracy. Evaluation covers GSM8K, MATH-500, and AIME-2025.

**Data.** GSM8K, MATH-500, AIME-2025 — all public HF datasets. Base models are R1-distilled Qwen 1.5B/7B. GRPO pipeline and cold-start budget annotations are the reproduction bottleneck.

**Weak points.**
- Savings drop 61%→48% going 1.5B→7B. The trend suggests gains shrink as models improve — the opposite of what you want from a method pitched at expensive frontier reasoning.
- Evaluated only at 1.5B and 7B, the two scales where CoT is most bloated relative to its usefulness.
- "Nearly undiminished accuracy" on AIME-2025 (30 problems) at 1.5B is measuring noise.
- The model both sets the budget and is scored on hitting it. Adherence is trivially improved by predicting generously; whether the estimate is *calibrated* is the question, and adherence metrics cannot answer it.
- Inherits token elasticity (see TALE, §1): budgets set too low can produce *more* tokens than looser ones.
- An ICLR 2026 submission of this work was withdrawn before the ACL Findings version.

### CoThink — The Price of a Second Thought: Evaluating Reasoning Efficiency
[arXiv:2505.22017](https://arxiv.org/abs/2505.22017) · May 2025

**Summary.** The field lacks a clean definition of reasoning efficiency, so the paper supplies one: token efficiency τ(M,D) = accuracy / average generated tokens, measured at fixed temperature so counts are comparable. Using that lens it locates where reasoning models waste compute and proposes CoThink, where a cheaper model drafts a solution outline and a reasoning model completes it — sidestepping the need to estimate difficulty up front. Evaluation is a clean 3x3: three reasoning models (DAPO, R1-Distill, QwQ) against three baselines (SoloThink, Best-of-N, NoThinking) on three benchmarks. Averaged over the nine settings, CoThink cuts total tokens 22.3% (up to 41.8%) with accuracy within 0.42%. The authors note consistent cross-model trends they suggest may indicate a reasoning-efficiency scaling law, and explicitly flag this as speculative. Fixing temperature at 0.6 across all models is a methodological detail more papers should copy.

**Data.** GSM8K, MATH-500, AIME-2024 — public.

**Weak points.**
- 22.3% average is modest; 41.8% is the best case and leads.
- Two-model pipelines need the drafter's tokens in the total to be meaningful, and the comparison is not matched on latency (two serial calls) or memory (two models resident).
- Accuracy/tokens as a ratio rewards answering instantly and badly. It encodes no notion of a task's minimum necessary reasoning, yet is used to compare across benchmarks of different difficulty.
- AIME-2024 is 30 problems; a "0.42% margin" is an artifact of averaging nine settings, not a resolvable measurement.
- The scaling-law gesture is unsupported by a 3x3 grid and will be cited without the authors' hedge.

---

## §4–§6 — not yet written

All arXiv IDs below were verified against the arXiv API (title, authors, and date confirmed); the annotated entries have not been written yet. Listed so nothing is lost.

**§4 Input-side compression.** LLMLingua [2310.05736], LongLLMLingua [2310.06839], LLMLingua-2 [2403.12968], Gist Tokens [2304.08467], ICAE [2307.06945], AutoCompressor [2305.14788], RECOMP [2310.04408], xRAG [2405.13792], Provence [2501.16214], Activation Beacon [2401.03462], Skeleton-of-Thought [2307.15337], Cache-Augmented Generation [2412.15605]. KV-budget lineage: H2O [2306.14048], StreamingLLM [2309.17453], SnapKV [2404.14469], PyramidKV [2406.02069], MInference [2407.02490], DuoAttention [2410.10819]. Adaptive depth: LayerSkip [2404.16710], Mixture-of-Depths [2404.02258].

**§5 Agents and multi-agent economics.** More Agents Is All You Need [2402.05120], Are More LLM Calls All You Need? [2403.02419] (the key compute-matching critique), Multiagent Debate [2305.14325], Why Do Multi-Agent LLM Systems Fail? [2503.13657], Small Language Models are the Future of Agentic AI [2506.02153], The Danger of Overthinking [2502.08235], MemGPT [2310.08560], Chain of Agents [2406.02818], Adaptive-RAG [2403.14403], Self-RAG [2310.11511], FLARE [2305.06983], Let's Think Dot by Dot [2404.15758], Following Length Constraints in Instructions [2406.17744], Length-Controlled AlpacaEval [2404.04475], speculative decoding [2211.17192].

**§6 Surveys, benchmarks, economics.** Stop Overthinking [2503.16419], Harnessing the Reasoning Economy [2503.24377], Survey of Efficient Reasoning for LRMs [2503.21614], Towards Reasoning Era [2503.09567], Efficient Inference survey [2404.14294], Cost-of-Pass [2504.13359], Underthinking [2501.18585], A Long Way To Go: length correlations in RLHF [2310.03716], Same Task More Tokens [2402.14848], Power Hungry Processing [2311.16863], The Price of Prompting [2407.16893].

**Note on §3.** Chen et al.'s *Are More LLM Calls All You Need?* [2403.02419] is listed under §5 but is load-bearing for §2 and §3 as well: it shows Vote and Filter-Vote performance can **rise then fall** with more LM calls, because more calls help easy queries and hurt hard ones, so a task mixing both produces non-monotone scaling. That result is the basis for the BAVT critique above and is the single best argument for compute-matched baselines in this literature.
