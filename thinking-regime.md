# Thinking regime — same claim, different length law

**Verdict.** Frozen Qwen3-8B still generates; tiny probes still estimate **V** = P(correct) and **T** = remaining tokens; the score is still `V · P(T ≤ B_rem)` versus V-only and a chain at a matched token budget. What failed is the regime, not the claim. Non-thinking MATH under a custom one-line-per-step prompt yields ~300-token traces. Width-3 search at every newline lost to a chain; V-only ≈ V+T. An oracle on labels still used 11.3% fewer tokens than V-only (14.3% on Level 5) after a ~32-token extra-candidate tax, so T is real and a wide tree on short CoT cannot pay for itself. Native thinking on AIME is the same backbone with a different `(π, exam, length law)`: remaining length in the thousands. Prefer a shorter thinking path at similar accuracy — do not compress thinking to native 300-token CoT.

## What already ran (and lost)

Policy: Qwen3-8B **non-thinking** plus a custom `"Reasoning Steps:"` system prompt — not native Qwen. Mean MATH-500 traces **291–319 tokens**. Search: width-3 every newline.

MATH-500, seed 0:

| arm | B=512 | B=1024 |
|---|---|---|
| chain | acc 0.59 / 291 tok | 0.65 / 319 tok |
| V+T tree | 0.28 / 479 tok | 0.58 / 730 tok |

V-only matched V+T; the tree lost.

The T probe was a **weak selector, not broken**: sibling ranking **0.65** vs k=8 ceiling **0.71** at HF layer 26 (27/36 = 0.75 depth). T bins used `max_tokens=1024`. Tele-Lens ([arXiv:2602.02103](https://arxiv.org/abs/2602.02103)) already found that early CoT does not predict global length on MATH-like tasks, and that the best answer layer sits mid-late (~0.75).

The step protocol was **instrumentation, not theory**. Newline cuts made hidden-state caching and a ReProbe-style comparison cheap (MATH non-thinking ~204 tokens, [arXiv:2511.06209](https://arxiv.org/abs/2511.06209)). Native non-thinking is a different π. It is the wrong length target on AIME.

Oracle V+T still beat V-only on labels, tax included. Remaining length on that CoT is tens to a few hundred tokens. Expanding every newline at width 3 spends candidates faster than sibling gaps repay. BAVT ([arXiv:2603.12634](https://arxiv.org/abs/2603.12634)) uses 2k–8k budgets and rare/annealed search, not every-step branching on 300-token math. ReProbe is the V-only arm; it did not beat the chain once search tokens were charged.

## AIME thinking audit (n=6)

Same Qwen3-8B, `enable_thinking=True`, no step prompt. High truncation at 4096. Sibling spread from a 1024-token prefix, completions capped at 8192:

- 3 pairs finished, both correct, gaps **405, 447, 937** tokens
- 3 pairs both truncated at 8192, both wrong, gap censored to **0**
- `frac_gap_gt_256 = 0.5`, `go = true`
- no mixed correct/incorrect siblings

Thinking remaining length is thousands. A discarded full completion costs thousands. A 256-token PEEK can be paid by a 400–900 token gap.

This is not the overthinking result (*When More Thinking Hurts*): extra tokens on one chain can flip an answer. V+T is narrower — among siblings with similar V, keep the one that still fits `B_rem`.

## Old probe is unusable

T was binned to 1024 tokens and trained on non-thinking step-prompt hidden states. Thinking prefixes are OOD on the residual stream and on the length support. *How Much is Left?* ([arXiv:2607.05316](https://arxiv.org/abs/2607.05316)) is a linear T-probe with a 1024 cap that **drops** truncated traces (survivorship). STAR ([arXiv:2510.13668](https://arxiv.org/abs/2510.13668)) predicts remaining length in the thousands on R1-Distill. We need that scale, without dropping the truncations that dominate AIME at 4096.

Retrain T and V on thinking prefixes, cut at **chunk** boundaries, not `Step N`. Do not headline Qwen3-8B MATH-500 thinking (97.4% vs 87.4% non-thinking, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)): V is saturated there. AIME is where length still has room.

## Next experiment

| | Non-thinking MATH (done) | Thinking AIME (next) |
|---|---|---|
| π | custom step prompt | native `enable_thinking=True` |
| typical length | 291–319 tok traces | remaining T in the thousands |
| search vs tax | ~32 tok/candidate; tree loses | 256-tok PEEK vs 400–900 gaps |
| exam | MATH-500 (V not saturated) | AIME (not MATH thinking 97.4%) |

Same backbone. Compare V+T vs V-only vs chain at a matched budget. Prefer a shorter thinking path with similar accuracy. Do not retarget thinking to 300 native tokens.

## 20 GiB MIG constraints (this retrain)

Qwen3-8B bf16 is already on disk (~16 GiB). Do **not** download R1-Distill. Capture **HF layer 26 only**, last token of parent+peek (~1.3k tokens), never the 4–8k completion and never four layers. Generation is **batch size 1**. Completions are JSON counts, not hidden maps. Pilot disk is <1 MiB.

`--complete-tokens 4096` is the memory/time default (KV ≈ 1 GiB on top of the 16 GiB weights). The AIME audit already ran 8192 on this slice; raise the cap only after 4096 fits. `--limit 20 --k 2` is a **pipeline smoke**, not a publishable probe (~40 states). Scale `limit` after it finishes without OOM.

## Next GPU command

```bash
python scripts/generate_thinking_siblings.py --limit 20 --k 2 --width 2 \
    --parent-tokens 1024 --peek-tokens 256 --complete-tokens 4096 \
    --dtype bfloat16
```

After `states.jsonl` exists: `build_thinking_cache.py` then `train_probe.py --max-tokens 8192 --layers 26 --no-same-trace`.
