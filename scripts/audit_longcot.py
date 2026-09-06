#!/usr/bin/env python3
"""Length/accuracy audit for long-CoT policies. No hidden states, tiny on disk.

  # reuse the Qwen3-8B already cached; thinking ON; no short-step system prompt
  python scripts/audit_longcot.py --dataset aime24 --limit 8 --samples 2 \
      --max-new-tokens 4096 --thinking --dtype bfloat16

Writes one JSONL of (n_tokens, truncated, correct). Print mean/p50/p90.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from tokenaware.config import ARTIFACTS_DIR, MODEL_ID
from tokenaware.data import load_eval_set
from tokenaware.generate import load_model
from tokenaware.scoring import rollout_correct
from tokenaware.steps import extract_boxed

THINK_PROMPT = (
    "Solve the problem. Put the final answer after the reasoning, in \\boxed{}."
)


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_think_prompt(tokenizer, problem: str, thinking: bool) -> str:
    messages = [
        {"role": "user", "content": f"{THINK_PROMPT}\n\n{problem}"},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=thinking, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="aime24", choices=("aime24", "aime25", "aime", "math500"))
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode")
    p.add_argument("--model", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    problems = load_eval_set(args.dataset)[: args.limit]
    out = Path(args.out) if args.out else ARTIFACTS_DIR / "audit" / f"{args.dataset}_longcot.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{stamp()}] {args.dataset}: {len(problems)} problems × {args.samples} "
          f"max_new={args.max_new_tokens} thinking={args.thinking}")
    model, tokenizer = load_model(model_id=args.model, dtype=args.dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id

    ntok, flags = [], []
    started = time.monotonic()
    with out.open("w") as fh:
        for i, q in enumerate(problems):
            prompt = build_think_prompt(tokenizer, q["problem"], args.thinking)
            for s in range(args.samples):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    gen = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos,
                    )
                new_ids = gen[0, inputs["input_ids"].shape[1] :].tolist()
                if tokenizer.pad_token_id is not None:
                    new_ids = [t for t in new_ids if t != tokenizer.pad_token_id]
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                truncated = len(new_ids) >= args.max_new_tokens
                rec = {
                    "problem_id": q["problem_id"],
                    "sample_id": s,
                    "n_tokens": len(new_ids),
                    "truncated": truncated,
                    "boxed": extract_boxed(text),
                    "correct": rollout_correct(text, q["gold"]),
                    "gold": q["gold"],
                    "text_head": text[:400],
                }
                ntok.append(len(new_ids))
                flags.append(rec)
                fh.write(json.dumps(rec) + "\n")
                print(
                    f"[{stamp()}] {i+1}/{len(problems)} s{s} {q['problem_id']} "
                    f"tok={len(new_ids)} trunc={truncated} ok={rec['correct']}",
                    flush=True,
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    arr = np.array(ntok, dtype=np.float64)
    print(
        json.dumps(
            {
                "n": len(arr),
                "mean": round(float(arr.mean()), 1),
                "p50": round(float(np.median(arr)), 1),
                "p90": round(float(np.quantile(arr, 0.9)), 1),
                "max": int(arr.max()),
                "trunc_rate": round(float(np.mean([r["truncated"] for r in flags])), 3),
                "acc": round(float(np.mean([r["correct"] for r in flags])), 3),
                "boxed_rate": round(float(np.mean([r["boxed"] is not None for r in flags])), 3),
                "seconds": round(time.monotonic() - started, 1),
                "out": str(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
