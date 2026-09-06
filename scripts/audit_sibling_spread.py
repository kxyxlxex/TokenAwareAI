#!/usr/bin/env python3
"""Sibling remaining-length spread under thinking CoT. No hidden states.

From a shared prefix, sample two independent completions and measure |T1 − T2|.
If that gap is usually smaller than one discarded sibling, V+T still cannot
beat a chain. High truncation on *both* completions censors the gap toward 0 —
raise --complete-tokens rather than calling that a kill.

  python scripts/audit_sibling_spread.py --dataset aime24 --limit 6 \
      --parent-tokens 1024 --complete-tokens 8192 --thinking --dtype bfloat16
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

from tokenaware.config import ARTIFACTS_DIR
from tokenaware.data import load_eval_set
from tokenaware.generate import load_model
from tokenaware.scoring import rollout_correct
from tokenaware.steps import extract_boxed

THINK_PROMPT = (
    "Solve the problem. Put the final answer after the reasoning, in \\boxed{}."
)


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


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def generate_ids(model, tokenizer, input_ids, max_new_tokens: int) -> list[int]:
    device = input_ids.device
    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = gen[0, input_ids.shape[1] :].tolist()
    pad = tokenizer.pad_token_id
    if pad is not None:
        new_ids = [t for t in new_ids if t != pad]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return new_ids


def decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="aime24", choices=("aime24", "aime25", "aime", "math500"))
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--width", type=int, default=2)
    p.add_argument("--parent-tokens", type=int, default=1024)
    p.add_argument("--complete-tokens", type=int, default=8192)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    problems = load_eval_set(args.dataset)[: args.limit]
    out = (
        Path(args.out)
        if args.out
        else ARTIFACTS_DIR / "audit" / f"{args.dataset}_sibling_spread.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["problem_id"] for line in out.open() if line.strip()} if out.exists() else set()

    print(
        f"[{stamp()}] {args.dataset}: {len(problems)} problems width={args.width} "
        f"parent={args.parent_tokens} complete={args.complete_tokens} thinking={args.thinking}",
        flush=True,
    )
    model, tokenizer = load_model(model_id=args.model, dtype=args.dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device

    gaps: list[float] = []
    started = time.monotonic()
    mode = "a" if done else "w"
    with out.open(mode) as fh:
        for i, q in enumerate(problems):
            if q["problem_id"] in done:
                print(f"[{stamp()}] skip {q['problem_id']}", flush=True)
                continue
            prompt = build_think_prompt(tokenizer, q["problem"], args.thinking)
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            parent_ids = generate_ids(model, tokenizer, prompt_ids, args.parent_tokens)
            parent_text = decode(tokenizer, parent_ids)
            if extract_boxed(parent_text) is not None or len(parent_ids) < args.parent_tokens:
                rec = {
                    "problem_id": q["problem_id"],
                    "skipped": True,
                    "reason": "parent_finished",
                    "parent_tokens": len(parent_ids),
                    "correct": rollout_correct(parent_text, q["gold"]),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(
                    f"[{stamp()}] {i+1}/{len(problems)} {q['problem_id']} parent finished tok={len(parent_ids)}",
                    flush=True,
                )
                continue

            prefix = torch.cat([prompt_ids, torch.tensor(parent_ids, device=device).unsqueeze(0)], dim=1)
            siblings = []
            for b in range(args.width):
                cont = generate_ids(model, tokenizer, prefix, args.complete_tokens)
                text = parent_text + decode(tokenizer, cont)
                sib = {
                    "branch": b,
                    "t": len(cont),
                    "truncated": len(cont) >= args.complete_tokens,
                    "boxed": extract_boxed(text),
                    "correct": rollout_correct(text, q["gold"]),
                }
                siblings.append(sib)
                print(
                    f"[{stamp()}] {i+1}/{len(problems)} {q['problem_id']} b{b} "
                    f"T={sib['t']} trunc={sib['truncated']} ok={sib['correct']}",
                    flush=True,
                )

            ts = [s["t"] for s in siblings]
            gap = float(max(ts) - min(ts)) if len(ts) >= 2 else 0.0
            gaps.append(gap)
            rec = {
                "problem_id": q["problem_id"],
                "skipped": False,
                "parent_tokens": len(parent_ids),
                "siblings": siblings,
                "gap": gap,
                "gold": q["gold"],
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    gaps = []
    n_trunc_both = 0
    for line in out.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("skipped") or "gap" not in rec:
            continue
        gaps.append(float(rec["gap"]))
        sibs = rec.get("siblings") or []
        if sibs and all(s.get("truncated") for s in sibs):
            n_trunc_both += 1

    arr = np.array(gaps, dtype=np.float64) if gaps else np.array([0.0])
    print(
        json.dumps(
            {
                "n_pairs": len(gaps),
                "gap_mean": round(float(arr.mean()), 1),
                "gap_p50": round(float(np.median(arr)), 1),
                "gap_p90": round(float(np.quantile(arr, 0.9)), 1),
                "frac_gap_gt_256": round(float(np.mean(arr > 256)), 3),
                "frac_gap_gt_1024": round(float(np.mean(arr > 1024)), 3),
                "n_both_truncated": n_trunc_both,
                "seconds": round(time.monotonic() - started, 1),
                "out": str(out),
                "go": bool(len(gaps) >= 4 and float(np.mean(arr > 256)) >= 0.4),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
