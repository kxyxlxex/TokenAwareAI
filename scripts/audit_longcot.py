#!/usr/bin/env python3
"""Length / pass@1 audit for a long-CoT reasoner on AIME (no hidden states).

This is the gate before rebuilding an MC corpus. We only need: mean/p50/p90
output tokens, truncation rate, pass@1. If mean length is not in the thousands,
AIME+this-model is the wrong scale.

  export CUDA_VISIBLE_DEVICES=MIG-...
  python scripts/audit_longcot.py --limit 30 --max-new-tokens 4096 --dtype bfloat16

Disk: text-only JSONL (megabytes). Do not pass --capture-hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from tokenaware.config import ARTIFACTS_DIR
from tokenaware.data import load_aime
from tokenaware.generate import load_model
from tokenaware.scoring import answers_equal
from tokenaware.steps import extract_boxed

R1_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def elapsed(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def build_r1_prompt(tokenizer, problem: str) -> str:
    """Native R1 template (think then answer). Not the short MATH CoT prompt."""
    messages = [{"role": "user", "content": problem}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False)


def summarize(records: list[dict]) -> dict:
    toks = np.array([r["n_tokens"] for r in records], dtype=np.float64)
    return {
        "n": len(records),
        "pass_at_1": round(float(np.mean([r["correct"] for r in records])), 4),
        "tokens_mean": round(float(toks.mean()), 1),
        "tokens_p50": round(float(np.median(toks)), 1),
        "tokens_p90": round(float(np.quantile(toks, 0.9)), 1),
        "tokens_max": int(toks.max()),
        "truncation_rate": round(float(np.mean([r["truncated"] for r in records])), 4),
        "boxed_rate": round(float(np.mean([r["boxed"] is not None for r in records])), 4),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=R1_ID)
    p.add_argument("--year", type=int, default=2024, choices=(2024, 2025))
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    problems = load_aime(args.year)[args.offset :]
    if args.limit:
        problems = problems[: args.limit]

    out_dir = Path(args.out) if args.out else ARTIFACTS_DIR / "audit_longcot"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"aime{args.year}_s0.jsonl"

    done = {}
    if dest.is_file():
        for line in dest.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["problem_id"]] = rec
    pending = [q for q in problems if q["problem_id"] not in done]
    print(
        f"[{stamp()}] AIME {args.year}: {len(problems)} listed, "
        f"{len(done)} cached, {len(pending)} to generate, cap={args.max_new_tokens}"
    )
    if not pending:
        recs = [done[q["problem_id"]] for q in problems if q["problem_id"] in done]
        print(json.dumps(summarize(recs), indent=2))
        return 0

    model, tokenizer = load_model(model_id=args.model, dtype=args.dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    started = time.monotonic()

    with dest.open("a") as fh:
        for i, q in enumerate(pending, start=1):
            prompt = build_r1_prompt(tokenizer, q["problem"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            t0 = time.monotonic()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen = out[0, inputs["input_ids"].shape[1] :]
            n = int(gen.shape[0])
            text = tokenizer.decode(gen, skip_special_tokens=True)
            boxed = extract_boxed(text)
            rec = {
                "problem_id": q["problem_id"],
                "gold": q["gold"],
                "n_tokens": n,
                "truncated": n >= args.max_new_tokens,
                "boxed": boxed,
                "correct": bool(answers_equal(boxed, q["gold"])),
                "seconds": round(time.monotonic() - t0, 1),
                "text_head": text[:400],
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[q["problem_id"]] = rec
            print(
                f"[{stamp()}] [{i}/{len(pending)}] {q['problem_id']} "
                f"tok={n} trunc={rec['truncated']} correct={rec['correct']} "
                f"in {elapsed(rec['seconds'])}",
                flush=True,
            )

    recs = [done[q["problem_id"]] for q in problems if q["problem_id"] in done]
    summary = summarize(recs)
    (out_dir / f"aime{args.year}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{stamp()}] finished in {elapsed(time.monotonic() - started)}")
    print(json.dumps(summary, indent=2))
    if summary["tokens_mean"] < 800:
        print(
            "WARNING: mean length is still MATH-scale. Do not build an MC corpus "
            "on this pair; pick a longer-thinking model or a harder set."
        )
    elif summary["truncation_rate"] > 0.3:
        print(
            "WARNING: >30% hit the cap; rerun with --max-new-tokens 8192 before "
            "trusting T labels."
        )
    else:
        print("Length scale looks usable for a rare-branch V+T search.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
