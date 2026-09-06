#!/usr/bin/env python3
"""Length / accuracy audit for a long-CoT reasoner. No probes, no tree.

Decides whether the next corpus should be built on this π. We need mean
generation length in the thousands, a truncation rate that is not ~100%,
and pass@1 clearly below ceiling (so V has something to do).

  export CUDA_VISIBLE_DEVICES=MIG-...
  python scripts/audit_longcot.py --dataset aime24 --limit 30 --dtype bfloat16
  python scripts/audit_longcot.py --dataset math500 --levels 5 --limit 30 --dtype bfloat16

Default model: DeepSeek-R1-Distill-Qwen-7B (STAR's remaining-length backbone).
Cap 8192; R1 eval often uses 32k — raise --max-new-tokens if p90 hugs the cap.
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
from tokenaware.data import load_aime24, load_math500
from tokenaware.generate import build_r1_prompt, load_model
from tokenaware.scoring import rollout_correct
from tokenaware.steps import extract_boxed

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# DeepSeek R1-distill eval defaults (not Qwen3 non-thinking 0.7 / top-k 20).
R1_TEMPERATURE = 0.6
R1_TOP_P = 0.95


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def elapsed(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def summarize(records: list[dict]) -> dict:
    n = len(records)
    toks = np.array([r["n_tokens"] for r in records], dtype=np.float64)
    return {
        "n": n,
        "pass_at_1": round(float(np.mean([r["correct"] for r in records])), 4) if n else 0.0,
        "boxed_rate": round(float(np.mean([r["boxed"] is not None for r in records])), 4) if n else 0.0,
        "truncation_rate": round(float(np.mean([r["truncated"] for r in records])), 4) if n else 0.0,
        "tokens_mean": round(float(toks.mean()), 1) if n else 0.0,
        "tokens_p50": round(float(np.median(toks)), 1) if n else 0.0,
        "tokens_p90": round(float(np.quantile(toks, 0.9)), 1) if n else 0.0,
        "tokens_max": int(toks.max()) if n else 0,
    }


def go_nogo(stats: dict, cap: int) -> str:
    """One-line recommendation. Thresholds are gates, not paper claims."""
    if stats["n"] < 10:
        return "TOO_FEW: run at least ~20 problems"
    if stats["tokens_mean"] < 800:
        return "WRONG_SCALE: mean < 800 tokens — this is still short-CoT; pick a thinkier π or harder set"
    if stats["truncation_rate"] > 0.5:
        return "CAP_TOO_LOW: >50% hit max_new_tokens — raise the cap and rerun before building labels"
    if stats["pass_at_1"] > 0.92:
        return "V_SATURATED: pass@1 too high for search to matter"
    if stats["pass_at_1"] < 0.05 and stats["boxed_rate"] < 0.2:
        return "BROKEN: almost no boxed answers — prompt/template/scoring, not the task"
    return "GO: length scale looks right; next is a 200-problem root corpus + MC prefixes"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dataset", choices=("aime24", "math500"), default="aime24")
    p.add_argument("--levels", type=int, nargs="+", default=None)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=R1_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=R1_TOP_P)
    p.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    problems = load_aime24() if args.dataset == "aime24" else load_math500()
    if args.levels:
        problems = [q for q in problems if q.get("level") in set(args.levels)]
    problems = problems[args.offset :]
    if args.limit:
        problems = problems[: args.limit]
    if not problems:
        print("no problems", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else ARTIFACTS_DIR / "longcot" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{args.dataset}_n{len(problems)}.jsonl"
    done_ids = set()
    records = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                records.append(rec)
                done_ids.add(rec["problem_id"])
    pending = [q for q in problems if q["problem_id"] not in done_ids]
    print(
        f"[{stamp()}] {args.dataset}: {len(problems)} problems, "
        f"{len(done_ids)} already done, {len(pending)} pending, "
        f"model={args.model} cap={args.max_new_tokens}"
    )

    if pending:
        model, tokenizer = load_model(model_id=args.model, dtype=args.dtype)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        device = next(model.parameters()).device
        started = time.monotonic()
        with jsonl_path.open("a") as fh:
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
                # Drop trailing pad
                ids = gen.tolist()
                pad = tokenizer.pad_token_id
                eos = tokenizer.eos_token_id
                eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
                cut = []
                finished = False
                for t in ids:
                    if t in eos_ids:
                        cut.append(t)
                        finished = True
                        break
                    if pad is not None and t == pad:
                        break
                    cut.append(t)
                text = tokenizer.decode(cut, skip_special_tokens=True)
                rec = {
                    "problem_id": q["problem_id"],
                    "level": q.get("level"),
                    "gold": q["gold"],
                    "n_tokens": len(cut),
                    "truncated": (not finished) and len(cut) >= args.max_new_tokens,
                    "boxed": extract_boxed(text),
                    "correct": rollout_correct(text, q["gold"]),
                    "seconds": round(time.monotonic() - t0, 1),
                    "text": text,
                    "model": args.model,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                }
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(
                    f"[{stamp()}] [{i}/{len(pending)}] {q['problem_id']} "
                    f"tok={rec['n_tokens']} boxed={rec['boxed']!r} "
                    f"ok={rec['correct']} trunc={rec['truncated']} "
                    f"in {elapsed(rec['seconds'])}",
                    flush=True,
                )
        print(f"[{stamp()}] generation {elapsed(time.monotonic() - started)}")

    stats = summarize(records)
    stats["model"] = args.model
    stats["dataset"] = args.dataset
    stats["max_new_tokens"] = args.max_new_tokens
    stats["verdict"] = go_nogo(stats, args.max_new_tokens)
    (out_dir / f"{args.dataset}_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("verdict:", stats["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
