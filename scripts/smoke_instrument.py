#!/usr/bin/env python3
"""Step 1 smoke test: one MATH problem, generate, parse steps, cache 4-layer h."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.config import ARTIFACTS_DIR, PROBE_LAYER_INDICES
from tokenaware.data import load_split
from tokenaware.generate import generate_rollout, load_model


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="HF id or local dir. Default: Hub Qwen/Qwen3-8B")
    p.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16"))
    args = p.parse_args()

    split_path = ARTIFACTS_DIR / "splits" / "math_probe_split.json"
    if not split_path.exists():
        raise SystemExit(f"Run scripts/make_splits.py first (missing {split_path})")
    problem = load_split(split_path)["train"][0]
    print(f"problem_id={problem['problem_id']} level={problem['level']} {problem['subject']}")
    print(problem["problem"][:200], "...")

    model, tokenizer = load_model(model_id=args.model, dtype=args.dtype)
    rollout = generate_rollout(
        model,
        tokenizer,
        problem=problem["problem"],
        gold=problem["gold"],
        problem_id=problem["problem_id"],
        sample_id=0,
    )
    print(f"tokens={rollout.n_tokens} truncated={rollout.truncated} correct={rollout.correct}")
    print(f"boxed={rollout.boxed!r} gold={problem['gold']!r}")
    print(f"n_steps={len(rollout.steps)}")
    for s in rollout.steps[:8]:
        print(f"  [{s['index']}] rem={s['tokens_remaining_this_trace']} {s['text'][:80]}")
    for li in PROBE_LAYER_INDICES:
        vecs = rollout.hidden.get(str(li), [])
        dim = len(vecs[0]) if vecs else 0
        print(f"  layer {li + 1} (hf {li}): {len(vecs)} step vectors, dim={dim}")

    out = ARTIFACTS_DIR / "smoke" / "one_rollout.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "problem_id": rollout.problem_id,
        "text": rollout.text,
        "n_tokens": rollout.n_tokens,
        "truncated": rollout.truncated,
        "boxed": rollout.boxed,
        "correct": rollout.correct,
        "steps": [{k: v for k, v in s.items() if k != "hidden"} for s in rollout.steps],
        "hidden_shapes": {k: [len(v), len(v[0]) if v else 0] for k, v in rollout.hidden.items()},
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
