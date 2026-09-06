#!/usr/bin/env python3
"""Thinking-mode sibling MC labels for a new V/T probe. Fits a 20 GiB MIG.

Same frozen Qwen3-8B. Native ``<think>`` (no Step-N prompt). One sequence at a
time. Hidden states: HF layer 26 last token of (parent + peek) only — not the
8k completion, not four layers.

  # overnight-ish pilot (~20 L5 problems). Do not download another model.
  python scripts/generate_thinking_siblings.py --limit 20 --k 2 --width 2 \
      --parent-tokens 1024 --peek-tokens 256 --complete-tokens 4096 \
      --dtype bfloat16

Disk: JSONL + hidden_L26.npy (20 problems × 2 siblings × 8 KiB ≈ 0.3 MiB).
Resume by problem_id. Completions are label-only (no hidden, no full text).
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
from tokenaware.data import load_math_train, load_split
from tokenaware.generate import (
    build_think_prompt,
    capture_state_vectors,
    load_model,
)
from tokenaware.scoring import rollout_correct
from tokenaware.steps import extract_boxed


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def generate_ids(model, tokenizer, input_ids: torch.Tensor, max_new: int) -> list[int]:
    """One sequence. Thinking sampling (T=0.6) as in the AIME audit."""
    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_ids = gen[0, input_ids.shape[1] :].tolist()
    pad = tokenizer.pad_token_id
    if pad is not None:
        new_ids = [t for t in new_ids if t != pad]
    del gen
    if input_ids.device.type == "cuda":
        torch.cuda.empty_cache()
    return new_ids


def load_pool(artifacts: Path) -> list[dict]:
    split_path = artifacts / "splits" / "math_probe_split.json"
    if split_path.is_file():
        return load_split(split_path)["train"]
    return load_math_train()


def parse_level_counts(specs: list[str]) -> dict[int, int]:
    """``['5:100', '4:60']`` → ``{5: 100, 4: 60}``."""
    out: dict[int, int] = {}
    for spec in specs:
        level_s, n_s = spec.split(":", 1)
        out[int(level_s)] = int(n_s)
    return out


def load_problems(
    artifacts: Path,
    levels: list[int],
    limit: int,
    level_counts: dict[int, int] | None = None,
) -> list[dict]:
    rows = load_pool(artifacts)
    if level_counts:
        by_level: dict[int, list[dict]] = {}
        for row in rows:
            by_level.setdefault(int(row.get("level") or 0), []).append(row)
        picked: list[dict] = []
        for level, n in sorted(level_counts.items(), reverse=True):
            bucket = by_level.get(level) or []
            if len(bucket) < n:
                print(
                    f"warning: only {len(bucket)} train problems at level {level}, want {n}",
                    flush=True,
                )
            picked.extend(bucket[:n])
        return picked
    rows = [r for r in rows if int(r.get("level") or 0) in set(levels)]
    return rows[:limit]


def append_hidden(path: Path, vector: np.ndarray) -> int:
    vec = np.asarray(vector, dtype=np.float16).reshape(1, -1)
    if path.exists():
        old = np.load(path)
        row = int(old.shape[0])
        np.save(path, np.concatenate([old, vec], axis=0))
        return row
    np.save(path, vec)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--levels", type=int, nargs="+", default=[5])
    p.add_argument(
        "--level-counts",
        nargs="+",
        default=None,
        metavar="L:N",
        help="quota per MATH level, e.g. 5:100 4:60. Overrides --limit/--levels.",
    )
    p.add_argument("--k", type=int, default=2, help="MC completions per sibling")
    p.add_argument("--width", type=int, default=2)
    p.add_argument("--parent-tokens", type=int, default=1024)
    p.add_argument("--peek-tokens", type=int, default=256)
    p.add_argument("--complete-tokens", type=int, default=4096)
    p.add_argument("--layer", type=int, default=26, help="HF 0-index residual layer")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else ARTIFACTS_DIR / "thinking" / "siblings"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "states.jsonl"
    hidden_path = out_dir / f"hidden_L{args.layer}.npy"
    done = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            rec = json.loads(line)
            done.add(rec["problem_id"])

    counts = parse_level_counts(args.level_counts) if args.level_counts else None
    problems = load_problems(ARTIFACTS_DIR, args.levels, args.limit, counts)
    pending = [q for q in problems if q["problem_id"] not in done]
    print(
        f"[{stamp()}] thinking siblings: {len(pending)}/{len(problems)} pending "
        f"quotas={counts or args.levels} width={args.width} k={args.k} "
        f"parent={args.parent_tokens} peek={args.peek_tokens} "
        f"complete={args.complete_tokens} layer={args.layer}",
        flush=True,
    )
    if not pending:
        print("every problem already labelled")
        return 0

    model, tokenizer = load_model(model_id=args.model, dtype=args.dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device

    started = time.monotonic()
    with jsonl_path.open("a") as fh:
        for i, q in enumerate(pending):
            prompt = build_think_prompt(tokenizer, q["problem"], thinking=True)
            prompt_t = tokenizer(prompt, return_tensors="pt").to(device)
            parent_ids = generate_ids(model, tokenizer, prompt_t["input_ids"], args.parent_tokens)
            parent_text = tokenizer.decode(parent_ids, skip_special_tokens=True)
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
                    f"[{stamp()}] {i+1}/{len(pending)} {q['problem_id']} skip parent tok={len(parent_ids)}",
                    flush=True,
                )
                continue

            prefix = torch.cat(
                [
                    prompt_t["input_ids"],
                    torch.tensor(parent_ids, device=device).unsqueeze(0),
                ],
                dim=1,
            )
            group_id = f"{q['problem_id']}:p{args.parent_tokens}"
            for b in range(args.width):
                peek_ids = generate_ids(model, tokenizer, prefix, args.peek_tokens)
                state_ids = parent_ids + peek_ids
                full_ids = prompt_t["input_ids"][0].tolist() + state_ids
                hid = capture_state_vectors(
                    model,
                    tokenizer,
                    [{"full_ids": full_ids}],
                    layers=(args.layer,),
                )[0]
                vec = hid[str(args.layer)].reshape(-1).numpy()
                hidden_row = append_hidden(hidden_path, vec)

                state = torch.cat(
                    [prefix, torch.tensor(peek_ids, device=device).unsqueeze(0)], dim=1
                )
                draws = []
                for _ in range(args.k):
                    cont = generate_ids(model, tokenizer, state, args.complete_tokens)
                    text = parent_text + tokenizer.decode(
                        peek_ids + cont, skip_special_tokens=True
                    )
                    draws.append(
                        {
                            "n_tokens": len(cont),
                            "truncated": len(cont) >= args.complete_tokens,
                            "correct": rollout_correct(text, q["gold"]),
                        }
                    )
                n_ok = sum(1 for d in draws if d["correct"])
                rec = {
                    "problem_id": q["problem_id"],
                    "skipped": False,
                    "level": int(q["level"]),
                    "subject": q.get("subject", "algebra"),
                    "group_id": group_id,
                    "branch": b,
                    "parent_tokens": len(parent_ids),
                    "peek_tokens": len(peek_ids),
                    "tokens_so_far": len(state_ids),
                    "problem_chars": len(q["problem"]),
                    "hidden_row": hidden_row,
                    "layer": args.layer,
                    "v_mc": n_ok / len(draws),
                    "t_mc_mean": float(np.mean([d["n_tokens"] for d in draws])),
                    "mc_k": len(draws),
                    "n_correct": n_ok,
                    "n_truncated": sum(1 for d in draws if d["truncated"]),
                    "continuations": draws,
                    "gold": q["gold"],
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(
                    f"[{stamp()}] {i+1}/{len(pending)} {q['problem_id']} b{b} "
                    f"T̄={rec['t_mc_mean']:.0f} V={rec['v_mc']:.2f} "
                    f"trunc={rec['n_truncated']}/{args.k}",
                    flush=True,
                )
            del prefix
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "seconds": round(time.monotonic() - started, 1),
                "jsonl": str(jsonl_path),
                "hidden": str(hidden_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
