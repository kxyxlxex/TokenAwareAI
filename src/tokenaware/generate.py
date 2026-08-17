"""Non-thinking Qwen3 generation + step-boundary hidden-state cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .config import (
    MAX_NEW_TOKENS,
    MODEL_DIR,
    MODEL_ID,
    PROBE_LAYER_INDICES,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from .hooks import hidden_state_hooks, last_token_vectors
from .scoring import rollout_correct
from .steps import attach_token_offsets, extract_boxed, parse_steps


@dataclass
class Rollout:
    problem_id: str
    sample_id: int
    text: str
    gen_ids: list[int]
    n_tokens: int
    truncated: bool
    boxed: str | None
    correct: bool
    steps: list[dict]
    # HF layer index -> FP16 tensor [steps, hidden_width]
    hidden: dict[str, torch.Tensor]


def _local_snapshot_ready(path) -> bool:
    p = Path(path)
    return p.is_dir() and any(p.glob("model-*.safetensors"))


def load_model(
    model_id: str | None = None,
    device_map: str = "auto",
    dtype: str = "auto",
):
    """Load Qwen3-8B from a complete local snapshot or the Hub."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = model_id or (
        str(MODEL_DIR) if _local_snapshot_ready(MODEL_DIR) else MODEL_ID
    )
    local_only = _local_snapshot_ready(source)
    print(f"loading model from {source} (local_only={local_only}, dtype={dtype})")

    tok_kwargs = {"local_files_only": local_only}
    tokenizer = AutoTokenizer.from_pretrained(source, **tok_kwargs)

    model_kwargs: dict = {
        "device_map": device_map,
        "local_files_only": local_only,
        "attn_implementation": "sdpa",
    }
    if dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    model.eval()
    attn = getattr(model.config, "_attn_implementation", None)
    print(f"attn_implementation={attn}")
    return model, tokenizer


def build_prompt(tokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    # Qwen3: disable the <think> block.
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _finished_generation(
    gen_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | list[int] | None,
) -> bool:
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
    return len(gen_ids) < max_new_tokens or bool(gen_ids and gen_ids[-1] in eos_ids)


def _eos_ids(tokenizer) -> set[int]:
    eos = tokenizer.eos_token_id
    return {eos} if isinstance(eos, int) else set(eos or [])


def _trim_generated_ids(ids: list[int], tokenizer) -> list[int]:
    """Drop right-padding and keep EOS when present."""
    eos_ids = _eos_ids(tokenizer)
    pad_id = tokenizer.pad_token_id
    for i, token_id in enumerate(ids):
        if token_id in eos_ids:
            return ids[: i + 1]
        if pad_id is not None and token_id == pad_id:
            return ids[:i]
    return ids


def _prepare_tokenizer_for_batch(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def _rollout_from_generated(
    model,
    tokenizer,
    *,
    gen_ids: list[int],
    problem_id: str,
    sample_id: int,
    gold: str,
    max_new_tokens: int,
) -> Rollout:
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    steps = attach_token_offsets(parse_steps(text), text, gen_ids, tokenizer)
    hidden = {
        str(layer_idx): torch.empty(
            (0, model.config.hidden_size), dtype=torch.float16
        )
        for layer_idx in PROBE_LAYER_INDICES
    }

    total = len(gen_ids)
    step_dicts = []
    for step in steps:
        offset = step.last_token_offset if step.last_token_offset is not None else 0
        data = asdict(step)
        data["tokens_remaining_this_trace"] = max(0, total - offset - 1)
        step_dicts.append(data)

    return Rollout(
        problem_id=problem_id,
        sample_id=sample_id,
        text=text,
        gen_ids=gen_ids,
        n_tokens=total,
        truncated=not _finished_generation(
            gen_ids, max_new_tokens, tokenizer.eos_token_id
        ),
        boxed=extract_boxed(text),
        correct=rollout_correct(text, gold),
        steps=step_dicts,
        hidden=hidden,
    )


def _capture_batch_hidden(
    model,
    tokenizer,
    rollouts: list[Rollout],
    prompt_ids: list[torch.Tensor],
) -> None:
    """Replay completed sequences once and attach step-boundary FP16 tensors."""
    sequences = [
        torch.cat(
            (
                prompt.squeeze(0),
                torch.tensor(
                    rollout.gen_ids, dtype=prompt.dtype, device=prompt.device
                ),
            )
        )
        for rollout, prompt in zip(rollouts, prompt_ids)
    ]
    max_length = max(sequence.shape[0] for sequence in sequences)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full(
        (len(sequences), max_length),
        pad_id,
        dtype=sequences[0].dtype,
        device=sequences[0].device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for batch_index, sequence in enumerate(sequences):
        input_ids[batch_index, : sequence.shape[0]] = sequence
        attention_mask[batch_index, : sequence.shape[0]] = 1

    with hidden_state_hooks(model, PROBE_LAYER_INDICES) as cache:
        model(input_ids, attention_mask=attention_mask, use_cache=False)
        cpu_cache = {
            idx: t.to(dtype=torch.float16, device="cpu") for idx, t in cache.items()
        }

    for batch_index, (rollout, prompt) in enumerate(zip(rollouts, prompt_ids)):
        layer_vectors: dict[str, list[torch.Tensor]] = {
            str(index): [] for index in PROBE_LAYER_INDICES
        }
        for step in rollout.steps:
            offset = step.get("last_token_offset")
            if offset is None:
                continue
            absolute_index = prompt.shape[1] + offset
            for layer_index, vector in last_token_vectors(
                cpu_cache, absolute_index, batch=batch_index
            ).items():
                layer_vectors[str(layer_index)].append(vector)
        rollout.hidden = {
            layer_index: (
                torch.stack(vectors)
                if vectors
                else torch.empty(
                    (0, model.config.hidden_size), dtype=torch.float16
                )
            )
            for layer_index, vectors in layer_vectors.items()
        }


@torch.inference_mode()
def generate_rollouts_batch(
    model,
    tokenizer,
    requests: list[dict],
    max_new_tokens: int = MAX_NEW_TOKENS,
    capture_hidden: bool = True,
) -> list[Rollout]:
    """Generate heterogeneous root rollouts in one padded model call."""
    if not requests:
        return []
    _prepare_tokenizer_for_batch(tokenizer)
    prompts = [build_prompt(tokenizer, request["problem"]) for request in requests]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]

    generated = model.generate(
        **inputs,
        do_sample=True,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    results = []
    prompt_rows = []
    for batch_index, request in enumerate(requests):
        raw_ids = generated[batch_index, input_width:].tolist()
        gen_ids = _trim_generated_ids(raw_ids, tokenizer)
        prompt_ids = inputs["input_ids"][batch_index][
            inputs["attention_mask"][batch_index].bool()
        ].unsqueeze(0)
        prompt_rows.append(prompt_ids)
        results.append(
            _rollout_from_generated(
                model,
                tokenizer,
                gen_ids=gen_ids,
                problem_id=request["problem_id"],
                sample_id=request.get("sample_id", batch_index),
                gold=request["gold"],
                max_new_tokens=max_new_tokens,
            )
        )
    if capture_hidden and any(rollout.steps for rollout in results):
        _capture_batch_hidden(model, tokenizer, results, prompt_rows)
    return results


@torch.inference_mode()
def generate_continuations_batch(
    model,
    tokenizer,
    requests: list[dict],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[dict]:
    """Generate heterogeneous prefix continuations in one padded model call."""
    if not requests:
        return []
    _prepare_tokenizer_for_batch(tokenizer)
    eos_ids = _eos_ids(tokenizer)
    sequences = []
    for request in requests:
        prompt_ids = tokenizer(build_prompt(tokenizer, request["problem"]))["input_ids"]
        prefix_ids = request.get("prefix_token_ids")
        if prefix_ids is None:
            prefix_ids = tokenizer(
                request["prefix"], add_special_tokens=False
            )["input_ids"]
        if prefix_ids and prefix_ids[-1] in eos_ids:
            prefix_ids = prefix_ids[:-1]
        sequences.append(prompt_ids + prefix_ids)
    inputs = tokenizer.pad(
        {
            "input_ids": sequences,
            "attention_mask": [[1] * len(sequence) for sequence in sequences],
        },
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]

    generated = model.generate(
        **inputs,
        do_sample=True,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    results = []
    for batch_index, request in enumerate(requests):
        raw_ids = generated[batch_index, input_width:].tolist()
        gen_ids = _trim_generated_ids(raw_ids, tokenizer)
        continuation = tokenizer.decode(gen_ids, skip_special_tokens=True)
        full_text = request["prefix"] + continuation
        results.append(
            {
                "text": continuation,
                "n_tokens": len(gen_ids),
                "truncated": not _finished_generation(
                    gen_ids, max_new_tokens, tokenizer.eos_token_id
                ),
                "boxed": extract_boxed(full_text),
                "correct": rollout_correct(full_text, request["gold"]),
            }
        )
    return results


@torch.inference_mode()
def generate_rollout(
    model,
    tokenizer,
    problem: str,
    gold: str,
    problem_id: str,
    sample_id: int = 0,
    max_new_tokens: int = MAX_NEW_TOKENS,
    capture_hidden: bool = True,
) -> Rollout:
    return generate_rollouts_batch(
        model,
        tokenizer,
        [
            {
                "problem": problem,
                "gold": gold,
                "problem_id": problem_id,
                "sample_id": sample_id,
            }
        ],
        max_new_tokens=max_new_tokens,
        capture_hidden=capture_hidden,
    )[0]
