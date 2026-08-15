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
from .steps import Step, attach_token_offsets, extract_boxed, parse_steps


@dataclass
class Rollout:
    problem_id: str
    sample_id: int
    text: str
    n_tokens: int
    truncated: bool
    boxed: str | None
    correct: bool
    steps: list[dict]
    # layer_idx (HF 0-based) -> list of last-token vectors, one per step
    hidden: dict[str, list[list[float]]]


def _local_snapshot_ready(path) -> bool:
    p = Path(path)
    return p.is_dir() and any(p.glob("model-*.safetensors"))


def load_model(
    model_id: str | None = None,
    device_map: str = "auto",
    dtype: str = "auto",
    load_in_4bit: bool = False,
):
    """Load Qwen3-8B from a local snapshot if complete, otherwise the Hub.

    Colab T4 (16GB): use load_in_4bit=True.
    Colab L4/A100: dtype='bfloat16' or 'auto' is fine.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = model_id or (
        str(MODEL_DIR) if _local_snapshot_ready(MODEL_DIR) else MODEL_ID
    )
    local_only = _local_snapshot_ready(source)
    print(f"loading model from {source} (local_only={local_only}, 4bit={load_in_4bit})")

    tok_kwargs = {"local_files_only": local_only}
    tokenizer = AutoTokenizer.from_pretrained(source, **tok_kwargs)

    model_kwargs: dict = {
        "device_map": device_map,
        "local_files_only": local_only,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    model.eval()
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


@torch.inference_mode()
def generate_continuation(
    model,
    tokenizer,
    problem: str,
    prefix: str,
    gold: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Sample a fresh completion after an exact generated reasoning prefix.

    The returned token count covers only newly generated continuation tokens.
    Correctness is scored on prefix + continuation, which forms the full solution.
    """
    prompt = build_prompt(tokenizer, problem)
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"]
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = torch.cat((prompt_ids, prefix_ids), dim=1)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    input_len = input_ids.shape[1]

    eos_ids = (
        {tokenizer.eos_token_id}
        if isinstance(tokenizer.eos_token_id, int)
        else set(tokenizer.eos_token_id or [])
    )
    if prefix_ids.numel() and int(prefix_ids[0, -1]) in eos_ids:
        prefix_ids = prefix_ids[:, :-1]
        input_ids = torch.cat((prompt_ids, prefix_ids), dim=1).to(device)
        input_len = input_ids.shape[1]
    attention_mask = torch.cat(
        (prompt_inputs["attention_mask"], torch.ones_like(prefix_ids)), dim=1
    ).to(device)

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = gen[0, input_len:].tolist()
    continuation = tokenizer.decode(gen_ids, skip_special_tokens=True)
    full_text = prefix + continuation
    return {
        "text": continuation,
        "n_tokens": len(gen_ids),
        "truncated": not _finished_generation(
            gen_ids, max_new_tokens, tokenizer.eos_token_id
        ),
        "boxed": extract_boxed(full_text),
        "correct": rollout_correct(full_text, gold),
    }


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
    prompt = build_prompt(tokenizer, problem)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    gen = model.generate(
        **inputs,
        do_sample=True,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = gen[0, prompt_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    truncated = not _finished_generation(
        gen_ids, max_new_tokens, tokenizer.eos_token_id
    )

    steps = parse_steps(text)
    steps = attach_token_offsets(steps, text, gen_ids, tokenizer)
    hidden: dict[str, list[list[float]]] = {str(i): [] for i in PROBE_LAYER_INDICES}

    if capture_hidden and steps:
        full_ids = gen
        with hidden_state_hooks(model, PROBE_LAYER_INDICES) as cache:
            model(full_ids, use_cache=False)
        for step in steps:
            # Absolute index in the full sequence = prompt_len + offset
            abs_idx = prompt_len + (step.last_token_offset or 0)
            abs_idx = min(abs_idx, full_ids.shape[1] - 1)
            vecs = last_token_vectors(cache, abs_idx)
            for layer_idx, vec in vecs.items():
                hidden[str(layer_idx)].append(vec)

    remaining = []
    total = len(gen_ids)
    for step in steps:
        off = step.last_token_offset if step.last_token_offset is not None else 0
        remaining.append(max(0, total - off - 1))

    step_dicts = []
    for step, rem in zip(steps, remaining):
        d = asdict(step)
        d["tokens_remaining_this_trace"] = rem
        step_dicts.append(d)

    return Rollout(
        problem_id=problem_id,
        sample_id=sample_id,
        text=text,
        n_tokens=len(gen_ids),
        truncated=bool(truncated),
        boxed=extract_boxed(text),
        correct=rollout_correct(text, gold),
        steps=step_dicts,
        hidden=hidden,
    )
