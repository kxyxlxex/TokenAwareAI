"""Capture residual-stream hidden states at chosen layers / token positions."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .config import PROBE_LAYER_INDICES


@contextmanager
def hidden_state_hooks(model, layer_indices: tuple[int, ...] = PROBE_LAYER_INDICES):
    """Register forward hooks on `model.model.layers[i]`.

    After a forward / generate call, `cache[layer_idx]` is a tensor
    [batch, seq, hidden] for that layer's output (post-block).
    """
    cache: dict[int, torch.Tensor] = {}
    handles = []

    def _make(idx: int):
        def hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            cache[idx] = hidden.detach()

        return hook

    layers = model.model.layers
    for idx in layer_indices:
        handles.append(layers[idx].register_forward_hook(_make(idx)))
    try:
        yield cache
    finally:
        for h in handles:
            h.remove()


def last_token_vectors(
    cache: dict[int, torch.Tensor],
    token_index: int,
    batch: int = 0,
) -> dict[int, torch.Tensor]:
    """Read h[batch, token_index] at each hooked layer as CPU FP16 tensors."""
    out: dict[int, torch.Tensor] = {}
    for idx, t in cache.items():
        out[idx] = t[batch, token_index].to(dtype=torch.float16, device="cpu")
    return out
