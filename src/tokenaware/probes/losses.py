"""Losses for outcome-draw training.

Everything here consumes *realised draws*: one continuation, its 0/1
correctness, its length in tokens after the state, and whether the generator hit
the hard cap (right-censored).

Censoring matters. *How Much is Left?* drops truncated sequences, which biases
both V (survivorship) and T (the tail is exactly what a budget cares about).
The discretised likelihood keeps censored draws by scoring only the statement
"T fell in the last bin or beyond", which is all the data supports.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_mean(values: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if weight is None:
        return values.mean()
    total = weight.sum().clamp_min(1e-8)
    return (values * weight).sum() / total


def v_bce_loss(
    logits: torch.Tensor,
    correct: torch.Tensor,
    weight: torch.Tensor | None = None,
    pos_weight: float = 3.0,
) -> torch.Tensor:
    """BCE on single Bernoulli draws. ``pos_weight`` follows ReProbe (3.0)."""
    target = correct.float()
    per_sample = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype),
    )
    return weighted_mean(per_sample, weight)


def t_censored_nll(
    logits: torch.Tensor,
    bin_index: torch.Tensor,
    censored: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Discretised NLL; censored draws contribute ``-log P(T >= bin_index)``."""
    log_pmf = F.log_softmax(logits, dim=-1)
    n_bins = logits.shape[-1]
    idx = bin_index.clamp(0, n_bins - 1)
    exact = -log_pmf.gather(1, idx.unsqueeze(1)).squeeze(1)

    positions = torch.arange(n_bins, device=logits.device).unsqueeze(0)
    tail_mask = positions >= idx.unsqueeze(1)
    tail_log_prob = torch.logsumexp(
        log_pmf.masked_fill(~tail_mask, float("-inf")), dim=-1
    )
    censored_loss = -tail_log_prob

    is_censored = censored.bool()
    per_sample = torch.where(is_censored, censored_loss, exact)
    return weighted_mean(per_sample, weight)


def pinball_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...],
    weight: torch.Tensor | None = None,
    censored: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantile regression. Censored draws only penalise under-prediction."""
    q = torch.tensor(quantiles, device=predictions.device, dtype=predictions.dtype)
    error = target.unsqueeze(1) - predictions
    per_q = torch.maximum(q * error, (q - 1.0) * error)
    if censored is not None:
        keep = torch.where(
            censored.bool().unsqueeze(1), (error > 0).to(per_q.dtype), torch.ones_like(per_q)
        )
        per_q = per_q * keep
    return weighted_mean(per_q.mean(dim=1), weight)


def l1_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    censored: torch.Tensor | None = None,
) -> torch.Tensor:
    error = predictions - target
    per_sample = error.abs()
    if censored is not None:
        # Under-prediction of a censored draw is a real error; over-prediction is not.
        per_sample = torch.where(
            censored.bool() & (error > 0), torch.zeros_like(per_sample), per_sample
        )
    return weighted_mean(per_sample, weight)
