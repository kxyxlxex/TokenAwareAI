"""Probe architectures for V and T.

Trunks
  ``linear``  no trunk; heads read the input directly (the *How Much is Left?*
              linear-probe diagnostic: is the signal even linear?)
  ``star``    4-layer MLP 2048-512-64 from STAR / ARES, ~9.5M params at d=4096
  ``mlp``     lighter 512-512 trunk
  ``attn``    ReProbe-style: one transformer encoder layer (hidden 512, 16
              heads) over the last W step vectors, then attention-pool

Heads
  ``VHead``       one logit; trained with BCE on realised 0/1 outcomes
  ``TDist``       logits over log-spaced remaining-token bins (the primary T
                  head). Gives ``P(T <= B)`` exactly, which is what the
                  cost-aware score needs, and handles right-censored traces
                  through a censored likelihood.
  ``TQuantile``   monotone quantile outputs with pinball loss
  ``TPoint``      single scalar, L1 in raw or log space (STAR's own recipe)
  ``JointVT``     shared trunk, V logit + T distribution
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


# --------------------------------------------------------------------------- #
# bins
# --------------------------------------------------------------------------- #
def make_bin_edges(max_tokens: int = 1024, n_bins: int = 32) -> torch.Tensor:
    """Log-spaced upper edges over ``[0, max_tokens]`` plus one overflow bin.

    Bin ``i`` covers ``(edges[i-1], edges[i]]``. The ``n_bins - 1`` finite edges
    end exactly at ``max_tokens``, so the finite bins are precisely "terminated
    within the generation cap" and the final infinite bin is precisely "did not".
    ``P(T <= B)`` for ``B >= max_tokens`` therefore tops out at the probability of
    terminating at all, which is the honest answer: a trace that never finished
    did not finish inside any budget.
    """
    if n_bins < 3:
        raise ValueError("n_bins must be at least 3 (two finite bins plus overflow)")
    lo = math.log1p(0.0)
    hi = math.log1p(float(max_tokens))
    finite = torch.expm1(torch.linspace(lo, hi, n_bins - 1))
    finite[-1] = float(max_tokens)
    return torch.cat([finite, torch.tensor([float("inf")])])


def bin_of(length: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return torch.searchsorted(edges.contiguous(), length.float().contiguous())


def bin_centers(edges: torch.Tensor, max_tokens: int = 1024) -> torch.Tensor:
    finite = edges.clone()
    finite[-1] = float(max_tokens) * 1.5
    lower = torch.cat([torch.zeros(1, device=edges.device), finite[:-1]])
    return 0.5 * (lower + finite)


# --------------------------------------------------------------------------- #
# trunks
# --------------------------------------------------------------------------- #
class LinearTrunk(nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.d_out = d_in

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        return x


class MLPTrunk(nn.Module):
    def __init__(
        self,
        d_in: int,
        widths: tuple[int, ...] = (2048, 512, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for width in widths:
            layers += [nn.Linear(prev, width), nn.GELU(), nn.Dropout(dropout)]
            prev = width
        self.net = nn.Sequential(*layers)
        self.d_out = prev

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        return self.net(x)


class AttnTrunk(nn.Module):
    """One transformer encoder layer over the step-vector history."""

    def __init__(
        self,
        d_in: int,
        hidden: int = 512,
        heads: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(d_in, hidden)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.query = nn.Parameter(torch.randn(hidden) * 0.02)
        self.d_out = hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: [n, W, d_in]; mask True where padded.
        if x.dim() == 2:
            x = x.unsqueeze(1)
            mask = None
        h = self.project(x)
        h = self.encoder(h, src_key_padding_mask=mask)
        scores = (h * self.query).sum(dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (h * weights).sum(dim=1)


def build_trunk(name: str, d_in: int, dropout: float = 0.1, **kwargs) -> nn.Module:
    if name == "linear":
        return LinearTrunk(d_in)
    if name == "star":
        return MLPTrunk(d_in, widths=(2048, 512, 64), dropout=dropout)
    if name == "mlp":
        return MLPTrunk(d_in, widths=(512, 512), dropout=dropout)
    if name == "attn":
        return AttnTrunk(d_in, dropout=dropout, **kwargs)
    raise ValueError(f"unknown trunk {name!r}")


TRUNK_NAMES = ("linear", "star", "mlp", "attn")


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
@dataclass
class ProbeConfig:
    task: str = "joint"  # v | t | joint
    trunk: str = "star"
    d_in: int = 4096
    dropout: float = 0.1
    t_head: str = "dist"  # dist | quantile | point
    n_bins: int = 32
    max_tokens: int = 1024
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    log_target: bool = True

    def to_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["quantiles"] = list(self.quantiles)
        return payload

    @staticmethod
    def from_dict(payload: dict) -> "ProbeConfig":
        payload = dict(payload)
        payload["quantiles"] = tuple(payload.get("quantiles", DEFAULT_QUANTILES))
        return ProbeConfig(**payload)


class Probe(nn.Module):
    """One trunk with a V logit head and/or a T head."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = build_trunk(config.trunk, config.d_in, dropout=config.dropout)
        d = self.trunk.d_out
        self.v_head = nn.Linear(d, 1) if config.task in ("v", "joint") else None
        self.t_head: nn.Module | None = None
        if config.task in ("t", "joint"):
            if config.t_head == "dist":
                self.t_head = nn.Linear(d, config.n_bins)
            elif config.t_head == "quantile":
                self.t_head = nn.Linear(d, len(config.quantiles))
            elif config.t_head == "point":
                self.t_head = nn.Linear(d, 1)
            else:
                raise ValueError(f"unknown t_head {config.t_head!r}")
        self.register_buffer(
            "bin_edges", make_bin_edges(config.max_tokens, config.n_bins)
        )
        self.register_buffer(
            "bin_centers_buf", bin_centers(self.bin_edges, config.max_tokens)
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        h = self.trunk(x, mask)
        out: dict[str, torch.Tensor] = {}
        if self.v_head is not None:
            out["v_logit"] = self.v_head(h).squeeze(-1)
        if self.t_head is not None:
            raw = self.t_head(h)
            if self.config.t_head == "dist":
                out["t_logits"] = raw
            elif self.config.t_head == "quantile":
                # Monotone in q: first output free, rest positive increments.
                first = raw[:, :1]
                increments = F.softplus(raw[:, 1:])
                out["t_quantiles"] = torch.cat(
                    [first, first + torch.cumsum(increments, dim=1)], dim=1
                )
            else:
                out["t_point"] = raw.squeeze(-1)
        return out

    # -- inference-time conveniences --------------------------------------- #
    def v_prob(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(out["v_logit"])

    def t_pmf(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.softmax(out["t_logits"], dim=-1)

    def t_mean(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.config.t_head == "dist":
            return (self.t_pmf(out) * self.bin_centers_buf).sum(dim=-1)
        if self.config.t_head == "quantile":
            mid = self.config.quantiles.index(0.5) if 0.5 in self.config.quantiles else 0
            value = out["t_quantiles"][:, mid]
            return torch.expm1(value) if self.config.log_target else value
        value = out["t_point"]
        return torch.expm1(value) if self.config.log_target else value

    def t_cdf_at(
        self, out: dict[str, torch.Tensor], budget: torch.Tensor
    ) -> torch.Tensor:
        """``P(T <= budget)``. ``budget`` broadcasts against the batch."""
        if self.config.t_head == "dist":
            pmf = self.t_pmf(out)
            edges = self.bin_edges.unsqueeze(0)
            budget = budget.reshape(-1, 1) if budget.dim() else budget.reshape(1, 1)
            included = (edges <= budget).to(pmf.dtype)
            return (pmf * included).sum(dim=-1)
        # Quantile / point heads: logistic surrogate around the median with a
        # scale read off the predicted spread.
        mean = self.t_mean(out)
        if self.config.t_head == "quantile":
            q = out["t_quantiles"]
            spread = (q[:, -1] - q[:, 0]).abs().clamp_min(1e-3)
            if self.config.log_target:
                spread = torch.expm1(q[:, -1]) - torch.expm1(q[:, 0])
                spread = spread.abs().clamp_min(1.0)
        else:
            spread = mean.abs().clamp_min(1.0) * 0.5
        budget = budget.reshape(-1) if budget.dim() else budget.reshape(1)
        return torch.sigmoid((budget - mean) / (spread / 2.197))
