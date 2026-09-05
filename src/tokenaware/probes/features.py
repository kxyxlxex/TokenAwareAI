"""Turn cache rows into probe inputs and outcome-draw training tables.

Two rules drive the design.

**Causality.** A feature is only allowed if it is computable at search time from
the prefix alone. ``n_steps`` and the depth *fraction* of a step within its
finished trace are therefore forbidden: they encode the future. ``tokens_so_far``,
``step_index``, problem length, level and subject are all legal.

**Draws, not means.** Every training target is one realised outcome of one
continuation: ``(hidden, correct, length, censored)``. A same-trace state
contributes one draw; a Monte-Carlo state contributes ``k`` draws from the same
hidden vector. Bernoulli / discretised-count likelihoods on single draws are
unbiased for ``V(s)`` and for the law of ``T(s)``, so no soft-label machinery is
needed and the abundant same-trace corpus becomes first-class training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .cache import SOURCE_ROOT, SUBJECTS, ProbeCache

N_LEVELS = 5


@dataclass
class FeatureSpec:
    layers: list[int]
    use_hidden: bool = True
    use_delta: bool = False
    use_pos: bool = True
    use_meta: bool = False
    history: int = 1

    @property
    def n_scalar(self) -> int:
        n = 0
        if self.use_pos:
            n += 4
        if self.use_meta:
            n += N_LEVELS + len(SUBJECTS)
        return n

    def describe(self) -> str:
        parts = [f"L{'+'.join(str(x) for x in self.layers)}"] if self.use_hidden else ["nohidden"]
        if self.use_delta:
            parts.append("delta")
        if self.use_pos:
            parts.append("pos")
        if self.use_meta:
            parts.append("meta")
        if self.history > 1:
            parts.append(f"hist{self.history}")
        return "-".join(parts)


@dataclass
class Normalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def to(self, device) -> "Normalizer":
        return Normalizer(self.mean.to(device), self.std.to(device))

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @staticmethod
    def from_state_dict(payload: dict) -> "Normalizer":
        return Normalizer(payload["mean"], payload["std"])


def scalar_matrix(cache: ProbeCache, spec: FeatureSpec) -> np.ndarray:
    """Causal, non-hidden features for every state in the cache."""
    states = cache.states
    blocks: list[np.ndarray] = []
    if spec.use_pos:
        tokens = states["tokens_so_far"].astype(np.float32)
        step = states["step_index"].astype(np.float32)
        chars = states["problem_chars"].astype(np.float32)
        blocks.append(
            np.stack(
                [
                    np.log1p(tokens),
                    tokens / 1024.0,
                    step / 16.0,
                    np.log1p(chars) / 8.0,
                ],
                axis=1,
            )
        )
    if spec.use_meta:
        level = np.clip(states["level"].astype(np.int64) - 1, 0, N_LEVELS - 1)
        level_oh = np.zeros((level.shape[0], N_LEVELS), dtype=np.float32)
        level_oh[np.arange(level.shape[0]), level] = 1.0
        subject = np.clip(
            states["subject_idx"].astype(np.int64), 0, len(SUBJECTS) - 1
        )
        subject_oh = np.zeros((subject.shape[0], len(SUBJECTS)), dtype=np.float32)
        subject_oh[np.arange(subject.shape[0]), subject] = 1.0
        blocks.append(level_oh)
        blocks.append(subject_oh)
    if not blocks:
        return np.zeros((cache.n_states, 0), dtype=np.float32)
    return np.concatenate(blocks, axis=1).astype(np.float32)


class FeatureStore:
    """Holds hidden states and scalars on one device and gathers rows on demand."""

    def __init__(
        self,
        cache: ProbeCache,
        spec: FeatureSpec,
        device: torch.device | str = "cpu",
        preload: bool = True,
    ) -> None:
        self.cache = cache
        self.spec = spec
        self.device = torch.device(device)
        self.preload = preload
        self.width = cache.hidden_dim
        self.d_hidden = self.width * len(spec.layers) if spec.use_hidden else 0

        self._hidden: torch.Tensor | None = None
        if spec.use_hidden and preload:
            # Fill layer by layer so the fp16 corpus is never duplicated in RAM.
            self._hidden = torch.empty(
                (cache.n_states, self.d_hidden),
                dtype=torch.float16,
                device=self.device,
            )
            chunk = 8192
            for i, layer in enumerate(spec.layers):
                lo_col, hi_col = i * self.width, (i + 1) * self.width
                source = cache.hidden[layer]
                for start in range(0, cache.n_states, chunk):
                    stop = min(start + chunk, cache.n_states)
                    block = np.array(source[start:stop], dtype=np.float16)
                    self._hidden[start:stop, lo_col:hi_col] = torch.from_numpy(
                        block
                    ).to(self.device)

        scalars = scalar_matrix(cache, spec)
        self._scalars = torch.from_numpy(scalars).to(self.device)

        row_in_trace = cache.states["row_in_trace"].astype(np.int64)
        prev = np.arange(cache.n_states, dtype=np.int64) - 1
        prev[row_in_trace == 0] = np.arange(cache.n_states, dtype=np.int64)[
            row_in_trace == 0
        ]
        self._prev = torch.from_numpy(prev).to(self.device)
        self._has_prev = torch.from_numpy((row_in_trace > 0).astype(np.float32)).to(
            self.device
        )
        self._hist_start = torch.from_numpy(
            cache.states["hist_start"].astype(np.int64)
        ).to(self.device)

        self.normalizer: Normalizer | None = None

    # -- hidden access ----------------------------------------------------- #
    def _hidden_rows(self, rows: torch.Tensor) -> torch.Tensor:
        if self._hidden is not None:
            return self._hidden.index_select(0, rows).float()
        block = self.cache.stack_hidden(
            rows.detach().cpu().numpy(), self.spec.layers
        )
        return torch.from_numpy(np.array(block, dtype=np.float16)).to(
            self.device
        ).float()

    @property
    def d_in(self) -> int:
        d = self.d_hidden * (2 if self.spec.use_delta else 1)
        return d + self.spec.n_scalar

    def fit_normalizer(self, rows: np.ndarray, max_rows: int = 50_000) -> Normalizer:
        if rows.shape[0] > max_rows:
            idx = np.linspace(0, rows.shape[0] - 1, max_rows).astype(np.int64)
            rows = rows[idx]
        with torch.no_grad():
            sample = self.raw(torch.from_numpy(rows).to(self.device))
            mean = sample.mean(dim=0)
            std = sample.std(dim=0).clamp_min(1e-3)
        self.normalizer = Normalizer(mean, std)
        return self.normalizer

    def raw(self, rows: torch.Tensor) -> torch.Tensor:
        """Un-normalised features for ``rows``: [n, d_in]."""
        parts: list[torch.Tensor] = []
        if self.spec.use_hidden:
            current = self._hidden_rows(rows)
            parts.append(current)
            if self.spec.use_delta:
                prev_rows = self._prev.index_select(0, rows)
                previous = self._hidden_rows(prev_rows)
                mask = self._has_prev.index_select(0, rows).unsqueeze(1)
                parts.append((current - previous) * mask)
        if self.spec.n_scalar:
            parts.append(self._scalars.index_select(0, rows))
        if not parts:
            raise ValueError(
                "feature spec selects nothing; enable hidden states or a scalar group"
            )
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        x = self.raw(rows)
        if self.normalizer is not None:
            x = (x - self.normalizer.mean) / self.normalizer.std
        return x

    def history(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequence features for the ReProbe-style trunk.

        Returns ``(x, mask)`` with ``x`` of shape [n, W, d_in] ordered oldest to
        newest and ``mask`` True where the slot is padding.
        """
        window = self.spec.history
        offsets = torch.arange(window - 1, -1, -1, device=self.device)
        candidate = rows.unsqueeze(1) - offsets.unsqueeze(0)
        start = self._hist_start.index_select(0, rows).unsqueeze(1)
        pad = candidate < start
        clamped = torch.clamp(candidate, min=0)
        clamped = torch.where(pad, rows.unsqueeze(1), clamped)
        flat = clamped.reshape(-1)
        x = self(flat).reshape(rows.shape[0], window, -1)
        return x, pad


# --------------------------------------------------------------------------- #
# outcome draws
# --------------------------------------------------------------------------- #
@dataclass
class DrawTable:
    rows: np.ndarray  # int64 index into cache states
    correct: np.ndarray  # uint8
    length: np.ndarray  # int32 tokens after the state
    censored: np.ndarray  # uint8, generation hit the hard cap
    weight: np.ndarray  # float32
    origin: np.ndarray  # uint8, 0 = same-trace, 1 = Monte-Carlo

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def summary(self) -> dict:
        return {
            "n_draws": len(self),
            "n_same_trace": int((self.origin == 0).sum()),
            "n_mc": int((self.origin == 1).sum()),
            "n_states": int(np.unique(self.rows).shape[0]),
            "correct_rate": round(float(self.correct.mean()), 4) if len(self) else 0.0,
            "censored_rate": round(float(self.censored.mean()), 4) if len(self) else 0.0,
            "length_mean": round(float(self.length.mean()), 2) if len(self) else 0.0,
            "length_p90": (
                round(float(np.quantile(self.length, 0.9)), 2) if len(self) else 0.0
            ),
        }


def build_draws(
    cache: ProbeCache,
    split: int,
    use_same_trace: bool = True,
    use_mc: bool = True,
    mc_weight: float = 1.0,
    same_weight: float = 1.0,
) -> DrawTable:
    states = cache.states
    in_split = states["split"] == split
    rows_all, correct_all, length_all, censored_all, weight_all, origin_all = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    if use_same_trace:
        eligible = np.nonzero(in_split & (states["source"] == SOURCE_ROOT))[0]
        rows_all.append(eligible.astype(np.int64))
        correct_all.append(states["y_same"][eligible].astype(np.uint8))
        length_all.append(states["t_same"][eligible].astype(np.int32))
        censored_all.append(states["trace_truncated"][eligible].astype(np.uint8))
        weight_all.append(np.full(eligible.shape[0], same_weight, dtype=np.float32))
        origin_all.append(np.zeros(eligible.shape[0], dtype=np.uint8))

    if use_mc and cache.n_mc:
        mc_rows = cache.mc_rows_for_split(split)
        if mc_rows.shape[0]:
            state_rows = cache.mc["state_row"][mc_rows].astype(np.int64)
            valid = cache.mc["draw_valid"][mc_rows].astype(bool)
            lengths = cache.mc["draw_len"][mc_rows]
            corrects = cache.mc["draw_correct"][mc_rows]
            truncs = cache.mc["draw_truncated"][mc_rows]
            repeat = valid.sum(axis=1)
            rows_all.append(np.repeat(state_rows, repeat))
            correct_all.append(corrects[valid].astype(np.uint8))
            length_all.append(lengths[valid].astype(np.int32))
            censored_all.append(truncs[valid].astype(np.uint8))
            weight_all.append(
                np.full(int(valid.sum()), mc_weight, dtype=np.float32)
            )
            origin_all.append(np.ones(int(valid.sum()), dtype=np.uint8))

    if not rows_all:
        empty_i = np.zeros(0, dtype=np.int64)
        return DrawTable(
            empty_i,
            np.zeros(0, np.uint8),
            np.zeros(0, np.int32),
            np.zeros(0, np.uint8),
            np.zeros(0, np.float32),
            np.zeros(0, np.uint8),
        )
    return DrawTable(
        rows=np.concatenate(rows_all),
        correct=np.concatenate(correct_all),
        length=np.concatenate(length_all),
        censored=np.concatenate(censored_all),
        weight=np.concatenate(weight_all),
        origin=np.concatenate(origin_all),
    )


@dataclass
class EvalStates:
    """Monte-Carlo-labelled states used for every reported metric."""

    rows: np.ndarray
    problem_idx: np.ndarray
    group_id: np.ndarray
    fraction: np.ndarray
    source: np.ndarray
    step_index: np.ndarray
    tokens_so_far: np.ndarray
    level: np.ndarray
    v_mc: np.ndarray
    t_mc_mean: np.ndarray
    draw_len: np.ndarray
    draw_correct: np.ndarray
    draw_valid: np.ndarray
    mc_k: np.ndarray
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def subset(self, mask: np.ndarray) -> "EvalStates":
        """Row subset; every per-state array is indexed by the same mask."""
        mask = np.asarray(mask, dtype=bool)
        return EvalStates(
            rows=self.rows[mask],
            problem_idx=self.problem_idx[mask],
            group_id=self.group_id[mask],
            fraction=self.fraction[mask],
            source=self.source[mask],
            step_index=self.step_index[mask],
            tokens_so_far=self.tokens_so_far[mask],
            level=self.level[mask],
            v_mc=self.v_mc[mask],
            t_mc_mean=self.t_mc_mean[mask],
            draw_len=self.draw_len[mask],
            draw_correct=self.draw_correct[mask],
            draw_valid=self.draw_valid[mask],
            mc_k=self.mc_k[mask],
            extra=dict(self.extra),
        )


def build_eval_states(cache: ProbeCache, split: int) -> EvalStates:
    mc_rows = cache.mc_rows_for_split(split)
    state_rows = cache.mc["state_row"][mc_rows].astype(np.int64)
    states = cache.states
    return EvalStates(
        rows=state_rows,
        problem_idx=states["problem_idx"][state_rows].astype(np.int64),
        group_id=states["group_id"][state_rows].astype(np.int64),
        fraction=cache.mc["fraction"][mc_rows].astype(np.float32),
        source=states["source"][state_rows].astype(np.uint8),
        step_index=states["step_index"][state_rows].astype(np.int64),
        tokens_so_far=states["tokens_so_far"][state_rows].astype(np.int64),
        level=states["level"][state_rows].astype(np.int64),
        v_mc=cache.mc["v_mc"][mc_rows].astype(np.float64),
        t_mc_mean=cache.mc["t_mc_mean"][mc_rows].astype(np.float64),
        draw_len=cache.mc["draw_len"][mc_rows].astype(np.int64),
        draw_correct=cache.mc["draw_correct"][mc_rows].astype(np.int64),
        draw_valid=cache.mc["draw_valid"][mc_rows].astype(bool),
        mc_k=cache.mc["mc_k"][mc_rows].astype(np.int64),
    )
