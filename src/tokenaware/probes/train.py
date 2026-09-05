"""Training loop for V/T probes over outcome draws.

Default schedule, and the reason for it: stage A trains on *every* draw in the
train split — the ~96K same-trace outcomes plus the ~96K Monte-Carlo
continuations — because each is one unbiased sample from ``π(·|s)`` and the
likelihoods used here are correct on single draws. Stage B then fine-tunes on
Monte-Carlo draws only at a tenth of the learning rate, so the final model is
adapted to the low-noise, on-distribution label set that evaluation uses,
without throwing away the 8x larger noisy corpus.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .cache import ProbeCache, SPLIT_TRAIN, SPLIT_VAL
from .features import DrawTable, FeatureSpec, FeatureStore, build_draws
from .heads import DEFAULT_QUANTILES, Probe, ProbeConfig, bin_of
from .losses import l1_loss, pinball_loss, t_censored_nll, v_bce_loss


@dataclass
class TrainConfig:
    # features
    layers: list[int] = field(default_factory=lambda: [35])
    use_hidden: bool = True
    use_delta: bool = False
    use_pos: bool = True
    use_meta: bool = False
    history: int = 1
    # architecture
    task: str = "joint"
    trunk: str = "star"
    t_head: str = "dist"
    n_bins: int = 32
    max_tokens: int = 1024
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    log_target: bool = True
    dropout: float = 0.1
    # optimisation
    epochs: int = 6
    mc_finetune_epochs: int = 2
    batch_size: int = 512
    lr: float = 5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    pos_weight: float = 3.0
    t_loss_weight: float = 1.0
    v_loss_weight: float = 1.0
    aux_l1_weight: float = 0.0
    # data
    use_same_trace: bool = True
    use_mc: bool = True
    mc_weight: float = 1.0
    same_weight: float = 1.0
    # runtime
    seed: int = 0
    device: str = "cuda"
    preload: bool = True
    amp: bool = True
    patience: int = 3

    def feature_spec(self) -> FeatureSpec:
        return FeatureSpec(
            layers=list(self.layers),
            use_hidden=self.use_hidden,
            use_delta=self.use_delta,
            use_pos=self.use_pos,
            use_meta=self.use_meta,
            history=self.history,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["quantiles"] = list(self.quantiles)
        return payload

    @staticmethod
    def from_dict(payload: dict) -> "TrainConfig":
        payload = dict(payload)
        payload["quantiles"] = tuple(payload.get("quantiles", DEFAULT_QUANTILES))
        payload["layers"] = list(payload.get("layers", [35]))
        known = set(TrainConfig.__dataclass_fields__)
        return TrainConfig(**{k: v for k, v in payload.items() if k in known})

    def tag(self) -> str:
        bits = [self.task, self.trunk]
        if self.task in ("t", "joint"):
            bits.append(self.t_head)
        bits.append(self.feature_spec().describe())
        return "_".join(bits)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DrawBatcher:
    """Keeps a draw table on the training device and yields shuffled batches."""

    def __init__(self, table: DrawTable, device: torch.device, edges: torch.Tensor):
        self.rows = torch.from_numpy(table.rows).to(device)
        self.correct = torch.from_numpy(table.correct.astype(np.float32)).to(device)
        self.length = torch.from_numpy(table.length.astype(np.float32)).to(device)
        self.censored = torch.from_numpy(table.censored.astype(np.float32)).to(device)
        self.weight = torch.from_numpy(table.weight).to(device)
        self.bins = bin_of(self.length, edges.to(device)).long()
        self.log_length = torch.log1p(self.length)
        self.n = self.rows.shape[0]
        self.device = device

    def batches(self, batch_size: int, shuffle: bool = True, generator=None):
        order = (
            torch.randperm(self.n, device=self.device, generator=generator)
            if shuffle
            else torch.arange(self.n, device=self.device)
        )
        for start in range(0, self.n, batch_size):
            idx = order[start : start + batch_size]
            yield {
                "rows": self.rows.index_select(0, idx),
                "correct": self.correct.index_select(0, idx),
                "length": self.length.index_select(0, idx),
                "log_length": self.log_length.index_select(0, idx),
                "censored": self.censored.index_select(0, idx),
                "weight": self.weight.index_select(0, idx),
                "bins": self.bins.index_select(0, idx),
            }


def compute_loss(
    probe: Probe, out: dict, batch: dict, cfg: TrainConfig
) -> tuple[torch.Tensor, dict]:
    parts: dict[str, float] = {}
    total = torch.zeros((), device=batch["correct"].device)
    if "v_logit" in out:
        loss_v = v_bce_loss(
            out["v_logit"], batch["correct"], batch["weight"], cfg.pos_weight
        )
        total = total + cfg.v_loss_weight * loss_v
        parts["v_bce"] = float(loss_v.detach())
    if cfg.t_head == "dist" and "t_logits" in out:
        loss_t = t_censored_nll(
            out["t_logits"], batch["bins"], batch["censored"], batch["weight"]
        )
        total = total + cfg.t_loss_weight * loss_t
        parts["t_nll"] = float(loss_t.detach())
    elif cfg.t_head == "quantile" and "t_quantiles" in out:
        target = batch["log_length"] if cfg.log_target else batch["length"]
        loss_t = pinball_loss(
            out["t_quantiles"],
            target,
            cfg.quantiles,
            batch["weight"],
            batch["censored"],
        )
        total = total + cfg.t_loss_weight * loss_t
        parts["t_pinball"] = float(loss_t.detach())
    elif cfg.t_head == "point" and "t_point" in out:
        target = batch["log_length"] if cfg.log_target else batch["length"]
        loss_t = l1_loss(out["t_point"], target, batch["weight"], batch["censored"])
        total = total + cfg.t_loss_weight * loss_t
        parts["t_l1"] = float(loss_t.detach())
    if cfg.aux_l1_weight > 0 and "t_logits" in out:
        mean = probe.t_mean(out)
        aux = l1_loss(
            torch.log1p(mean.clamp_min(0.0)),
            batch["log_length"],
            batch["weight"],
            batch["censored"],
        )
        total = total + cfg.aux_l1_weight * aux
        parts["t_aux_l1"] = float(aux.detach())
    parts["total"] = float(total.detach())
    return total, parts


def forward_batch(probe: Probe, store: FeatureStore, rows: torch.Tensor) -> dict:
    if store.spec.history > 1:
        x, mask = store.history(rows)
        return probe(x, mask)
    return probe(store(rows))


@torch.no_grad()
def evaluate_loss(
    probe: Probe,
    store: FeatureStore,
    batcher: DrawBatcher,
    cfg: TrainConfig,
    batch_size: int = 4096,
) -> dict:
    probe.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    for batch in batcher.batches(batch_size, shuffle=False):
        out = forward_batch(probe, store, batch["rows"])
        _, parts = compute_loss(probe, out, batch, cfg)
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1
    probe.train()
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def train_probe(
    cache: ProbeCache,
    cfg: TrainConfig,
    verbose: bool = True,
) -> dict:
    """Train one probe. Returns a dict with the probe, feature store, and history."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    spec = cfg.feature_spec()
    store = FeatureStore(cache, spec, device=device, preload=cfg.preload)

    train_rows = np.nonzero(cache.split_mask(SPLIT_TRAIN))[0]
    if train_rows.shape[0] == 0:
        raise RuntimeError("cache has no train-split states")
    store.fit_normalizer(train_rows)

    probe_cfg = ProbeConfig(
        task=cfg.task,
        trunk=cfg.trunk,
        d_in=store.d_in,
        dropout=cfg.dropout,
        t_head=cfg.t_head,
        n_bins=cfg.n_bins,
        max_tokens=cfg.max_tokens,
        quantiles=tuple(cfg.quantiles),
        log_target=cfg.log_target,
    )
    probe = Probe(probe_cfg).to(device)

    train_all = build_draws(
        cache,
        SPLIT_TRAIN,
        use_same_trace=cfg.use_same_trace,
        use_mc=cfg.use_mc,
        mc_weight=cfg.mc_weight,
        same_weight=cfg.same_weight,
    )
    train_mc = build_draws(
        cache, SPLIT_TRAIN, use_same_trace=False, use_mc=True, mc_weight=1.0
    )
    val_mc = build_draws(cache, SPLIT_VAL, use_same_trace=False, use_mc=True)
    if len(val_mc) == 0:
        # No MC labels on the val side; fall back to same-trace draws for early stop.
        val_mc = build_draws(cache, SPLIT_VAL, use_same_trace=True, use_mc=False)

    edges = probe.bin_edges.detach().cpu()
    stage_a = DrawBatcher(train_all, device, edges)
    stage_b = DrawBatcher(train_mc, device, edges) if len(train_mc) else None
    val_batcher = DrawBatcher(val_mc, device, edges) if len(val_mc) else None

    if verbose:
        print(
            f"  d_in={store.d_in} params={probe.n_params/1e6:.2f}M "
            f"stage_a={len(train_all)} draws stage_b={len(train_mc)} "
            f"val={len(val_mc)}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    use_amp = cfg.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16
    history: list[dict] = []
    best = {"loss": float("inf"), "epoch": -1, "state": None}
    bad_epochs = 0
    started = time.monotonic()

    schedule = [("A", stage_a, cfg.epochs, cfg.lr)]
    if stage_b is not None and cfg.mc_finetune_epochs > 0:
        schedule.append(("B", stage_b, cfg.mc_finetune_epochs, cfg.lr / 10.0))

    global_epoch = 0
    for stage_name, batcher, n_epochs, lr in schedule:
        for group in optimizer.param_groups:
            group["lr"] = lr
        for epoch in range(n_epochs):
            global_epoch += 1
            probe.train()
            running: dict[str, float] = {}
            n_batches = 0
            for batch in batcher.batches(
                cfg.batch_size, shuffle=True, generator=generator
            ):
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        out = forward_batch(probe, store, batch["rows"])
                        loss, parts = compute_loss(probe, out, batch, cfg)
                else:
                    out = forward_batch(probe, store, batch["rows"])
                    loss, parts = compute_loss(probe, out, batch, cfg)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg.grad_clip)
                optimizer.step()
                for key, value in parts.items():
                    running[key] = running.get(key, 0.0) + value
                n_batches += 1
            train_stats = {k: v / max(n_batches, 1) for k, v in running.items()}
            val_stats = (
                evaluate_loss(probe, store, val_batcher, cfg)
                if val_batcher is not None
                else {}
            )
            record = {
                "stage": stage_name,
                "epoch": global_epoch,
                "lr": lr,
                "train": train_stats,
                "val": val_stats,
                "seconds": round(time.monotonic() - started, 1),
            }
            history.append(record)
            if verbose:
                val_total = val_stats.get("total", float("nan"))
                print(
                    f"  [{stage_name}{epoch + 1}/{n_epochs}] "
                    f"train={train_stats.get('total', float('nan')):.4f} "
                    f"val={val_total:.4f} ({record['seconds']:.0f}s)",
                    flush=True,
                )
            val_total = val_stats.get("total", float("nan"))
            if not np.isnan(val_total) and val_total < best["loss"] - 1e-5:
                best = {
                    "loss": val_total,
                    "epoch": global_epoch,
                    "state": {
                        k: v.detach().clone() for k, v in probe.state_dict().items()
                    },
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if cfg.patience and bad_epochs >= cfg.patience and stage_name == "A":
                    if verbose:
                        print("  early stop (stage A)", flush=True)
                    break

    if best["state"] is not None:
        probe.load_state_dict(best["state"])
    probe.eval()
    return {
        "probe": probe,
        "store": store,
        "history": history,
        "best_epoch": best["epoch"],
        "best_val_loss": best["loss"],
        "train_draws": train_all.summary(),
        "mc_draws": train_mc.summary(),
        "val_draws": val_mc.summary(),
        "seconds": round(time.monotonic() - started, 1),
    }


def save_probe(
    path: str | Path,
    probe: Probe,
    store: FeatureStore,
    cfg: TrainConfig,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe_state": probe.state_dict(),
        "probe_config": probe.config.to_dict(),
        "train_config": cfg.to_dict(),
        "normalizer": store.normalizer.state_dict() if store.normalizer else None,
        "feature_spec": asdict(store.spec),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_probe(
    path: str | Path, cache: ProbeCache, device: str = "cuda", preload: bool = True
) -> tuple[Probe, FeatureStore, TrainConfig]:
    from .features import Normalizer

    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig.from_dict(payload["train_config"])
    resolved = resolve_device(device)
    spec = FeatureSpec(**payload["feature_spec"])
    store = FeatureStore(cache, spec, device=resolved, preload=preload)
    if payload.get("normalizer"):
        store.normalizer = Normalizer.from_state_dict(payload["normalizer"]).to(resolved)
    probe = Probe(ProbeConfig.from_dict(payload["probe_config"])).to(resolved)
    probe.load_state_dict(payload["probe_state"])
    probe.eval()
    return probe, store, cfg


def dump_history(path: str | Path, result: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "history": result["history"],
                "best_epoch": result["best_epoch"],
                "best_val_loss": result["best_val_loss"],
                "train_draws": result["train_draws"],
                "mc_draws": result["mc_draws"],
                "val_draws": result["val_draws"],
                "seconds": result["seconds"],
            },
            indent=2,
        )
    )
