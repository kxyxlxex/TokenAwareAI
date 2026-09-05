"""On-disk probe cache: a flat state table joined to memory-mapped hidden states.

Layout under ``<cache_dir>``::

    meta.json              layers, hidden dim, subject vocab, provenance, counts
    hidden_L8.npy          float16 [n_states, hidden_dim]  (one file per layer)
    states.npz             the state table (see STATE_COLUMNS)
    mc.npz                 Monte-Carlo labels, one row per MC-labelled state

One *state* is one reasoning-step boundary at which a hidden vector exists.
Rows for a single (problem, sample) trace are contiguous and ordered by step,
so a state's step history is ``rows[hist_start : row + 1]``.

Every state carries a cheap, biased *same-trace* label pair (the realised
outcome and realised remaining length of the one trace it came from). A subset
also carries low-noise *Monte-Carlo* labels from k fresh continuations.
"""

from __future__ import annotations

import io
import json
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

META_NAME = "meta.json"
STATES_NAME = "states.npz"
MC_NAME = "mc.npz"
HIDDEN_FMT = "hidden_L{layer}.npy"

SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

SOURCE_ROOT = 0
SOURCE_BRANCH = 1

SPLIT_TRAIN = 0
SPLIT_VAL = 1

# dtype per state column; the writer and reader agree on this dict alone.
STATE_COLUMNS: dict[str, str] = {
    "problem_ord": "int32",
    "problem_idx": "int32",
    "sample_id": "int16",
    "step_index": "int16",
    "n_steps": "int16",
    "row_in_trace": "int16",
    "hist_start": "int32",
    "tokens_so_far": "int32",
    "trace_len": "int32",
    "t_same": "int32",
    "y_same": "uint8",
    "trace_truncated": "uint8",
    "level": "uint8",
    "subject_idx": "uint8",
    "problem_chars": "int32",
    "split": "uint8",
    "source": "uint8",
    "group_id": "int32",
    "mc_row": "int32",
}

MC_COLUMNS: dict[str, str] = {
    "state_row": "int32",
    "fraction": "float32",
    "v_mc": "float32",
    "t_mc_mean": "float32",
    "mc_k": "int16",
    "n_correct": "int16",
    "n_truncated": "int16",
    "source_tokens_remaining": "int32",
}
# 2-D MC columns, shape [n_mc, max_k]
MC_MATRIX_COLUMNS: dict[str, str] = {
    "draw_len": "int32",
    "draw_correct": "uint8",
    "draw_truncated": "uint8",
    "draw_valid": "uint8",
}


# --------------------------------------------------------------------------- #
# torch-free .pt inspection
# --------------------------------------------------------------------------- #
class _StorageStub:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype


class _TensorStub:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype


def _rebuild_tensor(storage, _offset, size, _stride, *_rest):
    dtype = getattr(storage, "dtype", "unknown")
    return _TensorStub(tuple(size), dtype)


class _MetadataUnpickler(pickle.Unpickler):
    """Read tensor shapes out of a torch archive without importing torch."""

    def find_class(self, module: str, name: str):  # noqa: D102
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        return type(name, (), {})

    def persistent_load(self, pid):  # noqa: D102
        storage_type = pid[1] if len(pid) > 1 else "unknown"
        dtype = getattr(storage_type, "__name__", str(storage_type))
        return _StorageStub(dtype)


def pt_row_counts(path: Path) -> list[int]:
    """Rows per sample in a root hidden-state ``.pt``, without loading torch.

    The archive holds a list (one entry per rollout) of ``{layer: [rows, dim]}``.
    Returns the row count of the first layer of each entry.
    """
    with zipfile.ZipFile(path) as zf:
        pkl_names = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not pkl_names:
            raise ValueError("no data.pkl in archive")
        payload = zf.read(pkl_names[0])
    obj = _MetadataUnpickler(io.BytesIO(payload)).load()
    if not isinstance(obj, list):
        raise ValueError(f"expected a list of per-sample dicts, got {type(obj)}")
    counts = []
    for entry in obj:
        if not isinstance(entry, dict) or not entry:
            counts.append(0)
            continue
        first = next(iter(entry.values()))
        counts.append(int(first.shape[0]) if getattr(first, "shape", None) else 0)
    return counts


# --------------------------------------------------------------------------- #
# reader
# --------------------------------------------------------------------------- #
@dataclass
class ProbeCache:
    path: Path
    meta: dict
    states: dict[str, np.ndarray]
    mc: dict[str, np.ndarray]
    hidden: dict[int, np.ndarray]

    @property
    def layers(self) -> list[int]:
        return list(self.meta["layers"])

    @property
    def hidden_dim(self) -> int:
        return int(self.meta["hidden_dim"])

    @property
    def n_states(self) -> int:
        return int(self.states["problem_ord"].shape[0])

    @property
    def n_mc(self) -> int:
        return int(self.mc["state_row"].shape[0]) if self.mc else 0

    @property
    def problems(self) -> list[dict]:
        return self.meta.get("problems", [])

    def split_mask(self, split: int) -> np.ndarray:
        return self.states["split"] == split

    def mc_rows_for_split(self, split: int) -> np.ndarray:
        """Indices into the MC table whose state belongs to ``split``."""
        if not self.n_mc:
            return np.zeros(0, dtype=np.int64)
        state_split = self.states["split"][self.mc["state_row"]]
        return np.nonzero(state_split == split)[0]

    def stack_hidden(self, rows: np.ndarray, layers: list[int]) -> np.ndarray:
        """Gather ``rows`` for ``layers`` and concatenate on the feature axis."""
        blocks = [np.asarray(self.hidden[layer][rows]) for layer in layers]
        return np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]


def load_cache(
    path: str | Path,
    layers: list[int] | None = None,
    mmap: bool = True,
) -> ProbeCache:
    path = Path(path).expanduser().resolve()
    meta = json.loads((path / META_NAME).read_text())
    states_npz = np.load(path / STATES_NAME)
    states = {k: states_npz[k] for k in states_npz.files}
    mc: dict[str, np.ndarray] = {}
    mc_path = path / MC_NAME
    if mc_path.is_file():
        mc_npz = np.load(mc_path)
        mc = {k: mc_npz[k] for k in mc_npz.files}

    wanted = layers if layers is not None else list(meta["layers"])
    missing = [layer for layer in wanted if layer not in meta["layers"]]
    if missing:
        raise ValueError(
            f"cache {path} has layers {meta['layers']}, requested {wanted}"
        )
    hidden = {}
    for layer in wanted:
        file = path / HIDDEN_FMT.format(layer=layer)
        hidden[layer] = np.load(file, mmap_mode="r" if mmap else None)
    return ProbeCache(path=path, meta=meta, states=states, mc=mc, hidden=hidden)


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #
class CacheWriter:
    """Accumulate state rows in memory and stream hidden vectors to disk."""

    def __init__(
        self,
        path: str | Path,
        layers: list[int],
        hidden_dim: int,
        n_states: int,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.layers = list(layers)
        self.hidden_dim = int(hidden_dim)
        self.n_states = int(n_states)
        self.cursor = 0
        self._hidden = {
            layer: np.lib.format.open_memmap(
                self.path / HIDDEN_FMT.format(layer=layer),
                mode="w+",
                dtype=np.float16,
                shape=(self.n_states, self.hidden_dim),
            )
            for layer in self.layers
        }
        self._states: dict[str, list] = {k: [] for k in STATE_COLUMNS}
        self._mc: dict[str, list] = {k: [] for k in MC_COLUMNS}
        self._mc_matrices: dict[str, list] = {k: [] for k in MC_MATRIX_COLUMNS}

    def add_states(self, rows: list[dict], hidden: dict[int, np.ndarray]) -> int:
        """Append ``len(rows)`` states. ``hidden[layer]`` is [len(rows), dim]."""
        n = len(rows)
        if n == 0:
            return self.cursor
        start = self.cursor
        if start + n > self.n_states:
            raise ValueError(
                f"cache sized for {self.n_states} states, tried to write "
                f"{start + n}; recount before writing"
            )
        for layer in self.layers:
            block = np.asarray(hidden[layer], dtype=np.float16)
            if block.shape != (n, self.hidden_dim):
                raise ValueError(
                    f"layer {layer} block {block.shape} != {(n, self.hidden_dim)}"
                )
            self._hidden[layer][start : start + n] = block
        for row in rows:
            for key in STATE_COLUMNS:
                self._states[key].append(row.get(key, 0))
        self.cursor = start + n
        return start

    def add_mc(self, row: dict, draws: list[dict], max_k: int) -> int:
        mc_index = len(self._mc["state_row"])
        for key in MC_COLUMNS:
            self._mc[key].append(row.get(key, 0))
        lengths = np.zeros(max_k, dtype=np.int32)
        correct = np.zeros(max_k, dtype=np.uint8)
        truncated = np.zeros(max_k, dtype=np.uint8)
        valid = np.zeros(max_k, dtype=np.uint8)
        for i, draw in enumerate(draws[:max_k]):
            lengths[i] = int(draw.get("n_tokens", 0))
            correct[i] = 1 if draw.get("correct") else 0
            truncated[i] = 1 if draw.get("truncated") else 0
            valid[i] = 1
        self._mc_matrices["draw_len"].append(lengths)
        self._mc_matrices["draw_correct"].append(correct)
        self._mc_matrices["draw_truncated"].append(truncated)
        self._mc_matrices["draw_valid"].append(valid)
        return mc_index

    def set_state_field(self, state_row: int, key: str, value) -> None:
        if key not in STATE_COLUMNS:
            raise KeyError(key)
        self._states[key][state_row] = value

    def set_state_mc_row(self, state_row: int, mc_row: int) -> None:
        self._states["mc_row"][state_row] = mc_row

    def finalize(self, meta: dict) -> None:
        for layer in self.layers:
            self._hidden[layer].flush()
        if self.cursor != self.n_states:
            # Shrink the memmaps to the rows actually written.
            for layer in self.layers:
                file = self.path / HIDDEN_FMT.format(layer=layer)
                full = np.load(file, mmap_mode="r")
                trimmed = np.array(full[: self.cursor], dtype=np.float16)
                del full
                np.save(file, trimmed)
        states = {
            key: np.asarray(values, dtype=np.dtype(STATE_COLUMNS[key]))
            for key, values in self._states.items()
        }
        np.savez(self.path / STATES_NAME, **states)
        mc = {
            key: np.asarray(values, dtype=np.dtype(MC_COLUMNS[key]))
            for key, values in self._mc.items()
        }
        for key, dtype in MC_MATRIX_COLUMNS.items():
            stack = self._mc_matrices[key]
            mc[key] = (
                np.stack(stack).astype(np.dtype(dtype))
                if stack
                else np.zeros((0, 0), dtype=np.dtype(dtype))
            )
        np.savez(self.path / MC_NAME, **mc)
        payload = dict(meta)
        payload.update(
            {
                "layers": self.layers,
                "hidden_dim": self.hidden_dim,
                "n_states": self.cursor,
                "n_mc": int(len(self._mc["state_row"])),
            }
        )
        (self.path / META_NAME).write_text(json.dumps(payload, indent=2))
