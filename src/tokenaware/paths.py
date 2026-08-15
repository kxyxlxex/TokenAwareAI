"""Resolve repo root whether the file was imported, run as a script, or pasted in Colab."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("TOKENAWARE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    try:
        here = Path(__file__).resolve().parents[2]
        if (here / "src" / "tokenaware").is_dir():
            return here
    except NameError:
        pass
    cwd = Path.cwd()
    if (cwd / "src" / "tokenaware").is_dir():
        return cwd
    colab = Path("/content/TokenAwareAI")
    if (colab / "src" / "tokenaware").is_dir():
        return colab
    return cwd


def ensure_src_on_path() -> Path:
    root = repo_root()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root
