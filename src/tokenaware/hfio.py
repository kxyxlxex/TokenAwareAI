"""Hugging Face Hub transfer for the artifact corpus.

The corpus is too large to keep on a laptop, so the canonical copy lives in a
private Hub dataset repo as one or more archives. A GPU box pulls the archives,
extracts them into ``TOKENAWARE_ARTIFACTS``, and pushes derived outputs
(caches, checkpoints, reports) back.
"""

from __future__ import annotations

import json
import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO_ID = os.environ.get(
    "TOKENAWARE_HF_REPO", "kxyxlxex/tokenaware-artifacts"
)
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar.zst", ".tar", ".zip")
# Archives may or may not carry a leading `artifacts/` component.
KNOWN_TOP_LEVEL = ("splits", "rollouts", "labels", "logs", "branches", "reports")


def resolve_token(explicit: str | None = None) -> str | None:
    """Token from the CLI, then the usual environment variables, then the CLI cache."""
    if explicit:
        return explicit
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value.strip()
    for path in (
        Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "token",
        Path("~/.huggingface/token").expanduser(),
    ):
        if path.is_file():
            value = path.read_text().strip()
            if value:
                return value
    return None


def enable_fast_transfer() -> None:
    """Opt into hf_transfer when the wheel is importable."""
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        return
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


@dataclass
class RemoteFile:
    path: str
    size: int | None

    @property
    def is_archive(self) -> bool:
        return self.path.endswith(ARCHIVE_SUFFIXES)


def list_repo(
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
) -> list[RemoteFile]:
    from huggingface_hub import HfApi

    api = HfApi(token=resolve_token(token))
    info = api.repo_info(
        repo_id=repo_id, repo_type=repo_type, revision=revision, files_metadata=True
    )
    files = []
    for sibling in info.siblings or []:
        files.append(RemoteFile(sibling.rfilename, getattr(sibling, "size", None)))
    return sorted(files, key=lambda f: f.path)


def download_file(
    remote_path: str,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=remote_path,
        revision=revision,
        token=resolve_token(token),
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(local)


def _strip_leading_artifacts(name: str) -> str | None:
    """Normalise an archive member path to be relative to the artifacts root.

    Returns ``None`` for members that must not be written (absolute paths,
    parent-directory escapes, or the archive's own top-level directory entry).
    """
    parts = [p for p in Path(name).parts if p not in (".", "")]
    if not parts:
        return None
    if any(p == ".." for p in parts) or name.startswith("/"):
        return None
    # Drop any number of leading wrapper dirs until a known top-level appears.
    while len(parts) > 1 and parts[0] not in KNOWN_TOP_LEVEL:
        parts = parts[1:]
    return str(Path(*parts))


def _safe_targets(names: list[str], dest: Path) -> dict[str, Path]:
    """Map archive member names to absolute destinations, rejecting escapes."""
    dest = dest.resolve()
    mapping: dict[str, Path] = {}
    for name in names:
        rel = _strip_leading_artifacts(name)
        if rel is None:
            continue
        target = (dest / rel).resolve()
        if dest not in target.parents and target != dest:
            continue
        mapping[name] = target
    return mapping


def extract_archive(archive: Path, dest: Path, overwrite: bool = False) -> int:
    """Extract a tar/zip archive into ``dest``. Returns the file count written."""
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            members = [i for i in zf.infolist() if not i.is_dir()]
            targets = _safe_targets([m.filename for m in members], dest)
            for member in members:
                target = targets.get(member.filename)
                if target is None:
                    continue
                if target.exists() and not overwrite:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
                written += 1
        return written

    mode = "r:*"
    with tarfile.open(archive, mode) as tf:
        for member in tf:
            if not member.isfile():
                continue
            targets = _safe_targets([member.name], dest)
            target = targets.get(member.name)
            if target is None:
                continue
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            written += 1
    return written


def upload_folder(
    folder: Path,
    path_in_repo: str,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = "dataset",
    token: str | None = None,
    allow_patterns: list[str] | None = None,
    commit_message: str | None = None,
) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=resolve_token(token))
    api.create_repo(
        repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True
    )
    return api.upload_folder(
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=allow_patterns,
        commit_message=commit_message or f"upload {path_in_repo}",
    )


def upload_file(
    local_path: Path,
    path_in_repo: str,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = "dataset",
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=resolve_token(token))
    api.create_repo(
        repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True
    )
    return api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message or f"upload {path_in_repo}",
    )


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
