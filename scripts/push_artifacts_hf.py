#!/usr/bin/env python3
"""Push derived outputs back to the Hugging Face dataset repo.

The GPU box is ephemeral, so anything you want to keep has to leave before you
terminate it: reports, sweeps, probe checkpoints, and any newly generated
rollouts / MC labels / sibling branches.

  export HF_TOKEN=hf_...
  python scripts/push_artifacts_hf.py --preset results
  python scripts/push_artifacts_hf.py --preset branches
  python scripts/push_artifacts_hf.py --paths reports probes --name my-run

Directories are tarred (one archive per invocation) and uploaded under
``results/`` so the corpus archives stay untouched. ``--raw`` uploads a folder
file-by-file instead, which is better for a handful of small JSON reports.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.config import ARTIFACTS_DIR
from tokenaware.hfio import (
    DEFAULT_REPO_ID,
    enable_fast_transfer,
    resolve_token,
    upload_file,
    upload_folder,
)

PRESETS: dict[str, list[str]] = {
    "results": ["reports", "sweeps", "probes"],
    "branches": ["branches"],
    "rollouts": ["rollouts", "labels"],
    "cache": ["cache"],
    "logs": ["logs"],
}


def stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")


def make_archive(artifacts: Path, paths: list[str], out: Path) -> tuple[int, int]:
    files = 0
    with tarfile.open(out, "w:gz") as tf:
        for rel in paths:
            source = artifacts / rel
            if not source.exists():
                print(f"  skip missing {rel}")
                continue
            for item in sorted(source.rglob("*")):
                if not item.is_file() or item.name.endswith(".tmp"):
                    continue
                tf.add(item, arcname=str(item.relative_to(artifacts)))
                files += 1
    return files, out.stat().st_size


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--repo-type", default="dataset", choices=("dataset", "model"))
    p.add_argument("--token", default=None)
    p.add_argument("--artifacts", default=str(ARTIFACTS_DIR))
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    p.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help="paths relative to the artifacts dir",
    )
    p.add_argument("--name", default=None, help="archive base name")
    p.add_argument(
        "--raw",
        action="store_true",
        help="upload files individually instead of tarring them",
    )
    p.add_argument("--path-in-repo", default=None)
    p.add_argument("--message", default=None)
    args = p.parse_args()

    token = resolve_token(args.token)
    if token is None:
        print("no Hugging Face token found; set HF_TOKEN", file=sys.stderr)
        return 2
    enable_fast_transfer()

    paths = args.paths or (PRESETS.get(args.preset) if args.preset else None)
    if not paths:
        print("pass --preset or --paths", file=sys.stderr)
        return 2

    artifacts = Path(args.artifacts).expanduser().resolve()
    name = args.name or (args.preset or "artifacts")

    if args.raw:
        for rel in paths:
            source = artifacts / rel
            if not source.is_dir():
                print(f"skip missing {rel}")
                continue
            target = args.path_in_repo or f"results/{name}/{rel}"
            print(f"uploading {source} -> {target}")
            url = upload_folder(
                source,
                target,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                token=token,
                commit_message=args.message,
            )
            print(f"  {url}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{name}-{stamp()}.tar.gz"
        print(f"packing {paths} -> {archive.name}")
        started = time.monotonic()
        files, size = make_archive(artifacts, paths, archive)
        if files == 0:
            print("nothing to upload", file=sys.stderr)
            return 1
        print(
            f"  {files} files, {size / (1 << 20):.1f} MiB "
            f"in {time.monotonic() - started:.0f}s"
        )
        target = args.path_in_repo or f"results/{archive.name}"
        url = upload_file(
            archive,
            target,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            token=token,
            commit_message=args.message or f"upload {target}",
        )
        print(f"uploaded {target}\n{url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
