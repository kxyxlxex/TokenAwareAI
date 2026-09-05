#!/usr/bin/env python3
"""Pull the artifact corpus from a private Hugging Face dataset repo.

Run this first on a fresh GPU box. Nothing is downloaded to your laptop.

  export HF_TOKEN=hf_...
  python scripts/fetch_artifacts_hf.py --list            # see what is in the repo
  python scripts/fetch_artifacts_hf.py                   # download + extract everything
  python scripts/fetch_artifacts_hf.py --include 'rollouts*'

Archives (.tar.gz/.tgz/.tar/.zip) are extracted into $TOKENAWARE_ARTIFACTS with
any leading `artifacts/` component stripped. Loose files are copied in place.
Re-running is cheap: already-extracted files are left alone unless --overwrite.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenaware.config import ARTIFACTS_DIR
from tokenaware.hfio import (
    DEFAULT_REPO_ID,
    download_file,
    enable_fast_transfer,
    extract_archive,
    list_repo,
    resolve_token,
    write_manifest,
)

SKIP_NAMES = {".gitattributes", "README.md", ".gitignore"}


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def human(n: int | None) -> str:
    if n is None:
        return "?"
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TiB"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--repo-type", default="dataset", choices=("dataset", "model"))
    p.add_argument("--revision", default=None)
    p.add_argument("--token", default=None, help="defaults to $HF_TOKEN")
    p.add_argument("--dest", default=str(ARTIFACTS_DIR))
    p.add_argument(
        "--include",
        action="append",
        default=None,
        help="glob on the remote path; repeatable. Default: everything.",
    )
    p.add_argument("--exclude", action="append", default=None)
    p.add_argument("--list", action="store_true", help="list remote files and exit")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--keep-archives",
        action="store_true",
        help="keep the downloaded archive in the HF cache after extraction",
    )
    args = p.parse_args()

    token = resolve_token(args.token)
    if token is None:
        print(
            "no Hugging Face token found. Set HF_TOKEN or pass --token. "
            "A private repo cannot be read anonymously.",
            file=sys.stderr,
        )
        return 2
    enable_fast_transfer()

    dest = Path(args.dest).expanduser().resolve()
    print(f"[{stamp()}] repo={args.repo_id} ({args.repo_type}) dest={dest}", flush=True)

    try:
        files = list_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001 - surface auth/404 verbatim
        print(f"could not list repo: {exc}", file=sys.stderr)
        return 2

    if args.list:
        total = 0
        for f in files:
            total += f.size or 0
            kind = "archive" if f.is_archive else "file"
            print(f"  {f.path:<60} {human(f.size):>10}  {kind}")
        print(f"\n{len(files)} files, {human(total)} total")
        return 0

    selected = []
    for f in files:
        name = Path(f.path).name
        if name in SKIP_NAMES:
            continue
        if args.include and not any(
            fnmatch.fnmatch(f.path, pat) for pat in args.include
        ):
            continue
        if args.exclude and any(fnmatch.fnmatch(f.path, pat) for pat in args.exclude):
            continue
        selected.append(f)

    if not selected:
        print("nothing selected; run with --list to see the repo contents")
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "fetched_at": stamp(),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "files": [],
    }
    started = time.monotonic()
    for i, f in enumerate(selected, start=1):
        print(
            f"[{stamp()}] ({i}/{len(selected)}) get {f.path} ({human(f.size)})",
            flush=True,
        )
        local = download_file(
            f.path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            token=token,
        )
        entry = {"remote": f.path, "size": f.size}
        if f.is_archive:
            written = extract_archive(local, dest, overwrite=args.overwrite)
            entry["extracted_files"] = written
            print(f"[{stamp()}]   extracted {written} files", flush=True)
            if not args.keep_archives:
                # hf_hub_download returns a symlink into the blob store; unlink
                # the blob so a 4 GB tarball does not sit on the box twice.
                try:
                    blob = local.resolve()
                    local.unlink(missing_ok=True)
                    blob.unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            target = dest / f.path
            if target.exists() and not args.overwrite:
                entry["skipped"] = True
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local, target)
            entry["local"] = str(target.relative_to(dest))
        manifest["files"].append(entry)

    write_manifest(dest / "reports" / "fetch_manifest.json", manifest)
    print(
        f"[{stamp()}] done in {time.monotonic() - started:.0f}s. "
        f"Next: python scripts/inventory_artifacts.py",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
