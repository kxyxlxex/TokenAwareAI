#!/usr/bin/env python3
"""Pull completed Lambda artifacts to the Mac whenever you want.

Run on your Mac while the instance is still up:

  python3 scripts/pull_artifacts.py \\
    --remote ubuntu@INSTANCE_IP:~/tokenaware-data/artifacts \\
    --local /Users/kylexu/TokenAwareAI/artifacts

Copies every completed remote file. Temporary `.tmp` files are never pulled.
Root rollouts are pulled only when both the `.jsonl` and `.pt` exist, so partial
pairs cannot land locally. Uses `rsync --checksum`, so already-identical local
files are skipped by content, not by size. Safe to rerun mid-job or after the
job finishes. Does not transfer the Hugging Face model cache.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def parse_remote(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise SystemExit("--remote must look like user@host:/path")
    host, path = spec.split(":", 1)
    return host, path


def list_completed_remote_files(host: str, remote_root: str) -> list[str]:
    """Return relative paths of completed remote artifacts."""
    cmd = [
        "ssh",
        host,
        (
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            f"root = Path({remote_root!r}).expanduser().resolve()\n"
            "files = {\n"
            "    path.relative_to(root).as_posix()\n"
            "    for path in root.rglob('*')\n"
            "    if path.is_file() and not path.name.endswith('.tmp')\n"
            "}\n"
            "completed = set()\n"
            "for rel in sorted(files):\n"
            "    path = Path(rel)\n"
            "    if path.suffix == '.jsonl' and path.parts[:2] == ('rollouts', 'root'):\n"
            "        sibling = path.with_suffix('.pt').as_posix()\n"
            "        if sibling in files:\n"
            "            completed.add(rel)\n"
            "            completed.add(sibling)\n"
            "        continue\n"
            "    if path.suffix == '.pt' and path.parts[:2] == ('rollouts', 'root'):\n"
            "        continue  # added with its .jsonl pair\n"
            "    completed.add(rel)\n"
            "for rel in sorted(completed):\n"
            "    print(rel)\n"
            "PY"
        ),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--remote",
        required=True,
        help="user@host:~/tokenaware-data/artifacts",
    )
    p.add_argument(
        "--local",
        type=Path,
        default=Path("/Users/kylexu/TokenAwareAI/artifacts"),
    )
    args = p.parse_args()

    host, remote_root = parse_remote(args.remote)
    args.local.mkdir(parents=True, exist_ok=True)

    completed = list_completed_remote_files(host, remote_root)
    print(f"completed remote files: {len(completed)}", flush=True)
    if not completed:
        print("nothing to copy")
        return

    list_path = args.local / ".rsync-completed.txt"
    list_path.write_text("\n".join(completed) + "\n")
    try:
        cmd = [
            "rsync",
            "-av",
            "--partial",
            "--checksum",
            "--files-from",
            str(list_path),
            f"{host}:{remote_root.rstrip('/')}/",
            f"{args.local.as_posix().rstrip('/')}/",
        ]
        print(" ".join(shlex.quote(part) for part in cmd), flush=True)
        subprocess.run(cmd, check=True)
    finally:
        list_path.unlink(missing_ok=True)
    print("done")


if __name__ == "__main__":
    main()
