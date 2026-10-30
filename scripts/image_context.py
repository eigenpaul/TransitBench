"""Copy the public dev set into the survey image's build context, sha-checked.

Writes <out>/dev/truth.json (a copy of dev/truth.json) and <out>/dev/stars/<id>.npz for exactly the stars
listed in dev/files.json, read from data/built/dev/. Every star file must match its sha256 in dev/files.json,
the list must match files.json's digest, and truth.json must name the same stars. <out>/dev is rebuilt from
scratch, so it holds nothing else. image/Dockerfile.survey copies it to /app/data.

Run from the repo root (it reads data/, so as python -I):
    uv run python -I scripts/image_context.py [--out image/context]
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "dev" / "files.json"
TRUTH = ROOT / "dev" / "truth.json"
BUILT = ROOT / "data" / "built" / "dev"
OUT = ROOT / "image" / "context"
N_DEV = 196


class ContextError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def files_digest(files: dict[str, str]) -> str:
    """The rule in manifests/release_r1.json: SHA-256 of '<star_id> <file sha256>\\n', sorted by star id."""
    return sha256("".join(f"{sid} {h}\n" for sid, h in sorted(files.items())).encode())


def read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ContextError(f"{path.name}: missing or not a regular file")
    return path.read_bytes()


def build_context(out: Path = OUT, *, files_json: Path = FILES, truth_json: Path = TRUTH, built: Path = BUILT,
                  expect: int | None = N_DEV) -> dict:
    """Rebuild <out>/dev from the allowlist and return a summary. Raises ContextError on any mismatch."""
    listing = json.loads(files_json.read_text())
    files: dict[str, str] = listing["files"]
    if listing.get("release") != "dev" or listing.get("sample") != "dev":
        raise ContextError("files.json is not the dev listing")
    if files_digest(files) != listing["digest"]:
        raise ContextError("files.json digest does not match its file list")
    if expect is not None and len(files) != expect:
        raise ContextError(f"files.json lists {len(files)} stars, expected {expect}")
    truth_bytes = read_regular(truth_json)
    truth = json.loads(truth_bytes)
    if truth.get("release") != "dev" or set(truth["stars"]) != set(files):
        raise ContextError("truth.json does not name the stars in files.json")

    dest = out / "dev"
    tmp = out / ".dev.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "stars").mkdir(parents=True)
    try:
        for sid, want in sorted(files.items()):
            data = read_regular(built / f"{sid}.npz")
            if sha256(data) != want:
                raise ContextError(f"{sid}: sha256 does not match files.json")
            (tmp / "stars" / f"{sid}.npz").write_bytes(data)  # the bytes just checked
        (tmp / "truth.json").write_bytes(truth_bytes)
        for path in (tmp, tmp / "stars"):
            os.chmod(path, 0o755)
        for path in [*(tmp / "stars").iterdir(), tmp / "truth.json"]:
            os.chmod(path, 0o644)
        shutil.rmtree(dest, ignore_errors=True)
        tmp.rename(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"stars": len(files), "files_digest": listing["digest"], "truth_sha256": sha256(truth_bytes),
            "bytes": sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="context directory (default image/context)")
    args = parser.parse_args(argv)
    try:
        summary = build_context(args.out)
    except ContextError as e:
        print(f"image_context: {e}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
