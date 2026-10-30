"""Tests for the image inputs (image/, scripts/image_context.py): the survey build context holds exactly the public
dev set, sha-checked; the base image copies no data; the image libraries are the uv.lock versions.

The context test needs the built dev set (data/built/dev) and skips without it; the others always run.
"""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "image"
SCRIPT = ROOT / "scripts" / "image_context.py"
DEV_FILES = ROOT / "dev" / "files.json"
DEV_TRUTH = ROOT / "dev" / "truth.json"
DEV_BUILT = ROOT / "data" / "built" / "dev"
LIBRARIES = ("numpy", "scipy", "astropy", "scikit-learn")

spec = importlib.util.spec_from_file_location("image_context", SCRIPT)
image_context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_context)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree(path: Path) -> set[str]:
    return {p.relative_to(path).as_posix() for p in path.rglob("*")}


def test_context_is_exactly_the_dev_set(tmp_path):
    if not DEV_BUILT.is_dir():
        pytest.skip("dev set not built (data/built/dev)")
    out = tmp_path / "context"
    (out / "dev" / "stars").mkdir(parents=True)
    (out / "dev" / "stars" / "stale.npz").write_bytes(b"left over")
    run = subprocess.run([sys.executable, "-I", str(SCRIPT), "--out", str(out)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    files = json.loads(DEV_FILES.read_text())["files"]
    assert len(files) == 196
    want = {"dev", "dev/stars", "dev/truth.json"} | {f"dev/stars/{sid}.npz" for sid in files}
    assert tree(out) == want  # 196 stars plus truth.json, nothing else; the stale file is gone
    for sid, digest in files.items():
        assert sha256((out / "dev" / "stars" / f"{sid}.npz").read_bytes()) == digest, sid
    assert (out / "dev" / "truth.json").read_bytes() == DEV_TRUTH.read_bytes()
    assert json.loads(run.stdout)["stars"] == 196


def fake_dev(tmp_path: Path, n: int = 3) -> tuple[Path, Path, Path]:
    built = tmp_path / "built"
    built.mkdir()
    files = {}
    for i in range(n):
        data = f"star {i}".encode()
        (built / f"dev-{i:03d}.npz").write_bytes(data)
        files[f"dev-{i:03d}"] = sha256(data)
    (built / "dev-999.npz").write_bytes(b"not listed")
    listing = tmp_path / "files.json"
    listing.write_text(json.dumps({"release": "dev", "sample": "dev", "digest": image_context.files_digest(files),
                                   "files": files}))
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"release": "dev", "sample": "dev", "stars": {sid: {} for sid in files}}))
    return built, listing, truth


def test_context_copies_only_listed_files(tmp_path):
    built, listing, truth = fake_dev(tmp_path)
    out = tmp_path / "out"
    image_context.build_context(out, files_json=listing, truth_json=truth, built=built, expect=3)
    assert tree(out / "dev") == {"stars", "truth.json", "stars/dev-000.npz", "stars/dev-001.npz", "stars/dev-002.npz"}


@pytest.mark.parametrize("damage", ["tamper", "missing", "symlink", "digest", "truth", "count"])
def test_context_refuses_any_mismatch(tmp_path, damage):
    built, listing, truth = fake_dev(tmp_path)
    out = tmp_path / "out"
    image_context.build_context(out, files_json=listing, truth_json=truth, built=built, expect=3)
    before = {p: p.read_bytes() for p in (out / "dev").rglob("*") if p.is_file()}
    expect = 3
    if damage == "tamper":
        (built / "dev-001.npz").write_bytes(b"star X")
    elif damage == "missing":
        (built / "dev-002.npz").unlink()
    elif damage == "symlink":
        (built / "dev-002.npz").unlink()
        (built / "dev-002.npz").symlink_to(built / "dev-001.npz")
    elif damage == "digest":
        doc = json.loads(listing.read_text())
        doc["digest"] = "0" * 64
        listing.write_text(json.dumps(doc))
    elif damage == "truth":
        doc = json.loads(truth.read_text())
        doc["stars"]["dev-003"] = {}
        truth.write_text(json.dumps(doc))
    elif damage == "count":
        expect = 196
    with pytest.raises(image_context.ContextError):
        image_context.build_context(out, files_json=listing, truth_json=truth, built=built, expect=expect)
    assert {p: p.read_bytes() for p in (out / "dev").rglob("*") if p.is_file()} == before  # old context kept
    assert not (out / ".dev.tmp").exists()


def test_base_image_copies_no_data():
    lines = [line.split() for line in (IMAGE / "Dockerfile").read_text().splitlines()]
    copies = [words for words in lines if words and words[0] in ("COPY", "ADD")]
    assert copies == [["COPY", "install.sh", "requirements.txt", "/tmp/tb-image/"]]
    assert "context" in (IMAGE / "Dockerfile.dockerignore").read_text().split()


def requirement_pins() -> dict[str, tuple[str, set[str]]]:
    pins, current = {}, None
    for line in (IMAGE / "requirements.txt").read_text().splitlines():
        if m := re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", line):
            current = m.group(1).lower()
            pins[current] = (m.group(2), set())
        elif (m := re.search(r"--hash=sha256:([0-9a-f]{64})", line)) and current:
            pins[current][1].add(m.group(1))
    return pins


def test_requirements_match_uv_lock():
    lock = {p["name"]: p for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    closure, todo = set(), list(LIBRARIES)
    while todo:
        name = todo.pop()
        if name not in closure:
            closure.add(name)
            todo += [d["name"] for d in lock[name].get("dependencies", [])]
    pins = requirement_pins()
    assert set(pins) == closure
    for name, (version, hashes) in pins.items():
        assert version == lock[name]["version"], name
        locked = {w["hash"].removeprefix("sha256:") for w in lock[name].get("wheels", [])}
        assert locked <= hashes, name  # every wheel uv.lock allows is accepted, at the same hash
