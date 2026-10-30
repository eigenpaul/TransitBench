import hashlib
import io
import json
import random
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from transitbench_env import hidden
from transitbench_env.grading import GradingError

REPO = Path(__file__).resolve().parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def synthetic_release(data_dir: Path, n: int = 30, size: int = 40_000, release: str = "syn",
                      sample: str = "X") -> dict:
    """A synthetic release under data_dir (random bytes as star files) and its manifest; never real data."""
    rng = random.Random(f"{release}-{sample}")
    built = data_dir / "built" / release / sample
    built.mkdir(parents=True)
    files = {}
    for i in range(n):
        data = rng.randbytes(size + i)
        sid = f"syn-{sample}-{i:03d}"
        (built / f"{sid}.npz").write_bytes(data)
        files[sid] = sha(data)
    truth = {"release": release, "sample": sample, "window": [10, 11, 12, 13],
             "stars": {sid: {"population": "null", "host": "random", "scored": True, "coverage": [[0.0, 1.0]],
                             "signals": []} for sid in files}}
    private = data_dir / "private" / "releases" / release
    private.mkdir(parents=True)
    truth_raw = json.dumps(truth).encode()
    (private / f"truth_{sample}.json").write_bytes(truth_raw)
    digest = hidden.files_digest(files)
    (private / f"files_{sample}.json").write_text(json.dumps(
        {"release": release, "sample": sample, "digest": digest, "files": files}))
    return {"release": release, "samples": {sample: {"stars": n, "truth_sha256": sha(truth_raw),
                                                     "star_files_sha256": digest}}}


@pytest.fixture
def release(tmp_path, monkeypatch):
    manifest = synthetic_release(tmp_path)
    monkeypatch.setattr(hidden, "release_manifest", lambda r: manifest if r == "syn" else pytest.fail(r))
    monkeypatch.setattr(hidden, "CHUNK_BYTES", 1 << 20)
    return tmp_path


def members(chunk) -> list[tuple[tarfile.TarInfo, bytes]]:
    data = chunk if isinstance(chunk, bytes) else Path(chunk).read_bytes()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        return [(m, tar.extractfile(m).read()) for m in tar]


def test_survey_set_from_bundle(release):
    (s,) = hidden.survey_sets(release, "syn", ["X"])
    assert s.name == "X" and s.in_image is None and len(s.stars) == 30 and set(s.truth["stars"]) == set(s.stars)
    assert len(s.chunks) == 2 and all(isinstance(c, Path) and c.stat().st_size <= 1 << 20 for c in s.chunks)
    got = [(m, data) for c in s.chunks for m, data in members(c)]
    assert [m.name for m, _ in got] == sorted(f"{sid}.npz" for sid in s.stars)
    for m, data in got:
        assert m.isreg() and (m.mode, m.uid, m.gid, m.mtime, m.uname, m.gname) == (0o644, 0, 0, 0, "", "")
        assert sha(data) == s.stars[m.name[:-4]]
    meta = json.loads((release / "private" / "bundles" / "syn" / "X" / "bundle.json").read_text())
    assert meta["files_digest"] == s.digest and [c["n"] for c in meta["chunks"]] == [25, 5]


def test_bundles_are_deterministic(release):
    first = [p.read_bytes() for p in hidden.survey_sets(release, "syn", ["X"])[0].chunks]
    shutil.rmtree(release / "private" / "bundles")
    again = [p.read_bytes() for p in hidden.survey_sets(release, "syn", ["X"])[0].chunks]
    assert first == again
    assert hidden.chunk_tar([("a.npz", b"1")]) == hidden.chunk_tar([("a.npz", b"1")])


def test_bundle_is_reused_then_rebuilt_on_a_bad_chunk(release):
    (chunk, _) = hidden.survey_sets(release, "syn", ["X"])[0].chunks
    inode, data = chunk.stat().st_ino, chunk.read_bytes()
    assert hidden.survey_sets(release, "syn", ["X"])[0].chunks[0].stat().st_ino == inode  # reused, not rebuilt
    chunk.write_bytes(data[:-1] + b"\x01")
    rebuilt = hidden.survey_sets(release, "syn", ["X"])[0].chunks[0]
    assert rebuilt.read_bytes() == data


def test_changed_built_file_raises(release):
    hidden.survey_sets(release, "syn", ["X"])
    star = release / "built" / "syn" / "X" / "syn-X-007.npz"
    data = star.read_bytes()
    star.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(GradingError) as e:
        hidden.survey_sets(release, "syn", ["X"])
    assert str(e.value) == "TB-E001: evaluator data missing or does not match its digest"
    assert e.value.code == "TB-E001" and e.value.__suppress_context__


def test_missing_built_file_raises(release):
    (release / "built" / "syn" / "X" / "syn-X-003.npz").unlink()
    with pytest.raises(GradingError, match="TB-E001"):
        hidden.survey_sets(release, "syn", ["X"])
    assert not (release / "private" / "bundles" / "syn" / "X").exists()
    assert not list((release / "private" / "bundles" / "syn").glob(".build-*"))


@pytest.mark.parametrize("damage", ["truth", "files", "listing", "manifest_count", "missing_truth", "sample"])
def test_release_data_must_match_the_manifest(release, monkeypatch, damage):
    private = release / "private" / "releases" / "syn"
    if damage == "truth":
        doc = json.loads((private / "truth_X.json").read_text())
        doc["window"] = [10]
        (private / "truth_X.json").write_text(json.dumps(doc))
    elif damage == "files":
        doc = json.loads((private / "files_X.json").read_text())
        doc["files"]["syn-X-000"] = "0" * 64
        (private / "files_X.json").write_text(json.dumps(doc))
    elif damage == "listing":
        doc = json.loads((private / "files_X.json").read_text())
        del doc["files"]["syn-X-000"]
        doc["digest"] = hidden.files_digest(doc["files"])
        (private / "files_X.json").write_text(json.dumps(doc))
    elif damage == "manifest_count":
        manifest = hidden.release_manifest("syn")
        manifest["samples"]["X"]["stars"] = 31
    elif damage == "missing_truth":
        (private / "truth_X.json").unlink()
    with pytest.raises(GradingError, match="TB-E001"):
        hidden.survey_sets(release, "syn", ["Y" if damage == "sample" else "X"])


def test_real_release_manifest_is_packaged_or_in_the_repo():
    manifest = hidden.release_manifest("r1")
    assert set(manifest["samples"]) == {"A", "B"}
    with pytest.raises(GradingError, match="TB-E001"):
        hidden.release_manifest("r0")


def test_dev_set():
    dev = hidden.dev_set()
    files = json.loads((REPO / "dev" / "files.json").read_text())
    assert dev.name == "dev" and dev.in_image == "/app/data/stars" and dev.chunks == ()
    assert dev.stars == files["files"] and len(dev.stars) == 196 and dev.digest == files["digest"]
    assert set(dev.truth["stars"]) == set(dev.stars)
    assert [s.name for s in hidden.survey_sets(Path("/nonexistent"), "dev", ["dev"])] == ["dev"]


def test_mini_set():
    rng = random.Random(1)
    files = {f"h-{i:02d}": rng.randbytes(150_000) for i in rng.sample(range(16), 16)}
    draw = SimpleNamespace(hidden_files=files, hidden_truth={"stars": {sid: {} for sid in files}})
    a, b = hidden.mini_set(draw), hidden.mini_set(draw)
    assert a.name == "mini" and a.in_image is None and len(a.chunks) == 1 and a.chunks == b.chunks
    assert a.stars == {sid: sha(data) for sid, data in files.items()}
    assert [(m.name, data) for m, data in members(a.chunks[0])] == [(f"{s}.npz", files[s]) for s in sorted(files)]
    with pytest.raises(ValueError):
        hidden.mini_set(SimpleNamespace(hidden_files=files, hidden_truth={"stars": {}}))


def test_pack_respects_the_chunk_size():
    sizes = [(f"s{i}", 100_000 + 37 * i) for i in range(40)]
    groups = hidden.pack(sizes, 1 << 20)
    assert [sid for g in groups for sid in g] == [sid for sid, _ in sizes]
    for g in groups:
        tar = hidden.chunk_tar([(f"{sid}.npz", b"\0" * dict(sizes)[sid]) for sid in g])
        assert len(tar) <= 1 << 20
    assert hidden.pack([("big", 5 << 20), ("small", 10)], 1 << 20) == [["big"], ["small"]]
