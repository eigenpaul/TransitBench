"""Hidden star sets for grading.

A HiddenSet is one sample the grader runs: star id -> sha256 of its .npz file, the truth document (evaluator only),
and where the stars come from: tar chunks that grading.stage uploads into the grading box, or a directory already in
the image.

Survey test samples are read from the evaluator data directory and anchored to the committed release manifest
(manifests/release_<release>.json, packaged as transitbench_env/data/release_<release>.json): files_<S>.json must
have the manifest's star_files_sha256 and truth_<S>.json its truth_sha256. Their stars travel as a bundle:
deterministic tar chunks of at most CHUNK_BYTES under <data>/private/bundles/<release>/<S>/ with a bundle.json,
built lazily under a file lock from <data>/built/<release>/<S>/, each star file checked against files_<S>.json as it
is packed. Each load checks every chunk against bundle.json and every source file's stat against the one recorded;
any difference rebuilds the bundle, and a source file that is missing or no longer matches its sha256 raises TB-E001.

The dev set is baked into the survey image at /app/data/stars and is checked in place by grading.stage; its listing
and truth are the packaged copies of dev/files.json and dev/truth.json. A mini set is packed in memory from a draw.

Evaluator only: this module reads data/ and never goes into a box. It does blocking I/O, so call survey_sets from a
thread in async code.
"""

import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from transitbench_env.grading import GradingError

if TYPE_CHECKING:
    from transitbench_env.mini_draw import MiniDraw

CHUNK_BYTES = 16 << 20
BUNDLE_VERSION = 1
BLOCK, RECORD = 512, 20 * 512
DEV_STARS = "/app/data/stars"
DATA = Path(__file__).resolve().parent / "data"
REPO = Path(__file__).resolve().parents[1]
# Packaged file -> its source in a repo checkout, where the wheel's force-includes do not exist.
SOURCES = {"dev_files.json": "dev/files.json", "dev_truth.json": "dev/truth.json"}
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True, eq=False)
class HiddenSet:
    name: str
    stars: dict[str, str]  # star id -> sha256 of its .npz file
    truth: dict  # truth document (score.parse_truth input); evaluator only
    chunks: tuple[Path | bytes, ...]  # tar chunks of <id>.npz members, for stage() to upload
    in_image: str | None  # directory in the image that already holds <id>.npz, checked in place

    @property
    def digest(self) -> str:
        """The files digest of the set (SHA-256 of '<id> <sha256>\\n' lines sorted by id), safe to publish."""
        return files_digest(self.stars)


def survey_sets(data_dir: Path, release: str, samples: Sequence[str]) -> list[HiddenSet]:
    """The survey sets in sample order; "dev" is the in-image dev set. TB-E001 on missing data or a digest mismatch."""
    return [dev_set() if s == "dev" else _release_set(Path(data_dir), release, s) for s in samples]


def dev_set() -> HiddenSet:
    """The public dev set, checked in place in the survey image at /app/data/stars."""
    files, truth = _json(_packaged("dev_files.json")), _json(_packaged("dev_truth.json"))
    stars = _listing(files, "dev", "dev")
    if truth.get("release") != "dev" or set(truth.get("stars", ())) != set(stars):
        raise GradingError("TB-E001", "dev truth does not name the dev stars") from None
    return HiddenSet("dev", stars, truth, (), DEV_STARS)


def mini_set(draw: "MiniDraw") -> HiddenSet:
    """The 16 hidden stars of a mini draw, packed in memory into deterministic chunks."""
    files = {sid: draw.hidden_files[sid] for sid in sorted(draw.hidden_files)}
    for sid in files:
        _check_id(sid)
    if set(draw.hidden_truth["stars"]) != set(files):
        raise ValueError("mini draw: hidden truth and hidden files name different stars")
    stars = {sid: hashlib.sha256(data).hexdigest() for sid, data in files.items()}
    sizes = [(sid, len(data)) for sid, data in files.items()]
    chunks = tuple(chunk_tar([(f"{sid}.npz", files[sid]) for sid in group]) for group in pack(sizes, CHUNK_BYTES))
    return HiddenSet("mini", stars, draw.hidden_truth, chunks, None)


def release_manifest(release: str) -> dict:
    """The committed release manifest: the packaged copy, else manifests/ in a repo checkout."""
    name = f"release_{_check_id(release)}.json"
    for path in (DATA / name, REPO / "manifests" / name):
        try:
            return _json(path.read_bytes())
        except OSError:
            continue
    raise GradingError("TB-E001", f"no packaged manifest for release {release}") from None


def files_digest(stars: dict[str, str]) -> str:
    """SHA-256 of the lines '<star_id> <file sha256>\\n', sorted by star id (scripts/build.py files_digest)."""
    return hashlib.sha256("".join(f"{sid} {h}\n" for sid, h in sorted(stars.items())).encode()).hexdigest()


def pack(sizes: Sequence[tuple[str, int]], chunk_bytes: int) -> list[list[str]]:
    """Group ids, in order, into consecutive chunks whose tar (see chunk_tar) is at most chunk_bytes. A file too big
    for any chunk gets one of its own."""
    groups, current, used = [], [], 0
    for sid, size in sizes:
        member = BLOCK + _blocks(size)
        if current and _padded(used + member) > chunk_bytes:
            groups.append(current)
            current, used = [], 0
        current.append(sid)
        used += member
    if current:
        groups.append(current)
    return groups


def chunk_tar(members: Sequence[tuple[str, bytes]]) -> bytes:
    """A deterministic USTAR tar of regular files in the given order: mode 0644, uid and gid 0, no owner names,
    mtime 0."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.uid, info.gid, info.mtime = len(data), 0o644, 0, 0, 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ---- survey test samples --------------------------------------------------------------------------------------------

def _release_set(data_dir: Path, release: str, sample: str) -> HiddenSet:
    _check_id(release)
    _check_id(sample)
    entry = release_manifest(release).get("samples", {}).get(sample)
    if not isinstance(entry, dict):
        raise GradingError("TB-E001", f"release {release} has no sample {sample}") from None
    private = data_dir / "private" / "releases" / release
    stars = _listing(_json(_read(private / f"files_{sample}.json")), release, sample)
    raw = _read(private / f"truth_{sample}.json")
    truth = _json(raw)
    if files_digest(stars) != entry.get("star_files_sha256") or len(stars) != entry.get("stars"):
        raise GradingError("TB-E001", f"{release}/{sample}: file listing does not match the manifest") from None
    if hashlib.sha256(raw).hexdigest() != entry.get("truth_sha256"):
        raise GradingError("TB-E001", f"{release}/{sample}: truth does not match the manifest") from None
    if (truth.get("release"), truth.get("sample")) != (release, sample) or set(truth.get("stars", ())) != set(stars):
        raise GradingError("TB-E001", f"{release}/{sample}: truth does not name the listed stars") from None
    chunks = _bundle(data_dir, release, sample, stars)
    return HiddenSet(sample, stars, truth, chunks, None)


def _bundle(data_dir: Path, release: str, sample: str, stars: dict[str, str]) -> tuple[Path, ...]:
    """The bundle's chunk paths, after checking it under the lock and rebuilding it if anything differs."""
    parent = data_dir / "private" / "bundles" / release
    root, source = parent / sample, data_dir / "built" / release / sample
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _locked(parent / f".{sample}.lock"):
        chunks = _load(root, source, stars, CHUNK_BYTES)
        if chunks is None:
            _build(parent, root, source, release, sample, stars, CHUNK_BYTES)
            chunks = _load(root, source, stars, CHUNK_BYTES)
        if chunks is None:
            raise GradingError("TB-E001", f"{release}/{sample}: bundle changed while it was built") from None
        return chunks


def _load(root: Path, source: Path, stars: dict[str, str], chunk_bytes: int) -> tuple[Path, ...] | None:
    """The chunk paths if bundle.json describes this listing and chunk size, every source file still has its recorded
    stat and every chunk its recorded sha256; else None."""
    try:
        meta = json.loads((root / "bundle.json").read_bytes())
        if (meta["version"], meta["files_digest"], meta["chunk_bytes"]) != (BUNDLE_VERSION, files_digest(stars),
                                                                            chunk_bytes):
            return None
        if set(meta["sources"]) != set(stars) or any(_signature(source / f"{sid}.npz") != sig
                                                      for sid, sig in meta["sources"].items()):
            return None
        paths = []
        for c in meta["chunks"]:
            path = root / c["name"]
            if not SAFE.fullmatch(c["name"]) or _sha256_file(path) != c["sha256"]:
                return None
            paths.append(path)
        return tuple(paths)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _build(parent: Path, root: Path, source: Path, release: str, sample: str, stars: dict[str, str],
           chunk_bytes: int) -> None:
    """Pack the source star files, each checked against its sha256, into a fresh bundle that replaces root."""
    tmp = parent / f".build-{sample}-{uuid.uuid4().hex}"
    tmp.mkdir(mode=0o700)
    try:
        sizes, sources = [], {}
        for sid in sorted(stars):
            path = source / f"{_check_id(sid)}.npz"
            sources[sid] = _signature(path)
            if sources[sid] is None:
                raise GradingError("TB-E001", f"{release}/{sample}: a built star file is missing") from None
            sizes.append((sid, sources[sid][0]))
        chunks = []
        for i, group in enumerate(pack(sizes, chunk_bytes)):
            members = []
            for sid in group:
                data = _read(source / f"{sid}.npz")
                if hashlib.sha256(data).hexdigest() != stars[sid] or _signature(source / f"{sid}.npz") != sources[sid]:
                    raise GradingError("TB-E001", f"{release}/{sample}: a built star file does not match its "
                                                  "sha256") from None
                members.append((f"{sid}.npz", data))
            tar = chunk_tar(members)
            name = f"chunk-{i:02d}.tar"
            _write(tmp / name, tar)
            chunks.append({"name": name, "sha256": hashlib.sha256(tar).hexdigest(), "bytes": len(tar),
                           "n": len(group)})
        meta = {"version": BUNDLE_VERSION, "release": release, "sample": sample, "files_digest": files_digest(stars),
                "chunk_bytes": chunk_bytes, "chunks": chunks, "sources": sources}
        _write(tmp / "bundle.json", (json.dumps(meta, sort_keys=True, indent=1) + "\n").encode())
        old = parent / f".old-{sample}-{uuid.uuid4().hex}"
        if root.exists():
            root.rename(old)
        tmp.rename(root)
        shutil.rmtree(old, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- helpers --------------------------------------------------------------------------------------------------------

def _listing(doc: dict, release: str, sample: str) -> dict[str, str]:
    """The star -> sha256 map of a files_<S>.json document, checked against its own digest."""
    files = doc.get("files")
    ok = (doc.get("release"), doc.get("sample")) == (release, sample) and isinstance(files, dict) and files and all(
        isinstance(sid, str) and SAFE.fullmatch(sid) and isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)
        for sid, h in files.items())
    if not ok or files_digest(files) != doc.get("digest"):
        raise GradingError("TB-E001", f"{release}/{sample}: file listing is malformed or fails its digest") from None
    return dict(files)


def _packaged(name: str) -> bytes:
    for path in (DATA / name, REPO / SOURCES[name]):
        try:
            return path.read_bytes()
        except OSError:
            continue
    raise GradingError("TB-E001", f"packaged {name} is missing") from None


def _read(path: Path) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError:
        raise GradingError("TB-E001", f"cannot read {Path(path).name}") from None


def _json(raw: bytes) -> dict:
    try:
        doc = json.loads(raw)
    except ValueError:
        raise GradingError("TB-E001", "malformed JSON in evaluator data") from None
    if not isinstance(doc, dict):
        raise GradingError("TB-E001", "malformed JSON in evaluator data") from None
    return doc


def _check_id(name: str) -> str:
    if not (isinstance(name, str) and SAFE.fullmatch(name)):
        raise ValueError(f"unsafe name {name!r}")
    return name


def _signature(path: Path) -> list[int] | None:
    """[size, mtime_ns, ctime_ns, inode] of a regular file, None if it is missing or not a regular file."""
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    return [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]


def _sha256_file(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return None


def _write(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


@contextmanager
def _locked(path: Path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _blocks(size: int) -> int:
    return -(-size // BLOCK) * BLOCK


def _padded(members: int) -> int:
    """Bytes of a tar whose members take `members` bytes: two end blocks, padded to tarfile's 10240-byte record."""
    return -(-(members + 2 * BLOCK) // RECORD) * RECORD
