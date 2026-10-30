"""The canonical solution archive (docs/ENV_DESIGN.md section 2.5).

A solution directory, or a tar of one collected from a box, becomes one canonical uncompressed PAX tar whose members
sit under app/solution/: regular files and directories only, sorted, mode 0644 (0755 for directories and for files
with any exec bit), uid and gid 0, empty owner names, mtime 0. Its sha256 names the solution.

The limits come first and count everything in the solution, including what is dropped below: at most max_files
entries (files plus directories, bytecode included), at most max_files other entries (symlinks, FIFOs, sockets,
devices) and at most max_bytes of file content (bytecode included). The capture probe in transitbench_env/grading.py
counts the same way, so run_dev and grading reach the same verdict. A missing solution becomes an empty
app/solution/ ("empty"); one over a limit, or unreadable, becomes the same empty tree ("rejected"; too_many_files
outranks too_large, which outranks unreadable). Content is cut or left unread only past max_bytes, so every file in an
"ok" archive is whole.

Then, and only within the limits, the other entries, names that are not UTF-8, tar members outside app/solution and
Python bytecode (`__pycache__` directories and everything in them, `.pyc` and `.pyo` files) are dropped and listed,
so a solution's code is always its source, which the static scan reads. A hard link to a regular file earlier in the
same tar becomes a copy of that file, as a directory walk would see it. Solution content never raises.

Standard library only: this file ships into agent and grading boxes.
"""

import base64
import gzip
import hashlib
import io
import os
import stat
import tarfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

MAX_BYTES, MAX_FILES, ROOT = 1 << 20, 256, "app/solution"
MAX_TAR_BYTES = 8 << 20
BYTECODE_SUFFIXES = (".pyc", ".pyo")
DIR_MODE, FILE_MODE, EXEC_MODE = 0o755, 0o644, 0o755
READ_FLAGS = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_CUT = object()  # the body of a file not read in full, which happens only past max_bytes; never in an archive


@dataclass(frozen=True)
class Canonical:
    tar: bytes
    sha256: str
    status: Literal["ok", "empty", "rejected"]
    reason: str | None          # "missing" | "too_large" | "too_many_files" | "unreadable" | None
    files: tuple[dict, ...]     # {"path", "size", "sha256"}, path relative to app/solution
    dropped: tuple[str, ...]    # symlinks, FIFOs, devices, sockets, bytecode, bad names and stray members, relative
                                # to app/solution


class _Rejected(Exception):
    """The solution breaks a limit or cannot be read; args[0] is the reason."""


def from_dir(path: Path, *, max_bytes: int = MAX_BYTES, max_files: int = MAX_FILES) -> Canonical:
    """Canonicalise a solution directory. A symlink or non-directory at path counts as missing."""
    try:
        if not stat.S_ISDIR(os.lstat(path).st_mode):
            return empty("missing")
    except OSError:
        return empty("missing")
    try:
        return _canonical(*_walk(Path(path), max_bytes, max_files), max_bytes, max_files)
    except _Rejected as e:
        return empty(e.args[0])


def from_tar(data: bytes, *, max_bytes: int = MAX_BYTES, max_files: int = MAX_FILES) -> Canonical:
    """Canonicalise a tar holding app/solution (as `tar -cf - -C / app/solution` writes it); gzip is accepted."""
    if not data:
        return empty("missing")
    try:
        if data[:2] == b"\x1f\x8b":
            data = _gunzip(data, MAX_TAR_BYTES)
        entries, dropped, total, others, root_seen = _members(data, max_bytes, max_files)
        if not root_seen and not entries:
            return empty("missing")
        return _canonical(entries, dropped, total, others, False, max_bytes, max_files)
    except _Rejected as e:
        return empty(e.args[0])
    except (tarfile.TarError, OSError, EOFError, ValueError, zlib.error):
        return empty("unreadable")


def empty(reason: str | None) -> Canonical:
    """An empty app/solution/; status "empty" for a missing or empty solution, else "rejected"."""
    tar = _tar({})
    status = "empty" if reason in (None, "missing") else "rejected"
    return Canonical(tar, hashlib.sha256(tar).hexdigest(), status, reason, (), ())


def extract(c: Canonical, dest: Path) -> Path:
    """Write the canonical tree to dest/app/solution with its canonical modes and return that directory.

    Refuses an existing dest/app/solution, so the result holds exactly the archive.
    """
    target = Path(dest) / ROOT
    if os.path.lexists(target):
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(c.tar), mode="r:") as tar:
        for m in tar:
            parts = PurePosixPath(m.name).parts
            if parts[:2] != ("app", "solution") or ".." in parts or m.name.startswith("/"):
                raise ValueError(f"not a canonical member: {m.name!r}")
            out = Path(dest).joinpath(*parts)
            if m.isdir():
                out.mkdir(exist_ok=True)
                out.chmod(DIR_MODE)
            elif m.isfile():
                out.write_bytes(tar.extractfile(m).read())
                out.chmod(EXEC_MODE if m.mode & 0o111 else FILE_MODE)
            else:
                raise ValueError(f"not a canonical member: {m.name!r}")
    return target


def to_b64gz(tar: bytes) -> str:
    return base64.b64encode(gzip.compress(tar, compresslevel=9, mtime=0)).decode("ascii")


def from_b64gz(s: str, max_bytes: int = MAX_TAR_BYTES) -> bytes:
    return _gunzip(base64.b64decode(s, validate=True), max_bytes)


def _gunzip(data: bytes, max_bytes: int) -> bytes:
    d = zlib.decompressobj(wbits=31)
    out = d.decompress(data, max_bytes + 1)
    if len(out) > max_bytes or d.unconsumed_tail:
        raise ValueError(f"archive inflates past {max_bytes} bytes")
    if not d.eof:
        raise ValueError("truncated gzip stream")
    return out


def _walk(root: Path, max_bytes: int, max_files: int) -> tuple[dict, list, int, int, bool]:
    """Everything under root, without following links, in the capture probe's order: entries {relpath: (bytes, None
    for a directory, or _CUT; executable)}, dropped paths, bytes of file content, the number of other entries, and
    whether reading failed (which ends the walk, as it ends the probe's). Raises too_many_files at the first entry or
    other entry past max_files, where the probe stops, and reads no content past max_bytes."""
    entries, dropped, total, others = {}, [], 0, 0
    stack = [("", root)]
    try:
        while stack:
            rel, d = stack.pop()
            with os.scandir(d) as it:
                for e in it:
                    r = f"{rel}/{e.name}" if rel else e.name
                    st = e.stat(follow_symlinks=False)
                    if stat.S_ISDIR(st.st_mode):
                        entries[r] = (None, False)
                        stack.append((r, Path(e.path)))
                    elif not stat.S_ISREG(st.st_mode):
                        others += 1
                        dropped.append(_show(r))
                    elif total > max_bytes:
                        entries[r] = (_CUT, False)       # already over: only the counts still matter
                    elif (got := _read(e.path, max_bytes - total + 1)) is None:
                        others += 1                      # no longer a regular file
                        dropped.append(_show(r))
                    else:
                        total += len(got[0])
                        entries[r] = got if total <= max_bytes else (_CUT, False)
                    if len(entries) > max_files or others > max_files:
                        raise _Rejected("too_many_files")
    except OSError:
        return entries, dropped, total, others, True
    return entries, dropped, total, others, False


def _read(path: str, limit: int) -> tuple[bytes, bool] | None:
    """Up to limit bytes of a regular file, and whether it is executable; None if it is no longer a regular file."""
    fd = os.open(path, READ_FLAGS)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None
        chunks, n = [], 0
        while n < limit:
            chunk = os.read(fd, min(1 << 20, limit - n))
            if not chunk:
                break
            chunks.append(chunk)
            n += len(chunk)
        return b"".join(chunks), bool(st.st_mode & 0o111)
    finally:
        os.close(fd)


def _members(data: bytes, max_bytes: int, max_files: int) -> tuple[dict, list, int, int, bool]:
    """Entries, dropped paths, bytes of file content and the number of other entries from a tar, plus whether
    app/solution itself was a directory member. Raises too_many_files past max_files other entries and reads no
    content past max_bytes."""
    entries, dropped, sizes, total, others, root_seen = {}, [], {}, 0, 0, False
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        for m in tar:
            parts = PurePosixPath(m.name).parts
            if m.name.startswith("/") or ".." in parts or parts[:2] != ("app", "solution"):
                dropped.append(_show(m.name))
                continue
            rel = "/".join(parts[2:])
            if not rel:
                root_seen = root_seen or m.isdir()
                if not m.isdir():
                    dropped.append(".")
                continue
            if rel in entries:
                raise _Rejected("unreadable")
            if m.isdir():
                entries[rel] = (None, False)
            elif m.isfile() or (m.islnk() and _link_target(m) in sizes):
                sizes[rel] = sizes[_link_target(m)] if m.islnk() else m.size
                total += sizes[rel]
                if total > max_bytes:
                    entries[rel] = (_CUT, False)
                elif m.islnk():
                    entries[rel] = entries[_link_target(m)]
                else:
                    entries[rel] = (tar.extractfile(m).read(), bool(m.mode & 0o111))
            else:
                others += 1
                dropped.append(_show(rel))
                if others > max_files:
                    raise _Rejected("too_many_files")
    return entries, dropped, total, others, root_seen


def _bytecode(rel: str, body: bytes | None) -> bool:
    """A __pycache__ directory or anything in one, or a .pyc or .pyo file."""
    return "__pycache__" in rel.split("/") or (body is not None and rel.endswith(BYTECODE_SUFFIXES))


def _link_target(m: tarfile.TarInfo) -> str:
    parts = PurePosixPath(m.linkname).parts
    return "/".join(parts[2:]) if parts[:2] == ("app", "solution") and ".." not in parts else ""


def _canonical(entries: dict, dropped: list, total: int, others: int, error: bool, max_bytes: int,
               max_files: int) -> Canonical:
    """The canonical tar of the entries plus their parent directories. The limits are checked on everything found,
    as the capture probe counts it, before anything is dropped: too many entries outranks too many bytes, which
    outranks a read error. Then bytecode and names that are not UTF-8 are dropped, whole directories at a time."""
    for rel in list(entries):
        parts = rel.split("/")
        for i in range(1, len(parts)):
            parent = entries.setdefault("/".join(parts[:i]), (None, False))
            if parent[0] is not None:
                raise _Rejected("unreadable")      # a file used as a directory (only a crafted tar can do this)
    if len(entries) > max_files or others > max_files:
        raise _Rejected("too_many_files")
    if total > max_bytes or any(body is _CUT for body, _ in entries.values()):
        raise _Rejected("too_large")        # the second test never decides alone: a file is cut only past max_bytes
    if error:
        raise _Rejected("unreadable")
    for rel in [r for r, (body, _) in entries.items() if _bytecode(r, body) or not _valid_name(r)]:
        del entries[rel]
        dropped.append(_show(rel))
    tar = _tar(entries)
    files = tuple({"path": r, "size": len(entries[r][0]), "sha256": hashlib.sha256(entries[r][0]).hexdigest()}
                  for r in sorted(entries, key=_key) if entries[r][0] is not None)
    return Canonical(tar, hashlib.sha256(tar).hexdigest(), "ok" if files else "empty", None, files,
                     tuple(sorted(dropped)))


def _tar(entries: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        tar.addfile(_info(ROOT, None, False))
        for rel in sorted(entries, key=_key):
            body, executable = entries[rel]
            tar.addfile(_info(f"{ROOT}/{rel}", body, executable), None if body is None else io.BytesIO(body))
    return buf.getvalue()


def _info(name: str, body: bytes | None, executable: bool) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    if body is None:
        ti.type, ti.mode = tarfile.DIRTYPE, DIR_MODE
    else:
        ti.size, ti.mode = len(body), EXEC_MODE if executable else FILE_MODE
    ti.uid = ti.gid = ti.mtime = 0
    ti.uname = ti.gname = ""
    return ti


def _key(rel: str) -> list[str]:
    return rel.split("/")


def _valid_name(rel: str) -> bool:
    try:
        rel.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return "\x00" not in rel


def _show(name: str) -> str:
    return name.encode("utf-8", "backslashreplace").decode("utf-8")
