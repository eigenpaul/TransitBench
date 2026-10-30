import base64
import gzip
import hashlib
import io
import os
import random
import stat
import tarfile

import pytest

from transitbench import archive


def make_solution(root, files=None):
    sol = root / "app" / "solution"
    sol.mkdir(parents=True)
    for rel, body in (files or {"solution.py": "print('1,1,1')\n", "pkg/helper.py": "X = 1\n"}).items():
        (sol / rel).parent.mkdir(parents=True, exist_ok=True)
        (sol / rel).write_text(body)
    return sol


def tar_of(root):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar:
        tar.add(root / "app" / "solution", arcname="app/solution")
    return buf.getvalue()


def test_dir_and_tar_give_the_same_sha(tmp_path):
    sol = make_solution(tmp_path)
    (sol / "run.sh").write_text("echo\n")
    (sol / "run.sh").chmod(0o750)
    (sol / "empty_dir").mkdir()
    a, b = archive.from_dir(sol), archive.from_tar(tar_of(tmp_path))
    assert a.status == b.status == "ok" and a.reason is None
    assert a.sha256 == b.sha256 and a.tar == b.tar and a.files == b.files
    assert [f["path"] for f in a.files] == ["pkg/helper.py", "run.sh", "solution.py"]


def test_canonical_members(tmp_path):
    sol = make_solution(tmp_path)
    (sol / "solution.py").chmod(0o700)
    c = archive.from_dir(sol)
    with tarfile.open(fileobj=io.BytesIO(c.tar)) as tar:
        assert tar.format == tarfile.PAX_FORMAT
        members = tar.getmembers()
    assert [m.name for m in members] == ["app/solution", "app/solution/pkg", "app/solution/pkg/helper.py",
                                         "app/solution/solution.py"]
    assert {(m.uid, m.gid, m.uname, m.gname, m.mtime) for m in members} == {(0, 0, "", "", 0)}
    assert [oct(m.mode) for m in members] == ["0o755", "0o755", "0o644", "0o755"]


def test_mtimes_and_non_exec_modes_do_not_change_the_sha(tmp_path):
    a = archive.from_dir(make_solution(tmp_path / "a"))
    sol = make_solution(tmp_path / "b")
    os.utime(sol / "solution.py", (1, 1))
    (sol / "pkg" / "helper.py").chmod(0o600)
    assert archive.from_dir(sol).sha256 == a.sha256


def test_links_and_fifos_are_dropped_and_hard_links_copied(tmp_path):
    sol = make_solution(tmp_path)
    (sol / "escape").symlink_to("/")
    os.mkfifo(sol / "pipe")
    os.link(sol / "solution.py", sol / "copy.py")
    a, b = archive.from_dir(sol), archive.from_tar(tar_of(tmp_path))
    assert a.sha256 == b.sha256
    assert a.dropped == ("escape", "pipe") and b.dropped == ("escape", "pipe")
    assert [f["path"] for f in a.files] == ["copy.py", "pkg/helper.py", "solution.py"]


def test_missing_and_symlinked_roots_are_empty(tmp_path):
    missing = archive.from_dir(tmp_path / "nothing")
    assert missing.status == "empty" and missing.reason == "missing" and missing.files == ()
    (tmp_path / "link").symlink_to(tmp_path)
    assert archive.from_dir(tmp_path / "link").sha256 == missing.sha256
    assert archive.from_tar(b"").reason == "missing"


def test_empty_directory_is_empty_without_a_reason(tmp_path):
    sol = tmp_path / "solution"
    sol.mkdir()
    c = archive.from_dir(sol)
    assert c.status == "empty" and c.reason is None and c.sha256 == archive.empty(None).sha256


@pytest.mark.parametrize("files, reason", [
    ({f"f{i}.py": "" for i in range(5)}, "too_many_files"),
    ({"a.py": "x" * 60, "b.py": "x" * 60}, "too_large"),
    ({f"f{i}.py": "x" * 60 for i in range(5)}, "too_many_files"),
])
def test_limits_reject(tmp_path, files, reason):
    sol = make_solution(tmp_path, files)
    for c in (archive.from_dir(sol, max_bytes=100, max_files=4),
              archive.from_tar(tar_of(tmp_path), max_bytes=100, max_files=4)):
        assert c.status == "rejected" and c.reason == reason
        assert c.sha256 == archive.empty("missing").sha256 and c.files == ()


def test_directories_count_as_entries(tmp_path):
    sol = make_solution(tmp_path, {"a/b/c/d.py": ""})
    assert archive.from_dir(sol, max_files=3).reason == "too_many_files"
    assert archive.from_dir(sol, max_files=4).status == "ok"


def test_bad_tars_never_raise(tmp_path):
    assert archive.from_tar(b"not a tar" * 100).reason == "unreadable"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in ("app/solution/x.py", "app/solution/x.py", "etc/passwd"):
            info = tarfile.TarInfo(name)
            tar.addfile(info, io.BytesIO(b""))
    assert archive.from_tar(buf.getvalue()).reason == "unreadable"


def test_members_outside_the_root_are_dropped(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in ("app/solution/solution.py", "app/other.py", "../evil.py", "app/solution/../x.py"):
            info = tarfile.TarInfo(name)
            info.size = 3
            tar.addfile(info, io.BytesIO(b"abc"))
    c = archive.from_tar(buf.getvalue())
    assert c.status == "ok" and [f["path"] for f in c.files] == ["solution.py"] and len(c.dropped) == 3


def test_gzip_tar_and_b64gz_round_trip(tmp_path):
    sol = make_solution(tmp_path)
    c = archive.from_dir(sol)
    assert archive.from_tar(gzip.compress(tar_of(tmp_path))).sha256 == c.sha256
    assert archive.from_b64gz(archive.to_b64gz(c.tar)) == c.tar
    assert archive.to_b64gz(c.tar) == archive.to_b64gz(c.tar)


def test_b64gz_refuses_a_bomb():
    bomb = base64.b64encode(gzip.compress(b"\0" * (archive.MAX_TAR_BYTES + 1))).decode()
    with pytest.raises(ValueError):
        archive.from_b64gz(bomb)


def test_extract_round_trips_with_canonical_modes(tmp_path):
    sol = make_solution(tmp_path / "src")
    (sol / "solution.py").chmod(0o711)
    c = archive.from_dir(sol)
    out = archive.extract(c, tmp_path / "snap")
    assert out == tmp_path / "snap" / "app" / "solution"
    assert archive.from_dir(out).sha256 == c.sha256
    assert oct(out.stat().st_mode & 0o777) == "0o755"
    assert oct((out / "solution.py").stat().st_mode & 0o777) == "0o755"
    assert oct((out / "pkg" / "helper.py").stat().st_mode & 0o777) == "0o644"
    with pytest.raises(FileExistsError):
        archive.extract(c, tmp_path / "snap")


def test_empty_archive_extracts_to_an_empty_solution(tmp_path):
    out = archive.extract(archive.empty("too_large"), tmp_path)
    assert out.is_dir() and list(out.iterdir()) == []


def test_bytecode_is_dropped_so_the_code_is_its_source(tmp_path):
    src = tmp_path / "with"
    (src / "__pycache__").mkdir(parents=True)
    (src / "solution.py").write_text("print('1,1,1')\n")
    (src / "__pycache__" / "solution.cpython-312.pyc").write_bytes(b"\x00" * 64)
    (src / "helper.pyc").write_bytes(b"\x00" * 32)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "solution.py").write_text("print('1,1,1')\n")
    got, want = archive.from_dir(src), archive.from_dir(clean)
    assert got.sha256 == want.sha256 and got.status == "ok"
    assert [f["path"] for f in got.files] == ["solution.py"]
    assert set(got.dropped) == {"__pycache__", "__pycache__/solution.cpython-312.pyc", "helper.pyc"}
    assert archive.from_tar(got.tar).sha256 == want.sha256


# ---- the limits count everything, before anything is dropped -------------------------------------------------------

def build(sol, tree):
    """tree: path -> bytes (a file) or str (a symlink to that target); parent directories are made as needed."""
    sol.mkdir(parents=True, exist_ok=True)
    for rel, body in tree.items():
        (sol / rel).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, str):
            (sol / rel).symlink_to(body)
        else:
            (sol / rel).write_bytes(body)
    return sol


def assert_whole(c, sol):
    """Every file in an "ok" archive has exactly its source's bytes."""
    if c.status != "ok":
        return
    with tarfile.open(fileobj=io.BytesIO(c.tar)) as tar:
        for m in tar:
            if m.isfile():
                assert tar.extractfile(m).read() == (sol / m.name.removeprefix("app/solution/")).read_bytes(), m.name
    for f in c.files:
        body = (sol / f["path"]).read_bytes()
        assert (f["size"], f["sha256"]) == (len(body), hashlib.sha256(body).hexdigest())


SRC = b"print('1,1,1')\n"            # 15 bytes
PYC = "__pycache__/solution.cpython-312.pyc"
LIMIT_CASES = {                      # at max_bytes=1000, max_files=8
    "bytecode_tips_over": ({"solution.py": b"x" * 600, PYC: b"\0" * 401}, ("rejected", "too_large")),
    "bytes_at_limit": ({"solution.py": b"x" * 600, PYC: b"\0" * 400}, ("ok", None)),
    "entries_at_limit": ({"solution.py": SRC, **{f"__pycache__/m{i}.pyc": b"" for i in range(6)}}, ("ok", None)),
    "entries_one_over": ({"solution.py": SRC, **{f"__pycache__/m{i}.pyc": b"" for i in range(7)}},
                         ("rejected", "too_many_files")),
    "links_at_limit": ({"solution.py": SRC, **{f"l{i}": "solution.py" for i in range(8)}}, ("ok", None)),
    "links_one_over": ({"solution.py": SRC, **{f"l{i}": "solution.py" for i in range(9)}},
                       ("rejected", "too_many_files")),
    "bytecode_and_links": ({"solution.py": SRC, PYC: b"\0" * 900, "helper.pyc": b"\0" * 80,
                            **{f"l{i}": "/" for i in range(8)}}, ("ok", None)),
    "bytecode_and_links_over": ({"solution.py": SRC, **{f"__pycache__/m{i}.pyc": b"" for i in range(7)},
                                 **{f"l{i}": "/" for i in range(8)}}, ("rejected", "too_many_files")),
    # The root is read before pkg/, so a.pyc is read first: once, dropping it hid that pkg/mod.py had been cut
    # (a.pyc under the limit) or blanked (a.pyc over it), and the solution came out "ok".
    "pyc_cuts_a_later_file": ({"a.pyc": b"\0" * 950, "solution.py": SRC, "pkg/mod.py": b"y" * 100},
                              ("rejected", "too_large")),
    "pyc_blanks_a_later_file": ({"a.pyc": b"\0" * 1001, "solution.py": SRC, "pkg/mod.py": b"y" * 100},
                                ("rejected", "too_large")),
    "over_both_limits": ({"a_big.bin": b"z" * 2000, **{f"f{i}.py": b"" for i in range(9)}},
                         ("rejected", "too_many_files")),
}


@pytest.mark.parametrize("case", list(LIMIT_CASES))
def test_limits_count_what_is_dropped(tmp_path, case):
    tree, want = LIMIT_CASES[case]
    sol = build(tmp_path / "app" / "solution", tree)
    a = archive.from_dir(sol, max_bytes=1000, max_files=8)
    b = archive.from_tar(tar_of(tmp_path), max_bytes=1000, max_files=8)
    assert (a.status, a.reason) == (b.status, b.reason) == want
    assert (a.sha256, a.files, a.dropped) == (b.sha256, b.files, b.dropped)
    assert_whole(a, sol)
    if a.status == "ok":
        assert {f["path"] for f in a.files} == {r for r, body in tree.items()
                                                 if isinstance(body, bytes) and not r.endswith(".pyc")}
        assert set(a.dropped) >= {r for r in tree if r.endswith(".pyc") or isinstance(tree[r], str)}


def reference(sol, max_bytes, max_files):
    """The rule TASK.md states, counted here without archive.py: the reason, or None within the limits."""
    entries = others = size = 0
    for d, dirs, files in os.walk(sol):
        for name in dirs + files:
            st = os.lstat(os.path.join(d, name))
            if stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode):
                entries += 1
                size += st.st_size if stat.S_ISREG(st.st_mode) else 0
            else:
                others += 1
    if entries > max_files or others > max_files:
        return "too_many_files"
    return "too_large" if size > max_bytes else None


def random_tree(rng):
    tree = {}
    for i in range(rng.randrange(12)):
        size = rng.randrange(rng.choice((40, 200, 500)))
        tree[rng.choice((f"m{i}.py", f"m{i}.pyc", f"__pycache__/m{i}.cpython-312.pyc", f"pkg/s{i % 2}/m{i}.py",
                         f"pkg/__pycache__/m{i}.pyc"))] = rng.randbytes(size)
        if rng.random() < 0.3:
            tree[f"l{i}"] = rng.choice(("m0.py", "/", "pkg"))
    return tree


def test_ok_archives_hold_whole_files_and_follow_the_stated_rule(tmp_path):
    rng, seen = random.Random(7), set()
    for i in range(250):
        root = tmp_path / str(i)
        sol = build(root / "app" / "solution", random_tree(rng))
        a = archive.from_dir(sol, max_bytes=600, max_files=7)
        b = archive.from_tar(tar_of(root), max_bytes=600, max_files=7)
        want = reference(sol, 600, 7)
        assert a.reason == b.reason == want and (a.status, a.sha256, a.dropped) == (b.status, b.sha256, b.dropped)
        assert_whole(a, sol)
        assert_whole(b, sol)
        seen.add(a.status if a.status != "rejected" else a.reason)
    assert seen == {"ok", "empty", "too_large", "too_many_files"}


def test_a_cut_file_never_reaches_an_archive():
    with pytest.raises(archive._Rejected, match="too_large"):
        archive._canonical({"a.py": (archive._CUT, False)}, [], 0, 0, False, 100, 10)


def test_names_that_are_not_utf8_count_and_are_dropped(tmp_path):
    sol = make_solution(tmp_path, {"solution.py": "print(1)\n"})
    bad = os.path.join(os.fsencode(sol), b"\xff")
    try:
        os.mkdir(bad)
    except OSError:
        pytest.skip("this file system takes only UTF-8 names")
    for i in range(3):
        with open(os.path.join(bad, f"f{i}.py".encode()), "wb") as f:
            f.write(b"x" * 100)
    for c in (archive.from_dir(sol, max_bytes=1000, max_files=8),
              archive.from_tar(tar_of(tmp_path), max_bytes=1000, max_files=8)):
        assert c.status == "ok" and [f["path"] for f in c.files] == ["solution.py"] and len(c.dropped) == 4
    for kw, reason in (({"max_files": 4}, "too_many_files"), ({"max_bytes": 300}, "too_large")):
        assert archive.from_dir(sol, **kw).reason == archive.from_tar(tar_of(tmp_path), **kw).reason == reason
