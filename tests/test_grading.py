import asyncio
import contextlib
import hashlib
import io
import json
import os
import pickle
import random
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from transitbench import archive, runner
from transitbench_env import grading, hidden
from transitbench_env.grading import GradingError, Plan
from transitbench_env.hidden import HiddenSet

SRC = Path(__file__).resolve().parents[1]  # the repo root, where the packages live
IMAGE = "transitbench-base:local"
CAPS = {"star_cap_s": 30.0, "workers": 2, "memory_bytes": 512 << 20, "stdout_max_bytes": 65536}
# A synthetic solution: one row derived from the star's bytes, so every record can be checked on the evaluator.
SOLUTION = (b"import hashlib, sys\nd = sys.stdin.buffer.read()\nh = int(hashlib.sha256(d).hexdigest()[:8], 16)\n"
            b"print(f'{1 + h % 50}.5,{h % 7}.25,{len(d)}')\n")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def expected_row(data: bytes) -> list[float]:
    h = int(sha(data)[:8], 16)
    return [1 + h % 50 + 0.5, h % 7 + 0.25, float(len(data))]


def synthetic_stars(n: int, prefix: str = "syn", size: int = 40_000) -> dict[str, bytes]:
    rng = random.Random(prefix)
    return {f"{prefix}-{i:03d}": rng.randbytes(size + i) for i in range(n)}


def chunked_set(name: str, stars: dict[str, bytes], chunk_bytes: int = 1 << 20) -> HiddenSet:
    groups = hidden.pack([(sid, len(d)) for sid, d in sorted(stars.items())], chunk_bytes)
    chunks = tuple(hidden.chunk_tar([(f"{sid}.npz", stars[sid]) for sid in g]) for g in groups)
    return HiddenSet(name, {sid: sha(d) for sid, d in stars.items()}, {"stars": {}}, chunks, None)


# ---- errors ---------------------------------------------------------------------------------------------------------

def test_grading_error_is_fixed_text():
    e = GradingError("TB-E202", "star syn-X-001 had no record")
    assert str(e) == "TB-E202: the grader's results are missing or invalid" and e.code == "TB-E202"
    assert "syn-X" not in str(e) and "syn-X" in e.detail
    again = pickle.loads(pickle.dumps(e))
    assert (str(again), again.code, again.detail) == (str(e), e.code, e.detail)
    assert all(code.startswith("TB-E") and not any(c.isdigit() for c in text) for code, text in grading.MESSAGES.items())
    with pytest.raises(KeyError):
        GradingError("TB-E999")


# ---- results and status -----------------------------------------------------------------------------------------------

def rec(star: str, **kw) -> dict:
    return {"star": star, "status": "ok", "cpu_s": 1.5, "wall_s": 2.0, "rows": [[3.5, 1.0, 9.0], [2.0, 0.5, 4.0]],
            "n_rows": 2, "n_invalid": 0, "isolation": "none", "slot": 0, "leak": 0, **kw}


def jsonl(*records) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in records).encode()


PLAN = Plan([("A", "b"), ("dev", "a")], "0" * 64, "none")


def test_check_results_accepts_runner_records():
    data = jsonl(rec("a", status="exit:1", rows=[], n_rows=0), rec("b", n_rows=7, n_invalid=2, stderr="x",
                                                                   rows=[[1.0, 2, 3], [1.0, 2, 3], [5, 0, -1]]))
    out = grading.check_results(data, PLAN)
    assert list(out) == ["b", "a"] and out["a"]["status"] == "exit:1"
    assert grading.check_results(jsonl(rec("a", status="signal:9", rows=[], n_rows=0, leak=1), rec("b")), PLAN)


def drop(key):
    r = rec("b")
    del r[key]
    return r


BAD_RESULTS = {
    "missing star": jsonl(rec("a")),
    "duplicate star": jsonl(rec("a"), rec("a"), rec("b")),
    "unknown star": jsonl(rec("a"), rec("b"), rec("c")),
    "extra key": jsonl(rec("a"), rec("b", extra=1)),
    "missing key": jsonl(rec("a"), drop("leak")),
    "bad status": jsonl(rec("a"), rec("b", status="crashed")),
    "exit 0": jsonl(rec("a"), rec("b", status="exit:0")),
    "nan": jsonl(rec("a"), rec("b", cpu_s=float("nan"))),
    "negative time": jsonl(rec("a"), rec("b", wall_s=-1.0)),
    "bool slot": jsonl(rec("a"), rec("b", slot=True)),
    "leak 2": jsonl(rec("a"), rec("b", leak=2)),
    "isolation": jsonl(rec("a"), rec("b", isolation="jail")),
    "four rows": jsonl(rec("a"), rec("b", rows=[[1.0, 0, 1]] * 4, n_rows=4)),
    "unsorted": jsonl(rec("a"), rec("b", rows=[[1.0, 0, 1], [1.0, 0, 2]])),
    "period 0": jsonl(rec("a"), rec("b", rows=[[0.0, 0, 1]], n_rows=1)),
    "string number": jsonl(rec("a"), rec("b", rows=[["1", 0, 1]], n_rows=1)),
    "short row": jsonl(rec("a"), rec("b", rows=[[1.0, 0]], n_rows=1)),
    "n_rows": jsonl(rec("a"), rec("b", n_rows=5)),
    "rows when not ok": jsonl(rec("a"), rec("b", status="timeout")),
    "not utf-8": jsonl(rec("a"), rec("b")) + b"\xff\n",
    "no newline": jsonl(rec("a"), rec("b"))[:-1],
    "blank line": jsonl(rec("a")) + b"\n" + jsonl(rec("b")),
    "duplicate key": jsonl(rec("a")) + json.dumps(rec("b")).replace('"leak": 0', '"leak": 0, "slot": 1').encode()
    + b"\n",
    "not an object": jsonl(rec("a"), rec("b")) + b"[1]\n",
}


@pytest.mark.parametrize("case", list(BAD_RESULTS))
def test_check_results_is_strict(case):
    with pytest.raises(GradingError) as e:
        grading.check_results(BAD_RESULTS[case], PLAN)
    assert e.value.code == "TB-E202" and str(e.value) == "TB-E202: the grader's results are missing or invalid"


def status(beat=1, done=False, n=2, error=None, results=None, isolation="none"):
    return {"version": 1, "done": done, "planned": n, "written": n if done else 0, "beat": beat,
            "isolation": isolation, "image": None, "runner_sha256": "0" * 64, "error": error,
            "results_sha256": None if results is None else sha(results)}


def test_check_status():
    assert grading.check_status(json.dumps(status()).encode())["beat"] == 1
    for bad in ({**status(), "version": 2}, {**status(), "done": 1}, {**status(), "error": "oops"},
                {**status(), "beat": -1}, {**status(), "results_sha256": "x"}, [1]):
        with pytest.raises(ValueError):
            grading.check_status(json.dumps(bad).encode())


class Scripted:
    """A fake grading box that serves a script of status documents (or exceptions) and a results file."""

    def __init__(self, statuses, results=b""):
        self.statuses, self.results, self.launched = list(statuses), results, None

    async def run_background(self, argv, env, log):
        self.launched = (argv, log)

    async def read(self, path, max_bytes=None):
        if path == grading.STATUS_PATH:
            item = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            if isinstance(item, Exception):
                raise item
            return json.dumps(item).encode()
        assert path == grading.RESULTS_PATH and max_bytes == grading.RESULTS_MAX_BYTES
        return self.results

    async def run(self, argv, env):
        return SimpleNamespace(exit_code=0, stdout="runner: hash: details", stderr="")


def run_scripted(box, **kw):
    plan = Plan([("A", "b"), ("dev", "a")], "0" * 64, "none")
    return plan, asyncio.run(grading.run(box, plan, **{"poll_s": 0.001, **kw}))


def test_run_returns_checked_records():
    results = jsonl(rec("a"), rec("b"))
    box = Scripted([FileNotFoundError(), status(1), status(2), status(3, done=True, results=results)], results)
    plan, records = run_scripted(box)
    assert list(records) == ["b", "a"] and plan.status["done"] is True
    assert box.launched == ([*grading.RUNNER, "--plan", grading.PLAN_PATH], grading.LOG_PATH)


@pytest.mark.parametrize("case, statuses, code", [
    ("stale beat", [status(1)], "TB-E201"),
    ("never starts", [FileNotFoundError()], "TB-E201"),
    ("poll errors", [status(1)] + [OSError()] * 12, "TB-E201"),
    ("runner error", [status(1, done=True, error="hash")], "TB-E202"),
    ("sha mismatch", [status(1, done=True, results=b"other")], "TB-E202"),
    ("count mismatch", [status(1, done=True, n=3, results=jsonl(rec("a"), rec("b")))], "TB-E202"),
    ("isolation mismatch", [status(1, done=True, isolation="jail", results=jsonl(rec("a"), rec("b")))], "TB-E202"),
])
def test_run_failures(case, statuses, code):
    box = Scripted(statuses, jsonl(rec("a"), rec("b")))
    with pytest.raises(GradingError) as e:
        run_scripted(box, stale_s=0.05 if case != "poll errors" else 60)
    assert e.value.code == code and e.value.__suppress_context__


# ---- private records --------------------------------------------------------------------------------------------------

def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_write_private(tmp_path):
    p = grading.write_private(tmp_path, "grades", "r1/trace-1", {"b": 1, "a": [1]})
    assert p == tmp_path / "private" / "grades" / "r1" / "trace-1.json" and mode(p) == 0o600
    assert json.loads(p.read_text()) == {"a": [1], "b": 1} and mode(p.parent) == 0o700
    assert grading.write_private(tmp_path, "grades", "r1/trace-1", {"c": 2}) == p and json.loads(p.read_text()) == {
        "c": 2}
    tar = archive.empty("missing").tar
    a = grading.write_private(tmp_path, "archives", sha(tar), tar)
    assert a.name == f"{sha(tar)}.tar" and a.read_bytes() == tar and mode(a) == 0o600
    assert grading.write_private(tmp_path, "archives", sha(tar), tar) == a
    with pytest.raises(ValueError):
        grading.write_private(tmp_path, "archives", "0" * 64, tar)
    log = grading.write_private(tmp_path, "logs", "trace-1", {"code": "TB-E202"})
    grading.write_private(tmp_path, "logs", "trace-1", b"raw detail\n")
    assert log.read_bytes() == b'{"code":"TB-E202"}\nraw detail\n'
    assert not list(tmp_path.rglob("*.tmp"))
    for kind, name in [("grades", "../x"), ("grades", "a/b/c"), ("logs", ""), ("logs", ".hidden"), ("logs", "a b"),
                       ("secrets", "x")]:
        with pytest.raises(ValueError):
            grading.write_private(tmp_path, kind, name, b"x")


# ---- a fake box on this machine -------------------------------------------------------------------------------------

class Box:
    """A fake runtime whose /app and /opt/tb-grade live under root on this machine, with python3 = this interpreter.
    It rewrites those paths in argv and in written JSON, so the real in-box scripts and the real runner run here."""

    def __init__(self, root: Path):
        self.root, self.procs, self.json = root, [], {}
        (root / "app").mkdir(parents=True, exist_ok=True)

    def map(self, s: str) -> str:
        return s.replace("/opt/tb-grade", f"{self.root}/opt/tb-grade").replace("/app/", f"{self.root}/app/")

    def argv(self, argv: list[str]) -> list[str]:
        argv = [sys.executable if a == "python3" else self.map(a) for a in argv]
        return argv[2:] if argv[0] == "timeout" else argv

    async def run(self, argv, env):
        p = await asyncio.create_subprocess_exec(*self.argv(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                 cwd=self.root / "app")
        out, err = await p.communicate()
        return SimpleNamespace(exit_code=p.returncode, stdout=out.decode(), stderr=err.decode())

    async def run_background(self, argv, env, log):
        with open(self.map(log), "wb") as f:  # noqa: ASYNC230 - a local fake
            self.procs.append(await asyncio.create_subprocess_exec(*self.argv(argv), stdout=f, stderr=f,
                                                                   cwd=self.root / "app"))

    async def write(self, path, data):
        if path.endswith(".json"):
            self.json[path] = data
            data = self.map(data.decode()).encode()
        target = Path(self.map(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def read(self, path, max_bytes=None):
        data = Path(self.map(path)).read_bytes()
        if max_bytes is not None and len(data) > max_bytes:
            raise RuntimeError("over the byte limit")
        return data


@pytest.fixture
def box(tmp_path):
    b = Box(tmp_path / "box")
    yield b
    for p in b.procs:
        with contextlib.suppress(ProcessLookupError):
            p.kill()


def test_stage_and_run_in_a_fake_box(box):
    staged = synthetic_stars(30)
    in_image = synthetic_stars(3, prefix="img", size=1000)
    images = box.root / "app" / "data" / "stars"
    images.mkdir(parents=True)
    for sid, data in in_image.items():
        (images / f"{sid}.npz").write_bytes(data)
    (box.root / "app" / "solution").mkdir()
    (box.root / "app" / "solution" / "solution.py").write_bytes(SOLUTION)
    sets = [chunked_set("X", staged),
            HiddenSet("dev", {sid: sha(d) for sid, d in in_image.items()}, {"stars": {}}, (), "/app/data/stars")]
    assert len(sets[0].chunks) == 2

    plan = asyncio.run(grading.stage(box, sets, CAPS, require_jail=False, order_key="k" * 64))
    order = sorted(list(staged) + list(in_image), key=lambda sid: sha(f"{'k' * 64}{sid}".encode()))
    assert [sid for _, sid in plan.stars] == order and plan.isolation in ("none", "jail")
    assert {name for name, _ in plan.stars} == {"X", "dev"}
    written = json.loads(box.json[grading.PLAN_PATH])
    assert plan.digest == sha(box.json[grading.PLAN_PATH])
    assert (written["cap_s"], written["workers"], written["memory_bytes"], written["stdout_max_bytes"],
            written["solution"], written["out"]) == (30.0, 2, 512 << 20, 65536, "/app/solution", "/opt/tb-grade/out")
    assert written["stars"][0]["path"].startswith("/opt/tb-grade/stars/X/" if order[0] in staged else "/app/data/")
    grade = box.root / "opt" / "tb-grade"
    assert not list((grade / "in").iterdir())
    assert mode(grade / "stars" / "X") == 0o700 and {mode(p) for p in (grade / "stars" / "X").iterdir()} == {0o600}

    records = asyncio.run(grading.run(box, plan, poll_s=0.2))
    assert list(records) == order and plan.status["done"] and plan.status["results_sha256"]
    for sid, data in {**staged, **in_image}.items():
        assert records[sid]["status"] == "ok" and records[sid]["rows"] == [expected_row(data)]


def test_stage_refuses_a_corrupt_star(box):
    stars = synthetic_stars(4)
    s = chunked_set("X", stars)
    bad = HiddenSet("X", {**s.stars, "syn-002": "0" * 64}, {}, s.chunks, None)
    with pytest.raises(GradingError) as e:
        asyncio.run(grading.stage(box, [bad], CAPS, require_jail=False, order_key=""))
    assert e.value.code == "TB-E103"


def test_stage_refuses_a_chunk_damaged_in_transit(box):
    class Lossy(Box):
        async def write(self, path, data):
            await super().write(path, data[:-1] + b"\x01" if path.endswith(".tar") else data)

    lossy = Lossy(box.root)
    with pytest.raises(GradingError) as e:
        asyncio.run(grading.stage(lossy, [chunked_set("X", synthetic_stars(3))], CAPS, require_jail=False,
                                  order_key=""))
    assert e.value.code == "TB-E103"


def test_stage_requires_the_jail_when_asked(box):
    if runner.jail_works():
        pytest.skip("the jail works here")
    with pytest.raises(GradingError) as e:
        asyncio.run(grading.stage(box, [chunked_set("X", synthetic_stars(2))], CAPS, require_jail=True, order_key=""))
    assert e.value.code == "TB-E102"


def test_stage_needs_the_runner(box):
    class NoRunner(Box):
        async def run(self, argv, env):
            if "--probe" in argv:
                return SimpleNamespace(exit_code=1, stdout="", stderr="No module named transitbench")
            return await super().run(argv, env)

    with pytest.raises(GradingError) as e:
        asyncio.run(grading.stage(NoRunner(box.root), [chunked_set("X", synthetic_stars(2))], CAPS,
                                  require_jail=False, order_key=""))
    assert e.value.code == "TB-E103"


def test_stage_rejects_bad_sets(box):
    s = chunked_set("X", synthetic_stars(2))
    for sets in ([s, chunked_set("Y", synthetic_stars(2))],                 # the same ids twice
                 [HiddenSet("X", s.stars, {}, s.chunks, "/app/data/stars")],  # chunks and in_image
                 [s, s], [], [HiddenSet("../X", s.stars, {}, s.chunks, None)]):
        with pytest.raises(ValueError):
            asyncio.run(grading.stage(box, sets, CAPS, require_jail=False, order_key=""))


# ---- capture -------------------------------------------------------------------------------------------------------

def tree_tar(sol: Path) -> bytes:
    """A GNU tar of a solution directory, as app/solution."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar:
        tar.add(sol, arcname="app/solution")
    return buf.getvalue()


def local_collect(box: Box, calls: list):
    """verifiers' collect, emulated with this machine's tarfile (the Docker test below runs the real one)."""

    async def collect(runtime, cap):
        calls.append(cap)
        data = tree_tar(box.root / "app" / "solution")
        if len(data) > cap:
            raise RuntimeError("over the collection cap")
        return data

    return collect


def make_solution(root: Path, case: str) -> None:
    sol = root / "app" / "solution"
    if case == "missing":
        return
    if case == "file":
        sol.write_text("print(1)\n")
        return
    if case == "root_symlink":
        sol.symlink_to("/")
        return
    sol.mkdir()
    (sol / "solution.py").write_bytes(SOLUTION)
    if case == "oversize":
        (sol / "big.bin").write_bytes(os.urandom(2 << 20))
    elif case == "sparse":
        with open(sol / "sparse.bin", "wb") as f:
            f.truncate(10 << 30)
    elif case == "300_files":
        for i in range(300):
            (sol / f"f{i}.py").write_text(f"X = {i}\n")
    elif case == "300_links":
        for i in range(300):
            (sol / f"l{i}").symlink_to("solution.py")
    elif case == "symlink":
        (sol / "link").symlink_to("/etc/passwd")
    elif case == "fifo":
        os.mkfifo(sol / "pipe")


CAPTURE_CASES = {
    "missing": ("empty", "missing", False),
    "file": ("empty", "missing", False),
    "root_symlink": ("empty", "missing", False),
    "oversize": ("rejected", "too_large", False),
    "sparse": ("rejected", "too_large", False),
    "300_files": ("rejected", "too_many_files", False),
    "300_links": ("rejected", "too_many_files", False),
    "symlink": ("ok", None, True),
    "fifo": ("ok", None, True),
    "ok": ("ok", None, True),
}


@pytest.mark.parametrize("case", list(CAPTURE_CASES))
def test_capture_in_a_fake_box(box, monkeypatch, case):
    calls = []
    monkeypatch.setattr(grading, "_collect", local_collect(box, calls))
    make_solution(box.root, case)
    got = asyncio.run(grading.capture_solution(box, max_bytes=1 << 20, max_files=256))
    want_status, want_reason, collected = CAPTURE_CASES[case]
    assert (got.status, got.reason, bool(calls)) == (want_status, want_reason, collected)
    if got.status == "ok":
        assert [f["path"] for f in got.files] == ["solution.py"]
        assert got.sha256 == archive.from_dir(box.root / "app" / "solution").sha256
        assert calls == [6 << 20]


def failing(box_cls, *, probe=True, collect=True, alive=True):
    class Failing(box_cls):
        async def run(self, argv, env):
            if argv == ["true"]:
                if not alive:
                    raise ConnectionError("box gone")
            elif not probe:
                return SimpleNamespace(exit_code=127, stdout="", stderr="python3: not found")
            return await super().run(argv, env)

    return Failing


@pytest.mark.parametrize("probe, alive, outcome", [
    (True, True, "TB-E203"),     # the box reads fine and the solution is within limits: infrastructure
    (False, True, "unreadable"),  # the box runs, but its own tools fail on the solution: agent content
    (False, False, "TB-E203"),   # the box is gone
])
def test_capture_when_collection_fails(box, monkeypatch, probe, alive, outcome):
    async def broken(runtime, cap):
        raise RuntimeError("failed to collect artifact")

    monkeypatch.setattr(grading, "_collect", broken)
    make_solution(box.root, "ok")
    fake = failing(Box, probe=probe, alive=alive)(box.root)
    if outcome.startswith("TB-"):
        with pytest.raises(GradingError) as e:
            asyncio.run(grading.capture_solution(fake, max_bytes=1 << 20, max_files=256))
        assert e.value.code == outcome and e.value.__suppress_context__
    else:
        got = asyncio.run(grading.capture_solution(fake, max_bytes=1 << 20, max_files=256))
        assert (got.status, got.reason) == ("rejected", outcome)


def test_capture_rejects_a_solution_that_grew_past_the_limits(box, monkeypatch):
    make_solution(box.root, "ok")

    async def grow(runtime, cap):
        (box.root / "app" / "solution" / "big.bin").write_bytes(os.urandom(2 << 20))
        raise RuntimeError("over the collection cap")

    monkeypatch.setattr(grading, "_collect", grow)
    got = asyncio.run(grading.capture_solution(box, max_bytes=1 << 20, max_files=256))
    assert (got.status, got.reason) == ("rejected", "too_large")


def test_capture_collects_when_the_probe_cannot_run(box, monkeypatch):
    calls = []
    monkeypatch.setattr(grading, "_collect", local_collect(box, calls))
    make_solution(box.root, "ok")
    got = asyncio.run(grading.capture_solution(failing(Box, probe=False)(box.root), max_bytes=1 << 20,
                                               max_files=256))
    assert got.status == "ok" and calls


def test_probe_counts_like_the_archive(tmp_path):
    sol = tmp_path / "app" / "solution"
    (sol / "pkg" / "sub").mkdir(parents=True)
    (sol / "solution.py").write_bytes(b"x" * 700)
    (sol / "pkg" / "sub" / ("n" * 120 + ".py")).write_bytes(b"y")
    (sol / "link").symlink_to("t" * 150)
    out = subprocess.run([sys.executable, "-I", "-S", "-c", grading.PROBE, str(sol), "10"],
                         capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    assert (r["kind"], r["entries"], r["others"], r["bytes"], r["error"]) == ("dir", 4, 1, 701, False)
    c = archive.from_dir(sol)
    assert len(c.files) + 2 == r["entries"] and c.dropped == ("link",)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar:
        tar.add(sol, arcname=str(sol)[1:])
    assert r["tar"] >= len(buf.getvalue().rstrip(b"\0")) + 1024


# ---- parity: run_dev (archive.from_dir), the capture probe and archive.from_tar agree --------------------------------

MiB, LIMITS = 1 << 20, {"max_bytes": 1 << 20, "max_files": 256}
PARITY = {  # tree: path -> bytes (a file) or str (a symlink to that target); want (status, reason)
    "bytecode_tips_over": ({"solution.py": SOLUTION + b"#" * (700 << 10),
                            "__pycache__/solution.cpython-312.pyc": b"\0" * (400 << 10)}, ("rejected", "too_large")),
    "at_every_limit": ({"solution.py": SOLUTION, **{f"__pycache__/m{i}.pyc": b"\0" * 1000 for i in range(253)},
                        "data.txt": b"d" * (MiB - len(SOLUTION) - 253_000),
                        **{f"l{i}": "solution.py" for i in range(256)}}, ("ok", None)),
    "one_byte_over": ({"solution.py": SOLUTION, "__pycache__/m.pyc": b"\0" * (MiB - len(SOLUTION) + 1)},
                      ("rejected", "too_large")),
    "one_entry_over": ({"solution.py": SOLUTION, **{f"__pycache__/m{i}.pyc": b"" for i in range(255)}},
                       ("rejected", "too_many_files")),
    "300_links": ({"solution.py": SOLUTION, **{f"l{i}": "solution.py" for i in range(300)}},
                  ("rejected", "too_many_files")),
    "bytecode_and_links": ({"solution.py": SOLUTION, **{f"__pycache__/m{i}.pyc": b"\0" * 4000 for i in range(100)},
                            **{f"l{i}": "/etc/passwd" for i in range(200)}}, ("ok", None)),
    "bytecode_and_links_over": ({"solution.py": SOLUTION, **{f"__pycache__/m{i}.pyc": b"" for i in range(250)},
                                 **{f"l{i}": "/" for i in range(257)}}, ("rejected", "too_many_files")),
    # The root is read before pkg/, so a.pyc comes first: once, dropping it hid that pkg/mod.py had been cut (a.pyc
    # under the limit) or blanked (a.pyc over it), and run_dev ran a solution grading rejects.
    "pyc_cuts_a_later_file": ({"a.pyc": b"\0" * (MiB - 300), "solution.py": SOLUTION, "pkg/mod.py": b"y" * 400},
                              ("rejected", "too_large")),
    "pyc_blanks_a_later_file": ({"a.pyc": b"\0" * (MiB + 1), "solution.py": SOLUTION, "pkg/mod.py": b"y" * 400},
                                ("rejected", "too_large")),
    "over_both_limits": ({"a_big.bin": b"z" * (2 * MiB), **{f"f{i}.py": b"" for i in range(300)}},
                         ("rejected", "too_many_files")),
}


def build_tree(sol: Path, tree: dict) -> None:
    sol.mkdir(parents=True)
    for rel, body in tree.items():
        (sol / rel).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, str):
            (sol / rel).symlink_to(body)
        else:
            (sol / rel).write_bytes(body)


def assert_whole(c: archive.Canonical, sol: Path) -> None:
    """Every file in an "ok" archive has exactly its source's bytes."""
    if c.status == "ok":
        with tarfile.open(fileobj=io.BytesIO(c.tar)) as tar:
            for m in tar:
                if m.isfile():
                    assert tar.extractfile(m).read() == (sol / m.name.removeprefix("app/solution/")).read_bytes()


def assert_same_verdict(box: Box, monkeypatch, want: tuple[str, str | None]) -> tuple[dict, archive.Canonical]:
    """The probe plus _verdict, capture_solution, from_dir (run_dev) and from_tar of a GNU tar of the box's solution
    all reach want, with the same archive; every "ok" archive holds whole files. Returns the probe's report and the
    from_dir archive."""
    sol, calls = box.root / "app" / "solution", []
    monkeypatch.setattr(grading, "_collect", local_collect(box, calls))
    report = asyncio.run(grading._probe(box, LIMITS["max_files"]))
    probe = grading._verdict(report, LIMITS["max_bytes"], LIMITS["max_files"],
                             LIMITS["max_bytes"] + grading.COLLECT_SLACK)
    captured = asyncio.run(grading.capture_solution(box, **LIMITS))
    run_dev = archive.from_dir(sol, **LIMITS)
    tarred = archive.from_tar(tree_tar(sol), **LIMITS)
    assert probe == want[1] and calls == ([6 << 20] if want[0] == "ok" else [])
    for c in (captured, run_dev, tarred):
        assert (c.status, c.reason) == want
        assert (c.sha256, c.files, c.dropped) == (run_dev.sha256, run_dev.files, run_dev.dropped)
        assert_whole(c, sol)
    return report, run_dev


@pytest.mark.parametrize("case", list(PARITY))
def test_run_dev_and_grading_reach_the_same_verdict(box, monkeypatch, case):
    tree, want = PARITY[case]
    build_tree(box.root / "app" / "solution", tree)
    _, run_dev = assert_same_verdict(box, monkeypatch, want)
    if want[0] == "ok":
        assert {f["path"] for f in run_dev.files} <= {"solution.py", "data.txt"}


def test_long_names_within_the_limits_are_collected(box, monkeypatch):
    """250 symlinks with 4,000-byte targets, 1 MiB of content, and files at the end of a chain of 255-byte directory
    names whose paths reach PATH_MAX: within every limit, but over the old 2 MiB collection cap."""
    sol = box.root / "app" / "solution"
    sol.mkdir()
    if os.pathconf(sol, "PC_PATH_MAX") < 4096:
        pytest.skip("needs Linux path and link lengths")
    room = 4095 - len(os.fsencode(sol)) - 1   # a relative path this long makes the longest path the probe can stat
    dirs = ["d" * 255] * -(-(room - 255) // 256)
    name = room - 256 * len(dirs)                # 0 to 255 bytes; borrow from the last directory if under 8
    if name < 8:
        dirs[-1], name = "d" * (255 - 8 + name), 8
    deep = "/".join(dirs)
    tree = {"solution.py": SOLUTION, "data.txt": b"d" * (MiB - len(SOLUTION) - 400),
            **{f"{deep}/{str(i) * name}": b"y" * 100 for i in range(4)},
            **{f"l{i}": "t" * 4000 for i in range(250)}}
    sol.rmdir()
    build_tree(sol, tree)
    assert {len(os.fsencode(p)) for p in (sol / deep).iterdir()} == {4095}
    report, run_dev = assert_same_verdict(box, monkeypatch, ("ok", None))
    assert report["others"] == 250 and report["bytes"] == MiB and report["tar"] > 2 * MiB
    assert len(run_dev.files) == 6 and len(run_dev.dropped) == 250


def test_the_collection_cap_covers_every_solution_within_the_limits():
    name = 512 + (4095 + 512) // 512 * 512  # the GNU long-name blocks for the longest path or link target
    worst = 1536 + archive.MAX_FILES * (512 + name + 511) + archive.MAX_FILES * (512 + 2 * name) + archive.MAX_BYTES
    assert worst == 4_982_016 and worst + (1 << 20) < archive.MAX_BYTES + grading.COLLECT_SLACK
    assert archive.MAX_BYTES + grading.COLLECT_SLACK <= archive.MAX_TAR_BYTES


def test_the_limits_outrank_an_unreadable_directory(box, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip("root reads every directory")
    monkeypatch.setattr(grading, "_collect", local_collect(box, []))
    sol = box.root / "app" / "solution"
    build_tree(sol, {"solution.py": SOLUTION, "locked/x.py": b"x"})
    (sol / "locked").chmod(0)
    try:
        for extra, want in ((b"", "unreadable"), (b"b" * (2 * MiB), "too_large")):
            (sol / "big.bin").write_bytes(extra)
            report = asyncio.run(grading._probe(box, LIMITS["max_files"]))
            assert report["error"] and grading._verdict(report, LIMITS["max_bytes"], LIMITS["max_files"],
                                                        LIMITS["max_bytes"] + grading.COLLECT_SLACK) == want
            for c in (asyncio.run(grading.capture_solution(box, **LIMITS)), archive.from_dir(sol, **LIMITS)):
                assert (c.status, c.reason) == ("rejected", want)
    finally:
        (sol / "locked").chmod(0o755)


# ---- Docker: verifiers' Docker runtime and the base image -----------------------------------------------------------

def docker_skip() -> str | None:
    try:
        import verifiers.v1  # noqa: F401
    except ImportError:
        return "needs verifiers (the env extra)"
    if shutil.which("docker") is None:
        return "needs docker"
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=False).returncode != 0:
        return f"needs the {IMAGE} image (built as in image/Dockerfile)"
    return None


@contextlib.asynccontextmanager
async def docker_box():
    import verifiers.v1 as vf
    from verifiers.v1.runtimes import provision_runtime

    async with asyncio.timeout(600), provision_runtime(vf.DockerConfig(image=IMAGE, workdir="/app", cpu=2,
                                                                       memory=1)) as runtime:
        yield runtime


async def put_science(runtime) -> None:
    """What the task's setup writes into every box (box.put_files): the transitbench package."""
    for name in ("__init__", "archive", "match", "score", "runner"):
        await runtime.write(f"/app/transitbench/{name}.py", (SRC / "transitbench" / f"{name}.py").read_bytes())


@pytest.mark.skipif(docker_skip() is not None, reason=str(docker_skip()))
def test_docker_stage_and_grade_30_synthetic_stars(tmp_path, monkeypatch):
    from verifiers.v1.utils.artifacts import restore

    stars = synthetic_stars(30)
    data_dir = tmp_path / "data"
    built = data_dir / "built" / "syn" / "X"
    built.mkdir(parents=True)
    for sid, data in stars.items():
        (built / f"{sid}.npz").write_bytes(data)
    files = {sid: sha(d) for sid, d in stars.items()}
    truth = json.dumps({"release": "syn", "sample": "X", "stars": {sid: {} for sid in stars}}).encode()
    private = data_dir / "private" / "releases" / "syn"
    private.mkdir(parents=True)
    (private / "truth_X.json").write_bytes(truth)
    (private / "files_X.json").write_text(json.dumps({"release": "syn", "sample": "X", "files": files,
                                                      "digest": hidden.files_digest(files)}))
    manifest = {"samples": {"X": {"stars": 30, "truth_sha256": sha(truth),
                                  "star_files_sha256": hidden.files_digest(files)}}}
    monkeypatch.setattr(hidden, "release_manifest", lambda release: manifest)
    monkeypatch.setattr(hidden, "CHUNK_BYTES", 1 << 20)
    (sets,) = [hidden.survey_sets(data_dir, "syn", ["X"])]
    assert len(sets[0].chunks) == 2

    async def main():
        async with docker_box() as rt:
            await put_science(rt)
            await rt.write("/app/solution/solution.py", SOLUTION)
            cap = await grading.capture_solution(rt, max_bytes=1 << 20, max_files=256)
            assert cap.status == "ok" and [f["path"] for f in cap.files] == ["solution.py"]
            await rt.run(["rm", "-rf", "/app/solution"], {})
            await restore(rt, {"/app/solution": cap.tar})  # as verifiers restores it in the grading box
            plan = await grading.stage(rt, sets, CAPS, require_jail=False, order_key=cap.sha256)
            listing = ("ls -A /opt/tb-grade/in | wc -l; stat -c %a /opt/tb-grade /opt/tb-grade/stars/X; "
                       "stat -c %a /opt/tb-grade/stars/X/* | sort -u")
            check = await rt.run(["sh", "-c", listing], {})
            records = await grading.run(rt, plan, poll_s=0.5)
            return plan, check.stdout.split(), records

    plan, check, records = asyncio.run(main())
    assert plan.isolation == "none" and len(plan.stars) == 30
    assert check == ["0", "700", "700", "600"]
    assert len(records) == 30 and set(records) == set(stars)
    for sid, data in stars.items():
        assert records[sid]["status"] == "ok" and records[sid]["isolation"] == "none"
        assert records[sid]["rows"] == [expected_row(data)]
    assert plan.status["done"] and plan.status["image"] and plan.status["runner_sha256"] == sha(
        (SRC / "transitbench" / "runner.py").read_bytes())


DOCKER_CASES = {
    "missing": ("", ("empty", "missing")),
    "oversize": ("mkdir /app/solution && head -c 2097152 /dev/urandom > /app/solution/big.bin",
                 ("rejected", "too_large")),
    "300_files": ("mkdir /app/solution && for i in $(seq 300); do echo x=$i > /app/solution/f$i.py; done",
                  ("rejected", "too_many_files")),
    "symlink": (("mkdir /app/solution && echo 'print(1)' > /app/solution/solution.py && "
                 "ln -s /etc/passwd /app/solution/link && ln -s / /app/solution/root"), ("ok", None)),
    "fifo": (("mkdir /app/solution && echo 'print(1)' > /app/solution/solution.py && mkfifo /app/solution/pipe && "
              "python3 -c \"import socket; socket.socket(socket.AF_UNIX).bind('/app/solution/sock')\""), ("ok", None)),
    "root_symlink": ("ln -s / /app/solution", ("empty", "missing")),
    "file": ("echo 'print(1)' > /app/solution", ("empty", "missing")),
    "sparse": ("mkdir /app/solution && truncate -s 10G /app/solution/sparse.bin", ("rejected", "too_large")),
    "empty": ("mkdir /app/solution", ("empty", None)),
}


@pytest.mark.skipif(docker_skip() is not None, reason=str(docker_skip()))
def test_docker_capture_never_raises_on_agent_content():
    async def main():
        out = {}
        async with docker_box() as rt:
            for case, (command, _) in DOCKER_CASES.items():
                await rt.run(["sh", "-c", f"rm -rf /app/solution; {command}"], {})
                out[case] = await grading.capture_solution(rt, max_bytes=1 << 20, max_files=256)
        return out

    got = asyncio.run(main())
    for case, (_, want) in DOCKER_CASES.items():
        assert (got[case].status, got[case].reason) == want, case
    assert [f["path"] for f in got["symlink"].files] == ["solution.py"] and got["symlink"].dropped == ("link", "root")
    assert [f["path"] for f in got["fifo"].files] == ["solution.py"]
