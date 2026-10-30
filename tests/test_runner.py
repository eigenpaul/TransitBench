"""The runner (transitbench/runner.py). None mode runs everywhere; tests marked jail need root on Linux with
bubblewrap (run them in `docker run --privileged` with `pytest -m jail`) and are skipped elsewhere."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from transitbench import archive, runner
from transitbench.runner import RunnerError, run_catalogue, run_plan

FIXTURES = Path(__file__).parent / "fixtures"
SENTINEL = [424242.0, 42.0, 42.0]
jail = pytest.mark.jail
needs_jail = pytest.mark.skipif(not runner.jail_works(), reason="the jail needs root on Linux with bubblewrap")


def load_helper():
    spec = importlib.util.spec_from_file_location("fixture_helper", FIXTURES / "runner_fixture" / "helper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPER = load_helper()


def snapshot(tmp_path, files: dict[str, str]) -> Path:
    """A canonical, root-readable solution directory, as run_dev and grading make it."""
    src = tmp_path / "src"
    for rel, body in files.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(textwrap.dedent(body))
    return archive.extract(archive.from_dir(src), tmp_path / "snap")


def fixture_solution(tmp_path, *names: str) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for name in names:
        shutil.copy(FIXTURES / name, src / Path(name).name)
    return archive.extract(archive.from_dir(src), tmp_path / "snap")


def attack_solution(tmp_path, attack: Path) -> Path:
    src = tmp_path / f"src-{attack.stem}"
    src.mkdir()
    shutil.copy(attack, src / "solution.py")
    return archive.extract(archive.from_dir(src), tmp_path / f"snap-{attack.stem}")


def stars(tmp_path, blobs: dict[str, bytes]) -> list[dict]:
    d = tmp_path / "stars"
    d.mkdir(exist_ok=True)
    out = []
    for sid, data in blobs.items():
        (d / f"{sid}.npz").write_bytes(data)
        out.append({"id": sid, "path": str(d / f"{sid}.npz"), "sha256": hashlib.sha256(data).hexdigest()})
    return out


def plan(solution, star_list, isolation="none", **kw) -> dict:
    return {"version": 1, "solution": str(solution), "cap_s": 10.0, "workers": 2, "isolation": isolation,
            "stars": star_list, **kw}


def blob_for(mode: str, start: int = 0) -> bytes:
    """Star bytes for which the runner fixture takes the given behaviour."""
    for i in range(start, start + 10_000):
        data = f"star-{i}".encode()
        if HELPER.mode(data) == mode:
            return data
    raise AssertionError(mode)


ECHO = {"solution.py": """
    import os, stat, sys
    data = sys.stdin.buffer.read()
    fifo = stat.S_ISFIFO(os.fstat(0).st_mode)
    clean = os.environ["PYTHONHASHSEED"] == "0" and os.environ["HOME"] == os.environ["TMPDIR"] == os.getcwd()
    print(f"{len(data)},{int(fifo)},{int(clean)}")
"""}

BURN = {"solution.py": """
    import os, sys, time
    sys.stdin.buffer.read()
    def burn(seconds):
        end = time.process_time() + seconds
        while time.process_time() < end:
            pass
    pid = os.fork()
    if pid == 0:
        burn(0.6)
        os._exit(0)
    burn(0.6)
    os.waitpid(pid, 0)
    print("1,1,1")
"""}


# ---- none mode ---------------------------------------------------------------------------------------------------

def test_plan_records_in_plan_order_with_streaming(tmp_path):
    sol = snapshot(tmp_path, ECHO)
    blobs = {f"s{i}": bytes(1000 * (i + 1)) for i in range(5)}
    seen = []
    records = run_plan(plan(sol, stars(tmp_path, blobs)), seen.append)
    assert [r["star"] for r in records] == list(blobs)
    assert sorted(r["star"] for r in seen) == sorted(blobs)
    for r, (sid, data) in zip(records, blobs.items()):
        assert r["status"] == "ok" and r["isolation"] == "none" and r["rows"] == [[float(len(data)), 1.0, 1.0]]
        assert set(r) == {"star", "status", "cpu_s", "wall_s", "rows", "n_rows", "n_invalid", "isolation", "slot",
                          "leak"}
        assert r["slot"] in (0, 1) and r["leak"] == 0


def test_plan_checks_hashes_first(tmp_path):
    sol = snapshot(tmp_path, ECHO)
    star_list = stars(tmp_path, {"a": b"x", "b": b"y"})
    star_list[1]["sha256"] = "0" * 64
    seen = []
    with pytest.raises(RunnerError) as e:
        run_plan(plan(sol, star_list), seen.append)
    assert e.value.code == "hash" and seen == []


@pytest.mark.parametrize("change", [{"version": 2}, {"isolation": "box"}, {"cap_s": 0}, {"workers": 1.5},
                                    {"stars": [{"id": "a"}]}])
def test_malformed_plans_are_refused(tmp_path, change):
    with pytest.raises(ValueError):
        runner.check_plan({**plan(tmp_path, []), **change})


def test_duplicate_star_ids_are_refused(tmp_path):
    star_list = stars(tmp_path, {"a": b"x"})
    with pytest.raises(ValueError):
        runner.check_plan(plan(tmp_path, star_list + star_list))


def test_stdout_overflow_gives_no_rows(tmp_path):
    sol = snapshot(tmp_path, {"solution.py": "import sys\nprint('5,1,1')\nsys.stdout.write('x' * 70000)\n"})
    small = snapshot(tmp_path / "small", {"solution.py": "print('5,1,1')\n"})
    s = stars(tmp_path, {"a": b"x"})
    assert run_plan(plan(sol, s))[0]["status"] == "stdout_overflow"
    assert run_plan(plan(sol, s))[0]["rows"] == []
    assert run_plan(plan(sol, s, stdout_max_bytes=1 << 20))[0]["status"] == "ok"
    assert run_plan(plan(small, s))[0]["rows"] == [[5.0, 1.0, 1.0]]


def test_signal_and_exit_statuses(tmp_path):
    killed = snapshot(tmp_path / "k", {"solution.py": "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n"})
    crashed = snapshot(tmp_path / "c", {"solution.py": "print('1,1,1')\nraise SystemExit(4)\n"})
    s = stars(tmp_path, {"a": b"x"})
    assert run_plan(plan(killed, s))[0]["status"] == "signal:9"
    assert run_plan(plan(crashed, s))[0]["status"] == "exit:4"


def test_stderr_tail_when_asked(tmp_path):
    sol = snapshot(tmp_path, {"solution.py": "import sys\nsys.stderr.write('a' * 5000 + 'END')\n"})
    s = stars(tmp_path, {"a": b"x"})
    assert "stderr" not in run_plan(plan(sol, s))[0]
    assert run_plan(plan(sol, s, stderr_tail_bytes=100))[0]["stderr"] == "a" * 97 + "END"


def test_isolation_resolution_without_a_jail():
    if runner.jail_works():
        pytest.skip("the jail works here")
    assert runner.resolve_isolation("auto") == "none"
    with pytest.raises(RunnerError) as e:
        runner.resolve_isolation("jail")
    assert e.value.code == "jail"


def test_run_catalogue_is_back_compatible(tmp_path):
    sol = snapshot(tmp_path, ECHO)
    s = stars(tmp_path, {"a": b"abc", "b": b"de"})
    runs = run_catalogue(sol, {r["id"]: Path(r["path"]) for r in s}, 10, 2)
    assert list(runs) == ["a", "b"]
    assert runs["a"]["rows"][0].period == 3.0 and runs["b"]["rows"][0].period == 2.0
    assert all(r["status"] == "ok" and isinstance(r["cpu_s"], float) for r in runs.values())


def test_runner_fixture_in_none_mode(tmp_path):
    sol = fixture_solution(tmp_path, "runner_fixture/solution.py", "runner_fixture/helper.py")
    s = stars(tmp_path, {"sleep": blob_for("sleep"), "raise": blob_for("raise"), "probe": blob_for("probe")})
    records = {r["star"]: r for r in run_plan(plan(sol, s, cap_s=3.0))}
    assert records["sleep"]["status"] == "timeout" and records["sleep"]["rows"] == []
    assert records["raise"]["status"] == "exit:1"
    probe = records["probe"]
    assert probe["status"] == "ok" and len(probe["rows"]) == 2
    (flags, _, _), (_, pool_sum, _) = probe["rows"]
    assert HELPER.decode(flags)["tmpdir_writable"] and pool_sum == HELPER.POOL_SUM


def test_cpu_counts_the_solution_and_its_children(tmp_path):
    rec = run_plan(plan(snapshot(tmp_path, BURN), stars(tmp_path, {"a": b"x"}), workers=1))[0]
    assert rec["status"] == "ok" and 1.2 <= rec["cpu_s"] < 1.6


def run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "transitbench.runner", *map(str, args)], capture_output=True,
                          text=True, timeout=120, check=False)


def test_plan_cli_writes_results_and_status(tmp_path):
    sol = snapshot(tmp_path, ECHO)
    p = plan(sol, stars(tmp_path, {"a": b"abc", "b": b"de"}), out=str(tmp_path / "out"))
    (tmp_path / "plan.json").write_text(json.dumps(p))
    r = run_cli("--plan", tmp_path / "plan.json")
    assert r.returncode == 0, r.stderr
    results = (tmp_path / "out" / "results.jsonl").read_bytes()
    status = json.loads((tmp_path / "out" / "status.json").read_text())
    assert sorted(json.loads(line)["star"] for line in results.splitlines()) == ["a", "b"]
    assert status["done"] and status["error"] is None and status["planned"] == status["written"] == 2
    assert status["results_sha256"] == hashlib.sha256(results).hexdigest() and status["isolation"] == "none"
    assert status["runner_sha256"] == hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()
    assert status["beat"] >= 2 and status["version"] == 1


def test_plan_cli_reports_a_hash_error(tmp_path):
    sol = snapshot(tmp_path, ECHO)
    star_list = stars(tmp_path, {"a": b"abc"})
    star_list[0]["sha256"] = "f" * 64
    (tmp_path / "plan.json").write_text(json.dumps(plan(sol, star_list, out=str(tmp_path / "out"))))
    assert run_cli("--plan", tmp_path / "plan.json").returncode == 1
    status = json.loads((tmp_path / "out" / "status.json").read_text())
    assert status["done"] and status["error"] == "hash" and status["results_sha256"] is None


def test_directory_cli(tmp_path):
    sol = snapshot(tmp_path, ECHO)
    stars(tmp_path, {"a": b"abc", "b": b"de"})
    r = run_cli("--solution", sol, "--stars", tmp_path / "stars", "--out", tmp_path / "preds.jsonl",
                "--cap", 10, "--workers", 2, "--isolation", "none")
    assert r.returncode == 0, r.stderr
    lines = [json.loads(line) for line in (tmp_path / "preds.jsonl").read_text().splitlines()]
    assert {(x["star"], x["rows"][0][0]) for x in lines} == {("a", 3.0), ("b", 2.0)}


# ---- jail mode ---------------------------------------------------------------------------------------------------

@pytest.fixture
def hidden_dirs():
    """/app/data and /opt/tb-grade exist on the host, as in a box, so the jail has something to hide."""
    made = [d for d in (Path("/app/data"), Path("/opt/tb-grade")) if not d.exists()]
    for d in made:
        d.mkdir(parents=True)
        (d / "secret").write_text("x")
    yield
    for d in made:
        shutil.rmtree(d)


@jail
@needs_jail
def test_probe_passes():
    report = runner.probe()
    assert report["ok"], report.get("checks")


@jail
@needs_jail
def test_runner_fixture_in_the_jail(tmp_path, hidden_dirs):
    sol = fixture_solution(tmp_path, "runner_fixture/solution.py", "runner_fixture/helper.py")
    blobs = {"sleep": blob_for("sleep"), "raise": blob_for("raise")}
    blobs.update({f"probe{i}": blob_for("probe", 100 * i) for i in range(4)})
    records = {r["star"]: r for r in run_plan(plan(sol, stars(tmp_path, blobs), "jail", cap_s=3.0))}
    assert records["sleep"]["status"] == "timeout" and records["raise"]["status"] == "exit:1"
    for sid in blobs:
        assert records[sid]["isolation"] == "jail" and records[sid]["leak"] == 0
        if not sid.startswith("probe"):
            continue
        (flags, uid, _), (cpus, pool_sum, _) = records[sid]["rows"]
        assert HELPER.decode(flags) == {"tmpdir_writable": True, "tmp_writable": False, "shm_writable": True,
                                        "solution_writable": False, "data_visible": False, "grade_visible": False,
                                        "network": False}
        assert uid == runner.SLOT_UID + records[sid]["slot"] and cpus == 1 and pool_sum == HELPER.POOL_SUM


@jail
@needs_jail
def test_jail_statuses(tmp_path):
    killed = snapshot(tmp_path / "k", {"solution.py": "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n"})
    crashed = snapshot(tmp_path / "c", {"solution.py": "raise SystemExit(4)\n"})
    s = stars(tmp_path, {"a": b"x"})
    assert run_plan(plan(killed, s, "jail"))[0]["status"] == "signal:9"
    assert run_plan(plan(crashed, s, "jail"))[0]["status"] == "exit:4"


def without_cgroup(monkeypatch, memory: str) -> None:
    if memory == "rlimit":
        monkeypatch.setattr(runner, "CGROUP_ROOT", Path("/nonexistent"))
    elif runner._cgroup_slots(2, 512 << 20) is None:
        pytest.skip("no writable cgroup v2 here")


@jail
@needs_jail
@pytest.mark.parametrize("memory", ["cgroup", "rlimit"])
def test_jail_cpu_matches_none_mode(tmp_path, monkeypatch, memory):
    # Regression: without a cgroup, wait4 on bubblewrap saw none of the solution's CPU (cpu_s 0.001 for 1.5 s).
    without_cgroup(monkeypatch, memory)
    sol, s = snapshot(tmp_path, BURN), stars(tmp_path, {"a": b"x"})
    free = run_plan(plan(sol, s, "none", workers=1))[0]
    jailed = run_plan(plan(sol, s, "jail", workers=1))[0]
    assert free["status"] == jailed["status"] == "ok" and free["cpu_s"] >= 1.2
    assert abs(jailed["cpu_s"] - free["cpu_s"]) <= 0.1 * free["cpu_s"], (free, jailed)


@jail
@needs_jail
@pytest.mark.parametrize("memory", ["cgroup", "rlimit"])
def test_jail_cpu_on_timeout(tmp_path, monkeypatch, memory):
    without_cgroup(monkeypatch, memory)
    sol = snapshot(tmp_path, {"solution.py": "while True:\n    pass\n"})
    rec = run_plan(plan(sol, stars(tmp_path, {"a": b"x"}), "jail", workers=1, cap_s=2.0))[0]
    assert rec["status"] == "timeout" and 1.0 <= rec["cpu_s"] <= 2.2, rec


@jail
@needs_jail
@pytest.mark.parametrize("memory", ["cgroup", "rlimit"])
@pytest.mark.parametrize("attack", sorted((FIXTURES / "attacks").glob("*.py")), ids=lambda p: p.stem)
def test_attacks_are_contained(tmp_path, monkeypatch, hidden_dirs, attack, memory):
    without_cgroup(monkeypatch, memory)
    bad = attack_solution(tmp_path, attack)
    good = snapshot(tmp_path, {"solution.py": "import sys\nsys.stdin.buffer.read()\nprint('5,1,1')\n"})
    s = stars(tmp_path, {f"s{i}": f"star {i}".encode() for i in range(4)})
    attacked = run_plan(plan(bad, s, "jail", memory_bytes=512 << 20, cap_s=5.0))
    assert all(SENTINEL not in r["rows"] for r in attacked), attacked
    assert all(r["leak"] == 0 for r in attacked), attacked
    assert not any(runner._uid_pids(runner.SLOT_UID + k) for k in range(2))
    assert [r["status"] for r in run_plan(plan(good, s, "jail"))] == ["ok"] * 4
