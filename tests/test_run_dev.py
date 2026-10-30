"""run_dev.py, the agent's front end to the grading runner (transitbench/run_dev.py).

The fast tests build a box in a temporary directory from box.box_files, with a few synthetic stars and a fake
solution, and call run_dev.main in this process. The last test runs the box's own run_dev.py on 10 dev stars with
the starter, in none mode; it needs the built dev set (data/built/dev, git-ignored) and skips without it.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from transitbench import run_dev, score
from transitbench_env import box

REPO = Path(__file__).resolve().parents[1]
DEV_STARS = REPO / "data" / "built" / "dev"
CAPS = {"preset": "survey", "star_cap_s": 10.0, "command_cap_s": 60.0, "workers": 2, "max_turns": 60,
        "max_tokens": 200_000, "wall_s": 14_400, "solution_max_bytes": 1 << 20, "solution_max_files": 256,
        "stdout_max_bytes": 1 << 16, "memory_bytes": 2 << 30, "hidden": {"A": 1000, "B": 1000, "dev": 196},
        "labelled": 3, "libraries": {"numpy": "2.5.4"}}
COVERAGE = [[0.0, 100.0]]
TRUTH = {"release": "dev", "sample": "dev", "window": [10, 11, 12, 13], "stars": {
    "s-0": {"population": "inj", "host": "random", "scored": True, "coverage": COVERAGE,
            "signals": [{"period": 10.0, "t0": 5.0, "duration": 3.0, "scored": True, "kind": "injected",
                         "mes": 12.0}]},
    "s-1": {"population": "null", "host": "random", "scored": True, "coverage": COVERAGE, "signals": []},
    "s-2": {"population": "null", "host": "tce_only", "scored": True, "coverage": COVERAGE, "signals": []}}}
STARS = {"s-0": b"inj", "s-1": b"null-1", "s-2": b"null-2"}
# Claims the injected planet on s-0, prints a false row on star bytes ending in "!", crashes on "crash", sleeps
# past the cap on "slow" and prints nothing elsewhere.
FAKE = textwrap.dedent("""
    import sys, time
    data = sys.stdin.buffer.read()
    if data == b"inj":
        print("10.0,5.0,9.0")
        print("not a row")
    elif data.endswith(b"!"):
        print("3.3,1.0,1.0")
    elif data == b"crash":
        print("1.0,1.0,1.0")
        print("boom", file=sys.stderr)
        sys.exit(3)
    elif data == b"slow":
        time.sleep(60)
""")


def make_box(root: Path, caps=CAPS, solution=None, stars=STARS, truth=TRUTH) -> Path:
    """Write the box files under root/app, as put_files would under /, plus data/; return root/app."""
    solution = {"solution.py": FAKE.encode()} if solution is None else solution
    data = {"truth.json": json.dumps(truth).encode(), **{f"stars/{k}.npz": v for k, v in stars.items()}}
    for path, (body, mode) in box.box_files(caps, box.render_task_md(caps), solution, data).items():
        target = root / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        target.chmod(mode)
    return root / "app"


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    """run(*argv, **box) builds a box, runs run_dev.main(argv) in it and returns (exit code, stdout)."""
    def go(*argv, **kw):
        app = make_box(tmp_path, **kw)
        monkeypatch.setattr(run_dev, "ROOT", app)
        code = run_dev.main(list(argv))
        return code, capsys.readouterr().out
    return go


def json_lines(out: str) -> tuple[dict, list[dict], dict]:
    lines = [json.loads(line) for line in out.splitlines()]
    assert [x["kind"] for x in lines] == ["start"] + ["record"] * (len(lines) - 2) + ["summary"]
    return lines[0], [x["record"] for x in lines[1:-1]], lines[-1]


def test_text_run_scores_the_subset(run):
    code, out = run()
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("solution: ok, 1 file, ")
    assert lines[1].startswith("isolation: none (") if sys.platform == "darwin" else lines[1].startswith("isolation:")
    assert lines[2] == "running 3 stars on 2 workers, 10 s cap per star"
    assert sorted(line.split()[1] for line in lines[3:6]) == ["s-0", "s-1", "s-2"]
    s0 = next(line for line in lines if " s-0 " in line)
    assert " ok " in s0 and "rows 1 (+1 invalid lines): 10,5,9" in s0
    assert "statuses: ok 3" in out
    assert "S = 1.0000  C = 1.0000  F = 0.00" in out


def test_json_run(run):
    code, out = run("--json")
    assert code == 0
    start, records, summary = json_lines(out)
    assert start["solution"]["status"] == "ok" and start["solution"]["n_files"] == 1
    assert start["caps"] == CAPS and sorted(start["stars"]) == ["s-0", "s-1", "s-2"]
    assert start["isolation"] in ("none", "jail")
    assert {r["star"] for r in records} == {"s-0", "s-1", "s-2"}
    assert all(r["status"] == "ok" and r["isolation"] == start["isolation"] and "stderr" in r for r in records)
    assert summary["statuses"] == {"ok": 3}
    assert summary["report"]["headline"]["S"] == 1.0
    assert summary["report"]["inputs"]["missing_stars"] == 0


def test_false_rows_crashes_and_timeouts(run):
    stars = {"s-0": b"inj", "s-1": b"null-1!", "s-2": b"crash"}
    code, out = run(stars=stars)
    assert code == 0
    assert "exit:3" in out and "\n    boom\n" in out
    # One planet claimed, one false row on s-1; the crashed star's printed row does not count.
    assert "S = 0.0000  C = 1.0000  F = 33.33" in out
    code, out = run("--json", "--stars", "s-0", "s-1", caps={**CAPS, "star_cap_s": 1.0},
                    stars={"s-0": b"slow", "s-1": b"crash", "s-2": b"x"})
    _, records, summary = json_lines(out)
    by_star = {r["star"]: r for r in records}
    assert by_star["s-0"]["status"] == "timeout" and by_star["s-0"]["rows"] == []
    assert by_star["s-1"]["status"] == "exit:3" and by_star["s-1"]["rows"] == []
    assert "boom" in by_star["s-1"]["stderr"]
    assert summary["statuses"] == {"timeout": 1, "exit:3": 1}
    assert summary["report"]["headline"]["S"] == 0.0 and summary["report"]["inputs"]["not_ok"] == {
        "timeout": 1, "exit:3": 1}


def test_a_rejected_solution_runs_as_grading_runs_it(run):
    solution = {"solution.py": FAKE.encode(), "a.py": b"", "b.py": b""}
    code, out = run(caps={**CAPS, "solution_max_files": 2}, solution=solution)
    assert code == 0
    assert out.splitlines()[0].startswith("solution: rejected (too_many_files), 0 files")
    assert out.splitlines()[0].endswith("; graded as an empty solution")
    assert "statuses: exit:2 3" in out          # an empty solution/ has no solution.py
    assert "S = 0.0000" in out


def test_dropped_files_are_listed(tmp_path, monkeypatch, capsys):
    app = make_box(tmp_path)
    (app / "solution" / "link.py").symlink_to(app / "solution" / "solution.py")
    os.mkfifo(app / "solution" / "fifo")
    (app / "solution" / "helper.pyc").write_bytes(b"\0" * 16)
    monkeypatch.setattr(run_dev, "ROOT", app)
    assert run_dev.main([]) == 0
    assert "\n  dropped (links, special files, bytecode, non-UTF-8 names): fifo, helper.pyc, link.py\n" in \
        capsys.readouterr().out
    assert run_dev.main(["--json"]) == 0
    start, records, _ = json_lines(capsys.readouterr().out)
    assert start["solution"]["dropped"] == ["fifo", "helper.pyc", "link.py"] and start["solution"]["n_files"] == 1
    assert len(records) == 3


def test_selection(run):
    _, out = run("--json", "--stars", "s-2", "s-0")
    assert sorted(r["star"] for r in json_lines(out)[1]) == ["s-0", "s-2"]
    _, first = run("--json", "--n", "2", "--seed", "4")
    _, second = run("--json", "--n", "2", "--seed", "4")
    picked = sorted(json_lines(first)[0]["stars"])
    assert len(picked) == 2 and picked == sorted(json_lines(second)[0]["stars"])
    _, out = run("--json", "--n", "9")
    assert len(json_lines(out)[1]) == 3
    with pytest.raises(SystemExit) as e:
        run("--stars", "s-9")
    assert e.value.code == 2


def test_missing_box_files_fail_cleanly(tmp_path, monkeypatch, capsys):
    app = make_box(tmp_path, stars={"s-0": b"inj", "s-1": b"x"})      # truth lists s-2, which has no file
    monkeypatch.setattr(run_dev, "ROOT", app)
    assert run_dev.main([]) == 1
    assert "cannot read a star file" in capsys.readouterr().err
    (app / "caps.json").unlink()
    assert run_dev.main([]) == 1
    assert "cannot read the box files" in capsys.readouterr().err


@pytest.mark.skipif(not (DEV_STARS / "dev-000.npz").is_file(), reason="needs the built dev set in data/built/dev")
def test_the_box_run_dev_on_ten_dev_stars(tmp_path):
    """The box's own run_dev.py (the shim and the shipped package) runs the starter on 10 dev stars."""
    caps = {**CAPS, "star_cap_s": 60.0, "workers": 4, "labelled": 196}
    app = make_box(tmp_path, caps=caps, solution=box.starter(), stars={},
                   truth=json.loads((REPO / "dev" / "truth.json").read_text()))
    (app / "data" / "stars").mkdir()
    for star in sorted(DEV_STARS.glob("dev-*.npz")):
        (app / "data" / "stars" / star.name).symlink_to(star)
    result = subprocess.run([sys.executable, "run_dev.py", "--n", "10", "--seed", "0", "--json"], cwd=app,
                            capture_output=True, text=True, timeout=900, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    start, records, summary = json_lines(result.stdout)
    assert start["isolation"] == "none" and start["solution"]["status"] == "ok"
    assert len(records) == 10 and len(set(start["stars"])) == 10
    assert all(r["status"] == "ok" and r["isolation"] == "none" for r in records), records
    assert summary["statuses"] == {"ok": 10}
    truth = [s for s in score.load_truth(REPO / "dev" / "truth.json") if s.star_id in set(start["stars"])]
    catalogue, _ = score.catalogue_from_runs({r["star"]: r for r in records})
    assert summary["report"]["headline"] == json.loads(json.dumps(score.report(truth, catalogue)["headline"]))
