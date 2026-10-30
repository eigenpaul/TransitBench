import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from transitbench.match import Candidate, Signal
from transitbench.runner import parse_rows, run_star
from transitbench.score import Star, score

YEAR = np.array([[0.0, 370.0]])


def stars():
    return [
        Star("p1", "inj", YEAR, [Signal(10.0, 3.0, 4.0)]),
        Star("n1", "null", YEAR),
        Star("e1", "eb", YEAR),
        Star("r1", "real", YEAR, [Signal(7.0, 2.0, 3.0)], scored=False),
    ]


def test_empty_catalogue_scores_zero():
    out = score(stars(), {})
    assert out["S"] == 0.0 and out["C"] == 0.0 and out["F_per_100"] == 0.0


def test_truth_scores_one_and_real_is_outside_s():
    cat = {"p1": [Candidate(10.0, 3.0, 1.0)], "r1": [Candidate(1.0, 0.1, 1.0)]}
    out = score(stars(), cat)
    assert out["S"] == 1.0 and out["planets"] == 1 and out["scored_stars"] == 3
    assert out["populations"]["real"]["false"] == 1


def test_claims_on_null_and_eb_cost_points():
    cat = {"p1": [Candidate(10.0, 3.0, 1.0)], "n1": [Candidate(3.3, 1.0, 1.0)], "e1": [Candidate(2.0, 0.5, 1.0)]}
    out = score(stars(), cat)
    assert out["S"] == -1.0 and out["eb_as_planet"] == 1.0
    assert out["F_per_100"] == pytest.approx(200 / 3)


def test_only_top_three_rows_count():
    rows = [Candidate(10.0, 3.0, 0.1)] + [Candidate(1.0 + i, 0.2, 1.0) for i in range(3)]
    out = score(stars()[:1], {"p1": rows})
    assert out["C"] == 0.0 and out["S"] == -3.0


def test_parse_rows_drops_malformed_lines():
    text = "10,3,1\nbad line\n1,2\nnan,1,1\n-1,1,1\n5.5,2.25,0.5\n"
    assert parse_rows(text) == [Candidate(10.0, 3.0, 1.0), Candidate(5.5, 2.25, 0.5)]


def write_solution(tmp_path, body):
    sol = tmp_path / "solution"
    sol.mkdir()
    (sol / "solution.py").write_text(textwrap.dedent(body))
    star = tmp_path / "star.npz"
    np.savez(star, time=np.arange(10.0), flux=np.ones(10), quarter=np.zeros(10))
    return sol, star


def test_runner_reads_stdin_and_parses_stdout(tmp_path):
    sol, star = write_solution(tmp_path, """
        import io, sys, numpy as np
        star = np.load(io.BytesIO(sys.stdin.buffer.read()))
        print(f"{star['time'].size},1.5,0.9")
    """)
    result = run_star(sol, star, cap_seconds=30, python=sys.executable)
    assert result["status"] == "ok" and result["rows"] == [Candidate(10.0, 1.5, 0.9)]
    assert result["isolation"] == "none" and result["n_invalid"] == 0 and result["leak"] == 0


def test_runner_keeps_the_top_three_rows_by_score(tmp_path):
    sol, star = write_solution(tmp_path, """
        for line in ["1,1,0.1", "2,1,0.9", "junk", "3,1,0.5", "4,1,0.9", "nan,1,1", "5,1,0.2"]:
            print(line)
    """)
    result = run_star(sol, star, cap_seconds=30, python=sys.executable)
    assert [c.period for c in result["rows"]] == [2.0, 4.0, 3.0]
    assert result["n_rows"] == 5 and result["n_invalid"] == 2


def test_runner_kills_on_timeout(tmp_path):
    sol, star = write_solution(tmp_path, "import time\nprint('1,1,1', flush=True)\ntime.sleep(60)\n")
    result = run_star(sol, star, cap_seconds=1, python=sys.executable)
    assert result["status"] == "timeout" and result["rows"] == []


def test_runner_crash_gives_no_rows(tmp_path):
    sol, star = write_solution(tmp_path, "print('1,1,1')\nraise SystemExit(3)\n")
    result = run_star(sol, star, cap_seconds=30, python=sys.executable)
    assert result["status"] == "exit:3" and result["rows"] == [] and result["n_rows"] == 0


def test_runner_accepts_relative_paths(tmp_path, monkeypatch):
    sol, star = write_solution(tmp_path, "print('2,1,1')\n")
    monkeypatch.chdir(tmp_path)
    result = run_star(Path(sol.name), Path(star.name), cap_seconds=30, python=sys.executable)
    assert result["status"] == "ok" and result["rows"] == [Candidate(2.0, 1.0, 1.0)]
