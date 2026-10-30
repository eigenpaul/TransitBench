"""The clean-clone smoke test (scripts/smoke_test.py): each check passes on the committed files and fails when the
thing it checks is broken. Needs no data/ directory; the starter tests need astropy (the build group)."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from transitbench import score
from transitbench.data import STAR_KEYS
from transitbench.inject import TOO_DEEP

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("smoke_test", ROOT / "scripts" / "smoke_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load()
RESULTS, RELEASE, DEV = ROOT / "results" / "baselines.json", ROOT / "manifests" / "release_r1.json", ROOT / "dev"


def test_scorer_check_passes_on_the_dev_truth():
    assert smoke.check_scorer(DEV / "truth.json") == []


def test_scorer_check_fails_for_a_scorer_that_pays_for_nothing(monkeypatch):
    monkeypatch.setattr(smoke.score, "score", lambda stars, catalogue: {"S": 1.0, "C": 1.0, "false": 0, "planets": 1})
    problems = smoke.check_scorer(DEV / "truth.json")
    assert any("no rows scored" in p for p in problems) and any("wrong row" in p for p in problems)


def test_synthetic_stars_look_like_star_files_and_their_truth_loads(tmp_path):
    truth = smoke.synthetic_stars(tmp_path)
    stars = {s.star_id: s for s in score.parse_truth(truth)}
    assert sorted(stars) == ["smoke-eclipse", "smoke-noise", "smoke-planet"]
    assert score.score(list(stars.values()), smoke.oracle(list(stars.values())))["S"] == 1.0
    for sid in stars:
        with np.load(tmp_path / f"{sid}.npz") as f:
            assert set(f.files) == set(STAR_KEYS)
            flux, quarter = f["flux"], f["quarter"]
        assert [np.median(flux[quarter == q]) for q in range(4)] == pytest.approx([1.0] * 4)
    with np.load(tmp_path / "smoke-eclipse.npz") as f:
        assert 1.0 - f["flux"].min() > 2 * TOO_DEEP  # the starter's depth cut rejects it
    with np.load(tmp_path / "smoke-planet.npz") as f:
        assert 1.0 - f["flux"].min() < TOO_DEEP


def test_synthetic_stars_are_deterministic(tmp_path):
    smoke.synthetic_stars(tmp_path / "a")
    smoke.synthetic_stars(tmp_path / "b")
    for sid in ("smoke-planet", "smoke-eclipse", "smoke-noise"):
        assert (tmp_path / "a" / f"{sid}.npz").read_bytes() == (tmp_path / "b" / f"{sid}.npz").read_bytes()


def test_starter_check_passes():
    pytest.importorskip("astropy", reason="the starter needs astropy (uv sync --group build)")
    assert smoke.check_starter(ROOT / "workspace" / "solution") == []


def test_starter_check_fails_for_a_solution_that_claims_every_star(tmp_path):
    (tmp_path / "solution.py").write_text("print('1.2345,0.5,1')\n")
    problems = smoke.check_starter(tmp_path)
    assert problems and all("expected 1" in p for p in problems)


def test_starter_check_fails_for_a_crashing_solution(tmp_path):
    (tmp_path / "solution.py").write_text("raise SystemExit(3)\n")
    problems = smoke.check_starter(tmp_path)
    assert sum("ended exit:3" in p for p in problems) == 3


def test_starter_check_fails_for_a_missing_solution(tmp_path):
    assert smoke.check_starter(tmp_path / "nothing") == ["the solution snapshot is empty (missing)"]


def test_results_check_passes_on_the_committed_results():
    assert smoke.check_results(RESULTS, RELEASE, DEV) == []


def test_results_check_catches_a_number_that_does_not_follow_from_its_counts(tmp_path):
    res = json.loads(RESULTS.read_text())
    res["rows"]["starter"]["A"]["S"] += 0.01
    (tmp_path / "r.json").write_text(json.dumps(res))
    assert smoke.check_results(tmp_path / "r.json", RELEASE, DEV) == [
        "row starter, set A: S does not follow from the counts"]


def test_results_check_catches_results_from_another_release(tmp_path):
    res = json.loads(RESULTS.read_text())
    res["sets"]["B"]["truth_sha256"] = "0" * 64
    del res["rows"]["learned_vetter"]
    (tmp_path / "r.json").write_text(json.dumps(res))
    assert smoke.check_results(tmp_path / "r.json", RELEASE, DEV) == [
        "set B: the digests differ from the committed release", "row learned_vetter is missing"]


def test_star_files_digest_follows_the_release_rule():
    files = {"b": "2" * 64, "a": "1" * 64}
    assert smoke.star_files_digest(files) == hashlib.sha256(
        f"a {'1' * 64}\nb {'2' * 64}\n".encode()).hexdigest()


def test_main_reports_every_check(capsys, monkeypatch):
    monkeypatch.setattr(smoke, "check_starter", lambda solution_dir: ["boom"])
    assert smoke.main() == 1
    out = capsys.readouterr().out
    assert "PASS scorer" in out and "FAIL starter" in out and "PASS results" in out
    assert "smoke test: 2 of 3 checks passed" in out


def test_every_relative_link_in_the_readme_points_into_the_repo():
    text = (ROOT / "README.md").read_text()
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    local = [t for t in targets if not t.startswith(("http://", "https://", "#"))]
    assert local, "the README links to repo files"
    assert [t for t in local if not (ROOT / t).exists()] == []
    anchors = {re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-")
               for h in re.findall(r"^#+ (.+)$", text, flags=re.M)}
    assert [t for t in targets if t.startswith("#") and t[1:] not in anchors] == []
