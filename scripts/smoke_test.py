"""Smoke test for a clean clone: no data/ directory, no evaluator data, no network and no model.

Three checks, one PASS or FAIL line each:

  scorer    The official scorer (transitbench/score.py) on the public dev truth (dev/truth.json, 196 stars).
            No rows scores S = 0, the true ephemerides score S = 1, and one wrong row added to every star scores
            below 1.
  starter   The naive starter (workspace/solution) is snapshotted as grading snapshots it (transitbench.archive) and
            run by the grading runner (transitbench.runner, without the jail) on three synthetic stars made here:
            an injected planet, an eclipse 6% deep and noise only. It must find the planet at its true ephemeris and
            print nothing for the other two, so the scorer gives S = 1. White noise is far easier than real Kepler
            light curves: this checks the code path, not the science.
  results   results/baselines.json, the baseline table on samples A, B and dev. Its set digests must equal the
            committed release (manifests/release_r1.json, dev/truth.json, dev/files.json), and each row's S, C and F
            must follow from its own counts. The table is printed. The star files behind it are private, so the
            scores themselves cannot be recomputed here.

Run from the repo root after `uv sync --group build`:
    uv run --group build python scripts/smoke_test.py
Exit status: 0 when every check passes, 1 otherwise.
"""

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from transitbench import archive, runner, score
from transitbench.data import QUARTERS_BKJD, write_star
from transitbench.inject import CADENCE_DAYS, transit_drop
from transitbench.match import Candidate, coverage_from_times

ROOT = Path(__file__).resolve().parent.parent
WINDOW = (10, 11, 12, 13)
SEED = 20261028
NOISE = 3e-4
PLANET = {"period": 7.3, "t0_after_start": 2.1, "duration": 3.5, "ror": 0.03, "b": 0.3}
ECLIPSE = {"period": 3.9, "t0_after_start": 1.3, "duration": 4.0, "ror": 0.25, "b": 0.0}
CAP_S = 60.0
ROWS = ("empty", "starter", "naive_v0", "learned_vetter", "reference_detector")


def oracle(stars: list[score.Star]) -> dict[str, list[Candidate]]:
    """The true ephemeris of every scored signal, at most score.MAX_ROWS per star."""
    return {s.star_id: [Candidate(x.period, x.t0, 1.0) for x in s.signals if x.scored][:score.MAX_ROWS]
            for s in stars}


def check_scorer(truth_path: Path) -> list[str]:
    """Problems with the scorer on a truth file; an empty list means it passed."""
    stars = score.load_truth(truth_path)
    empty = score.score(stars, {})["S"]
    truth = score.score(stars, oracle(stars))
    wrong = {sid: rows + [Candidate(0.7311, 0.123, 0.5)] for sid, rows in oracle(stars).items()}
    hedged = score.score(stars, wrong)["S"]
    problems = []
    if empty != 0.0:
        problems.append(f"no rows scored S = {empty}, expected 0")
    if not (truth["S"] == 1.0 and truth["C"] == 1.0 and truth["false"] == 0):
        problems.append(f"the true ephemerides scored S = {truth['S']}, C = {truth['C']}, expected 1 and 1")
    if not hedged < 1.0:
        problems.append(f"one wrong row per star scored S = {hedged}, expected below 1")
    print(f"  {len(stars)} stars, {truth['planets']} scored planets: S(no rows) = {empty}, S(truth) = {truth['S']}, "
          f"S(truth + one wrong row per star) = {hedged:.4f}")
    return problems


def synthetic_stars(directory: Path, seed: int = SEED) -> dict:
    """Write three star files (planet, eclipse, noise) to directory; return their truth document.

    Each file has exactly the keys a real star file has (transitbench.data.STAR_KEYS): one year of Kepler long
    cadence over quarters 10-13, white noise of NOISE per cadence, flux over each quarter's median."""
    rng = np.random.default_rng(seed)
    time = np.concatenate([np.arange(*QUARTERS_BKJD[q], CADENCE_DAYS) for q in WINDOW])
    quarter = np.concatenate([np.full(np.arange(*QUARTERS_BKJD[q], CADENCE_DAYS).size, i, dtype=np.int8)
                              for i, q in enumerate(WINDOW)])
    coverage = coverage_from_times(time, max_gap=0.5).tolist()
    stellar = {"teff": 5800.0, "logg": 4.4, "radius": 1.0, "mass": 1.0}

    def ephemeris(shape: dict) -> dict:
        return {"period": shape["period"], "t0": float(time[0]) + shape["t0_after_start"],
                "duration": shape["duration"]}

    def write(name: str, shape: dict | None) -> None:
        flux = 1.0 + NOISE * rng.standard_normal(time.size)
        if shape is not None:
            e = ephemeris(shape)
            flux *= 1.0 - transit_drop(time, e["period"], e["t0"], e["duration"], shape["ror"], shape["b"])
        for i in range(len(WINDOW)):
            flux[quarter == i] /= np.median(flux[quarter == i])
        write_star(directory / f"{name}.npz", {"time": time, "flux": flux, "quarter": quarter, **stellar})

    write("smoke-planet", PLANET)
    write("smoke-eclipse", ECLIPSE)
    write("smoke-noise", None)
    planet = {**ephemeris(PLANET), "scored": True, "kind": "injected", "mes": None}
    eclipse = {**ephemeris(ECLIPSE), "evidence": {"too_deep": True}}
    return {"release": "smoke", "sample": "smoke", "window": list(WINDOW), "stars": {
        "smoke-planet": {"population": "inj", "host": "random", "scored": True, "coverage": coverage,
                         "signals": [planet]},
        "smoke-eclipse": {"population": "eb", "host": "eb", "scored": True, "coverage": coverage, "signals": [],
                          "eb": eclipse},
        "smoke-noise": {"population": "null", "host": "random", "scored": True, "coverage": coverage,
                        "signals": []},
    }}


def check_starter(solution_dir: Path) -> list[str]:
    """Problems with the starter run end to end on the synthetic stars; an empty list means it passed."""
    with tempfile.TemporaryDirectory(prefix="tb-smoke-") as tmp:
        tmp = Path(tmp)
        truth = synthetic_stars(tmp / "stars")
        snapshot = archive.from_dir(solution_dir)
        if snapshot.status != "ok":
            return [f"the solution snapshot is {snapshot.status} ({snapshot.reason})"]
        solution = archive.extract(snapshot, tmp / "box")
        stars = {sid: tmp / "stars" / f"{sid}.npz" for sid in truth["stars"]}
        records = runner.run_catalogue(solution, stars, CAP_S, workers=1, isolation="none")
    catalogue, inputs = score.catalogue_from_runs(records)
    head = score.score(score.parse_truth(truth), catalogue)
    for sid, rec in records.items():
        rows = ", ".join(f"P = {r.period:.4f} d" for r in rec["rows"]) or "no rows"
        cpu = "CPU time unknown" if rec.get("cpu_s") is None else f"{rec['cpu_s']:.1f} s CPU"
        print(f"  {sid}: {rec['status']}, {cpu}, {rows}")
    print(f"  S = {head['S']}, C = {head['C']}, false candidates = {head['false']}")
    problems = [f"{sid} ended {rec['status']}" for sid, rec in records.items() if rec["status"] != "ok"]
    if head["S"] != 1.0:
        problems.append(f"S = {head['S']}, expected 1 (planet found, nothing on the eclipse or the noise)")
    return problems


def star_files_digest(files: dict[str, str]) -> str:
    """The rule in manifests/release_r1.json: SHA-256 of '<star_id> <file sha256>\\n' lines sorted by star id."""
    return hashlib.sha256("".join(f"{sid} {files[sid]}\n" for sid in sorted(files)).encode()).hexdigest()


def check_results(results_path: Path, release_path: Path, dev_dir: Path) -> list[str]:
    """Problems with the baseline results file; an empty list means it passed. Prints the table."""
    res = json.loads(results_path.read_text())
    release = json.loads(release_path.read_text())
    dev_files = json.loads((dev_dir / "files.json").read_text())
    expected = {s: {k: release["samples"][s][k] for k in ("truth_sha256", "star_files_sha256")} for s in ("A", "B")}
    expected["dev"] = {"truth_sha256": hashlib.sha256((dev_dir / "truth.json").read_bytes()).hexdigest(),
                       "star_files_sha256": star_files_digest(dev_files["files"])}
    problems = []
    if dev_files["digest"] != expected["dev"]["star_files_sha256"]:
        problems.append("dev/files.json: its digest does not follow from its file list")
    for s, want in expected.items():
        got = {k: res["sets"].get(s, {}).get(k) for k in want}
        if got != want:
            problems.append(f"set {s}: the digests differ from the committed release")
    print(f"  {'row':<20} {'S on A':>8} {'S on B':>8} {'S on dev':>9} {'C on A':>8} {'F/100 on A':>11} "
          f"{'EB-as-planet on A':>18}")
    for name in ROWS:
        row = res["rows"].get(name)
        if row is None:
            problems.append(f"row {name} is missing")
            continue
        for s in ("A", "B", "dev"):
            r = res["rows"][name][s]
            planets, stars = max(r["planets"], 1), max(r["scored_stars"], 1)
            implied = {"S": (r["credit"] - r["false"]) / planets, "C": r["credit"] / planets,
                       "F_per_100": 100.0 * r["false"] / stars}
            for key, value in implied.items():
                if not (math.isclose(r[key], value, abs_tol=1e-9)
                        and math.isclose(res["table"][s][name][key], value, abs_tol=1e-9)):
                    problems.append(f"row {name}, set {s}: {key} does not follow from the counts")
        a, b, d = row["A"], row["B"], row["dev"]
        print(f"  {name:<20} {a['S']:>8.4f} {b['S']:>8.4f} {d['S']:>9.4f} {a['C']:>8.4f} {a['F_per_100']:>11.2f} "
              f"{a['eb_as_planet']:>18.3f}")
    return problems


def main() -> int:
    checks = (
        ("scorer", lambda: check_scorer(ROOT / "dev" / "truth.json")),
        ("starter", lambda: check_starter(ROOT / "workspace" / "solution")),
        ("results", lambda: check_results(ROOT / "results" / "baselines.json", ROOT / "manifests" / "release_r1.json",
                                          ROOT / "dev")),
    )
    failed = 0
    for name, check in checks:
        print(f"{name}:")
        try:
            problems = check()
        except Exception as e:  # a crash is a failed check, reported like the others
            problems = [f"{type(e).__name__}: {e}"]
        for p in problems:
            print(f"  problem: {p}")
        print(f"{'PASS' if not problems else 'FAIL'} {name}")
        failed += bool(problems)
    print(f"smoke test: {len(checks) - failed} of {len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
