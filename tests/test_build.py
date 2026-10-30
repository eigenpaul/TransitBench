"""Tests for the data build (scripts/build.py): what a star file holds, the truth frame, privacy.

Tests that need built or private data skip when it is absent (a clean clone has neither).
"""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from transitbench.data import QUARTERS_BKJD, STAR_KEYS, STELLAR_KEYS, deliver_flux, round_sig, write_star
from transitbench.inject import CADENCE_DAYS, in_transit, running_median, transit_drop
from transitbench.match import coverage_from_times

pytest.importorskip("astropy")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build  # noqa: E402

DEV_TRUTH = ROOT / "dev" / "truth.json"
DEV_MAP = ROOT / "dev" / "map.json"
DEV_BUILT = ROOT / "data" / "built" / "dev"
POOLS = ROOT / "data" / "private" / "pools.json"
SELECTION = ROOT / "data" / "private" / "releases" / "r1" / "selection.json"
KIC_LIKE = re.compile(r"(?<![0-9A-Za-z.])(\d{6,9})(?![0-9A-Za-z.])")


def built_dev():
    if not (DEV_TRUTH.exists() and DEV_BUILT.exists()):
        pytest.skip("dev set not built")
    return json.loads(DEV_TRUTH.read_text())["stars"]


def synthetic_quarters(seed=0):
    """Float32 'archive' flux in four quarters and its reduction, the way fetch_lightcurves.py makes it."""
    rng = np.random.default_rng(seed)
    times, raws, quarters = [], [], []
    for i, q in enumerate(range(10, 14)):
        t = np.arange(*QUARTERS_BKJD[q], CADENCE_DAYS)
        times.append(t)
        raws.append((52000.0 * (1 + i * 0.01) * (1 + 2e-4 * rng.standard_normal(t.size))).astype(np.float32))
        quarters.append(np.full(t.size, i, dtype=np.int8))
    time, quarter = np.concatenate(times), np.concatenate(quarters)
    raw = np.concatenate(raws).astype(float)
    qmedian = np.array([np.median(raw[quarter == i]) for i in range(4)])
    return time, raw / qmedian[quarter], quarter, qmedian


def test_round_sig_keeps_two_significant_figures():
    assert [round_sig(x) for x in (5923.0, 4.456, 0.991, 1.05, 6499.0, 0.0149)] == [5900.0, 4.5, 0.99, 1.1, 6500.0,
                                                                                     0.015]
    assert round_sig(0.0) == 0.0 and math.isnan(round_sig(math.nan))


def test_untouched_star_is_delivered_bit_for_bit():
    _, flux, quarter, qmedian = synthetic_quarters()
    assert np.array_equal(deliver_flux(flux, quarter, qmedian), flux)


def float32_exact(flux, scale) -> np.ndarray:
    """Which cadences are exactly a float32 value once multiplied back by the quarter's raw median."""
    raw = flux * scale
    return np.abs(raw.astype(np.float32).astype(float) - raw) <= 1e-12 * np.abs(raw)


def test_injected_star_keeps_the_archive_fingerprint():
    time, flux, quarter, qmedian = synthetic_quarters()
    drop = transit_drop(time, 7.3, 910.0, 4.0, 0.03, 0.2)
    inside = in_transit(time, 7.3, 910.0, 4.0)
    out = deliver_flux(flux, quarter, qmedian, drop)
    assert out[inside].mean() < out[~inside].mean() - 5e-4
    raw = ((flux * qmedian[quarter]).astype(np.float32).astype(float) * (1 - drop)).astype(np.float32).astype(float)
    for i in range(4):
        sel = quarter == i
        assert np.median(out[sel]) == pytest.approx(1.0, abs=1e-15)
        assert float32_exact(out[sel], np.median(raw[sel])).all()
    naive = flux * (1 - drop)
    assert not float32_exact(naive[inside & (quarter == 0)], qmedian[0]).all()


def test_star_file_holds_only_the_allowed_keys_and_is_reproducible(tmp_path):
    arrays = {"time": np.arange(3.0), "flux": np.ones(3), "quarter": np.zeros(3, np.int8),
              **{k: np.float64(1.0) for k in STELLAR_KEYS}}
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    write_star(a, arrays)
    write_star(b, arrays)
    assert a.read_bytes() == b.read_bytes()
    assert sorted(np.load(a).files) == sorted(STAR_KEYS)
    with pytest.raises(ValueError):
        write_star(tmp_path / "c.npz", {**arrays, "flux_err": np.ones(3)})


def test_offset_moves_data_and_truth_together():
    time, flux, quarter, _ = synthetic_quarters()
    signal = {"period": 9.1, "t0": 912.4, "duration": 5.0}
    offset = -3127.25
    shifted = build.shift(signal, offset)
    assert np.array_equal(in_transit(time, signal["period"], signal["t0"], signal["duration"]),
                          in_transit(time + offset, shifted["period"], shifted["t0"], shifted["duration"]))


def test_coverage_splits_at_gaps():
    time = np.concatenate([np.arange(0, 10, CADENCE_DAYS), np.arange(12, 20, CADENCE_DAYS)])
    cov = coverage_from_times(time, max_gap=0.5)
    assert cov.shape == (2, 2) and cov[0, 0] == 0 and cov[1, 0] == 12


def test_release_secret_is_created_once(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "SECRET", tmp_path / "secret")
    monkeypatch.setattr(build, "RELEASE_MANIFEST", tmp_path / "release.json")
    first = build.release_secret(new=True)
    assert build.release_secret() == first
    with pytest.raises(SystemExit):
        build.release_secret(new=True)


def test_built_dev_files_hold_only_what_a_pipeline_sees():
    stars = built_dev()
    for sid in stars:
        d = np.load(DEV_BUILT / f"{sid}.npz")
        assert sorted(d.files) == sorted(STAR_KEYS)
        assert d["time"].dtype == np.float64 and d["flux"].dtype == np.float64 and d["quarter"].dtype == np.int8
        assert set(np.unique(d["quarter"])) <= {0, 1, 2, 3}
        for k in STELLAR_KEYS:
            assert d[k].shape == () and (math.isnan(float(d[k])) or float(d[k]) == round_sig(float(d[k])))


def test_built_dev_truth_is_in_the_delivered_frame():
    stars = built_dev()
    mapping = json.loads(DEV_MAP.read_text())["stars"]
    lo, hi = QUARTERS_BKJD[10][0] - 0.01, QUARTERS_BKJD[13][1] + 0.01
    for sid, entry in stars.items():
        d = np.load(DEV_BUILT / f"{sid}.npz")
        time, flux = d["time"], d["flux"]
        bkjd = time - mapping[sid]["time_offset"]
        assert lo <= bkjd.min() and bkjd.max() <= hi
        assert np.allclose(np.asarray(entry["coverage"]), coverage_from_times(time, max_gap=0.5), rtol=0, atol=0)
        detrended = flux / running_median(time, flux)
        for s in entry["signals"]:
            if s["kind"] == "injected":
                assert detrended[in_transit(time, s["period"], s["t0"], s["duration"])].mean() < 1.0


def test_dev_truth_loads_in_the_scorer():
    from transitbench.score import load_truth

    built_dev()
    assert len(load_truth(DEV_TRUTH)) == len(json.loads(DEV_TRUTH.read_text())["stars"])


def test_samples_are_disjoint_and_come_from_their_pools():
    if not (POOLS.exists() and SELECTION.exists()):
        pytest.skip("private pools or selection absent")
    pools = json.loads(POOLS.read_text())
    sel = json.loads(SELECTION.read_text())
    lists = ("random", "tce_only", "eb", "real_7_12", "real_12_up")
    pool = {name: {k for p in lists for k in pools[p][name]} for name in ("reserve", "dev", "train")}
    pool["natural"] = set(pools["natural"])
    sets = {s: {k for v in sel[s].values() for k in v} for s in ("dev", "A", "B")}
    assert not sets["A"] & sets["B"] and not sets["A"] & sets["dev"] and not sets["B"] & sets["dev"]
    assert sets["A"] | sets["B"] <= pool["reserve"] and sets["dev"] <= pool["dev"]
    assert not (sets["A"] | sets["B"] | sets["dev"]) & (pool["train"] | pool["natural"])


def test_no_reserve_kic_in_committed_files():
    if not POOLS.exists():
        pytest.skip("private pools absent")
    pools = json.loads(POOLS.read_text())
    reserve = {int(k) for p in ("random", "tce_only", "eb", "real_7_12", "real_12_up") for k in pools[p]["reserve"]}
    reserve |= {int(k) for k in pools["natural"]}
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
                            capture_output=True, check=True).stdout.decode().split("\0")
    leaks = {}
    for name in filter(None, listed):
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        found = {int(m) for m in KIC_LIKE.findall(text)} & reserve
        if found:
            leaks[name] = len(found)
    assert not leaks, f"test-reserve KICs in committable files: {leaks}"


R1_TRUTHS = [ROOT / "data" / "private" / "releases" / "r1" / f"truth_{s}.json" for s in ("A", "B")]


def all_truths():
    paths = [p for p in [DEV_TRUTH, *R1_TRUTHS] if p.exists()]
    if not paths:
        pytest.skip("no truth built")
    return {p.name: json.loads(p.read_text())["stars"] for p in paths}


def test_injected_durations_are_physical():
    for name, stars in all_truths().items():
        for sid, entry in stars.items():
            for s in entry["signals"]:
                if s["kind"] == "injected":
                    assert s["duration"] / (24.0 * s["period"]) < 0.25, (name, sid)


def test_stratified_targets_fill_every_stratum_once():
    for name, stars in all_truths().items():
        cells = {}
        for entry in stars.values():
            if entry["population"] == "inj":
                cells.setdefault(entry["host"], []).append(entry["signals"][0]["mes_target"])
        for host, targets in cells.items():
            n = len(targets)
            assert sorted(int((t - 4.0) / 16.0 * n) for t in targets) == list(range(n)), (name, host)


def test_assign_targets_is_a_jittered_grid():
    cells = {"inj_random": list(range(100, 140)), "inj_tce_only": list(range(200, 240))}
    targets = build.assign_targets("dev", cells, "seed")
    assert targets == build.assign_targets("dev", cells, "seed")
    for cell, kics in cells.items():
        values = sorted(targets[k] for k in kics)
        assert [int((t - 4.0) / 16.0 * 40) for t in values] == list(range(40))


def test_ebs_carry_variability_and_quiet():
    for name, stars in all_truths().items():
        for entry in stars.values():
            if entry["population"] == "eb":
                assert {"variability", "quiet", "secondary_phase"} <= set(entry["eb"])
                assert isinstance(entry["eb"]["quiet"], bool)


def test_delivered_stellar_parameters_are_finite():
    built_dev()
    for path in DEV_BUILT.glob("*.npz"):
        d = np.load(path)
        assert all(math.isfinite(float(d[k])) for k in STELLAR_KEYS), path.name


def test_real_mes_is_on_the_injection_scale():
    from transitbench.inject import mes

    time, flux, _, _ = synthetic_quarters(5)
    ld = (0.4, 0.2, 0.1, -0.05)
    s = {"period": 6.3, "t0": 908.0, "duration": 3.5, "ror": 0.02, "impact": 0.3, "depth_ppm": 520.0}
    drop = transit_drop(time, s["period"], s["t0"], s["duration"], s["ror"], s["impact"], ld)
    value, method = build.real_mes(time, flux, s, ld)
    assert method == "shape" and value == pytest.approx(mes(time, flux, 6.3, 908.0, 3.5, drop))
    fallback, method = build.real_mes(time, flux, {**s, "ror": math.nan, "depth_ppm": drop.max() * 1e6}, ld)
    assert method == "depth" and fallback == pytest.approx(value, rel=0.1)

