"""Build the TransitBench v1 data: the public dev set and test release r1, samples A and B.

This is the evaluator's tool, shipped to show how the dev set and the test samples are built. It needs the evaluator's
private data and inputs that are not in this repository.

Each star is one .npz holding only what a pipeline may see (transitbench.data.STAR_KEYS): time (BKJD plus a random
per-star offset), flux (PDCSAP over each quarter's median, quality-0 cadences only), quarter (0-3), and Teff, logg,
radius and mass rounded to 2 significant figures. Truth is one JSON per sample, checked by transitbench.score.load_truth
as it is written.

Steps, one subcommand each (`all` runs select, fetch, measure, select, build, validate):
  select     dev = the dev pool lists; r1 = samples A and B from the reserve, after trimming the host and REAL
             reserves on stellar-table features to the stars least distinguishable from the other populations
             (propensity trimming); EBs are drawn at random. REAL needs measure first, so select runs twice
  fetch      download and reduce dev, every REAL reserve host and the selected reserve stars (cached, data/lc/)
  measure    per star, before delivery: REAL signals with their one-year MES, the EB ephemeris and in-window
             evidence, and non-signal features (cached by light-curve content)
  build      inject, shift, round and write star files, truth, maps and manifests, and results/m2_build.json
  validate   per-pair adversarial validation on each built sample: results/m2_adversarial.json
  baselines  the naive starter and the reference vetter on dev only: results/m2_dev_baselines.json

Test draws, offsets and injections are seeded by the release secret (data/private/releases/r1/secret), which
`select --new-secret` creates once and never replaces. Dev uses the public seed DEV_SEED. Test truth, maps, offsets
and seeds stay under data/private/releases/r1/; only digests and counts are committed (manifests/release_r1.json).
TCE-only stars listed under "drop" in data/private/tce_audit.json (the visual audit) are never NULL or INJ hosts.

Run from the repo root: uv run --group build python -I scripts/build.py STEP [--new-secret] [--workers N]
"""

import argparse
import hashlib
import json
import math
import os
import secrets
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")  # small models in many processes; threads only contend
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from transitbench.data import (QUARTERS_BKJD, STELLAR_KEYS, deliver_flux, eb_ephemerides, injection_eb_periods,
                               kirk_ephemerides, read_csv, round_sig, verify_tables, write_star)
from transitbench.inject import (CADENCE_DAYS, TOO_DEEP, draw_injection, eb_tests, in_transit, mes, robust_scatter,
                                 running_median, transit_count, transit_drop)
from transitbench.match import Candidate, Signal, compare, coverage_from_times

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_lightcurves  # noqa: E402

TABLES = ROOT / "data" / "tables"
LC = ROOT / "data" / "lc" / "mast"
POOLS = ROOT / "data" / "private" / "pools.json"
AUDIT = ROOT / "data" / "private" / "tce_audit.json"
PRIVATE = ROOT / "data" / "private" / "releases" / "r1"
SECRET = PRIVATE / "secret"
MEASURED = PRIVATE / "measure.json"
SELECTION = PRIVATE / "selection.json"
BUILT = ROOT / "data" / "built"
DEV_DIR = ROOT / "dev"
DEV_LIGHTCURVES = ROOT / "manifests" / "dev_lightcurves.json"
RELEASE_MANIFEST = ROOT / "manifests" / "release_r1.json"
M0_COUNTS = ROOT / "results" / "m0_counts.json"
OUT_BUILD = ROOT / "results" / "m2_build.json"
OUT_ADVERSARIAL = ROOT / "results" / "m2_adversarial.json"
OUT_BASELINES = ROOT / "results" / "m2_dev_baselines.json"
SOLUTIONS = {"starter": ROOT / "workspace" / "solution", "naive_v0": ROOT / "baselines" / "naive_v0",
             "vetter": ROOT / "baselines" / "vetter"}

RELEASE = "r1"
WINDOW = [10, 11, 12, 13]
DEV_SEED = "transitbench-v1-dev"
OFFSET_DAYS = 5000.0
MES_MIN = 7.1
REAL_SPLIT = 12.0
P_RANGE = (0.5, 100.0)
POOL_LISTS = ("random", "tce_only", "eb", "real_7_12", "real_12_up")
TEST_CELLS = {"inj_random": 200, "inj_tce_only": 200, "null_random": 175, "null_tce_only": 175, "eb": 150,
              "real_7_12": 50, "real_12_up": 50}
DEV_CELLS = {"inj_random": 40, "inj_tce_only": 40, "null_random": 35, "null_tce_only": 35, "eb": 30,
             "real_7_12": 10, "real_12_up": 10}
SAMPLES = ("A", "B")
WORKERS = 8
CAP_SECONDS = 20.0
MASK_PAD_HOURS = 1.0
ECLIPSE_EDGE = 0.05
ORBIT_HARMONICS = 4
MEASURE_VERSION = 2
FEATURES = ("teff", "logg", "radius", "mass", "n_cadences", "n_quarters", "p2p", "scatter_1h", "scatter_3h",
            "scatter_6h", "red_noise_3h", "red_noise", "var_slow", "var_mad", "quarter_noise_ratio", "pos_outliers")
VARIABILITY = "red_noise_3h"
QUIET_PERCENTILE = 90.0
MES_HIST_EDGES = (4.0, 6.0, 7.1, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
INJ_CELLS = ("inj_random", "inj_tce_only")
TABLE_FEATURES = ("teff", "logg", "radius", "mass", "kepmag", "log_cdpp_3h", "log_cdpp_6h")
PAIRS = (("inj", "null"), ("inj", "eb"), ("null", "eb"), ("inj", "real"), ("null", "real"), ("eb", "real"))
AUC_GATE = 0.6
BOOTSTRAP = 2000
NULL_RUNS = 40


def digest(*parts) -> bytes:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()


def unit(*parts) -> float:
    return int.from_bytes(digest(*parts)[:8], "big") / 2**64


def seed_of(*parts) -> int:
    return int.from_bytes(digest(*parts)[:8], "big")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=float) + "\n")


def release_secret(new: bool = False) -> str:
    if new:
        if SECRET.exists():
            raise SystemExit(f"{SECRET} exists; refusing to replace it (that would redraw release {RELEASE})")
        SECRET.parent.mkdir(parents=True, exist_ok=True)
        SECRET.write_text(secrets.token_hex(32) + "\n")
    if not SECRET.exists():
        raise SystemExit(f"{SECRET} is missing. Restore it; select --new-secret starts a new release")
    secret = SECRET.read_text().strip()
    committed = json.loads(RELEASE_MANIFEST.read_text()).get("secret_sha256") if RELEASE_MANIFEST.exists() else None
    if committed and committed != sha256_bytes(secret.encode()):
        raise SystemExit(f"release secret does not match {RELEASE_MANIFEST.relative_to(ROOT)}")
    return secret


def audit_drops() -> tuple[set[int], dict]:
    if not AUDIT.exists():
        return set(), {"file": None, "sha256": None, "drop": 0}
    drop = {int(k) for k in json.loads(AUDIT.read_text())["drop"]}
    return drop, {"file": str(AUDIT.relative_to(ROOT)), "sha256": sha256_file(AUDIT), "drop": len(drop)}


def load_tables() -> dict:
    hashes = verify_tables(TABLES, ROOT / "manifests" / "tables.json")
    return {"hashes": hashes,
            "stellar": read_csv(TABLES / "dr25_stellar.csv", text=("st_quarters",)),
            "koi": read_csv(TABLES / "cumulative_koi.csv", text=("kepoi_name", "kepler_name")),
            "tce": read_csv(TABLES / "dr25_obs_tce.csv"),
            "koi_shape": read_csv(TABLES / "cumulative_koi_shape.csv", text=("kepoi_name",)),
            "tce_shape": read_csv(TABLES / "dr25_tce_shape.csv"),
            "kirk": read_csv(TABLES / "kirk2016_eb.tsv", delimiter="\t", comment="#", text=("KIC",))}


def rows_by_kic(table) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i, k in enumerate(table["kepid"]):
        out.setdefault(int(k), []).append(i)
    return out


def stellar_rows(stellar) -> dict[int, dict]:
    out = {}
    for i, k in enumerate(stellar["kepid"]):
        row = {key: float(stellar[key][i]) for key in STELLAR_KEYS}
        row["ld"] = tuple(0.0 if not math.isfinite(c) else float(c)
                          for c in (stellar[f"limbdark_coeff{j}"][i] for j in range(1, 5)))
        out[int(k)] = row
    return out


def load_lc(kic: int):
    d = np.load(LC / f"{kic}.npz")
    order = np.argsort(d["time"], kind="stable")
    return d["time"][order], d["flux"][order], d["quarter"][order], d["qmedian"]


def lc_sha256(kic: int) -> str:
    """Hash of a reduced light curve's arrays (not the file, whose zip timestamps change on every fetch)."""
    d = np.load(LC / f"{kic}.npz")
    return sha256_bytes(b"".join(np.ascontiguousarray(d[k]).tobytes() for k in ("time", "flux", "quarter", "qmedian")))


def mad(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(x - np.median(x))))


def phase(time, period, t0):
    return (np.asarray(time) - t0 + 0.5 * period) % period - 0.5 * period


def finite(*values) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


# ---- measurements on one light curve -------------------------------------------------------------------------------

def mean_to_peak(ld) -> float:
    """Mean in-transit drop over the central drop of a small central transit with this limb darkening."""
    t = np.linspace(-0.6, 0.6, 4001)
    drop = transit_drop(t, 10.0, 0.0, 24.0, 0.05, 0.0, ld)
    return float(drop[np.abs(t) < 0.5].mean() / drop.max())


def real_mes(time, flux, s: dict, ld) -> tuple[float | None, str | None]:
    """One-year MES of a catalogue signal on the INJ scale (inject.mes of a limb-darkened drop): the transit at the
    catalogue radius ratio and impact parameter, or, without those, a box at the catalogue depth times the host's
    mean-to-peak ratio. Returns (MES, method)."""
    period, t0, duration = s["period"], s["t0"], s["duration"]
    ror, b = s.get("ror"), s.get("impact")
    if finite(ror, b) and ror > 0 and 0 <= b < 1 + ror:
        drop = transit_drop(time, period, t0, duration, ror, b, ld)
        return float(mes(time, flux, period, t0, duration, drop)), "shape"
    if finite(s["depth_ppm"]) and s["depth_ppm"] > 0:
        drop = np.where(in_transit(time, period, t0, duration), s["depth_ppm"] * 1e-6 * mean_to_peak(ld), 0.0)
        return float(mes(time, flux, period, t0, duration, drop)), "depth"
    return None, None


def real_signals(time, flux, kois: list[dict], tces: list[dict], ld) -> list[dict]:
    """Every KOI on a REAL host, plus DR25 TCEs that match none of them; ephemerides in the frame of time.

    Scored exactly when CONFIRMED, P 0.5-100 d and one-year MES >= 7.1; everything else is neutral.
    """
    coverage = coverage_from_times(time, max_gap=0.5)
    out = []
    for k in kois:
        if not finite(k["period"], k["t0"], k["duration"]) or k["period"] <= 0 or k["duration"] <= 0:
            continue
        m, method = real_mes(time, flux, k, ld)
        kind = "confirmed" if k["disposition"] == "CONFIRMED" else "koi"
        scored = kind == "confirmed" and P_RANGE[0] <= k["period"] <= P_RANGE[1] and m is not None and m >= MES_MIN
        out.append({"period": k["period"], "t0": k["t0"], "duration": k["duration"], "scored": scored, "kind": kind,
                    "mes": m, "mes_method": method, "depth_ppm": k["depth_ppm"], "name": k["name"],
                    "disposition": k["disposition"],
                    "n_transits": transit_count(time, k["period"], k["t0"], k["duration"])})
    known = [Signal(s["period"], s["t0"], s["duration"]) for s in out]
    for t in tces:
        if not finite(t["period"], t["t0"], t["duration"]) or t["period"] <= 0 or t["duration"] <= 0:
            continue
        if any(compare(s, Candidate(t["period"], t["t0"]), coverage) > 0 for s in known):
            continue
        m, method = real_mes(time, flux, t, ld)
        out.append({"period": t["period"], "t0": t["t0"], "duration": t["duration"], "scored": False, "kind": "tce",
                    "mes": m, "mes_method": method, "depth_ppm": t["depth_ppm"], "name": t["name"],
                    "disposition": "TCE", "n_transits": transit_count(time, t["period"], t["t0"], t["duration"])})
    return out


def eclipse_duration(time, flux, period, t0):
    """Primary eclipse width for an EB without a catalogue duration (Kirk 2016), measured on the folded year.

    Bins of half a cadence within a quarter period (at most 1 d) of the epoch, smoothed over 3 bins; the eclipse is
    the run of bins around the deepest one within 1.5 h of the epoch whose deficit exceeds ECLIPSE_EDGE of that
    depth and 2 sigma, and its width is the duration. Its ratio to the catalogue duration on KOI EBs is reported in
    results/m2_build.json (reserve_eb). Returns (hours, depth, centre offset in hours), or None when the eclipse is
    not seen at SNR 10.
    """
    resid = flux / running_median(time, flux) - 1.0
    sigma = 1.4826 * mad(resid)
    dt = phase(time, period, t0)
    width = CADENCE_DAYS / 2
    nb = int(math.ceil(min(0.25 * period, 1.0) / width))
    idx = np.floor(dt / width).astype(int) + nb
    ok = (idx >= 0) & (idx < 2 * nb)
    meds = np.full(2 * nb, np.nan)
    counts = np.bincount(idx[ok], minlength=2 * nb)
    order = np.flatnonzero(ok)[np.argsort(idx[ok], kind="stable")]
    bins, starts = np.unique(idx[order], return_index=True)
    for i, chunk in zip(bins, np.split(resid[order], starts[1:])):
        meds[i] = np.median(chunk)
    smooth = np.array([np.nanmedian(meds[max(i - 1, 0):i + 2]) if np.isfinite(meds[max(i - 1, 0):i + 2]).any()
                       else np.nan for i in range(2 * nb)])
    centres = (np.arange(2 * nb) - nb + 0.5) * width
    near = np.flatnonzero((np.abs(centres) < 1.5 / 24) & np.isfinite(smooth))
    if near.size == 0:
        return None
    c = int(near[np.argmin(smooth[near])])
    depth = -float(smooth[c])
    if not depth > 0:
        return None
    floor = 2.0 * 1.2533 * sigma / np.sqrt(np.maximum(counts, 1))

    def deep(i):
        return 0 <= i < 2 * nb and np.isfinite(smooth[i]) and -smooth[i] > max(ECLIPSE_EDGE * depth, floor[i])

    left, right = c, c
    while deep(left - 1):
        left -= 1
    while deep(right + 1):
        right += 1
    duration = (right - left + 1) * width
    centre = 0.5 * (centres[left] + centres[right])
    inside = np.abs(dt - centre) < duration / 2
    if inside.sum() < 3 or sigma <= 0:
        return None
    snr = -float(resid[inside].mean()) / (sigma / math.sqrt(inside.sum()))
    if snr < 10:
        return None
    return duration * 24.0, depth, centre * 24.0


def eb_truth(time, flux, eph: dict) -> dict:
    """Ephemeris, duration and in-window evidence of an EB (inject.eb_tests); ephemeris fields None when it is not
    usable here. secondary_phase is the measured secondary when it is significant (eccentric orbits included)."""
    period, t0, duration, source = eph["period"], eph["t0"], eph.get("duration"), eph["source"]
    measured = eclipse_duration(time, flux, period, t0) if finite(period, t0) and period > 0 else None
    if source == "kirk":
        duration = measured[0] if measured else None
    usable = finite(period, t0, duration) and period > 0 and duration > 0
    flags = {"secondary": False, "odd_even": False, "too_deep": False}
    if not usable:
        return {"period": None, "t0": None, "duration": None, "evidence": flags, "source": source, "depth": None,
                "secondary_phase": None, "secondary_z": None, "odd_even_z": None, "calibrated": False,
                "measured_duration": None, "centre_offset_h": None}
    tests = eb_tests(time, flux, period, t0, duration)
    resid = flux / running_median(time, flux) - 1.0
    core = np.abs(phase(time, period, t0)) < max(duration / 96.0, CADENCE_DAYS)
    depth = -float(resid[core].mean()) if core.any() else None
    return {"period": period, "t0": t0, "duration": duration, "evidence": {k: tests[k] for k in flags},
            "source": source, "depth": depth, "secondary_phase": tests["secondary_phase"],
            "secondary_z": tests["secondary_z"], "odd_even_z": tests["odd_even_z"],
            "calibrated": tests["calibrated"], "measured_duration": measured[0] if measured else None,
            "centre_offset_h": measured[2] if measured else None}


def signal_mask(time, signals: list[dict], eb: dict | None) -> np.ndarray:
    """Cadences within half a duration plus MASK_PAD_HOURS of any truth signal, and of an EB's eclipses: the primary,
    phase 0.5 and the measured secondary phase when there is one."""
    mask = np.zeros(np.asarray(time).size, dtype=bool)
    for s in signals:
        mask |= np.abs(phase(time, s["period"], s["t0"])) < s["duration"] / 48.0 + MASK_PAD_HOURS / 24.0
    if eb and eb.get("duration"):
        half = eb["duration"] / 48.0 + MASK_PAD_HOURS / 24.0
        dt = np.abs(phase(time, eb["period"], eb["t0"]))
        mask |= (dt < half) | (np.abs(dt - 0.5 * eb["period"]) < half)
        if eb.get("secondary_phase") is not None:
            mask |= np.abs(phase(time, eb["period"], eb["t0"] + eb["secondary_phase"] * eb["period"])) < half
    return mask


def remove_orbit(time, flux, period, t0) -> np.ndarray:
    """Flux with a least-squares series of ORBIT_HARMONICS harmonics of an orbital period divided out (ellipsoidal
    variation, reflection and beaming of a binary)."""
    ph = 2.0 * math.pi * (np.asarray(time) - t0) / period
    harmonics = [g(k * ph) for k in range(1, ORBIT_HARMONICS + 1) for g in (np.cos, np.sin)]
    A = np.column_stack([np.ones_like(ph)] + harmonics)
    coef, *_ = np.linalg.lstsq(A, flux - 1.0, rcond=None)
    return flux - A[:, 1:] @ coef[1:]


def star_features(time, flux, quarter, mask, stellar: dict, orbit: dict | None = None) -> dict:
    """Non-signal features of one star: what a pipeline could read off it besides the transits and eclipses.

    The delivered stellar parameters, cadence and quarter counts, and noise and variability measured with the truth
    signals masked: point-to-point scatter, duration-binned robust scatter at 1, 3 and 6 h after detrending, red
    noise (3-h and 6-h scatter over their white-noise expectation), slow variability (5-95% range of a 1-day
    running median), total robust scatter, the noisiest over the quietest quarter, and the share of 5-sigma upward
    outliers. With an orbit (an EB's period and t0), its out-of-eclipse harmonics are removed first (a diagnostic).
    """
    keep = ~mask
    t, f, q = time[keep], flux[keep], quarter[keep]
    if orbit:
        f = remove_orbit(t, f, orbit["period"], orbit["t0"])
    p2p = 1.4826 * mad(np.diff(f)) / math.sqrt(2.0)
    trend = running_median(t, f, 1.0)
    scatter = {h: robust_scatter(t, f, h) for h in (1.0, 3.0, 6.0)}
    per_quarter = [1.4826 * mad(np.diff(f[q == i])) for i in np.unique(q) if (q == i).sum() > 100]
    return {
        **{k: stellar[k] for k in STELLAR_KEYS},
        "n_cadences": float(np.asarray(time).size), "n_quarters": float(np.unique(quarter).size),
        "p2p": p2p, "scatter_1h": scatter[1.0], "scatter_3h": scatter[3.0], "scatter_6h": scatter[6.0],
        "red_noise_3h": scatter[3.0] / (p2p / math.sqrt(3.0 / 24.0 / CADENCE_DAYS)) if p2p > 0 else math.nan,
        "red_noise": scatter[6.0] / (p2p / math.sqrt(6.0 / 24.0 / CADENCE_DAYS)) if p2p > 0 else math.nan,
        "var_slow": float(np.percentile(trend, 95) - np.percentile(trend, 5)),
        "var_mad": 1.4826 * mad(f),
        "quarter_noise_ratio": max(per_quarter) / min(per_quarter) if per_quarter and min(per_quarter) > 0 else 1.0,
        "pos_outliers": float(np.mean(f / trend - 1.0 > 5.0 * p2p)),
    }


def stratum(signals: list[dict]) -> str | None:
    scored = [s["mes"] for s in signals if s["scored"]]
    if not scored:
        return None
    return "real_7_12" if min(scored) < REAL_SPLIT else "real_12_up"


def measure_star(job: dict) -> dict:
    """Everything select needs about one star, on its light curve before delivery (BKJD, no injection)."""
    time, flux, quarter, _ = load_lc(job["kic"])
    out = {"lc_sha256": job["lc_sha256"], "key": job["key"], "role": job["role"], "pool": job["pool"],
           "list": job["list"], "n_cadences": int(time.size), "quarters": sorted(int(q) for q in np.unique(quarter))}
    signals, eb = [], None
    if job["role"] == "real":
        signals = real_signals(time, flux, job["kois"], job["tces"], job["ld"])
        out["signals"] = signals
        out["stratum"] = stratum(signals)
    elif job["role"] == "eb":
        eb = eb_truth(time, flux, job["eph"])
        out["eb"] = eb
    out["features"] = star_features(time, flux, quarter, signal_mask(time, signals, eb), job["stellar"])
    return out


# ---- fetch and measure ----------------------------------------------------------------------------------------------

def pool_members(pools: dict, pool: str) -> list[tuple[int, str]]:
    return [(int(k), name) for name in POOL_LISTS for k in pools[name][pool]]


def cmd_fetch(args) -> None:
    """Download dev, every REAL reserve host (their strata need the window MES) and the selected reserve stars."""
    pools = json.loads(POOLS.read_text())
    reserve = json.loads(SELECTION.read_text())["fetch_reserve"]
    for kics, manifest in (([k for k, _ in pool_members(pools, "dev")], DEV_LIGHTCURVES),
                           (reserve, PRIVATE / "lightcurves.json")):
        fetch_lightcurves.fetch([{"kic": k, "kind": "mast"} for k in kics], manifest, args.workers)


def measure_jobs(pools: dict, tables: dict, wanted: set[int]) -> list[dict]:
    stellar = stellar_rows(tables["stellar"])
    koi, tce = tables["koi"], tables["tce"]
    koi_rows, tce_rows = rows_by_kic(koi), rows_by_kic(tce)
    koi_ebs, kirk = eb_ephemerides(koi), kirk_ephemerides(tables["kirk"])
    ks, ts = tables["koi_shape"], tables["tce_shape"]
    koi_shape = {n: (float(r), float(b)) for n, r, b in zip(ks["kepoi_name"], ks["ror"], ks["impact"])}
    tce_shape = {(int(k), int(n)): (float(r), float(b))
                 for k, n, r, b in zip(ts["kepid"], ts["tce_plnt_num"], ts["ror"], ts["impact"])}
    jobs = []
    for pool in ("dev", "reserve"):
        for kic, name in pool_members(pools, pool):
            path = LC / f"{kic}.npz"
            if kic not in wanted or not path.exists() or "qmedian" not in np.load(path).files:
                continue
            role = "host" if name in ("random", "tce_only") else "eb" if name == "eb" else "real"
            job = {"kic": kic, "pool": pool, "list": name, "role": role,
                   "stellar": {k: round_sig(stellar[kic][k]) for k in STELLAR_KEYS}}
            if role == "real":
                job["ld"] = stellar[kic]["ld"]
                job["kois"] = [{"name": koi["kepoi_name"][i], "disposition": koi["koi_disposition"][i],
                                "period": float(koi["period_d"][i]), "t0": float(koi["epoch_bkjd"][i]),
                                "duration": float(koi["dur_hr"][i]), "depth_ppm": float(koi["depth_ppm"][i]),
                                **dict(zip(("ror", "impact"), koi_shape.get(koi["kepoi_name"][i], (None, None))))}
                               for i in koi_rows.get(kic, [])]
                job["tces"] = [{"name": f"TCE {int(tce['tce_plnt_num'][i])}", "period": float(tce["period_d"][i]),
                                "t0": float(tce["epoch_bkjd"][i]), "duration": float(tce["dur_hr"][i]),
                                "depth_ppm": float(tce["depth_ppm"][i]),
                                **dict(zip(("ror", "impact"),
                                           tce_shape.get((kic, int(tce["tce_plnt_num"][i])), (None, None))))}
                               for i in tce_rows.get(kic, [])]
            elif role == "eb":
                if kic in koi_ebs:
                    e = koi_ebs[kic]
                    job["eph"] = {"period": e["period"], "t0": e["t0"], "duration": e["duration"], "source": "koi"}
                else:
                    e = kirk[kic]
                    job["eph"] = {"period": e["period"], "t0": e["t0"], "duration": None, "source": "kirk"}
            job["lc_sha256"] = lc_sha256(kic)
            job["key"] = sha256_bytes(json.dumps([MEASURE_VERSION, job], sort_keys=True, default=str).encode())
            jobs.append(job)
    return jobs


def wanted_stars(pools: dict) -> set[int]:
    return {k for k, _ in pool_members(pools, "dev")} | set(json.loads(SELECTION.read_text())["fetch_reserve"])


def cmd_measure(args) -> None:
    pools = json.loads(POOLS.read_text())
    wanted = wanted_stars(pools)
    jobs = measure_jobs(pools, load_tables(), wanted)
    cache = json.loads(MEASURED.read_text())["stars"] if MEASURED.exists() else {}
    todo = [j for j in jobs if cache.get(str(j["kic"]), {}).get("key") != j["key"]]
    print(f"measuring {len(todo)} of {len(jobs)} stars", file=sys.stderr)
    with ProcessPoolExecutor(args.workers) as pool:
        for job, result in zip(todo, pool.map(measure_star, todo, chunksize=4)):
            cache[str(job["kic"])] = result
    write_json(MEASURED, {"version": MEASURE_VERSION, "stars": cache})
    print(f"{len(cache)} measured; {len(wanted) - len(jobs)} wanted stars still without a light curve",
          file=sys.stderr)


# ---- select ---------------------------------------------------------------------------------------------------------

def table_features(stellar) -> dict[int, list[float]]:
    """Per KIC, the stellar-table columns the populations are matched on: the four delivered parameters as rounded,
    Kepler magnitude, and log CDPP at 3 and 6 h."""
    out = {}
    for i, k in enumerate(stellar["kepid"]):
        cdpp = [stellar[c][i] for c in ("rrmscdpp03p0", "rrmscdpp06p0")]
        out[int(k)] = [round_sig(stellar[c][i]) for c in STELLAR_KEYS] + [float(stellar["kepmag"][i])] + \
                      [math.log10(c) if c > 0 else math.nan for c in cdpp]
    return out


def propensity(groups: dict[str, list[int]], features: dict[int, list[float]]) -> dict[int, float]:
    """Out-of-fold probability that a star belongs to its own group (host, eb, real) given its table features, with
    every group weighted equally. High means easy to tell apart."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    names = list(groups)
    kics = [k for g in names for k in groups[g]]
    y = np.array([i for i, g in enumerate(names) for _ in groups[g]])
    X = np.array([features[k] for k in kics], dtype=float)
    weight = np.array([len(kics) / (len(names) * len(groups[names[i]])) for i in y])
    prob = np.zeros((len(kics), len(names)))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                                               early_stopping=False, random_state=0)
        model.fit(X[train], y[train], sample_weight=weight[train])
        prob[test] = model.predict_proba(X[test])
    return {k: float(prob[i, y[i]]) for i, k in enumerate(kics)}


def split_roles(kics: list[int], quotas: list[tuple[str, str, int]], base: str, tag: str) -> dict:
    """Assign stars to (sample, cell) slots in the order of a seeded hash; returns {sample: {cell: [kics]}}."""
    order = sorted(kics, key=lambda k: unit(base, "split", tag, k))
    out: dict[str, dict[str, list[int]]] = {}
    i = 0
    for sample, cell, n in quotas:
        out.setdefault(sample, {})[cell] = sorted(order[i:i + n])
        i += n
    return out


def merge(into: dict, part: dict) -> None:
    for sample, cells in part.items():
        into.setdefault(sample, {}).update(cells)


def eb_mix(kics, koi_ebs: dict, kirk: dict) -> dict:
    """Catalogue depth mix and period quartiles of a set of EBs (Kirk-only EBs have no catalogue depth)."""
    bins = depth_bins(koi_ebs[k]["depth_ppm"] * 1e-6 if k in koi_ebs else None for k in kics)
    bins["no_catalogue_depth"] = bins.pop("not measured")
    periods = [koi_ebs[k]["period"] if k in koi_ebs else kirk[k]["period"] for k in kics]
    return {"stars": len(kics), "depth_bins": bins,
            "period_quartiles_d": [round_sig(float(x), 3) for x in np.percentile(periods, [25, 50, 75])]}


def real_cells(kics: list[int], measured: dict) -> dict[str, list[int]] | None:
    """REAL hosts by the stratum of their window MES; None until every one of them is measured."""
    if any(str(k) not in measured for k in kics):
        return None
    cells = {"real_7_12": [], "real_12_up": [], "real_below_7_1": []}
    for k in kics:
        cells[measured[str(k)]["stratum"] or "real_below_7_1"].append(k)
    return {c: sorted(v) for c, v in cells.items()}


def select_dev(pools: dict, drops: set[int], measured: dict, usable) -> dict:
    """Dev is the dev pool lists as they are, minus audit drops and stars without a DR25 mass and radius; roles and
    REAL strata as in the test samples."""
    dev: dict[str, dict] = {}
    for name in ("random", "tce_only"):
        kics = [k for k in pools[name]["dev"] if k not in drops and usable(k)]
        n_inj = DEV_CELLS[f"inj_{name}"]
        merge(dev, split_roles(kics, [("dev", f"inj_{name}", n_inj), ("dev", f"null_{name}", len(kics) - n_inj)],
                               DEV_SEED, name))
    dev["dev"]["eb"] = sorted(k for k in pools["eb"]["dev"] if usable(k))
    real = real_cells([k for k in pools["real_7_12"]["dev"] + pools["real_12_up"]["dev"] if usable(k)], measured)
    dev["dev"].update(real or {"real_pending": sorted(pools["real_7_12"]["dev"] + pools["real_12_up"]["dev"])})
    return dev["dev"]


def cmd_select(args) -> None:
    """Steer the test draws before download: trim the host and REAL reserves to the stars whose table features look
    least like their own population, draw the EBs at random (their CDPP carries their eclipses, so trimming them
    would drop the deep, short-period ones and change the natural depth mix), then split A and B by the release
    secret. REAL strata need the window MES, so REAL is completed on a second run, after measure. Stars without a
    finite DR25 mass and radius are left out of every population: an injection needs them, and a NaN parameter
    left only on EB or REAL stars would label them."""
    secret = release_secret(args.new_secret)
    pools = json.loads(POOLS.read_text())
    measured = json.loads(MEASURED.read_text())["stars"] if MEASURED.exists() else {}
    drops, audit = audit_drops()
    tables = load_tables()
    features = table_features(tables["stellar"])
    stellar = stellar_rows(tables["stellar"])

    def usable(k):
        return finite(stellar[k]["mass"], stellar[k]["radius"])

    reserve = {name: list(pools[name]["reserve"]) for name in POOL_LISTS}
    real = reserve["real_7_12"] + reserve["real_12_up"]
    groups = {"host": reserve["random"] + reserve["tce_only"], "eb": reserve["eb"], "real": real}
    score = propensity(groups, features)

    def keep(kics, n, steer=True):
        """The n stars least identifiable as their own population, or n at random; the hash breaks exact ties."""
        return sorted(kics, key=lambda k: (score[k] if steer else 0.0, unit(secret, "draw", k)))[:n]

    test: dict[str, dict] = {}
    used = {}
    for name in ("random", "tce_only"):
        eligible = [k for k in reserve[name] if k not in drops and usable(k)]
        chosen = keep(eligible, 2 * (TEST_CELLS[f"inj_{name}"] + TEST_CELLS[f"null_{name}"]))
        used[name] = {"reserve": len(reserve[name]), "dropped_by_audit": sum(k in drops for k in reserve[name]),
                      "no_mass_or_radius": sum(not usable(k) for k in reserve[name]), "selected": len(chosen)}
        quotas = [(s, f"{role}_{name}", TEST_CELLS[f"{role}_{name}"]) for s in SAMPLES for role in ("inj", "null")]
        merge(test, split_roles(chosen, quotas, secret, name))
    cells = {"eb": [k for k in reserve["eb"] if usable(k)]}
    by_stratum = real_cells([k for k in real if usable(k)], measured)
    if by_stratum:
        cells.update({c: by_stratum[c] for c in ("real_7_12", "real_12_up")})
    for cell, kics in cells.items():
        n = 2 * TEST_CELLS[cell]
        chosen = keep(kics, n, steer=cell != "eb")
        used[cell] = {"reserve": len(kics), "needed": n, "selected": len(chosen),
                      "steered": cell != "eb" and len(kics) > n}
        half = len(chosen) // 2
        merge(test, split_roles(chosen, [("A", cell, half), ("B", cell, len(chosen) - half)], secret, cell))
    used["eb"]["no_mass_or_radius"] = len(reserve["eb"]) - len(cells["eb"])
    if by_stratum:
        used["real_below_7_1"] = {"reserve": len(by_stratum["real_below_7_1"]), "selected": 0}

    koi_ebs, kirk = eb_ephemerides(tables["koi"]), kirk_ephemerides(tables["kirk"])
    drawn_eb = test["A"]["eb"] + test["B"]["eb"]
    counterfactual = {"reserve": eb_mix(reserve["eb"], koi_ebs, kirk), "drawn": eb_mix(drawn_eb, koi_ebs, kirk),
                      "if_steered": eb_mix(keep(reserve["eb"], len(drawn_eb)), koi_ebs, kirk)}
    dev = select_dev(pools, drops, measured, usable)
    chosen = {k for cells in test.values() for v in cells.values() for k in v}
    order = [k for c in ("real_7_12", "real_12_up", "eb", "tce_only", "random") for k in reserve[c]
             if k in chosen or c.startswith("real")]
    doc = {"release": RELEASE, "complete": by_stratum is not None and "real_pending" not in dev, "dev": dev, **test,
           "audit": audit, "used": used, "eb_steering_counterfactual": counterfactual, "fetch_reserve": order,
           "propensity": {str(k): v for k, v in score.items()},
           "trimmed": {g: sorted(k for k in v if k not in chosen and k not in drops) for g, v in groups.items()}}
    write_json(SELECTION, doc)
    print(json.dumps({"complete": doc["complete"], "dev": {c: len(v) for c, v in dev.items()},
                      **{s: {c: len(v) for c, v in test[s].items()} for s in SAMPLES}, "used": used,
                      "audit": audit, "fetch_reserve": len(order)}, indent=1))


# ---- build ----------------------------------------------------------------------------------------------------------

POPULATION = {"inj_random": ("inj", "random"), "inj_tce_only": ("inj", "tce_only"),
              "null_random": ("null", "random"), "null_tce_only": ("null", "tce_only"), "eb": ("eb", "eb"),
              "real_7_12": ("real", "real"), "real_12_up": ("real", "real"), "real_below_7_1": ("real", "real")}

_EB_PERIODS: list[float] = []
_EB_PERIOD_SET: set[float] = set()


def _init_worker(eb_periods):
    _EB_PERIODS[:] = eb_periods
    _EB_PERIOD_SET.update(eb_periods)


def shift(signal: dict, offset: float) -> dict:
    return {**signal, "t0": signal["t0"] + offset}


def build_star(job: dict) -> dict:
    """Write one delivered star file and return its truth entry, map entry and file hash, or failed=True when an INJ
    star cannot reach its assigned MES target (nothing is written then)."""
    kic, cell = job["kic"], job["cell"]
    population, host = POPULATION[cell]
    time, flux, quarter, qmedian = load_lc(kic)
    offset = float(np.random.default_rng(seed_of(job["base"], "offset", kic)).uniform(-OFFSET_DAYS, OFFSET_DAYS))
    entry = {"population": population, "host": host, "scored": population in ("inj", "null"), "coverage": None,
             "signals": []}
    mapped = {"kic": kic, "cell": cell, "time_offset": offset}
    drop = None
    if population == "inj":
        inject_seed = seed_of(job["base"], "inject", kic)
        try:
            _, sig = draw_injection(kic, time, flux, job["star"], np.random.default_rng(inject_seed), _EB_PERIODS,
                                    target=job["target"])
        except RuntimeError:
            return {"star_id": job["star_id"], "failed": True}
        drop = transit_drop(time, sig["period"], sig["t0"], sig["duration"], sig["ror"], sig["b"], job["star"]["ld"])
        entry["signals"] = [shift({"period": sig["period"], "t0": sig["t0"], "duration": sig["duration"],
                                   "scored": True, "kind": "injected", "mes": sig["mes"],
                                   **{k: sig[k] for k in ("mes_target", "ror", "b", "e", "omega", "depth_ppm")},
                                   "eb_period_copy": sig["period"] in _EB_PERIOD_SET}, offset)]
        mapped["inject_seed"] = inject_seed
    delivered_time = time + offset
    delivered_flux = deliver_flux(flux, quarter, qmedian, drop)
    eb = None
    if population == "real":
        kois = [shift(k, offset) for k in job["kois"]]
        tces = [shift(t, offset) for t in job["tces"]]
        entry["signals"] = real_signals(delivered_time, delivered_flux, kois, tces, job["ld"])
        mapped["stratum"] = stratum(entry["signals"])
    elif population == "eb":
        eph = {**job["eph"], "t0": job["eph"]["t0"] + offset}
        eb = eb_truth(delivered_time, delivered_flux, eph)
        entry["eb"] = {k: eb[k] for k in ("period", "t0", "duration", "evidence", "source", "secondary_phase")}
        entry["scored"] = any(eb["evidence"].values())
        mapped.update(eb_depth=eb["depth"], secondary_z=eb["secondary_z"], odd_even_z=eb["odd_even_z"],
                      calibrated=eb["calibrated"])
    mask = signal_mask(delivered_time, entry["signals"], entry.get("eb"))
    mapped["variability"] = star_features(delivered_time, delivered_flux, quarter, mask, job["stellar"])[VARIABILITY]
    if not finite(mapped["variability"]):  # eclipses cover nearly the whole orbit: no 3-h bin survives the mask
        mapped["variability"] = star_features(delivered_time, delivered_flux, quarter, np.zeros_like(mask),
                                              job["stellar"])[VARIABILITY]
        mapped["variability_unmasked"] = True
    entry["coverage"] = coverage_from_times(delivered_time, max_gap=0.5).tolist()
    first = int(quarter.min())
    guess = delivered_time[quarter == first].min() - QUARTERS_BKJD[WINDOW[first]][0]
    mapped["offset_guess_error_d"] = float(guess - offset)
    path = Path(job["path"])
    write_star(path, {"time": delivered_time, "flux": delivered_flux, "quarter": quarter.astype(np.int8),
                      **{k: np.float64(job["stellar"][k]) for k in STELLAR_KEYS}})
    return {"star_id": job["star_id"], "truth": entry, "map": mapped, "sha256": sha256_file(path)}


def assign_targets(sample: str, cells: dict, base: str) -> dict[int, float]:
    """Stratified MES targets: within each INJ host cell of n stars, target_i = 4 + 16 (i + u_i) / n with
    u_i ~ U(0, 1), assigned to the stars in a seeded random order."""
    out = {}
    for cell in INJ_CELLS:
        kics = sorted(cells.get(cell, []), key=lambda k: unit(base, "target-order", sample, cell, k))
        for i, k in enumerate(kics):
            out[k] = 4.0 + 16.0 * (i + unit(base, "target-jitter", sample, cell, i)) / len(kics)
    return out


def build_sample(pool, jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Build every star; an INJ star that cannot reach its target swaps targets with the star of its cell holding
    the lowest target not yet swapped (both rebuilt; failures handled in star-id order)."""
    results = {r["star_id"]: r for r in pool.map(build_star, jobs, chunksize=2)}
    swaps = []
    for job in sorted((j for j in jobs if results[j["star_id"]].get("failed")), key=lambda j: j["star_id"]):
        partners = sorted((j for j in jobs if j["cell"] == job["cell"] and j is not job and not j.get("swapped")
                           and not results[j["star_id"]].get("failed")), key=lambda j: j["target"])
        for partner in partners:
            a, b = {**job, "target": partner["target"]}, {**partner, "target": job["target"]}
            ra, rb = pool.map(build_star, [a, b])
            if not ra.get("failed") and not rb.get("failed"):
                swaps.append({"cell": job["cell"], "targets": [job["target"], partner["target"]]})
                job.update(target=a["target"], swapped=True)
                partner.update(target=b["target"], swapped=True)
                results[job["star_id"]], results[partner["star_id"]] = ra, rb
                break
        else:
            raise SystemExit(f"{job['star_id']}: no target swap in its cell reaches both targets")
    return [results[j["star_id"]] for j in jobs], swaps


def mark_quiet(results: list[dict]) -> float:
    """Give each EB its variability and quiet flag: at or below the QUIET_PERCENTILE of the sample's INJ and NULL
    hosts. Variability is the 3-h red noise with the truth signals masked; when the eclipse mask leaves no full 3-h bin,
    it is measured on the whole light curve. Returns the threshold."""
    hosts = [r["map"]["variability"] for r in results if r["truth"]["population"] in ("inj", "null")]
    threshold = float(np.nanpercentile(hosts, QUIET_PERCENTILE))
    for r in results:
        if r["truth"]["population"] == "eb":
            v = r["map"]["variability"]
            r["truth"]["eb"].update(variability=v, quiet=bool(finite(v) and v <= threshold))
    return threshold


def star_ids(kics: list[int], base: str, sample: str) -> dict[int, str]:
    order = sorted(kics, key=lambda k: unit(base, "id", sample, k))
    if sample == "dev":
        return {k: f"dev-{i:03d}" for i, k in enumerate(order)}
    return {k: f"{RELEASE}-{sample}-{i:04d}" for i, k in enumerate(order)}


def sample_jobs(sample: str, cells: dict, base: str, tables_ctx: dict, measure_jobs_by_kic: dict) -> list[dict]:
    kics = [k for v in cells.values() for k in v]
    ids = star_ids(kics, base, sample)
    targets = assign_targets(sample, cells, base)
    folder = BUILT / "dev" if sample == "dev" else BUILT / RELEASE / sample
    jobs = []
    for cell, members in cells.items():
        for kic in members:
            m = measure_jobs_by_kic[kic]
            jobs.append({"kic": kic, "cell": cell, "base": base, "star_id": ids[kic],
                         "path": str(folder / f"{ids[kic]}.npz"), "stellar": m["stellar"],
                         "star": tables_ctx["stellar"][kic], "kois": m.get("kois", []), "tces": m.get("tces", []),
                         "ld": m.get("ld"), "eph": m.get("eph"), "target": targets.get(kic)})
    return sorted(jobs, key=lambda j: j["star_id"])


def files_digest(files: dict[str, str]) -> str:
    return sha256_bytes("".join(f"{sid} {h}\n" for sid, h in sorted(files.items())).encode())


def write_sample(sample: str, results: list[dict]) -> dict:
    from transitbench.score import load_truth

    truth = {"release": "dev" if sample == "dev" else RELEASE, "sample": sample, "window": WINDOW,
             "stars": {r["star_id"]: r["truth"] for r in results}}
    mapping = {r["star_id"]: r["map"] for r in results}
    files = {r["star_id"]: r["sha256"] for r in results}
    folder = DEV_DIR if sample == "dev" else PRIVATE
    suffix = "" if sample == "dev" else f"_{sample}"
    truth_path = folder / f"truth{suffix}.json"
    write_json(truth_path, truth)
    write_json(folder / f"map{suffix}.json", {"release": truth["release"], "sample": sample, "stars": mapping})
    write_json(folder / f"files{suffix}.json", {"release": truth["release"], "sample": sample,
                                                 "digest": files_digest(files), "files": files})
    load_truth(truth_path)
    return {"truth": truth, "map": mapping, "truth_sha256": sha256_file(truth_path),
            "files_digest": files_digest(files), "files": files}


def quantiles(values, private: bool = False) -> dict:
    """Percentiles of the finite values. Private (test) samples get only p10-p90, rounded to 3 significant figures,
    so no committed number pins down one test star; dev keeps exact values with min and max."""
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0}
    names = ("p10", "p25", "p50", "p75", "p90") if private else ("min", "p10", "p25", "p50", "p75", "p90", "max")
    qs = np.percentile(v, [0.0 if n == "min" else 100.0 if n == "max" else float(n[1:]) for n in names])
    return {"n": int(v.size), **{n: round_sig(float(x), 3) if private else float(x) for n, x in zip(names, qs)}}


def mes_histogram(mes_values) -> dict:
    counts, _ = np.histogram([m for m in mes_values], bins=MES_HIST_EDGES)
    return {f"{lo:g}-{hi:g}": int(c) for lo, hi, c in zip(MES_HIST_EDGES[:-1], MES_HIST_EDGES[1:], counts)}


def sample_summary(built: dict, private: bool, threshold: float, swaps: list[dict]) -> dict:
    truth, mapping = built["truth"]["stars"], built["map"]
    cells: dict[str, int] = {}
    for m in mapping.values():
        cells[m["cell"]] = cells.get(m["cell"], 0) + 1
    pops: dict[str, int] = {}
    for t in truth.values():
        pops[t["population"]] = pops.get(t["population"], 0) + 1
    ebs = [t for t in truth.values() if t["population"] == "eb"]
    inj = {c: [truth[s]["signals"][0] for s, m in mapping.items() if m["cell"] == c] for c in INJ_CELLS}
    inj_all = inj[INJ_CELLS[0]] + inj[INJ_CELLS[1]]
    real = [t for t in truth.values() if t["population"] == "real"]
    real_scored = [s for t in real for s in t["signals"] if s["scored"]]
    real_all = [s for t in real for s in t["signals"]]
    return {
        "stars": len(truth), "cells": cells, "populations": pops,
        "scored_stars": sum(t["scored"] for t in truth.values()),
        "planets_scored": sum(s["scored"] for t in truth.values() if t["scored"] for s in t["signals"]),
        "inj_mes": quantiles((s["mes"] for s in inj_all), private),
        "inj_mes_histogram": {c: mes_histogram(s["mes"] for s in v) for c, v in inj.items()},
        "inj_share_mes_ge_7_1": {**{c: sum(s["mes"] >= MES_MIN for s in v) / max(len(v), 1) for c, v in inj.items()},
                                 "all": sum(s["mes"] >= MES_MIN for s in inj_all) / max(len(inj_all), 1)},
        "inj_target_swaps": len(swaps),
        "inj_depth_above_3pct": sum(s["depth_ppm"] > 3e4 for s in inj_all),
        "inj_eb_period_copies": sum(s["eb_period_copy"] for s in inj_all),
        "inj_depth_ppm": quantiles((s["depth_ppm"] for s in inj_all), private),
        "offset_recovered_from_quarter_start_abs_error_d": quantiles(
            (abs(m["offset_guess_error_d"]) for m in mapping.values()), private),
        "eb": {"stars": len(ebs), "scored": sum(t["scored"] for t in ebs),
               "evidence_share": sum(t["scored"] for t in ebs) / max(len(ebs), 1),
               "flags": {f: sum(t["eb"]["evidence"][f] for t in ebs) for f in ("secondary", "odd_even", "too_deep")},
               "secondary_away_from_phase_0_5": sum(t["eb"]["secondary_phase"] is not None
                                                    and abs(t["eb"]["secondary_phase"] - 0.5) > 0.05 for t in ebs),
               "tests_calibrated_off_eclipse": sum(bool(m.get("calibrated")) for m in mapping.values()
                                                   if m["cell"] == "eb"),
               "quiet": sum(t["eb"]["quiet"] for t in ebs),
               "quiet_and_scored": sum(t["eb"]["quiet"] and t["scored"] for t in ebs),
               "variability_unmasked_fallback": sum(bool(m.get("variability_unmasked")) for m in mapping.values()),
               "variability_feature": VARIABILITY,
               "quiet_threshold": round_sig(threshold, 3) if private else threshold,
               "source": {s: sum(t["eb"]["source"] == s for t in ebs) for s in ("koi", "kirk")},
               "no_usable_ephemeris": sum(t["eb"]["duration"] is None for t in ebs),
               "depth_bins": depth_bins(m.get("eb_depth") for m in mapping.values() if m["cell"] == "eb"),
               "scored_depth_bins": depth_bins(mapping[s].get("eb_depth") for s, t in truth.items()
                                               if t["population"] == "eb" and t["scored"])},
        "real": {"stars": len(real), "scored_planets": len(real_scored),
                 "neutral_signals": sum(not s["scored"] for s in real_all),
                 "neutral_by_kind": {k: sum(s["kind"] == k and not s["scored"] for s in real_all)
                                     for k in ("confirmed", "koi", "tce")},
                 "mes_method": {k: sum(s["mes_method"] == k for s in real_all) for k in ("shape", "depth", None)},
                 "scored_mes": quantiles((s["mes"] for s in real_scored), private),
                 "scored_with_fewer_than_3_transits": sum(s["n_transits"] < 3 for s in real_scored),
                 "hosts_without_scored_planet": sum(not any(s["scored"] for s in t["signals"]) for t in real)},
    }


def depth_bins(depths) -> dict:
    bins = {"<1%": 0, "1-5%": 0, ">5%": 0, "not measured": 0}
    for d in depths:
        if d is None or not math.isfinite(d):
            bins["not measured"] += 1
        else:
            bins["<1%" if d < 0.01 else "1-5%" if d < 0.05 else ">5%"] += 1
    return bins


def real_recount(measured: dict, pools: dict) -> dict:
    """The pool lists' REAL strata use a four-year proxy MES; this is where the window's own MES puts their hosts."""
    out = {}
    for pool in ("reserve", "dev"):
        table = {}
        for proxy in ("real_7_12", "real_12_up"):
            row = {"real_7_12": 0, "real_12_up": 0, "below_7_1": 0, "no_light_curve": 0}
            for k in pools[proxy][pool]:
                m = measured.get(str(k))
                row["no_light_curve" if m is None else m["stratum"] or "below_7_1"] += 1
            table[f"proxy_{proxy}"] = row
        hosts = [str(k) for k in pools["real_7_12"][pool] + pools["real_12_up"][pool] if str(k) in measured]
        signals = [s for k in hosts for s in measured[k]["signals"]]
        planets = [s for s in signals if s["kind"] == "confirmed" and P_RANGE[0] <= s["period"] <= P_RANGE[1]]
        out[pool] = {"hosts_by_proxy_stratum": table,
                     "confirmed_planets_in_period_range": len(planets),
                     "of_which_mes_ge_7_1": sum(s["scored"] for s in planets),
                     "mes_method": {k: sum(s["mes_method"] == k for s in signals) for k in ("shape", "depth", None)}}
    return out


def prevalence(tables: dict, ebs: list[dict]) -> dict:
    """pi, pi_eb and the natural TCE-only share for the Q10-Q13 parent sample."""
    m0 = json.loads(M0_COUNTS.read_text())
    pops, parent = m0["populations"], m0["parent_with_window"]
    st = tables["stellar"]
    in_window = np.array([sum(q.strip('"')[i - 1] == "1" for i in WINDOW) >= 3 for q in st["st_quarters"]])
    base = (st["teff"] >= 4500) & (st["teff"] <= 6500) & (st["logg"] >= 4.0) & (st["logg"] <= 4.7) & \
           (st["kepmag"] >= 10) & (st["kepmag"] <= 15) & in_window & (st["nkoi"] == 0)
    tce = tables["tce"]
    vetted = {int(k) for k, r in zip(tce["kepid"], tce["tce_rogue_flag"]) if r == 0}
    no_koi = st["kepid"][base].astype(int)
    evidence = sum(any(e["evidence"].values()) for e in ebs) / max(len(ebs), 1)
    pi_eb = pops["eb"] / parent
    return {
        "parent_with_window": parent,
        "pi": {"confirmed": m0["pi"]["pi_confirmed"], "with_candidates": m0["pi"]["pi_with_candidates"],
               "mes": "four-year proxy; see real_recount for how the window MES moves REAL hosts"},
        "pi_eb": pi_eb,
        "pi_eb_with_in_window_evidence": pi_eb * evidence,
        "eb_evidence_share_dev_and_r1": evidence, "eb_dev_and_r1": len(ebs),
        "eb_population": pops["eb"],
        "tce_only_share": {
            "of_null_eligible": pops["tce_only"] / (pops["random"] + pops["tce_only"]),
            "null_eligible": pops["random"] + pops["tce_only"], "tce_only": pops["tce_only"],
            "of_all_no_koi_parent": float(np.mean([k in vetted for k in no_koi])), "no_koi_parent": int(no_koi.size)},
    }


def eb_evidence_white_noise(time, flux, period, t0, duration_hours) -> dict:
    """The first build's EB tests (one white-noise sigma per cadence, secondary at phase 0.5 only). Kept only to
    measure their false-fire rate on NULL stars against the current inject.eb_tests."""
    resid = flux / running_median(time, flux) - 1.0
    sigma = 1.4826 * mad(resid)
    half = duration_hours / 48.0
    dt = phase(time, period, t0)
    primary, secondary = np.abs(dt) < half, np.abs(np.abs(dt) - 0.5 * period) < half
    out = {"secondary": False, "odd_even": False, "too_deep": False}
    if primary.sum() < 3:
        return out
    out["too_deep"] = bool(-resid[primary].mean() > TOO_DEEP)
    if secondary.sum() >= 3:
        out["secondary"] = bool(-resid[secondary].mean() > 3.0 * sigma / math.sqrt(secondary.sum()))
    epoch = np.round((time - t0) / period).astype(int)
    odd, even = primary & (epoch % 2 == 1), primary & (epoch % 2 == 0)
    if odd.sum() >= 3 and even.sum() >= 3:
        diff = abs(resid[odd].mean() - resid[even].mean())
        out["odd_even"] = bool(diff > 3.0 * sigma * math.sqrt(1.0 / odd.sum() + 1.0 / even.sum()))
    return out


def null_fire(item) -> dict:
    """Both EB test versions on one NULL star at a borrowed real EB ephemeris (a hashed EB of the same sample, its
    period and duration, at a hashed phase)."""
    path, period, t0, duration = item
    d = np.load(path)
    time, flux = d["time"], d["flux"]
    return {"current": eb_tests(time, flux, period, t0, duration),
            "white_noise": eb_evidence_white_noise(time, flux, period, t0, duration)}


def null_false_fire(pool, sample: str, built: dict) -> dict:
    truth = built["truth"]["stars"]
    ebs = [t["eb"] for _, t in sorted(truth.items()) if t["population"] == "eb" and t["eb"]["duration"]]
    folder = BUILT / "dev" if sample == "dev" else BUILT / RELEASE / sample
    items = []
    for sid, t in sorted(truth.items()):
        if t["population"] == "null":
            eb = ebs[int(unit("null-fire", sample, sid) * len(ebs))]
            t0 = t["coverage"][0][0] + unit("null-fire-phase", sample, sid) * eb["period"]
            items.append((str(folder / f"{sid}.npz"), eb["period"], t0, eb["duration"]))
    runs = list(pool.map(null_fire, items, chunksize=4))
    out = {"null_stars": len(runs)}
    for version in ("current", "white_noise"):
        out[version] = {f: sum(r[version][f] for r in runs) / len(runs) for f in ("secondary", "odd_even", "too_deep")}
        out[version]["any"] = sum(any(r[version][f] for f in ("secondary", "odd_even", "too_deep"))
                                  for r in runs) / len(runs)
    return out


def cmd_build(args) -> None:
    tables = load_tables()
    pools = json.loads(POOLS.read_text())
    selection = json.loads(SELECTION.read_text())
    measured = json.loads(MEASURED.read_text())["stars"]
    secret = release_secret()
    if selection["audit"] != audit_drops()[1]:
        raise SystemExit("the audit drop list changed since select; rerun select")
    if not selection["complete"]:
        raise SystemExit("selection is incomplete (REAL strata need measure); run measure, then select again")
    jobs_by_kic = {j["kic"]: j for j in measure_jobs(pools, tables, wanted_stars(pools))}
    missing = [k for s in ("dev", *SAMPLES) for v in selection[s].values() for k in v if k not in jobs_by_kic]
    if missing:
        raise SystemExit(f"{len(missing)} selected stars have no light curve; run fetch and measure")
    stellar = stellar_rows(tables["stellar"])
    eb_periods = injection_eb_periods(tables["koi"], tables["kirk"])
    ctx = {"stellar": stellar}
    built, summaries, fire = {}, {}, {}
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(eb_periods,)) as pool:
        for sample in ("dev", *SAMPLES):
            base = DEV_SEED if sample == "dev" else secret
            jobs = sample_jobs(sample, selection[sample], base, ctx, jobs_by_kic)
            folder = Path(jobs[0]["path"]).parent
            for old in folder.glob("*.npz") if folder.exists() else []:
                old.unlink()
            results, swaps = build_sample(pool, jobs)
            threshold = mark_quiet(results)
            built[sample] = write_sample(sample, results)
            mismatch = sum(r["map"].get("stratum") != r["map"]["cell"] for r in results
                           if r["map"]["cell"] in ("real_7_12", "real_12_up"))
            if mismatch:
                raise SystemExit(f"{sample}: {mismatch} REAL hosts changed stratum between measure and build")
            summaries[sample] = sample_summary(built[sample], sample != "dev", threshold, swaps)
            fire[sample] = null_false_fire(pool, sample, built[sample])
            print(f"{sample}: {len(results)} stars, {len(swaps)} target swaps", file=sys.stderr)

    secret_sha = sha256_bytes(secret.encode())
    write_json(RELEASE_MANIFEST, {
        "release": RELEASE, "window": WINDOW, "secret_sha256": secret_sha,
        "pool_salt_sha256": (ROOT / "manifests" / "pool_salt.sha256").read_text().strip(),
        "tce_audit_sha256": selection["audit"]["sha256"],
        "reserve_lightcurves_manifest_sha256": sha256_file(PRIVATE / "lightcurves.json"),
        "samples": {s: {"stars": summaries[s]["stars"], "populations": summaries[s]["populations"],
                        "cells": summaries[s]["cells"], "truth_sha256": built[s]["truth_sha256"],
                        "star_files_sha256": built[s]["files_digest"]} for s in SAMPLES},
        "star_files_sha256_rule": "SHA-256 of the lines '<star_id> <file sha256>\\n', sorted by star id",
    })
    drawn = [k for s in SAMPLES for k in selection[s]["eb"]]
    eb_all = [measured[str(k)]["eb"] for k in drawn + selection["dev"]["eb"]]
    report = {
        "window": WINDOW,
        "inputs": {"tables_sha256": tables["hashes"], "release_secret_sha256": secret_sha,
                   "dev_seed": DEV_SEED, "tce_audit": selection["audit"],
                   "dev_truth_sha256": built["dev"]["truth_sha256"]},
        "samples": summaries,
        "selection": selection["used"],
        "not_selected": {g: len(v) for g, v in selection["trimmed"].items()},
        "eb_steering_counterfactual": selection["eb_steering_counterfactual"],
        "eb_ephemerides": {"r1_drawn": eb_ephemeris_summary([measured[str(k)]["eb"] for k in drawn], True),
                           "dev": eb_ephemeris_summary([measured[str(k)]["eb"] for k in selection["dev"]["eb"]],
                                                       False)},
        "eb_evidence_false_fire_on_null": fire,
        "real_recount": real_recount(measured, pools),
        "prevalence": prevalence(tables, eb_all),
        "injection_eb_periods": len(eb_periods),
    }
    write_json(OUT_BUILD, report)
    print(json.dumps({s: {k: v for k, v in summaries[s].items() if k in ("stars", "populations", "scored_stars",
                                                                         "planets_scored", "inj_target_swaps")}
                      for s in summaries}, indent=1))


def eb_ephemeris_summary(ebs: list[dict], private: bool) -> dict:
    """Sources, usable ephemerides, evidence and depth of a set of EBs, and how the Kirk duration measurement reads
    against catalogue durations on the KOI EBs."""
    return {"stars": len(ebs),
            "source": {s: sum(e["source"] == s for e in ebs) for s in ("koi", "kirk")},
            "no_usable_ephemeris": {s: sum(e["source"] == s and e["duration"] is None for e in ebs)
                                    for s in ("koi", "kirk")},
            "with_evidence": sum(any(e["evidence"].values()) for e in ebs),
            "depth_bins": depth_bins(e["depth"] for e in ebs),
            "kirk_duration_h": quantiles((e["duration"] for e in ebs if e["source"] == "kirk"), private),
            "koi_measured_over_catalogue_duration": quantiles(
                (e["measured_duration"] / e["duration"] for e in ebs
                 if e["source"] == "koi" and e["duration"] and e.get("measured_duration")), private)}


# ---- validate -------------------------------------------------------------------------------------------------------

def built_features(item) -> tuple[str, dict, dict, dict, float | None]:
    """Features of a built star with its truth signals (or a NULL star's pseudo-signal) masked; the same with an
    EB's orbital harmonics removed; the same with an EB's measured secondary left unmasked (the first build's mask);
    and for an INJ star the largest relative change any feature shows against its host's light curve before
    injection under the same mask."""
    sid, path, entry, pseudo, host = item
    d = np.load(path)
    time, flux, quarter = d["time"], d["flux"], d["quarter"]
    stellar = {k: float(d[k]) for k in STELLAR_KEYS}
    eb = entry.get("eb")
    eb = eb if eb and eb.get("duration") else None
    signals = entry["signals"] or ([pseudo] if pseudo else [])
    mask = signal_mask(time, signals, eb)
    masked = star_features(time, flux, quarter, mask, stellar)
    change = None
    if host is not None:
        t, f, q, _ = load_lc(host)
        before = star_features(t, f, q, mask, stellar)
        change = max(abs(masked[k] - before[k]) / abs(before[k]) for k in FEATURES if before[k])
    orbit = star_features(time, flux, quarter, mask, stellar, eb) if eb else masked
    old = masked
    if eb and eb.get("secondary_phase") is not None:
        old = star_features(time, flux, quarter, signal_mask(time, signals, {**eb, "secondary_phase": None}), stellar)
    return sid, masked, orbit, old, change


def adversarial(X: np.ndarray, y: np.ndarray, seed: int = 0, names=FEATURES, null_runs: int = 0) -> dict:
    """Cross-validated AUC of telling y=1 from y=0 (3 x 5 folds, out-of-fold probabilities averaged), a star
    bootstrap 95% interval, a logistic-regression check, permutation importances and univariate AUCs. With
    null_runs, the same procedure on shuffled labels gives the chance level for this sample size."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    def gb():
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=10,
                                              early_stopping=False, random_state=0)

    def lr():
        return make_pipeline(QuantileTransformer(n_quantiles=min(200, len(y)), output_distribution="normal"),
                             LogisticRegression(max_iter=2000))

    rng = np.random.default_rng(seed)
    Xf = np.where(np.isfinite(X), X, np.nan)
    Xl = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    prob, prob_lr = np.zeros(len(y)), np.zeros(len(y))
    importance = np.zeros(X.shape[1])
    repeats = 3
    for r in range(repeats):
        for train, test in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(Xf, y):
            model = gb().fit(Xf[train], y[train])
            prob[test] += model.predict_proba(Xf[test])[:, 1] / repeats
            prob_lr[test] += lr().fit(Xl[train], y[train]).predict_proba(Xl[test])[:, 1] / repeats
            if r == 0 and len(np.unique(y[test])) == 2:
                base = roc_auc_score(y[test], model.predict_proba(Xf[test])[:, 1])
                for j in range(X.shape[1]):
                    drops = []
                    for _ in range(5):
                        Xp = Xf[test].copy()
                        Xp[:, j] = rng.permutation(Xp[:, j])
                        drops.append(base - roc_auc_score(y[test], model.predict_proba(Xp)[:, 1]))
                    importance[j] += np.mean(drops) / 5
    auc = float(roc_auc_score(y, prob))
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boot = [roc_auc_score(y[i], prob[i]) for i in
            (np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)]) for _ in range(BOOTSTRAP))]
    null = []
    for _ in range(null_runs):
        shuffled = rng.permutation(y)
        p = np.zeros(len(y))
        for r in range(repeats):
            for train, test in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(Xf, shuffled):
                p[test] += gb().fit(Xf[train], shuffled[train]).predict_proba(Xf[test])[:, 1] / repeats
        null.append(roc_auc_score(shuffled, p))
    univariate = {}
    for j, f in enumerate(names):
        ok = np.isfinite(X[:, j])
        u = roc_auc_score(y[ok], X[ok, j]) if len(np.unique(X[ok, j])) > 1 else 0.5
        univariate[f] = round(float(max(u, 1 - u)), 4)
    return {"n_pos": int(pos.size), "n_neg": int(neg.size), "auc": auc,
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "auc_logistic": float(roc_auc_score(y, prob_lr)),
            **({"shuffled_label_auc_p95": float(np.percentile(null, 95)),
                "p_vs_shuffled": (1 + sum(a >= auc for a in null)) / (1 + null_runs)} if null_runs else {}),
            "permutation_importance": dict(sorted(((f, round(float(v), 4)) for f, v in zip(names, importance)),
                                                  key=lambda kv: -kv[1])),
            "univariate_auc": dict(sorted(univariate.items(), key=lambda kv: -kv[1]))}


def pair_tasks(key: tuple, features: dict, population: dict, pairs, names=FEATURES) -> list[tuple]:
    """One adversarial task per pair of populations: (key, pair, X, y, feature names)."""
    tasks = []
    for a, b in pairs:
        ids = [s for s in features if population[s] in (a, b)]
        if len({population[s] for s in ids}) < 2:
            continue
        X = np.array([[features[s][f] for f in names] if isinstance(features[s], dict) else features[s]
                      for s in ids], dtype=float)
        y = np.array([population[s] == a for s in ids], dtype=int)
        tasks.append((key, f"{a}-{b}", X, y, names))
    return tasks


def pseudo_signal(sample: str, sid: str, entry: dict, stars: dict) -> dict | None:
    """A NULL star has nothing to mask, so it is masked at the period and duration of a hashed INJ star of the same
    sample at a hashed phase: INJ and NULL then lose the same share of cadences to masking."""
    inj = [e["signals"][0] for _, e in sorted(stars.items()) if e["population"] == "inj"]
    if entry["population"] != "null" or not inj:
        return None
    s = inj[int(unit("pseudo", sample, sid) * len(inj))]
    t0 = entry["coverage"][0][0] + unit("pseudo-phase", sample, sid) * s["period"]
    return {"period": s["period"], "t0": t0, "duration": s["duration"]}


def run_task(task) -> tuple:
    key, pair, X, y, names = task
    return key, pair, adversarial(X, y, names=names, null_runs=NULL_RUNS if key[0] == "samples" else 0)


def cross_task(task) -> tuple:
    """Fit on one sample, score the other: boosted and logistic AUC."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    pair, direction, Xtr, ytr, Xte, yte = task
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=10,
                                        early_stopping=False, random_state=0).fit(Xtr, ytr)
    fill = np.nanmedian(Xtr, axis=0)
    lr = make_pipeline(QuantileTransformer(n_quantiles=min(200, len(ytr)), output_distribution="normal"),
                       LogisticRegression(max_iter=2000)).fit(np.where(np.isfinite(Xtr), Xtr, fill), ytr)
    return pair, direction, {"auc": float(roc_auc_score(yte, gb.predict_proba(Xte)[:, 1])),
                             "auc_logistic": float(roc_auc_score(
                                 yte, lr.predict_proba(np.where(np.isfinite(Xte), Xte, fill))[:, 1]))}


def eb_labels(stars: dict, mapping: dict) -> dict[str, dict[str, str]]:
    """Population labels with EBs split by score and by in-window depth and period, for the scored-EB rows and the
    EB subgroups."""
    def sub(test):
        return {s: (test(e, mapping[s]) or "other") if e["population"] == "eb" else e["population"]
                for s, e in stars.items()}

    def deep(m):
        return m.get("eb_depth") is not None and m["eb_depth"] >= 0.01

    return {
        "scored": sub(lambda e, m: "eb_scored" if e["scored"] else "eb_unscored"),
        "depth": sub(lambda e, m: "eb_deep" if deep(m) else "eb_shallow"),
        "deep_long": sub(lambda e, m: "eb_deep_long" if deep(m) and (e["eb"]["period"] or 0) >= 15 else None),
        "quiet": sub(lambda e, m: "eb_quiet" if e["eb"].get("quiet") else "eb_variable"),
    }


SCORED_PAIRS = (("inj", "eb_scored"), ("null", "eb_scored"), ("eb_scored", "real"))
SUBGROUP_PAIRS = {"scored": (("inj", "eb_unscored"),), "depth": (("inj", "eb_deep"), ("inj", "eb_shallow")),
                  "deep_long": (("inj", "eb_deep_long"),), "quiet": (("inj", "eb_quiet"), ("inj", "eb_variable"))}


def cmd_validate(args) -> None:
    pools = json.loads(POOLS.read_text())
    selection = json.loads(SELECTION.read_text())
    tasks, medians, injection_change, gate_rows = [], {}, {}, {}
    for sample in ("dev", *SAMPLES):
        private = sample != "dev"
        truth_path = DEV_DIR / "truth.json" if sample == "dev" else PRIVATE / f"truth_{sample}.json"
        folder = BUILT / "dev" if sample == "dev" else BUILT / RELEASE / sample
        stars = json.loads(truth_path.read_text())["stars"]
        mapping = json.loads(truth_path.with_name(truth_path.name.replace("truth", "map")).read_text())["stars"]
        items = [(sid, str(folder / f"{sid}.npz"), entry, pseudo_signal(sample, sid, entry, stars),
                  mapping[sid]["kic"] if entry["population"] == "inj" else None) for sid, entry in stars.items()]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(built_features, items, chunksize=4))
        features = {s: f for s, f, _, _, _ in rows}
        population = {s: e["population"] for s, e in stars.items()}
        labels = eb_labels(stars, mapping)
        medians[sample] = {pop: {f: round_sig(float(np.nanmedian([features[s][f] for s in features
                                                                  if population[s] == pop])), 3)
                                 for f in FEATURES} for pop in sorted(set(population.values()))}
        injection_change[sample] = quantiles((c for *_, c in rows if c is not None), private)
        tasks += pair_tasks(("samples", sample), features, population, PAIRS)
        tasks += pair_tasks(("samples", sample), features, labels["scored"], SCORED_PAIRS)
        for name, pairs in SUBGROUP_PAIRS.items():
            tasks += pair_tasks(("eb_subgroups", sample), features, labels[name], pairs)
        eb_pairs = [p for p in PAIRS if "eb" in p]
        tasks += pair_tasks(("eb_orbit_removed", sample), {s: f for s, _, f, _, _ in rows}, population, eb_pairs)
        tasks += pair_tasks(("eb_without_secondary_mask", sample), {s: f for s, _, _, f, _ in rows}, population,
                            eb_pairs)
        gate_rows[sample] = (features, labels["scored"], population)

    cross = []
    for a, b in PAIRS + SCORED_PAIRS:
        for train, test in (("A", "B"), ("B", "A")):
            (ftr, ltr, ptr), (fte, lte, pte) = gate_rows[train], gate_rows[test]
            labels_tr = ltr if "eb_scored" in (a, b) else ptr
            labels_te = lte if "eb_scored" in (a, b) else pte
            sets = []
            for feats, lab in ((ftr, labels_tr), (fte, labels_te)):
                ids = [s for s in feats if lab[s] in (a, b)]
                sets += [np.array([[feats[s][f] for f in FEATURES] for s in ids], dtype=float),
                         np.array([lab[s] == a for s in ids], dtype=int)]
            cross.append((f"{a}-{b}", f"{train}->{test}", *sets))

    table = table_features(load_tables()["stellar"])
    groups = {"host": [], "eb": [], "real": []}
    for name in POOL_LISTS:
        groups["host" if name in ("random", "tce_only") else "eb" if name == "eb" else "real"] += pools[name]["reserve"]
    selected = {k for s in SAMPLES for v in selection[s].values() for k in v}
    group_pairs = (("host", "eb"), ("host", "real"), ("eb", "real"))
    for key, keep in (("reserve_table_before_trimming", None), ("reserve_table_after_trimming", selected)):
        members = {k: g for g, ks in groups.items() for k in ks if keep is None or k in keep}
        tasks += pair_tasks((key,), {k: table[k] for k in members}, members, group_pairs, TABLE_FEATURES)

    report = {"gate": ("INJ-NULL must reach max(boosted, logistic) AUC <= 0.6 in A and "
                       "B and in the cross-sample check (fit on one sample, test on the other). EB and REAL pairs "
                       "are reported, not gated."),
              "features": list(FEATURES), "table_features": list(TABLE_FEATURES), "mask_pad_hours": MASK_PAD_HOURS,
              "orbit_harmonics": ORBIT_HARMONICS, "null_pseudo_mask": True, "samples": {}, "eb_subgroups": {},
              "eb_orbit_removed": {}, "eb_without_secondary_mask": {}, "cross_sample": {},
              "feature_medians": medians, "inj_feature_change_vs_host": injection_change}
    with ProcessPoolExecutor(args.workers) as pool:
        for key, pair, result in pool.map(run_task, tasks):
            node = report
            for part in key:
                node = node.setdefault(part, {})
            node[pair] = result
        for pair, direction, result in pool.map(cross_task, cross):
            report["cross_sample"].setdefault(pair, {})[direction] = result
    for node in [report["samples"][s] for s in ("dev", *SAMPLES)] + list(report["cross_sample"].values()):
        for v in node.values():
            v["max_auc"] = max(v["auc"], v["auc_logistic"])
    inj_null = [report["samples"][s]["inj-null"]["max_auc"] for s in SAMPLES]
    inj_null += [v["max_auc"] for v in report["cross_sample"]["inj-null"].values()]
    report["gate_passed"] = all(a <= AUC_GATE for a in inj_null)
    report["reported_not_gated_above_gate"] = {
        f"{s}:{p}": round(v["max_auc"], 3) for s in SAMPLES for p, v in report["samples"][s].items()
        if p != "inj-null" and v["max_auc"] > AUC_GATE}
    write_json(OUT_ADVERSARIAL, report)
    for s in ("dev", *SAMPLES):
        print(s, {p: round(v["max_auc"], 3) for p, v in report["samples"][s].items()}, file=sys.stderr)
    print(json.dumps({"gate_passed": report["gate_passed"],
                      "cross_inj_null": report["cross_sample"]["inj-null"]}, indent=1))


# ---- baselines on dev -----------------------------------------------------------------------------------------------

def cmd_baselines(args) -> None:
    from transitbench.runner import run_catalogue
    from transitbench.score import catalogue_from_runs, load_preds, load_truth, report

    truth_path = DEV_DIR / "truth.json"
    stars = load_truth(truth_path)
    paths = {s.star_id: BUILT / "dev" / f"{s.star_id}.npz" for s in stars}
    out = {"truth_sha256": sha256_file(truth_path), "cap_seconds": CAP_SECONDS, "solutions": {}}
    for name, solution in SOLUTIONS.items():
        runs = run_catalogue(solution.resolve(), {k: v.resolve() for k, v in paths.items()}, CAP_SECONDS,
                             args.workers)
        preds = ROOT / "outputs" / "m2" / f"dev_{name}.jsonl"
        preds.parent.mkdir(parents=True, exist_ok=True)
        with open(preds, "w") as f:
            for sid, r in runs.items():
                f.write(json.dumps({"star": sid, "status": r["status"], "cpu_s": r["cpu_s"],
                                    "rows": [[c.period, c.t0, c.score] for c in r["rows"]]}) + "\n")
        catalogue, inputs = catalogue_from_runs(load_preds(preds))
        rep = report(stars, catalogue)
        h = rep["headline"]
        cpu = np.array([r["cpu_s"] for r in runs.values()])
        out["solutions"][name] = {
            "S": h["S"], "C": h["C"], "F_per_100": h["F_per_100"], "eb_as_planet": h["eb_as_planet"],
            "planets": h["planets"], "scored_stars": h["scored_stars"], "populations": h["populations"],
            "by_host": h["by_host"], "null_false_alarms": rep["null_false_alarms"],
            "eb_rates": rep["eb_as_planet"], "completeness_by_mes": rep["completeness"]["by_mes"],
            "real": {k: rep["real"][k] for k in ("stars", "planets", "recall", "predicted_from_inj", "rows")},
            "runs": inputs, "cpu_s": {"median": float(np.median(cpu)), "p95": float(np.percentile(cpu, 95)),
                                      "max": float(cpu.max())}}
        print(name, {k: round(h[k], 4) for k in ("S", "C", "F_per_100", "eb_as_planet")}, file=sys.stderr)
        if name == "starter":
            starter = catalogue
    with ProcessPoolExecutor(args.workers) as pool:
        variability = dict(pool.map(unmasked_variability, sorted(paths.items()), chunksize=4))
    out["starter_abstain_on_variable_stars"] = abstain_reference(stars, starter, variability)
    print("starter + abstain", {k: round(v, 4) for k, v in out["starter_abstain_on_variable_stars"].items()
                                if isinstance(v, float)}, file=sys.stderr)
    write_json(OUT_BASELINES, out)


def unmasked_variability(item) -> tuple[str, float]:
    """The variability feature as a pipeline could compute it: on the whole delivered light curve, nothing masked."""
    sid, path = item
    d = np.load(path)
    stellar = {k: float(d[k]) for k in STELLAR_KEYS}
    return sid, star_features(d["time"], d["flux"], d["quarter"], np.zeros(d["time"].size, bool), stellar)[VARIABILITY]


def abstain_reference(stars, catalogue: dict, variability: dict) -> dict:
    """In-sample reference for the EB variability shortcut: the starter's rows, dropped on every star whose unmasked
    variability exceeds a threshold tuned on dev for the best S. Its EB-specific part is the gain beyond silencing
    EBs at the rate the rule silences INJ and NULL hosts (a counterfactual in which the rule is blind to EBs)."""
    from transitbench.score import score

    def silenced(threshold):
        return {s.star_id for s in stars if finite(variability[s.star_id]) and variability[s.star_id] > threshold}

    def run(quiet_out):
        return score(stars, {k: ([] if k in quiet_out else v) for k, v in catalogue.items()})

    values = sorted(v for v in variability.values() if finite(v))
    best = max(((run(silenced(t))["S"], t) for t in np.percentile(values, np.arange(50, 100, 1.0))),
               key=lambda st: (st[0], st[1]))
    threshold = float(best[1])
    out_ids = silenced(threshold)
    base, after = run(set()), run(out_ids)
    scored_eb = [s for s in stars if s.population == "eb" and s.scored]
    hosts = [s for s in stars if s.population in ("inj", "null")]
    host_rate = sum(s.star_id in out_ids for s in hosts) / len(hosts)
    eb_base = score(scored_eb, catalogue)["false"]
    eb_after = score(scored_eb, {k: ([] if k in out_ids else v) for k, v in catalogue.items()})["false"]
    eb_part = ((eb_base - eb_after) - host_rate * eb_base) / base["planets"]
    return {"in_sample": True, "feature": VARIABILITY, "threshold": threshold,
            "S": after["S"], "C": after["C"], "F_per_100": after["F_per_100"], "eb_as_planet": after["eb_as_planet"],
            "S_starter": base["S"], "gain": after["S"] - base["S"], "eb_specific_gain": eb_part,
            "other_gain": after["S"] - base["S"] - eb_part,
            "silenced_share": {"inj_and_null": host_rate,
                               "scored_eb": sum(s.star_id in out_ids for s in scored_eb) / max(len(scored_eb), 1)},
            "eb_false_rows": {"starter": eb_base, "after": eb_after}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("fetch", "measure", "select", "build", "validate", "baselines", "all"))
    parser.add_argument("--new-secret", action="store_true", help="select: create the release secret (once)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    steps = {"fetch": cmd_fetch, "measure": cmd_measure, "select": cmd_select, "build": cmd_build,
             "validate": cmd_validate, "baselines": cmd_baselines}
    order = ("select", "fetch", "measure", "select", "build", "validate")
    for name in order if args.step == "all" else (args.step,):
        steps[name](args)


if __name__ == "__main__":
    main()
