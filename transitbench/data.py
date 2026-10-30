"""Readers for the pinned Kepler tables and the Kepler quarter calendar.

Quarter times are the first and last long-cadence mid-times of each quarter from the Kepler Data Release 25 Notes
(KSCI-19065-002, Tables 1-2), converted with BKJD = MJD - 54832.5. They agree with the FITS TIME column of real
DR25 light curves to about 0.002 d, which is enough to assign transit midpoints to quarters.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

QUARTERS_BKJD = {
    1: (131.5110, 164.9812),
    2: (169.5175, 258.4649),
    3: (260.2222, 349.4966),
    4: (352.3777, 442.2038),
    5: (443.4912, 538.1600),
    6: (539.4473, 629.2939),
    7: (630.1725, 719.5491),
    8: (735.3647, 802.3460),
    9: (808.5170, 905.9239),
    10: (906.8434, 1000.2659),
    11: (1001.2058, 1098.3267),
    12: (1099.4097, 1182.0227),
    13: (1182.7379, 1273.0544),
    14: (1274.1374, 1371.3196),
    15: (1373.4855, 1471.1377),
    16: (1472.0980, 1557.9601),
    17: (1559.2269, 1591.0011),
}

# Long cadences per quarter, from the same tables.
QUARTER_CADENCES = {
    1: 1639, 2: 4354, 3: 4370, 4: 4397, 5: 4634, 6: 4398, 7: 4375, 8: 3279, 9: 4768,
    10: 4573, 11: 4754, 12: 4044, 13: 4421, 14: 4757, 15: 4780, 16: 4203, 17: 1556,
}


def observed_quarters(st_quarters: str) -> list[int]:
    """Quarters with data from the DR25 stellar table's st_quarters string (left-most character is Q1)."""
    return [i + 1 for i, c in enumerate(st_quarters.strip().strip('"')) if c == "1"]


def quarter_coverage(quarters) -> np.ndarray:
    """[start, end] BKJD intervals for the given quarters, ignoring gaps inside a quarter."""
    rows = [QUARTERS_BKJD[q] for q in sorted(quarters)]
    return np.array(rows, dtype=float).reshape(-1, 2)


def _number(text: str):
    if text in ("", "null", "nan", "NaN"):
        return math.nan
    try:
        return float(text)
    except ValueError:
        return None


def _columns(names, rows, text=()) -> dict[str, np.ndarray]:
    for i, r in enumerate(rows):
        if len(r) != len(names):
            raise ValueError(f"row {i} has {len(r)} fields, header has {len(names)} (truncated or reformatted table)")
    cols = {}
    for k, name in enumerate(names):
        values = [r[k] for r in rows]
        numbers = [None] if name in text else [_number(v) for v in values]
        if all(n is not None for n in numbers):
            cols[name] = np.array(numbers, dtype=float)
        else:
            cols[name] = np.array(values, dtype=object)
    return cols


def read_csv(path, delimiter: str = ",", comment: str | None = None, text=()) -> dict[str, np.ndarray]:
    """Read a delimited table into columns; columns that parse as numbers become float arrays unless named in text."""
    with open(path, newline="") as f:
        lines = [line for line in f if not (comment and line.startswith(comment)) and line.strip()]
    reader = csv.reader(lines, delimiter=delimiter)
    names = [n.strip() for n in next(reader)]
    return _columns(names, list(reader), text=text)


def read_ipac(path) -> dict[str, np.ndarray]:
    """Read an IPAC ASCII table (backslash keywords, then |name|type|unit|null| rows, then data)."""
    names = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("\\"):
                continue
            if line.startswith("|"):
                if names is None:
                    names = [n.strip() for n in line.strip().strip("|").split("|")]
                continue
            if line.strip():
                rows.append(line.split())
    return _columns(names, rows)


def verify_tables(tables_dir, manifest_path) -> dict[str, str]:
    """Check every downloaded table against its committed SHA-256; returns {file: sha256}, raises on any mismatch."""
    manifest = json.loads(Path(manifest_path).read_text())
    hashes = {}
    for entry in manifest["tables"]:
        digest = hashlib.sha256((Path(tables_dir) / entry["file"]).read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['file']}: sha256 {digest} does not match the manifest")
        hashes[entry["file"]] = digest
    return hashes


def stellar_params(stellar) -> dict[int, dict]:
    """Mass, radius (solar units) and the four Claret limb-darkening coefficients per KIC, from the stellar table."""
    out = {}
    for i, k in enumerate(stellar["kepid"]):
        ld = tuple(float(stellar[f"limbdark_coeff{j}"][i]) for j in range(1, 5))
        out[int(k)] = {"mass": float(stellar["mass"][i]), "radius": float(stellar["radius"][i]),
                       "ld": tuple(0.0 if not math.isfinite(c) else c for c in ld)}
    return out


def eb_ephemerides(koi) -> dict[int, dict]:
    """Per EB host, the deepest stellar-eclipse FALSE POSITIVE KOI (ephemeris-match FPs dropped), P 0.5-100 d."""
    out = {}
    for k, d, ss, ec, p, t0, dur, depth in zip(koi["kepid"], koi["koi_disposition"], koi["koi_fpflag_ss"],
                                               koi["koi_fpflag_ec"], koi["period_d"], koi["epoch_bkjd"],
                                               koi["dur_hr"], koi["depth_ppm"]):
        if d == "FALSE POSITIVE" and ss == 1 and ec == 0 and 0.5 <= p <= 100:
            if int(k) not in out or depth > out[int(k)]["depth_ppm"]:
                out[int(k)] = {"period": p, "t0": t0, "duration": dur, "depth_ppm": depth}
    return out


def select_twins(plti, quarters_by_kic, eligible_hosts, rng, n, window, mes_range=(6.0, 10.0)) -> list[dict]:
    """Pick n DR25 INJ1 injections on eligible hosts for paired twin tests, in a random order from rng.

    Kept: P <= 100 d, T14 <= 15 h, all window quarters observed, at least 3 transits in the window, and an expected
    one-year MES (NASA's Expected_MES times the square root of the window's share of the transits) in mes_range.
    """
    from transitbench.inject import CADENCE_DAYS, transit_count

    picked = []
    for i in rng.permutation(len(plti["KIC_ID"])):
        kic = int(plti["KIC_ID"][i])
        p, e, d, mes4 = plti["i_period"][i], plti["i_epoch"][i], plti["i_dur"][i], plti["Expected_MES"][i]
        if kic not in eligible_hosts or p > 100 or d > 15 or not math.isfinite(mes4) or plti["N_Transit"][i] <= 0:
            continue
        qs = quarters_by_kic.get(kic, [])
        if not all(q in qs for q in window):
            continue
        grid = np.concatenate([np.arange(*QUARTERS_BKJD[q], CADENCE_DAYS) for q in window])
        n_win = transit_count(grid, p, e, d)
        if n_win < 3:
            continue
        mes1 = mes4 * math.sqrt(n_win / plti["N_Transit"][i])
        if mes_range[0] <= mes1 <= mes_range[1]:
            picked.append({"kic": kic, "plti_row": int(i), "mes_1yr_expected": round(float(mes1), 3),
                           "n_transits_window": n_win})
        if len(picked) == n:
            break
    return picked


# What a pipeline sees for one star. Nothing else may be written into a delivered star file.
STAR_KEYS = ("time", "flux", "quarter", "teff", "logg", "radius", "mass")
STELLAR_KEYS = ("teff", "logg", "radius", "mass")


def round_sig(x: float, digits: int = 2) -> float:
    """Round to the given number of significant figures; NaN and zero pass through."""
    x = float(x)
    if not math.isfinite(x) or x == 0.0:
        return x
    return round(x, digits - 1 - math.floor(math.log10(abs(x))))


def deliver_flux(flux, quarter, qmedian, drop=None) -> np.ndarray:
    """Quarter-normalised flux as delivered: the archive's float32 values times (1 - drop), rounded back to float32,
    then divided by each quarter's median. With no drop this returns the reduction's flux bit for bit, so injected
    and untouched stars carry the same float32 fingerprint and every quarter's median is 1."""
    quarter = np.asarray(quarter)
    raw = (np.asarray(flux, dtype=float) * np.asarray(qmedian, dtype=float)[quarter]).astype(np.float32)
    if drop is not None:
        raw = (raw.astype(float) * (1.0 - np.asarray(drop, dtype=float))).astype(np.float32)
    raw = raw.astype(float)
    out = np.empty_like(raw)
    for q in np.unique(quarter):
        sel = quarter == q
        out[sel] = raw[sel] / np.median(raw[sel])
    return out


def write_star(path, arrays: dict) -> None:
    """Write a star .npz with only STAR_KEYS, in a fixed order and with fixed zip timestamps, so the bytes depend on
    the content alone (no build time or build order leaks into the file)."""
    import zipfile

    if set(arrays) != set(STAR_KEYS):
        raise ValueError(f"star keys {sorted(arrays)} differ from {STAR_KEYS}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in STAR_KEYS:
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w") as f:
                np.lib.format.write_array(f, np.asarray(arrays[key]), allow_pickle=False)


def star_bytes(arrays: dict) -> bytes:
    """The exact bytes write_star writes for these arrays, built in memory (mini tasks hand star files to a box
    without touching disk)."""
    import io
    import zipfile

    if set(arrays) != set(STAR_KEYS):
        raise ValueError(f"star keys {sorted(arrays)} differ from {STAR_KEYS}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in STAR_KEYS:
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w") as f:
                np.lib.format.write_array(f, np.asarray(arrays[key]), allow_pickle=False)
    return buf.getvalue()

KIRK_BKJD_OFFSET = 54833.0  # Kirk 2016 BJD0 is BJD - 2400000; BKJD = BJD - 2454833


def kirk_ephemerides(kirk, p_range=(0.5, 100.0)) -> dict[int, dict]:
    """Per Kirk 2016 EB, period and primary epoch in BKJD (the table has no duration), P within p_range."""
    out = {}
    for k, p, bjd0, morph in zip(kirk["KIC"], kirk["Per"], kirk["BJD0"], kirk["Morph"]):
        if not str(k).strip().isdigit():
            continue  # the unit and separator rows under the header
        p, bjd0, m = (_number(str(v).strip()) for v in (p, bjd0, morph))
        if p is None or not math.isfinite(p) or not p_range[0] <= p <= p_range[1] or bjd0 is None:
            continue
        out.setdefault(int(k), {"period": p, "t0": bjd0 - KIRK_BKJD_OFFSET, "morph": math.nan if m is None else m})
    return out


def injection_eb_periods(koi, kirk) -> list[float]:
    """Real EB periods that injections copy 10% of the time: every stellar-eclipse FP KOI (eb_ephemerides) plus every
    Kirk 2016 EB without one, P 0.5-100 d, sorted so a seeded draw is reproducible."""
    koi_ebs = eb_ephemerides(koi)
    periods = [v["period"] for v in koi_ebs.values()]
    periods += [v["period"] for k, v in kirk_ephemerides(kirk).items() if k not in koi_ebs]
    return sorted(float(p) for p in periods)
