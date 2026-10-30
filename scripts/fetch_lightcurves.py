"""Download one window of Kepler long-cadence light curves and reduce each star to one .npz.

Sources, per quarter (file names carry a fixed timestamp per quarter):
- mast: DR25 PDCSAP, https://archive.stsci.edu/missions/kepler/lightcurves/<kic4>/<kic9>/kplr<kic9>-<ts>_llc.fits
- inj1: NASA's pixel-level injected light curves, the INJ1 run of DR25 (exoplanetarchive Simulated data)

Reduction: keep cadences with finite TIME and PDCSAP_FLUX and SAP_QUALITY == 0, divide flux by each quarter's
median. Output data/lc/<kind>/<kic>.npz with time (BKJD), flux, quarter (0-based index within the window) and
qmedian (each window quarter's median PDCSAP flux, NaN if absent), so flux * qmedian[quarter] recovers the archive's
float32 values exactly. Raw files stay in data/lc/raw/; their hashes go in the manifest given with --manifest. A
quarter the archive does not have (HTTP 404) leaves an empty .404 marker so reruns do not ask again.

Run from the repo root: uv run --group build python -I scripts/fetch_lightcurves.py STARS.json --manifest OUT.json
where STARS.json is a list of {"kic": int, "kind": "mast" | "inj1"}.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "lc" / "raw"
REDUCED = ROOT / "data" / "lc"
WINDOW = (10, 11, 12, 13)
TIMESTAMPS = {  # long-cadence file timestamp per quarter, as in the MAST file names
    9: "2011177032512", 10: "2011271113734", 11: "2012004120508", 12: "2012088054726", 13: "2012179063303",
    14: "2012277125453", 15: "2013011073258", 16: "2013098041711", 17: "2013131215648",
}
INJ1 = "https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/light_curves/injected-light-curves-dr25-inj1"


def url(kind: str, kic: int, quarter: int) -> str:
    k9 = f"{kic:09d}"
    ts = TIMESTAMPS[quarter]
    if kind == "mast":
        return f"https://archive.stsci.edu/missions/kepler/lightcurves/{k9[:4]}/{k9}/kplr{k9}-{ts}_llc.fits"
    return f"{INJ1}/q{quarter}/kplr{k9}-{ts}_INJECTED-inj1_llc.fits.gz"


def download(kind: str, kic: int, quarter: int) -> Path | str:
    """The cached or downloaded file, "absent" if the archive has no such quarter, or "failed"."""
    suffix = ".fits" if kind == "mast" else ".fits.gz"
    dest = RAW / kind / f"{kic:09d}-q{quarter}{suffix}"
    absent = dest.with_name(dest.name + ".404")
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if absent.exists():
        return "absent"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    result = subprocess.run(
        ["curl", "-sSL", "--retry", "3", "--retry-delay", "2", "--max-time", "300", "-w", "%{http_code}", "-o",
         str(tmp), url(kind, kic, quarter)], capture_output=True, text=True)
    code = result.stdout.strip()
    if result.returncode != 0 or code != "200":
        tmp.unlink(missing_ok=True)
        if code == "404":
            absent.touch()
            return "absent"
        return "failed"
    tmp.rename(dest)
    return dest


def reduce_star(kind: str, kic: int, files: dict[int, Path]) -> dict:
    times, fluxes, quarters = [], [], []
    qmedian = np.full(len(WINDOW), np.nan)
    for i, q in enumerate(WINDOW):
        path = files.get(q)
        if path is None:
            continue
        with fits.open(path, memmap=False) as hdul:
            d = hdul[1].data
            t = np.asarray(d["TIME"], dtype=float)
            f = np.asarray(d["PDCSAP_FLUX"], dtype=float)
            good = np.isfinite(t) & np.isfinite(f) & (np.asarray(d["SAP_QUALITY"]) == 0)
        if good.sum() < 100:
            continue
        qmedian[i] = np.median(f[good])
        times.append(t[good])
        fluxes.append(f[good] / qmedian[i])
        quarters.append(np.full(good.sum(), i, dtype=np.int8))
    out = REDUCED / kind / f"{kic}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if times:
        np.savez_compressed(out, time=np.concatenate(times), flux=np.concatenate(fluxes),
                            quarter=np.concatenate(quarters), qmedian=qmedian)
    return {"kic": kic, "kind": kind, "quarters": [WINDOW[i] for i in sorted({int(q[0]) for q in quarters})],
            "cadences": int(sum(t.size for t in times)),
            "raw_sha256": {q: hashlib.sha256(p.read_bytes()).hexdigest() for q, p in files.items()}}


def fetch_star(kind: str, kic: int) -> dict:
    """Download one star's window quarters and reduce it; a star with a failed (not absent) quarter is not reduced,
    so a rerun retries it rather than leaving a light curve with a quarter silently missing."""
    got = {q: download(kind, kic, q) for q in WINDOW}
    files = {q: p for q, p in got.items() if isinstance(p, Path)}
    failed = [q for q, p in got.items() if p == "failed"]
    if failed:
        return {"kic": kic, "kind": kind, "failed_quarters": failed}
    return {**reduce_star(kind, kic, files), "absent_quarters": [q for q, p in got.items() if p == "absent"]}


def fetch(stars: list[dict], manifest: Path, workers: int = 8) -> dict:
    """Download and reduce every star in the list, in list order; write and return the manifest."""
    keys = list(dict.fromkeys((s["kind"], int(s["kic"])) for s in stars))
    with ThreadPoolExecutor(workers) as pool:
        entries = list(pool.map(lambda k: fetch_star(*k), keys))
    entries.sort(key=lambda e: (e["kind"], e["kic"]))
    failed = [e for e in entries if "failed_quarters" in e]
    out = {"window": list(WINDOW), "files_absent": sum(len(e.get("absent_quarters", [])) for e in entries),
           "stars_failed": len(failed), "stars": entries}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(entries)} stars, {len(failed)} with a failed download (rerun to retry), "
          f"{out['files_absent']} quarters absent from the archive", file=sys.stderr)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stars", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    fetch(json.loads(args.stars.read_text()), args.manifest, args.workers)


if __name__ == "__main__":
    main()
