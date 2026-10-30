"""Build the mini pool (the stars every mini task draws from) and create the evaluator secret.

This is the evaluator's tool, shipped to show how the mini pool is built. It needs the evaluator's private data and
inputs that are not in this repository.

Uses the public train and dev pools only. Of data/private/pools.json only the "train" and "dev" lists are used; the
reserve and natural lists, and the test samples' KICs, are read only to assert that the pool is disjoint from them.

Steps, in order (each caches its output under outputs/mini_pool/):
  candidates  every train-pool TCE-only and EB star, every train-pool random star whose four window quarters are
              already cached under data/lc/raw from earlier downloads, and the dev pool's random, TCE-only and EB
              lists. Stars without a finite DR25 mass and radius and audit drops are left out
  fetch       reduce the train candidates not already reduced for an earlier train selection into
              data/lc/mini_pool/mast with fetch_lightcurves.fetch (raw files cached under data/lc/raw; the few TCE-only
              quarters not cached are downloaded); that selection's and the dev light curves are reused as they are
  measure     per star, in the BKJD frame: quarters, the variability (3-h red noise, build.star_features; an EB with
              its eclipses masked exactly as build.build_star masks them) and, for an EB, the ephemeris and in-window
              evidence (build.eb_truth)
  build       pids, roles and files: data/mini_pool/<sha16>/{manifest.json, stars/<pid>.npz, kic_map.json} and
              transitbench_env/data/mini_pool.json. Asserts the pool is disjoint from every reserve and test list
  secret      --new creates data/private/mini/secret (32 random bytes, hex) once and refuses to replace it

Run from the repo root: uv run --group build python -I scripts/mini_pool.py STEP [--workers 4]
"""

import argparse
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import sys
import tempfile
import time as clock
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build  # noqa: E402  (sets OMP_NUM_THREADS=1)
import fetch_lightcurves  # noqa: E402

from transitbench.data import STELLAR_KEYS, eb_ephemerides, injection_eb_periods, kirk_ephemerides  # noqa: E402
from transitbench.data import round_sig  # noqa: E402
from transitbench_env import mini_draw  # noqa: E402

DATA = ROOT / "data"
POOLS = DATA / "private" / "pools.json"
RAW = DATA / "lc" / "raw" / "mast"
LC_STUDY = DATA / "lc" / "starter_study" / "mast"
LC_DEV = DATA / "lc" / "mast"
LC_EXT_ROOT = DATA / "lc" / "mini_pool"
LC_EXT = LC_EXT_ROOT / "mast"
STUDY_SELECTION = ROOT / "outputs" / "starter_study" / "selection.json"
OUT = ROOT / "outputs" / "mini_pool"
CANDIDATES = OUT / "candidates.json"
MEASURED = OUT / "measure.json"
POOL_JSON = ROOT / "transitbench_env" / "data" / "mini_pool.json"

POOL_SEED = "transitbench-v1-mini-pool"
LISTS = ("random", "tce_only", "eb")
SPLITS = ("train", "dev")
MIN_QUARTERS = 3
MEASURE_VERSION = 1
WORKERS = 4
EB_HIDDEN_SHARE = 0.5  # of a split's EBs with in-window evidence, by hash rank; every other EB is box-only


def log(*parts) -> None:
    print(clock.strftime("%H:%M:%S"), *parts, file=sys.stderr, flush=True)


def write_json(path: Path, doc, indent=1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=indent) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


# ---- candidates, fetch, measure ------------------------------------------------------------------------------------

def raw_cached(kic: int) -> bool:
    """Every window quarter is cached under data/lc/raw: a file, or the archive's 404 marker."""
    for q in fetch_lightcurves.WINDOW:
        f = RAW / f"{kic:09d}-q{q}.fits"
        if not ((f.exists() and f.stat().st_size > 0) or f.with_name(f.name + ".404").exists()):
            return False
    return True


def cmd_candidates(args) -> None:
    pools = read_json(POOLS)
    lists = {split: {name: sorted(int(k) for k in pools[name][split]) for name in LISTS} for split in SPLITS}
    del pools
    tables = build.load_tables()
    stellar = build.stellar_rows(tables["stellar"])
    drops, audit = build.audit_drops()
    study = {int(k) for v in read_json(STUDY_SELECTION)["order"].values() for k in v}

    def usable(k):
        return k in stellar and build.finite(stellar[k]["mass"], stellar[k]["radius"])

    out, counts = [], {}
    for split in SPLITS:
        for name in LISTS:
            kics = lists[split][name]
            c = {"list": len(kics), "no_mass_or_radius": 0, "audit_drop": 0, "not_cached": 0, "candidates": 0}
            for k in kics:
                if not usable(k):
                    c["no_mass_or_radius"] += 1
                elif name != "eb" and k in drops:
                    c["audit_drop"] += 1
                elif split == "train" and name == "random" and not raw_cached(k):
                    c["not_cached"] += 1
                else:
                    lc = LC_DEV if split == "dev" else LC_STUDY if k in study else LC_EXT
                    out.append({"kic": k, "split": split, "list": name, "lc": str(lc.relative_to(ROOT)),
                                "study": k in study})
                    c["candidates"] += 1
            counts[f"{split}/{name}"] = c
    write_json(CANDIDATES, {"seed": POOL_SEED, "audit": audit, "counts": counts, "stars": out})
    log("candidates", counts)


def cmd_fetch(args) -> None:
    stars = read_json(CANDIDATES)["stars"]
    todo = [s["kic"] for s in stars if s["lc"] == str(LC_EXT.relative_to(ROOT))]
    fetch_lightcurves.REDUCED = LC_EXT_ROOT  # reduced files land in data/lc/mini_pool/mast
    out = fetch_lightcurves.fetch([{"kic": k, "kind": "mast"} for k in todo], OUT / "lightcurves.json",
                                  args.workers)
    log("fetch", {"stars": len(out["stars"]), "failed": out["stars_failed"], "absent": out["files_absent"]})


def load_lc(path: Path):
    """A reduced light curve sorted by time (as build.load_lc), with qmedian."""
    d = np.load(path)
    order = np.argsort(d["time"], kind="stable")
    return d["time"][order], d["flux"][order], d["quarter"][order], d["qmedian"]


def lc_sha256(time, flux, quarter, qmedian) -> str:
    return build.sha256_bytes(b"".join(np.ascontiguousarray(a).tobytes() for a in (time, flux, quarter, qmedian)))


def measure_star(job: dict) -> dict:
    """Quarters, variability and (EB) ephemeris with in-window evidence, on the light curve before delivery."""
    path = ROOT / job["lc"] / f"{job['kic']}.npz"
    if not path.exists() or "qmedian" not in np.load(path).files:
        return {"kic": job["kic"], "missing": True}
    time, flux, quarter, qmedian = load_lc(path)
    out = {"kic": job["kic"], "key": job["key"], "lc_sha256": lc_sha256(time, flux, quarter, qmedian),
           "quarters": int(np.unique(quarter).size), "n_cadences": int(time.size), "eb": None,
           "variability_unmasked": False}
    mask = np.zeros(time.size, dtype=bool)
    if job["list"] == "eb":
        eb = build.eb_truth(time, flux, job["eph"])
        out["eb"] = {k: eb[k] for k in ("period", "t0", "duration", "evidence", "source", "secondary_phase")}
        out["eb_depth"] = eb["depth"]
        mask = build.signal_mask(time, [], out["eb"])
    v = build.star_features(time, flux, quarter, mask, job["stellar"])[build.VARIABILITY]
    if not build.finite(v):  # eclipses cover nearly the whole orbit: no 3-h bin survives the mask (as build_star)
        v = build.star_features(time, flux, quarter, np.zeros_like(mask), job["stellar"])[build.VARIABILITY]
        out["variability_unmasked"] = True
    out["variability"] = float(v)
    return out


def measure_jobs() -> list[dict]:
    cands = read_json(CANDIDATES)["stars"]
    tables = build.load_tables()
    stellar = build.stellar_rows(tables["stellar"])
    koi_ebs, kirk = eb_ephemerides(tables["koi"]), kirk_ephemerides(tables["kirk"])
    jobs = []
    for c in cands:
        k = c["kic"]
        job = {**c, "stellar": {key: round_sig(stellar[k][key]) for key in STELLAR_KEYS}, "eph": None}
        if c["list"] == "eb":
            if k in koi_ebs:
                e = koi_ebs[k]
                job["eph"] = {"period": e["period"], "t0": e["t0"], "duration": e["duration"], "source": "koi"}
            else:
                e = kirk[k]
                job["eph"] = {"period": e["period"], "t0": e["t0"], "duration": None, "source": "kirk"}
        job["key"] = build.sha256_bytes(json.dumps([MEASURE_VERSION, job], sort_keys=True).encode())
        jobs.append(job)
    return jobs


def cmd_measure(args) -> None:
    jobs = measure_jobs()
    cache = read_json(MEASURED)["stars"] if MEASURED.exists() else {}
    todo = [j for j in jobs if cache.get(f"{j['split']}/{j['kic']}", {}).get("key") != j["key"]]
    log("measuring", len(todo), "of", len(jobs))
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (job, res) in enumerate(zip(todo, pool.map(measure_star, todo, chunksize=8)), 1):
            cache[f"{job['split']}/{job['kic']}"] = res
            if i % 250 == 0:
                log("measured", i)
                write_json(MEASURED, {"version": MEASURE_VERSION, "stars": cache}, indent=None)
    write_json(MEASURED, {"version": MEASURE_VERSION, "stars": cache}, indent=None)
    log("measured", len(cache), "missing", sum(bool(v.get("missing")) for v in cache.values()))


# ---- build ---------------------------------------------------------------------------------------------------------

def npz_bytes(arrays: dict) -> bytes:
    """A .npz of these arrays, in this order and with fixed zip timestamps, so its bytes depend on content only."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, value in arrays.items():
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w") as f:
                np.lib.format.write_array(f, np.asarray(value), allow_pickle=False)
    return buf.getvalue()


def other_kics() -> tuple[set, set, set, set]:
    """Every reserve, natural and test-sample KIC (read only to assert disjointness), and the dev and train lists."""
    pools = read_json(POOLS)
    lists = ("random", "tce_only", "eb", "real_7_12", "real_12_up")
    reserve = {int(k) for p in lists for k in pools[p]["reserve"]} | {int(k) for k in pools["natural"]}
    dev = {int(k) for p in lists for k in pools[p]["dev"]}
    train = {int(k) for p in lists for k in pools[p]["train"]}
    del pools
    test = set()
    for path in sorted((DATA / "private" / "releases").glob("*/map_*.json")):
        test |= {int(v["kic"]) for v in read_json(path)["stars"].values()}
    return reserve, test, dev, train


def assert_disjoint(by_split: dict[str, set[int]]) -> dict:
    """The pool holds no reserve, natural or test-sample star; train stars come from the train lists and dev stars
    from the dev lists. Returns counts only."""
    reserve, test, dev, train = other_kics()
    pool = by_split["train"] | by_split["dev"]
    if pool & reserve or pool & test:
        raise SystemExit("the pool holds a reserve, natural or test star")
    if not by_split["train"] <= train or by_split["train"] & dev or not by_split["dev"] <= dev:
        raise SystemExit("a pool star is outside its split's pool lists")
    return {"reserve_and_natural_stars": len(reserve), "test_sample_stars": len(test), "overlap": 0}


def quiet_threshold(hosts: dict[str, list[float]]) -> float:
    """The build's QUIET_PERCENTILE of host variability (build.mark_quiet), over the split's hosts with the two host
    types weighted equally, as a sample holds them half and half."""
    values = np.concatenate([np.asarray(v, float) for v in hosts.values()])
    weights = np.concatenate([np.full(len(v), 1.0 / len(v)) for v in hosts.values()])
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(cum, build.QUIET_PERCENTILE / 100.0)])


def cmd_build(args) -> None:
    jobs = {f"{j['split']}/{j['kic']}": j for j in measure_jobs()}
    measured = read_json(MEASURED)["stars"]
    tables = build.load_tables()
    stellar = build.stellar_rows(tables["stellar"])
    eb_periods = injection_eb_periods(tables["koi"], tables["kirk"])
    keep, dropped = {split: [] for split in SPLITS}, {}
    for key, j in sorted(jobs.items()):
        m = measured.get(key)
        if m is None or m.get("missing") or m["key"] != j["key"]:
            raise SystemExit(f"{key}: not measured; run measure")
        reason = ("few_quarters" if m["quarters"] < MIN_QUARTERS else
                  "no_variability" if not build.finite(m["variability"]) else
                  "no_eb_ephemeris" if j["list"] == "eb" and m["eb"]["duration"] is None else None)
        if reason:
            dropped[f"{j['split']}/{reason}"] = dropped.get(f"{j['split']}/{reason}", 0) + 1
        else:
            keep[j["split"]].append((j, m))
    disjoint = assert_disjoint({split: {j["kic"] for j, _ in v} for split, v in keep.items()})
    base = DATA / mini_draw.POOL_DIR
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".build-", dir=base))
    (work / "stars").mkdir()
    stars, kic_map, splits = [], {}, {}
    for split, members in keep.items():
        flagged = sorted((jm for jm in members if jm[0]["list"] == "eb" and any(jm[1]["eb"]["evidence"].values())),
                         key=lambda jm: build.unit(POOL_SEED, "eb-role", jm[0]["kic"]))
        hidden = {j["kic"] for j, _ in flagged[:math.ceil(len(flagged) * EB_HIDDEN_SHARE)]}
        splits[split] = {"quiet_threshold": quiet_threshold(
            {t: [m["variability"] for j, m in members if j["list"] == t] for t in mini_draw.HOST_TYPES})}
        prefix = "t" if split == "train" else "d"
        for i, (j, m) in enumerate(sorted(members, key=lambda jm: build.unit(POOL_SEED, "pid", split,
                                                                              jm[0]["kic"]))):
            pid, kic, is_eb = f"{prefix}-{i:05d}", j["kic"], j["list"] == "eb"
            time, flux, quarter, qmedian = load_lc(ROOT / j["lc"] / f"{kic}.npz")
            if lc_sha256(time, flux, quarter, qmedian) != m["lc_sha256"]:
                raise SystemExit(f"{split}/{kic}: light curve changed since measure")
            data = npz_bytes({"time": time, "flux": flux, "quarter": quarter, "qmedian": qmedian})
            (work / "stars" / f"{pid}.npz").write_bytes(data)
            s = stellar[kic]
            stars.append({
                "pid": pid, "split": split, "host": j["list"],
                "role": ("hidden" if kic in hidden else "box") if is_eb else "any",
                "stellar": j["stellar"],
                "inject": None if is_eb else {"mass": s["mass"], "radius": s["radius"], "ld": list(s["ld"])},
                "eb": {**m["eb"], "flagged": any(m["eb"]["evidence"].values()), "depth": m["eb_depth"]}
                if is_eb else None,
                "variability": m["variability"], "variability_unmasked": m["variability_unmasked"],
                "sha256": hashlib.sha256(data).hexdigest()})
            kic_map[pid] = {"kic": kic, "list": j["list"]}
    manifest = {"version": 1, "builder": mini_draw.BUILDER, "seed": POOL_SEED, "window": build.WINDOW,
                "variability": build.VARIABILITY, "splits": splits, "stars": stars, "eb_periods": eb_periods}
    raw = (json.dumps(manifest, indent=1) + "\n").encode()
    sha16 = hashlib.sha256(raw).hexdigest()[:16]
    (work / "manifest.json").write_bytes(raw)
    write_json(work / "kic_map.json", {"pool": sha16, "stars": kic_map})
    dest = base / sha16
    if dest.exists():
        shutil.rmtree(work)  # same manifest digest, so the same star files
    else:
        work.rename(dest)
    write_json(POOL_JSON, {"sha": sha16})
    write_json(OUT / "build.json", {"pool": sha16, "dropped": dropped, "disjoint": disjoint})
    log("pool", sha16, {"stars": len(stars), "dropped": dropped, "disjoint": disjoint})


# ---- secret --------------------------------------------------------------------------------------------------------

def create_secret(path: Path) -> None:
    """Write a new evaluator secret (SECRET_BYTES random bytes as hex) to path, readable by its owner only. Refuses to
    replace an existing one: that would redraw every mini task."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise SystemExit("the mini secret exists; refusing to replace it (that would redraw every mini task)") from None
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_hex(mini_draw.SECRET_BYTES) + "\n")


def cmd_secret(args) -> None:
    if args.new:
        create_secret(args.data_dir / mini_draw.SECRET)
        log("created the mini secret")
    mini_draw.read_secret(args.data_dir)
    log("mini secret present and well formed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("candidates", "fetch", "measure", "build", "secret"))
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--new", action="store_true", help="secret: create it (once)")
    parser.add_argument("--data-dir", type=Path, default=DATA, help="secret: the evaluator data directory")
    args = parser.parse_args()
    if args.workers > 4:
        raise SystemExit("at most 4 workers")
    {"candidates": cmd_candidates, "fetch": cmd_fetch, "measure": cmd_measure, "build": cmd_build,
     "secret": cmd_secret}[args.step](args)


if __name__ == "__main__":
    main()
