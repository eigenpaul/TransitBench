"""Download the public Kepler tables listed below into data/tables/ and record them in manifests/tables.json.

Run from the repo root: uv run python -I scripts/fetch_tables.py [--only FILE ...]
--only fetches just the named tables and updates their manifest entries, leaving every other pinned table as it is.
TAP doubles come back rounded to 6 significant figures unless selected as `col+0 AS newname`, so every
ephemeris column below uses that form. Downloads use curl, which the archive's firewall accepts.
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tables"
MANIFEST = ROOT / "manifests" / "tables.json"
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
SIM = "https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated"

STELLAR_COLUMNS = (
    "kepid,teff,logg,kepmag,radius,mass,dens,st_quarters,nkoi,ntce,nconfp,dataspan,dutycycle,timeoutsumry,"
    "mesthres01p5,mesthres02p0,mesthres02p5,mesthres03p0,mesthres03p5,mesthres04p5,mesthres05p0,mesthres06p0,"
    "mesthres07p5,mesthres09p0,mesthres10p5,mesthres12p0,mesthres12p5,mesthres15p0,rrmscdpp03p0,rrmscdpp06p0,"
    "limbdark_coeff1,limbdark_coeff2,limbdark_coeff3,limbdark_coeff4"
)
KOI_COLUMNS = (
    "kepid,kepoi_name,kepler_name,koi_disposition,koi_pdisposition,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,"
    "koi_fpflag_ec,koi_period+0 as period_d,koi_time0bk+0 as epoch_bkjd,koi_duration+0 as dur_hr,"
    "koi_depth+0 as depth_ppm,koi_max_mult_ev+0 as mes,koi_model_snr+0 as snr,koi_tce_plnt_num,koi_tce_delivname"
)
TCE_COLUMNS = (
    "kepid,tce_plnt_num,tce_rogue_flag,tce_period+0 as period_d,tce_time0bk+0 as epoch_bkjd,"
    "tce_duration+0 as dur_hr,tce_depth+0 as depth_ppm,tce_max_mult_ev+0 as mes"
)

SOURCES = [
    ("dr25_stellar.csv", "tap", f"select {STELLAR_COLUMNS} from q1_q17_dr25_ks"),
    ("cumulative_koi.csv", "tap", f"select {KOI_COLUMNS} from cumulative"),
    ("dr25_obs_tce.csv", "tap", f"select {TCE_COLUMNS} from q1_q17_dr25_tce"),
    ("cumulative_koi_shape.csv", "tap",
     "select kepid,kepoi_name,koi_ror+0 as ror,koi_impact+0 as impact from cumulative"),
    ("dr25_tce_shape.csv", "tap",
     "select kepid,tce_plnt_num,tce_ror+0 as ror,tce_impact+0 as impact from q1_q17_dr25_tce"),
    ("keplernames.csv", "tap", "select kepid,pl_name,koi_name,kepler_name from keplernames"),
    (
        "pscomppars_kepler.csv",
        "tap",
        "select p.pl_name,p.ttv_flag,p.pl_controv_flag from pscomppars p join keplernames k on p.pl_name=k.pl_name",
    ),
    ("kplr_dr25_inj1_plti.txt", "url", f"{SIM}/kplr_dr25_inj1_plti.txt"),
    ("kplr_dr25_inj1_tces.txt", "url", f"{SIM}/kplr_dr25_inj1_tces.txt"),
    (
        "kirk2016_eb.tsv",
        "url",
        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/AJ/151/68/catalog&-out.max=unlimited&-out.all",
    ),
    (
        "kplr_droplist_inv.txt",
        "url",
        "https://raw.githubusercontent.com/stevepur/DR25-occurrence-public/"
        "d200f54b6f0df49e0dae530e69983cdce5397bfb/data/kplr_droplist_inv.txt",
    ),
]


def curl(args: list[str], dest: Path) -> None:
    cmd = ["curl", "-sSfL", "--retry", "3", "--max-time", "900", "-o", str(dest), *args]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="fetch only these files; keep every other manifest entry")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    kept = json.loads(MANIFEST.read_text())["tables"] if args.only and MANIFEST.exists() else []
    kept = [e for e in kept if e["file"] not in args.only] if args.only else []
    entries = []
    for name, kind, source in SOURCES:
        if args.only and name not in args.only:
            continue
        dest = OUT / name
        print(f"fetching {name}", file=sys.stderr)
        if kind == "tap":
            curl(["-G", TAP, "--data-urlencode", "format=csv", "--data-urlencode", f"query={source}"], dest)
        else:
            curl([source], dest)
        body = dest.read_bytes()
        if body.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
            raise SystemExit(f"{name}: got an HTML page instead of data (firewall or error page)")
        entries.append(
            {
                "file": name,
                "kind": kind,
                "source": source,
                "bytes": len(body),
                "lines": body.count(b"\n"),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fetched = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.only:
        for e in entries:
            e["fetched_utc"] = fetched
        previous = json.loads(MANIFEST.read_text())["fetched_utc"]
        doc = {"fetched_utc": previous, "tables": kept + entries}
    else:
        doc = {"fetched_utc": fetched, "tables": entries}
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
