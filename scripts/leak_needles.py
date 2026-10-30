"""Needles for the leak scans: what must never leave the evaluator.

needles() reads the evaluator's private data (data/private), which is not in this repository, and raises
FileNotFoundError without it. The natural sample is the evaluator's real-survey check: Kepler stars drawn at random,
with nothing injected, that the site reports (https://transitbench.eigenpaul.com/, section "Natural sample"). Its star
ids and names are needles only when asked for (below), and its map, secret and names only where its private files
exist.

- test star ids, the pattern r1-A-NNNN / r1-B-NNNN (four digits); the natural sample's n1-NNNN only with names (below),
  counted as test_id;
- KICs: every star of a reserve list or the natural list in data/private/pools.json, and every star of the test
  maps (data/private/releases/r1/map_A.json, map_B.json) and of the natural map (data/private/natural/map_n1.json,
  when it exists). A KIC counts as a whole number, leading zeros dropped: a
  digit run that is not part of a longer one, does not follow a decimal point (the fraction of a float) and is not
  inside a hex digest (16 or more hex characters with a letter among them);
- secrets: the release secret, the pool salt and the mini secret, and the natural secret when it exists, as hex in
  either case and as raw bytes;
- names, only when asked for (needles(names=True), --names): the natural star ids n1-NNNN (as test_id), and the KOI
  names (KNNNNN.NN) and Kepler names (Kepler-N b) of the natural truth's KOI signals with their common spellings, in
  any case: for each KOI host number N, KOI-N, KOI N, KOI_N or KOIN with or without zeros and a .NN suffix, and the
  zero-padded KNNNNN with any suffix (never K0*N, which would match a spectral type such as K5V for a one-digit
  host); for each Kepler number K, Kepler-K, Kepler K, Kepler_K or KeplerK with any planet letter. All from
  data/private/natural/truth_n1.json when it exists. The default leaves them out: a trace that names a well-known
  Kepler system leaks nothing, since the natural truth never enters a box.

Used by tests/test_wheel.py and tests/test_env.py for the wheel and image/context/; the command below scans any files, such as a run
directory. Reports never show a needle, only where each kind was found and how often.

Run from the repo root (it reads data/, so as python -I):
    uv run python -I scripts/leak_needles.py PATH [PATH ...]      exit 1 if any file holds a needle
"""

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TEST_ID = re.compile(rb"\br1-[AB]-\d{4}\b")
NATURAL_ID = re.compile(rb"\bn1-\d{4}\b")  # a test_id with names only; the default needles leave it out
NUMBER = re.compile(rb"(?<![\d.])\d{5,9}(?!\d)")
HEX = re.compile(rb"[0-9a-fA-F]{16,}")
SECRETS = ("private/releases/r1/secret", "private/pool_salt", "private/mini/secret", "private/natural/secret")
MAPS = ("private/releases/r1/map_A.json", "private/releases/r1/map_B.json", "private/natural/map_n1.json")
OPTIONAL = frozenset(("private/natural/secret", "private/natural/map_n1.json"))  # read only when they exist
NATURAL_TRUTH = "private/natural/truth_n1.json"
KOI_NAME = re.compile(r"K0*(\d+)\.\d+")
KEPLER_NAME = re.compile(r"Kepler-(\d+)")
KINDS = ("test_id", "kic", "secret", "name")


@dataclass(frozen=True, repr=False)
class Needles:
    kics: frozenset[bytes]
    hex_secrets: tuple[bytes, ...]  # lower case
    raw_secrets: tuple[bytes, ...]
    names: tuple[bytes, ...] = ()  # regex pieces for the natural KOI and Kepler names (truth_names), any case
    natural: bool = False  # natural star ids count as test_id (needles(names=True))

    def __post_init__(self):
        object.__setattr__(self, "_names", names_pattern(self.names))

    def __repr__(self) -> str:  # never the needles themselves: a failing test prints its arguments
        return f"Needles({len(self.kics)} KICs, {len(self.hex_secrets)} secrets)"

    def scan(self, data: bytes) -> dict[str, int]:
        """Hits of each kind in data."""
        lower = data.lower()
        return {"test_id": len(TEST_ID.findall(data)) + (len(NATURAL_ID.findall(data)) if self.natural else 0),
                "kic": sum(m.lstrip(b"0") in self.kics for m in NUMBER.findall(HEX.sub(_mask, data))),
                "secret": sum(map(lower.count, self.hex_secrets)) + sum(map(data.count, self.raw_secrets)),
                "name": len(self._names.findall(data)) if self._names else 0}


def _mask(m: re.Match) -> bytes:
    return m.group() if m.group().isdigit() else b"x" * len(m.group())


def needles(data_dir: Path = DATA, *, names: bool = False) -> Needles:
    """The needles from the evaluator's private data; FileNotFoundError when data/private is absent."""
    private = Path(data_dir) / "private"
    if not private.is_dir():
        raise FileNotFoundError(f"{private} is absent")
    pools = json.loads((private / "pools.json").read_text())
    kics = {int(k) for k in pools["natural"]}
    for lists in pools.values():
        if isinstance(lists, dict):
            kics.update(int(k) for k in lists["reserve"])
    for rel in MAPS:
        if rel in OPTIONAL and not (Path(data_dir) / rel).exists():
            continue
        kics.update(int(entry["kic"]) for entry in json.loads((Path(data_dir) / rel).read_text())["stars"].values())
    hexes = [(Path(data_dir) / rel).read_text().strip().lower() for rel in SECRETS
             if rel not in OPTIONAL or (Path(data_dir) / rel).exists()]
    return Needles(frozenset(str(k).encode() for k in kics), tuple(h.encode() for h in hexes),
                   tuple(bytes.fromhex(h) for h in hexes),
                   natural_names(Path(data_dir) / NATURAL_TRUTH) if names else (), natural=names)


def names_pattern(names) -> re.Pattern | None:
    """One case-insensitive pattern for the name pieces, longest first; None for no pieces."""
    return re.compile(b"|".join(sorted(names, key=len, reverse=True)), re.IGNORECASE) if names else None


def truth_names(truth: dict) -> tuple[bytes, ...]:
    """Regex pieces for a natural truth's names (matched in any case): every KOI name and Kepler name of a KOI signal,
    escaped; per KOI host number N, KOI[-_ ]?0*N with an optional .NN and the zero-padded KNNNNN, neither followed
    by a digit; per Kepler number K, Kepler[-_ ]?K not followed by a digit."""
    found = set()
    for entry in truth["stars"].values():
        for s in entry.get("signals") or []:
            if (s.get("source"), s.get("phase", "primary")) != ("koi", "primary"):
                continue  # TCE, Kirk and EB-secondary names are generic ("TCE 1"), not system names
            name, kepler = (str(s.get(k) or "").strip() for k in ("name", "kepler_name"))
            for full in (name, kepler):
                if full:
                    found.add(re.escape(full.encode()))
            if m := KOI_NAME.fullmatch(name):
                n = int(m.group(1))
                found.add(rb"KOI[-_ ]?0*%d(?:\.\d{1,2})?(?!\d)" % n)
                found.add(rb"K%05d(?!\d)" % n)
            if m := KEPLER_NAME.match(kepler):
                found.add(rb"Kepler[-_ ]?%d(?!\d)" % int(m.group(1)))
    return tuple(sorted(found))


def natural_names(path: Path) -> tuple[bytes, ...]:
    """truth_names of the natural truth at path; empty when it does not exist."""
    if not path.exists():
        return ()
    return truth_names(json.loads(path.read_text()))


def files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        path = Path(path)
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink())
        elif path.is_file():
            yield path


def scan_paths(paths: Iterable[Path], n: Needles) -> dict[str, dict[str, int]]:
    """{file: {kind: hits}} for every file under paths that holds a needle."""
    found = {}
    for path in files(paths):
        hits = n.scan(path.read_bytes())
        if any(hits.values()):
            found[str(path)] = {k: v for k, v in hits.items() if v}
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan files for test ids, reserve KICs and secrets.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=DATA, help="evaluator data directory (default: data/)")
    parser.add_argument("--names", action="store_true",
                        help="also natural star ids and the natural truth's KOI and Kepler names")
    args = parser.parse_args(argv)
    n = needles(args.data, names=args.names)
    found = scan_paths(args.paths, n)
    scanned = sum(1 for _ in files(args.paths))
    print(json.dumps({"files": scanned, "kics": len(n.kics), "secrets": len(n.hex_secrets), "names": len(n.names),
                      "found": found}, indent=1))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
