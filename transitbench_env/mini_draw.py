"""Mini tasks, the RL training unit: the pool, the evaluator secret and the draw.

A mini task gives the agent 12 labelled stars and grades its frozen solution on 16 hidden stars. Both sets come from
a fixed pool of train-pool light curves (dev-pool ones for the held-out dev split), and every choice is a pure
function of the pool, the evaluator secret and the task's token:

    HMAC-SHA256(secret, f"{BUILDER}|{pool}|{token}|{pid}|{purpose}")

picks the stars (with a slot label in place of the pid), and per star the time offset (U[-5000, 5000) d, as in the
build), the injection's random stream and its stratified MES target, 4 + 16 (k + u) / n. TaskData carries only the
token, which means nothing without the secret. Hidden stars and their truth exist only in evaluator memory and are
rebuilt from the three.

Each star is delivered exactly as scripts/build.py delivers the public dev set: the same injection
(transitbench.inject.draw_injection with a target), the same float32 delivery (transitbench.data.deliver_flux) and
the same file bytes (transitbench.data.star_bytes). EB ephemerides, in-window evidence and variability were measured
once per pool light curve by scripts/mini_pool.py; the draw only moves them into the delivered frame.

Platforms. A token's stars are bit-defined per evaluator platform, not across platforms: draws on Mac arm64 and on
Linux x86_64 differ by about 1 ulp in the time offsets, from FMA in numpy's uniform. The grade's inputs record the
evaluator's platform, and hidden_digest differs whenever the draw does.

Variability matching (off by default; MiniData.match_variability turns it on), against the EB tell: EB hosts vary
more than INJ and NULL hosts even with their eclipses masked, so staying silent on variable stars avoids some EBs
without vetting them (the site's Limitations, https://transitbench.eigenpaul.com/#limitations). With matching, every
INJ and NULL slot draws a variability bin with the probability the set's EBs fall in it, then a host of its type
uniformly within that bin; the bins are cut between the EBs' own values. The variability is the build's: 3-h red
noise (build.star_features), with an EB's eclipses masked. Without matching, every slot's host is drawn uniformly
from its type.

Evaluator only: this module reads data/ and the secret and never goes into a box.
"""

import hashlib
import hmac
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transitbench.data import STELLAR_KEYS, deliver_flux, star_bytes
from transitbench.inject import draw_injection, transit_drop
from transitbench.match import coverage_from_times

BUILDER = "tb-mini-1"
BOX_MIX = {"inj_random": 3, "inj_tce_only": 3, "null": 4, "eb": 2}  # the 12 labelled stars
HIDDEN_MIX = {"inj_random": 4, "inj_tce_only": 4, "null": 5, "eb_flagged": 3}  # the 16 hidden stars
WINDOW = [10, 11, 12, 13]
OFFSET_DAYS = 5000.0
MES_LOW, MES_SPAN = 4.0, 16.0
NULL_TCE_SHARE = 0.5
MATCH_BINS = 8
MATCH_MIN_HOSTS = 6  # hosts of each type per bin; a bin a task has used up hands over to its nearest neighbour
POOL_DIR = "mini_pool"
SECRET = Path("private") / "mini" / "secret"
SECRET_BYTES = 32
HOST_TYPES = ("random", "tce_only")
POPULATION = {"inj_random": ("inj", "random"), "inj_tce_only": ("inj", "tce_only"),
              "null_random": ("null", "random"), "null_tce_only": ("null", "tce_only"), "eb": ("eb", "eb")}

_REF = Path(__file__).resolve().parent / "data" / "mini_pool.json"
MINI_POOL = json.loads(_REF.read_text()) if _REF.exists() else {"sha": None}

E_POOL = "TB-E002: mini pool missing or does not match its digest"
E_SECRET = "TB-E002: mini secret missing or malformed"
E_DRAW = "TB-E003: no injection reached its MES target in this draw"


class MiniPoolError(RuntimeError):
    """A mini pool, secret or draw failure. The message is one fixed "TB-Exxx: ..." text and carries no data."""


@dataclass(frozen=True, eq=False)
class Pool:
    name: str  # manifest sha256[:16]
    root: Path  # <data_dir>/mini_pool/<name>
    stars: dict  # pid -> manifest entry
    eb_periods: tuple
    quiet: dict  # split -> EB quiet threshold (90th percentile of the split's hosts)
    sets: dict  # split -> {"hosts": {type: [pid]}, "eb": {part: [pid]}, "bins": {part: [Bin]}}


@dataclass(frozen=True)
class Bin:
    upper: float  # variability upper edge (inf for the last bin)
    share: float  # share of the part's EBs in this bin
    hosts: dict  # host type -> [pid], sorted


@dataclass(frozen=True)
class Pick:
    name: str  # m-00..m-11 or h-00..h-15
    pid: str
    cell: str  # inj_random, inj_tce_only, null_random, null_tce_only or eb
    target: float | None  # MES target of an INJ star


@dataclass(frozen=True)
class MiniDraw:
    token: str
    split: str
    box_files: dict  # star id (m-00..m-11) -> .npz bytes
    box_truth: dict  # truth document, schema of dev/truth.json
    hidden_files: dict  # star id (h-00..h-15, shuffled) -> .npz bytes
    hidden_truth: dict
    roles: dict  # star id -> cell, box and hidden
    digest: str  # sha256 over the hidden files and truth


# ---- pool and secret -----------------------------------------------------------------------------------------------

def load_pool(data_dir: Path, pool: str) -> Pool:
    """The pool named by its manifest digest; MiniPoolError (TB-E002) when it is missing or the digest differs."""
    root = Path(data_dir) / POOL_DIR / str(pool)
    try:
        raw = (root / "manifest.json").read_bytes()
    except OSError:
        raise MiniPoolError(E_POOL) from None
    if hashlib.sha256(raw).hexdigest()[:16] != pool:
        raise MiniPoolError(E_POOL)
    doc = json.loads(raw)
    stars = {s["pid"]: s for s in doc["stars"]}
    sets = {split: _sets(stars, split) for split in doc["splits"]}
    return Pool(pool, root, stars, tuple(doc["eb_periods"]),
                {split: v["quiet_threshold"] for split, v in doc["splits"].items()}, sets)


def read_secret(data_dir: Path) -> bytes:
    """The evaluator secret (<data_dir>/private/mini/secret, hex); MiniPoolError (TB-E002) when missing or short."""
    try:
        key = bytes.fromhex((Path(data_dir) / SECRET).read_text().strip())
    except (OSError, ValueError):
        raise MiniPoolError(E_SECRET) from None
    if len(key) < SECRET_BYTES:
        raise MiniPoolError(E_SECRET)
    return key


def _sets(stars: dict, split: str) -> dict:
    """Drawable stars of a split: hosts by type, box-only EBs, hidden-only EBs with in-window evidence, and the
    matching bins of each part."""
    mine = [s for pid, s in sorted(stars.items()) if s["split"] == split]
    hosts = {t: [s["pid"] for s in mine if s["host"] == t] for t in HOST_TYPES}
    eb = {"box": [s["pid"] for s in mine if s["host"] == "eb" and s["role"] == "box"],
          "hidden": [s["pid"] for s in mine if s["host"] == "eb" and s["role"] == "hidden" and s["eb"]["flagged"]]}
    var = {s["pid"]: s["variability"] for s in mine}
    return {"hosts": hosts, "eb": eb,
            "bins": {part: match_bins([var[p] for p in eb[part]], hosts, var) for part in ("box", "hidden")}}


def match_bins(ref: list[float], hosts: dict, var: dict, n_bins: int = MATCH_BINS,
               min_hosts: int = MATCH_MIN_HOSTS) -> list[Bin]:
    """Variability bins for matching hosts to the reference EBs.

    Up to n_bins bins (at least 3 EBs each) with edges at the geometric midpoint between consecutive EBs at equal
    rank steps; each bin's share is the exact share of the EBs in it. A bin holding fewer than min_hosts hosts of
    either type merges with its neighbour toward the middle, until none does. Pure Python, so the bins do not depend
    on the numpy build.
    """
    ref = sorted(float(v) for v in ref)
    n = len(ref)
    if n == 0:
        raise ValueError("no reference EBs to match")
    k = max(1, min(n_bins, n // 3))
    cuts = sorted({round(i * n / k) for i in range(1, k)} - {0, n})
    uppers = [math.sqrt(ref[c - 1] * ref[c]) for c in cuts] + [math.inf]
    counts = [c - p for p, c in zip([0] + cuts, cuts + [n])]

    def members(lo, hi):
        return {t: sorted(p for p in hosts[t] if lo < var[p] <= hi) for t in HOST_TYPES}

    while True:
        lowers = [-math.inf] + uppers[:-1]
        bins = [Bin(hi, cnt / n, members(lo, hi)) for lo, hi, cnt in zip(lowers, uppers, counts)]
        thin = [i for i, b in enumerate(bins) if min(len(v) for v in b.hosts.values()) < min_hosts]
        if not thin or len(bins) == 1:
            return bins
        i = thin[0]
        j = i + 1 if i < (len(bins) - 1) / 2 else i - 1  # merge toward the middle
        lo, hi = min(i, j), max(i, j)
        del uppers[lo]
        counts[lo:hi + 1] = [counts[lo] + counts[hi]]


# ---- per-star randomness -------------------------------------------------------------------------------------------

class _Keyed:
    """HMAC-SHA256(secret, f"{BUILDER}|{pool}|{token}|{pid}|{purpose}") as a unit float or a 64-bit seed."""

    def __init__(self, secret: bytes, pool: str, token: str):
        self.key, self.prefix = secret, f"{BUILDER}|{pool}|{token}|"

    def seed(self, pid: str, purpose: str) -> int:
        mac = hmac.new(self.key, f"{self.prefix}{pid}|{purpose}".encode(), hashlib.sha256).digest()
        return int.from_bytes(mac[:8], "big")

    def unit(self, pid: str, purpose: str) -> float:
        return self.seed(pid, purpose) / 2**64


# ---- selection -----------------------------------------------------------------------------------------------------

def plan(pool: Pool, secret: bytes, split: str, token: str, *, match_variability: bool = False) -> tuple[list, list]:
    """Which stars a task holds, without building them: (box, hidden) lists of Pick, each sorted by name.

    Hidden: HIDDEN_MIX, 3 EBs from the hidden-only EBs with in-window evidence and 2 or 3 of the 5 NULL on TCE-only
    hosts at equal odds. Box: BOX_MIX from the box-only EBs and the hosts the hidden set left. A host's role (INJ or
    NULL) is drawn per task, so a host can be INJ in one task and NULL in another, never both in one task.
    """
    if split not in pool.sets:
        raise ValueError(f"split {split!r} not in the pool")
    keyed = _Keyed(secret, pool.name, token)
    sets = pool.sets[split]
    rank: dict[str, float] = {}

    def rank_of(pid):
        if pid not in rank:
            rank[pid] = keyed.unit(pid, "pick")
        return rank[pid]

    used: set[str] = set()
    parts = []
    for part, prefix, mix, n_eb in (("hidden", "h", HIDDEN_MIX, HIDDEN_MIX["eb_flagged"]),
                                    ("box", "m", BOX_MIX, BOX_MIX["eb"])):
        null_tce = math.floor(mix["null"] * NULL_TCE_SHARE + keyed.unit(prefix, "null-tce"))
        cells = [("inj_random", mix["inj_random"]), ("inj_tce_only", mix["inj_tce_only"]),
                 ("null_random", mix["null"] - null_tce), ("null_tce_only", null_tce)]
        picks = []
        for cell, n in cells:
            host = POPULATION[cell][1]
            for j in range(n):
                slot = f"{prefix}:{cell}:{j}"
                if match_variability:
                    pid = _matched_host(sets["bins"][part], host, keyed.unit(slot, "bin"), used, rank_of)
                else:
                    pid = min((p for p in sets["hosts"][host] if p not in used), key=rank_of)
                used.add(pid)
                picks.append((pid, cell))
        ebs = sorted(sets["eb"][part], key=rank_of)[:n_eb]
        if len(ebs) < n_eb:
            raise MiniPoolError(E_POOL)
        picks += [(pid, "eb") for pid in ebs]
        targets = {}
        for cell in ("inj_random", "inj_tce_only"):
            members = sorted((pid for pid, c in picks if c == cell), key=lambda p: keyed.unit(p, "target-order"))
            for i, pid in enumerate(members):
                targets[pid] = MES_LOW + MES_SPAN * (i + keyed.unit(pid, "target-jitter")) / len(members)
        order = sorted(picks, key=lambda pc: keyed.unit(pc[0], "order"))
        parts.append([Pick(f"{prefix}-{i:02d}", pid, cell, targets.get(pid)) for i, (pid, cell) in enumerate(order)])
    hidden, box = parts
    return box, hidden


def _matched_host(bins: list[Bin], host: str, u: float, used: set, rank_of) -> str:
    """The first-ranked unused host of the bin u falls in (bins weighted by their EB share); if every host of that bin
    is taken, the nearest bin with one left."""
    cum, b = 0.0, len(bins) - 1
    for i, bn in enumerate(bins):
        cum += bn.share
        if u < cum:
            b = i
            break
    for i in sorted(range(len(bins)), key=lambda i: (abs(i - b), i)):
        free = [p for p in bins[i].hosts[host] if p not in used]
        if free:
            return min(free, key=rank_of)
    raise MiniPoolError(E_POOL)


# ---- delivery ------------------------------------------------------------------------------------------------------

def deliver(lc: tuple, stellar: dict, offset: float, injection: tuple | None = None) -> tuple[dict, dict | None]:
    """One star as scripts/build.py's build_star delivers it: (arrays for star_bytes, injected signal or None).

    lc is (time, flux, quarter, qmedian) in BKJD, sorted by time; stellar the delivered (rounded) parameters;
    injection (star, rng, eb_periods, target) for an INJ star, star holding mass, radius and ld. The signal's t0 is
    in the delivered frame. Raises RuntimeError when no injection reaches the target.
    """
    time, flux, quarter, qmedian = lc
    drop, signal = None, None
    if injection is not None:
        star, rng, eb_periods, target = injection
        _, sig = draw_injection("mini", time, flux, star, rng, eb_periods, target=target)
        drop = transit_drop(time, sig["period"], sig["t0"], sig["duration"], sig["ror"], sig["b"], star["ld"])
        signal = {"period": sig["period"], "t0": sig["t0"] + offset, "duration": sig["duration"], "scored": True,
                  "kind": "injected", "mes": sig["mes"],
                  **{k: sig[k] for k in ("mes_target", "ror", "b", "e", "omega", "depth_ppm")}}
    arrays = {"time": time + offset, "flux": deliver_flux(flux, quarter, qmedian, drop),
              "quarter": quarter.astype(np.int8), **{k: np.float64(stellar[k]) for k in STELLAR_KEYS}}
    return arrays, signal


def _load(pool: Pool, pid: str) -> tuple:
    try:
        raw = (pool.root / "stars" / f"{pid}.npz").read_bytes()
    except OSError:
        raise MiniPoolError(E_POOL) from None
    if hashlib.sha256(raw).hexdigest() != pool.stars[pid]["sha256"]:
        raise MiniPoolError(E_POOL)
    d = np.load(io.BytesIO(raw))
    return d["time"], d["flux"], d["quarter"], d["qmedian"]


def _build(pool: Pool, keyed: _Keyed, split: str, pick: Pick, target: float | None) -> tuple[bytes, dict]:
    """File bytes and truth entry of one picked star (RuntimeError when an injection misses its target)."""
    entry = pool.stars[pick.pid]
    population, host = POPULATION[pick.cell]
    lc = _load(pool, pick.pid)
    offset = float(np.random.default_rng(keyed.seed(pick.pid, "offset")).uniform(-OFFSET_DAYS, OFFSET_DAYS))
    injection = None
    if population == "inj":
        star = {"mass": entry["inject"]["mass"], "radius": entry["inject"]["radius"],
                "ld": tuple(entry["inject"]["ld"])}
        injection = (star, np.random.default_rng(keyed.seed(pick.pid, "inject")), pool.eb_periods, target)
    arrays, signal = deliver(lc, entry["stellar"], offset, injection)
    truth = {"population": population, "host": host, "scored": population != "eb", "coverage": None, "signals": []}
    if signal:
        truth["signals"] = [{**signal, "eb_period_copy": signal["period"] in _period_set(pool)}]
    if population == "eb":
        eb = entry["eb"]
        truth["eb"] = {"period": eb["period"], "t0": eb["t0"] + offset, "duration": eb["duration"],
                       "evidence": dict(eb["evidence"]), "source": eb["source"],
                       "secondary_phase": eb["secondary_phase"], "variability": entry["variability"],
                       "quiet": bool(entry["variability"] <= pool.quiet[split])}
        truth["scored"] = any(eb["evidence"].values())
    truth["coverage"] = coverage_from_times(arrays["time"], max_gap=0.5).tolist()
    return star_bytes(arrays), truth


_PERIOD_SETS: dict = {}


def _period_set(pool: Pool) -> frozenset:
    if pool.name not in _PERIOD_SETS:
        _PERIOD_SETS[pool.name] = frozenset(pool.eb_periods)
    return _PERIOD_SETS[pool.name]


def _build_part(pool: Pool, keyed: _Keyed, split: str, picks: list[Pick]) -> dict[str, tuple[bytes, dict]]:
    """Build every star of a part. An INJ star whose target cannot be reached swaps targets with the star of its cell
    holding the lowest target not yet swapped, both rebuilt, as scripts/build.py does; no swap raises TB-E003."""
    out, failed = {}, []
    for p in picks:
        try:
            out[p.name] = _build(pool, keyed, split, p, p.target)
        except RuntimeError as e:
            if isinstance(e, MiniPoolError) or p.target is None:
                raise
            failed.append(p)
    swapped: set[str] = set()
    for p in failed:
        partners = sorted((q for q in picks if q.cell == p.cell and q.name != p.name and q.name not in swapped
                           and q.name in out), key=lambda q: q.target)
        for q in partners:
            try:
                a, b = _build(pool, keyed, split, p, q.target), _build(pool, keyed, split, q, p.target)
            except RuntimeError as e:
                if isinstance(e, MiniPoolError):
                    raise
                continue
            out[p.name], out[q.name] = a, b
            swapped |= {p.name, q.name}
            break
        else:
            raise MiniPoolError(E_DRAW)
    return out


def draw(pool: Pool, secret: bytes, split: str, token: str, *, match_variability: bool = False) -> MiniDraw:
    """One mini task, a pure function of (pool, secret, split, token, match_variability)."""
    keyed = _Keyed(secret, pool.name, token)
    box, hidden = plan(pool, secret, split, token, match_variability=match_variability)
    docs = {}
    for part, picks in (("box", box), ("hidden", hidden)):
        built = _build_part(pool, keyed, split, picks)
        docs[part] = ({p.name: built[p.name][0] for p in picks},
                      {"release": f"mini-{split}", "sample": part, "token": token, "window": WINDOW,
                       "stars": {p.name: built[p.name][1] for p in picks}})
    (box_files, box_truth), (hidden_files, hidden_truth) = docs["box"], docs["hidden"]
    return MiniDraw(token, split, box_files, box_truth, hidden_files, hidden_truth,
                    {p.name: p.cell for p in box + hidden}, hidden_digest(hidden_files, hidden_truth))


def hidden_digest(files: dict, truth: dict) -> str:
    """sha256 of the canonical JSON of {"files": {id: sha256}, "truth": truth}."""
    doc = {"files": {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}, "truth": truth}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
