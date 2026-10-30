"""Score a catalogue of candidates against the hidden truth (plan section 8); docs/SCORING.md defines every number.

S = (sum of match credit - false candidates) / planets, over the scored stars (INJ, NULL and EBs with in-window
evidence), always shown with C (completeness), F (false candidates per 100 scored stars) and the EB-as-planet rate
(share of scored EB stars with any false row). Each star keeps at most MAX_ROWS rows, the highest-scoring ones; a
star missing from the catalogue, or whose run did not end "ok", has none. REAL hosts and EBs without in-window
evidence never enter S. The RL reward is the same sum over the scored stars, over a fixed norm, clipped to [-1, 1].
The science report adds C by MES bin, NULL false alarms with exact intervals, EB rates, the score sweep and the
REAL transfer check; it never ranks.

Standard library and numpy only: this file ships to agents in an offline box.

CLI: python -m transitbench.score --truth TRUTH.json --preds PREDS.jsonl [--out REPORT.json]
"""

import argparse
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from transitbench.match import Candidate, Signal, StarResult, match_star

MAX_ROWS = 3
MINI_NORM = 8.0
MES_THRESHOLD = 7.1
REAL_PERIOD_RANGE = (0.5, 100.0)
MES_BINS = ((4.0, 7.1), (7.1, 10.0), (10.0, 15.0), (15.0, 20.0))
REAL_BINS = MES_BINS[1:] + ((20.0, math.inf),)
F_LIMIT = 1.0
CI_LEVEL = 0.95
HOSTS = {"inj": ("random", "tce_only"), "null": ("random", "tce_only"), "eb": ("eb",), "real": ("real",)}
KINDS = ("injected", "confirmed", "koi", "tce")


@dataclass(frozen=True)
class TruthSignal(Signal):
    kind: str = "injected"
    mes: float | None = None


@dataclass
class Star:
    star_id: str
    population: str
    coverage: object
    signals: list[Signal] = field(default_factory=list)
    scored: bool = True
    host: str = ""
    quiet: bool | None = None


@dataclass
class _Graded:
    star: Star
    rows: list[Candidate]
    prefix: list[tuple[float, int]]
    result: StarResult
    signal_credit: list[float]


def parse_truth(doc: dict) -> list[Star]:
    """Stars from a truth document; raises ValueError naming the first star that breaks a truth rule."""
    return [_parse_star(sid, entry) for sid, entry in doc["stars"].items()]


def load_truth(path) -> list[Star]:
    return parse_truth(json.loads(Path(path).read_text()))


def _parse_star(sid: str, t: dict) -> Star:
    signals = [TruthSignal(float(s["period"]), float(s["t0"]), float(s["duration"]), bool(s["scored"]), s["kind"],
                           None if s.get("mes") is None else float(s["mes"])) for s in t["signals"]]
    coverage = np.asarray(t["coverage"], dtype=float).reshape(-1, 2)
    quiet = (t.get("eb") or {}).get("quiet") if t["population"] == "eb" else None
    star = Star(sid, t["population"], coverage[np.argsort(coverage[:, 0])], signals, bool(t["scored"]), t["host"],
                quiet)
    problem = _broken_rule(star, t.get("eb"))
    if problem:
        raise ValueError(f"truth star {sid}: {problem}")
    return star


def _broken_rule(star: Star, eb: dict | None) -> str:
    """The first truth rule (plan 6.4) the star breaks, or an empty string."""
    pop, signals = star.population, star.signals
    if star.host not in HOSTS.get(pop, ()):
        return f"population {pop!r} with host {star.host!r}"
    if pop == "eb" and not (eb and isinstance(eb.get("evidence"), dict)):
        return "EB without evidence flags"
    if pop == "eb" and "quiet" in eb and not isinstance(eb["quiet"], bool):
        return "EB 'quiet' must be a bool"
    if pop == "eb" and "variability" in eb and not _finite_number(eb["variability"]):
        return "EB 'variability' must be a finite number"
    expected = any(eb["evidence"].values()) if pop == "eb" else pop != "real"
    if star.scored != expected:
        return f"{pop} star with scored={star.scored}, expected {expected}"
    if star.coverage.size == 0:
        return "no coverage"
    if any(s.kind not in KINDS or not (s.period > 0 and s.duration > 0 and math.isfinite(s.t0)) for s in signals):
        return "a signal with an unknown kind or a degenerate ephemeris"
    if pop == "inj" and not (signals and all(s.scored for s in signals)):
        return "INJ stars carry scored signals only, at least one"
    if pop in ("null", "eb") and signals:
        return "NULL and EB stars carry no signals; the EB ephemeris goes in 'eb'"
    if pop == "real" and any(s.scored != _real_target(s) for s in signals):
        return f"REAL signals are scored exactly when confirmed, P in {REAL_PERIOD_RANGE} d and MES >= {MES_THRESHOLD}"
    return ""


def _finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _real_target(s: TruthSignal) -> bool:
    lo, hi = REAL_PERIOD_RANGE
    return s.kind == "confirmed" and lo <= s.period <= hi and (s.mes or 0.0) >= MES_THRESHOLD


def load_preds(path) -> dict[str, dict]:
    """Runner records (JSON lines of star, status, cpu_s, rows) keyed by star; a star may appear only once."""
    runs = {}
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            rec = json.loads(line)
            if rec["star"] in runs:
                raise ValueError(f"{path}:{n}: star {rec['star']} appears twice")
            runs[rec["star"]] = rec
    return runs


def catalogue_from_runs(runs: dict[str, dict]) -> tuple[dict[str, list[Candidate]], dict]:
    """Catalogue from runner records, plus counts for the report.

    A star whose status is not "ok" (crash, timeout) has no rows; a missing status counts as "ok". Rows may be
    Candidates or [period, t0, score] lists; a row that is not three finite numbers with period > 0 is dropped.
    """
    catalogue, not_ok, invalid = {}, Counter(), 0
    for sid, rec in runs.items():
        status = rec.get("status", "ok")
        if status != "ok":
            not_ok[status] += 1
        rows = [_candidate(r) for r in rec.get("rows", [])] if status == "ok" else []
        invalid += rows.count(None)
        catalogue[sid] = [r for r in rows if r is not None]
    return catalogue, {"runs": len(runs), "not_ok": dict(not_ok), "rows_invalid": invalid}


def _candidate(row) -> Candidate | None:
    values = (row.period, row.t0, row.score) if isinstance(row, Candidate) else row
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    if not all(_finite_number(v) for v in values):
        return None
    return Candidate(*map(float, values)) if values[0] > 0 else None


def _grade(star: Star, rows: list[Candidate]) -> _Graded:
    """Match the star's best MAX_ROWS valid rows. Greedy matching by score is prefix-stable, so matching each
    prefix of the sorted rows gives the score sweep and which signal each row claimed, with the same matcher."""
    rows = sorted((c for c in map(_candidate, rows) if c is not None), key=lambda c: -c.score)[:MAX_ROWS]
    results = [match_star(star.signals, rows[:k], star.coverage) for k in range(len(rows) + 1)]
    credit = [0.0] * len(star.signals)
    for before, after in itertools.pairwise(results):
        for j, (was, now) in enumerate(zip(before.claimed, after.claimed)):
            if now and not was:
                credit[j] = after.credit - before.credit
    return _Graded(star, rows, [(r.credit, r.n_false) for r in results], results[-1], credit)


def _grade_all(stars: list[Star], catalogue: dict[str, list[Candidate]]) -> list[_Graded]:
    return [_grade(s, catalogue.get(s.star_id, [])) for s in stars]


def _group(graded: list[_Graded]) -> dict:
    """Counts over a group of stars, with its completeness and false candidates per 100 stars."""
    planets = sum(s.scored for g in graded for s in g.star.signals)
    credit = sum(g.result.credit for g in graded)
    false = sum(g.result.n_false for g in graded)
    return {"stars": len(graded), "planets": planets, "credit": credit, "false": false,
            "stars_with_false": sum(g.result.n_false > 0 for g in graded),
            "stars_with_rows": sum(bool(g.rows) for g in graded),
            "C": credit / max(planets, 1), "F_per_100": 100.0 * false / max(len(graded), 1)}


def _headline(graded: list[_Graded]) -> dict:
    scored = [g for g in graded if g.star.scored]
    total = _group(scored)
    eb = [g for g in scored if g.star.population == "eb"]
    planets = max(total["planets"], 1)
    by_host = {pop: {host: _group([g for g in graded if g.star.population == pop and g.star.host == host])
                     for host in HOSTS[pop]}
               for pop in ("inj", "null") if any(g.star.population == pop for g in graded)}
    return {
        "S": (total["credit"] - total["false"]) / planets,
        "C": total["credit"] / planets,
        "F_per_100": total["F_per_100"],
        "eb_as_planet": sum(g.result.n_false > 0 for g in eb) / max(len(eb), 1),
        "planets": total["planets"],
        "scored_stars": total["stars"],
        "credit": total["credit"],
        "false": total["false"],
        "populations": {pop: _group([g for g in graded if g.star.population == pop])
                        for pop in dict.fromkeys(g.star.population for g in graded)},
        "by_host": by_host,
    }


def score(stars: list[Star], catalogue: dict[str, list[Candidate]]) -> dict:
    """Headline numbers over the scored stars, per population and per host (random vs TCE-only)."""
    return _headline(_grade_all(stars, catalogue))


def reward(stars: list[Star], catalogue: dict[str, list[Candidate]], norm: float = MINI_NORM) -> float:
    """RL reward (plan section 8): clip((sum of credit - false candidates) / norm, -1, 1) over the scored stars.

    Build the catalogue with catalogue_from_runs, so a crash counts as an empty catalogue and invalid rows drop.
    """
    graded = _grade_all([s for s in stars if s.scored], catalogue)
    net = sum(g.result.credit - g.result.n_false for g in graded)
    return min(1.0, max(-1.0, net / norm))


def _binomial_tail(lo: int, hi: int, n: int, p: float) -> float:
    """P(lo <= X <= hi) for X ~ Binomial(n, p)."""
    if p <= 0.0 or p >= 1.0:
        return float(lo <= (0 if p <= 0.0 else n) <= hi)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p)
            + (n - i) * math.log1p(-p) for i in range(lo, hi + 1)]
    return sum(math.exp(v) for v in logs)


def _root(f, iterations: int = 100) -> float:
    """Root in [0, 1] of an increasing function with f(0) < 0 < f(1), by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, level: float = CI_LEVEL) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval for a binomial rate k / n, by bisection on the binomial CDF."""
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return 0.0, 1.0
    tail = (1.0 - level) / 2
    lo = 0.0 if k == 0 else _root(lambda p: _binomial_tail(k, n, n, p) - tail)
    hi = 1.0 if k == n else _root(lambda p: tail - _binomial_tail(0, k, n, p))
    return lo, hi


def _rate(graded: list[_Graded]) -> dict:
    """Share of stars with any row, with its exact 95% interval."""
    k, n = sum(bool(g.rows) for g in graded), len(graded)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": list(clopper_pearson(k, n))}


def _eb_rates(eb: list[_Graded], null: list[_Graded]) -> dict:
    """EB-as-planet by slice, and the abstention check: each scored slice's rate minus the NULL any-row rate, the
    rate a pipeline gets by claiming on EBs as often as on NULL stars. The quiet/variable split needs the truth's
    'quiet' flag on every scored EB; without it those slices are None."""
    flagged = [g for g in eb if g.star.scored]
    split = all(g.star.quiet is not None for g in flagged)
    rates = {"flagged": _rate(flagged), "four_year_label": _rate([g for g in eb if not g.star.scored]),
             "flagged_quiet": _rate([g for g in flagged if g.star.quiet]) if split else None,
             "flagged_variable": _rate([g for g in flagged if not g.star.quiet]) if split else None}
    null_rate = _rate(null)["rate"]
    check = {"null_rate": null_rate}
    for name in ("flagged", "flagged_quiet", "flagged_variable"):
        rate = (rates[name] or {}).get("rate")
        check[f"{name}_minus_null"] = None if rate is None or null_rate is None else rate - null_rate
    return {**rates, "abstention_check": check}


def _bin_label(lo: float, hi: float) -> str:
    return f"{lo:g}+" if math.isinf(hi) else f"{lo:g}-{hi:g}"


def _completeness(graded: list[_Graded], bins) -> list[dict]:
    """Credit of the scored signals with a MES, binned by MES: n, mean MES and C per bin (alias credit 0.5)."""
    pairs = [[] for _ in bins]
    for g in graded:
        for s, c in zip(g.star.signals, g.signal_credit):
            m = getattr(s, "mes", None)
            if not s.scored or m is None:
                continue
            for i, (lo, hi) in enumerate(bins):
                if lo <= m < hi or m == hi == bins[-1][1]:
                    pairs[i].append((m, c))
                    break
    out = []
    for (lo, hi), p in zip(bins, pairs):
        credit = sum(c for _, c in p)
        out.append({"mes": _bin_label(lo, hi), "n": len(p), "mean_mes": sum(m for m, _ in p) / len(p) if p else None,
                    "credit": credit, "C": credit / len(p) if p else None})
    return out


def _mes50(curve: list[dict]) -> tuple[float | None, str]:
    """MES where C first reaches 0.5, interpolating linearly between bins at their mean MES."""
    points = [(b["mean_mes"], b["C"]) for b in curve if b["n"]]
    if not points:
        return None, "no planets with a MES"
    for i, (x, y) in enumerate(points):
        if y >= 0.5:
            if i == 0:
                return None, "C >= 0.5 already in the lowest bin"
            x0, y0 = points[i - 1]
            return x0 + (0.5 - y0) * (x - x0) / (y - y0), "interpolated"
    return None, "C < 0.5 in every bin"


def _sweep(scored: list[_Graded]) -> dict:
    """C and F as the score threshold falls, row by row; the best C with F <= F_LIMIT per 100 stars."""
    planets = max(sum(s.scored for g in scored for s in g.star.signals), 1)
    n = len(scored)
    events = sorted(((row.score, i, k + 1) for i, g in enumerate(scored) for k, row in enumerate(g.rows)),
                    key=lambda e: -e[0])
    kept = [0] * n
    credit, false = 0.0, 0
    points = [(None, 0.0, 0)]
    for idx, (threshold, i, k) in enumerate(events):
        old = scored[i].prefix[kept[i]]
        kept[i] = max(kept[i], k)
        new = scored[i].prefix[kept[i]]
        credit, false = credit + new[0] - old[0], false + new[1] - old[1]
        if idx + 1 == len(events) or events[idx + 1][0] != threshold:
            points.append((threshold, credit, false))
    best = max((p for p in points if 100 * p[2] <= F_LIMIT * n), key=lambda p: p[1])
    per_100 = 100.0 / max(n, 1)
    return {
        "F_limit_per_100": F_LIMIT,
        "C_at_F_limit": best[1] / planets, "S_at_F_limit": (best[1] - best[2]) / planets,
        "F_per_100_at_F_limit": best[2] * per_100, "threshold": best[0],
        "curve": [{"threshold": t, "C": c / planets, "F_per_100": f * per_100} for t, c, f in points],
    }


def _real(real: list[_Graded], inj_curve: list[dict]) -> dict:
    """Recall on REAL hosts, the gap to the recall INJ completeness predicts at their MES, and their rows."""
    planets = [(s.mes, c) for g in real for s, c in zip(g.star.signals, g.signal_credit)
               if s.scored and getattr(s, "mes", None) is not None]
    points = [(b["mean_mes"], b["C"]) for b in inj_curve if b["n"]]
    observed = sum(c for _, c in planets) / len(planets) if planets else None
    predicted = None
    if planets and points:
        xs, ys = zip(*points)
        predicted = float(np.mean(np.interp([m for m, _ in planets], xs, ys)))
    outcomes = Counter(o for g in real for o in g.result.outcomes)
    return {
        "stars": len(real), "planets": len(planets),
        "recall": observed, "recall_by_mes": _completeness(real, REAL_BINS),
        "predicted_from_inj": predicted,
        "transfer_gap": observed - predicted if predicted is not None else None,
        "rows": {"total": sum(outcomes.values()), "matched": outcomes["exact"] + outcomes["alias"],
                 "neutral": outcomes["neutral"], "duplicate": outcomes["duplicate"], "unknown": outcomes["false"],
                 "unknown_per_100_stars": 100.0 * outcomes["false"] / max(len(real), 1)},
    }


def _pick(graded: list[_Graded], pop: str, host: str | None = None) -> list[_Graded]:
    return [g for g in graded if g.star.population == pop and host in (None, g.star.host)]


def report(stars: list[Star], catalogue: dict[str, list[Candidate]]) -> dict:
    """The headline plus the science report (plan section 8). Only the headline ranks."""
    graded = _grade_all(stars, catalogue)
    inj, null, eb = _pick(graded, "inj"), _pick(graded, "null"), _pick(graded, "eb")
    inj_curve = _completeness(inj, MES_BINS)
    mes50, note = _mes50(inj_curve)
    unbinned = sum(s.scored for g in inj for s in g.star.signals) - sum(b["n"] for b in inj_curve)
    ids = {s.star_id for s in stars}
    return {
        "headline": _headline(graded),
        "completeness": {"by_mes": inj_curve, "unbinned": unbinned, "mes50": mes50, "mes50_note": note},
        "null_false_alarms": {"all": _rate(null), "random": _rate(_pick(null, "null", "random")),
                              "tce_only": _rate(_pick(null, "null", "tce_only"))},
        "eb_as_planet": _eb_rates(eb, null),
        "sweep": _sweep([g for g in graded if g.star.scored]),
        "real": _real(_pick(graded, "real"), inj_curve),
        "inputs": {"missing_stars": sum(sid not in catalogue for sid in ids),
                   "unknown_stars": sum(sid not in ids for sid in catalogue),
                   "rows_over_cap": sum(max(len(catalogue.get(sid, [])) - MAX_ROWS, 0) for sid in ids)},
    }


def format_headline(rep: dict) -> str:
    h, i = rep["headline"], rep["inputs"]
    return (f"S = {h['S']:.4f}  C = {h['C']:.4f}  F = {h['F_per_100']:.2f} per 100 stars  "
            f"EB-as-planet = {h['eb_as_planet']:.4f}\n"
            f"{h['planets']} planets on {h['scored_stars']} scored stars; runs not ok: {i.get('not_ok', {})}, "
            f"invalid rows: {i.get('rows_invalid', 0)}, rows over cap: {i['rows_over_cap']}, "
            f"stars without preds: {i['missing_stars']}, preds for unknown stars: {i['unknown_stars']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Score a catalogue against the truth (docs/SCORING.md).")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    doc = json.loads(args.truth.read_text())
    catalogue, inputs = catalogue_from_runs(load_preds(args.preds))
    rep = {"release": doc.get("release"), "sample": doc.get("sample"), "window": doc.get("window"),
           **report(parse_truth(doc), catalogue)}
    rep["inputs"].update(inputs)
    print(format_headline(rep))
    if args.out:
        args.out.write_text(json.dumps(rep, indent=2) + "\n")


if __name__ == "__main__":
    main()
