"""Match candidate transit ephemerides to true signals on one star (plan section 8).

A candidate matches a signal when their transit times line up inside the data: at least 75% of the signal's
in-data transits have a candidate transit within tau, and at least 75% of the candidate's in-data transits land on
signal transits. tau is half the signal's duration plus half an hour. A candidate at 1/3, 1/2, 2 or 3 times the
period that overlaps partly is an alias and earns half credit. Each scored signal can be claimed once; a row that
matches nothing, or a signal already claimed, is a false candidate. Rows on neutral signals are ignored.

Units: periods and epochs in days, durations in hours, coverage as [start, end] intervals in days.
"""

import math
from dataclasses import dataclass

import numpy as np

EXACT_FRACTION = 0.75
ALIAS_MIN_FRACTION = 0.2
ALIAS_RATIOS = (1 / 3, 1 / 2, 2.0, 3.0)
ALIAS_RATIO_TOLERANCE = 0.01
TAU_PAD_HOURS = 0.5
MAX_TRANSITS = 100_000


@dataclass(frozen=True)
class Signal:
    period: float
    t0: float
    duration: float
    scored: bool = True


@dataclass(frozen=True)
class Candidate:
    period: float
    t0: float
    score: float = 0.0


@dataclass(frozen=True)
class StarResult:
    credit: float
    n_false: int
    outcomes: tuple[str, ...]
    claimed: tuple[bool, ...]


def coverage_from_times(times, max_gap: float) -> np.ndarray:
    """Merge sorted cadence times into [start, end] intervals, splitting at gaps longer than max_gap days."""
    t = np.sort(np.asarray(times, dtype=float))
    if t.size == 0:
        return np.empty((0, 2))
    breaks = np.flatnonzero(np.diff(t) > max_gap)
    starts = np.concatenate([t[:1], t[breaks + 1]])
    ends = np.concatenate([t[breaks], t[-1:]])
    return np.column_stack([starts, ends])


def transit_times(period: float, t0: float, coverage) -> np.ndarray:
    """Transit midpoints t0 + n * period that fall inside the coverage intervals (sorted, non-overlapping).

    Empty when the ephemeris has more than MAX_TRANSITS transits across the coverage, or a non-finite count: a row
    such as 1e-9,0,1 would otherwise allocate about 1e11 times (docs/SCORING.md, guard)."""
    cov = np.asarray(coverage, dtype=float).reshape(-1, 2)
    if cov.size == 0:
        return np.empty(0)
    with np.errstate(over="ignore", invalid="ignore"):
        n_lo = np.ceil((cov[0, 0] - t0) / period)
        n_hi = np.floor((cov[-1, 1] - t0) / period)
        count = n_hi - n_lo + 1
    if not np.isfinite(count) or count > MAX_TRANSITS:
        return np.empty(0)
    times = t0 + period * np.arange(n_lo, n_hi + 1)
    idx = np.searchsorted(cov[:, 0], times, side="right") - 1
    inside = (idx >= 0) & (times <= cov[np.clip(idx, 0, None), 1])
    return times[inside]


def _span(coverage, pad: float) -> np.ndarray:
    cov = np.asarray(coverage, dtype=float).reshape(-1, 2)
    return np.array([[cov[0, 0] - pad, cov[-1, 1] + pad]])


def _fraction_near(a: np.ndarray, b: np.ndarray, tau: float) -> float:
    """Fraction of the times in a that have some time in b within tau."""
    if a.size == 0 or b.size == 0:
        return 0.0
    b = np.sort(b)
    idx = np.searchsorted(b, a)
    left = b[np.clip(idx - 1, 0, b.size - 1)]
    right = b[np.clip(idx, 0, b.size - 1)]
    nearest = np.minimum(np.abs(a - left), np.abs(a - right))
    return float(np.mean(nearest <= tau))


def compare(signal: Signal, candidate: Candidate, coverage) -> float:
    """Match credit of one candidate against one signal: 1 exact, 0.5 alias, 0 no match.

    Each side's transits are counted inside the coverage, but their partners are looked up on the other side's
    full ephemeris, so a partner that falls just inside a gap still counts. Exact credit needs the periods to
    agree within ALIAS_RATIO_TOLERANCE; a candidate at twice the period that skips only unobserved transits is an
    alias, not an exact match. Degenerate candidates (non-finite values, period <= 0) never match.
    """
    if not (math.isfinite(candidate.period) and math.isfinite(candidate.t0) and candidate.period > 0):
        return 0.0
    tau = (0.5 * signal.duration + TAU_PAD_HOURS) / 24.0
    true_in = transit_times(signal.period, signal.t0, coverage)
    cand_in = transit_times(candidate.period, candidate.t0, coverage)
    if true_in.size == 0 or cand_in.size == 0:
        return 0.0
    span = _span(coverage, tau)
    h_true = _fraction_near(true_in, transit_times(candidate.period, candidate.t0, span), tau)
    h_cand = _fraction_near(cand_in, transit_times(signal.period, signal.t0, span), tau)
    ratio = candidate.period / signal.period
    if abs(ratio - 1.0) <= ALIAS_RATIO_TOLERANCE and h_true >= EXACT_FRACTION and h_cand >= EXACT_FRACTION:
        return 1.0
    is_alias_ratio = any(abs(ratio / r - 1.0) <= ALIAS_RATIO_TOLERANCE for r in ALIAS_RATIOS)
    if is_alias_ratio and max(h_true, h_cand) >= EXACT_FRACTION and min(h_true, h_cand) >= ALIAS_MIN_FRACTION:
        return 0.5
    return 0.0


def match_star(signals, candidates, coverage) -> StarResult:
    """Assign candidates to signals greedily by score and count credit and false candidates."""
    signals = list(signals)
    order = sorted(range(len(candidates)), key=lambda i: -candidates[i].score)
    claimed = [False] * len(signals)
    outcomes = [""] * len(candidates)
    credit = 0.0
    n_false = 0
    for i in order:
        cand = candidates[i]
        credits = [compare(s, cand, coverage) for s in signals]
        open_scored = [j for j, s in enumerate(signals) if s.scored and not claimed[j] and credits[j] > 0]
        if open_scored:
            best = max(open_scored, key=lambda j: credits[j])
            claimed[best] = True
            credit += credits[best]
            outcomes[i] = "exact" if credits[best] == 1.0 else "alias"
        elif any(not s.scored and c > 0 for s, c in zip(signals, credits)):
            outcomes[i] = "neutral"
        else:
            n_false += 1
            outcomes[i] = "duplicate" if any(c > 0 for c in credits) else "false"
    return StarResult(credit=credit, n_false=n_false, outcomes=tuple(outcomes), claimed=tuple(claimed))
