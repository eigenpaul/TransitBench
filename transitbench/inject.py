"""Planet transits for injection into real light curves, and the one MES function.

The transit model is the overlap of two discs times the stellar intensity at the planet's centre, with the star's own
4-coefficient Claret law, a straight chord across the disc, and 7-point supersampling across each 29.4-minute
cadence. It is a small-planet approximation, which holds for the radius ratios injected here (below about 0.1).

Injections are this model, unscaled. It was checked against NASA's pixel-level DR25 INJ1 injections, which went
through PDC, on twins (a NASA injection and ours at its ephemeris, on the same star) at NASA's expected MES 6-10
(twin_comparison): against a 1-day running median NASA's transits come out as deep as ours, and a plain BLS search
(baselines/naive_v0) detects them as often. The site (https://transitbench.eigenpaul.com/, Data and Limitations)
reports this check.

MES is the mean in-transit deficit over the light curve's own robust scatter at the transit duration, times the
square root of the number of transits with at least half their in-transit cadences present.
"""

import math
from statistics import NormalDist

import numpy as np

G = 6.674e-11
SOLAR_MASS_KG = 1.989e30
SOLAR_RADIUS_M = 6.957e8
CADENCE_DAYS = 0.0204336  # Kepler long cadence, 29.4244 min
DETREND_WINDOW_DAYS = 2.0
EB_PERIOD_SHARE = 0.10
EB_SIGMA = 3.0
EB_FALSE_ALARM = 0.00135
EB_MIN_OFF_PHASES = 8
TOO_DEEP = 0.03


def _circle_overlap(z: np.ndarray, k: float) -> np.ndarray:
    """Area of the stellar disc (radius 1) covered by a planet disc of radius k at centre distance z."""
    overlap = np.zeros_like(z)
    full = z <= abs(1.0 - k)
    overlap[full] = math.pi * min(k, 1.0) ** 2
    partial = (~full) & (z < 1.0 + k)
    d = z[partial]
    if d.size:
        a = np.clip((d**2 + k**2 - 1.0) / (2.0 * d * k), -1.0, 1.0)
        b = np.clip((d**2 + 1.0 - k**2) / (2.0 * d), -1.0, 1.0)
        rad = np.maximum(0.0, (-d + k + 1.0) * (d + k - 1.0) * (d - k + 1.0) * (d + k + 1.0))
        overlap[partial] = k**2 * np.arccos(a) + np.arccos(b) - 0.5 * np.sqrt(rad)
    return overlap


def _intensity(mu: np.ndarray, ld) -> np.ndarray:
    out = np.ones_like(mu)
    for i, c in enumerate(ld, start=1):
        out -= c * (1.0 - mu ** (i / 2.0))
    return np.maximum(out, 0.05)


def _mean_intensity(ld) -> float:
    return max(1.0 - sum(c * i / (i + 4.0) for i, c in enumerate(ld, start=1)), 0.1)


def transit_drop(time, period, t0, duration_hours, ror, b, ld=(0.0, 0.0, 0.0, 0.0), supersample: int = 7):
    """Fractional flux drop at each cadence time for a transit of the given ephemeris and shape."""
    time = np.asarray(time, dtype=float)
    offsets = np.linspace(-0.5 * CADENCE_DAYS, 0.5 * CADENCE_DAYS, supersample)
    t = time[:, None] + offsets[None, :]
    dt = (t - t0 + 0.5 * period) % period - 0.5 * period
    x = np.abs(dt) / (duration_hours / 48.0)
    chord = max((1.0 + ror) ** 2 - b**2, 1e-8)
    z = np.sqrt(b**2 + x**2 * chord)
    covered = _circle_overlap(z.ravel(), ror).reshape(z.shape) / math.pi
    mu = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(z, 1.0) ** 2))
    drop = covered * _intensity(mu, ld) / _mean_intensity(ld)
    drop[x >= 1.0] = 0.0
    return drop.mean(axis=1)


def duration_hours(period, ror, b, mass, radius, e: float = 0.0, omega: float = math.pi / 2) -> float:
    """First-to-last-contact duration (T14) from the orbit and the host star's mass and radius (solar units).

    Raises ValueError on a non-finite mass, radius or semi-major axis: the clamps below would otherwise turn a NaN
    into a transit lasting a third of the orbit.
    """
    if not (math.isfinite(mass) and math.isfinite(radius) and mass > 0 and radius > 0):
        raise ValueError(f"host mass {mass} and radius {radius} must be finite and positive")
    seconds = period * 86400.0
    a = (G * mass * SOLAR_MASS_KG * seconds**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0) / (radius * SOLAR_RADIUS_M)
    if not math.isfinite(a):
        raise ValueError(f"non-finite semi-major axis for period {period}")
    arg = math.sqrt(max(0.0, (1.0 + ror) ** 2 - b**2)) / math.sqrt(max(1e-8, a**2 - b**2))
    ecc = math.sqrt(1.0 - e**2) / (1.0 + e * math.sin(omega))
    return period * 24.0 * math.asin(min(1.0, arg)) * ecc / math.pi


def running_median(time, flux, window_days: float = DETREND_WINDOW_DAYS) -> np.ndarray:
    """Approximate running median: medians in window-wide bins every quarter window, interpolated to each cadence."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    step = window_days / 4.0
    centres = np.arange(time.min(), time.max() + step, step)
    meds = np.full(centres.size, np.nan)
    lo = np.searchsorted(time, centres - window_days / 2)
    hi = np.searchsorted(time, centres + window_days / 2)
    for i, (a, c) in enumerate(zip(lo, hi)):
        if c > a:
            meds[i] = np.median(flux[a:c])
    good = np.isfinite(meds)
    return np.interp(time, centres[good], meds[good])


def in_transit(time, period, t0, duration_hours) -> np.ndarray:
    dt = (np.asarray(time) - t0 + 0.5 * period) % period - 0.5 * period
    return np.abs(dt) < duration_hours / 48.0


def transit_count(time, period, t0, duration_hours, min_fraction: float = 0.5) -> int:
    """Transits with at least min_fraction of their expected in-transit cadences present."""
    time = np.asarray(time, dtype=float)
    expected = max(duration_hours / 24.0 / CADENCE_DAYS, 1.0)
    n0 = math.ceil((time.min() - t0) / period - 0.5)
    n1 = math.floor((time.max() - t0) / period + 0.5)
    count = 0
    for n in range(n0, n1 + 1):
        centre = t0 + n * period
        present = np.count_nonzero(np.abs(time - centre) < duration_hours / 48.0)
        count += int(present >= min_fraction * expected)
    return count


def robust_scatter(time, flux, duration_hours) -> float:
    """1.4826 x MAD of the detrended light curve averaged in bins one transit duration wide."""
    time = np.asarray(time, dtype=float)
    resid = np.asarray(flux, dtype=float) / running_median(time, flux) - 1.0
    width = duration_hours / 24.0
    edges = np.arange(time.min(), time.max() + width, width)
    idx = np.digitize(time, edges)
    counts = np.bincount(idx, minlength=edges.size + 1)
    sums = np.bincount(idx, weights=resid, minlength=edges.size + 1)
    full = counts >= 0.5 * max(width / CADENCE_DAYS, 1.0)
    means = sums[full] / counts[full]
    return float(1.4826 * np.median(np.abs(means - np.median(means))))


def _event_means(time, resid, period, t0, duration_hours, phases):
    """Mean residual of every eclipse event (at least half its cadences present, else NaN) at each trial phase:
    an array [phase, cycle], and the cycle numbers."""
    half = duration_hours / 48.0
    expected = max(duration_hours / 24.0 / CADENCE_DAYS, 1.0)
    cycles = np.arange(math.floor((time.min() - t0) / period) - 1, math.ceil((time.max() - t0) / period) + 2)
    centres = t0 + (cycles[None, :] + np.asarray(phases)[:, None]) * period
    lo, hi = np.searchsorted(time, centres - half), np.searchsorted(time, centres + half)
    csum = np.concatenate([[0.0], np.cumsum(resid)])
    count = hi - lo
    means = np.where(count >= 0.5 * expected, (csum[hi] - csum[lo]) / np.maximum(count, 1), np.nan)
    return means, cycles


def _depth_z(means, sigma):
    """Deficit significance per phase: mean event deficit over sigma / sqrt(events)."""
    n = np.sum(np.isfinite(means), axis=1)
    deficit = -np.nanmean(np.where(n[:, None] > 0, means, 0.0), axis=1)
    return np.where(n > 0, deficit / (sigma / np.sqrt(np.maximum(n, 1))), np.nan)


def _odd_even_z(means, cycles, sigma):
    odd, even = means[:, cycles % 2 == 1], means[:, cycles % 2 == 0]
    n_odd, n_even = np.sum(np.isfinite(odd), axis=1), np.sum(np.isfinite(even), axis=1)
    ok = (n_odd >= 2) & (n_even >= 2)
    diff = np.nanmean(np.where(ok[:, None], odd, 0.0), axis=1) - np.nanmean(np.where(ok[:, None], even, 0.0), axis=1)
    err = sigma * np.sqrt(1.0 / np.maximum(n_odd, 1) + 1.0 / np.maximum(n_even, 1))
    return np.where(ok, diff / err, np.nan)


def _calibrate(z, off):
    """Centre and scale of a statistic at off-eclipse phases (median and 1.4826 MAD, scale at least 1); the
    theoretical (0, 1) when fewer than EB_MIN_OFF_PHASES off phases exist."""
    v = z[off & np.isfinite(z)]
    if v.size < EB_MIN_OFF_PHASES:
        return 0.0, 1.0, False
    centre = float(np.median(v))
    return centre, max(1.4826 * float(np.median(np.abs(v - centre))), 1.0), True


def eb_tests(time, flux, period, t0, duration_hours) -> dict:
    """Binary evidence visible in this light curve at a catalogue ephemeris, with the numbers behind it.

    Each eclipse event is averaged over its duration; the noise of one event is the duration-binned robust scatter
    (robust_scatter at the eclipse duration), so a mean over n events has sigma / sqrt(n). Trial phases every half
    duration around the orbit, away from the primary, give each test its own null on this light curve (the median
    and robust spread of the statistic there).
    - secondary: the deepest trial phase anywhere (eccentric orbits included), significant after calibration at a
      false-alarm probability of EB_FALSE_ALARM over the independent trial phases;
    - odd_even: odd and even primary depths differ by more than EB_SIGMA calibrated sigma;
    - too_deep: the mean primary deficit exceeds TOO_DEEP (a fixed 3% cut, not a significance test).
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    out = {"secondary": False, "odd_even": False, "too_deep": False, "secondary_phase": None,
           "secondary_z": None, "odd_even_z": None, "primary_depth": None, "calibrated": False}
    width = duration_hours / 24.0 / period
    sigma = robust_scatter(time, flux, duration_hours)
    if not (math.isfinite(sigma) and sigma > 0 and width < 1.0):
        return out
    resid = flux / running_median(time, flux) - 1.0
    phases = np.concatenate([[0.0], np.arange(1.5 * width, 1.0 - 1.5 * width + 1e-12, 0.5 * width)])
    means, cycles = _event_means(time, resid, period, t0, duration_hours, phases)
    if not np.isfinite(means[0]).any():
        return out
    depth = -float(np.nanmean(means[0]))
    z, z_oe = _depth_z(means, sigma), _odd_even_z(means, cycles, sigma)
    out.update(primary_depth=depth, too_deep=bool(depth > TOO_DEEP))
    trial = np.arange(phases.size) > 0
    off = trial
    if (trial & np.isfinite(z)).any():
        best = int(np.nanargmax(np.where(trial, z, np.nan)))
        off = trial & (np.abs((phases - phases[best] + 0.5) % 1.0 - 0.5) >= width)
        centre, scale, calibrated = _calibrate(z, off)
        independent = max((1.0 - 2.0 * width) / width, 1.0)
        threshold = NormalDist().inv_cdf(1.0 - EB_FALSE_ALARM / independent)
        out["secondary_z"] = (float(z[best]) - centre) / scale
        out["secondary"] = bool(out["secondary_z"] > threshold)
        out["secondary_phase"] = float(phases[best]) if out["secondary"] else None
        out["calibrated"] = calibrated
    if math.isfinite(z_oe[0]):
        oe_centre, oe_scale, _ = _calibrate(z_oe, off)
        out["odd_even_z"] = (float(z_oe[0]) - oe_centre) / oe_scale
        out["odd_even"] = bool(abs(out["odd_even_z"]) > EB_SIGMA)
    return out


def eb_evidence(time, flux, period, t0, duration_hours) -> dict:
    """The three in-window evidence flags of eb_tests: secondary, odd_even, too_deep."""
    tests = eb_tests(time, flux, period, t0, duration_hours)
    return {k: tests[k] for k in ("secondary", "odd_even", "too_deep")}


def mes(time, flux, period, t0, duration_hours, drop) -> float:
    """MES of a transit with the given per-cadence drop, against the scatter of flux (the light curve it sits in)."""
    inside = in_transit(time, period, t0, duration_hours)
    if not inside.any():
        return 0.0
    deficit = float(np.mean(np.asarray(drop)[inside]))
    sigma = robust_scatter(time, flux, duration_hours)
    n = transit_count(time, period, t0, duration_hours)
    return deficit / sigma * math.sqrt(n) if sigma > 0 else math.inf


def twin_comparison(nasa, orig, period, t0, duration_hours, model_drop, window_days: float = 1.0) -> dict:
    """Compare NASA's injected light curve with ours at the same ephemeris (each a (time, flux) pair, on the same star).

    depth_ratio_local: NASA's in-transit deficit over ours, each measured against its own light curve's running
    median (window_days), which is the contrast a pipeline sees. depth_ratio_raw: NASA's deficit relative to the
    original light curve over model_drop's, on globally normalised flux; it also picks up baseline differences
    between NASA's and the archive's separately processed light curves, so a baseline lifted around the transits
    makes it read low where the local ratio does not (tests/test_inject.py).
    """
    _, ia, ib = np.intersect1d(np.round(nasa[0], 5), np.round(orig[0], 5), return_indices=True)
    t, fn, fo = orig[0][ib], nasa[1][ia], orig[1][ib]
    model = np.asarray(model_drop)[ib]
    inside = in_transit(t, period, t0, duration_hours)
    if not inside.any() or model[inside].mean() <= 0:
        return {"depth_ratio_local": math.nan, "depth_ratio_raw": math.nan, "in_transit_cadences": 0}
    ours = fo * (1.0 - model)
    local_nasa = 1.0 - float(np.mean((fn / running_median(t, fn, window_days))[inside]))
    local_ours = 1.0 - float(np.mean((ours / running_median(t, ours, window_days))[inside]))
    raw = float(np.mean((fo - fn)[inside]) / model[inside].mean())
    return {"depth_ratio_local": local_nasa / local_ours if local_ours > 0 else math.nan, "depth_ratio_raw": raw,
            "out_of_transit_rms_ppm": float(np.std((fo - fn)[~inside]) * 1e6), "in_transit_cadences": int(inside.sum())}


def solve_ror(time, flux, period, t0, b, star, e, omega, target):
    """Bisect the radius ratio so the injected transit's MES on this light curve hits the target."""
    lo, hi = 0.002, 0.3
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        dur = duration_hours(period, mid, b, star["mass"], star["radius"], e, omega)
        if dur <= 0:
            lo = mid
            continue
        m = mes(time, flux, period, t0, dur, transit_drop(time, period, t0, dur, mid, b, star["ld"]))
        lo, hi = (mid, hi) if m < target else (lo, mid)
    ror = 0.5 * (lo + hi)
    dur = duration_hours(period, ror, b, star["mass"], star["radius"], e, omega)
    return ror, dur


def draw_injection(kic, time, flux, star, rng, eb_periods, target=None):
    """Draw one planet for this light curve and return (injected flux, truth).

    MES_1yr ~ U[4, 20]; P log-uniform 0.5-100 d with at least 3 transits in the data, or copied from a real EB
    ephemeris with probability EB_PERIOD_SHARE; b ~ U(0, 1); e ~ Beta(0.867, 3.03); omega ~ U(0, 2 pi). The radius
    ratio is solved so the MES on this light curve hits the target. star holds mass, radius and ld. A given target
    (stratified samples) is kept across retries; the random stream is consumed exactly as without one.
    """
    fixed = target
    for _ in range(200):
        if rng.random() < EB_PERIOD_SHARE and eb_periods:
            period = float(rng.choice(eb_periods))
        else:
            period = float(math.exp(rng.uniform(math.log(0.5), math.log(100.0))))
        t0 = float(time.min() + rng.uniform(0, period))
        b = float(rng.uniform(0, 1))
        e = float(rng.beta(0.867, 3.03))
        omega = float(rng.uniform(0, 2 * math.pi))
        drawn = float(rng.uniform(4, 20))
        target = drawn if fixed is None else float(fixed)
        nominal = duration_hours(period, 0.02, b, star["mass"], star["radius"], e, omega)
        if nominal <= 0 or transit_count(time, period, t0, nominal) < 3:
            continue
        ror, dur = solve_ror(time, flux, period, t0, b, star, e, omega, target)
        drop = transit_drop(time, period, t0, dur, ror, b, star["ld"])
        achieved = mes(time, flux, period, t0, dur, drop)
        if transit_count(time, period, t0, dur) < 3 or not (0.9 * target < achieved < 1.1 * target):
            continue
        truth = {"period": period, "t0": t0, "duration": dur, "ror": ror, "b": b, "e": e, "omega": omega,
                 "mes": achieved, "mes_target": target, "depth_ppm": float(drop.max() * 1e6)}
        return flux * (1.0 - drop), truth
    raise RuntimeError(f"no valid injection for KIC {kic}")
