"""Candidate generation and vetting features for the learned-vetter baseline (plan section 11, baseline 4).

For one star: detrend with a 1-day running median, run BLS on the starter's grid, and take up to three peaks above
SNR 7.1, masking each signal (primary and phase 0.5) before searching again. Every peak gets one feature row; a
gradient-boosted classifier trained on labelled train-pool stars then turns rows into scored claims.
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares

P_MIN, P_MAX = 0.5, 100.0
DURATIONS_HOURS = (1.0, 2.5, 5.0, 10.0)
SNR_MIN = 7.1
MAX_PEAKS = 3
FEATURES = (
    "snr", "log_period", "duration_h", "log_depth", "n_transits", "single_event_share", "odd_even_sigma",
    "secondary_sigma", "shape_ratio", "duration_ratio", "max_quarter_share", "rank", "snr_ratio_prev",
    "red_noise", "variability", "period_ratio_prev",
)


def detrend(time, flux, window=1.0):
    step = window / 4.0
    centres = np.arange(time.min(), time.max() + step, step)
    lo = np.searchsorted(time, centres - window / 2)
    hi = np.searchsorted(time, centres + window / 2)
    meds = np.array([np.median(flux[a:b]) if b > a else np.nan for a, b in zip(lo, hi)])
    good = np.isfinite(meds)
    trend = np.interp(time, centres[good], meds[good])
    return flux / trend, trend


def period_grid(baseline):
    periods = [P_MIN]
    while periods[-1] < P_MAX:
        p = periods[-1]
        duration = max(13.0 / 24.0 * (p / 365.25) ** (1.0 / 3.0), 1.0 / 24.0)
        periods.append(p * (1.0 + duration / (3.0 * baseline)))
    return np.array(periods)


def phase(time, period, t0):
    return (time - t0 + 0.5 * period) % period - 0.5 * period


def _binned_scatter(time, resid, width):
    edges = np.arange(time.min(), time.max() + width, width)
    idx = np.digitize(time, edges)
    counts = np.bincount(idx)
    sums = np.bincount(idx, weights=resid)
    full = counts >= max(2, 0.5 * width / 0.0204)
    means = sums[full] / counts[full]
    return 1.4826 * np.median(np.abs(means - np.median(means))), np.median(counts[full])


def star_features(time, flux, raw):
    resid = flux - 1.0
    point = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    binned, n = _binned_scatter(time, resid, 6.0 / 24.0)
    red = binned / (point / np.sqrt(max(n, 1))) if point > 0 else 1.0
    var = 1.4826 * np.median(np.abs(raw - np.median(raw))) / point if point > 0 else 1.0
    return point, float(red), float(var)


def peak_features(time, flux, quarter, period, t0, duration, snr, point):
    dt = phase(time, period, t0)
    inside = np.abs(dt) < duration / 2
    epochs = np.round((time - t0) / period).astype(int)
    depths, errs, numbers, quarters = [], [], [], []
    for n in np.unique(epochs[inside]):
        sel = inside & (epochs == n)
        if sel.sum() >= 3:
            depths.append(1.0 - flux[sel].mean())
            errs.append(point / np.sqrt(sel.sum()))
            numbers.append(n)
            quarters.append(int(np.bincount(quarter[sel]).argmax()))
    depths, errs, numbers, quarters = map(np.asarray, (depths, errs, numbers, quarters))
    out = {"n_transits": float(depths.size)}
    if depths.size == 0:
        return {**out, "single_event_share": 1.0, "odd_even_sigma": 0.0, "secondary_sigma": 0.0,
                "shape_ratio": 1.0, "max_quarter_share": 1.0, "log_depth": -9.0}
    s2 = (depths / errs) ** 2
    out["single_event_share"] = float(s2.max() / max(s2.sum(), 1e-12))
    by_quarter = np.bincount(quarters, weights=s2)
    out["max_quarter_share"] = float(by_quarter.max() / max(s2.sum(), 1e-12))
    w = 1.0 / errs**2
    depth = float(np.sum(w * depths) / np.sum(w))
    out["log_depth"] = float(np.log10(max(depth, 1e-6)))
    odd, even = numbers % 2 == 1, numbers % 2 == 0
    if odd.any() and even.any():
        def mean(m):
            ww = w[m]
            return np.sum(ww * depths[m]) / np.sum(ww), 1.0 / np.sqrt(np.sum(ww))
        (do, eo), (de, ee) = mean(odd), mean(even)
        out["odd_even_sigma"] = float(abs(do - de) / np.hypot(eo, ee))
    else:
        out["odd_even_sigma"] = 0.0
    sec = np.abs(phase(time, period, t0 + 0.5 * period)) < duration / 2
    out["secondary_sigma"] = float((1.0 - flux[sec].mean()) / (point / np.sqrt(sec.sum()))) if sec.sum() >= 3 else 0.0
    core = np.abs(dt) < duration / 4
    edge = inside & ~core
    if core.sum() >= 3 and edge.sum() >= 3 and depth > 0:
        out["shape_ratio"] = float((1.0 - flux[edge].mean()) / max(1.0 - flux[core].mean(), 1e-9))
    else:
        out["shape_ratio"] = 1.0
    return out


def candidates(time, flux, quarter):
    """Up to MAX_PEAKS candidates as (period, t0, feature dict)."""
    order = np.argsort(time)
    time, raw, quarter = time[order], flux[order], quarter[order]
    flux, trend = detrend(time, raw)
    point, red, var = star_features(time, flux, trend)
    keep = np.ones(time.size, bool)
    out, prev = [], None
    for rank in range(MAX_PEAKS):
        if keep.sum() < 100:
            break
        bls = BoxLeastSquares(time[keep], flux[keep], dy=np.full(keep.sum(), point))
        res = bls.power(period_grid(time.max() - time.min()), np.array(DURATIONS_HOURS) / 24.0, objective="snr")
        i = int(np.argmax(res.power))
        period, t0, duration, snr = (float(res.period[i]), float(res.transit_time[i]), float(res.duration[i]),
                                     float(res.depth_snr[i]))
        if snr < SNR_MIN:
            break
        f = peak_features(time[keep], flux[keep], quarter[keep], period, t0, duration, snr, point)
        f.update({"snr": snr, "log_period": float(np.log10(period)), "duration_h": duration * 24.0, "rank": float(rank),
                  "snr_ratio_prev": snr / prev[1] if prev else 1.0,
                  "period_ratio_prev": period / prev[0] if prev else 1.0,
                  "duration_ratio": duration * 24.0 / (13.0 * (period / 365.25) ** (1.0 / 3.0)),
                  "red_noise": red, "variability": var})
        out.append((period, t0, f))
        prev = (period, snr)
        keep &= ~((np.abs(phase(time, period, t0)) < duration) | (np.abs(phase(time, period, t0 + 0.5 * period)) < duration))
    return out
