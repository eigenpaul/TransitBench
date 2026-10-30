"""Reference vetter for the M1 kill test: the starter's BLS search plus standard vetting, up to three signals.

Written by Claude on 2026-10-26 as a quick classical reference. It is not Paul's hand-vetted baseline (plan section
11, baseline 3), which stays his to write. Same input and output contract as the starter.

For each BLS peak above SNR 7.1 (up to three, masking each signal before searching again) it rejects:
- fewer than 3 transits with data;
- one transit carrying more than half the signal (a glitch);
- odd and even transit depths that differ by more than 3 sigma (an eclipsing binary at twice the period);
- a dip at phase 0.5 deeper than 3 sigma (a secondary eclipse);
- a depth above 3% (too deep for a planet around a dwarf);
- a period within 2% of one already handled, or of its double or half (the same signal found again).
"""

import io
import sys

import numpy as np
from astropy.timeseries import BoxLeastSquares

P_MIN, P_MAX = 0.5, 100.0
DURATIONS_HOURS = (1.0, 2.5, 5.0, 10.0)
SNR_MIN = 7.1
MAX_SIGNALS = 3


def detrend(time, flux, window=1.0):
    step = window / 4.0
    centres = np.arange(time.min(), time.max() + step, step)
    lo = np.searchsorted(time, centres - window / 2)
    hi = np.searchsorted(time, centres + window / 2)
    meds = np.array([np.median(flux[a:b]) if b > a else np.nan for a, b in zip(lo, hi)])
    good = np.isfinite(meds)
    return flux / np.interp(time, centres[good], meds[good])


def period_grid(baseline):
    periods = [P_MIN]
    while periods[-1] < P_MAX:
        p = periods[-1]
        duration = max(13.0 / 24.0 * (p / 365.25) ** (1.0 / 3.0), 1.0 / 24.0)
        periods.append(p * (1.0 + duration / (3.0 * baseline)))
    return np.array(periods)


def search(time, flux, scatter):
    bls = BoxLeastSquares(time, flux, dy=np.full(time.size, scatter))
    result = bls.power(period_grid(time.max() - time.min()), np.array(DURATIONS_HOURS) / 24.0, objective="snr")
    i = int(np.argmax(result.power))
    return float(result.period[i]), float(result.transit_time[i]), float(result.duration[i]), float(result.depth_snr[i])


def phase(time, period, t0):
    return (time - t0 + 0.5 * period) % period - 0.5 * period


def event_depths(time, flux, period, t0, duration, scatter):
    """Mean drop and its error for each transit with at least 3 in-transit cadences."""
    epochs = np.round((time - t0) / period)
    inside = np.abs(phase(time, period, t0)) < duration / 2
    depths, errors, numbers = [], [], []
    for n in np.unique(epochs[inside]):
        sel = inside & (epochs == n)
        if sel.sum() >= 3:
            depths.append(1.0 - flux[sel].mean())
            errors.append(scatter / np.sqrt(sel.sum()))
            numbers.append(int(n))
    return np.array(depths), np.array(errors), np.array(numbers)


def mean_depth(depths, errors):
    w = 1.0 / errors**2
    return float(np.sum(w * depths) / np.sum(w)), float(1.0 / np.sqrt(np.sum(w)))


def vet(time, flux, period, t0, duration, scatter):
    depths, errors, numbers = event_depths(time, flux, period, t0, duration, scatter)
    if depths.size < 3:
        return False
    snr2 = (depths / errors) ** 2
    if snr2.max() > 0.5 * max(snr2.sum(), 1e-12):
        return False
    odd, even = numbers % 2 == 1, numbers % 2 == 0
    if odd.any() and even.any():
        d_odd, e_odd = mean_depth(depths[odd], errors[odd])
        d_even, e_even = mean_depth(depths[even], errors[even])
        if abs(d_odd - d_even) > 3.0 * np.hypot(e_odd, e_even):
            return False
    d_all, _ = mean_depth(depths, errors)
    if d_all > 0.03:
        return False
    sec = np.abs(phase(time, period, t0 + 0.5 * period)) < duration / 2
    if sec.sum() >= 3:
        d_sec = 1.0 - flux[sec].mean()
        if d_sec > 3.0 * scatter / np.sqrt(sec.sum()):
            return False
    return True


def main():
    star = np.load(io.BytesIO(sys.stdin.buffer.read()))
    order = np.argsort(star["time"])
    time = np.asarray(star["time"], float)[order]
    flux = detrend(time, np.asarray(star["flux"], float)[order])
    scatter = 1.4826 * np.median(np.abs(flux - np.median(flux)))
    keep = np.ones(time.size, bool)
    seen = []
    for _ in range(MAX_SIGNALS):
        if keep.sum() < 100:
            break
        period, t0, duration, snr = search(time[keep], flux[keep], scatter)
        if snr < SNR_MIN:
            break
        repeat = any(abs(period / (p * r) - 1.0) < 0.02 for p in seen for r in (0.5, 1.0, 2.0))
        seen.append(period)
        if not repeat and vet(time[keep], flux[keep], period, t0, duration, scatter):
            print(f"{period:.7f},{t0:.6f},{snr:.3f}")
        primary = np.abs(phase(time, period, t0)) < duration
        secondary = np.abs(phase(time, period, t0 + 0.5 * period)) < duration
        keep &= ~(primary | secondary)


if __name__ == "__main__":
    main()
