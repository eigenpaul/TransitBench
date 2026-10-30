"""Naive transit search: detrend, run BLS on a fixed grid, report the strongest peak if it clears SNR 7.1.

Input: one star as an .npz on stdin with arrays `time` (days), `flux` (normalised to 1 per quarter) and `quarter`.
Output: CSV on stdout, at most 3 rows of `period_days,t0_days,score`, one row per planet you claim. The score ranks
your claims; rows that match no planet cost points, and printing nothing is allowed.

This file is a working starting point, not a good pipeline. It reports one peak per star and never rejects one.
`vet()` is the place to start rejecting eclipsing binaries and glitches; anything else may change too.
"""

import io
import sys

import numpy as np
from astropy.timeseries import BoxLeastSquares

P_MIN, P_MAX = 0.5, 100.0
DURATIONS_HOURS = (1.0, 2.5, 5.0, 10.0)
SNR_MIN = 7.1


def detrend(time, flux, window=1.0):
    """Divide by a running median: medians in window-wide bins every quarter window, interpolated."""
    step = window / 4.0
    centres = np.arange(time.min(), time.max() + step, step)
    lo = np.searchsorted(time, centres - window / 2)
    hi = np.searchsorted(time, centres + window / 2)
    meds = np.array([np.median(flux[a:b]) if b > a else np.nan for a, b in zip(lo, hi)])
    good = np.isfinite(meds)
    return flux / np.interp(time, centres[good], meds[good])


def period_grid(baseline):
    """Log-spaced periods with steps of a third of a Sun-like star's transit duration over the baseline."""
    periods = [P_MIN]
    while periods[-1] < P_MAX:
        p = periods[-1]
        duration = max(13.0 / 24.0 * (p / 365.25) ** (1.0 / 3.0), 1.0 / 24.0)
        periods.append(p * (1.0 + duration / (3.0 * baseline)))
    return np.array(periods)


def search(time, flux):
    """Best BLS peak as (period, t0, duration_days, depth, snr)."""
    scatter = 1.4826 * np.median(np.abs(flux - np.median(flux)))
    bls = BoxLeastSquares(time, flux, dy=np.full(time.size, scatter))
    periods = period_grid(time.max() - time.min())
    result = bls.power(periods, np.array(DURATIONS_HOURS) / 24.0, objective="snr")
    i = int(np.argmax(result.power))
    return (float(result.period[i]), float(result.transit_time[i]), float(result.duration[i]),
            float(result.depth[i]), float(result.depth_snr[i]))


def vet(candidate, time, flux):
    """Return True to keep a candidate. The starter keeps everything."""
    return True


def main():
    star = np.load(io.BytesIO(sys.stdin.buffer.read()))
    time, flux = np.asarray(star["time"], float), np.asarray(star["flux"], float)
    order = np.argsort(time)
    time, flux = time[order], detrend(time[order], flux[order])
    candidate = search(time, flux)
    period, t0, _, _, snr = candidate
    if snr >= SNR_MIN and vet(candidate, time, flux):
        print(f"{period:.7f},{t0:.6f},{snr:.3f}")


if __name__ == "__main__":
    main()
