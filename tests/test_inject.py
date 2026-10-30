import math

import numpy as np
import pytest

from transitbench.inject import (
    CADENCE_DAYS,
    duration_hours,
    mes,
    robust_scatter,
    transit_count,
    transit_drop,
)

TIME = np.arange(0.0, 366.0, CADENCE_DAYS)


def test_no_drop_outside_transit():
    drop = transit_drop(TIME, period=10.0, t0=5.0, duration_hours=4.0, ror=0.05, b=0.3)
    far = np.abs(((TIME - 5.0 + 5.0) % 10.0) - 5.0) > 0.2
    assert np.all(drop[far] == 0.0)


def test_central_depth_is_ror_squared_without_limb_darkening():
    drop = transit_drop(np.array([5.0]), period=10.0, t0=5.0, duration_hours=4.0, ror=0.05, b=0.0)
    assert drop[0] == pytest.approx(0.05**2, rel=0.02)


def test_limb_darkening_deepens_the_centre():
    ld = (0.4, 0.2, 0.1, -0.05)
    plain = transit_drop(np.array([5.0]), 10.0, 5.0, 4.0, 0.05, 0.0)[0]
    darkened = transit_drop(np.array([5.0]), 10.0, 5.0, 4.0, 0.05, 0.0, ld)[0]
    assert darkened > plain


def test_earth_sun_duration_is_about_13_hours():
    assert duration_hours(365.25, 0.0092, 0.0, 1.0, 1.0) == pytest.approx(13.0, abs=0.3)


def test_eccentricity_shortens_transit_at_periastron():
    circ = duration_hours(10.0, 0.03, 0.2, 1.0, 1.0)
    ecc = duration_hours(10.0, 0.03, 0.2, 1.0, 1.0, e=0.3, omega=math.pi / 2)
    assert ecc < circ


def test_transit_count_skips_gaps():
    gap = TIME[(TIME < 100) | (TIME > 200)]
    full = transit_count(TIME, 10.0, 5.0, 4.0)
    assert full == 37
    assert transit_count(gap, 10.0, 5.0, 4.0) == 27


def test_robust_scatter_matches_white_noise():
    rng = np.random.default_rng(1)
    flux = 1.0 + rng.normal(0, 1e-3, TIME.size)
    per_bin = 4.0 / 24 / CADENCE_DAYS
    assert robust_scatter(TIME, flux, 4.0) == pytest.approx(1e-3 / math.sqrt(per_bin), rel=0.15)


def test_mes_scales_with_depth_and_transit_count():
    rng = np.random.default_rng(2)
    flux = 1.0 + rng.normal(0, 1e-3, TIME.size)
    drop = transit_drop(TIME, 10.0, 5.0, 4.0, 0.02, 0.0)
    one = mes(TIME, flux, 10.0, 5.0, 4.0, drop)
    two = mes(TIME, flux, 10.0, 5.0, 4.0, 2 * drop)
    assert two == pytest.approx(2 * one, rel=1e-9)
    short = TIME < 92
    assert mes(TIME[short], flux[short], 10.0, 5.0, 4.0, drop[short]) == pytest.approx(one / 2, rel=0.2)


def _noisy(seed=3):
    return 1.0 + np.random.default_rng(seed).normal(0, 3e-4, TIME.size)


def test_planet_shows_no_binary_evidence():
    from transitbench.inject import eb_evidence
    flux = _noisy() * (1 - transit_drop(TIME, 5.0, 2.0, 3.0, 0.03, 0.2))
    assert not any(eb_evidence(TIME, flux, 5.0, 2.0, 3.0).values())


def test_secondary_and_odd_even_are_flagged():
    from transitbench.inject import eb_evidence
    primary = transit_drop(TIME, 5.0, 2.0, 3.0, 0.05, 0.2)
    secondary = 0.3 * transit_drop(TIME, 5.0, 4.5, 3.0, 0.05, 0.2)
    flags = eb_evidence(TIME, _noisy() * (1 - primary - secondary), 5.0, 2.0, 3.0)
    assert flags["secondary"]
    # An EB found at half its true period: alternate eclipses differ in depth.
    deep = transit_drop(TIME, 10.0, 2.0, 3.0, 0.06, 0.1)
    shallow = transit_drop(TIME, 10.0, 7.0, 3.0, 0.04, 0.1)
    flags = eb_evidence(TIME, _noisy() * (1 - deep - shallow), 5.0, 2.0, 3.0)
    assert flags["odd_even"]


def test_twin_comparison_recovers_a_known_depth_ratio():
    from transitbench.inject import twin_comparison
    orig = (TIME, _noisy(4))
    model = transit_drop(TIME, 10.0, 5.0, 4.0, 0.03, 0.2)
    nasa = (TIME, orig[1] * (1 - 0.9 * model))
    out = twin_comparison(nasa, orig, 10.0, 5.0, 4.0, model)
    assert out["depth_ratio_local"] == pytest.approx(0.9, rel=0.02)
    assert out["depth_ratio_raw"] == pytest.approx(0.9, rel=0.02)


def test_twin_comparison_local_ratio_ignores_a_baseline_lift_around_the_transits():
    from transitbench.inject import twin_comparison
    orig = (TIME, _noisy(6))
    model = transit_drop(TIME, 10.0, 5.0, 4.0, 0.03, 0.2)
    phase = (TIME - 5.0 + 5.0) % 10.0 - 5.0
    lift = 0.1 * model.max() * np.exp(-0.5 * (phase / 1.5) ** 2)  # days-wide, as around NASA's transits
    nasa = (TIME, orig[1] * (1 - model) * (1 + lift))
    out = twin_comparison(nasa, orig, 10.0, 5.0, 4.0, model)
    assert out["depth_ratio_local"] == pytest.approx(1.0, rel=0.02)
    assert out["depth_ratio_raw"] < 0.95


def test_duration_refuses_a_missing_host_mass():
    with pytest.raises(ValueError):
        duration_hours(10.0, 0.02, 0.3, math.nan, 1.0)
    with pytest.raises(ValueError):
        duration_hours(10.0, 0.02, 0.3, 1.0, math.nan)


def test_eccentric_secondary_is_found_at_its_own_phase():
    from transitbench.inject import eb_tests
    primary = transit_drop(TIME, 5.0, 2.0, 3.0, 0.05, 0.2)
    secondary = 0.3 * transit_drop(TIME, 5.0, 2.0 + 0.37 * 5.0, 3.0, 0.05, 0.2)
    out = eb_tests(TIME, _noisy() * (1 - primary - secondary), 5.0, 2.0, 3.0)
    assert out["secondary"] and out["secondary_phase"] == pytest.approx(0.37, abs=0.02)
    assert not out["too_deep"]


def test_noise_alone_rarely_fires():
    from transitbench.inject import eb_evidence
    fired = sum(any(eb_evidence(TIME, _noisy(seed), 7.3, 1.1, 4.0).values()) for seed in range(40))
    assert fired <= 2
