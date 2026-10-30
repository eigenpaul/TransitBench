import time

import numpy as np
import pytest

from transitbench.match import MAX_TRANSITS, Candidate, Signal, compare, coverage_from_times, match_star, transit_times
from transitbench.score import Star, score

YEAR = np.array([[0.0, 370.0]])
PLANET = Signal(period=10.0, t0=3.0, duration=4.0)


def test_transit_times_inside_coverage_only():
    cov = np.array([[0.0, 25.0], [40.0, 60.0]])
    assert transit_times(10.0, 3.0, cov).tolist() == [3.0, 13.0, 23.0, 43.0, 53.0]


def test_coverage_from_times_splits_on_gaps():
    times = np.concatenate([np.arange(0, 10, 0.02), np.arange(15, 20, 0.02)])
    cov = coverage_from_times(times, max_gap=0.1)
    assert cov.shape == (2, 2)
    assert cov[1, 0] == pytest.approx(15.0)


def test_empty_catalogue_scores_zero():
    result = match_star([PLANET], [], YEAR)
    assert result.credit == 0.0 and result.n_false == 0


def test_truth_is_an_exact_match():
    result = match_star([PLANET], [Candidate(10.0, 3.0, 1.0)], YEAR)
    assert result.credit == 1.0 and result.n_false == 0 and result.outcomes == ("exact",)


def test_epoch_one_period_away_is_still_exact():
    assert compare(PLANET, Candidate(10.0, 13.0), YEAR) == 1.0


@pytest.mark.parametrize("factor", [0.5, 2.0, 1 / 3, 3.0])
def test_aliases_earn_half_credit(factor):
    assert compare(PLANET, Candidate(10.0 * factor, 3.0), YEAR) == 0.5


def test_epoch_shifted_by_two_tau_is_no_match():
    tau = (0.5 * PLANET.duration + 0.5) / 24
    assert compare(PLANET, Candidate(10.0, 3.0 + 2 * tau), YEAR) == 0.0


def test_period_off_enough_to_drift_out_is_no_match():
    # 0.5% off: after 37 transits the candidate is 1.8 d away from the truth
    assert compare(PLANET, Candidate(10.05, 3.0), YEAR) == 0.0


def test_hedging_across_periods_is_penalized():
    rows = [Candidate(10.0, 3.0, 0.9), Candidate(5.0, 3.0, 0.8), Candidate(20.0, 3.0, 0.7)]
    result = match_star([PLANET], rows, YEAR)
    assert result.credit == 1.0 and result.n_false == 2
    assert result.outcomes == ("exact", "duplicate", "duplicate")


def test_greedy_by_score_lets_the_alias_win_if_ranked_first():
    rows = [Candidate(5.0, 3.0, 0.9), Candidate(10.0, 3.0, 0.1)]
    result = match_star([PLANET], rows, YEAR)
    assert result.credit == 0.5 and result.n_false == 1


def test_unrelated_candidate_is_false():
    result = match_star([PLANET], [Candidate(7.3, 1.1, 1.0)], YEAR)
    assert result.credit == 0.0 and result.n_false == 1 and result.outcomes == ("false",)


def test_neutral_signal_is_ignored():
    known = Signal(period=7.3, t0=1.1, duration=3.0, scored=False)
    result = match_star([PLANET, known], [Candidate(7.3, 1.1, 1.0)], YEAR)
    assert result.credit == 0.0 and result.n_false == 0 and result.outcomes == ("neutral",)


def test_two_planets_each_claimed_once():
    other = Signal(period=31.0, t0=20.0, duration=6.0)
    rows = [Candidate(31.0, 20.0, 0.5), Candidate(10.0, 3.0, 0.4)]
    result = match_star([PLANET, other], rows, YEAR)
    assert result.credit == 2.0 and result.n_false == 0 and result.claimed == (True, True)


def test_candidate_with_no_transit_in_data_is_false():
    cov = np.array([[0.0, 4.0]])
    result = match_star([Signal(2.0, 1.0, 2.0)], [Candidate(50.0, 10.0, 1.0)], cov)
    assert result.n_false == 1


def test_gaps_do_not_break_a_true_match():
    cov = np.array([[0.0, 90.0], [100.0, 180.0], [190.0, 270.0], [280.0, 370.0]])
    assert compare(PLANET, Candidate(10.0, 3.0), cov) == 1.0


def test_partner_in_a_gap_still_counts():
    # Review F1: the candidate's matching transit at 210.05 falls in a gap the truth's 210.0 does not.
    cov = np.array([[5.0, 210.03], [211.0, 300.0]])
    result = match_star([Signal(100.0, 10.0, 4.0)], [Candidate(100.0, 10.05, 1.0)], cov)
    assert result.credit == 1.0 and result.n_false == 0


def test_double_period_skipping_gaps_is_an_alias_not_exact():
    # Review F6: every other true transit falls in a gap, so 2P hits all observed ones.
    cov = np.array([[0.0, 6.0], [18.0, 26.0], [38.0, 46.0], [58.0, 66.0]])
    assert compare(Signal(10.0, 3.0, 4.0), Candidate(20.0, 3.0), cov) == 0.5


@pytest.mark.parametrize("period, t0", [(0.0, 3.0), (-5.0, 3.0), (float("nan"), 3.0), (10.0, float("inf"))])
def test_degenerate_candidates_never_match_and_never_crash(period, t0):
    assert compare(PLANET, Candidate(period, t0), YEAR) == 0.0


def test_quadruple_period_is_not_an_alias():
    assert compare(PLANET, Candidate(40.0, 3.0), YEAR) == 0.0
    assert compare(PLANET, Candidate(2.5, 3.0), YEAR) == 0.0


def test_guard_limit():
    assert MAX_TRANSITS == 100_000
    assert transit_times(370.0 / (MAX_TRANSITS - 1), 0.0, YEAR).size == MAX_TRANSITS
    assert transit_times(370.0 / (MAX_TRANSITS + 1), 0.0, YEAR).size == 0


@pytest.mark.parametrize("period, t0", [(1e-9, 0.0), (5e-324, 0.0), (1e-300, 1e300)])
def test_guard_returns_no_transits_for_absurd_ephemerides(period, t0):
    assert transit_times(period, t0, YEAR).size == 0


def test_absurd_row_is_one_false_candidate_and_fast():
    # Without the guard, the row 1e-9,0,1 would allocate about 3.7e11 transit times over a year of data.
    star = Star("p1", "inj", YEAR, [PLANET])
    start = time.perf_counter()
    result = match_star([PLANET], [Candidate(1e-9, 0.0, 1.0)], YEAR)
    out = score([star], {"p1": [Candidate(1e-9, 0.0, 1.0)]})
    assert time.perf_counter() - start < 0.05
    assert result.n_false == 1 and result.credit == 0.0 and result.outcomes == ("false",)
    assert out["false"] == 1 and out["S"] == -1.0


def test_ragged_tables_are_rejected(tmp_path):
    from transitbench.data import read_csv
    path = tmp_path / "t.csv"
    path.write_text("a,b,c\n1,2,3\n4,5\n")
    with pytest.raises(ValueError, match="fields"):
        read_csv(path)
