"""The official scorer: gold oracle, hedging, aliases, reward, truth rules and the report."""

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from transitbench.match import Candidate, Signal
from transitbench.score import (
    Star,
    TruthSignal,
    catalogue_from_runs,
    clopper_pearson,
    load_preds,
    load_truth,
    main,
    report,
    reward,
    score,
)

QUARTERS = [[0.0, 90.0], [92.0, 180.0], [182.0, 270.0], [272.0, 370.0]]
YEAR = np.array(QUARTERS)
NO_EVIDENCE = {"secondary": False, "odd_even": False, "too_deep": False}


def signal(period, t0, duration, mes, kind="injected", scored=True, **extra):
    return {"period": period, "t0": t0, "duration": duration, "scored": scored, "kind": kind, "mes": mes, **extra}


def entry(population, host, signals=(), scored=True, **extra):
    return {"population": population, "host": host, "scored": scored, "coverage": QUARTERS, "signals": list(signals),
            **extra}


def truth_doc():
    """A tiny sample with every population and host, plus extra keys the scorer must ignore."""
    eb = {"period": 2.2, "t0": 0.7, "duration": 3.0, "source": "kirk2016"}
    return {"release": "test", "sample": "synthetic", "window": [10, 11, 12, 13], "build": {"seed_sha256": "x"},
            "stars": {
                "inj-r": entry("inj", "random", [signal(10.0, 3.0, 4.0, 12.5, depth_ppm=310.0, b=0.3)], kic_hash="y"),
                "inj-t": entry("inj", "tce_only", [signal(3.7, 1.2, 2.5, 8.2)]),
                "null-r": entry("null", "random"),
                "null-t": entry("null", "tce_only"),
                "eb-f": entry("eb", "eb", eb={**eb, "evidence": {**NO_EVIDENCE, "secondary": True}, "quiet": True,
                                              "variability": 0.0012}),
                "eb-u": entry("eb", "eb", scored=False, eb={**eb, "evidence": NO_EVIDENCE, "quiet": False,
                                                           "variability": 0.03}),
                "real": entry("real", "real", scored=False, signals=[
                    signal(7.0, 2.0, 3.0, 9.0, kind="confirmed"),
                    signal(23.1, 5.5, 5.0, None, kind="koi", scored=False),
                    signal(41.0, 11.0, 6.0, 5.0, kind="confirmed", scored=False)]),
            }}


def load(tmp_path, doc=None):
    path = tmp_path / "truth.json"
    path.write_text(json.dumps(doc or truth_doc()))
    return load_truth(path)


def only(stars, *ids):
    return [s for s in stars if s.star_id in ids]


def inj(sid, mes, period=10.0, t0=3.0, host="random"):
    return Star(sid, "inj", YEAR, [TruthSignal(period, t0, 4.0, True, "injected", mes)], True, host)


def null(sid, host="random"):
    return Star(sid, "null", YEAR, [], True, host)


def real(sid, *planets):
    signals = [TruthSignal(p, 1.0 + i, 4.0, True, "confirmed", mes) for i, (p, mes) in enumerate(planets)]
    return Star(sid, "real", YEAR, signals, False, "real")


def oracle_runs(doc) -> list[dict]:
    """One runner record per truth star with one row per scored signal, built from the raw truth file."""
    return [{"star": sid, "status": "ok", "cpu_s": 0.0,
             "rows": [[s["period"], s["t0"], 1.0] for s in t["signals"] if s["scored"]]}
            for sid, t in doc["stars"].items()]


def assert_gold(truth_path, tmp_path):
    """Gold scorer test: the truth oracle scores exactly 1 through the CLI, and every other report field agrees with
    a perfect catalogue."""
    doc = json.loads(truth_path.read_text())
    runs = oracle_runs(doc)
    preds, out = tmp_path / "oracle.jsonl", tmp_path / "oracle_report.json"
    preds.write_text("".join(json.dumps(r) + "\n" for r in runs))
    main(["--truth", str(truth_path), "--preds", str(preds), "--out", str(out)])
    rep = json.loads(out.read_text())

    stars = load_truth(truth_path)
    catalogue, _ = catalogue_from_runs({r["star"]: r for r in runs})
    h = rep["headline"]
    assert h == score(stars, catalogue)
    planets = sum(s["scored"] for t in doc["stars"].values() if t["scored"] for s in t["signals"])
    assert (h["S"], h["C"], h["false"], h["F_per_100"], h["eb_as_planet"]) == (1.0, 1.0, 0, 0.0, 0.0)
    assert h["planets"] == planets and h["scored_stars"] == sum(t["scored"] for t in doc["stars"].values())
    for group in [g for pop in h["by_host"].values() for g in pop.values()]:
        assert group["false"] == 0 and group["credit"] == group["planets"]
    assert reward(stars, catalogue) == min(1.0, planets / 8)

    assert all(b["C"] == 1.0 for b in rep["completeness"]["by_mes"] if b["n"])
    assert rep["completeness"]["mes50"] is None
    eb = rep["eb_as_planet"]
    slices = [eb[k] for k in ("flagged", "four_year_label", "flagged_quiet", "flagged_variable") if eb[k]]
    for rate in [*rep["null_false_alarms"].values(), *slices]:
        assert rate["k"] == 0 and rate["ci95"] == list(clopper_pearson(0, rate["n"]))
    assert all(v in (0.0, None) for v in eb["abstention_check"].values())
    sweep = rep["sweep"]
    assert (sweep["C_at_F_limit"], sweep["S_at_F_limit"], sweep["F_per_100_at_F_limit"]) == (1.0, 1.0, 0.0)
    assert sweep["curve"][-1] == {"threshold": 1.0, "C": 1.0, "F_per_100": 0.0}

    per_host = [sum(s["scored"] for s in t["signals"]) for t in doc["stars"].values() if t["population"] == "real"]
    r = rep["real"]
    assert r["rows"]["unknown"] == r["rows"]["neutral"] == r["rows"]["duplicate"] == 0
    assert r["rows"]["matched"] == sum(min(n, 3) for n in per_host)
    if sum(per_host):
        assert r["recall"] == sum(min(n, 3) for n in per_host) / sum(per_host)
        assert r["predicted_from_inj"] == 1.0 and r["transfer_gap"] == r["recall"] - 1.0
    i = rep["inputs"]
    assert (i["missing_stars"], i["unknown_stars"], i["rows_invalid"], i["not_ok"]) == (0, 0, 0, {})


def test_truth_oracle_scores_one(tmp_path):
    path = tmp_path / "truth.json"
    path.write_text(json.dumps(truth_doc()))
    assert_gold(path, tmp_path)


def test_empty_catalogue_scores_zero(tmp_path):
    stars = load(tmp_path)
    h = score(stars, {})
    assert (h["S"], h["C"], h["F_per_100"], h["eb_as_planet"]) == (0.0, 0.0, 0.0, 0.0)
    rep = report(stars, {})
    assert rep["sweep"]["C_at_F_limit"] == 0.0 and rep["sweep"]["threshold"] is None
    assert rep["inputs"]["missing_stars"] == len(stars) and reward(stars, {}) == 0.0


@pytest.mark.parametrize("extra", [
    [[5.0, 3.0, 0.5]],
    [[20.0, 3.0, 0.5]],
    [[5.0, 3.0, 0.5], [20.0, 3.0, 0.4]],
    [[20.0, 3.0, 0.95]],
    [[13.3, 4.4, 0.5]],
    [[10.0, 3.0, 0.5]],
], ids=["half-period", "double-period", "both-aliases", "alias-ranked-first", "other-period", "same-row-twice"])
def test_hedging_is_penalized(tmp_path, extra):
    stars = load(tmp_path)
    alone, _ = catalogue_from_runs({"inj-r": {"rows": [[10.0, 3.0, 0.9]]}})
    hedged, _ = catalogue_from_runs({"inj-r": {"rows": [[10.0, 3.0, 0.9], *extra]}})
    assert score(stars, alone)["S"] == 0.5
    assert score(stars, hedged)["S"] < score(stars, alone)["S"]
    assert reward(stars, hedged) < reward(stars, alone)


def test_hedging_between_aliases_scores_below_one_alias(tmp_path):
    stars = only(load(tmp_path), "inj-r")
    one = score(stars, {"inj-r": [Candidate(20.0, 3.0, 0.9)]})
    two = score(stars, {"inj-r": [Candidate(20.0, 3.0, 0.9), Candidate(5.0, 3.0, 0.8)]})
    assert one["S"] == 0.5 and two["S"] == -0.5


@pytest.mark.parametrize("factor", [0.5, 2.0, 1 / 3, 3.0])
def test_alias_earns_half(tmp_path, factor):
    stars = only(load(tmp_path), "inj-r")
    h = score(stars, {"inj-r": [Candidate(10.0 * factor, 3.0, 1.0)]})
    assert (h["credit"], h["false"], h["C"], h["S"]) == (0.5, 0, 0.5, 0.5)


def test_reward_normalises_and_clips():
    stars = [inj(f"i{k}", 12.0) for k in range(12)] + [null(f"n{k}") for k in range(5)]
    found = {f"i{k}": [Candidate(10.0, 3.0, 1.0)] for k in range(12)}
    three = {k: found[k] for k in ("i0", "i1", "i2")}
    assert reward(stars, three) == 3 / 8
    assert reward(stars, {**three, "n0": [Candidate(7.7, 1.0, 1.0)]}) == 2 / 8
    assert reward(stars, three, norm=4.0) == 0.75
    assert reward(stars, found) == 1.0
    spam = {f"n{k}": [Candidate(1.1 + j, 0.3, 1.0) for j in range(3)] for k in range(5)}
    assert reward(stars, spam) == -1.0
    h = score(stars, {**three, "n0": [Candidate(7.7, 1.0, 1.0)]})
    assert reward(stars, {**three, "n0": [Candidate(7.7, 1.0, 1.0)]}) == pytest.approx(h["S"] * h["planets"] / 8)


def test_reward_ignores_unscored_stars():
    stars = [inj("i", 12.0), real("r", (7.0, 9.0))]
    cat = {"i": [Candidate(10.0, 3.0, 1.0)]}
    assert reward(stars, {**cat, "r": [Candidate(2.3, 0.1, 1.0)] * 3}) == reward(stars, cat) == 1 / 8


def test_crash_or_missing_star_means_no_rows(tmp_path):
    stars = load(tmp_path)
    runs = {"inj-r": {"star": "inj-r", "status": "timeout", "cpu_s": 20.0, "rows": [[10.0, 3.0, 1.0]]},
            "inj-t": {"star": "inj-t", "status": "exit 1", "cpu_s": 1.0, "rows": [[3.7, 1.2, 1.0]]},
            "null-r": {"star": "null-r", "status": "ok", "cpu_s": 1.0, "rows": []}}
    catalogue, inputs = catalogue_from_runs(runs)
    assert catalogue == {"inj-r": [], "inj-t": [], "null-r": []}
    assert inputs == {"runs": 3, "not_ok": {"timeout": 1, "exit 1": 1}, "rows_invalid": 0}
    assert score(stars, catalogue) == score(stars, {}) and reward(stars, catalogue) == 0.0
    assert report(stars, catalogue)["inputs"]["missing_stars"] == len(stars) - 3


def test_invalid_rows_are_dropped_and_counted():
    rows = [[10.0, 3.0, 1.0], [math.nan, 1.0, 1.0], [-2.0, 1.0, 1.0], [0, 1, 1], [1.0, 2.0], ["5", 1, 1],
            [True, 1, 1], [5.0, 1.0, math.inf], "10,3,1", Candidate(4.0, 1.0, 0.5)]
    catalogue, inputs = catalogue_from_runs({"s": {"status": "ok", "rows": rows}})
    assert catalogue == {"s": [Candidate(10.0, 3.0, 1.0), Candidate(4.0, 1.0, 0.5)]}
    assert inputs["rows_invalid"] == 8
    stars = [inj("s", 12.0)]
    raw = [Candidate(math.nan, 3.0, 1.0), Candidate(-1.0, 3.0, 1.0), Candidate(10.0, 3.0, math.inf)]
    assert score(stars, {"s": raw})["false"] == 0 and reward(stars, {"s": raw}) == 0.0


def test_rows_over_the_cap_are_cut_and_counted(tmp_path):
    stars = load(tmp_path)
    rows = [Candidate(1.1 + j, 0.3, 1.0) for j in range(5)]
    rep = report(stars, {"null-r": rows})
    assert rep["headline"]["false"] == 3 and rep["inputs"]["rows_over_cap"] == 2


def test_duplicate_star_in_preds_is_rejected(tmp_path):
    path = tmp_path / "preds.jsonl"
    path.write_text('{"star": "a", "status": "ok", "rows": []}\n{"star": "a", "status": "ok", "rows": []}\n')
    with pytest.raises(ValueError, match="twice"):
        load_preds(path)


def test_real_stays_outside_s(tmp_path):
    stars = load(tmp_path)
    base = {"inj-r": [Candidate(10.0, 3.0, 1.0)]}
    with_real = {**base, "real": [Candidate(7.0, 2.0, 0.9), Candidate(13.3, 4.4, 0.8), Candidate(2.9, 0.4, 0.7)]}
    h0, h1 = score(stars, base), score(stars, with_real)
    for key in ("S", "C", "F_per_100", "planets", "scored_stars", "credit", "false", "by_host"):
        assert h0[key] == h1[key]
    assert reward(stars, base) == reward(stars, with_real)
    r = report(stars, with_real)["real"]
    assert r["recall"] == 1.0 and r["planets"] == 1
    assert r["rows"] == {"total": 3, "matched": 1, "neutral": 0, "duplicate": 0, "unknown": 2,
                         "unknown_per_100_stars": 200.0}


def test_unflagged_eb_is_neutral_and_flagged_eb_counts(tmp_path):
    stars = load(tmp_path)
    on_unflagged = {"eb-u": [Candidate(2.2, 0.7, 1.0)]}
    h = score(stars, on_unflagged)
    assert (h["S"], h["false"], h["eb_as_planet"]) == (0.0, 0, 0.0)
    eb = report(stars, on_unflagged)["eb_as_planet"]
    assert (eb["four_year_label"]["k"], eb["four_year_label"]["n"], eb["flagged"]["k"]) == (1, 1, 0)
    h = score(stars, {"eb-f": [Candidate(2.2, 0.7, 1.0)]})
    assert (h["S"], h["false"], h["eb_as_planet"]) == (-0.5, 1, 1.0)


def with_variable_eb(doc):
    eb = {**doc["stars"]["eb-f"]["eb"], "quiet": False, "variability": 0.05}
    doc["stars"]["eb-v"] = entry("eb", "eb", eb=eb)
    return doc


def test_eb_rates_split_by_quiet_hosts_with_abstention_check(tmp_path):
    stars = load(tmp_path, with_variable_eb(truth_doc()))
    cat = {"eb-v": [Candidate(2.2, 0.7, 1.0)], "null-t": [Candidate(9.9, 1.0, 1.0)]}
    eb = report(stars, cat)["eb_as_planet"]
    assert (eb["flagged"]["k"], eb["flagged"]["n"]) == (1, 2)
    assert eb["flagged_quiet"] == {"k": 0, "n": 1, "rate": 0.0, "ci95": list(clopper_pearson(0, 1))}
    assert eb["flagged_variable"] == {"k": 1, "n": 1, "rate": 1.0, "ci95": list(clopper_pearson(1, 1))}
    assert eb["abstention_check"] == {"null_rate": 0.5, "flagged_minus_null": 0.0, "flagged_quiet_minus_null": -0.5,
                                      "flagged_variable_minus_null": 0.5}
    assert score(stars, cat)["eb_as_planet"] == 0.5
    no_null = report([s for s in stars if s.population != "null"], cat)["eb_as_planet"]["abstention_check"]
    assert set(no_null.values()) == {None}


def drop_quiet(*ids):
    def mutate(doc):
        for sid in ids:
            doc["stars"][sid]["eb"].pop("quiet")
            doc["stars"][sid]["eb"].pop("variability")
        return doc
    return mutate


@pytest.mark.parametrize("mutate", [drop_quiet("eb-f", "eb-u", "eb-v"), drop_quiet("eb-v")],
                         ids=["older-truth", "one-flagged-eb-without-quiet"])
def test_quiet_split_is_unavailable_without_the_flag(tmp_path, mutate):
    stars = load(tmp_path, mutate(with_variable_eb(truth_doc())))
    cat = {"eb-v": [Candidate(2.2, 0.7, 1.0)], "null-t": [Candidate(9.9, 1.0, 1.0)]}
    eb = report(stars, cat)["eb_as_planet"]
    assert eb["flagged_quiet"] is None and eb["flagged_variable"] is None
    assert (eb["flagged"]["k"], eb["flagged"]["n"]) == (1, 2)
    assert eb["abstention_check"] == {"null_rate": 0.5, "flagged_minus_null": 0.0, "flagged_quiet_minus_null": None,
                                      "flagged_variable_minus_null": None}


def test_neutral_signals_are_ignored(tmp_path):
    stars = load(tmp_path)
    on_neutral = {"real": [Candidate(23.1, 5.5, 1.0), Candidate(41.0, 11.0, 0.9)]}
    r = report(stars, on_neutral)["real"]
    assert (r["rows"]["neutral"], r["rows"]["unknown"], r["recall"]) == (2, 0, 0.0)
    tce = TruthSignal(7.3, 1.1, 3.0, False, "tce", None)
    star = Star("x", "inj", YEAR, [TruthSignal(10.0, 3.0, 4.0, True, "injected", 12.0), tce], True, "random")
    h = score([star], {"x": [Candidate(7.3, 1.1, 1.0)]})
    assert (h["S"], h["false"], h["planets"]) == (0.0, 0, 1)


def test_per_host_breakdown(tmp_path):
    stars = load(tmp_path)
    cat = {"inj-r": [Candidate(10.0, 3.0, 1.0)], "inj-t": [Candidate(6.1, 0.2, 1.0)],
           "null-t": [Candidate(9.9, 1.0, 1.0)]}
    rep = report(stars, cat)
    by = rep["headline"]["by_host"]
    assert (by["inj"]["random"]["C"], by["inj"]["random"]["false"]) == (1.0, 0)
    assert (by["inj"]["tce_only"]["C"], by["inj"]["tce_only"]["F_per_100"]) == (0.0, 100.0)
    assert (by["null"]["random"]["stars_with_rows"], by["null"]["tce_only"]["stars_with_rows"]) == (0, 1)
    far = rep["null_false_alarms"]
    assert (far["all"]["k"], far["all"]["n"], far["random"]["k"], far["tce_only"]["k"]) == (1, 2, 0, 1)
    assert far["tce_only"]["ci95"] == pytest.approx([0.025, 1.0])


@pytest.mark.parametrize("k, n, lo, hi", [
    (0, 10, 0.0, 0.3084971078187608),
    (5, 10, 0.18708602844739855, 0.8129139715526015),
    (10, 10, 0.6915028921812392, 1.0),
    (3, 350, 0.0017711347806370586, 0.02484371110923991),
    (17, 175, 0.0576134728355015, 0.1509768234794437),
])
def test_clopper_pearson_known_values(k, n, lo, hi):
    # references: scipy.stats.beta.ppf(0.025, k, n - k + 1) and beta.ppf(0.975, k + 1, n - k)
    assert clopper_pearson(k, n) == pytest.approx((lo, hi), abs=1e-12)


def test_clopper_pearson_edges():
    assert clopper_pearson(0, 10)[1] == pytest.approx(1 - 0.025 ** (1 / 10), abs=1e-12)
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        clopper_pearson(4, 3)


@pytest.mark.parametrize("tie", [False, True])
def test_score_sweep_on_a_hand_built_case(tie):
    stars = [inj(f"i{k}", 12.0) for k in range(4)] + [null(f"n{k}") for k in range(96)]
    cat = {"i0": [Candidate(10.0, 3.0, 0.9)],
           "i1": [Candidate(10.0, 3.0, 0.5), Candidate(20.0, 3.0, 0.3)],
           "i2": [Candidate(10.0, 3.0, 0.2)],
           "n0": [Candidate(7.7, 1.0, 0.8)],
           "n1": [Candidate(7.7, 1.0, 0.4)]}
    if tie:
        cat["n2"] = [Candidate(7.7, 1.0, 0.5)]
    sweep = report(stars, cat)["sweep"]
    curve = [(p["threshold"], p["C"], p["F_per_100"]) for p in sweep["curve"]]
    extra = 1.0 if tie else 0.0
    assert curve == [(None, 0.0, 0.0), (0.9, 0.25, 0.0), (0.8, 0.25, 1.0), (0.5, 0.5, 1.0 + extra),
                     (0.4, 0.5, 2.0 + extra), (0.3, 0.5, 3.0 + extra), (0.2, 0.75, 3.0 + extra)]
    best = (sweep["threshold"], sweep["C_at_F_limit"], sweep["F_per_100_at_F_limit"], sweep["S_at_F_limit"])
    assert best == ((0.9, 0.25, 0.0, 0.25) if tie else (0.5, 0.5, 1.0, 0.25))
    h = score(stars, cat)
    assert (h["C"], h["F_per_100"]) == curve[-1][1:]


def mes_case():
    """INJ completeness 0, 0.75, 1, 1 by bin (mean MES 5.5, 8.5, 12, 19) and one planet above the bins."""
    found = {5.0: 0, 6.0: 0, 7.5: 1, 8.0: 1, 9.0: 0, 9.5: 1, 12.0: 1, 18.0: 1, 20.0: 1, 25.0: 1}
    stars = [inj(f"i{m}", m) for m in found]
    cat = {f"i{m}": [Candidate(10.0, 3.0, 1.0)] for m, hit in found.items() if hit}
    return stars, cat


def test_completeness_by_mes_and_mes50():
    stars, cat = mes_case()
    comp = report(stars, cat)["completeness"]
    assert [(b["mes"], b["n"], b["mean_mes"], b["C"]) for b in comp["by_mes"]] == [
        ("4-7.1", 2, 5.5, 0.0), ("7.1-10", 4, 8.5, 0.75), ("10-15", 1, 12.0, 1.0), ("15-20", 2, 19.0, 1.0)]
    assert comp["mes50"] == pytest.approx(5.5 + 0.5 * 3.0 / 0.75) and comp["unbinned"] == 1


def test_real_recall_by_mes_and_transfer_gap():
    stars, cat = mes_case()
    stars += [real("r1", (10.0, 8.5)), real("r2", (10.0, 30.0)), real("r3", (10.0, 10.25))]
    cat |= {"r1": [Candidate(10.0, 1.0, 1.0)], "r3": [Candidate(20.0, 1.0, 1.0)]}
    r = report(stars, cat)["real"]
    assert [(b["mes"], b["n"], b["C"]) for b in r["recall_by_mes"]] == [
        ("7.1-10", 1, 1.0), ("10-15", 1, 0.5), ("15-20", 0, None), ("20+", 1, 0.0)]
    assert r["recall"] == 0.5
    assert r["predicted_from_inj"] == pytest.approx((0.75 + 1.0 + 0.875) / 3)
    assert r["transfer_gap"] == pytest.approx(0.5 - 0.875)


def test_credit_is_kept_per_signal_on_a_multi_planet_host():
    star = real("r", (10.0, 9.0), (31.0, 16.0))
    cat = {"r": [Candidate(62.0, 2.0, 0.9), Candidate(10.0, 1.0, 0.5)]}
    r = report([star], cat)["real"]
    assert [(b["mes"], b["C"]) for b in r["recall_by_mes"] if b["n"]] == [("7.1-10", 1.0), ("15-20", 0.5)]
    assert r["recall"] == 0.75 and r["rows"]["matched"] == 2


def test_load_truth_round_trip(tmp_path):
    doc = truth_doc()
    doc["stars"]["null-r"]["coverage"] = QUARTERS[2:] + QUARTERS[:2]
    stars = load(tmp_path, doc)
    assert [s.star_id for s in stars] == list(doc["stars"])
    for star in stars:
        t = doc["stars"][star.star_id]
        assert (star.population, star.host, star.scored) == (t["population"], t["host"], t["scored"])
        assert star.coverage.tolist() == QUARTERS
        assert star.quiet == t.get("eb", {}).get("quiet")
        assert [(s.period, s.t0, s.duration, s.scored, s.kind, s.mes) for s in star.signals] == [
            (s["period"], s["t0"], s["duration"], s["scored"], s["kind"], s["mes"]) for s in t["signals"]]


def set_star(sid, **fields):
    return lambda doc: doc["stars"][sid].update(fields)


def set_eb(sid, **fields):
    return lambda doc: doc["stars"][sid]["eb"].update(fields)


@pytest.mark.parametrize("mutate", [
    set_star("null-r", scored=False),
    set_star("inj-r", scored=False),
    set_star("real", scored=True),
    set_star("eb-f", scored=False),
    set_star("eb-u", scored=True),
    set_star("eb-f", eb={"period": 2.2, "t0": 0.7, "duration": 3.0}),
    set_star("null-t", signals=[signal(5.0, 1.0, 3.0, None, kind="tce", scored=False)]),
    set_star("eb-f", signals=[signal(2.2, 0.7, 3.0, None, kind="koi", scored=False)]),
    set_star("inj-r", signals=[]),
    set_star("inj-r", signals=[signal(10.0, 3.0, 4.0, 12.5, scored=False)]),
    set_star("inj-r", signals=[signal(10.0, 3.0, 4.0, 12.5, kind="planet")]),
    set_star("inj-r", signals=[signal(-10.0, 3.0, 4.0, 12.5)]),
    set_star("real", signals=[signal(7.0, 2.0, 3.0, 9.0, kind="confirmed", scored=False)]),
    set_star("real", signals=[signal(7.0, 2.0, 3.0, 6.0, kind="confirmed")]),
    set_star("real", signals=[signal(7.0, 2.0, 3.0, 9.0, kind="koi")]),
    set_star("inj-r", host="eb"),
    set_star("inj-r", population="twin"),
    set_star("null-r", coverage=[]),
    set_star("real", signals=[signal(150.0, 2.0, 3.0, 9.0, kind="confirmed")]),
    set_eb("eb-f", quiet="yes"),
    set_eb("eb-f", quiet=None),
    set_eb("eb-u", quiet=1),
    set_eb("eb-f", variability="high"),
    set_eb("eb-f", variability=math.nan),
    set_eb("eb-u", variability=True),
], ids=["null-unscored", "inj-unscored", "real-scored", "flagged-eb-unscored", "unflagged-eb-scored",
        "eb-without-evidence", "neutral-tce-on-null", "signal-on-eb", "inj-without-signal", "inj-neutral-signal",
        "unknown-kind", "negative-period", "real-confirmed-unscored", "real-low-mes-scored", "real-koi-scored",
        "wrong-host", "unknown-population", "no-coverage", "real-long-period-scored", "quiet-string", "quiet-null",
        "quiet-int", "variability-string", "variability-nan", "variability-bool"])
def test_load_truth_rejects_rule_breaks(tmp_path, mutate):
    doc = truth_doc()
    mutate(doc)
    with pytest.raises(ValueError, match="truth star"):
        load(tmp_path, doc)


def test_score_keeps_the_legacy_star_interface():
    stars = [Star("p", "inj_random", YEAR, [Signal(10.0, 3.0, 4.0)]), Star("e", "eb", YEAR, scored=False)]
    h = score(stars, {"p": [Candidate(10.0, 3.0, 1.0)], "e": [Candidate(2.0, 0.5, 1.0)]})
    assert (h["S"], h["scored_stars"], h["populations"]["eb"]["false"]) == (1.0, 1, 1)


def test_cli_end_to_end(tmp_path):
    truth, preds, out = tmp_path / "truth.json", tmp_path / "preds.jsonl", tmp_path / "report.json"
    truth.write_text(json.dumps(truth_doc()))
    records = [
        {"star": "inj-r", "status": "ok", "cpu_s": 3.1, "rows": [[10.0, 3.0, 0.9], [5.0, 3.0, 0.3]]},
        {"star": "inj-t", "status": "exit 1", "cpu_s": 0.2, "rows": [[3.7, 1.2, 1.0]]},
        {"star": "null-r", "status": "ok", "cpu_s": 3.0, "rows": [["bad", 1, 1], [7.7, 2.0, 0.5]]},
        {"star": "eb-f", "status": "ok", "cpu_s": 2.9, "rows": []},
        {"star": "eb-u", "status": "timeout", "cpu_s": 20.0, "rows": []},
        {"star": "real", "status": "ok", "cpu_s": 3.3, "rows": [[7.0, 2.0, 1.0]]},
        {"star": "ghost", "status": "ok", "cpu_s": 1.0, "rows": [[1.0, 1.0, 1.0]]},
    ]
    preds.write_text("".join(json.dumps(r) + "\n" for r in records))
    cmd = [sys.executable, "-m", "transitbench.score", "--truth", str(truth), "--preds", str(preds)]
    done = subprocess.run([*cmd, "--out", str(out)], capture_output=True, text=True, check=True)
    assert done.stdout.splitlines()[0] == "S = -0.5000  C = 0.5000  F = 40.00 per 100 stars  EB-as-planet = 0.0000"
    rep = json.loads(out.read_text())
    assert (rep["release"], rep["sample"], rep["window"]) == ("test", "synthetic", [10, 11, 12, 13])
    assert rep["inputs"] == {"missing_stars": 1, "unknown_stars": 1, "rows_over_cap": 0, "runs": 7,
                             "not_ok": {"exit 1": 1, "timeout": 1}, "rows_invalid": 1}
    assert rep["real"]["recall"] == 1.0 and rep["null_false_alarms"]["all"]["k"] == 1
    assert set(rep) == {"release", "sample", "window", "headline", "completeness", "null_false_alarms",
                        "eb_as_planet", "sweep", "real", "inputs"}


def test_prefix_matching_agrees_with_one_pass():
    # The report reads per-signal credit and the sweep off prefixes of the sorted rows; check that on random rows,
    # with a 2:1 pair of planets and a neutral KOI that compete for the same rows.
    from transitbench.match import match_star
    from transitbench.score import _grade

    planets = [TruthSignal(p, t0, 4.0, mes >= 7.1, kind, mes) for p, t0, kind, mes in
               [(10.0, 1.0, "confirmed", 9.0), (20.0, 1.0, "confirmed", 12.0), (31.0, 3.0, "confirmed", 16.0),
                (5.0, 1.0, "koi", 0.0)]]
    star = Star("r", "real", YEAR, planets, False, "real")
    choices = [(10.0, 1.0), (20.0, 1.0), (5.0, 1.0), (40.0, 1.0), (31.0, 3.0), (62.0, 3.0), (7.7, 1.0)]
    rng = np.random.default_rng(0)
    for _ in range(300):
        picks = rng.choice(len(choices), size=rng.integers(0, 6))
        rows = [Candidate(*choices[i], float(rng.choice([0.1, 0.5, 0.9]))) for i in picks]
        g = _grade(star, rows)
        direct = match_star(planets, sorted(rows, key=lambda c: -c.score)[:3], YEAR)
        assert g.result == direct and g.prefix[-1] == (direct.credit, direct.n_false)
        assert sum(g.signal_credit) == direct.credit
        assert [c > 0 for c in g.signal_credit] == list(direct.claimed)
