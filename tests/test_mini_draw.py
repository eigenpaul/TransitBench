"""Tests for the mini pool and draw (transitbench_env/mini_draw.py, scripts/mini_pool.py).

Tests that need the built pool, the dev set or private data skip when they are absent (a clean clone has none). No
test uses the evaluator secret: draws here use a test key.
"""

import hashlib
import io
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from transitbench import score
from transitbench.data import STAR_KEYS, STELLAR_KEYS, deliver_flux, star_bytes, write_star
from transitbench_env import mini_draw

pytest.importorskip("astropy")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build  # noqa: E402
import mini_pool  # noqa: E402

DATA = ROOT / "data"
POOLS = DATA / "private" / "pools.json"
KEY = hashlib.sha256(b"transitbench-v1-mini-tests").digest()  # a test key, not the evaluator secret
TOKENS = 1000
CELL_COUNTS = {"hidden": {"inj_random": 4, "inj_tce_only": 4, "null": 5, "eb": 3},
               "box": {"inj_random": 3, "inj_tce_only": 3, "null": 4, "eb": 2}}


@pytest.fixture(scope="module")
def pool():
    sha = mini_draw.MINI_POOL["sha"]
    if not sha or not (DATA / mini_draw.POOL_DIR / sha / "manifest.json").exists():
        pytest.skip("mini pool not built")
    return mini_draw.load_pool(DATA, sha)


@pytest.fixture(scope="module")
def plans(pool):
    """(split, match) -> [(box, hidden)] for TOKENS tokens."""
    return {(split, match): [mini_draw.plan(pool, KEY, split, f"{split}-{i:08d}", match_variability=match)
                             for i in range(TOKENS)]
            for split in ("train", "dev") for match in (True, False)}


def synthetic_star(seed=0) -> dict:
    rng = np.random.default_rng(seed)
    n = 4000
    return {"time": np.sort(rng.uniform(900, 1270, n)), "flux": 1 + 1e-4 * rng.standard_normal(n),
            "quarter": np.repeat(np.arange(4, dtype=np.int8), n // 4), **{k: np.float64(1.0) for k in STELLAR_KEYS}}


def test_star_bytes_are_the_bytes_write_star_writes(tmp_path):
    for seed in range(3):
        arrays = synthetic_star(seed)
        write_star(tmp_path / "s.npz", arrays)
        assert star_bytes(arrays) == (tmp_path / "s.npz").read_bytes()
    with pytest.raises(ValueError):
        star_bytes({**synthetic_star(), "flux_err": np.zeros(4000)})


def test_secret_is_created_once_and_never_replaced(tmp_path):
    with pytest.raises(mini_draw.MiniPoolError, match="^TB-E002"):
        mini_draw.read_secret(tmp_path)
    mini_pool.create_secret(tmp_path / mini_draw.SECRET)
    first = (tmp_path / mini_draw.SECRET).read_bytes()
    assert len(mini_draw.read_secret(tmp_path)) == mini_draw.SECRET_BYTES
    with pytest.raises(SystemExit):
        mini_pool.create_secret(tmp_path / mini_draw.SECRET)
    cli = [sys.executable, "-I", str(ROOT / "scripts" / "mini_pool.py"), "secret", "--new", "--data-dir", str(tmp_path)]
    run = subprocess.run(cli, capture_output=True, text=True, cwd=ROOT)
    assert run.returncode != 0 and "refusing to replace" in run.stderr
    assert (tmp_path / mini_draw.SECRET).read_bytes() == first
    (tmp_path / mini_draw.SECRET).write_text("abcd\n")
    with pytest.raises(mini_draw.MiniPoolError, match="^TB-E002"):
        mini_draw.read_secret(tmp_path)


def test_pool_and_star_files_fail_closed(tmp_path, pool):
    with pytest.raises(mini_draw.MiniPoolError, match="^TB-E002"):
        mini_draw.load_pool(tmp_path, pool.name)
    copy = tmp_path / mini_draw.POOL_DIR / pool.name
    (copy / "stars").mkdir(parents=True)
    raw = (pool.root / "manifest.json").read_bytes()
    (copy / "manifest.json").write_bytes(raw.replace(b'"version": 1', b'"version": 2'))
    with pytest.raises(mini_draw.MiniPoolError, match="^TB-E002"):
        mini_draw.load_pool(tmp_path, pool.name)
    (copy / "manifest.json").write_bytes(raw)
    other = mini_draw.load_pool(tmp_path, pool.name)
    pid = sorted(pool.stars)[0]
    data = bytearray((pool.root / "stars" / f"{pid}.npz").read_bytes())
    data[-1] ^= 1
    (copy / "stars" / f"{pid}.npz").write_bytes(bytes(data))
    with pytest.raises(mini_draw.MiniPoolError, match="^TB-E002"):
        mini_draw._load(other, pid)


def test_pool_files_match_the_manifest(pool):
    for pid, s in pool.stars.items():
        assert hashlib.sha256((pool.root / "stars" / f"{pid}.npz").read_bytes()).hexdigest() == s["sha256"], pid
    assert pool.name == hashlib.sha256((pool.root / "manifest.json").read_bytes()).hexdigest()[:16]


def test_pool_is_disjoint_from_every_reserve_and_test_list(pool):
    if not POOLS.exists():
        pytest.skip("private pools absent")
    kic_map = json.loads((pool.root / "kic_map.json").read_text())["stars"]
    assert set(kic_map) == set(pool.stars)
    pools = json.loads(POOLS.read_text())
    lists = ("random", "tce_only", "eb", "real_7_12", "real_12_up")
    reserve = {int(k) for p in lists for k in pools[p]["reserve"]} | {int(k) for k in pools["natural"]}
    split_lists = {s: {name: {int(k) for k in pools[name][s]} for name in ("random", "tce_only", "eb")}
                   for s in ("train", "dev")}
    test = set()
    for path in (DATA / "private" / "releases").glob("*/map_*.json"):
        test |= {int(v["kic"]) for v in json.loads(path.read_text())["stars"].values()}
    kics = {pid: v["kic"] for pid, v in kic_map.items()}
    assert not set(kics.values()) & (reserve | test)
    assert len(set(kics.values())) == len(kics)
    for pid, s in pool.stars.items():
        assert kics[pid] in split_lists[s["split"]][s["host"]], pid
        assert kic_map[pid]["list"] == s["host"]


def test_composition_and_disjointness_over_1000_tokens(pool, plans):
    for (split, match), drawn in plans.items():
        null_tce = Counter()
        for box, hidden in drawn:
            pids = [p.pid for p in box + hidden]
            assert len(set(pids)) == len(pids), (split, match)
            for part, picks in (("box", box), ("hidden", hidden)):
                prefix = "m" if part == "box" else "h"
                assert [p.name for p in picks] == [f"{prefix}-{i:02d}" for i in range(len(picks))]
                cells = Counter(p.cell for p in picks)
                want = CELL_COUNTS[part]
                assert cells["inj_random"] == want["inj_random"] and cells["inj_tce_only"] == want["inj_tce_only"]
                assert cells["null_random"] + cells["null_tce_only"] == want["null"] and cells["eb"] == want["eb"]
                assert cells["null_tce_only"] in ((2, 3) if part == "hidden" else (2,))
                null_tce[part] += cells["null_tce_only"]
                for p in picks:
                    s = pool.stars[p.pid]
                    assert s["split"] == split and s["host"] == mini_draw.POPULATION[p.cell][1]
                    if p.cell == "eb":
                        assert s["role"] == part and (part == "box" or s["eb"]["flagged"])
                    else:
                        assert s["role"] == "any"
                for cell in ("inj_random", "inj_tce_only"):
                    targets = sorted(p.target for p in picks if p.cell == cell)
                    n = len(targets)
                    assert all(4 + 16 * i / n <= t < 4 + 16 * (i + 1) / n for i, t in enumerate(targets))
                assert all(p.target is None for p in picks if not p.cell.startswith("inj"))
        assert 0.45 < null_tce["hidden"] / (5 * len(drawn)) < 0.55, (split, match)


def test_plan_depends_on_key_and_token(pool):
    a = mini_draw.plan(pool, KEY, "train", "train-00000001")
    assert a == mini_draw.plan(pool, KEY, "train", "train-00000001")
    assert a != mini_draw.plan(pool, KEY, "train", "train-00000002")
    assert a != mini_draw.plan(pool, hashlib.sha256(b"another key").digest(), "train", "train-00000001")


def variability_auc(pool, drawn) -> float:
    eb, host = [], []
    for _, hidden in drawn:
        for p in hidden:
            (eb if p.cell == "eb" else host).append(pool.stars[p.pid]["variability"])
    host = np.sort(host)
    below, upto = np.searchsorted(host, eb, "left"), np.searchsorted(host, eb, "right")
    return float(np.mean((below + upto) / 2) / host.size)


def test_matching_removes_the_variability_tell(pool, plans):
    matched, plain = variability_auc(pool, plans[("train", True)]), variability_auc(pool, plans[("train", False)])
    assert matched < 0.58 < 0.7 < plain


def test_match_bins_follow_the_reference():
    hosts = {"random": [f"r{i}" for i in range(100)], "tce_only": [f"t{i}" for i in range(100)]}
    var = {f"{c}{i}": 1.0 + i / 10 for c in "rt" for i in range(100)}
    bins = mini_draw.match_bins([1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], hosts, var, n_bins=3, min_hosts=6)
    assert [b.share for b in bins] == pytest.approx([1 / 3] * 3)
    assert bins[-1].upper == math.inf and bins[0].upper == pytest.approx(math.sqrt(12.0))
    assert sum(len(b.hosts["random"]) for b in bins) == 100
    thin = mini_draw.match_bins([1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 50.0, 60.0, 70.0], hosts, var, n_bins=3, min_hosts=6)
    assert [b.share for b in thin] == pytest.approx([1 / 3, 2 / 3])  # no host above 17: the top bin merged down


def test_pool_bins_hold_enough_hosts(pool):
    for split, sets in pool.sets.items():
        for part, bins in sets["bins"].items():
            assert sum(b.share for b in bins) == pytest.approx(1.0)
            assert all(len(v) >= mini_draw.MATCH_MIN_HOSTS for b in bins for v in b.hosts.values()) or len(bins) == 1
            assert [b.upper for b in bins] == sorted(b.upper for b in bins)


def test_eb_evidence_is_invariant_to_the_offset(pool):
    """The draw shifts the pool's EB evidence and variability into the delivered frame; recomputing them on the
    delivered star at two drawn offsets gives the same flags, secondary phase, duration and variability."""
    keyed = mini_draw._Keyed(KEY, pool.name, "offset-test")
    for pid, s in sorted(pool.stars.items()):
        if s["host"] != "eb":
            continue
        time_, flux, quarter, qmedian = mini_draw._load(pool, pid)
        eb = s["eb"]
        for purpose in ("offset", "offset-2"):
            off = float(np.random.default_rng(keyed.seed(pid, purpose)).uniform(-5000.0, 5000.0))
            delivered = deliver_flux(flux, quarter, qmedian)
            eph = {"period": eb["period"], "t0": eb["t0"] + off, "source": eb["source"],
                   "duration": eb["duration"] if eb["source"] == "koi" else None}
            again = build.eb_truth(time_ + off, delivered, eph)
            assert again["evidence"] == eb["evidence"], pid
            assert again["secondary_phase"] == eb["secondary_phase"] and again["duration"] == eb["duration"], pid
            entry = {k: again[k] for k in ("period", "t0", "duration", "secondary_phase")}
            v = build.star_features(time_ + off, delivered, quarter, build.signal_mask(time_ + off, [], entry),
                                    s["stellar"])[build.VARIABILITY]
            if not s["variability_unmasked"]:
                assert v == pytest.approx(s["variability"], rel=1e-9), pid


def test_dev_parity_reproduces_public_dev_stars_byte_for_byte():
    """mini_draw.deliver, the draw's delivery path, rebuilds 20 public dev stars from dev/map.json exactly."""
    paths = [ROOT / "dev" / f for f in ("map.json", "truth.json", "files.json")]
    if not all(p.exists() for p in paths) or not (DATA / "tables").exists():
        pytest.skip("dev set or tables absent")
    mapping = json.loads(paths[0].read_text())["stars"]
    truth = json.loads(paths[1].read_text())["stars"]
    files = json.loads(paths[2].read_text())["files"]
    tables = build.load_tables()
    stellar = build.stellar_rows(tables["stellar"])
    eb_periods = build.injection_eb_periods(tables["koi"], tables["kirk"])
    by_pop: dict[str, list[str]] = {}
    for sid in sorted(mapping):
        by_pop.setdefault(truth[sid]["population"], []).append(sid)
    chosen = by_pop["inj"][:8] + by_pop["null"][:6] + by_pop["eb"][:6]
    lc_dir = DATA / "lc" / "mast"
    if not all((lc_dir / f"{mapping[sid]['kic']}.npz").exists() for sid in chosen):
        pytest.skip("dev light curves absent")
    for sid in chosen:
        m, t = mapping[sid], truth[sid]
        kic = m["kic"]
        d = np.load(lc_dir / f"{kic}.npz")
        order = np.argsort(d["time"], kind="stable")
        lc = (d["time"][order], d["flux"][order], d["quarter"][order], d["qmedian"])
        injection = None
        if t["population"] == "inj":
            injection = (stellar[kic], np.random.default_rng(m["inject_seed"]), eb_periods,
                         t["signals"][0]["mes_target"])
        rounded = {k: build.round_sig(stellar[kic][k]) for k in STELLAR_KEYS}
        arrays, signal = mini_draw.deliver(lc, rounded, m["time_offset"], injection)
        assert hashlib.sha256(star_bytes(arrays)).hexdigest() == files[sid], sid
        if signal:
            expected = {k: v for k, v in t["signals"][0].items() if k != "eb_period_copy"}
            assert signal == expected, sid


def test_two_processes_draw_identical_tasks(pool):
    code = ("import json, sys\n"
            "from pathlib import Path\n"
            "from transitbench_env import mini_draw as md\n"
            "pool = md.load_pool(Path(sys.argv[1]), sys.argv[2])\n"
            "key = bytes.fromhex(sys.argv[3])\n"
            "print(json.dumps([md.draw(pool, key, s, t, match_variability=m).digest "
            "for s, t, m in json.loads(sys.argv[4])]))\n")
    jobs = [("train", f"train-{i:08d}", i % 4 != 3) for i in range(16)] + [("dev", f"dev-{i:08d}", True)
                                                                         for i in range(4)]
    argv = [sys.executable, "-I", "-c", code, str(DATA), pool.name, KEY.hex(), json.dumps(jobs)]
    procs = [subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    outs = [p.communicate(timeout=900) for p in procs]
    assert all(p.returncode == 0 for p in procs), [o[1][-500:] for o in outs]
    first, second = (json.loads(o[0]) for o in outs)
    assert first == second and len(set(first)) == len(jobs)


def test_draw_is_fast_and_its_truth_is_well_formed(pool):
    seconds = []
    for i in range(7):
        t0 = time.perf_counter()
        d = mini_draw.draw(pool, KEY, "train", f"timing-{i:08d}")
        seconds.append(time.perf_counter() - t0)
        assert d.digest == mini_draw.hidden_digest(d.hidden_files, d.hidden_truth)
        hidden, box = score.parse_truth(d.hidden_truth), score.parse_truth(d.box_truth)
        assert Counter(s.population for s in hidden) == {"inj": 8, "null": 5, "eb": 3}
        assert Counter(s.population for s in box) == {"inj": 6, "null": 4, "eb": 2}
        assert all(s.scored for s in hidden) and sum(len(s.signals) for s in hidden) == 8
        assert set(d.hidden_files) == set(d.hidden_truth["stars"]) and set(d.box_files) == set(d.box_truth["stars"])
        assert set(d.roles) == set(d.box_files) | set(d.hidden_files) and len(d.roles) == 28
        for name, data in {**d.box_files, **d.hidden_files}.items():
            assert set(np.load(io.BytesIO(data)).files) == set(STAR_KEYS)
            cell = d.roles[name]
            entry = (d.hidden_truth if name.startswith("h") else d.box_truth)["stars"][name]
            assert entry["population"] == mini_draw.POPULATION[cell][0]
    assert float(np.median(seconds)) < 6.0, seconds
