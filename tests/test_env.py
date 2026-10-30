"""The two tasksets (transitbench_env/task.py, transitbench_survey, transitbench_mini): plugin resolution,
the task data and its wire form, keys, the hooks against a fake box, scoring, and the eval TOMLs.

Needs verifiers, a dependency: uv run --group build pytest tests/test_env.py. Tests that read the
evaluator's private data or the mini pool skip without them. The end-to-end checks are vf-validate runs (README).
"""

import asyncio
import io
import itertools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

vf = pytest.importorskip("verifiers.v1", reason="needs verifiers (the env extra)")

from transitbench import archive  # noqa: E402
from transitbench_env import box, grading, hidden, task  # noqa: E402
from transitbench_env.harness import TransitBashConfig, TransitBashHarness  # noqa: E402
from transitbench_env.task import TransitTaskConfig  # noqa: E402
from transitbench_mini import MiniConfig, MiniData, MiniTask, MiniTaskset  # noqa: E402
from transitbench_survey import SurveyConfig, SurveyData, SurveyTask, SurveyTaskset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRIVATE = DATA / "private"
STARTER = ROOT / "workspace" / "solution" / "solution.py"
SURVEY_ID, MINI_ID = "transitbench-survey", "transitbench-mini"
PLUGINS = [(SURVEY_ID, SurveyTaskset, SurveyConfig, SurveyTask, SurveyData),
           (MINI_ID, MiniTaskset, MiniConfig, MiniTask, MiniData)]


def mini_pool_skip() -> str | None:
    from transitbench_env import mini_draw

    if not (DATA / mini_draw.POOL_DIR / mini_draw.MINI_POOL["sha"] / "manifest.json").is_file():
        return "needs the mini pool (scripts/mini_pool.py)"
    if not (DATA / mini_draw.SECRET).is_file():
        return "needs the mini secret"
    return None


def survey_tasks(**config) -> list[SurveyTask]:
    return list(SurveyTaskset(SurveyConfig(id=SURVEY_ID, **config)))


def mini_tasks(n: int, **config) -> list[MiniTask]:
    return list(itertools.islice(MiniTaskset(MiniConfig(id=MINI_ID, **config)), n))


def make_trace(t) -> vf.Trace:
    from verifiers.v1.trace import TraceTask

    return vf.Trace(task=TraceTask(type=type(t).__name__, data=t.data, key=t.key, hash=t.hash), state=vf.State(),
                    agent=vf.AgentInfo(config=vf.AgentConfig(runtime=vf.DockerConfig()), name="test", trainable=False))


class FakeBox:
    """Records what box.put_files writes and runs, and answers like a box where every command succeeds."""

    def __init__(self, name: str = "box", exit_code: int = 0):
        self.name, self.exit_code, self.writes, self.commands = name, exit_code, [], []
        self.config = vf.DockerConfig()

    def with_user(self, user):
        return self

    async def write(self, path, data):
        self.writes.append((path, data))

    async def run(self, argv, env):
        self.commands.append(argv)
        return SimpleNamespace(exit_code=self.exit_code, stdout="", stderr="tar: some detail\n")

    def files(self) -> dict[str, tuple[bytes, int]]:
        (_, data), = self.writes
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            return {f"/{m.name}": (tar.extractfile(m).read(), m.mode) for m in tar if m.isfile()}


# ---- plugins and task data -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("taskset_id, taskset_cls, config_cls, task_cls, data_cls", PLUGINS)
def test_ids_resolve_to_the_tasksets_with_the_capped_harness(taskset_id, taskset_cls, config_cls, task_cls, data_cls):
    from verifiers.v1.envs.single_agent import SingleAgentEnv
    from verifiers.v1.utils import loaders

    assert loaders.taskset_class(taskset_id) is taskset_cls
    assert loaders.taskset_config_type(taskset_id) is config_cls
    assert taskset_cls.task_type() is task_cls
    assert task_cls.data_type() is data_cls
    assert task_cls.config_type() is TransitTaskConfig
    assert config_cls().task == TransitTaskConfig()
    assert "__init__" not in vars(taskset_cls) and task_cls.NEEDS_CONTAINER
    # The taskset's module exports the harness, so it is the default.
    assert loaders.default_harness_id(taskset_id) == taskset_id
    assert loaders.harness_class(taskset_id) is TransitBashHarness
    assert loaders.harness_config_type(taskset_id) is TransitBashConfig
    assert loaders.environment_class(taskset_id) is SingleAgentEnv
    env = vf.resolve_env_config({"taskset": {"id": taskset_id}})
    (harness,) = env.agent_harnesses().values()
    assert isinstance(harness, TransitBashConfig) and harness.id == taskset_id


def test_survey_tasks():
    tasks = survey_tasks()
    assert [t.key for t in tasks] == ["survey-r1", "survey-dev"] == [t.data.id for t in tasks]
    r1, dev = (t.data for t in tasks)
    assert (r1.release, r1.samples, dev.release, dev.samples) == ("r1", ("A", "B", "dev"), "dev", ("dev",))
    for d in (r1, dev):
        assert (d.preset, d.star_cap_s, d.command_cap_s, d.workers, d.max_turns, d.max_tokens) == (
            "survey", 60.0, 940.0, 16, 60, 200_000)
        assert d.image == task.IMAGES["survey"] and d.workdir == "/app"
        assert d.resources == vf.TaskResources(cpu=16, memory=32, disk=10)
        assert d.timeout == vf.TaskTimeout(setup=1800, agent=14400, finalize=300, scoring=10800)
    caps = task.caps(r1)
    assert caps["hidden"] == {"A": 1000, "B": 1000, "dev": 196} and caps["labelled"] == 196
    assert task.caps(dev) == caps and r1.prompt == dev.prompt == box.render_task_md(caps)
    assert survey_tasks(image="transitbench-survey:local")[1].data.image == "transitbench-survey:local"


def test_mini_tasks():
    from transitbench_env import mini_draw

    tasks = mini_tasks(3)
    d = tasks[0].data
    assert [t.key for t in tasks] == ["train-00000000", "train-00000001", "train-00000002"]
    assert (d.preset, d.split, d.pool, d.builder, d.match_variability) == (
        "mini", "train", mini_draw.MINI_POOL["sha"], mini_draw.BUILDER, False)
    assert (d.star_cap_s, d.command_cap_s, d.workers, d.max_turns, d.max_tokens) == (20.0, 75.0, 4, 12, 32_768)
    assert d.image == task.IMAGES["base"] and d.resources == vf.TaskResources(cpu=4, memory=8, disk=5)
    assert d.timeout == vf.TaskTimeout(setup=600, agent=3600, finalize=120, scoring=900)
    caps = task.caps(d)
    assert caps["hidden"] == {"inj": 8, "null": 5, "eb": 3} and caps["labelled"] == 12
    assert d.prompt == box.render_task_md(caps)
    dev = list(MiniTaskset(MiniConfig(id=MINI_ID, split="dev", match_variability=True)))
    assert len(dev) == 32 and dev[-1].key == "dev-00000031" and dev[0].data.match_variability


@pytest.mark.parametrize("taskset_id", [SURVEY_ID, MINI_ID])
def test_task_data_is_public_and_shaped_for_isolated_grading(taskset_id):
    tasks = survey_tasks() if taskset_id == SURVEY_ID else mini_tasks(4)
    for t in tasks:
        d = t.data
        assert d.scoring.mode == "isolated" and d.network.allow == [] and d.network.network_restricted
        assert d.artifacts == [vf.Artifact(source="/app/solution")]
        # verifiers' fallback collect and vf-validate collect under grading's own cap (capture_solution).
        assert d.artifact_max_bytes == d.solution_max_bytes + grading.COLLECT_SLACK == 6 << 20
        assert set(task.caps(d)) == set(box.CAPS_KEYS) and task.caps(d)["wall_s"] == d.timeout.agent
        assert not [name for name, field in type(d).model_fields.items() if field.exclude]


def test_star_memory_caps_fit_the_box():
    """workers x memory_bytes + MEMORY_RESERVE leaves at least 1 GiB under each box's nominal memory, so it fits the
    box's MemTotal while the kernel keeps less than that (task.PRESETS)."""
    for preset, p in task.PRESETS.items():
        nominal = int(p["resources"].memory * 2**30)
        assert p["workers"] * p["memory_bytes"] + task.MEMORY_RESERVE <= nominal - (1 << 30), preset
    # The same value reaches caps.json, the grading plan (both from caps) and TASK.md.
    for t, shown in ((survey_tasks()[0], "1,792 MiB of memory"), (mini_tasks(1)[0], "1,280 MiB of memory")):
        assert task.caps(t.data)["memory_bytes"] == task.PRESETS[t.data.preset]["memory_bytes"]
        assert shown in t.data.prompt


def test_wire_round_trip_is_stable():
    from verifiers.v1.task import WireTaskData, task_key

    for t in survey_tasks() + mini_tasks(5):
        cls, d = type(t.data), t.data
        wire = d.model_dump(mode="json")
        assert cls.model_validate(wire) == d
        assert cls.model_validate_json(d.model_dump_json()) == d
        # As a client dispatches it and a worker rebuilds it (serve/server.py _build_task).
        rebuilt = cls.model_validate(WireTaskData.model_validate(wire).model_dump(mode="json"))
        assert rebuilt == d and task_key(rebuilt.model_dump(mode="json", exclude_none=True)) == t.hash
        assert type(t)(rebuilt, t.config).key == t.key


def test_grading_task_refuses_shared_scoring():
    for t in survey_tasks()[:1] + mini_tasks(1):
        with pytest.raises(ValueError, match="isolated"):
            t.grading_task(None, scoring=vf.ScoringConfig(mode="shared"))
        assert t.grading_task(None, scoring=vf.ScoringConfig(mode="isolated")) is t


def test_mini_keys_are_unique_and_stable():
    first = [t.key for t in mini_tasks(1000)]
    assert len(set(first)) == 1000
    assert [t.key for t in mini_tasks(1000)] == first
    assert all(re.fullmatch(r"train-\d{8}", k) for k in first)


def test_libraries_are_the_images():
    pins = dict(re.findall(r"^([a-z-]+)==(\S+)$", (ROOT / "image" / "requirements.in").read_text(), re.MULTILINE))
    libraries = {k: v for k, v in task.LIBRARIES.items() if k != "python"}
    assert list(libraries) == ["numpy", "scipy", "astropy", "scikit-learn"]
    assert {k: pins[k] for k in libraries} == libraries
    assert "FROM python:3.12-slim" in (ROOT / "image" / "Dockerfile").read_text()


@pytest.mark.skipif(not PRIVATE.is_dir(), reason="needs data/private")
def test_task_data_holds_no_needle():
    leak = load_script("leak_needles")
    needles = leak.needles(DATA)
    for t in survey_tasks() + mini_tasks(50):
        hits = needles.scan(t.data.model_dump_json().encode())
        assert not any(hits.values()), (t.key, hits)


def load_script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- hooks against a fake box ----------------------------------------------------------------------------------------

def test_survey_setup_writes_the_box(tmp_path):
    t = survey_tasks(task=TransitTaskConfig(data_dir=tmp_path))[1]
    rt = FakeBox()
    asyncio.run(t.setup(make_trace(t), rt))
    files = rt.files()
    assert files["/app/TASK.md"][0].decode() == t.data.prompt
    assert json.loads(files["/app/caps.json"][0]) == t.caps()
    assert files["/app/solution/solution.py"] == (STARTER.read_bytes(), 0o644)
    assert files["/app/run_dev.py"][1] == 0o755
    assert {p for p in files if p.startswith("/app/transitbench/")} == {
        f"/app/transitbench/{n}.py" for n in box.SCIENCE}
    assert not [p for p in files if p.startswith("/app/data")]          # the image holds the dev set
    assert rt.commands[0][2].startswith("rm -rf -- /app/solution /app/transitbench && tar -xf ")


def test_setup_installs_the_configured_solution_and_finalize_requires_it_back(tmp_path, monkeypatch):
    solution = tmp_path / "sol"
    (solution / "lib").mkdir(parents=True)
    (solution / "solution.py").write_text("print('hi')\n")
    (solution / "lib" / "run.sh").write_text("#!/bin/sh\n")
    (solution / "lib" / "run.sh").chmod(0o755)
    t = survey_tasks(task=TransitTaskConfig(data_dir=tmp_path / "data", solution=solution))[1]
    trace, rt = make_trace(t), FakeBox()
    asyncio.run(t.setup(trace, rt))
    files = rt.files()
    assert files["/app/solution/solution.py"] == (b"print('hi')\n", 0o644)
    assert files["/app/solution/lib/run.sh"] == (b"#!/bin/sh\n", 0o755)

    async def capture(runtime, *, max_bytes, max_files):
        return archive.from_dir(solution, max_bytes=max_bytes, max_files=max_files)

    monkeypatch.setattr(grading, "capture_solution", capture)
    asyncio.run(t.finalize(trace, rt))                  # the same solution comes back
    info = trace.info["transitbench"]
    sha = archive.from_dir(solution).sha256
    assert info["solution"]["sha256"] == sha and trace.state.artifacts == {"/app/solution": archive.from_dir(
        solution).tar}
    assert (tmp_path / "data" / "private" / "archives" / f"{sha}.tar").read_bytes() == trace.state.artifacts[
        "/app/solution"]

    (solution / "solution.py").write_text("print('changed')\n")
    t2 = survey_tasks(task=TransitTaskConfig(data_dir=tmp_path / "data", solution=tmp_path / "data" / "private" /
                                             "archives" / f"{sha}.tar"))[1]
    trace2 = make_trace(t2)
    with pytest.raises(grading.GradingError) as e:
        asyncio.run(t2.finalize(trace2, rt))
    assert str(e.value) == "TB-E204: the captured solution differs from the configured one"
    assert e.value.__cause__ is None and e.value.__suppress_context__
    log = json.loads((tmp_path / "data" / "private" / "logs" / f"{trace2.id}.log").read_text())
    assert log["code"] == "TB-E204" and log["hook"] == "finalize" and sha in log["detail"]
    assert trace2.info["transitbench"]["solution"]["sha256"] != sha


@pytest.mark.parametrize("problem", ["missing", "empty", "box"])
def test_setup_errors_are_fixed_text(tmp_path, problem):
    solution = tmp_path / "sol"
    solution.mkdir()
    config = TransitTaskConfig(data_dir=tmp_path / "data",
                               solution=solution if problem == "empty" else (tmp_path / "nothing"))
    t = survey_tasks(task=config if problem != "box" else TransitTaskConfig(data_dir=tmp_path / "data"))[1]
    trace = make_trace(t)
    with pytest.raises(grading.GradingError) as e:
        asyncio.run(t.setup(trace, FakeBox(exit_code=2)))
    want = "TB-E104: writing the task files into the box failed" if problem == "box" else (
        "TB-E004: the configured solution is missing, empty or over the limits")
    assert str(e.value) == want and e.value.__suppress_context__
    log = json.loads((tmp_path / "data" / "private" / "logs" / f"{trace.id}.log").read_text())
    assert log["hook"] == "setup" and log["code"] == want[:7]


def test_finalize_needs_a_data_dir(monkeypatch):
    monkeypatch.delenv(task.DATA_ENV, raising=False)
    t = survey_tasks()[1]

    async def capture(runtime, *, max_bytes, max_files):
        return archive.empty("missing")

    monkeypatch.setattr(grading, "capture_solution", capture)
    with pytest.raises(grading.GradingError, match="^TB-E001: "):
        asyncio.run(t.finalize(make_trace(t), FakeBox()))


def test_headline_grades_only_in_the_staged_box(tmp_path):
    t = survey_tasks(task=TransitTaskConfig(data_dir=tmp_path))[1]
    trace = make_trace(t)
    with pytest.raises(grading.GradingError) as e:
        asyncio.run(t.headline(trace, FakeBox("solver")))
    assert str(e.value) == "TB-E101: grading ran outside the staged verifier"
    assert json.loads((tmp_path / "private" / "logs" / f"{trace.id}.log").read_text())["code"] == "TB-E101"


def test_stage_verifier_records_public_digests(tmp_path, monkeypatch):
    t = survey_tasks(task=TransitTaskConfig(data_dir=tmp_path))[1]
    trace = make_trace(t)
    trace.info["transitbench"] = {"solution": {"sha256": "ab" * 32}}
    got = {}

    async def stage(runtime, sets, caps, *, require_jail, order_key):
        got.update(sets=sets, caps=caps, require_jail=require_jail, order_key=order_key)
        return grading.Plan([("dev", sid) for sid in sets[0].stars], "d" * 64, "none")

    monkeypatch.setattr(grading, "stage", stage)
    asyncio.run(t.stage_verifier(trace, FakeBox("grader")))
    (dev,) = got["sets"]
    assert (dev.name, got["require_jail"], got["order_key"], got["caps"]) == ("dev", False, "ab" * 32, t.caps())
    assert trace.info["transitbench"]["staged"] == {"dev": {"files_digest": dev.digest, "n": 196}}
    assert t._verifier == "grader" and t._plan.isolation == "none"


def test_finalize_stops_the_agent_before_the_capture(tmp_path, monkeypatch):
    from transitbench_env import harness

    t = survey_tasks(task=TransitTaskConfig(data_dir=tmp_path))[1]
    rt = FakeBox()
    commands_at_capture = []

    async def capture(runtime, *, max_bytes, max_files):
        commands_at_capture.append(list(rt.commands))
        return archive.empty("missing")

    monkeypatch.setattr(grading, "capture_solution", capture)
    asyncio.run(t.finalize(make_trace(t), rt))
    ((argv,),) = commands_at_capture
    assert harness.STOP_AGENT in argv and argv[-2:] == [harness.PROGRAM_SHA256, "1"]   # Docker: orphans too
    rt.config = vf.PrimeConfig()
    asyncio.run(t.finalize(make_trace(t), rt))
    assert rt.commands[-1][-1] == "0"


def test_concurrent_calls_for_one_task_draw_it_once(tmp_path, monkeypatch):
    import transitbench_mini
    from transitbench_env import mini_draw

    calls = []

    def draw(pool, secret, split, token, *, match_variability):
        calls.append((pool, secret, split, token, match_variability))
        time.sleep(0.3)
        if token == "train-00000002":
            raise mini_draw.MiniPoolError(mini_draw.E_DRAW)
        return SimpleNamespace(token=token)

    monkeypatch.setattr(mini_draw, "load_pool", lambda data_dir, pool: f"pool {pool}")
    monkeypatch.setattr(mini_draw, "read_secret", lambda data_dir: b"secret")
    monkeypatch.setattr(mini_draw, "draw", draw)

    def at_once(token, n=8):
        barrier, got = threading.Barrier(n), []

        def call():
            barrier.wait()
            try:
                got.append(transitbench_mini.cached_draw(tmp_path, "p", "train", token, False))
            except mini_draw.MiniPoolError as e:
                got.append(e)

        threads = [threading.Thread(target=call) for _ in range(n)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return got

    got = at_once("train-00000001")
    assert calls == [("pool p", b"secret", "train", "train-00000001", False)]
    assert len(got) == 8 and all(g is got[0] for g in got) and got[0].token == "train-00000001"
    assert transitbench_mini.cached_draw(tmp_path, "p", "train", "train-00000001", False) is got[0]
    assert len(calls) == 1
    # A failed draw reaches every waiter once and is not cached.
    failed = at_once("train-00000002")
    assert len(failed) == 8 and all(isinstance(e, mini_draw.MiniPoolError) for e in failed) and len(calls) == 2
    at_once("train-00000002", n=1)
    assert len(calls) == 3
    assert transitbench_mini.DRAW_CACHE >= 64


# ---- scoring ---------------------------------------------------------------------------------------------------------

def oracle_records(sets) -> dict[str, dict]:
    records = {}
    for s in sets:
        for sid, rows in task.oracle_rows(s.truth).items():
            records[sid] = {"star": sid, "status": "ok", "cpu_s": 1.0, "wall_s": 2.0, "rows": rows[:3],
                            "n_rows": len(rows), "n_invalid": 0, "isolation": "none", "slot": 0, "leak": 0}
    return records


def staged(t, sets, trace):
    t._sets = {s.name: s for s in sets}
    t._plan = grading.Plan([(s.name, sid) for s in sets for sid in sorted(s.stars)], "d" * 64, "none",
                           {"image": "transitbench-survey:local", "runner_sha256": "e" * 64})
    trace.info["transitbench"] = {"v": 1, "solution": {"sha256": "ab" * 32, "status": "ok", "bytes": 10},
                                  "scan": {"n_flags": 2}}


def test_survey_scoring_keeps_aggregates_in_the_trace(tmp_path):
    t = survey_tasks(task=TransitTaskConfig(data_dir=tmp_path))[1]
    trace, sets = make_trace(t), [hidden.dev_set()]
    staged(t, sets, trace)
    records = oracle_records(sets)
    inj = next(sid for sid, rows in task.oracle_rows(sets[0].truth).items() if rows)
    records[inj] = {**records[inj], "status": "timeout", "rows": [], "n_rows": 0, "wall_s": 60.0}
    assert t.score_records(trace, records) == trace.metrics["S_dev"] < 1.0
    assert trace.metrics["C_dev"] < 1.0 and trace.metrics["F_per_100_dev"] == 0.0
    grade = trace.info["transitbench"]["grade"]
    assert set(grade) == {"inputs", "cpu_s", "near_cap", "isolation", "rows_sha256", "headline", "runs", "sidecar"}
    assert grade["runs"]["dev"]["timeout"] == 1 and grade["runs"]["dev"]["ok"] == 195
    assert grade["inputs"]["image"] == "transitbench-survey:local" and grade["isolation"] == "none"
    assert grade["inputs"]["evaluator"] == {"system": platform.system(), "machine": platform.machine()}
    assert set(grade["headline"]["dev"]) == set(task.HEADLINE)
    assert not re.search(r"dev-\d{3}", json.dumps(trace.info)), "a star id reached the trace"
    assert trace.metrics["n_timeout"] == 1 and trace.metrics["isolated"] == 0.0 and trace.metrics["scan_flags"] == 2
    sidecar = json.loads((tmp_path / "private" / "grades" / "dev" / f"{trace.id}.json").read_text())
    assert sidecar["records"] == records and sidecar["reports"]["dev"]["headline"]["S"] == trace.metrics["S_dev"]
    assert grade["sidecar"] == f"dev/{trace.id}"

    perfect = make_trace(t)
    staged(t, sets, perfect)
    assert t.score_records(perfect, oracle_records(sets)) == 1.0
    assert task.records_sha256(oracle_records(sets)) == perfect.info["transitbench"]["grade"]["rows_sha256"]


def test_scoring_errors_are_fixed_text(tmp_path, monkeypatch):
    t = survey_tasks(task=TransitTaskConfig(data_dir=tmp_path))[1]
    trace, sets = make_trace(t), [hidden.dev_set()]
    staged(t, sets, trace)
    t._verifier = "grader"

    async def run(runtime, plan, *, poll_s):
        assert poll_s == SurveyTask.POLL_S
        return {}

    monkeypatch.setattr(grading, "run", run)
    with pytest.raises(grading.GradingError) as e:
        asyncio.run(t.headline(trace, FakeBox("grader")))
    assert str(e.value) == "TB-E205: scoring the grader's results failed" and e.value.__suppress_context__
    assert "KeyError" in json.loads((tmp_path / "private" / "logs" / f"{trace.id}.log").read_text())["detail"]


@pytest.mark.skipif(mini_pool_skip() is not None, reason=str(mini_pool_skip()))
def test_mini_setup_and_scoring():
    t = mini_tasks(1, task=TransitTaskConfig(data_dir=DATA))[0]
    trace, rt = make_trace(t), FakeBox()
    asyncio.run(t.setup(trace, rt))
    files = rt.files()
    draw = t.draw()
    truth = json.loads(files["/app/data/truth.json"][0])
    assert truth == draw.box_truth and len(truth["stars"]) == 12
    assert {p for p in files if p.startswith("/app/data/stars/")} == {
        f"/app/data/stars/{sid}.npz" for sid in draw.box_files}
    assert rt.commands[0][2].startswith("rm -rf -- /app/solution /app/transitbench /app/data && ")

    sets = t.hidden_sets()
    assert [s.name for s in sets] == ["mini"] and set(sets[0].stars) == set(draw.hidden_files)
    staged(t, sets, trace)
    trace.info["transitbench"]["solution"]["sha256"] = "00" * 32
    assert t.score_records(trace, oracle_records(sets)) == 1.0
    grade = trace.info["transitbench"]["grade"]
    assert grade["reward"] == 1.0 and grade["false"] == 0 and grade["inputs"]["hidden_digest"] == draw.digest
    assert set(grade["stars"]) == set(draw.hidden_files)
    assert {v["role"] for v in grade["stars"].values()} <= {"inj_random", "inj_tce_only", "null_random",
                                                             "null_tce_only", "eb"}
    assert trace.metrics["credit"] == grade["credit"] >= 8 and trace.metrics["solution_changed"] == 1.0


@pytest.mark.skipif(not (PRIVATE / "releases" / "r1").is_dir(), reason="needs the r1 release (data/private)")
def test_survey_r1_gold_is_scored_on_the_evaluator_only():
    t = survey_tasks(task=TransitTaskConfig(data_dir=DATA))[0]
    assert asyncio.run(t.validate(make_trace(t), None)) is True


def test_oracle_solution_prints_the_truth_rows(tmp_path):
    from transitbench import runner

    sets = [hidden.dev_set()]
    (tmp_path / "solution.py").write_bytes(task.oracle_solution(sets))
    built = DATA / "built" / "dev"
    oracle = task.oracle_rows(sets[0].truth)
    with_rows = [sid for sid, rows in oracle.items() if rows]
    for sid in with_rows[:2] + [next(sid for sid, rows in oracle.items() if not rows)]:
        expected = oracle[sid]
        path = built / f"{sid}.npz"
        if not path.is_file():
            pytest.skip("needs the built dev set")
        out = subprocess.run([sys.executable, str(tmp_path / "solution.py")], input=path.read_bytes(),
                             capture_output=True, check=True).stdout
        rows = runner.parse_rows(out.decode())
        assert [[r.period, r.t0, r.score] for r in rows] == expected


# ---- the eval configs ------------------------------------------------------------------------------------------------

EVAL_CONFIGS = sorted((ROOT / "configs").glob("*/*.toml"))


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_eval_toml_validates(path, tmp_path):
    """The eval configs are prime-rl `uv run eval` configs; their env block, selection and sampling are checked by
    `vf-eval --dry-run` on the equivalent vf-eval config."""
    import tomli_w

    name = f"{path.parent.name}-{path.stem}"
    config = tomllib.loads(path.read_text())
    (source,) = config["source"]
    assert source["name"] == source["env"]["taskset"]["id"]
    equivalent = {"num_rollouts": config["group_size"], "max_concurrent": config["concurrency"]["max_inflight"],
                  "select": config["select"], "sampling": config["sampling"], "env": source["env"]}
    (tmp_path / "eval.toml").write_text(tomli_w.dumps(equivalent))
    vf_eval = Path(sys.executable).parent / "vf-eval"
    result = subprocess.run([str(vf_eval), "@", str(tmp_path / "eval.toml"), "--dry-run", "--no-rich", "-o",
                             str(tmp_path / "out"), "--run.name", name], capture_output=True, text=True,
                            env={**os.environ, "NO_COLOR": "1"}, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    resolved = json.loads((tmp_path / "out" / name / "configs" / "resolved" / "eval.json").read_text())
    assert resolved["env"]["taskset"]["id"] == source["env"]["taskset"]["id"]
    assert resolved["env"]["agent"]["max_turns"] == source["env"]["agent"]["max_turns"]
    assert resolved["env"]["agent"]["max_total_tokens"] == source["env"]["agent"]["max_total_tokens"]
    assert resolved["env"]["agent"]["runtime"]["labels"] == source["env"]["agent"]["runtime"]["labels"]


# ---- Docker: finalize in a real box ----------------------------------------------------------------------------------

DOCKER_IMAGE = "transitbench-base:local"


def docker_skip() -> str | None:
    if shutil.which("docker") is None:
        return "needs docker"
    if subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE], capture_output=True, check=False).returncode != 0:
        return f"needs the {DOCKER_IMAGE} image (built as in image/Dockerfile)"
    return None


@pytest.mark.skipif(docker_skip() is not None, reason=str(docker_skip()))
def test_docker_finalize_stops_the_command_that_outlives_the_agent(tmp_path):
    """After an agent timeout the chat program's last command still runs. finalize kills it, its background job and a
    detached daemon before the capture, so solution/ stays as captured; verifiers' own processes are left alone."""
    from verifiers.v1.runtimes import provision_runtime

    from transitbench_env import harness

    t = survey_tasks(task=TransitTaskConfig(data_dir=tmp_path))[1]
    trace = make_trace(t)
    sol = "/app/solution"
    command = (f"(sleep 4; echo bg > {sol}/bg.py) >/dev/null 2>&1 & "
               f"setsid sh -c '(sleep 4; echo daemon > {sol}/daemon.py) &' </dev/null >/dev/null 2>&1; "
               f"touch /tmp/started; sleep 4; echo late > {sol}/solution.py")
    wait = "for i in $(seq 100); do test -e /tmp/started && exit 0; sleep 0.1; done; exit 1"

    async def main():
        config = vf.DockerConfig(image=DOCKER_IMAGE, workdir="/app", cpu=1, memory=1)
        async with asyncio.timeout(300), provision_runtime(config) as rt:
            await rt.write(f"{sol}/solution.py", b"print(1)\n")
            # The chat program as verifiers runs it (<scripts dir>/<sha256>.py), reduced to one bash call.
            chat = f"{rt.backend.scripts_dir}/{harness.PROGRAM_SHA256}.py"
            await rt.write(chat, f"import subprocess, sys\n{harness.RUN_BASH}\nrun_bash(sys.argv[1])\n".encode())
            await rt.run_background(["python3", chat, command], {harness.COMMAND_TIMEOUT_ENV: "60"}, "/tmp/chat.log")
            await rt.run_background(["sleep", "60"], {}, "/tmp/bystander.log")
            assert (await rt.run(["sh", "-c", wait], {})).exit_code == 0
            await t.finalize(trace, rt)
            await asyncio.sleep(6)
            after = await grading.capture_solution(rt, max_bytes=1 << 20, max_files=256)
            ps = await rt.run(["ps", "-e", "-o", "stat=,args="], {})
            return after, ps.stdout

    after, ps = asyncio.run(main())
    captured = trace.info["transitbench"]["solution"]
    assert [f["path"] for f in captured["files"]] == ["solution.py"]
    assert after.sha256 == captured["sha256"] and task.solution_tree(after)[0] == {"solution.py": b"print(1)\n"}
    live = [line.split(None, 1)[1] for line in ps.splitlines() if not line.lstrip().startswith("Z")]
    assert "sleep 60" in live                                       # exec'd by verifiers (parent 0): left alone
    assert not [a for a in live if "sleep 4" in a or harness.PROGRAM_SHA256 in a], live
