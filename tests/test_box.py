"""The agent box's files, TASK.md and put_files (transitbench_env/box.py)."""

import asyncio
import copy
import io
import json
import resource
import tarfile
from pathlib import Path

import pytest

from transitbench import runner, score
from transitbench_env import box

REPO = Path(__file__).resolve().parents[1]
LIBRARIES = {"python": "3.12", "numpy": "2.5.4", "scipy": "1.18.1", "astropy": "8.0.2", "scikit-learn": "1.9.1"}
SURVEY = {"preset": "survey", "star_cap_s": 60.0, "command_cap_s": 940.0, "workers": 16, "max_turns": 60,
          "max_tokens": 200_000, "wall_s": 14_400, "solution_max_bytes": 1 << 20, "solution_max_files": 256,
          "stdout_max_bytes": 1 << 16, "memory_bytes": 2 << 30, "hidden": {"A": 1000, "B": 1000, "dev": 196},
          "labelled": 196, "libraries": LIBRARIES}
MINI = {**SURVEY, "preset": "mini", "star_cap_s": 20.0, "command_cap_s": 75.0, "workers": 4, "max_turns": 12,
        "max_tokens": 32_768, "wall_s": 3600, "hidden": {"inj": 8, "null": 5, "eb": 3}, "labelled": 12}
PRESETS = {"survey": SURVEY, "mini": MINI}


def changed(value):
    """A different value of the same shape."""
    if isinstance(value, dict):
        key = min(value)
        return {**value, key: changed(value[key])}
    if isinstance(value, str):
        return value + ".1"
    return value * 2 + 1


def test_caps_keys_match_the_design():
    assert set(box.CAPS_KEYS) == set(SURVEY) == set(MINI)


@pytest.mark.parametrize("preset", PRESETS)
def test_task_md_states_every_cap(preset):
    caps = PRESETS[preset]
    text = box.render_task_md(caps)
    assert "{" not in text and "}" not in text and "<!--" not in text
    for key in box.CAPS_KEYS:
        other = copy.deepcopy(caps)
        if key == "preset":
            other["preset"] = "mini" if preset == "survey" else "survey"
            other["hidden"] = PRESETS[other["preset"]]["hidden"]
        else:
            other[key] = changed(caps[key])
        assert box.render_task_md(other) != text, f"TASK.md does not show caps[{key!r}]"
    for sub, n in caps["hidden"].items():
        other = copy.deepcopy(caps)
        other["hidden"][sub] = n + 1
        assert box.render_task_md(other) != text, f"TASK.md does not show hidden[{sub!r}]"


@pytest.mark.parametrize("preset", PRESETS)
def test_task_md_values(preset):
    caps = PRESETS[preset]
    text = box.render_task_md(caps)
    cap = int(caps["star_cap_s"])
    for phrase in (f"{cap} s of wall time", f"{cap + 2} s per process", f"after {int(caps['command_cap_s'])} s",
                   "2 GiB of memory", f"{caps['workers']} stars at a time", f"{caps['max_turns']} turns",
                   f"{caps['max_tokens']:,} tokens", "1 MiB of file content", "256 files and directories",
                   "More than 64 KiB of stdout", f"Only the {score.MAX_ROWS}", f"{runner.NPROC} processes",
                   "numpy 2.5.4", "scikit-learn 1.9.1", f"{caps['labelled']} labelled stars",
                   "/tmp` is not writable", "Background\n  processes end with the command"):
        assert phrase in text, phrase
    if preset == "survey":
        assert "after 4 h" in text and "S on sample A" in text and "A (1,000 stars)" in text
        assert "## Reward" not in text
    else:
        assert "after 1 h" in text and "/ 8, -1, 1)" in text and "on 16 hidden stars" in text
        assert "8 INJ" in text and "5 NULL" in text and "3 EB" in text and "## Score" not in text


def test_task_md_cpu_backstop_is_the_runners():
    """The CPU limit TASK.md states is the runner's RLIMIT_CPU soft limit."""
    cfg = runner._config(SURVEY["star_cap_s"], "none", 1)
    soft = {res: s for res, s, _ in runner._limits(cfg)}[resource.RLIMIT_CPU]
    assert f"{soft} s per process" in box.render_task_md(SURVEY)


@pytest.mark.parametrize("change", [{"extra": 1}, {"preset": "other"}])
def test_bad_caps_are_refused(change):
    with pytest.raises(ValueError):
        box.render_task_md({**SURVEY, **change})
    caps = dict(SURVEY)
    del caps["wall_s"]
    with pytest.raises(ValueError):
        box.render_task_md(caps)


def test_science_files_are_the_installed_package():
    files = box.science_files()
    assert set(files) == {f"transitbench/{n}.py" for n in ("__init__", "match", "score", "runner", "archive",
                                                          "run_dev")}
    for rel, body in files.items():
        assert body == (REPO / rel).read_bytes()


def test_starter_reads_the_starter_file():
    assert box.starter() == {"solution.py": (REPO / "workspace" / "solution" / "solution.py").read_bytes()}


def test_box_files():
    prompt = box.render_task_md(MINI)
    solution = {"solution.py": b"print()\n", "lib/helper.py": b"x = 1\n"}
    data = {"truth.json": b"{}", "stars/m-00.npz": b"npz"}
    files = box.box_files(MINI, prompt, solution, data)
    assert files["/app/TASK.md"] == (prompt.encode(), 0o644)
    assert json.loads(files["/app/caps.json"][0]) == MINI
    shim, mode = files["/app/run_dev.py"]
    assert mode == 0o755 and b"from transitbench.run_dev import main" in shim
    assert files["/app/solution/lib/helper.py"] == (b"x = 1\n", 0o644)
    assert files["/app/data/stars/m-00.npz"] == (b"npz", 0o644)
    assert files["/app/transitbench/runner.py"][0] == (REPO / "transitbench/runner.py").read_bytes()
    assert len(files) == 3 + 6 + 2 + 2
    survey = box.box_files(SURVEY, "x", box.starter(), None)
    assert not any(p.startswith("/app/data/") for p in survey)


@pytest.mark.parametrize("rel", ["/etc/passwd", "../x", "a/../../x", "./x", "a//b", ""])
def test_box_files_refuse_unsafe_paths(rel):
    with pytest.raises(ValueError):
        box.box_files(SURVEY, "x", {rel: b""}, None)


class FakeRuntime:
    """Records put_files' calls: one write and one command, as root."""

    def __init__(self, exit_code=0):
        self.user, self.calls, self.exit_code = "agent", [], exit_code

    def with_user(self, user):
        twin = FakeRuntime(self.exit_code)
        twin.user, twin.calls = user, self.calls
        return twin

    async def write(self, path, data):
        self.calls.append(("write", self.user, path, data))

    async def run(self, argv, env):
        self.calls.append(("run", self.user, argv, env))
        return type("Result", (), {"exit_code": self.exit_code, "stdout": "", "stderr": "tar: boom"})()


def test_put_files_writes_one_tar_and_extracts_it_as_root():
    files = box.box_files(SURVEY, "prompt", {"solution.py": b"print(1)\n"}, None)
    rt = FakeRuntime()
    asyncio.run(box.put_files(rt, files, clear=("/app/solution", "/app/transitbench")))
    (kind, user, path, data), (kind2, user2, argv, env) = rt.calls
    assert (kind, kind2, user, user2) == ("write", "run", None, None)
    assert argv[:2] == ["sh", "-c"] and env == {}
    assert argv[2] == f"rm -rf -- /app/solution /app/transitbench && tar -xf {path} -C / && rm -f {path}"
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        members = {m.name: m for m in tar.getmembers()}
        assert {n for n, m in members.items() if m.isfile()} == {p.lstrip("/") for p in files}
        assert {n for n, m in members.items() if m.isdir()} == {"app", "app/solution", "app/transitbench"}
        assert all(m.uid == 0 and m.gid == 0 and m.mtime == 0 for m in members.values())
        assert members["app/run_dev.py"].mode == 0o755 and members["app/TASK.md"].mode == 0o644
        assert tar.extractfile("app/solution/solution.py").read() == b"print(1)\n"
    assert box.files_tar(files) == data


def test_put_files_raises_when_the_command_fails():
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(box.put_files(FakeRuntime(exit_code=2), {"/app/x": (b"", 0o644)}))
    with pytest.raises(ValueError):
        asyncio.run(box.put_files(FakeRuntime(), {"/app/x": (b"", 0o644)}, clear=("/",)))
