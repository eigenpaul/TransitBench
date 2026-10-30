"""The capped Bash harness (transitbench_env/harness.py).

The run_bash tests read RUN_BASH from the source file and need neither verifiers nor a box: they run on macOS, and
in Linux containers, where the PID-namespace tests also run when `unshare --pid` works (a privileged container):
  docker run --rm --privileged -v "$PWD":/repo:ro -w /repo transitbench-base:local \
    sh -c 'pip install -q pytest && python -m pytest -q -p no:cacheprovider tests/test_harness.py'
The other tests need verifiers (the `env` extra) and skip without it.
"""

import ast
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "transitbench_env" / "harness.py"
CAP = 2


def source_constant(name: str) -> str:
    """A string constant assigned at the top of harness.py, read without importing verifiers."""
    for node in ast.parse(HARNESS.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


@pytest.fixture
def run_bash(monkeypatch):
    """The program's run_bash, with a 2 s cap."""
    monkeypatch.setenv(source_constant("COMMAND_TIMEOUT_ENV"), str(CAP))
    namespace = {"subprocess": subprocess}
    exec(source_constant("RUN_BASH"), namespace)  # noqa: S102 - our own program text
    namespace["run_bash"].pid_namespace = namespace["PID_NAMESPACE"]
    return namespace["run_bash"]


def wait_and_check_absent(*markers: Path, seconds: float = 4.0) -> None:
    time.sleep(seconds)
    assert not [m.name for m in markers if m.exists()], "a process outlived its command"


def test_a_command_under_the_cap_returns_its_output(run_bash):
    assert run_bash("echo out; echo err >&2") == "out\nerr\n"
    assert run_bash("exit 3") == "(no output)"
    assert run_bash("printf '\\377ok'") == "�ok"


def test_a_command_over_the_cap_is_killed_with_verifiers_message(run_bash, tmp_path):
    marker = tmp_path / "late"
    command = f"echo partial; (sleep 3; touch {marker}) & sleep 30"
    start = time.monotonic()
    output = run_bash(command)
    assert time.monotonic() - start < CAP + 3
    # The words verifiers' own 3600 s limit gives (str of subprocess.TimeoutExpired), after the output so far.
    assert output == f"partial\nerror: Command '{['bash', '-c', command]}' timed out after {CAP} seconds"
    wait_and_check_absent(marker)          # the backgrounded subshell was killed with the command


@pytest.mark.skipif(shutil.which("unshare") is None or shutil.which("setsid") is None, reason="needs util-linux")
def test_nothing_a_command_starts_outlives_it(run_bash, tmp_path):
    if not run_bash.pid_namespace:
        pytest.skip("unshare --pid does not work here (needs CAP_SYS_ADMIN, e.g. docker run --privileged)")
    background, daemon = tmp_path / "background", tmp_path / "daemon"
    start = time.monotonic()
    output = run_bash(f"(sleep 3; touch {background}) >/dev/null 2>&1 & "
                      f"setsid sh -c 'sleep 3; touch {daemon}' >/dev/null 2>&1 </dev/null & echo started")
    assert output == "started\n" and time.monotonic() - start < 1
    wait_and_check_absent(background, daemon)
    # The same at the cap: a setsid daemon escapes the process group, but not the namespace.
    late = tmp_path / "late"
    output = run_bash(f"setsid sh -c 'sleep 3; touch {late}' >/dev/null 2>&1 </dev/null & sleep 30")
    assert output.startswith("error: Command") and output.endswith(f"timed out after {CAP} seconds")
    wait_and_check_absent(late)
    if shutil.which("ps"):
        assert run_bash("ps -e -o pid= | wc -l").strip() in ("2", "3")   # bash, ps and wc: their own namespace


def capped_program_namespace(monkeypatch, cap: int) -> dict:
    """The program's RUN_BASH globals, run_bash among them, with a cap of `cap` seconds."""
    monkeypatch.setenv(source_constant("COMMAND_TIMEOUT_ENV"), str(cap))
    namespace = {"subprocess": subprocess}
    exec(source_constant("RUN_BASH"), namespace)  # noqa: S102 - our own program text
    return namespace


def test_endless_output_kills_the_command_and_keeps_head_and_tail(monkeypatch, tmp_path):
    program = capped_program_namespace(monkeypatch, 5)
    limit, keep = program["OUTPUT_MAX_BYTES"], program["OUTPUT_KEEP_BYTES"]
    marker = tmp_path / "after"
    start = time.monotonic()
    output = program["run_bash"](f"yes; touch {marker}")
    assert time.monotonic() - start < 5                     # stopped by the output budget, not by the cap
    notice = f"error: output exceeded {limit} bytes; command killed"
    assert output.startswith("y\ny\n") and output.endswith("\n" + notice)
    assert 2 * keep + len(notice) <= len(output) <= 2 * keep + len(notice) + 1
    assert not marker.exists()                              # the command was killed, not only cut


def test_long_output_under_the_budget_keeps_head_and_tail(monkeypatch):
    program = capped_program_namespace(monkeypatch, 30)
    keep = program["OUTPUT_KEEP_BYTES"]
    output = program["run_bash"]("echo first; head -c 3000000 /dev/zero | tr '\\0' a; echo; echo last >&2")
    assert output.startswith("first\naaa") and output.endswith("aaa\nlast\n")
    assert len(output) == 2 * keep + len("last\n")


def stop_agent_constant(name: str):
    """A constant of the STOP_AGENT script, read without running it."""
    for node in ast.parse(source_constant("STOP_AGENT")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def test_stop_agent_finds_the_commands_run_bash_starts(monkeypatch):
    program = capped_program_namespace(monkeypatch, 5)
    assert [a.decode() for a in stop_agent_constant("UNSHARE")] == program["_PID_NAMESPACE"] + ["bash", "-c"]


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="reads /proc (Linux)")
def test_stop_agent_kills_the_program_and_its_command_namespace(monkeypatch, tmp_path):
    """As on Prime: the chat program's last command, in its PID namespace, still runs when the agent stops."""
    program = capped_program_namespace(monkeypatch, 60)
    if not program["PID_NAMESPACE"]:
        pytest.skip("unshare --pid does not work here (needs CAP_SYS_ADMIN, e.g. docker run --privileged)")
    digest = "ab" * 32
    chat = tmp_path / "scripts" / f"{digest}.py"
    chat.parent.mkdir()
    chat.write_text("import subprocess, sys\n" + source_constant("RUN_BASH") + "\nrun_bash(sys.argv[1])\n")
    started, late, daemon = tmp_path / "started", tmp_path / "late", tmp_path / "daemon"
    command = (f"setsid sh -c '(sleep 3; touch {daemon}) &' </dev/null >/dev/null 2>&1; touch {started}; "
               f"sleep 3; touch {late}")
    agent = subprocess.Popen([sys.executable, str(chat), command], env={**os.environ, "TB_COMMAND_TIMEOUT": "60"})
    bystander = subprocess.Popen(["sleep", "30"])
    try:
        for _ in range(100):
            if started.exists():
                break
            time.sleep(0.05)
        stop = subprocess.run([sys.executable, "-I", "-S", "-c", source_constant("STOP_AGENT"), digest, "0"],
                              capture_output=True, text=True, timeout=60, check=True)
        report = json.loads(stop.stdout)
        assert report["left"] == 0 and report["killed"] >= 4        # python, unshare, bash, sleep, the daemon
        assert agent.wait(timeout=5) == -signal.SIGKILL
        wait_and_check_absent(late, daemon)
        assert bystander.poll() is None                             # not the agent's: left alone
    finally:
        for process in (agent, bystander):
            process.kill()
            process.wait()


# ---- with verifiers --------------------------------------------------------------------------------------------------

@pytest.fixture
def harness():
    pytest.importorskip("verifiers", reason="needs verifiers (the env extra)")
    from transitbench_env import harness
    return harness


def test_capped_program_is_verifiers_program_with_run_bash_inserted(harness):
    from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE, PEP_723_END

    program = harness.capped_program()
    assert program == harness.PROGRAM
    compile(program, "program", "exec")
    assert program.replace(harness.RUN_BASH, "", 1) == CHAT_PROGRAM_SOURCE
    assert program.startswith(CHAT_PROGRAM_SOURCE.split(PEP_723_END)[0] + PEP_723_END)
    assert program.index(harness.RUN_BASH) + len(harness.RUN_BASH) == program.index(harness.ENTRY_POINT)
    assert "timeout=3600" in CHAT_PROGRAM_SOURCE


@pytest.mark.parametrize("change", ["no_entry", "two_entries", "no_run_bash", "two_run_bash", "no_metadata"])
def test_capped_program_refuses_a_changed_shape(harness, change):
    from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE, PEP_723_END

    base = {
        "no_entry": CHAT_PROGRAM_SOURCE.replace(harness.ENTRY_POINT, "\nif True:\n"),
        "two_entries": CHAT_PROGRAM_SOURCE + harness.ENTRY_POINT + "    pass\n",
        "no_run_bash": CHAT_PROGRAM_SOURCE.replace("def run_bash(", "def run_shell("),
        "two_run_bash": CHAT_PROGRAM_SOURCE.replace("def run_edit(", "def run_bash(c):\n    pass\n\n\ndef run_edit("),
        "no_metadata": CHAT_PROGRAM_SOURCE.replace(PEP_723_END, ""),
    }[change]
    with pytest.raises(RuntimeError, match="changed shape"):
        harness.capped_program(base)


def test_the_program_dispatches_to_the_capped_run_bash(harness, monkeypatch):
    """Executed as a module (not __main__), the bundled program's run_bash is ours, with the cap from the env."""
    pytest.importorskip("openai")
    monkeypatch.setenv(harness.COMMAND_TIMEOUT_ENV, "7")
    namespace = {"__name__": "transitbench_program"}
    exec(compile(harness.PROGRAM, "program", "exec"), namespace)  # noqa: S102 - our own program text
    assert namespace["COMMAND_TIMEOUT_SECONDS"] == 7
    assert "PID_NAMESPACE" in namespace["run_bash"].__code__.co_names
    assert namespace["run_chat_loop"].__globals__ is namespace
    assert namespace["run_bash"]("echo hi") == "hi\n"


def test_config_and_harness_classes(harness):
    import pydantic
    from verifiers.v1.configs.harness import HarnessConfig
    from verifiers.v1.utils.generic import concrete_type

    config = harness.TransitBashConfig()
    assert config.compaction is None and config.search is False and config.edit is True
    with pytest.raises(pydantic.ValidationError):
        harness.TransitBashConfig(search=True)
    assert harness.TransitBashConfig(compaction={"summarize_at_tokens": 1000}).compaction.summarize_at_tokens == 1000
    # verifiers resolves a harness's config through its MRO (utils/loaders.py: harness_config_type).
    assert concrete_type(harness.TransitBashHarness, HarnessConfig) is harness.TransitBashConfig
    assert "__init__" not in vars(harness.TransitBashHarness)
    assert harness.TransitBashHarness.APPENDS_SYSTEM_PROMPT and harness.TransitBashHarness.SUPPORTS_RESUME


def capture_launches(monkeypatch, *modules) -> list[dict]:
    import verifiers.v1 as vf

    calls = []

    async def fake_launch(source, config, ctx, trace, runtime, endpoint, secret, mcp_urls, system_prompt, prompt,
                          **kw):
        calls.append({"source": source, "system_prompt": system_prompt, "prompt": prompt, "endpoint": endpoint,
                      "secret": secret, "mcp_urls": mcp_urls, **kw})
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")

    for module in modules:
        monkeypatch.setattr(module, "launch_chat_program", fake_launch)
    return calls


def test_launch_is_the_bash_harness_with_the_program_and_the_cap(harness, monkeypatch):
    import verifiers.v1 as vf
    from verifiers.v1.harnesses.bash import harness as bash_module
    from verifiers.v1.harnesses.bash.harness import BashHarness, BashHarnessConfig
    from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE

    class Data(vf.TaskData):
        command_cap_s: float = 75.0

    calls = capture_launches(monkeypatch, bash_module, harness)
    data = Data(id="t", prompt="solve it", system_prompt="be careful", workdir="/app")
    args = (None, None, None, "http://endpoint", "secret", {}, data)
    plain_config = BashHarnessConfig(compaction=None, env={"A": "1"})
    asyncio.run(BashHarness(plain_config).launch(*args, tool_interception_url="http://gate"))
    config = harness.TransitBashConfig(env={"A": "1", harness.COMMAND_TIMEOUT_ENV: "9"})
    asyncio.run(harness.TransitBashHarness(config).launch(*args, tool_interception_url="http://gate"))
    plain, capped = calls
    assert plain.pop("source") == CHAT_PROGRAM_SOURCE and capped.pop("source") == harness.PROGRAM
    assert plain.pop("env") == {"A": "1"}
    assert capped.pop("env") == {"A": "1", harness.COMMAND_TIMEOUT_ENV: "75"}
    assert plain == capped
    assert capped["extra_args"] == ["--bash", "--tool-interception-url=http://gate", "--edit"]
    assert capped["activate"] is False

    asyncio.run(harness.TransitBashHarness(harness.TransitBashConfig()).launch(
        None, None, None, "e", "s", {}, vf.TaskData(prompt="p")))
    assert calls[-1]["env"] == {harness.COMMAND_TIMEOUT_ENV: "3600"}


def test_setup_prepares_the_program(harness):
    prepared = []

    class Runtime:
        async def prepare_uv_script(self, script, env=None, *, activate=True):
            prepared.append((script, env))
            return ["python", "program.py"]

    config = harness.TransitBashConfig(env={"A": "1"})
    asyncio.run(harness.TransitBashHarness(config).setup(Runtime()))
    assert prepared == [(harness.PROGRAM, {"A": "1"})]


def test_the_program_runs_as_its_digest(harness):
    """stop_agent finds the chat program by PROGRAM_SHA256 in the path verifiers' prepare_uv_script gives it."""
    import hashlib

    import verifiers.v1 as vf

    class Backend:
        SUPPORTS_USER_SWITCHING, type, scripts_dir = True, "stub", "/tmp/vf-scripts"
        cache = SimpleNamespace(uv_interpreters={}, uv_locks={}, curl_ready=True)

        async def write(self, runtime, path, data):
            pass

        async def run(self, runtime, argv, env):
            return SimpleNamespace(exit_code=0, stdout="/venv/bin/python\n", stderr="")

    argv = asyncio.run(vf.Runtime(backend=Backend()).prepare_uv_script(harness.PROGRAM, activate=False))
    assert argv == ["/venv/bin/python", f"/tmp/vf-scripts/{harness.PROGRAM_SHA256}.py"]
    assert harness.PROGRAM_SHA256 == hashlib.sha256(harness.PROGRAM.encode()).hexdigest()


def test_stop_agent_is_bounded_and_never_raises(harness):
    import verifiers.v1 as vf

    class Box:
        def __init__(self, config, stdout="", fail=False):
            self.config, self.stdout, self.fail, self.argv = config, stdout, fail, None

        def with_user(self, user):
            assert user is None
            return self

        async def run(self, argv, env):
            self.argv = argv
            if self.fail:
                raise RuntimeError("box gone")
            return SimpleNamespace(exit_code=0, stdout=self.stdout, stderr="")

    docker = Box(vf.DockerConfig(), stdout='{"killed": 3, "left": 0}\n')
    assert asyncio.run(harness.stop_agent(docker)) == {"killed": 3, "left": 0}
    assert docker.argv[:2] == ["timeout", str(harness.STOP_S)] and docker.argv[-2:] == [harness.PROGRAM_SHA256, "1"]
    prime = Box(vf.PrimeConfig())
    assert asyncio.run(harness.stop_agent(prime)) is None and prime.argv[-1] == "0"   # no orphan sweep on Prime
    assert asyncio.run(harness.stop_agent(Box(vf.DockerConfig(), fail=True))) is None


def test_the_harness_has_no_per_call_tool_gate():
    """The budget hook must not switch on verifiers' /tool round trip before every command (TransitTask.budget_notice)."""
    from transitbench_env.harness import TransitBashHarness
    assert TransitBashHarness.SUPPORTS_TOOL_INTERCEPTION is False
