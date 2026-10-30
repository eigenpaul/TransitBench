"""The agent's harness: verifiers' Bash harness with each command capped and in its own PID namespace.

verifiers' chat program runs every bash call through `run_bash`, which allows `subprocess.run(..., timeout=3600)`
(`verifiers/v1/harnesses/utils/core.py`). `capped_program` takes verifiers' own bundled program and redefines
`run_bash` just before its entry point; the chat loop looks `run_bash` up when it dispatches a call, so the
redefinition is the one that runs. The new `run_bash`:

- reads its cap in seconds from TB_COMMAND_TIMEOUT, which `launch` sets from `TaskData.command_cap_s`;
- runs `bash -c <command>` under `unshare --pid --fork --kill-child --mount-proc` when a probe at program start shows
  that works (it needs CAP_SYS_ADMIN: Prime VMs, not local Docker). The command's bash is then the init of its own
  PID namespace, so every process it started, daemons included, dies when it exits or is killed: nothing an agent
  command starts outlives the command;
- starts the command in its own session and, at the cap, kills that process group and returns the output so far
  followed by verifiers' own timeout message: `error: Command '['bash', '-c', '<command>']' timed out after N
  seconds`. Without the namespace, the kill reaches the process group only;
- reads stdout and stderr as they come and keeps the first and last OUTPUT_KEEP_BYTES (1 MiB) of each, so a command
  that prints without end cannot grow the chat program. Once the two together pass OUTPUT_MAX_BYTES (64 MiB) it kills
  the process group and appends `error: output exceeded 67108864 bytes; command killed`;
- returns `(no output)` for a command that printed nothing, so every real result can carry the task's budget line
  (an empty result is what verifiers' tool gate probes with; see `TransitTask.budget_notice`).

The harness has no per-call tool gate in the box (`SUPPORTS_TOOL_INTERCEPTION = False`); verifiers' evaluator-side
check of the proposed calls (`session.gate_tool_calls`) still runs.

Everything else is verifiers' Bash harness: prompts, flags, the edit tool and the 20,000-byte bound on each tool
result. Compaction is off by default and search cannot be turned on.

At a rollout deadline verifiers only cancels its own wait for the chat program (`verifiers/v1/rollout.py`; Prime's
`run` polls a background job), so the program and the command it is running go on in the box. The task's finalize
calls `stop_agent` before it captures /app/solution: it kills the chat program, each command's process tree (with
its PID namespace) and, in Docker, every process init adopted.

A taskset module that exports `TransitBashHarness` in `__all__` makes it the default harness for its id.
"""

import asyncio
import hashlib
import json
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.harnesses.bash.harness import (
    BASH_SYSTEM_PROMPT,
    EDIT_SYSTEM_PROMPT,
    BashHarness,
    BashHarnessConfig,
)
from verifiers.v1.harnesses.utils.launch import (
    CHAT_PROGRAM_SOURCE,
    PEP_723_END,
    CompactionConfig,
    compaction_args,
    launch_chat_program,
)

COMMAND_TIMEOUT_ENV = "TB_COMMAND_TIMEOUT"
DEFAULT_COMMAND_TIMEOUT = 3600
"""verifiers' own per-command timeout, used when the task data carries no command_cap_s."""
ENTRY_POINT = '\nif __name__ == "__main__":\n'

RUN_BASH = r'''
import os as _os
import selectors as _selectors
import signal as _signal
import time as _time

_PID_NAMESPACE = ["unshare", "--pid", "--fork", "--kill-child", "--mount-proc"]


def _pid_namespace():
    try:
        probe = subprocess.run(_PID_NAMESPACE + ["true"], capture_output=True, timeout=10, check=False)
    except Exception:
        return []
    return _PID_NAMESPACE if probe.returncode == 0 else []


COMMAND_TIMEOUT_SECONDS = int(_os.environ["TB_COMMAND_TIMEOUT"])
PID_NAMESPACE = _pid_namespace()
OUTPUT_MAX_BYTES = 64 << 20
OUTPUT_KEEP_BYTES = 1 << 20
NO_OUTPUT = "(no output)"
_DRAIN_SECONDS = 5


class _Stream:
    """One pipe's first and last OUTPUT_KEEP_BYTES, and how many bytes it carried."""

    def __init__(self):
        self.head, self.tail, self.total = bytearray(), bytearray(), 0

    def add(self, chunk):
        self.total += len(chunk)
        room = OUTPUT_KEEP_BYTES - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self.tail += chunk
            del self.tail[:-OUTPUT_KEEP_BYTES]

    def text(self):
        text = bytes(self.head + self.tail).decode("utf-8", errors="replace")
        return text.replace("\r\n", "\n").replace("\r", "\n")


def _kill(process):
    try:
        _os.killpg(process.pid, _signal.SIGKILL)
    except OSError:
        pass


def _collect(process, streams):
    """Read the command's pipes into streams until both close and the command exits. Kills its process group at the
    cap ("timeout", then reads for up to _DRAIN_SECONDS more) or once the output passes OUTPUT_MAX_BYTES ("overflow").
    Returns None, "timeout" or "overflow"."""
    deadline = _time.monotonic() + COMMAND_TIMEOUT_SECONDS
    stopped = None
    with _selectors.DefaultSelector() as selector:
        for pipe in streams:
            selector.register(pipe, _selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                if stopped is not None:
                    break
                _kill(process)
                stopped, deadline = "timeout", _time.monotonic() + _DRAIN_SECONDS
                continue
            for key, _ in selector.select(remaining):
                chunk = _os.read(key.fd, 1 << 16)
                if chunk:
                    streams[key.fileobj].add(chunk)
                else:
                    selector.unregister(key.fileobj)
            if stopped is None and sum(s.total for s in streams.values()) > OUTPUT_MAX_BYTES:
                _kill(process)
                stopped = "overflow"
                break
    if stopped is None:
        try:
            process.wait(timeout=max(0.0, deadline - _time.monotonic()))
            return None
        except subprocess.TimeoutExpired:
            _kill(process)
            stopped = "timeout"
    try:
        process.wait(timeout=_DRAIN_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    return stopped


def run_bash(command: str) -> str:
    try:
        process = subprocess.Popen(
            PID_NAMESPACE + ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as e:
        return f"error: {e}"
    out, err = _Stream(), _Stream()
    try:
        stopped = _collect(process, {process.stdout: out, process.stderr: err})
    except Exception as e:
        _kill(process)
        process.kill()
        process.wait()
        return f"error: {e}"
    finally:
        process.stdout.close()
        process.stderr.close()
    output = out.text() + err.text()
    if stopped is None:
        return output or NO_OUTPUT
    if output and not output.endswith("\n"):
        output += "\n"
    if stopped == "overflow":
        return output + f"error: output exceeded {OUTPUT_MAX_BYTES} bytes; command killed"
    expired = subprocess.TimeoutExpired(["bash", "-c", command], COMMAND_TIMEOUT_SECONDS)
    return output + f"error: {expired}"
'''


def capped_program(base: str = CHAT_PROGRAM_SOURCE) -> str:
    """verifiers' bundled chat program with RUN_BASH inserted just before its entry point.

    Refuses a program whose body does not have exactly one entry point and one `run_bash` definition, so a change in
    verifiers fails here rather than in a box.
    """
    metadata, sep, body = base.partition(PEP_723_END)
    if not sep or body.count(ENTRY_POINT) != 1 or body.count("def run_bash(") != 1:
        raise RuntimeError("the verifiers chat program changed shape; check transitbench_env.harness against it")
    head, tail = body.split(ENTRY_POINT)
    program = f"{metadata}{PEP_723_END}{head}{RUN_BASH}{ENTRY_POINT}{tail}"
    compile(program, "transitbench_chat_program", "exec")
    return program


PROGRAM = capped_program()
PROGRAM_SHA256 = hashlib.sha256(PROGRAM.encode()).hexdigest()
"""The chat program runs as <scripts dir>/<PROGRAM_SHA256>.py (verifiers' `Runtime.prepare_uv_script`)."""

STOP_AGENT = r'''
import json, os, signal, sys, time

SCRIPT = ("/" + sys.argv[1] + ".py").encode()
ORPHANS = sys.argv[2] == "1"
UNSHARE = [b"unshare", b"--pid", b"--fork", b"--kill-child", b"--mount-proc", b"bash", b"-c"]


def table():
    """pid -> (state, ppid, sid, argv) for every process."""
    procs = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as f:
                stat = f.read()
            with open(f"/proc/{name}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        fields = stat[stat.rindex(b")") + 2:].split()
        procs[int(name)] = (fields[0], int(fields[1]), int(fields[3]), argv)
    return procs


def agent(procs, keep):
    """The chat program, each command's unshare and, with ORPHANS, every process init adopted; then everything they
    started and every member of a session they lead."""
    hit = {pid for pid, (_, ppid, _, argv) in procs.items()
           if any(SCRIPT in a for a in argv) or argv[:len(UNSHARE)] == UNSHARE or (ORPHANS and ppid == 1)}
    while True:
        more = {pid for pid, (_, ppid, sid, _) in procs.items() if pid not in hit and (ppid in hit or sid in hit)}
        if not more:
            return hit - keep
        hit |= more


procs = table()
keep, pid = {1}, os.getpid()
while pid > 1 and pid not in keep:  # this script and its ancestors
    keep.add(pid)
    pid = procs[pid][1] if pid in procs else 0
stopped = set()
for _ in range(20):  # stop first, so that nothing forks between finding and killing
    procs = table()
    found = {pid for pid in agent(procs, keep) - stopped if procs[pid][0] not in (b"Z", b"X")}
    if not found:
        break
    for pid in found:
        try:
            os.kill(pid, signal.SIGSTOP)
        except OSError:
            pass
    stopped |= found
for pid in stopped:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
deadline, left = time.monotonic() + 5, stopped
while left and time.monotonic() < deadline:
    time.sleep(0.05)
    procs = table()
    left = {pid for pid in left if pid in procs and procs[pid][0] not in (b"Z", b"X")}
print(json.dumps({"killed": len(stopped), "left": len(left)}))
'''
STOP_S = 20


async def stop_agent(runtime: vf.Runtime) -> dict | None:
    """Kill everything the agent may still run in its box; the task's finalize calls it before the capture.

    Targets the chat program (by PROGRAM_SHA256 in its command line), each command's `unshare` (its --kill-child takes
    the command's PID namespace with it), everything those started and every member of a session they lead. In
    Docker it adds every process init adopted: verifiers execs its own processes there with parent 0, so those are
    what the agent's commands left behind, daemons included, when the box has no PID namespaces. Stops them all, then
    kills them and waits up to 5 s for them to go. Best effort and bounded: returns {"killed", "left"}, or None if the
    script could not run.
    """
    orphans = isinstance(runtime.config, vf.DockerConfig)
    argv = ["timeout", str(STOP_S), "python3", "-I", "-S", "-c", STOP_AGENT, PROGRAM_SHA256, str(int(orphans))]
    try:
        async with asyncio.timeout(STOP_S + 10):
            result = await runtime.with_user(None).run(argv, {})
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001 - the capture that follows records what the box holds either way
        return None


def command_timeout(data: vf.TaskData) -> int:
    """The per-command cap in whole seconds: the task's command_cap_s, else verifiers' 3600."""
    return int(getattr(data, "command_cap_s", DEFAULT_COMMAND_TIMEOUT))


class TransitBashConfig(BashHarnessConfig):
    compaction: CompactionConfig | None = None
    """Off by default; a TOML can still set a policy."""

    search: Literal[False] = False
    """The box has no network, so the search tool is never offered."""


class TransitBashHarness(BashHarness, vf.Harness[TransitBashConfig]):
    """The Bash harness running PROGRAM, with the command cap passed in TB_COMMAND_TIMEOUT."""

    SUPPORTS_TOOL_INTERCEPTION = False
    """No per-call tool gate. The task's only request hook (`TransitTask.budget_notice`) never denies a call, and the
    gate would add an HTTP round trip without retries before every command, so one failed request would fail the
    rollout. verifiers still runs the gate's check on the evaluator (`session.gate_tool_calls`)."""

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM, self.config.resolved_env)

    async def launch(
        self,
        ctx,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
        tool_interception_url: str | None = None,
    ) -> vf.ProgramResult:
        # BashHarness.launch without search, with PROGRAM and the command cap in the program's environment.
        system_prompt, prompt = self.resolve_prompt(data)
        fragments = [BASH_SYSTEM_PROMPT] + ([EDIT_SYSTEM_PROMPT] if self.config.edit else [])
        harness_prompt = " ".join(fragments)
        system_prompt = "\n\n".join(p for p in (harness_prompt, system_prompt) if p)
        # A resumed transcript without a task system prompt already holds the harness prompt.
        if (
            data.system_prompt is None
            and prompt is not None
            and not isinstance(prompt, str)
            and any(message.role == "system" and message.content == harness_prompt for message in prompt)
        ):
            system_prompt = None
        args = ["--bash"]
        if tool_interception_url:
            args.append(f"--tool-interception-url={tool_interception_url}")
        args += compaction_args(self.config.compaction)
        if self.config.edit:
            args.append("--edit")
        env = {**self.config.resolved_env, COMMAND_TIMEOUT_ENV: str(command_timeout(data))}
        return await launch_chat_program(
            PROGRAM,
            self.config,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            system_prompt,
            prompt,
            extra_args=args,
            env=env,
            activate=False,
        )
