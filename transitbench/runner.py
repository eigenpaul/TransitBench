"""Run a frozen solution once per star, offline, and collect its catalogue (plan section 7; docs/ENV_DESIGN.md 2.5).

Each star runs in its own process with the star's bytes on a stdin pipe, a fresh empty directory as cwd, HOME and
TMPDIR, a cleared environment (PYTHONHASHSEED=0, BLAS and OpenMP threads 1), a wall-clock cap, rlimits (a CPU
backstop, open files, file size, no core files) and one core. stdout is CSV rows of period_days,t0_days,score: past
stdout_max_bytes the star gets status stdout_overflow and no rows; otherwise its record keeps the 3 highest-scoring
valid rows, in the order score._grade uses. A crash or timeout leaves no rows.

Isolation "jail" (root on Linux with bubblewrap) runs each star in a bubblewrap sandbox as the slot's own user, with
new PID, network, IPC and UTS namespaces, an allowlist of read-only mounts (/usr, /etc and the solution at
/app/solution), a private size-capped /work and /dev/shm, an unwritable /tmp, a per-slot memory cgroup (else an
address-space rlimit) and a process cap. Nothing of a star outlives it: its namespace dies with bubblewrap, the slot
cgroup is killed, and any process still running as the slot user is killed and counted as a leak. "none" (macOS,
local Docker, non-root) is the same runner without the sandbox. "auto" uses the jail when it works. Every record says
which mode ran it.

Numpy and the standard library only: this file ships into agent and grading boxes.

CLI:
  python -m transitbench.runner --plan PLAN.json        results.jsonl and status.json in the plan's out dir
  python -m transitbench.runner --solution DIR --stars DIR --out FILE [--cap S] [--workers N] [--isolation MODE]
  python -m transitbench.runner --probe                 exit 0 iff the jail works and enforces its properties
"""

import argparse
import contextlib
import errno
import functools
import hashlib
import json
import math
import os
import platform
import queue
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transitbench.match import Candidate
from transitbench.score import MAX_ROWS

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
SOLUTION_ENV = {"LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
                **{v: "1" for v in THREAD_VARS}}
JAIL_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/work", "TMPDIR": "/work", **SOLUTION_ENV}
MODES = ("jail", "auto", "none")
ERRORS = ("hash", "spawn", "jail")
CAP_S, STDOUT_MAX_BYTES, MEMORY_BYTES = 60.0, 1 << 16, 2 << 30
NPROC, NOFILE, FSIZE = 64, 256, 64 << 20
SHM_BYTES, WORK_BYTES = 64 << 20, 256 << 20
SLOT_UID = 2000
JAIL_PYTHON = "/usr/local/bin/python3"
CGROUP_ROOT = Path("/sys/fs/cgroup")
IMAGE_FILE = Path("/etc/transitbench-image")
POLL_S, DRAIN_S, SWEEP_S, STATUS_EVERY_S = 0.01, 0.5, 2.0, 5.0
HIDDEN = ("/app/data", "/opt/tb-grade", "/root", "/logs", "/home", "/var")
PROBE_STDIN = b"transitbench-probe\n"
# Runs as root inside the jail, between bubblewrap and setpriv: bubblewrap returns when its command exits and never
# reaps its own init, so wait4 on bubblewrap sees none of the solution's CPU. The wrapper forks the solution (which
# does not keep the report fd), waits for it, and writes its children's CPU seconds to that fd.
CPU_WRAPPER = """import os, resource, signal, sys
fd = int(sys.argv[1])
pid = os.fork()
if pid == 0:
    os.close(fd)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    try:
        os.execv(sys.argv[2], sys.argv[2:])
    finally:
        os._exit(127)
code = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
use = resource.getrusage(resource.RUSAGE_CHILDREN)
os.write(fd, repr(use.ru_utime + use.ru_stime).encode())
os._exit(code if code >= 0 else 128 - code)
"""

Record = dict[str, Any]
"""One star: {"star", "status", "cpu_s", "wall_s", "rows", "n_rows", "n_invalid", "isolation", "slot", "leak"},
plus "stderr" (the last stderr_tail_bytes) when the plan asks for it. status is "ok", "timeout", "exit:<n>",
"signal:<n>", "oom" or "stdout_overflow"; rows are at most 3 [period, t0, score] lists, sorted by -score (stable);
n_rows counts the valid rows printed and n_invalid the other non-blank lines (both 0 unless status is "ok"); leak is
1 when a process of the star had to be killed after it ended (jail only)."""


class RunnerError(RuntimeError):
    """A failure of the run, not of a star; code is "hash", "spawn" or "jail", as status.json reports it."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class _Config:
    cap_s: float
    mode: str
    memory_bytes: int = MEMORY_BYTES
    stdout_max_bytes: int = STDOUT_MAX_BYTES
    stderr_tail: int = 0
    python: str = sys.executable
    cgroups: tuple[Path, ...] | None = None
    cpus: tuple[int, ...] | None = None


# ---- rows ------------------------------------------------------------------------------------------------------------

def parse_rows(text: str) -> list[Candidate]:
    """Every valid row of a star's stdout, in print order: three finite numbers with period > 0."""
    return _parse(text)[0]


def _parse(text: str) -> tuple[list[Candidate], int]:
    rows, invalid = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        row = _row(line)
        if row is None:
            invalid += 1
        else:
            rows.append(row)
    return rows, invalid


def _row(line: str) -> Candidate | None:
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None
    try:
        period, t0, score = (float(x) for x in parts)
    except ValueError:
        return None
    if all(math.isfinite(v) for v in (period, t0, score)) and period > 0:
        return Candidate(period=period, t0=t0, score=score)
    return None


# ---- one star --------------------------------------------------------------------------------------------------------

def _run_star(sid: str, data: bytes, solution: Path, cfg: _Config, slot: int) -> tuple[Record, bytes]:
    """Run the solution on one star's bytes in the given slot; the record and the raw stdout (empty on overflow)."""
    cgroup = cfg.cgroups[slot] if cfg.cgroups else None
    cpu0, oom0 = _cgroup_stats(cgroup)
    workdir = tempfile.TemporaryDirectory(prefix="tb-star-", ignore_cleanup_errors=True) if cfg.mode == "none" \
        else contextlib.nullcontext("/")
    report_r, report_w = os.pipe() if cfg.mode == "jail" else (None, None)
    with workdir as work, _closing(report_r):
        argv, env = _command(cfg, solution, work, slot, report_w)
        start = time.monotonic()
        try:
            proc = subprocess.Popen(argv, bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE if cfg.stderr_tail else subprocess.DEVNULL, cwd=work,
                                    env=env, start_new_session=True, preexec_fn=_preexec(cfg, slot, cgroup),
                                    pass_fds=() if report_w is None else (report_w,))
        except (OSError, subprocess.SubprocessError) as e:
            raise RunnerError("spawn", f"star {sid}: {e}") from None
        finally:
            if report_w is not None:
                os.close(report_w)
        pipes = _Pipes(proc, data, cfg.stdout_max_bytes, cfg.stderr_tail)
        timed_out, sampled = False, None
        try:
            while not _exited(proc.pid):
                if not timed_out and time.monotonic() - start >= cfg.cap_s:
                    timed_out = True
                    if cfg.mode == "jail":
                        sampled = _uid_cpu(SLOT_UID + slot)     # the wrapper dies unreported with the namespace
                    _killpg(proc.pid)
                pipes.pump(POLL_S)
            wall = time.monotonic() - start
            _killpg(proc.pid)                               # the group leader is still a zombie: no pgid reuse
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            pipes.drain(DRAIN_S)
            reported = _read_report(report_r)
        finally:
            if proc.returncode is None:
                _killpg(proc.pid)
                proc.wait()
            pipes.close()
    leak = _clean(cfg, slot, cgroup)
    cpu1, oom1 = _cgroup_stats(cgroup)
    if cgroup:
        cpu = cpu1 - cpu0
    elif reported is not None:
        cpu = reported
    elif sampled is not None:
        cpu = sampled
    else:
        cpu = usage.ru_utime + usage.ru_stime
    state = _status(proc.returncode, timed_out, oom1 > oom0, pipes.overflow, cfg.mode)
    out = bytes(pipes.out)
    rows, invalid = _parse(out.decode("utf-8", "replace")) if state == "ok" else ([], 0)
    rec = {"star": sid, "status": state, "cpu_s": round(cpu, 3), "wall_s": round(wall, 3),
           "rows": [[c.period, c.t0, c.score] for c in sorted(rows, key=lambda c: -c.score)[:MAX_ROWS]],
           "n_rows": len(rows), "n_invalid": invalid, "isolation": cfg.mode, "slot": slot, "leak": leak}
    if cfg.stderr_tail:
        rec["stderr"] = pipes.err.decode("utf-8", "replace")
    return rec, out


def _status(code: int, timed_out: bool, oom: bool, overflow: bool, mode: str) -> str:
    if timed_out:
        return "timeout"
    if oom:
        return "oom"
    if overflow:
        return "stdout_overflow"
    if code == 0:
        return "ok"
    if code < 0:
        return f"signal:{-code}"
    if mode == "jail" and 128 < code <= 128 + 64:  # bubblewrap reports a signal death of its child as 128 + n
        return f"signal:{code - 128}"
    return f"exit:{code}"


def _command(cfg: _Config, solution: Path, work: str, slot: int, report_fd: int | None = None
             ) -> tuple[list[str], dict]:
    """argv and environment for one star. In the jail, report_fd (when given) receives the solution's CPU seconds
    from CPU_WRAPPER."""
    if cfg.mode == "none":
        work = os.path.realpath(work)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": work, "TMPDIR": work, **SOLUTION_ENV}
        return [cfg.python, "-s", str(solution / "solution.py")], env
    uid = str(SLOT_UID + slot)
    argv = [_tool("bwrap"), "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try",
            "--die-with-parent", "--new-session", "--clearenv",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc", "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin",
            "--proc", "/proc", "--dev", "/dev",
            "--perms", "1777", "--size", str(SHM_BYTES), "--tmpfs", "/dev/shm", "--tmpfs", "/tmp",
            "--perms", "1777", "--size", str(WORK_BYTES), "--tmpfs", "/work",
            "--dir", "/app", "--ro-bind", str(solution), "/app/solution", "--chdir", "/work",
            *[a for k, v in JAIL_ENV.items() for a in ("--setenv", k, v)],
            "--cap-add", "CAP_SETUID", "--cap-add", "CAP_SETGID",
            *([] if report_fd is None else [JAIL_PYTHON, "-I", "-S", "-c", CPU_WRAPPER, str(report_fd)]),
            _tool("setpriv"), f"--reuid={uid}", f"--regid={uid}", "--clear-groups", "--no-new-privs",
            JAIL_PYTHON, "-s", "/app/solution/solution.py"]
    return argv, {"PATH": "/usr/bin:/bin"}


def _preexec(cfg: _Config, slot: int, cgroup: Path | None) -> Callable[[], None]:
    """Runs in the child before exec, as the runner's user: only os-level calls, safe after fork in a threaded parent."""
    cpu = cfg.cpus[slot % len(cfg.cpus)] if cfg.cpus else None
    limits = _limits(cfg)
    procs = str(cgroup / "cgroup.procs") if cgroup else None

    def run():
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})
        for res, soft, hard in limits:
            resource.setrlimit(res, (soft, hard))
        if sys.platform == "linux":
            with contextlib.suppress(OSError):
                _write_raw("/proc/self/oom_score_adj", b"1000")
        if procs:
            _write_raw(procs, b"0")
    return run


def _limits(cfg: _Config) -> list[tuple[int, int, int]]:
    cap = math.ceil(cfg.cap_s)
    want = [(resource.RLIMIT_CPU, cap + 2, cap + 5), (resource.RLIMIT_NOFILE, NOFILE, NOFILE),
            (resource.RLIMIT_FSIZE, FSIZE, FSIZE), (resource.RLIMIT_CORE, 0, 0)]
    if cfg.mode == "jail":
        want.append((resource.RLIMIT_NPROC, NPROC, NPROC))  # per uid, and each slot has its own
    if cfg.cgroups is None and sys.platform == "linux":
        want.append((resource.RLIMIT_AS, cfg.memory_bytes, cfg.memory_bytes))
    out = []
    for res, soft, hard in want:
        current = resource.getrlimit(res)[1]
        if current != resource.RLIM_INFINITY:
            hard = min(hard, current)
        out.append((res, min(soft, hard), hard))
    return out


def _write_raw(path: str, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


@contextlib.contextmanager
def _closing(fd: int | None):
    try:
        yield fd
    finally:
        if fd is not None:
            os.close(fd)


def _read_report(fd: int | None) -> float | None:
    """CPU seconds CPU_WRAPPER wrote, or None (no jail, or the wrapper was killed)."""
    if fd is None:
        return None
    os.set_blocking(fd, False)
    try:
        value = float(os.read(fd, 64))
    except (OSError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _exited(pid: int) -> bool:
    """Whether the child has exited, without reaping it."""
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _killpg(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


class _Pipes:
    """Feeds the star to stdin and drains stdout (capped) and stderr (a tail) without blocking."""

    def __init__(self, proc: subprocess.Popen, data: bytes, stdout_max: int, stderr_tail: int):
        self.proc, self.data, self.pos = proc, memoryview(data), 0
        self.stdout_max, self.stderr_tail = stdout_max, stderr_tail
        self.out, self.err, self.overflow = bytearray(), bytearray(), False
        self.sel = selectors.DefaultSelector()
        for f, event in ((proc.stdin, selectors.EVENT_WRITE), (proc.stdout, selectors.EVENT_READ),
                         (proc.stderr, selectors.EVENT_READ)):
            if f is not None:
                os.set_blocking(f.fileno(), False)
                self.sel.register(f, event)
        if not data:
            self._done(proc.stdin)

    def pump(self, timeout: float) -> None:
        if not self.sel.get_map():
            time.sleep(timeout)
            return
        for key, _ in self.sel.select(timeout):
            if key.fileobj is self.proc.stdin:
                self._write(key.fileobj)
            else:
                self._read(key.fileobj)

    def drain(self, grace: float) -> None:
        """After the process exits: read what is left until EOF, or until grace runs out (an escaped holder)."""
        if self.proc.stdin in self._open():
            self._done(self.proc.stdin)
        deadline = time.monotonic() + grace
        while self._open() and time.monotonic() < deadline:
            self.pump(POLL_S)

    def close(self) -> None:
        for f in self._open():
            self._done(f)
        self.sel.close()

    def _open(self) -> list:
        return [key.fileobj for key in self.sel.get_map().values()]

    def _write(self, f) -> None:
        try:
            n = os.write(f.fileno(), self.data[self.pos:self.pos + (1 << 16)])
        except BlockingIOError:
            return
        except OSError:
            self._done(f)
            return
        self.pos += n
        if self.pos >= len(self.data):
            self._done(f)

    def _read(self, f) -> None:
        try:
            chunk = os.read(f.fileno(), 1 << 16)
        except BlockingIOError:
            return
        except OSError:
            chunk = b""
        if not chunk:
            self._done(f)
        elif f is self.proc.stdout:
            if self.overflow or len(self.out) + len(chunk) > self.stdout_max:
                self.overflow = True
                self.out.clear()
            else:
                self.out += chunk
        else:
            self.err += chunk
            del self.err[:-self.stderr_tail]

    def _done(self, f) -> None:
        self.sel.unregister(f)
        f.close()


# ---- cleanup and cgroups ---------------------------------------------------------------------------------------------

def _clean(cfg: _Config, slot: int, cgroup: Path | None) -> int:
    """After a jailed star: kill its cgroup, then wait for every process of the slot user to end with the namespace;
    kill any left after SWEEP_S and return 1 (a leak)."""
    if cfg.mode != "jail":
        return 0
    if cgroup:
        _cgroup_kill(cgroup)
    uid = SLOT_UID + slot
    deadline = time.monotonic() + SWEEP_S
    while pids := _uid_pids(uid):
        if time.monotonic() >= deadline:
            for pid in pids:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
            _wait(lambda: not _uid_pids(uid), SWEEP_S)
            return 1
        time.sleep(POLL_S)
    return 0


def _uid_pids(uid: int) -> list[int]:
    """Live processes whose real, effective, saved or filesystem uid is uid."""
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/status") as f:
                fields = dict(line.split(":", 1) for line in f.read().splitlines() if ":" in line)
        except (OSError, ValueError):
            continue
        if fields.get("State", "").strip()[:1] in ("Z", "X"):
            continue
        if str(uid) in fields.get("Uid", "").split():
            pids.append(int(name))
    return pids


def _uid_cpu(uid: int) -> float:
    """CPU seconds of the uid's live processes, their reaped children included (utime to cstime in /proc/<pid>/stat)."""
    ticks = 0
    for pid in _uid_pids(uid):
        try:
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            ticks += sum(int(x) for x in fields[11:15])
        except (OSError, IndexError, ValueError):
            continue
    return ticks / os.sysconf("SC_CLK_TCK")


def _wait(done: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not done():
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_S)
    return True


def _cgroup_kill(group: Path) -> None:
    def empty() -> bool:
        try:
            procs = (group / "cgroup.procs").read_text().split()
        except OSError:
            return True
        if procs:
            try:
                (group / "cgroup.kill").write_text("1")
            except OSError:
                for pid in procs:
                    with contextlib.suppress(ProcessLookupError, ValueError):
                        os.kill(int(pid), signal.SIGKILL)
        return not procs
    _wait(empty, SWEEP_S)


def _cgroup_stats(group: Path | None) -> tuple[float, int]:
    """CPU seconds used so far and the OOM-kill count of a slot cgroup."""
    if group is None:
        return 0.0, 0
    return _flat(group / "cpu.stat").get("usage_usec", 0) / 1e6, _flat(group / "memory.events").get("oom_kill", 0)


def _flat(path: Path) -> dict[str, int]:
    try:
        return {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())}
    except (OSError, ValueError):
        return {}


def _cgroup_slots(n: int, memory_bytes: int) -> tuple[Path, ...] | None:
    """Slot groups tb/slotNN with memory.max = memory_bytes, no swap, pids.max = NPROC, the whole group killed on
    OOM and one CPU of bandwidth; None when cgroup v2 with memory and pids is not writable here.

    A namespaced cgroup root that holds processes cannot hand controllers to children (EBUSY), so its processes
    first move to the leaf tb-init, as container inits do.
    """
    if sys.platform != "linux":
        return None
    try:
        if not {"memory", "pids"} <= set((CGROUP_ROOT / "cgroup.controllers").read_text().split()):
            return None
        try:
            _enable(CGROUP_ROOT, "memory", "pids")
        except OSError as e:
            if e.errno != errno.EBUSY:
                raise
            _evacuate(CGROUP_ROOT, CGROUP_ROOT / "tb-init")
            _enable(CGROUP_ROOT, "memory", "pids")
        base = CGROUP_ROOT / "tb"
        base.mkdir(exist_ok=True)
        _enable(base, "memory", "pids")
        with contextlib.suppress(OSError):
            _enable(CGROUP_ROOT, "cpu")
            _enable(base, "cpu")
        groups = []
        for k in range(n):
            group = base / f"slot{k:02d}"
            group.mkdir(exist_ok=True)
            (group / "memory.max").write_text(str(memory_bytes))
            (group / "pids.max").write_text(str(NPROC))
            for name, value in (("memory.swap.max", "0"), ("memory.oom.group", "1"), ("cpu.max", "100000 100000")):
                if (group / name).exists():
                    (group / name).write_text(value)
            groups.append(group)
        return tuple(groups)
    except OSError:
        return None


def _enable(group: Path, *controllers: str) -> None:
    have = (group / "cgroup.subtree_control").read_text().split()
    missing = [c for c in controllers if c not in have]
    if missing:
        (group / "cgroup.subtree_control").write_text(" ".join(f"+{c}" for c in missing))


def _evacuate(root: Path, leaf: Path) -> None:
    leaf.mkdir(exist_ok=True)
    for _ in range(10):
        pids = (root / "cgroup.procs").read_text().split()
        if not pids:
            return
        for pid in pids:
            with contextlib.suppress(OSError):
                (leaf / "cgroup.procs").write_text(pid)


# ---- the jail --------------------------------------------------------------------------------------------------------

@functools.cache
def _tool(name: str) -> str:
    path = shutil.which(name, path="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin")
    if path is None:
        raise RunnerError("jail", f"{name} not found")
    return path


def _jail_missing() -> str:
    """Why the jail cannot run here, or an empty string."""
    if sys.platform != "linux" or os.geteuid() != 0:
        return "the jail needs root on Linux"
    try:
        bwrap, _ = _tool("bwrap"), _tool("setpriv")
    except RunnerError as e:
        return str(e)
    if not os.access(JAIL_PYTHON, os.X_OK):
        return f"{JAIL_PYTHON} not found"
    r = subprocess.run([bwrap, "--ro-bind", "/", "/", "true"], capture_output=True, timeout=30, check=False)
    return "" if r.returncode == 0 else f"bwrap --ro-bind / / true failed: {r.stderr.decode(errors='replace')[-300:]}"


PROBE_SOLUTION = r'''
import json, os, resource, socket, sys, time


def writable(d):
    try:
        p = os.path.join(d, ".tb-probe")
        with open(p, "w") as f:
            f.write("x")
        os.unlink(p)
        return True
    except (OSError, TypeError):
        return False


def network():
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=2):
            return True
    except OSError:
        return False


def setuid_root():
    try:
        os.setuid(0)
        return True
    except OSError:
        return False


stdin = sys.stdin.buffer.read()
status = dict(line.split(":", 1) for line in open("/proc/self/status").read().splitlines() if ":" in line)
report = {
    "uid": os.getuid(), "gid": os.getgid(), "groups": os.getgroups(), "cpus": len(os.sched_getaffinity(0)),
    "write": {"tmpdir": writable(os.environ.get("TMPDIR")), "home": writable(os.environ.get("HOME")),
              "tmp": writable("/tmp"), "shm": writable("/dev/shm"),
              "solution": writable(os.path.dirname(os.path.abspath(__file__)))},
    "visible": {p: os.path.exists(p) for p in HIDDEN},
    "network": network(), "pids": sum(p.isdigit() for p in os.listdir("/proc")), "setuid0": setuid_root(),
    "no_new_privs": int(status["NoNewPrivs"]), "cap_eff": int(status["CapEff"], 16),
    "stdin": os.readlink("/proc/self/fd/0"), "stdin_bytes": len(stdin), "env": sorted(os.environ),
    "nproc": resource.getrlimit(resource.RLIMIT_NPROC)[0],
}
if os.fork() == 0:            # a setsid daemon that must not outlive the star
    os.setsid()
    if os.fork() == 0:
        time.sleep(60)
    os._exit(0)
os.wait()
print(json.dumps(report))
'''.replace("HIDDEN", repr(HIDDEN))


def probe(cap_s: float = 30.0) -> dict:
    """Run a probe solution in the jail on slot 0 and check what docs/ENV_DESIGN.md 2.5 and gate 4 require: the slot
    user, one CPU, a writable TMPDIR and /dev/shm, no writable /tmp or solution, hidden paths, no network, a PID
    namespace, no privileges, stdin a pipe, a cleared environment, the process cap and no survivor."""
    report = {"ok": False, "platform": platform.platform(), "error": _jail_missing()}
    if report["error"]:
        return report
    report["bwrap"] = subprocess.run([_tool("bwrap"), "--version"], capture_output=True, text=True,
                                    check=False).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="tb-probe-") as d:
        solution = Path(d) / "solution"
        solution.mkdir()
        solution.chmod(0o755)
        (solution / "solution.py").write_text(PROBE_SOLUTION)
        (solution / "solution.py").chmod(0o644)
        cfg = _config(cap_s, "jail", 1)
        try:
            rec, out = _run_star("probe", PROBE_STDIN, solution, cfg, 0)
        except RunnerError as e:
            report["error"] = str(e)
            return report
    report.update(record=rec, memory="cgroup" if cfg.cgroups else "rlimit")
    try:
        seen = json.loads(out)
    except ValueError:
        report["error"] = "the probe solution printed no report"
        return report
    checks = {
        "status_ok": rec["status"] == "ok",
        "slot_user": seen["uid"] == SLOT_UID and seen["gid"] == SLOT_UID and seen["groups"] == [],
        "one_cpu": seen["cpus"] == 1,
        "tmpdir_writable": seen["write"]["tmpdir"] and seen["write"]["home"],
        "tmp_not_writable": not seen["write"]["tmp"],
        "shm_writable": seen["write"]["shm"],
        "solution_read_only": not seen["write"]["solution"],
        "paths_hidden": not any(seen["visible"].values()),
        "no_network": not seen["network"],
        "pid_namespace": seen["pids"] <= 3,
        "no_privileges": seen["cap_eff"] == 0 and seen["no_new_privs"] == 1 and not seen["setuid0"],
        "stdin_pipe": seen["stdin"].startswith("pipe:") and seen["stdin_bytes"] == len(PROBE_STDIN),
        "clean_env": set(seen["env"]) <= {*JAIL_ENV, "PWD"},          # bubblewrap sets PWD for --chdir
        "process_cap": seen["nproc"] == NPROC,
        "no_survivor": rec["leak"] == 0,
    }
    report.update(observed=seen, checks=checks, ok=all(checks.values()))
    return report


@functools.cache
def jail_works() -> bool:
    """True iff the jail runs here and enforces every property probe() checks (cached per process)."""
    return probe()["ok"]


def resolve_isolation(mode: str) -> str:
    """The mode a run will use: "jail" or "none". Raises RunnerError("jail") if "jail" is required but fails."""
    if mode not in MODES:
        raise ValueError(f"isolation must be one of {MODES}, got {mode!r}")
    if mode == "none":
        return "none"
    if jail_works():
        return "jail"
    if mode == "jail":
        raise RunnerError("jail", "the jail probe failed (python -m transitbench.runner --probe)")
    return "none"


# ---- many stars ------------------------------------------------------------------------------------------------------

def _config(cap_s: float, mode: str, workers: int, **kw) -> _Config:
    if workers < 1 or not cap_s > 0:
        raise ValueError(f"need workers >= 1 and cap_s > 0, got {workers} and {cap_s}")
    memory = kw.get("memory_bytes", MEMORY_BYTES)
    cgroups = _cgroup_slots(workers, memory) if mode == "jail" else None
    cpus = tuple(sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else None
    return _Config(float(cap_s), mode, cgroups=cgroups, cpus=cpus, **kw)


def _run_all(solution: Path, stars: list[tuple[str, Path, str | None]], cfg: _Config, workers: int,
             on_record: Callable[[Record], None] | None = None) -> list[Record]:
    """Run the stars in list order on `workers` slots; records in list order. A star's file is checked against its
    sha256, when given, as it is read."""
    solution = Path(solution).resolve()
    free = queue.SimpleQueue()
    for slot in range(workers):
        free.put(slot)
    lock = threading.Lock()
    records: list[Record] = [{}] * len(stars)

    def task(i: int) -> None:
        sid, path, sha = stars[i]
        try:
            data = Path(path).read_bytes()
        except OSError as e:
            raise RunnerError("hash", f"star {sid}: {e}") from None
        if sha is not None and hashlib.sha256(data).hexdigest() != sha:
            raise RunnerError("hash", f"star {sid}: sha256 mismatch")
        slot = free.get()
        try:
            records[i] = _run_star(sid, data, solution, cfg, slot)[0]
        finally:
            free.put(slot)
        if on_record is not None:
            with lock:
                on_record(records[i])

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(task, i) for i in range(len(stars))]
        try:
            for f in as_completed(futures):
                f.result()
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    return records


def check_plan(plan: dict) -> dict:
    """The plan with defaults filled in; raises ValueError on a malformed plan."""
    if plan.get("version") != 1:
        raise ValueError("plan version must be 1")
    p = {"memory_bytes": MEMORY_BYTES, "stdout_max_bytes": STDOUT_MAX_BYTES, "stderr_tail_bytes": 0, **plan}
    if p.get("isolation") not in MODES:
        raise ValueError(f"plan isolation must be one of {MODES}")
    if not (isinstance(p.get("solution"), str) and isinstance(p.get("stars"), list)):
        raise ValueError("plan needs a solution path and a list of stars")
    for key in ("cap_s", "workers", "memory_bytes", "stdout_max_bytes"):
        if not (isinstance(p.get(key), (int, float)) and not isinstance(p[key], bool) and p[key] > 0):
            raise ValueError(f"plan {key} must be a positive number")
    if not (isinstance(p["workers"], int) and isinstance(p["stderr_tail_bytes"], int) and p["stderr_tail_bytes"] >= 0):
        raise ValueError("plan workers and stderr_tail_bytes must be integers")
    ids = [s.get("id") if isinstance(s, dict) else None for s in p["stars"]]
    if not all(isinstance(s, dict) and isinstance(s.get("id"), str) and isinstance(s.get("path"), str)
               and isinstance(s.get("sha256"), str) for s in p["stars"]) or len(set(ids)) != len(ids):
        raise ValueError("plan stars need unique string ids, paths and sha256 values")
    return p


def run_plan(plan: dict, on_record: Callable[[Record], None] | None = None) -> list[Record]:
    """Run a plan (the plan.json format of docs/ENV_DESIGN.md 2.5): check every star's sha256 first, resolve the
    isolation, run the stars in list order and return their records in plan order. on_record gets each record as
    its star finishes, one call at a time. Raises RunnerError for a hash mismatch, a failed spawn or a missing jail.

    Optional plan keys: memory_bytes, stdout_max_bytes, stderr_tail_bytes (0: stderr goes to /dev/null)."""
    plan = check_plan(plan)
    stars = [(s["id"], Path(s["path"]), s["sha256"]) for s in plan["stars"]]
    for sid, path, sha in stars:
        if _sha256_file(path) != sha:
            raise RunnerError("hash", f"star {sid}: sha256 mismatch or unreadable")
    mode = resolve_isolation(plan["isolation"])
    cfg = _config(plan["cap_s"], mode, plan["workers"], memory_bytes=int(plan["memory_bytes"]),
                  stdout_max_bytes=int(plan["stdout_max_bytes"]), stderr_tail=plan["stderr_tail_bytes"])
    return _run_all(Path(plan["solution"]), stars, cfg, plan["workers"], on_record)


def run_star(solution: Path, star: Path, cap_seconds: float, python: str = sys.executable, *,
             isolation: str = "none") -> dict:
    """Run the solution on one star: its record, with rows as Candidates."""
    sid = Path(star).stem
    return run_catalogue(solution, {sid: star}, cap_seconds, 1, python, isolation=isolation)[sid]


def run_catalogue(solution: Path, stars: dict[str, Path], cap_seconds: float, workers: int,
                  python: str = sys.executable, *, isolation: str = "none") -> dict[str, dict]:
    """Run the solution on every star (star_id -> .npz path) on `workers` slots: records keyed by star, in input
    order, with rows as Candidates (the M1 and M2 scripts read them so)."""
    cfg = _config(cap_seconds, resolve_isolation(isolation), workers, python=python)
    records = _run_all(Path(solution), [(sid, Path(p), None) for sid, p in stars.items()], cfg, workers)
    return {r["star"]: {**r, "rows": [Candidate(*row) for row in r["rows"]]} for r in records}


def _sha256_file(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return None


# ---- plan CLI --------------------------------------------------------------------------------------------------------

class _Output:
    """out/results.jsonl, appended as stars finish, and out/status.json, rewritten atomically every STATUS_EVERY_S."""

    def __init__(self, out: Path, planned: int, isolation: str):
        out.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.dir, self.lock, self.sha = out, threading.Lock(), hashlib.sha256()
        self.results = os.fdopen(os.open(out / "results.jsonl", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb")
        self.status = {"version": 1, "done": False, "planned": planned, "written": 0, "beat": 0,
                       "isolation": isolation, "image": _image(), "runner_sha256": _runner_sha256(),
                       "results_sha256": None, "error": None}
        self.stop = threading.Event()
        self.update()
        self.thread = threading.Thread(target=self._beat, daemon=True)
        self.thread.start()

    def record(self, rec: Record) -> None:
        line = (json.dumps(rec, separators=(",", ":"), allow_nan=False) + "\n").encode()
        with self.lock:
            self.results.write(line)
            self.results.flush()
            self.sha.update(line)
            self.status["written"] += 1

    def update(self, **fields) -> None:
        with self.lock:
            self.status.update(fields)
            self.status["beat"] += 1
            tmp = self.dir / ".status.json.tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(self.status) + "\n")
            os.replace(tmp, self.dir / "status.json")

    def finish(self, error: str | None = None) -> None:
        self.stop.set()
        self.thread.join()
        with self.lock:
            self.results.close()
        self.update(done=True, error=error, results_sha256=None if error else self.sha.hexdigest())

    def _beat(self) -> None:
        while not self.stop.wait(STATUS_EVERY_S):
            self.update()


def _image() -> str | None:
    try:
        return IMAGE_FILE.read_text().strip()
    except OSError:
        return None


def _runner_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _main_plan(path: Path) -> int:
    plan = json.loads(Path(path).read_text())
    out = _Output(Path(plan["out"]), len(plan.get("stars") or []), str(plan.get("isolation")))
    try:
        plan = check_plan(plan)
        mode = resolve_isolation(plan["isolation"])
        out.update(isolation=mode)
        run_plan({**plan, "isolation": mode}, out.record)
    except RunnerError as e:
        print(f"runner: {e}", file=sys.stderr)
        out.finish(error=e.code)
        return 1
    except Exception:
        traceback.print_exc()
        out.finish(error="spawn")
        return 1
    out.finish()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a frozen solution once per star (docs/ENV_DESIGN.md 2.5).")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--plan", type=Path, help="plan.json; writes results.jsonl and status.json in its out dir")
    what.add_argument("--solution", type=Path, help="solution directory (with --stars and --out)")
    what.add_argument("--probe", action="store_true", help="check the jail; exit 0 iff it works")
    parser.add_argument("--stars", type=Path, help="directory of <star>.npz files")
    parser.add_argument("--out", type=Path, help="JSON lines file of records")
    parser.add_argument("--cap", type=float, default=CAP_S)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--isolation", choices=MODES, default="auto")
    args = parser.parse_args(argv)
    if args.probe:
        report = probe()
        print(json.dumps(report, indent=1))
        return 0 if report["ok"] else 1
    if args.plan:
        return _main_plan(args.plan)
    if args.stars is None or args.out is None:
        parser.error("--solution needs --stars and --out")
    paths = sorted(args.stars.glob("*.npz"))
    plan = {"version": 1, "solution": str(args.solution), "cap_s": args.cap, "workers": args.workers,
            "isolation": args.isolation,
            "stars": [{"id": p.stem, "path": str(p), "sha256": _sha256_file(p)} for p in paths]}
    with open(args.out, "w") as f:
        def write(rec: Record) -> None:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        run_plan(plan, write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
