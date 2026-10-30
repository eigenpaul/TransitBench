"""The agent box's public files, and put_files, which writes them into a runtime.

The box holds /app/TASK.md (the task prompt), /app/caps.json (every cap, built by the preset from TaskData),
/app/run_dev.py, the science package at /app/transitbench (the runner, archive, matcher and scorer that grading
uses, read from the installed package), /app/solution and, for mini, the labelled stars in /app/data. The survey
image already holds its dev set in /app/data.

TASK.md is rendered from templates/TASK.md. A block between `<!-- survey -->` (or `<!-- mini -->`) and
`<!-- end -->` lines is kept only for that preset; `{name}` fields come from the caps and from the runner's and
scorer's own constants, so the page states the limits the code enforces.

Evaluator side: this module never ships into a box.
"""

import importlib.resources
import io
import json
import math
import posixpath
import re
import shlex
import tarfile
import uuid
from collections.abc import Iterable
from pathlib import Path

from transitbench import runner, score

APP = "/app"
SCIENCE = ("__init__", "match", "score", "runner", "archive", "run_dev")
STARTER_FILES = ("solution.py",)
STARTER_REPO_DIR = "workspace/solution"
"""Where the starter lives in a repo checkout; the wheel force-includes it as transitbench_env/data/starter."""
CAPS_KEYS = ("preset", "star_cap_s", "command_cap_s", "workers", "max_turns", "max_tokens", "wall_s",
             "solution_max_bytes", "solution_max_files", "stdout_max_bytes", "memory_bytes", "hidden", "labelled",
             "libraries")
PRESETS = ("survey", "mini")
FILE_MODE, EXEC_MODE, DIR_MODE = 0o644, 0o755, 0o755
BLOCK = re.compile(r"^<!-- (survey|mini) -->\n(.*?)^<!-- end -->\n", re.MULTILINE | re.DOTALL)

RUN_DEV_SHIM = '''#!/usr/bin/env python3
"""Run solution/ on the labelled stars as grading runs it, and score it: python3 run_dev.py --help."""

import sys

from transitbench.run_dev import main

sys.exit(main())
'''

Files = dict[str, tuple[bytes, int]]
"""Absolute box path -> (content, mode)."""


# ---- TASK.md ---------------------------------------------------------------------------------------------------------

def check_caps(caps: dict) -> None:
    """Raise ValueError unless caps has exactly CAPS_KEYS and a known preset."""
    missing, extra = set(CAPS_KEYS) - set(caps), set(caps) - set(CAPS_KEYS)
    if missing or extra:
        raise ValueError(f"caps keys differ: missing {sorted(missing)}, unexpected {sorted(extra)}")
    if caps["preset"] not in PRESETS:
        raise ValueError(f"caps preset must be one of {PRESETS}, got {caps['preset']!r}")


def render_task_md(caps: dict, template: str | None = None) -> str:
    """TASK.md for these caps: the template's blocks for caps["preset"], with every field filled. template replaces
    the packaged template's text, so an earlier template's task hash can be recomputed."""
    check_caps(caps)
    text = template if template is not None else (
        importlib.resources.files("transitbench_env") / "templates" / "TASK.md").read_text(encoding="utf-8")
    text = BLOCK.sub(lambda m: m.group(2) if m.group(1) == caps["preset"] else "", text)
    return text.format_map(_fields(caps))


def _fields(caps: dict) -> dict[str, str]:
    cap = caps["star_cap_s"]
    hidden = {f"hidden_{k}": _count(v) for k, v in caps["hidden"].items()}
    return {
        "star_cap": _seconds(cap),
        "cpu_backstop": _seconds(math.ceil(cap) + 2),        # runner._limits: RLIMIT_CPU (cap + 2, cap + 5)
        "command_cap": _seconds(caps["command_cap_s"]),
        "workers": _count(caps["workers"]),
        "max_turns": _count(caps["max_turns"]),
        "max_tokens": _count(caps["max_tokens"]),
        "wall": _duration(caps["wall_s"]),
        "solution_max_bytes": _bytes(caps["solution_max_bytes"]),
        "solution_max_files": _count(caps["solution_max_files"]),
        "stdout_max_bytes": _bytes(caps["stdout_max_bytes"]),
        "memory": _bytes(caps["memory_bytes"]),
        "labelled": _count(caps["labelled"]),
        "libraries": ", ".join(f"{name} {version}" for name, version in caps["libraries"].items()),
        "hidden_total": _count(sum(caps["hidden"].values())),
        **hidden,
        "max_rows": _count(score.MAX_ROWS),
        "norm": _count(score.MINI_NORM),
        "nproc": _count(runner.NPROC),
        "nofile": _count(runner.NOFILE),
        "fsize": _bytes(runner.FSIZE),
        "shm": _bytes(runner.SHM_BYTES),
        "work": _bytes(runner.WORK_BYTES),
    }


def _count(n) -> str:
    return f"{int(n):,}" if float(n).is_integer() else f"{n:,}"


def _seconds(s) -> str:
    return f"{_count(s)} s"


def _duration(s) -> str:
    for unit, name in ((3600, "h"), (60, "min")):
        if s >= unit and s % unit == 0:
            return f"{_count(s // unit)} {name}"
    return _seconds(s)


def _bytes(n: int) -> str:
    for shift, name in ((30, "GiB"), (20, "MiB"), (10, "KiB")):
        if n >= 1 << shift and n % (1 << shift) == 0:
            return f"{n >> shift:,} {name}"
    return f"{n:,} bytes"


# ---- box files -------------------------------------------------------------------------------------------------------

def science_files() -> dict[str, bytes]:
    """transitbench/{__init__, match, score, runner, archive, run_dev}.py from the installed package."""
    package = importlib.resources.files("transitbench")
    return {f"transitbench/{name}.py": (package / f"{name}.py").read_bytes() for name in SCIENCE}


def starter() -> dict[str, bytes]:
    """The starter solution: transitbench_env/data/starter in a wheel, workspace/solution in a repo checkout."""
    packaged = importlib.resources.files("transitbench_env") / "data" / "starter"
    if all((packaged / name).is_file() for name in STARTER_FILES):
        return {name: (packaged / name).read_bytes() for name in STARTER_FILES}
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "pyproject.toml").is_file():
        raise FileNotFoundError("the starter solution is neither packaged nor in a repo checkout")
    return {name: (repo / STARTER_REPO_DIR / name).read_bytes() for name in STARTER_FILES}


def caps_json(caps: dict) -> bytes:
    check_caps(caps)
    return (json.dumps(caps, indent=1, sort_keys=True) + "\n").encode()


def box_files(caps: dict, prompt: str, solution: dict[str, bytes], data: dict[str, bytes] | None = None) -> Files:
    """Every file setup writes into the agent box: TASK.md (the prompt), caps.json, run_dev.py, the science package,
    solution/ (relative path -> bytes) and, when given, data/ (relative path -> bytes)."""
    files = {f"{APP}/TASK.md": (prompt.encode(), FILE_MODE), f"{APP}/caps.json": (caps_json(caps), FILE_MODE),
             f"{APP}/run_dev.py": (RUN_DEV_SHIM.encode(), EXEC_MODE)}
    files.update({f"{APP}/{rel}": (body, FILE_MODE) for rel, body in science_files().items()})
    for under, tree in (("solution", solution), ("data", data or {})):
        for rel, body in tree.items():
            files[f"{APP}/{under}/{_relative(rel)}"] = (bytes(body), FILE_MODE)
    return files


def _relative(rel: str) -> str:
    if not rel or rel.startswith("/") or posixpath.normpath(rel) != rel or rel.split("/")[0] == "..":
        raise ValueError(f"not a plain relative path: {rel!r}")
    return rel


def _absolute(path: str) -> str:
    if not path.startswith("/") or path == "/" or posixpath.normpath(path) != path or "\x00" in path:
        raise ValueError(f"not a normalised absolute path below /: {path!r}")
    return path


# ---- writing into a runtime ------------------------------------------------------------------------------------------

def files_tar(files: Files) -> bytes:
    """One deterministic tar of the files, rooted at /: their parent directories (mode 0755) and the files, owned by
    uid and gid 0, mtime 0."""
    paths = sorted(_absolute(p) for p in files)
    dirs = sorted({d for p in paths for d in _parents(p)} - set(paths))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for d in dirs:
            tar.addfile(_info(d, tarfile.DIRTYPE, DIR_MODE, 0))
        for p in paths:
            body, mode = files[p]
            tar.addfile(_info(p, tarfile.REGTYPE, mode, len(body)), io.BytesIO(body))
    return buf.getvalue()


def _parents(path: str) -> Iterable[str]:
    parent = posixpath.dirname(path)
    while parent != "/":
        yield parent
        parent = posixpath.dirname(parent)


def _info(path: str, kind: bytes, mode: int, size: int) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(path.lstrip("/"))
    ti.type, ti.mode, ti.size = kind, mode, size
    ti.uid = ti.gid = ti.mtime = 0
    ti.uname = ti.gname = ""
    return ti


async def put_files(runtime, files: Files, *, clear: Iterable[str] = ()) -> None:
    """Write files into the runtime as root: one tar, one runtime.write, then one command that removes each path in
    clear, extracts the tar over / and deletes it. Raises RuntimeError if the command fails."""
    targets = [shlex.quote(_absolute(p)) for p in clear]
    tar = files_tar(files)
    root = runtime.with_user(None)
    path = f"/tmp/tb-box-{uuid.uuid4().hex}.tar"
    await root.write(path, tar)
    remove = f"rm -rf -- {' '.join(targets)} && " if targets else ""
    command = f"{remove}tar -xf {shlex.quote(path)} -C / && rm -f {shlex.quote(path)}"
    result = await root.run(["sh", "-c", command], {})
    if result.exit_code != 0:
        raise RuntimeError(f"writing the box files failed (exit {result.exit_code}): {result.stderr.strip()[-500:]}")
