"""Evaluator-side grading: capture, staging, the grader run and the private sidecar.

capture_solution (finalize, agent box): a bounded probe of /app/solution, then verifiers' collect of that one path,
then archive.from_tar on the evaluator. A missing, oversize, overfull or unreadable solution becomes an empty
canonical archive with its reason; agent content never raises. Only a box that cannot be read while the solution is
within the limits raises TB-E203.

stage (stage_verifier, grading box): probe the jail with `python -m transitbench.runner --probe` (required on Prime:
TB-E102), upload each set's tar chunks four at a time into /opt/tb-grade/in/, write plan.json, then one in-box
script checks every chunk's sha256, extracts it into /opt/tb-grade/stars/<set>/ (0700, files 0600), deletes it and
checks every planned star file, staged or in the image, against its sha256 (TB-E103 on any failure). The run order
is sha256(order_key + star id).

run (the reward, grading box): start the runner on the plan in the background, poll out/status.json, then read
out/results.jsonl back and check it strictly against the status and the plan (TB-E201 for a grader that stops
making progress, TB-E202 for missing or invalid results).

write_private keeps the evaluator's private records under <data>/private/{grades,archives,logs}.

Every hook error is a GradingError whose message is one fixed "TB-Exxx: <text>" with no data in it; the detail
(which may name stars) goes only to the private log. Raise it `from None` so no chained exception reaches a trace.

Evaluator only: never written into a box. Imports verifiers lazily (capture only), so the module loads without it.
"""

import asyncio
import hashlib
import itertools
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from transitbench import archive

if TYPE_CHECKING:
    from transitbench_env.hidden import HiddenSet

MESSAGES = {
    "TB-E001": "evaluator data missing or does not match its digest",
    "TB-E101": "grading ran outside the staged verifier",
    "TB-E102": "the jail is required here and its probe failed",
    "TB-E103": "staging the hidden stars failed",
    "TB-E201": "the grader stopped making progress",
    "TB-E202": "the grader's results are missing or invalid",
    "TB-E203": "the solution could not be captured from the box",
    "TB-E204": "the captured solution differs from the configured one",
}

SOLUTION = "/app/solution"
GRADE_DIR = "/opt/tb-grade"
IN_DIR, STARS_DIR, OUT_DIR = f"{GRADE_DIR}/in", f"{GRADE_DIR}/stars", f"{GRADE_DIR}/out"
PLAN_PATH, STAGE_PATH, LOG_PATH = f"{GRADE_DIR}/plan.json", f"{GRADE_DIR}/stage.json", f"{GRADE_DIR}/grade.log"
STATUS_PATH, RESULTS_PATH = f"{OUT_DIR}/status.json", f"{OUT_DIR}/results.jsonl"
RUNNER = ["python3", "-s", "-m", "transitbench.runner"]
SCRIPT = ["python3", "-I", "-S", "-c"]
UPLOADS = 4
PROBE_S, BOX_S = 60, 120
# The collection cap is max_bytes + COLLECT_SLACK, sized so the probe's tar bound never decides for a solution within
# the limits. A path the probe can stat, or a link target, is under PATH_MAX (at most 4,095 bytes), so a name costs at
# most 512 + 4,096 = 4,608 bytes of GNU long-name blocks. Within the limits the GNU tar is at most
#   256 entries x (512 header + 4,608 name + 511 padding)             1,441,536
# + 256 others x (512 header + 4,608 name + 4,608 link target)          2,490,368
# + root header and end blocks                                             1,536
# + 1 MiB of content                                                   1,048,576  = 4,982,016 bytes.
# 5 MiB makes the cap 6 MiB (6,291,456), which leaves 1,309,440 bytes for GNU tar's 10 KiB record padding and stays
# under archive.MAX_TAR_BYTES (8 MiB). The arithmetic holds for archive.MAX_FILES and MAX_BYTES (test_grading).
COLLECT_SLACK = 5 << 20
STATUS_MAX_BYTES, RESULTS_MAX_BYTES, LOG_TAIL_BYTES = 1 << 16, 8 << 20, 4000
POLL_ERRORS, READ_ATTEMPTS = 10, 3
MAX_ROWS = 3
STATUSES = ("ok", "timeout", "oom", "stdout_overflow")
STATUS_RE = re.compile(r"(exit|signal):[1-9][0-9]{0,2}")
RUN_ERRORS = ("hash", "spawn", "jail")
RECORD_KEYS = {"star", "status", "cpu_s", "wall_s", "rows", "n_rows", "n_invalid", "isolation", "slot", "leak"}
OPTIONAL_KEYS = {"stderr"}
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SHA_RE = re.compile(r"[0-9a-f]{64}")
PRIVATE = {"grades": ".json", "archives": ".tar", "logs": ".log"}


class GradingError(RuntimeError):
    """A grading failure. str() is the fixed "TB-Exxx: <text>" of MESSAGES; detail is for the private log only."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {MESSAGES[code]}")
        self.code, self.detail = code, detail

    def __reduce__(self):
        return type(self), (self.code, self.detail)


@dataclass
class Plan:
    stars: list[tuple[str, str]]  # (set name, star id) in run order
    digest: str  # sha256 of plan.json as written into the box
    isolation: str  # "jail" or "none": what the probe found, and what every record must say
    status: dict | None = None  # the final out/status.json, set by run() (image, runner_sha256, results_sha256)


# ---- capture (finalize, agent box) ---------------------------------------------------------------------------------

# Run as root in the agent box with python3 -I -S: argv = root, max_entries. Walks without following
# links and prints one JSON line. It counts as archive.from_dir does, in the same order: entries (directories and
# regular files, bytecode and names that are not UTF-8 included), others (everything else) and bytes (st_size of the
# regular files). It stops as soon as a count passes max_entries but sums sizes to the end, so a solution over both
# limits is too_many_files whatever the walk order. An OSError ends the walk. "tar" bounds the bytes of a GNU tar of
# the tree, as verifiers' collect makes it (headers, long names and links, padded contents, end blocks).
PROBE = r'''
import json, os, stat, sys
root = sys.argv[1]
max_entries = int(sys.argv[2])
r = {"kind": "dir", "entries": 0, "others": 0, "bytes": 0, "tar": 1024, "error": False}
def header(*names):
    return 512 + sum(512 + (len(n) + 512) // 512 * 512 for n in names if len(n) > 99)
def full():
    return r["entries"] > max_entries or r["others"] > max_entries
try:
    top = os.lstat(root)
except (FileNotFoundError, NotADirectoryError):
    r["kind"] = "missing"
except OSError:
    r["error"] = True
else:
    if not stat.S_ISDIR(top.st_mode):
        r["kind"] = "other"
    else:
        r["tar"] += header(os.fsencode(root)[1:] + b"/")
        stack = [root]
        try:
            while stack and not full():
                with os.scandir(stack.pop()) as it:
                    for e in it:
                        s = e.stat(follow_symlinks=False)
                        name = os.fsencode(e.path)[1:]
                        if stat.S_ISDIR(s.st_mode):
                            r["entries"] += 1
                            r["tar"] += header(name + b"/")
                            stack.append(e.path)
                        elif stat.S_ISREG(s.st_mode):
                            r["entries"] += 1
                            r["bytes"] += s.st_size
                            r["tar"] += header(name) + (s.st_size + 511) // 512 * 512
                        else:
                            r["others"] += 1
                            link = os.fsencode(os.readlink(e.path)) if stat.S_ISLNK(s.st_mode) else b""
                            r["tar"] += header(name, link)
                        if full():
                            break
        except OSError:
            r["error"] = True
print(json.dumps(r))
'''


async def capture_solution(runtime, *, max_bytes: int, max_files: int) -> archive.Canonical:
    """The canonical archive of /app/solution in the agent box. Never raises on agent content: missing, over the
    limits or unreadable gives an empty canonical archive with its status and reason. Raises TB-E203 only when the
    box cannot be read although the solution is within the limits."""
    cap = max_bytes + COLLECT_SLACK
    first = await _probe(runtime, max_files)
    if first is not None and (reason := _verdict(first, max_bytes, max_files, cap)):
        return archive.empty(reason)
    try:
        tar = await _collect(runtime, cap)
    except Exception as e:  # noqa: BLE001 - a failed collection is either agent content or infrastructure; see below
        again = await _probe(runtime, max_files)
        if again is not None and (reason := _verdict(again, max_bytes, max_files, cap)):
            return archive.empty(reason)
        if again is None and await _alive(runtime):
            return archive.empty("unreadable")  # the box runs but its own tools fail on the solution
        raise GradingError("TB-E203", f"collect failed: {type(e).__name__}: {e}"[:2000]) from None
    try:
        return archive.from_tar(tar, max_bytes=max_bytes, max_files=max_files)
    except Exception:  # noqa: BLE001 - from_tar handles bad tars itself; this is a last guard on agent bytes
        return archive.empty("unreadable")


async def _probe(runtime, max_files: int) -> dict | None:
    """The probe's report, or None if it could not run or said nothing valid."""
    argv = ["timeout", str(PROBE_S), *SCRIPT, PROBE, SOLUTION, str(max_files)]
    try:
        async with asyncio.timeout(PROBE_S + 30):
            result = await runtime.run(argv, {})
        report = json.loads(result.stdout.strip().splitlines()[-1]) if result.exit_code == 0 else None
    except Exception:  # noqa: BLE001
        return None
    keys = {"kind": str, "entries": int, "others": int, "bytes": int, "tar": int, "error": bool}
    if not (isinstance(report, dict) and set(report) == set(keys)
            and all(type(report[k]) is t for k, t in keys.items()) and report["kind"] in ("dir", "missing", "other")):
        return None
    return report


def _verdict(probe: dict, max_bytes: int, max_files: int, cap: int) -> str | None:
    """The reason the probed solution is empty or rejected, or None if it should be collected."""
    if probe["kind"] != "dir":
        return "missing"
    if probe["entries"] > max_files or probe["others"] > max_files:
        return "too_many_files"
    if probe["bytes"] > max_bytes or probe["tar"] > cap:
        return "too_large"
    if probe["error"]:
        return "unreadable"
    return None


async def _collect(runtime, cap: int) -> bytes:
    import verifiers.v1 as vf

    got = await vf.collect(runtime, [vf.Artifact(source=SOLUTION)], max_bytes=cap, sweep=False)
    return got[SOLUTION] or b""


async def _alive(runtime) -> bool:
    try:
        async with asyncio.timeout(BOX_S):
            return (await runtime.run(["true"], {})).exit_code == 0
    except Exception:  # noqa: BLE001
        return False


# ---- staging (stage_verifier, grading box) -------------------------------------------------------------------------

# Run as root in the grading box with python3 -I -S: argv = the stage.json path. Prints one JSON line; exit 0 iff ok.
STAGE = r'''
import hashlib, io, json, os, re, sys, tarfile
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\.npz")
def done(ok, **kw):
    print(json.dumps({"ok": ok, **kw}))
    sys.exit(0 if ok else 1)
with open(sys.argv[1], "rb") as f:
    spec = json.load(f)
files = 0
for c in spec["chunks"]:
    with open(c["path"], "rb") as f:
        data = f.read()
    if hashlib.sha256(data).hexdigest() != c["sha256"]:
        done(False, error="chunk")
    os.makedirs(c["dest"], mode=0o700, exist_ok=True)
    os.chmod(c["dest"], 0o700)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        for m in tar:
            if not (m.isreg() and NAME.fullmatch(m.name)):
                done(False, error="member")
            try:
                fd = os.open(os.path.join(c["dest"], m.name), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except OSError:
                done(False, error="duplicate")
            with os.fdopen(fd, "wb") as out:
                out.write(tar.extractfile(m).read())
            files += 1
    os.remove(c["path"])
with open(spec["plan"], "rb") as f:
    plan = json.load(f)
bad = 0
for s in plan["stars"]:
    try:
        with open(s["path"], "rb") as f:
            bad += hashlib.file_digest(f, "sha256").hexdigest() != s["sha256"]
    except OSError:
        bad += 1
done(bad == 0 and files == spec["files"], files=files, checked=len(plan["stars"]), bad=bad)
'''


async def stage(runtime, sets: "list[HiddenSet]", caps: dict, *, require_jail: bool, order_key: str) -> Plan:
    """Stage the sets in the grading box and write the plan. Raises TB-E102 when the jail is required and its probe
    fails, TB-E103 when the runner is missing or any upload or check fails, ValueError on bad input."""
    stars = _planned(sets, order_key)
    staged = sum(len(s.stars) for s in sets if s.chunks)
    prepare = (f"umask 077 && rm -rf {IN_DIR} {STARS_DIR} {OUT_DIR} {PLAN_PATH} {STAGE_PATH} && "
               f"mkdir -p {GRADE_DIR} {IN_DIR} {STARS_DIR} && chmod 0700 {GRADE_DIR} {IN_DIR} {STARS_DIR}")
    result = await _box(runtime, ["sh", "-c", prepare], "prepare")
    if result.exit_code != 0:
        raise GradingError("TB-E103", f"prepare: {result.stderr[-500:]}") from None

    result = await _box(runtime, [*RUNNER, "--probe"], "probe")
    try:
        report = json.loads(result.stdout)
        jail = bool(report["ok"]) and result.exit_code == 0
    except (ValueError, KeyError, TypeError):
        raise GradingError("TB-E103", f"runner probe printed no report (exit {result.exit_code}): "
                                      f"{result.stderr[-1000:]}") from None
    if require_jail and not jail:
        raise GradingError("TB-E102", json.dumps(report)[:4000]) from None
    isolation = "jail" if jail else "none"

    chunks = await _upload(runtime, sets)
    plan = {"version": 1, "solution": SOLUTION, "cap_s": float(caps["star_cap_s"]), "workers": int(caps["workers"]),
            "isolation": isolation, "memory_bytes": int(caps["memory_bytes"]),
            "stdout_max_bytes": int(caps["stdout_max_bytes"]),
            "stars": [{"id": sid, "path": path, "sha256": sha} for _, _, sid, path, sha in stars], "out": OUT_DIR}
    data = json.dumps(plan, separators=(",", ":")).encode()
    spec = json.dumps({"chunks": chunks, "plan": PLAN_PATH, "files": staged}).encode()
    try:
        await runtime.write(PLAN_PATH, data)
        await runtime.write(STAGE_PATH, spec)
    except Exception as e:  # noqa: BLE001
        raise GradingError("TB-E103", f"write plan: {type(e).__name__}: {e}"[:2000]) from None
    result = await _box(runtime, [*SCRIPT, STAGE, STAGE_PATH], "check")
    try:
        checked = json.loads(result.stdout)
    except ValueError:
        checked = None
    if result.exit_code != 0 or not (isinstance(checked, dict) and checked.get("ok") is True):
        raise GradingError("TB-E103", f"in-box check: {result.stdout[-500:]} {result.stderr[-1000:]}") from None
    return Plan([(name, sid) for _, name, sid, _, _ in stars], hashlib.sha256(data).hexdigest(), isolation)


def _planned(sets: "list[HiddenSet]", order_key: str) -> list[tuple[str, str, str, str, str]]:
    """(order, set, id, path in the box, sha256) for every star, sorted by order = sha256(order_key + id)."""
    names = [s.name for s in sets]
    if not sets or len(set(names)) != len(names) or not all(SAFE.fullmatch(n) for n in names):
        raise ValueError("stage needs at least one set, with unique safe names")
    stars, seen = [], set()
    for s in sets:
        if bool(s.chunks) == (s.in_image is not None):
            raise ValueError(f"set {s.name} needs exactly one of chunks and in_image")
        base = s.in_image.rstrip("/") if s.in_image is not None else f"{STARS_DIR}/{s.name}"
        for sid, sha in s.stars.items():
            if sid in seen or not SAFE.fullmatch(sid) or not SHA_RE.fullmatch(sha):
                raise ValueError("star ids must be unique and safe, with sha256 values")
            seen.add(sid)
            order = hashlib.sha256(f"{order_key}{sid}".encode()).hexdigest()
            stars.append((order, s.name, sid, f"{base}/{sid}.npz", sha))
    return sorted(stars)


async def _upload(runtime, sets: "list[HiddenSet]") -> list[dict]:
    """Write every chunk to IN_DIR, at most UPLOADS at a time; the stage.json entries in order."""
    gate = asyncio.Semaphore(UPLOADS)

    async def one(name: str, i: int, chunk: Path | bytes) -> dict:
        async with gate:
            data, sha = await asyncio.to_thread(_chunk, chunk)
            path = f"{IN_DIR}/{name}-{i:03d}.tar"
            await runtime.write(path, data)
        return {"path": path, "sha256": sha, "dest": f"{STARS_DIR}/{name}"}

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(one(s.name, i, c)) for s in sets for i, c in enumerate(s.chunks)]
    except Exception as e:  # noqa: BLE001 - TaskGroup raises an ExceptionGroup; cancellation passes through
        raise GradingError("TB-E103", f"upload: {e!r}"[:2000]) from None
    return [t.result() for t in tasks]


def _chunk(chunk: Path | bytes) -> tuple[bytes, str]:
    data = chunk if isinstance(chunk, bytes) else Path(chunk).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


async def _box(runtime, argv: list[str], step: str):
    try:
        async with asyncio.timeout(BOX_S):
            return await runtime.run(argv, {})
    except Exception as e:  # noqa: BLE001
        raise GradingError("TB-E103", f"{step}: {type(e).__name__}: {e}"[:2000]) from None


# ---- the grader run (reward, grading box) ---------------------------------------------------------------------------

async def run(runtime, plan: Plan, *, poll_s: float, stale_s: float = 180.0) -> dict[str, dict]:
    """Run the staged plan in the background and return each planned star's record, in plan order.

    Polls out/status.json every poll_s. Before the first status is read only the stale clock runs; after that, up
    to POLL_ERRORS consecutive failed polls are tolerated. TB-E201 if the beat does not change for stale_s or the
    polls keep failing; TB-E202 if the runner reports an error or the results do not match the status and plan.
    Sets plan.status to the final status."""
    try:
        await runtime.run_background([*RUNNER, "--plan", PLAN_PATH], {}, LOG_PATH)
    except Exception as e:  # noqa: BLE001
        raise GradingError("TB-E201", f"launch: {type(e).__name__}: {e}"[:2000]) from None
    loop = asyncio.get_running_loop()
    beat, changed, errors, seen = None, loop.time(), 0, False
    while True:
        await asyncio.sleep(poll_s)
        try:
            status = check_status(await runtime.read(STATUS_PATH, max_bytes=STATUS_MAX_BYTES))
        except Exception as e:  # noqa: BLE001 - a missed poll; counted below
            errors += seen
            if errors > POLL_ERRORS:
                raise GradingError("TB-E201", f"{errors} failed polls, last {type(e).__name__}: {e}"[:2000]
                                   + await _log_tail(runtime)) from None
        else:
            seen, errors = True, 0
            if status["beat"] != beat:
                beat, changed = status["beat"], loop.time()
            if status["done"]:
                break
        if loop.time() - changed > stale_s:
            raise GradingError("TB-E201", f"no beat for {stale_s:.0f} s" + await _log_tail(runtime)) from None

    plan.status = status
    n = len(plan.stars)
    if status["error"] is not None:
        raise GradingError("TB-E202", f"runner error {status['error']}" + await _log_tail(runtime)) from None
    if (status["planned"], status["written"], status["isolation"]) != (n, n, plan.isolation):
        raise GradingError("TB-E202", f"status {status['planned']}/{status['written']} {status['isolation']} for "
                                      f"{n} {plan.isolation}") from None
    data = None
    for attempt in range(READ_ATTEMPTS):
        try:
            data = await runtime.read(RESULTS_PATH, max_bytes=RESULTS_MAX_BYTES)
            break
        except Exception as e:  # noqa: BLE001
            if attempt + 1 == READ_ATTEMPTS:
                raise GradingError("TB-E202", f"read results: {type(e).__name__}: {e}"[:2000]) from None
            await asyncio.sleep(poll_s)
    if hashlib.sha256(data).hexdigest() != status["results_sha256"]:
        raise GradingError("TB-E202", "results do not match the status sha256") from None
    return check_results(data, plan)


def check_status(data: bytes) -> dict:
    """A status.json document as the runner writes it, checked; ValueError if malformed."""
    st = json.loads(data, parse_constant=_no_constant)
    ints = ("planned", "written", "beat")
    if not (isinstance(st, dict) and st.get("version") == 1 and type(st.get("done")) is bool
            and all(type(st.get(k)) is int and st[k] >= 0 for k in ints) and isinstance(st.get("isolation"), str)
            and st.get("error") in (None, *RUN_ERRORS)
            and (st.get("results_sha256") is None or SHA_RE.fullmatch(str(st["results_sha256"])))):
        raise ValueError("malformed status")
    return st


def check_results(data: bytes, plan: Plan) -> dict[str, dict]:
    """The records of results.jsonl keyed by star in plan order, after strict checks: UTF-8 JSON lines, each ending
    in a newline, one object per planned star and no other, exactly RECORD_KEYS (the keys runner.py writes per star)
    plus an optional stderr, with valid types and values, the plan's isolation, and rows as the runner writes them (at
    most 3 [period > 0, t0, score] lists of finite numbers, sorted by -score, min(n_rows, 3) of them, none unless ok).
    Raises TB-E202 on any failure."""
    planned = {sid for _, sid in plan.stars}
    records: dict[str, dict] = {}
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise GradingError("TB-E202", "results are not UTF-8") from None
    if text and not text.endswith("\n"):
        raise GradingError("TB-E202", "results do not end in a newline") from None
    for n, line in enumerate(text.splitlines(), 1):
        try:
            rec = json.loads(line, parse_constant=_no_constant, object_pairs_hook=_unique_keys)
        except ValueError:
            raise GradingError("TB-E202", f"line {n}: not one JSON object") from None
        problem = _record_problem(rec, plan.isolation)
        if problem is None and rec["star"] not in planned:
            problem = "star not in the plan"
        if problem is None and rec["star"] in records:
            problem = "star appears twice"
        if problem is not None:
            raise GradingError("TB-E202", f"line {n}: {problem}") from None
        records[rec["star"]] = rec
    if len(records) != len(planned):
        raise GradingError("TB-E202", f"{len(planned) - len(records)} planned stars have no record") from None
    return {sid: records[sid] for _, sid in plan.stars}


def _record_problem(rec: Any, isolation: str) -> str | None:
    if not isinstance(rec, dict):
        return "not an object"
    keys = set(rec)
    if not RECORD_KEYS <= keys or keys - RECORD_KEYS - OPTIONAL_KEYS:
        return "wrong keys"
    if not isinstance(rec["star"], str) or not rec["star"]:
        return "bad star"
    status = rec["status"]
    if not (isinstance(status, str) and (status in STATUSES or STATUS_RE.fullmatch(status))):
        return "bad status"
    if not all(_number(rec[k]) and rec[k] >= 0 for k in ("cpu_s", "wall_s")):
        return "bad times"
    if not all(type(rec[k]) is int and rec[k] >= 0 for k in ("n_rows", "n_invalid", "slot", "leak")) or rec["leak"] > 1:
        return "bad counts"
    if rec["isolation"] != isolation:
        return "wrong isolation"
    if "stderr" in rec and not isinstance(rec["stderr"], str):
        return "bad stderr"
    rows = rec["rows"]
    if not (isinstance(rows, list) and all(isinstance(r, list) and len(r) == 3 and all(map(_number, r)) and r[0] > 0
                                           for r in rows)):
        return "bad rows"
    if len(rows) != min(rec["n_rows"], MAX_ROWS) or any(a[2] < b[2] for a, b in itertools.pairwise(rows)):
        return "rows do not match n_rows or are not sorted by score"
    if status != "ok" and (rows or rec["n_rows"] or rec["n_invalid"]):
        return "rows on a star that is not ok"
    return None


def _number(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def _no_constant(name: str):
    raise ValueError(f"non-finite number {name}")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    out = dict(pairs)
    if len(out) != len(pairs):
        raise ValueError("duplicate key")
    return out


async def _log_tail(runtime) -> str:
    """The end of the grader's log, for the private detail only (it may name stars)."""
    try:
        async with asyncio.timeout(BOX_S):
            result = await runtime.run(["tail", "-c", str(LOG_TAIL_BYTES), LOG_PATH], {})
        return f"\n--- grade.log tail ---\n{result.stdout}"
    except Exception:  # noqa: BLE001
        return ""


# ---- private records on the evaluator ------------------------------------------------------------------------------

def write_private(data_dir: Path, kind: Literal["grades", "archives", "logs"], name: str,
                  payload: dict | bytes) -> Path:
    """Write <data>/private/<kind>/<name><ext> (grades .json, archives .tar, logs .log), directories 0700 and files
    0600, and return its path. name is one or two safe components ("<release>/<trace_id>" for grades). Grades are
    replaced atomically; an archive's name must be the sha256 of its bytes and an existing one is kept; logs are
    appended, a dict as one JSON line."""
    parts = name.split("/") if isinstance(name, str) else []
    if kind not in PRIVATE or not 1 <= len(parts) <= 2 or not all(SAFE.fullmatch(p) for p in parts):
        raise ValueError(f"bad private record {kind!r} {name!r}")
    if isinstance(payload, dict):
        data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode() if kind == "logs" \
            else (json.dumps(payload, sort_keys=True, indent=1) + "\n").encode()
    elif isinstance(payload, bytes):
        data = payload
    else:
        raise TypeError("payload must be a dict or bytes")
    if kind == "archives" and hashlib.sha256(data).hexdigest() != name:
        raise ValueError("an archive is named by the sha256 of its bytes")
    folder = Path(data_dir, "private", kind, *parts[:-1])
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = folder / f"{parts[-1]}{PRIVATE[kind]}"
    if kind == "logs":
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab") as f:
            f.write(data)
        return path
    if kind == "archives" and path.exists():
        return path
    tmp = folder / f".{parts[-1]}.{uuid.uuid4().hex}.tmp"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
