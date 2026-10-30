"""The shared half of the TransitBench tasksets.

transitbench-survey and transitbench-mini are thin tasksets over this module: the task data both carry, the run-time
config, the presets (caps, timeouts, resources), the hooks they share and the gold oracle.

TaskData carries public values only, since it travels to the workers and into traces: the caps that /app/caps.json
and TASK.md state, the image, the box shape and isolated scoring, plus the preset's release and sample names or mini
token. A task is graded only in isolated scoring, never in the agent's own box. The hooks, in rollout order:

- setup (agent box, and again in the grading box before the restore): TASK.md, caps.json, run_dev.py, the science
  package and the starter (or TaskConfig.solution) in /app, plus the preset's labelled data. Idempotent.
- finalize (agent box): harness.stop_agent kills whatever the agent still runs (after an agent timeout, its last
  command), then the canonical archive of /app/solution (grading.capture_solution) becomes the only artifact,
  so verifiers collects nothing itself. The archive, its sha256 and the static scan go into
  trace.info["transitbench"] and the archive into <data>/private/archives, so a failed grade can be regraded.
  With TaskConfig.solution set, the captured solution must be the one setup installed (TB-E204).
- stage_verifier (grading box): the preset's hidden sets staged by grading.stage, the jail required on Prime; their
  public digests and counts go into trace.info.
- headline, the one reward: the staged plan runs in the background (grading.run) and the preset scores the records
  on the evaluator.

At the model boundary, budget_notice (a request interceptor) ends each new tool result with the [budget] line: turns
and tokens used and agent time left, against the limits that end the session.

Errors: a GradingError carries one fixed "TB-Exxx: <text>"; the hooks write its detail only to
<data>/private/logs/<trace id>.log. HookError adds the codes the task hooks raise themselves.

The evaluator data directory is TransitTaskConfig.data_dir, else $TRANSITBENCH_DATA (TB-E001 without either).

Evaluator only: never written into a box.
"""

import asyncio
import contextlib
import functools
import hashlib
import importlib.resources
import io
import json
import os
import platform
import re
import tarfile
import time
from pathlib import Path
from typing import ClassVar, Literal

import verifiers.v1 as vf
from typing_extensions import TypeVar
from verifiers.v1.harnesses.utils.compaction import TOOL_OUTPUT_MAX_BYTES

from transitbench import archive, score
from transitbench_env import box, grading, hidden
from transitbench_env.grading import GradingError
from transitbench_env.harness import stop_agent
from transitbench_env.hidden import HiddenSet
from transitbench_env.scan import scan

DATA_ENV = "TRANSITBENCH_DATA"
SOLUTION = grading.SOLUTION
IMAGES = json.loads((importlib.resources.files("transitbench_env") / "data" / "images.json").read_text())
LIBRARIES = {"python": "3.12", "numpy": "2.5.4", "scipy": "1.18.1", "astropy": "8.0.2", "scikit-learn": "1.9.1"}
"""Python and the libraries both images install (image/requirements.in), as TASK.md lists them."""
INFO_VERSION = 1
NEAR_CAP = 0.8
"""An "ok" star whose wall time reached this share of the star cap counts as near the cap."""
HEADLINE = ("S", "C", "F_per_100", "eb_as_planet", "planets", "scored_stars", "credit", "false")

PRESETS = {
    "survey": {"star_cap_s": 60.0, "command_cap_s": 940.0, "workers": 16, "max_turns": 60, "max_tokens": 200_000,
               "memory_bytes": 1792 << 20,
               "timeout": vf.TaskTimeout(setup=1800, agent=14400, finalize=300, scoring=10800),
               "resources": vf.TaskResources(cpu=16, memory=32, disk=10)},
    "mini": {"star_cap_s": 20.0, "command_cap_s": 75.0, "workers": 4, "max_turns": 12, "max_tokens": 32_768,
             "memory_bytes": 1280 << 20,
             "timeout": vf.TaskTimeout(setup=600, agent=3600, finalize=120, scoring=900),
             "resources": vf.TaskResources(cpu=4, memory=8, disk=5)},
}
"""Per preset: the caps, the stage timeouts and the box shape.

command_cap_s is at least 1.25 x labelled stars x star_cap_s / workers: the time a run_dev.py pass over every labelled
star takes if each star uses its whole cap, plus a quarter.

memory_bytes is per star, so that workers x memory_bytes plus MEMORY_RESERVE fits the box's MemTotal and no star within
its cap can run the box out of memory. MemTotal is the box's nominal memory less what the kernel keeps. The sums leave
2 GiB under the survey box's nominal 32 GiB (16 x 1.75 GiB + 2 GiB = 30 GiB) and 1 GiB under the mini box's 8 GiB
(4 x 1.25 GiB + 2 GiB = 7 GiB), so they fit MemTotal as long as the kernel keeps less than 1 GiB of a box."""
MEMORY_RESERVE = 2 << 30
"""Box memory left over the stars' caps: the grader, the box's own processes and the page cache."""

BUDGET = "[budget] "
"""The prefix of the line budget_notice appends to a tool result."""
CUT = re.compile(r"\n\[\.\.\. (\d+) bytes truncated \.\.\.\]\n")
"""The marker verifiers' truncate_tool_output leaves where it cut a tool result."""

MESSAGES = {
    "TB-E004": "the configured solution is missing, empty or over the limits",
    "TB-E104": "writing the task files into the box failed",
    "TB-E205": "scoring the grader's results failed",
}


class HookError(GradingError):
    """A failure of the task hooks themselves, with a code grading.MESSAGES does not hold. str() is the fixed
    "TB-Exxx: <text>" of MESSAGES; detail is for the private log only."""

    def __init__(self, code: str, detail: str = ""):
        RuntimeError.__init__(self, f"{code}: {MESSAGES[code]}")
        self.code, self.detail = code, detail


class TransitTaskConfig(vf.TaskConfig):
    data_dir: Path | None = None
    """Evaluator data directory (data/ in a checkout); None reads $TRANSITBENCH_DATA."""
    solution: Path | None = None
    """A solution directory, .tar or .tar.gz that setup installs instead of the starter, for regrades, baselines and
    runner fixtures (run with vf-validate --only-noop, so the task data and its hash stay the same); finalize then
    requires the same solution back (TB-E204)."""


class TransitData(vf.TaskData):
    scoring: vf.ScoringConfig = vf.ScoringConfig(mode="isolated")
    preset: Literal["survey", "mini"]
    star_cap_s: float
    command_cap_s: float
    workers: int
    max_turns: int
    """Stated in TASK.md; the framework limit is the run's env.agent.max_turns."""
    max_tokens: int
    """Stated in TASK.md; the framework limit is the run's env.agent.max_total_tokens."""
    solution_max_bytes: int = archive.MAX_BYTES
    solution_max_files: int = archive.MAX_FILES
    stdout_max_bytes: int = 1 << 16
    memory_bytes: int
    """Per star: set by the preset so that every slot's cap together fits the box (PRESETS)."""


DataT = TypeVar("DataT", bound=TransitData)


@functools.cache
def _counts(preset: str) -> tuple[tuple[tuple[str, int], ...], int]:
    if preset == "survey":
        samples = hidden.release_manifest("r1")["samples"]
        n_dev = len(hidden.dev_set().stars)
        return (("A", samples["A"]["stars"]), ("B", samples["B"]["stars"]), ("dev", n_dev)), n_dev
    from transitbench_env import mini_draw

    mix = mini_draw.HIDDEN_MIX
    return ((("inj", mix["inj_random"] + mix["inj_tce_only"]), ("null", mix["null"]), ("eb", mix["eb_flagged"])),
            sum(mini_draw.BOX_MIX.values()))


def caps(data: TransitData) -> dict:
    """/app/caps.json (box.CAPS_KEYS): the task data's caps, the agent time limit, the hidden stars by sample (survey:
    release r1's A and B and the dev set) or population (mini), the labelled stars in the box and the libraries."""
    counts, labelled = _counts(data.preset)
    return {"preset": data.preset, "star_cap_s": data.star_cap_s, "command_cap_s": data.command_cap_s,
            "workers": data.workers, "max_turns": data.max_turns, "max_tokens": data.max_tokens,
            "wall_s": data.timeout.agent, "solution_max_bytes": data.solution_max_bytes,
            "solution_max_files": data.solution_max_files, "stdout_max_bytes": data.stdout_max_bytes,
            "memory_bytes": data.memory_bytes, "hidden": dict(counts), "labelled": labelled,
            "libraries": dict(LIBRARIES)}


@functools.cache
def _task_md(caps_json: str) -> str:
    return box.render_task_md(json.loads(caps_json))


def task_data(cls: type[DataT], *, image: str, **fields) -> DataT:
    """The task data of cls's preset: its caps, timeouts and resources, the image, /app, no network beyond the
    framework, /app/solution as the artifact with grading's collection cap (verifiers' fallback collect and
    vf-validate use it), and the prompt, TASK.md rendered from the caps."""
    preset = cls.model_fields["preset"].default
    max_bytes = fields.get("solution_max_bytes", cls.model_fields["solution_max_bytes"].default)
    common = {**PRESETS[preset], "image": image, "workdir": "/app", "network": vf.NetworkPolicyConfig(allow=[]),
              "artifacts": [vf.Artifact(source=SOLUTION)], "artifact_max_bytes": max_bytes + grading.COLLECT_SLACK,
              **fields}
    prompt = _task_md(json.dumps(caps(cls(**common))))  # keeps the key order TASK.md lists them in
    return cls(**common, prompt=prompt)


# ---- the gold oracle ------------------------------------------------------------------------------------------------

ORACLE = '''"""Gold oracle (transitbench_env.task): prints the truth rows of the star whose file sha256 it reads."""

import hashlib
import sys

ROWS = {rows!r}

for row in ROWS.get(hashlib.sha256(sys.stdin.buffer.read()).hexdigest(), ()):
    print(",".join(map(repr, row)))
'''


def oracle_rows(truth: dict) -> dict[str, list[list[float]]]:
    """Per star of a truth document, one [period, t0, 1.0] row per scored signal: a catalogue that scores 1."""
    return {sid: [[float(s["period"]), float(s["t0"]), 1.0] for s in t["signals"] if s["scored"]]
            for sid, t in truth["stars"].items()}


def oracle_solution(sets: list[HiddenSet]) -> bytes:
    """solution.py that prints, for each hidden star, its truth rows, looked up by the sha256 of its file."""
    rows = {}
    for s in sets:
        for sid, r in oracle_rows(s.truth).items():
            rows[s.stars[sid]] = r
    return ORACLE.format(rows=rows).encode()


def scorer_gold(s: HiddenSet) -> bool:
    """Whether the truth oracle of a set scores exactly S = 1 on the evaluator."""
    catalogue, _ = score.catalogue_from_runs({sid: {"rows": r} for sid, r in oracle_rows(s.truth).items()})
    return score.score(score.parse_truth(s.truth), catalogue)["S"] == 1.0


# ---- helpers --------------------------------------------------------------------------------------------------------

def solution_tree(c: archive.Canonical) -> tuple[dict[str, bytes], frozenset[str]]:
    """A canonical solution's files (path relative to /app/solution -> bytes) and which of them are executable."""
    files, executable = {}, set()
    with tarfile.open(fileobj=io.BytesIO(c.tar), mode="r:") as tar:
        for m in tar:
            if m.isfile():
                rel = m.name.removeprefix(f"{archive.ROOT}/")
                files[rel] = tar.extractfile(m).read()
                if m.mode & 0o111:
                    executable.add(rel)
    return files, frozenset(executable)


def solution_entries(files: dict[str, bytes], executable: frozenset[str] = frozenset()) -> box.Files:
    """The /app/solution entries put_files writes for these files."""
    return {f"{SOLUTION}/{rel}": (body, box.EXEC_MODE if rel in executable else box.FILE_MODE)
            for rel, body in files.items()}


def installed_sha(entries: box.Files, max_bytes: int, max_files: int) -> str:
    """The canonical sha256 of /app/solution once put_files has written these entries."""
    return archive.from_tar(box.files_tar(entries), max_bytes=max_bytes, max_files=max_files).sha256


def read_solution(path: Path, max_bytes: int, max_files: int) -> archive.Canonical:
    """The canonical archive of a configured solution directory, .tar or .tar.gz; TB-E004 unless it is "ok"."""
    path = Path(path)
    if path.is_dir():
        c = archive.from_dir(path, max_bytes=max_bytes, max_files=max_files)
    else:
        try:
            c = archive.from_tar(path.read_bytes(), max_bytes=max_bytes, max_files=max_files)
        except OSError:
            c = archive.empty("missing")
    if c.status != "ok":
        raise HookError("TB-E004", f"{c.status} {c.reason}") from None
    return c


def records_sha256(records: dict[str, dict]) -> str:
    """sha256 of the canonical JSON of [star, status, rows] for every record, sorted by star."""
    rows = [[sid, records[sid]["status"], records[sid]["rows"]] for sid in sorted(records)]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def truth_sha256(truth: dict) -> str:
    return hashlib.sha256(json.dumps(truth, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_counts(records: list[dict]) -> dict[str, int]:
    """Runs by outcome, with invalid rows and leaked processes."""
    statuses = [r["status"] for r in records]
    return {"ok": statuses.count("ok"), "timeout": statuses.count("timeout"),
            "crash": sum(s.startswith(("exit:", "signal:")) for s in statuses),
            "overflow": statuses.count("stdout_overflow"), "oom": statuses.count("oom"),
            "invalid_rows": sum(r["n_invalid"] for r in records), "leak": sum(r["leak"] for r in records)}


def cpu_quantiles(records: list[dict]) -> dict[str, float | None]:
    cpu = sorted(r["cpu_s"] for r in records)
    if not cpu:
        return {"p50": None, "p95": None}
    return {"p50": cpu[len(cpu) // 2], "p95": cpu[min(len(cpu) - 1, int(0.95 * len(cpu)))]}


# ---- the budget notice ----------------------------------------------------------------------------------------------

def budget_line(data: TransitData, trace: vf.Trace, now: float | None = None) -> str:
    """The [budget] line. Turns are trace.num_turns (model replies so far) and tokens trace.num_total_tokens, the
    numbers verifiers checks max_turns and max_total_tokens against before each model call. Their limits are the
    agent config's, the only ones verifiers enforces (RolloutLimits: None is no cap), so a limit the config leaves
    unset is not stated. Time left is the agent timeout (the config's timeout.rollout, else the task data's, which
    verifiers applies) less the time since the agent stage started, rounded down to the minute; it is left out while
    either is unknown.

    verifiers refuses a model call once num_turns reaches max_turns, before the request hooks run, so the highest
    count the agent sees is one under the limit, and its reply to that line is its last: the reply's commands run,
    but their output is never shown. The line says so then."""
    config = trace.agent.config
    turns, tokens = config.max_turns, config.max_total_tokens
    wall = config.timeout.rollout if config.timeout.rollout is not None else data.timeout.agent
    parts = [f"{trace.num_turns} turns used" if turns is None else f"{trace.num_turns} of {turns} turns used",
             f"{trace.num_total_tokens:,} tokens used" if tokens is None
             else f"{trace.num_total_tokens:,} of {tokens:,} tokens used"]
    if wall is not None and trace.timing.agent.start:
        elapsed = (time.time() if now is None else now) - trace.timing.agent.start
        hours, minutes = divmod(int(max(0.0, wall - elapsed) // 60), 60)
        left = f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"
        parts.append(f"{left} of {box._duration(wall)} left")
    last = " Your next reply is your last turn: its commands run, but you will not see their output." if (
        turns is not None and trace.num_turns + 1 == turns) else ""
    return (f"{BUDGET}{'; '.join(parts)}.{last} The session ends at the first limit, and solution/ is graded as it is "
            "then.")


def append_line(text: str, line: str, limit: int = TOOL_OUTPUT_MAX_BYTES) -> str | None:
    """text with line as its last line, in at most limit bytes of UTF-8, or None if it cannot fit.

    verifiers truncates every rewritten tool result over the limit again, which would cut a result the harness has
    already cut a second time and lose the first cut's marker. So a text that would pass the limit gives up the bytes
    just before its cut marker (the one nearest its middle, where verifiers puts it; else a new one at the middle),
    and the marker counts them."""
    tail = ("" if not text or text.endswith("\n") else "\n") + line
    if len(text.encode()) + len(tail.encode()) <= limit:
        return text + tail
    cut = min(CUT.finditer(text), key=lambda m: abs(m.start() - len(text) // 2), default=None)
    if cut is None:
        head, n, rest = text[:len(text) // 2], 0, text[len(text) // 2:]
    else:
        head, n, rest = text[:cut.start()], int(cut.group(1)), text[cut.end():]
    while True:
        out = f"{head}\n[... {n} bytes truncated ...]\n{rest}{tail}"
        over = len(out.encode()) - limit
        if over <= 0:
            return out
        data = head.encode()
        if over > len(data):
            return None
        head = data[:len(data) - over].decode(errors="ignore")
        n += len(data) - len(head.encode())


def with_budget(message: vf.ToolMessage, line: str) -> vf.ToolMessage | None:
    """The tool result with line appended, or None to leave it as it is.

    An empty result is left as it is: verifiers' tool gate runs the request interceptors on an empty result for each
    call the model proposes, before the call runs, and takes any rewrite of it as a verdict. With TransitBashHarness,
    which has no gate of its own, that verdict would end the rollout. Real results are never empty, because the
    harness's run_bash returns "(no output)" for a silent command. A result that already ends with a budget line (one
    pinned earlier) is left as it is too."""
    content = message.content
    if not content:
        return None
    if isinstance(content, str):
        text = content
    else:
        text = "\n".join(p.text for p in content if isinstance(p, vf.TextContentPart))
    if text.rsplit("\n", 1)[-1].startswith(BUDGET):
        return None
    if isinstance(content, str):
        content = append_line(content, line)
        if content is None:
            return None
    else:
        content = [*content, vf.TextContentPart(text="\n" + line)]
    return message.model_copy(update={"content": content})


# ---- the task -------------------------------------------------------------------------------------------------------

class TransitTask(vf.Task[DataT, vf.State, TransitTaskConfig]):
    """The hooks both presets share. A preset sets CLEAR and POLL_S and implements key, box_data, hidden_sets and
    score_records."""

    NEEDS_CONTAINER = True
    CLEAR: ClassVar[tuple[str, ...]] = (SOLUTION, "/app/transitbench")
    """Box paths setup removes before writing its files."""
    POLL_S: ClassVar[float] = 15.0
    """Seconds between polls of the grader's status."""

    _verifier: str | None = None
    """The grading box stage_verifier prepared; headline grades nowhere else."""
    _plan: grading.Plan | None = None
    _sets: dict[str, HiddenSet] | None = None

    # ---- per preset ----

    def box_data(self) -> dict[str, bytes] | None:
        """Files for /app/data (path relative to it -> bytes), or None to keep the image's. Blocking."""
        return None

    def hidden_sets(self) -> list[HiddenSet]:
        """The sets grading runs, in order. Blocking: call from a thread."""
        raise NotImplementedError

    def score_records(self, trace: vf.Trace, records: dict[str, dict]) -> float:
        """Score the grader's records on the evaluator, record metrics and info, and return the reward. Blocking."""
        raise NotImplementedError

    # ---- shared ----

    def caps(self) -> dict:
        return caps(self.data)

    def data_dir(self) -> Path:
        path = self.config.data_dir or os.environ.get(DATA_ENV)
        if not path:
            raise GradingError("TB-E001", f"no TransitTaskConfig.data_dir and no ${DATA_ENV}") from None
        return Path(path)

    def grading_task(self, runtime: vf.Runtime, *, scoring: vf.ScoringConfig):
        if scoring.mode != "isolated":
            raise ValueError("TransitBench tasks are graded only in isolated scoring: grading in the agent's own box "
                             "would run while its harness is still open")
        return super().grading_task(runtime, scoring=scoring)

    @contextlib.contextmanager
    def private_errors(self, trace: vf.Trace, hook: str):
        """Write a GradingError's code and detail to <data>/private/logs/<trace id>.log and let it propagate with its
        fixed text only."""
        try:
            yield
        except GradingError as e:
            with contextlib.suppress(Exception):
                grading.write_private(self.data_dir(), "logs", trace.id,
                                      {"code": e.code, "detail": e.detail, "hook": hook})
            raise

    async def solution_files(self) -> box.Files:
        """/app/solution as setup writes it: TaskConfig.solution, else the starter."""
        if self.config.solution is None:
            return solution_entries(box.starter())
        c = await asyncio.to_thread(read_solution, self.config.solution, self.data.solution_max_bytes,
                                    self.data.solution_max_files)
        return solution_entries(*solution_tree(c))

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        with self.private_errors(trace, "setup"):
            solution = await self.solution_files()
            data = await asyncio.to_thread(self.box_data)
            files = box.box_files(self.caps(), self.data.prompt, {}, data)
            files.update(solution)
            try:
                await box.put_files(runtime, files, clear=self.CLEAR)
            except RuntimeError as e:
                raise HookError("TB-E104", str(e)) from None

    @vf.intercept
    def budget_notice(self, request: vf.Request, trace: vf.Trace) -> vf.Request | None:
        """End a new tool result with the [budget] line (budget_line), so the agent can pace its turns, tokens and
        time; TASK.md says so.

        verifiers calls this once per new tool result, with the request cut just after it, so with parallel calls each
        result is the last message of its own call. Only that message changes: same messages, same tools. verifiers
        then pins the rewritten result, so the model keeps seeing the numbers as they were when it arrived."""
        last = request.messages[-1] if request.messages else None
        if not isinstance(last, vf.ToolMessage):
            return None
        rewritten = with_budget(last, budget_line(self.data, trace))
        if rewritten is None:
            return None
        return request.model_copy(update={"messages": [*request.messages[:-1], rewritten]})

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # After an agent timeout the chat program and its last command still run: stop them, so that /app/solution
        # is captured as it was when the session ended.
        await stop_agent(runtime)
        with self.private_errors(trace, "finalize"):
            expected = None
            if self.config.solution is not None:
                expected = installed_sha(await self.solution_files(), self.data.solution_max_bytes,
                                         self.data.solution_max_files)
            await self.capture(trace, runtime, expected=expected)

    async def capture(self, trace: vf.Trace, runtime: vf.Runtime, *, expected: str | None = None) -> archive.Canonical:
        """Capture /app/solution as the submission (finalize); TB-E204 if expected is set and differs."""
        d = self.data
        cap = await grading.capture_solution(runtime, max_bytes=d.solution_max_bytes, max_files=d.solution_max_files)
        trace.state.artifacts = {SOLUTION: cap.tar}
        report = await asyncio.to_thread(scan, cap.tar)
        await asyncio.to_thread(grading.write_private, self.data_dir(), "archives", cap.sha256, cap.tar)
        trace.info["transitbench"] = {
            "v": INFO_VERSION, "preset": d.preset,
            "solution": {"status": cap.status, "reason": cap.reason, "sha256": cap.sha256,
                         "bytes": sum(f["size"] for f in cap.files), "n_files": len(cap.files),
                         "files": list(cap.files), "dropped": list(cap.dropped),
                         "tar_gz_b64": archive.to_b64gz(cap.tar)},
            "scan": report}
        if expected is not None and cap.sha256 != expected:
            raise GradingError("TB-E204", f"captured {cap.sha256}, installed {expected}") from None
        return cap

    async def stage_verifier(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        info = trace.info["transitbench"]
        with self.private_errors(trace, "stage_verifier"):
            sets = await asyncio.to_thread(self.hidden_sets)
            prime = isinstance(runtime.config, vf.PrimeConfig)
            plan = await grading.stage(runtime, sets, self.caps(), require_jail=prime,
                                       order_key=info["solution"]["sha256"])
        self._sets, self._plan, self._verifier = {s.name: s for s in sets}, plan, runtime.name
        info["staged"] = {s.name: {"files_digest": s.digest, "n": len(s.stars)} for s in sets}

    @vf.reward(weight=1.0)
    async def headline(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """The preset's reward from one grader run over every staged set."""
        with self.private_errors(trace, "headline"):
            if self._plan is None or runtime.name != self._verifier:
                raise GradingError("TB-E101", f"runtime {runtime.name!r}, staged {self._verifier!r}") from None
            records = await grading.run(runtime, self._plan, poll_s=self.POLL_S)
            try:
                return await asyncio.to_thread(self.score_records, trace, records)
            except GradingError:
                raise
            except Exception as e:  # noqa: BLE001 - its message may name hidden stars
                raise HookError("TB-E205", f"{type(e).__name__}: {e}"[:4000]) from None

    async def validate(self, trace: vf.Trace, runtime: vf.Runtime) -> bool:
        """Gold check: an oracle solution.py that prints each hidden star's truth rows, captured by finalize and
        graded through the real isolated path, must score exactly 1."""
        with self.private_errors(trace, "validate"):
            sets = await asyncio.to_thread(self.hidden_sets)
            oracle = {f"{SOLUTION}/solution.py": (oracle_solution(sets), box.FILE_MODE)}
            try:
                await box.put_files(runtime, oracle, clear=(SOLUTION,))
            except RuntimeError as e:
                raise HookError("TB-E104", str(e)) from None
            await self.capture(trace, runtime)
        await vf.grade_submission(self, trace, runtime)
        reward = trace.rewards.get("headline")
        return reward is not None and reward.score == 1.0

    # ---- scoring helpers for the presets ----

    def grouped(self, records: dict[str, dict]) -> dict[str, dict[str, dict]]:
        """The records by staged set, in plan order."""
        groups = {name: {} for name in self._sets}
        for name, sid in self._plan.stars:
            groups[name][sid] = records[sid]
        return groups

    def grade_common(self, records: dict[str, dict]) -> dict:
        """The grade info both presets record: inputs, CPU time, stars near the cap, isolation and the rows digest."""
        status, cap = self._plan.status or {}, self.data.star_cap_s
        recs = list(records.values())
        return {"inputs": {"image": status.get("image"), "runner_sha256": status.get("runner_sha256"),
                           "caps": self.caps(), "evaluator": {"system": platform.system(),
                                                              "machine": platform.machine()}},
                "cpu_s": cpu_quantiles(recs),
                "near_cap": sum(r["status"] == "ok" and r["wall_s"] >= NEAR_CAP * cap for r in recs),
                "isolation": self._plan.isolation, "rows_sha256": records_sha256(records)}

    def solution_metrics(self, trace: vf.Trace) -> dict[str, float]:
        info = trace.info["transitbench"]
        return {"isolated": float(self._plan.isolation == "jail"), "scan_flags": float(info["scan"]["n_flags"])}
