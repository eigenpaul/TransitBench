"""Run solution/ on the labelled stars exactly as grading runs it, and score it.

  python3 run_dev.py                      every labelled star
  python3 run_dev.py --n 20 --seed 1      20 stars drawn at random (seed 1)
  python3 run_dev.py --stars ID [ID ...]  the named stars
  python3 run_dev.py --json               JSON lines instead of text
  python3 run_dev.py --solution DIR       another solution directory (default: solution/ in this box)

It snapshots the solution the way grading captures it (transitbench/archive.py: the same size and file limits,
counting everything; links, special files, bytecode and non-UTF-8 names then dropped; over a limit, the whole solution
is graded as empty), then runs the snapshot with the grading runner (transitbench/runner.py) under the caps in
caps.json. Each star runs in the runner's jail when it works here, as in grading; otherwise without it, and the first
lines say which and why. Each star prints as it finishes, with the end of its stderr when it did not end "ok". The
score of the subset follows (transitbench/score.py, the grading scorer).

--json prints one JSON object per line: {"kind": "start", ...}, then {"kind": "record", "record": {...}} for each
star as it finishes, then {"kind": "summary", ...} with the statuses, CPU times and the full score report.

Exit status: 0 after a run, 1 if the runner or the box files fail, 2 for bad arguments.

Numpy and the standard library only: this file ships into agent boxes.
"""

import argparse
import hashlib
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

from transitbench import archive, runner, score

ROOT = Path(__file__).resolve().parents[1]
"""The box directory: caps.json, data/, solution/ and the transitbench/ package that holds this file."""
STDERR_TAIL_BYTES = 2048
STDERR_LINES = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_dev.py", description="Run solution/ on the labelled stars as grading "
                                     "runs it, and score it.")
    parser.add_argument("--n", type=int, help="run N stars drawn at random")
    parser.add_argument("--seed", type=int, default=0, help="seed for --n (default 0)")
    parser.add_argument("--stars", nargs="+", metavar="ID", help="run only these stars")
    parser.add_argument("--json", action="store_true", help="print JSON lines")
    parser.add_argument("--solution", type=Path, default=None, help="solution directory (default: solution/)")
    args = parser.parse_args(argv)
    if args.n is not None and args.n < 1:
        parser.error("--n must be at least 1")
    out = _Json() if args.json else _Text()
    try:
        caps = json.loads((ROOT / "caps.json").read_text())
        truth = json.loads((ROOT / "data" / "truth.json").read_text())
    except (OSError, ValueError) as e:
        print(f"run_dev: cannot read the box files: {e}", file=sys.stderr)
        return 1
    known = sorted(truth["stars"])
    unknown = sorted(set(args.stars or ()) - set(known))
    if unknown:
        parser.error(f"unknown stars: {' '.join(unknown)}")
    ids = sorted(set(args.stars)) if args.stars else known
    if args.n is not None and args.n < len(ids):
        ids = sorted(random.Random(args.seed).sample(ids, args.n))
    try:
        stars = [{"id": sid, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for sid, path in ((sid, ROOT / "data" / "stars" / f"{sid}.npz") for sid in ids)]
    except OSError as e:
        print(f"run_dev: cannot read a star file: {e}", file=sys.stderr)
        return 1

    solution = archive.from_dir(args.solution or ROOT / "solution", max_bytes=caps["solution_max_bytes"],
                                max_files=caps["solution_max_files"])
    mode, why = _isolation()
    # The order grading uses: shuffled by the solution's sha, so no solution can count on it.
    stars.sort(key=lambda s: hashlib.sha256(f"{solution.sha256}{s['id']}".encode()).hexdigest())
    out.start(solution, mode, why, caps, [s["id"] for s in stars])
    with tempfile.TemporaryDirectory(prefix="tb-dev-") as tmp:
        snapshot = archive.extract(solution, Path(tmp))
        plan = {"version": 1, "solution": str(snapshot), "cap_s": caps["star_cap_s"], "workers": caps["workers"],
                "isolation": mode, "memory_bytes": caps["memory_bytes"], "stdout_max_bytes": caps["stdout_max_bytes"],
                "stderr_tail_bytes": STDERR_TAIL_BYTES, "stars": stars}
        try:
            records = runner.run_plan(plan, out.record)
        except runner.RunnerError as e:
            print(f"run_dev: the runner failed: {e}", file=sys.stderr)
            return 1
    catalogue, inputs = score.catalogue_from_runs({r["star"]: r for r in records})
    chosen = set(ids)
    report = score.report([s for s in score.parse_truth(truth) if s.star_id in chosen], catalogue)
    report["inputs"].update(inputs)
    out.summary(records, report, caps)
    return 0


def _isolation() -> tuple[str, str | None]:
    """The mode grading's "auto" resolves to here, and why the jail is not used when it is not."""
    mode = runner.resolve_isolation("auto")
    if mode == "jail":
        return mode, None
    probe = runner.probe()
    failed = [name for name, ok in (probe.get("checks") or {}).items() if not ok]
    return mode, probe.get("error") or f"the jail probe failed: {', '.join(failed)}"


def _cpu(records: list[dict]) -> dict:
    cpu = sorted(r["cpu_s"] for r in records)
    if not cpu:
        return {"p50": None, "p95": None, "max": None}
    return {"p50": cpu[min(len(cpu) - 1, len(cpu) // 2)], "p95": cpu[min(len(cpu) - 1, int(0.95 * len(cpu)))],
            "max": cpu[-1]}


def _statuses(records: list[dict]) -> dict[str, int]:
    return dict(Counter(r["status"] for r in records).most_common())


class _Text:
    def __init__(self):
        self.done, self.total = 0, 0

    def start(self, solution: archive.Canonical, mode: str, why: str | None, caps: dict, ids: list[str]) -> None:
        size = sum(f["size"] for f in solution.files)
        state = solution.status if solution.reason is None else f"{solution.status} ({solution.reason})"
        n = len(solution.files)
        line = f"solution: {state}, {n} file{'' if n == 1 else 's'}, {size:,} bytes, sha256 {solution.sha256[:16]}"
        if solution.status == "rejected":
            line += "; graded as an empty solution"
        print(line)
        if solution.dropped:
            print(f"  dropped (links, special files, bytecode, non-UTF-8 names): {', '.join(solution.dropped)}")
        print(f"isolation: {mode}" + (f" ({why})" if why else ""))
        print(f"running {len(ids)} stars on {caps['workers']} workers, {caps['star_cap_s']:g} s cap per star",
              flush=True)
        self.total = len(ids)

    def record(self, rec: dict) -> None:
        self.done += 1
        rows = "; ".join(",".join(f"{v:.9g}" for v in row) for row in rec["rows"])
        line = (f"[{self.done:>{len(str(self.total))}}/{self.total}] {rec['star']} {rec['status']} "
                f"cpu {rec['cpu_s']:.1f} s wall {rec['wall_s']:.1f} s rows {len(rec['rows'])}")
        if rec["n_invalid"]:
            line += f" (+{rec['n_invalid']} invalid lines)"
        print(line + (f": {rows}" if rows else ""))
        tail = (rec.get("stderr") or "").strip().splitlines()[-STDERR_LINES:]
        if rec["status"] != "ok":
            for text in tail:
                print(f"    {text}")
        sys.stdout.flush()

    def summary(self, records: list[dict], report: dict, caps: dict) -> None:
        cpu = _cpu(records)
        print("statuses: " + ", ".join(f"{k} {v}" for k, v in _statuses(records).items()))
        if cpu["max"] is not None:
            print(f"cpu per star: p50 {cpu['p50']:.1f} s, p95 {cpu['p95']:.1f} s, max {cpu['max']:.1f} s "
                  f"(cap {caps['star_cap_s']:g} s)")
        print(score.format_headline(report), flush=True)


class _Json:
    def start(self, solution: archive.Canonical, mode: str, why: str | None, caps: dict, ids: list[str]) -> None:
        self._print({"kind": "start", "isolation": mode, "isolation_note": why,
                     "solution": {"status": solution.status, "reason": solution.reason, "sha256": solution.sha256,
                                  "bytes": sum(f["size"] for f in solution.files), "n_files": len(solution.files),
                                  "files": list(solution.files), "dropped": list(solution.dropped)},
                     "caps": caps, "stars": ids})

    def record(self, rec: dict) -> None:
        self._print({"kind": "record", "record": rec})

    def summary(self, records: list[dict], report: dict, caps: dict) -> None:
        self._print({"kind": "summary", "statuses": _statuses(records), "cpu_s": _cpu(records), "report": report})

    @staticmethod
    def _print(obj: dict) -> None:
        print(json.dumps(obj), flush=True)


if __name__ == "__main__":
    sys.exit(main())
