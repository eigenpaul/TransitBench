"""The transitbench-mini taskset, the RL training unit.

Each task gives the agent 12 labelled stars in /app/data and grades its frozen solution on 16 hidden stars, drawn
from the mini pool. The task data carries only the token ("train-00000042"): the stars are a pure function of the
pool, the evaluator secret and the token (mini_draw.draw; variability matching is a task data flag, off by
default), rebuilt on the worker and never on the wire. The reward
is clip((credit - false) / 8, -1, 1) on the hidden stars; traces carry per-star detail, since the pool is public.

The train split never ends (select a bound with -n); the dev split holds 32 tasks.
"""

import collections
import concurrent.futures
import functools
import itertools
import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

from transitbench import score
from transitbench_env import box, hidden, mini_draw
from transitbench_env.harness import TransitBashHarness
from transitbench_env.hidden import HiddenSet
from transitbench_env.task import (
    IMAGES,
    SOLUTION,
    TransitData,
    TransitTask,
    TransitTaskConfig,
    installed_sha,
    run_counts,
    solution_entries,
    task_data,
)

__all__ = ["MiniTaskset", "TransitBashHarness"]

DEV_TASKS = 32
DRAW_CACHE = 64
"""Draws a worker keeps, about 5 MB each (4.6 MB of star files), above the 40 tokens in flight at a max_inflight of
320 with groups of 8."""


class MiniData(TransitData):
    preset: Literal["mini"] = "mini"
    split: Literal["train", "dev"]
    token: str
    """f"{split}-{i:08d}": meaningless without the evaluator secret."""
    pool: str
    """The pool manifest's sha256[:16]; a worker with another pool refuses the task (TB-E002)."""
    builder: str = mini_draw.BUILDER
    """The draw's code version; a worker with another one refuses the task (TB-E002)."""
    match_variability: bool = False
    """Draw INJ and NULL hosts matched to the EBs' variability (mini_draw); off by default, since matching makes
    abstaining on variable stars pay."""


def mini_data(split: str, token: str, pool: str, image: str, match_variability: bool = False) -> MiniData:
    return task_data(MiniData, image=image, id=f"mini-{token}", split=split, token=token, pool=pool,
                     match_variability=match_variability)


def shared_cache(maxsize: int) -> Callable[[Callable], Callable]:
    """functools.lru_cache for slow calls from many threads: concurrent calls with one key wait for a single
    computation instead of each running it. A failure is raised to every waiter and not cached."""

    def wrap(fn):
        lock, entries = threading.Lock(), collections.OrderedDict()

        @functools.wraps(fn)
        def cached(*key):
            with lock:
                future = entries.get(key)
                owner = future is None
                if owner:
                    future = entries[key] = concurrent.futures.Future()
                    while len(entries) > maxsize:
                        entries.popitem(last=False)
                else:
                    entries.move_to_end(key)
            if owner:
                try:
                    future.set_result(fn(*key))
                except BaseException as e:  # noqa: BLE001 - handed to every waiter, then raised
                    with lock:
                        if entries.get(key) is future:
                            del entries[key]
                    future.set_exception(e)
            return future.result()

        cached.cache_clear = entries.clear
        return cached

    return wrap


@shared_cache(maxsize=4)
def _pool(data_dir: Path, pool: str) -> mini_draw.Pool:
    return mini_draw.load_pool(data_dir, pool)


@shared_cache(maxsize=DRAW_CACHE)
def cached_draw(data_dir: Path, pool: str, split: str, token: str, match_variability: bool) -> mini_draw.MiniDraw:
    """One task's stars, drawn once per worker while the task is among the DRAW_CACHE most recently used: the rollouts
    of a group and every hook that needs them (setup in the agent and grading boxes, staging, scoring) share one
    draw, also while it is being computed. Blocking: call from a thread."""
    return mini_draw.draw(_pool(data_dir, pool), mini_draw.read_secret(data_dir), split, token,
                          match_variability=match_variability)


@functools.cache
def starter_sha(max_bytes: int, max_files: int) -> str:
    return installed_sha(solution_entries(box.starter()), max_bytes, max_files)


class MiniTask(TransitTask[MiniData]):
    CLEAR = (SOLUTION, "/app/transitbench", "/app/data")
    POLL_S = 2.0

    @property
    def key(self) -> str:
        return self.data.token

    def draw(self) -> mini_draw.MiniDraw:
        d = self.data
        if d.builder != mini_draw.BUILDER:
            raise mini_draw.MiniPoolError(mini_draw.E_POOL)
        return cached_draw(self.data_dir(), d.pool, d.split, d.token, d.match_variability)

    def box_data(self) -> dict[str, bytes]:
        draw = self.draw()
        return {"truth.json": (json.dumps(draw.box_truth, indent=1) + "\n").encode(),
                **{f"stars/{sid}.npz": body for sid, body in draw.box_files.items()}}

    def hidden_sets(self) -> list[HiddenSet]:
        return [hidden.mini_set(self.draw())]

    def score_records(self, trace: vf.Trace, records: dict[str, dict]) -> float:
        """The reward on the hidden stars, with per-star detail in the trace."""
        (name, recs), = self.grouped(records).items()
        draw = self.draw()
        stars = score.parse_truth(self._sets[name].truth)
        catalogue, _ = score.catalogue_from_runs(recs)
        reward = score.reward(stars, catalogue)
        per_star = {}
        for star in stars:
            r, h = recs[star.star_id], score.score([star], {star.star_id: catalogue[star.star_id]})
            per_star[star.star_id] = {"role": draw.roles[star.star_id], "status": r["status"], "rows": r["rows"],
                                      "n_invalid": r["n_invalid"], "cpu_s": r["cpu_s"], "wall_s": r["wall_s"],
                                      "credit": h["credit"], "false": h["false"]}
        total = score.score(stars, catalogue)
        runs = run_counts(list(recs.values()))
        info = trace.info["transitbench"]
        grade = self.grade_common(records)
        grade["inputs"]["hidden_digest"] = draw.digest
        info["grade"] = {**grade, "reward": reward, "credit": total["credit"], "false": total["false"],
                         "runs": runs, "stars": per_star}
        changed = info["solution"]["sha256"] != starter_sha(self.data.solution_max_bytes, self.data.solution_max_files)
        trace.record_metrics({**self.solution_metrics(trace), "credit": total["credit"], "false": total["false"],
                              "n_timeout": runs["timeout"], "n_crash": runs["crash"],
                              "solution_changed": float(changed)})
        return reward


class MiniConfig(vf.TasksetConfig):
    split: Literal["train", "dev"] = "train"
    image: str = IMAGES["base"]
    """The base image (no dev set); transitbench-base:local for local Docker."""
    pool: str = mini_draw.MINI_POOL["sha"]
    match_variability: bool = False
    """Match INJ and NULL hosts to the EBs' variability (mini_draw); off by default."""
    task: TransitTaskConfig = TransitTaskConfig()


class MiniTaskset(vf.Taskset[MiniTask, MiniConfig]):
    INFINITE = True
    """The train split; the dev split stops after DEV_TASKS. Every run passes a bound (-n) anyway."""

    def load(self) -> Iterator[MiniTask]:
        c = self.config
        for i in itertools.count() if c.split == "train" else range(DEV_TASKS):
            yield MiniTask(mini_data(c.split, f"{c.split}-{i:08d}", c.pool, c.image, c.match_variability), c.task)
