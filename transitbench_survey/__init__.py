"""The transitbench-survey taskset.

One submission, frozen and graded once on unseen stars. Two tasks give the agent the same box: the survey image's
196 labelled dev stars and the same TASK.md, which describes release r1.

- survey-r1 grades samples A and B and the dev set in one runner pass. The reward is S on A; S on B and on dev are
  metrics, so the dev-to-test gap comes from the same frozen solution and grader run.
- survey-dev grades only the dev set, whose stars are already in the image: the model-free checks (gold, noop,
  regrades) and the check that grading reproduces run_dev.py star by star. Its reward is S on dev.

Traces carry aggregates only, since a solution controls its rows, statuses and CPU times and could use them to carry
test-star fingerprints into a trace: per-star records and the full reports go to the private sidecar
<data>/private/grades/<release>/<trace id>.json.
"""

import asyncio
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from transitbench import score
from transitbench_env import grading, hidden
from transitbench_env.harness import TransitBashHarness
from transitbench_env.hidden import HiddenSet
from transitbench_env.task import (
    HEADLINE,
    IMAGES,
    TransitData,
    TransitTask,
    TransitTaskConfig,
    run_counts,
    scorer_gold,
    task_data,
    truth_sha256,
)

__all__ = ["SurveyTaskset", "TransitBashHarness"]

TASKS = (("r1", ("A", "B", "dev")), ("dev", ("dev",)))


class SurveyData(TransitData):
    preset: Literal["survey"] = "survey"
    release: Literal["r1", "dev"]
    samples: tuple[Literal["A", "B", "dev"], ...] = Field(min_length=1)
    """The sets graded, in one runner pass; the reward is S of the first."""


def survey_data(release: str, samples: tuple[str, ...], image: str) -> SurveyData:
    return task_data(SurveyData, image=image, id=f"survey-{release}", name=f"survey-{release}", release=release,
                     samples=samples)


class SurveyTask(TransitTask[SurveyData]):
    POLL_S = 15.0

    @property
    def key(self) -> str:
        return self.data.id

    def hidden_sets(self) -> list[HiddenSet]:
        return hidden.survey_sets(self.data_dir(), self.data.release, self.data.samples)

    async def validate(self, trace: vf.Trace, runtime: vf.Runtime) -> bool:
        """survey-dev: the oracle through grading (TransitTask.validate). survey-r1: the truth oracle scores S = 1 on
        every sample, on the evaluator only, since test truth never enters a box."""
        if self.data.release == "dev":
            return await super().validate(trace, runtime)
        with self.private_errors(trace, "validate"):
            sets = await asyncio.to_thread(self.hidden_sets)
        return all([await asyncio.to_thread(scorer_gold, s) for s in sets])

    def score_records(self, trace: vf.Trace, records: dict[str, dict]) -> float:
        """S per set; the sidecar keeps every record and report, the trace the aggregates."""
        reports, runs = {}, {}
        for name, recs in self.grouped(records).items():
            catalogue, inputs = score.catalogue_from_runs(recs)
            reports[name] = score.report(score.parse_truth(self._sets[name].truth), catalogue)
            reports[name]["inputs"].update(inputs)
            runs[name] = run_counts(list(recs.values()))
        info = trace.info["transitbench"]
        sidecar = f"{self.data.release}/{trace.id}"
        grading.write_private(self.data_dir(), "grades", sidecar, {
            "v": info["v"], "trace_id": trace.id, "task": self.data.id, "solution_sha256": info["solution"]["sha256"],
            "plan_digest": self._plan.digest, "status": self._plan.status, "records": records, "reports": reports})
        grade = self.grade_common(records)
        grade["inputs"]["truth_sha256"] = {name: truth_sha256(s.truth) for name, s in self._sets.items()}
        info["grade"] = {**grade, "headline": {name: {k: rep["headline"][k] for k in HEADLINE}
                                               for name, rep in reports.items()},
                         "runs": runs, "sidecar": sidecar}
        for name, rep in reports.items():
            for k in ("S", "C", "F_per_100", "eb_as_planet"):
                trace.record_metric(f"{k}_{name}", rep["headline"][k])
        trace.record_metrics({
            **self.solution_metrics(trace),
            "n_timeout": sum(r["timeout"] for r in runs.values()), "n_crash": sum(r["crash"] for r in runs.values()),
            "solution_bytes": info["solution"]["bytes"],
            "solution_rejected": float(info["solution"]["status"] == "rejected")})
        return reports[self.data.samples[0]]["headline"]["S"]


class SurveyConfig(vf.TasksetConfig):
    image: str = IMAGES["survey"]
    """The survey image (base plus the dev set); transitbench-survey:local for local Docker."""
    task: TransitTaskConfig = TransitTaskConfig()


class SurveyTaskset(vf.Taskset[SurveyTask, SurveyConfig]):
    def load(self) -> Iterator[SurveyTask]:
        # Always both tasks: grading checks the private data, load() reads none.
        for release, samples in TASKS:
            yield SurveyTask(survey_data(release, samples, self.config.image), self.config.task)
