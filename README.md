# TransitBench

TransitBench is a benchmark for autonomous scientific programming in astronomy. A coding agent gets one year of real
Kepler light curves for 196 labelled dev stars and a weak starter search, and builds a planet-search pipeline in
Python. When the session ends, the pipeline is frozen and runs once, with no language model, on 2,000 hidden stars.
There it has to find the planets I injected and stay quiet on eclipsing binaries and on stars where Kepler's own search
fired on something that wasn't a planet. It's scored on planets found minus false candidates.

The full results, every submission's trace and the write-up are at
[transitbench.eigenpaul.com](https://transitbench.eigenpaul.com/). The environment is on Prime Intellect's
Environments Hub as
[`eigenpaul/transitbench`](https://app.primeintellect.ai/dashboard/environments/eigenpaul/transitbench). MIT license
([LICENSE](LICENSE)).

## Install

You need git and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/eigenpaul/transitbench.git
cd transitbench
uv sync --python 3.12 --group build               # on macOS: --python 3.13
uv run python scripts/smoke_test.py               # ends with: smoke test: 3 of 3 checks passed
```

## The task and the score

- **Task.** The agent gets one prompt, TASK.md, rendered from [a template](transitbench_env/templates/TASK.md),
  and a shell in a box with no network. The box holds the 196 labelled dev stars and their answers, the starter in
  `solution/`, the grading runner and scorer, and `run_dev.py`, which runs the current solution on the dev stars the
  way grading runs it and prints the score. When the session ends, `solution/` is graded once in a fresh box on
  hidden stars that never enter the agent's box.
- **Output.** Grading runs `solution/solution.py` once per star, in a fresh process with 60 s, one core and no
  network. It prints one row of `period_days,t0_days,score` per planet it claims, and only the 3 rows with the
  highest score count. A crash or a timeout gives the star no rows.
- **Matching.** A row earns full credit for a planet when its period is within 1% of the planet's and at least 75%
  of the transits line up, counted from both sides ([transitbench/match.py](transitbench/match.py)). At a
  third, half, double or triple the period it earns half credit. Rows are matched highest score first, each planet
  can be claimed once, and every row left over is a false candidate.
- **Score.** S = (credit - false candidates) / planets ([transitbench/score.py](transitbench/score.py)).
  Reporting nothing scores 0 and a perfect pipeline 1, and a pipeline with more false candidates than planets found
  scores below 0. Next to S I report completeness C (credit per planet), F (false candidates per 100 scored stars)
  and the share of flagged eclipsing binaries that got a candidate. The site shows S as points, 100 S.
- **Stars.** The test set A and the replication set B hold 1,000 stars each: 400 with a planet I injected, 350
  untouched, 150 real eclipsing binaries and 100 hosts of confirmed planets. Half of the injected and untouched
  stars are random Kepler stars, and the other half are stars where Kepler's search fired on a signal its vetting
  later rejected. The injected planets' MES, roughly the signal-to-noise of all their transits together, runs from 4
  to 20, on both sides of the 7.1 Kepler needed for a detection. A binary only counts when its year of data gives it
  away, for example with a secondary eclipse, and the confirmed-planet hosts don't count at all. That leaves 864
  scored stars in A and 870 in B, with 400 planets each. A gives the score and B checks it on different stars.
- **Budget.** A session gets 60 turns, 200,000 tokens and at most 4 hours, the caps the site's model results ran
  with. The rendered TASK.md states them, and they come from `PRESETS` in
  [transitbench_env/task.py](transitbench_env/task.py).

## Baseline results

[results/baselines.json](results/baselines.json) holds five reference pipelines that use no language model:

| Row | What it is | S on A | S on B |
| --- | --- | ---: | ---: |
| `empty` | no rows on any star | 0 | 0 |
| `starter` | the BLS search the agent starts from ([workspace/solution](workspace/solution/solution.py)), which reports its strongest peak if that has SNR >= 8 and is no deeper than 3% | 0.046 | 0.076 |
| `naive_v0` | a plain BLS search with no vetting ([baselines/naive_v0](baselines/naive_v0/solution.py)) | -0.159 | -0.155 |
| `learned_vetter` | up to three BLS peaks per star, kept or dropped by a gradient-boosted classifier trained on labelled stars outside dev, A and B, then frozen ([features](baselines/learned_vetter/features.py)) | 0.696 | 0.673 |
| `reference_detector` | reads the answers and claims every planet with MES >= 7.1 at its true ephemeris | 0.805 | 0.805 |

- The starter finds 68% of the planets in A but raises 29 false candidates per 100 stars, so it barely beats
  printing nothing. `naive_v0` does no vetting at all, scores below printing nothing and claims a planet on every
  flagged binary in A.
- The learned vetter gets to about 0.7. The repository has its feature code, without the training code or the
  trained classifier.
- The reference detector cheats on purpose. It scores what a perfect search at Kepler's detection threshold would,
  and a pipeline that finds planets below MES 7.1 can beat it.

## Tasksets

TransitBench is two [verifiers](https://github.com/PrimeIntellect-ai/verifiers) tasksets, `transitbench-survey` for
the benchmark and `transitbench-mini` for RL, both in the Hub environment `eigenpaul/transitbench`:

| Taskset id | Tasks | Box | Reward |
| --- | --- | --- | --- |
| `transitbench-survey` | `survey-r1` (A, B and dev in one pass); `survey-dev` (dev only, for model-free checks) | survey image, 196 dev stars | S on A (`survey-dev`: S on dev); S on B and dev are metrics |
| `transitbench-mini` | `mini-train-00000000`, ... (never ends); `--taskset.split dev` gives 32, `mini-dev-00000000` to `mini-dev-00000031` | base image, 12 labelled stars | clip((credit - false) / 8, -1, 1) on 16 hidden stars |

Each mini task's 16 hidden stars hold 8 injected planets, drawn from the train and dev pools, which share no star
with A and B. There's no format reward, so a crash or broken output counts as reporting nothing.

There's one example prime-rl eval config per taskset, [configs/examples/survey.toml](configs/examples/survey.toml)
and [configs/examples/mini.toml](configs/examples/mini.toml), and each file's header says how to install and run
it. [scripts/build.py](scripts/build.py) builds the dev set and the test samples and
[scripts/mini_pool.py](scripts/mini_pool.py) the mini pool, from the Kepler light curves and tables that
[scripts/fetch_lightcurves.py](scripts/fetch_lightcurves.py) and [scripts/fetch_tables.py](scripts/fetch_tables.py)
download.
