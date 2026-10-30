# TransitBench: find transiting planets in Kepler light curves

Write a pipeline that reads one star's light curve and prints the transiting planets it finds. When your session
ends, `solution/` is frozen and graded once on hidden stars that never enter this box.

## This box (/app)

- `solution/solution.py`: your submission. It starts as a working naive search; change anything.
- `data/stars/*.npz` and `data/truth.json`: {labelled} labelled stars and their answers.
- `run_dev.py`: runs your solution on the labelled stars as grading runs it, and scores it (see below).
- `transitbench/`: the grading runner, matcher (`match.py`) and scorer (`score.py`). `caps.json`: the numbers here.

## Input and output

- Each run gets one star's `.npz` file on stdin, which is a pipe: read it whole, for example
  `np.load(io.BytesIO(sys.stdin.buffer.read()))`. It holds `time` (days, shifted by a per-star offset), `flux`
  (normalised to 1 in each quarter), `quarter` (0-3) and the star's `teff`, `logg`, `radius` and `mass`.
- Print one CSV row `period_days,t0_days,score` per planet you claim, with t0 on the file's `time` axis. A row is
  three finite numbers with a period above 0; other non-blank lines are ignored. Only the {max_rows}
  highest-scoring rows count. Printing nothing is allowed.
- Rows count only if the process exits with status 0 within the cap. More than {stdout_max_bytes} of stdout in all
  gives the star no rows. stderr is discarded.

## How grading runs each star

`python3 -s solution/solution.py`, a fresh process for every star, {workers} stars at a time, in shuffled order.
Each star:

- may run for {star_cap} of wall time; then all its processes are killed and it gets no rows. A CPU-time limit of
  {cpu_backstop} per process backs this up;
- gets one CPU core, {memory} of memory, at most {nproc} processes and {nofile} open files, files up to {fsize},
  and no network. A star that runs out of memory gets no rows;
- runs as an unprivileged user with a cleared environment (`PYTHONHASHSEED=0`, one BLAS and OpenMP thread);
- sees `/usr`, `/etc` and its solution at `/app/solution`, all read-only, and nothing else of this box: not `data/`,
  not `transitbench/`. Import only modules in `solution/` and the libraries below;
- starts in an empty private directory that is its working directory, `HOME` and `TMPDIR` ({work}); `/dev/shm` is
  private ({shm}) and `/tmp` is not writable. Nothing persists from one star to the next.

Libraries: {libraries}.

<!-- survey -->
## Score

Grading runs the captured `solution/` in a fresh box of this image and shape on hidden samples A ({hidden_A} stars)
and B ({hidden_B} stars) and on the {hidden_dev} dev stars. A sample holds the dev set's populations: INJ (a planet
injected into a real star, half on random stars and half on stars where Kepler's search fired on something that
was not a planet), NULL (the same two kinds of star, no planet), EB (eclipsing binaries) and REAL (hosts of
confirmed planets). A and B each have 400 INJ, 350 NULL, 150 EB and 100 REAL stars.

Your score is S on sample A; S on B and on dev are reported with it.
S = (sum of match credit - false candidates) / planets, over the INJ and NULL stars and the EB stars whose
delivered year shows EB evidence (a secondary eclipse, odd/even depths, or a depth too large for a planet). REAL
stars do not enter S; they are reported separately.
<!-- end -->
<!-- mini -->
## Reward

Grading runs the captured `solution/` in a fresh box with the same caps on {hidden_total} hidden stars drawn from
the same pool as `data/`: {hidden_inj} INJ (a planet injected into a real star), {hidden_null} NULL (no planet) and
{hidden_eb} EB (eclipsing binaries whose delivered year shows EB evidence). Your reward is
r = clip((sum of match credit - false candidates) / {norm}, -1, 1).
<!-- end -->

Matching (`transitbench/match.py`): a row earns 1 when its period is within 1% of a planet's and at least 75% of the
planet's transits in the data have a row transit within half the transit duration plus 30 minutes, and the other
way round; it earns 0.5 at 1/3, 1/2, 2 or 3 times the period with partial overlap. Rows are matched in score order
and each planet is claimed once. Every other row is a false candidate. A crash or timeout gives the star no rows.

## Your session

- {max_turns} turns and {max_tokens} tokens in total. The session stops after {wall} at the latest, and `solution/`
  is graded as it is at that moment. Every tool result ends with a `[budget]` line: the turns and tokens
  used so far and the time left. When the turn or token limit ends the session, the commands of your last turn still
  run, but you do not see their output. A reply that calls no tool also ends the session.
- Each bash command is killed after {command_cap} and returns its output so far with a timeout error. Background
  processes end with the command that started them. A tool result over 20,000 bytes keeps only its head and tail.
  There is no network.
- A command whose stdout and stderr together pass 64 MiB is killed, and its result ends with an error saying so.
- `python3 run_dev.py` snapshots `solution/` under the rules below, runs it on every labelled star with the grading
  runner and caps, prints each star as it finishes, then the score. `--n N [--seed S]` runs a random subset,
  `--stars ID ...` named stars, `--json` prints JSON lines. It sandboxes each star as grading does when this box
  allows it, and says which (`isolation: jail` or `none`).

## Submission rules

- `solution/` may hold {solution_max_bytes} of file content in at most {solution_max_files} files and directories,
  plus at most {solution_max_files} symlinks, FIFOs, sockets and devices. Everything counts, including what is dropped
  below. Over a limit, or unreadable, it is graded as an empty solution.
- Only regular files and directories are kept: symlinks, FIFOs, sockets, devices and names that are not UTF-8 are
  dropped, hard links become copies, and file modes become 0644 (0755 if executable). Python bytecode
  (`__pycache__/`, `.pyc`, `.pyo`) is dropped too, so only your source runs, but it still counts toward the limits.
- A static scan reports, and never penalises: number tables with more than 200 numbers, string or bytes constants
  over 8 KiB, calls that decode data (base64, binascii, `bytes.fromhex`, `codecs.decode`, zlib, gzip, bz2, lzma,
  pickle, marshal, `np.frombuffer`, `np.fromstring`, `struct.unpack`), files other than `.py`, and files that do
  not parse.
