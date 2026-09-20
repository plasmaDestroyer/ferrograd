# Phase-B workflow benchmark

## Result

Ferrograd was faster than both references in all 19 measured one-thread
workflow cases. Speedups ranged from **1.15x to 4.45x versus bounded-memory
NumPy** and **6.97x to 319.79x versus rliable 1.2.0**. These results apply to
the measured 2,000-repetition workflow matrix. They do not transfer the older
50,000-repetition IQM gate result to other operations.

The matrix contains 57 one-thread rows and six representative Rust scaling
rows. Each row records one untimed warm-up, five complete Python-call timing
samples, and a separate fresh-process peak RSS measurement. Raw samples and
outputs are in [workflow-phase-b.jsonl](results/workflow-phase-b.jsonl), with
environment and source hashes in
[workflow-phase-b.metadata.json](results/workflow-phase-b.metadata.json).

## One-thread results

Times are median seconds. Parentheses contain the minimum and maximum of five
samples. `N/F` and `R/F` are NumPy/Ferrograd and rliable/Ferrograd ratios.

| Operation | Data / thresholds | Ferrograd | NumPy | rliable | N/F | R/F |
|---|---|---:|---:|---:|---:|---:|
| Aggregates | SPR | 0.0419 (0.0413-0.0434) | 0.1067 (0.1010-0.1150) | 0.3122 (0.3090-0.3165) | 2.55x | 7.46x |
| Aggregates | DrQ(eps) | 0.0421 (0.0413-0.0425) | 0.1049 (0.1036-0.1066) | 0.3144 (0.3083-0.3191) | 2.49x | 7.47x |
| Aggregates | IRIS | 0.0045 (0.0043-0.0046) | 0.0062 (0.0060-0.0063) | 0.1906 (0.1870-0.1941) | 1.39x | 42.46x |
| Score profile | SPR / 25 | 0.0388 (0.0376-0.0398) | 0.1034 (0.1010-0.1075) | 0.2796 (0.2761-0.2822) | 2.67x | 7.21x |
| Score profile | DrQ(eps) / 25 | 0.0384 (0.0379-0.0387) | 0.1002 (0.0991-0.1033) | 0.2672 (0.2645-0.2725) | 2.61x | 6.97x |
| Score profile | IRIS / 25 | 0.0031 (0.0031-0.0032) | 0.0066 (0.0064-0.0067) | 0.1729 (0.1722-0.1748) | 2.09x | 55.12x |
| Score profile | SPR / 121 | 0.0873 (0.0861-0.0884) | 0.3796 (0.3683-0.3843) | 1.0395 (1.0282-1.0416) | 4.35x | 11.90x |
| Score profile | DrQ(eps) / 121 | 0.0863 (0.0860-0.0889) | 0.3840 (0.3794-0.3980) | 1.0632 (1.0572-1.0718) | 4.45x | 12.32x |
| Score profile | IRIS / 121 | 0.0072 (0.0070-0.0075) | 0.0226 (0.0218-0.0231) | 0.7289 (0.7254-0.7409) | 3.15x | 101.58x |
| Task profile | SPR / 25 | 0.0320 (0.0320-0.0321) | 0.0430 (0.0427-0.0491) | 0.4065 (0.4042-0.4271) | 1.34x | 12.69x |
| Task profile | DrQ(eps) / 25 | 0.0303 (0.0283-0.0318) | 0.0349 (0.0332-0.0402) | 0.4445 (0.4433-0.4607) | 1.15x | 14.67x |
| Task profile | IRIS / 25 | 0.0027 (0.0027-0.0029) | 0.0039 (0.0039-0.0045) | 0.3032 (0.3000-0.3067) | 1.42x | 110.65x |
| Task profile | SPR / 121 | 0.0292 (0.0290-0.0310) | 0.0403 (0.0391-0.0419) | 1.6051 (1.6007-1.6290) | 1.38x | 54.97x |
| Task profile | DrQ(eps) / 121 | 0.0288 (0.0286-0.0292) | 0.0378 (0.0371-0.0386) | 1.6361 (1.6257-1.6520) | 1.31x | 56.79x |
| Task profile | IRIS / 121 | 0.0038 (0.0038-0.0039) | 0.0104 (0.0103-0.0112) | 1.1920 (1.1812-1.2108) | 2.73x | 314.23x |
| Comparison | SPR-DrQ(eps) | 0.2955 (0.2947-0.2971) | 0.9440 (0.8983-1.0566) | 13.6711 (13.5586-13.6830) | 3.20x | 46.27x |
| Comparison | SPR-IRIS | 0.0415 (0.0410-0.0417) | 0.1280 (0.1267-0.1307) | 13.2628 (13.0570-13.5328) | 3.09x | 319.79x |
| Full workflow | 25 | 0.6126 (0.6073-0.6173) | 1.6675 (1.6346-1.6704) | 42.7125 (42.5749-43.0155) | 2.72x | 69.72x |
| Full workflow | 121 | 0.7101 (0.7001-0.7283) | 2.2647 (2.2435-2.3586) | 47.6496 (47.2725-51.9415) | 3.19x | 67.10x |

The comparison and full-workflow rows are the slow cases. A full workflow
computes four aggregates, both profile definitions, and all three pairwise
comparisons for SPR, DrQ(eps), and IRIS. The standalone comparison matrix
measures SPR-DrQ(eps) and SPR-IRIS; the full workflow also includes
DrQ(eps)-IRIS. IRIS has five runs while SPR and DrQ(eps) have 100, so the
SPR-IRIS and DrQ(eps)-IRIS comparisons cover unequal run counts.

On the full 121-threshold workflow, Ferrograd saved 1.5546 seconds per call
versus NumPy and 46.9395 seconds versus rliable at this repetition count.
This end-to-end result is separate from the historical IQM-only gate.

Peak process RSS, including Python and imports, ranged from 32.5-35.0 MiB for
Ferrograd, 38.7-97.5 MiB for NumPy, and 162.9-169.4 MiB for rliable. Increment
above the baseline measured before backend-specific imports was at most 1.9
MiB, 64.4 MiB, and 136.2 MiB, respectively. The NumPy batching budget uses
conservative estimates of 40 bytes per score cell for aggregates and 24 bytes
per score cell for profiles, with a 64 MiB target.

The NumPy comparison point estimate uses sorting and `searchsorted`. Its
bootstrap uses bounded batches of direct run-pair broadcasts, whose work grows
with the product of both run counts. It is an independent vectorized baseline,
not a claim about the fastest possible NumPy rank-based implementation.

## Rust scaling samples

Scaling uses SPR and one representative operation from each output family.
No two- or four-thread full-workflow row was measured.

| Operation | 1 thread | 2 threads | 4 threads | 1t/2t | 1t/4t |
|---|---:|---:|---:|---:|---:|
| Aggregates | 0.0419 | 0.0215 | 0.0116 | 1.94x | 3.61x |
| Score profile, 121 | 0.0873 | 0.0479 | 0.0273 | 1.82x | 3.20x |
| SPR-DrQ(eps) comparison | 0.2955 | 0.1714 | 0.1039 | 1.72x | 2.84x |

## Definitions and correctness

All inputs are two-dimensional `(runs, tasks)` float64 arrays. Atari scores
use per-game human normalization. Aggregate output contains IQM over all score
cells, mean of task means, median of task means, and optimality gap at gamma
1.0. Score profiles count individual scores strictly above each threshold;
task profiles count task means strictly above it. Comparisons average the
probability that a run from the left algorithm exceeds a run from the right,
with ties worth one half. Percentile intervals use 95% confidence and resample
runs independently within each task. This matrix does not use task bootstrap.

Ferrograd point estimates matched both references in every comparable row;
the largest absolute difference was `4.44e-16`. Bootstrap endpoints need not
match because the three backends use independent RNG implementations. Fixed
index tests independently calculate each bootstrap replicate before comparing
percentiles, including task bootstrap and unequal-run comparisons.

The measured native module matches the wheel member SHA-256
`eb72b8f3287e820f7f2c080756f4ca38fb84798d01b6ddeeee5751d1ff5ebb22`.
Core Rust source SHA-256 is
`84e1c5ecee2d52d481172dba7e27c34ff968e3f73bbd0c81bcf70c21b229fb7a`;
the Python wrapper is
`3026e5a2ac982c32a8c8950158cbb80bb95d8752ba70ce936c6910e6927fa016`.
No core or wrapper change separates this matrix from the final-package IQM
benchmark.

Hardware was an Intel Core i9-12900H with 14 physical cores and 20 logical
CPUs, running Linux 7.2.5. Single-thread calls were pinned to logical CPU 4;
two- and four-thread calls used one logical CPU per physical core. The CPU
governor was `powersave` with `balance_performance` energy preference. BLAS
thread counts were fixed at one. The pinned environment used Python 3.12.14,
NumPy 2.5.3, rliable 1.2.0, and arch 7.2.0.

## Reproduction

Use a new output path after any source or environment change. Run commands
sequentially; the shared lock prevents concurrent writers.

```sh
OUT=benchmarks/results/workflow-phase-b.jsonl
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output "$OUT" --operation aggregates --reps 2000
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output "$OUT" --operation profile-score --thresholds 25 --reps 2000
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output "$OUT" --operation profile-score --thresholds 121 --reps 2000
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output "$OUT" --operation profile-task --thresholds 25 --reps 2000
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output "$OUT" --operation profile-task --thresholds 121 --reps 2000
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output "$OUT" --operation compare --reps 2000
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output "$OUT" --operation workflow --thresholds 25 --reps 2000
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output "$OUT" --operation workflow --thresholds 121 --reps 2000
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output "$OUT" --operation aggregates --case spr --reps 2000 --scaling
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output "$OUT" --operation profile-score --case spr --thresholds 121 --reps 2000 --scaling
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output "$OUT" --operation compare --case spr-drq --reps 2000 --scaling
```
