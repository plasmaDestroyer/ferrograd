# Ferrograd 0.1: current evaluation contract

Ferrograd is a Rust/PyO3 backend for rliable-style analysis of **2D final
scores** `(runs, tasks)`. The earlier tensor autodiff proposal is archived in
[docs/HISTORICAL_AUTODIFF_SPEC.md](docs/HISTORICAL_AUTODIFF_SPEC.md) and does not
describe this implementation.

## Public API

`get_interval_estimates(score_dict, *, metrics=("iqm",), reps=50_000,
confidence=0.95, task_bootstrap=False, seed=0, threads=1, gamma=1.0)` returns
ordered metric point arrays and `(lower, upper)` percentile arrays. Supported
names are `iqm`, `mean`, `median`, `optimality_gap`; order follows `metrics`.
IQM flattens and trims `floor(n/4)` values from each end. Mean and median use
per-task means. Gap is `gamma - mean(min(score, gamma))`.

`create_performance_profile(score_dict, thresholds, *, reps=2000,
confidence=0.95, task_bootstrap=False, seed=0, threads=1,
use_score_distribution=True)` returns fractions strictly above each threshold.
The default counts all run/task scores; `False` counts per-task means.

`compare(score_pairs, *, reps=2000, confidence=0.95, seed=0, threads=1)` takes
named `(x, y)` pairs with matching task counts and computes the average over
tasks of all run-pair wins, with ties worth one half.

Each function returns two dictionaries with input insertion order. Point
arrays are one-dimensional. Interval arrays have shape `(2, number of points)`.
The run bootstrap is stratified by task. Optional task bootstrap additionally
resamples columns. Percentile endpoints use linear interpolation. Seeds produce
the same output across thread counts in a pinned build, but do not reproduce
rliable's NumPy RNG samples. All inputs must be finite, real, nonempty 2D
arrays; conversion yields owned contiguous float64 data for the Rust work.

## Evidence and decision

Correctness checks use identical precomputed resample indices against pinned
`rliable==1.2.0`/`arch==7.2.0` and NumPy percentiles. The performance gate
required 3× speed over rliable and 1.5× over batched NumPy at 50,000
replicates on Atari SPR. The installed final-package wheel passed in a
single-thread run: 0.926 s versus 5.820 s rliable (6.28×) and 2.117 s
batched NumPy (2.29×). The [final-package report](benchmarks/PACKAGE_REPORT.md)
has samples, the 100-configuration matrix, and environment details. The
earlier [partition gate](benchmarks/PARTITION_REPORT.md) also passed; the
original [sorting report](benchmarks/REPORT.md) documents the failed baseline.
These timings cover IQM, not every current API method or platform.

## Distribution and scope

CPython 3.11–3.14, NumPy 2.x, native extension `ferrograd._native`. CI is configured to build
interpreter-specific wheels on Linux x86-64 (manylinux2014), macOS x86-64 and
arm64, Windows x86-64, plus an sdist. Every wheel is installed and smoke
tested in its own native runner; Python 3.12 also runs pinned reference parity.
Artifacts are uploaded to CI, not published to a package index.

No 3D curves, GPU, CLI, callbacks, or cross-library RNG identity are promised.
