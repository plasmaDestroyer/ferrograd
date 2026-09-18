# Ferrograd

Rust-backed bootstrap statistics for reinforcement-learning benchmark scores.
Version 0.1 handles finite, two-dimensional final-score arrays shaped
`(runs, tasks)`. It is not an automatic differentiation engine; the older
design is preserved in [the historical specification](docs/HISTORICAL_AUTODIFF_SPEC.md).

## Install

Requires CPython 3.11–3.14 and NumPy 2.x. CI wheel jobs target Linux
x86-64, macOS Intel and Apple Silicon, and Windows x86-64. Build locally with:

```sh
python -m pip install .
```

For benchmark/reference tests, use Python 3.12 and install
`benchmarks/reference-requirements.txt`. The runtime package depends only on
NumPy; reference packages and plotting packages are optional.

## Use

```python
import numpy as np
import ferrograd

# One row per run, one column per task; values should be comparable scores.
scores = {
    "baseline": np.array([[0.2, 0.7], [0.5, 0.9]]),
    "candidate": np.array([[0.4, 0.8], [0.6, 1.1]]),
}
metrics = ("iqm", "mean", "median", "optimality_gap")
points, intervals = ferrograd.get_interval_estimates(
    scores, metrics=metrics, reps=50_000, gamma=1.0, seed=7)
profiles, profile_intervals = ferrograd.create_performance_profile(
    scores, thresholds=np.linspace(0, 1.2, 25), reps=2_000, seed=7)
wins, win_intervals = ferrograd.compare(
    {"candidate,baseline": (scores["candidate"], scores["baseline"])},
    reps=2_000, seed=7)
```

Each call returns `(points, intervals)` as dictionaries keyed like the input.
Metric points have shape `(number of metrics,)`, profile points have shape
`(number of thresholds,)`, and comparison points have shape `(1,)`. Each
interval has shape `(2, number of points)`; its rows are the lower and upper
percentile limits. `examples/plot_comparison.py` shows direct use with
`rliable.plot_utils` when plotting is needed.

## Statistics and boundaries

- IQM flattens all run/task scores, trims `floor(n/4)` from each tail, and
  averages the middle. Mean and median summarize task means. Optimality gap
  is `gamma - mean(min(score, gamma))`.
- Profiles report the fraction of scores strictly greater than each threshold.
  Set `use_score_distribution=False` to threshold task means instead.
- `compare` averages taskwise run-pair win probabilities, giving ties half
  credit. Paired arrays need the same number of tasks; run counts may differ.
- Bootstrap sampling draws runs independently within each task. Set
  `task_bootstrap=True` for interval estimates or profiles to resample task
  columns too. Comparison resamples runs in both algorithms independently.
  Intervals use percentile bounds with linear interpolation.
- `seed` reproduces results across thread counts in a pinned build. Ferrograd
  and NumPy/rliable use different random streams, so their seeded confidence
  intervals need not match sample for sample. Fixed-index tests check exact
  metric parity with rliable.
- Input arrays must be nonempty, finite, real numeric, and two-dimensional.
  Noncontiguous inputs are copied into contiguous float64 storage. `reps >= 2`,
  `threads >= 1`, and `0 < confidence < 1`.

The package has no 3D learning-curve interface, GPU backend, command-line
tool, arbitrary metric callbacks, or matching NumPy random stream. Current
contract and stop/go evidence are in [SPEC.md](SPEC.md) and the
[final-package benchmark](benchmarks/PACKAGE_REPORT.md). Its measured
50,000-rep single-thread Atari IQM call passed the 3× rliable and 1.5×
batched NumPy gates at 6.28× and 2.29×, respectively. The earlier
[partition gate](benchmarks/PARTITION_REPORT.md) and
[sorting baseline](benchmarks/REPORT.md) remain as historical evidence.

## License

Ferrograd code is dual-licensed MIT or Apache-2.0. See [NOTICE](NOTICE) for
benchmark data and reference-score attribution.
