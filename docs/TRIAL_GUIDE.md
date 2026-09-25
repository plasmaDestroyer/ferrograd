# Try Ferrograd on your own results

Start with two algorithms evaluated on the same tasks and at the same training
budget. Ferrograd analyzes existing scores; it does not train or evaluate agents.

## 1. Install

Use ordinary CPython 3.11–3.14. Create and activate a fresh environment:

```sh
python -m venv .venv-trial
```

On Linux/macOS: `source .venv-trial/bin/activate`.
On Windows PowerShell: `.venv-trial\Scripts\Activate.ps1`.

Download a wheel artifact from a successful
[Wheels and tests run](https://github.com/plasmaDestroyer/ferrograd/actions/workflows/wheels.yml).
Choose your OS, architecture, and Python version, then extract the artifact ZIP.
For example, `cp312` means CPython 3.12. Install the extracted wheel using its
actual filename:

```sh
python -m pip install /path/to/downloaded/ferrograd-wheel.whl
python -c "import ferrograd; print(ferrograd.__file__)"
```

The wheel filename above is a placeholder. No package-index release is assumed.
If an artifact has expired, clone the repository and run `python -m pip install .`
from its root with a Rust toolchain installed.

## 2. Prepare scores

Save each algorithm's final scores as a NumPy `.npy` array:

- Rows are independent experiment runs, not successive checkpoints or repeated
  episodes treated as independent training runs.
- Columns are tasks in exactly the same order for both algorithms.
- Run counts may differ. Every task within one array needs the same run count.
- Values must be finite real numbers. Missing runs need an explicit experimental
  policy; do not silently fill them with zero or drop difficult tasks.
- Higher must mean better. For aggregates across tasks, use a meaningful common
  scale and the same normalization references for both algorithms. Ferrograd
  does not normalize scores automatically.

Keep the task names and order alongside each file. Check those names before
analysis: equal column counts alone are insufficient. Use a consistent rule
for choosing the final checkpoint and evaluation budget.

## 3. Run a first comparison

Save this as `try_ferrograd.py` next to `baseline.npy` and `candidate.npy`.
Run it only after checking their task names and column order.

```python
import numpy as np
import ferrograd

baseline = np.load("baseline.npy", allow_pickle=False)
candidate = np.load("candidate.npy", allow_pickle=False)
scores = {"baseline": baseline, "candidate": candidate}
settings = dict(reps=2000, confidence=0.95, seed=7, threads=1)
metrics = ("iqm", "mean")

points, intervals = ferrograd.get_interval_estimates(
    scores, metrics=metrics, **settings
)
for name in scores:
    for index, metric in enumerate(metrics):
        low, high = intervals[name][:, index]
        print(f"{name} {metric}: {points[name][index]:.4f} [{low:.4f}, {high:.4f}]")

pair = "candidate vs baseline"
wins, bounds = ferrograd.compare(
    {pair: (candidate, baseline)}, **settings
)
low, high = bounds[pair][:, 0]
print(f"{pair}: {wins[pair][0]:.4f} [{low:.4f}, {high:.4f}]")

np.savez(
    "trial-results.npz",
    baseline_points=points["baseline"], baseline_intervals=intervals["baseline"],
    candidate_points=points["candidate"], candidate_intervals=intervals["candidate"],
    improvement=wins[pair], improvement_interval=bounds[pair],
)
```

```sh
python try_ferrograd.py
python -m pip freeze > trial-environment.txt
```

Keep the script, environment record, original arrays, task names, normalization
references, and `trial-results.npz` together. Two thousand bootstrap repetitions
are a starting point for exploration; more repetitions reduce bootstrap Monte
Carlo noise but do not compensate for too few independent runs.

## 4. Read the result

- **IQM** averages the middle half of pooled scores after trimming both tails.
  **Mean** averages task means. Both are higher-is-better here.
- **Improvement probability** averages run-pair wins across tasks, with half-credit
  for ties. A value of 0.60 means 60% wins including tie credit under that
  comparison—not 60% probability that an algorithm is truly superior.
- Brackets are nominal 95% percentile intervals. This example resamples runs
  while keeping tasks fixed. Overlap of separate aggregate intervals is not a
  dedicated test of the difference between algorithms.
- Task resampling is available for aggregates and profiles with
  `task_bootstrap=True`; it changes the uncertainty question. `compare` currently
  resamples each algorithm's runs independently within corresponding tasks.
- Nominal coverage is not guaranteed for every experiment. See the
  [coverage study](../benchmarks/COVERAGE_REPORT.md) for measured limitations.

For profiles, plots, and a complete working dataset, use the
[Atari notebook](../examples/atari_analysis.ipynb) and
[walkthrough](atari-analysis/README.md). That example uses Matplotlib directly;
plotting is optional and not needed for the script above.

## 5. Note what worked or blocked you

Record these five things after your trial:

1. OS, Python version, and any installation error.
2. Array shapes, task count, and effort needed to prepare comparable scores.
3. Any unclear statistic, interval, or error message.
4. Analysis time compared with your existing workflow, using the same settings.
5. Whether you would use Ferrograd again, and the main reason.

This is local feedback preparation; it does not send your results anywhere.
