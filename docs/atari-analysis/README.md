# Published Atari 100k analysis

This walkthrough analyzes every published final score for SPR (100 runs),
DrQ (epsilon, 100 runs), and IRIS (5 runs) on the same 26 Atari 100k tasks.
No training or data download is required. The executable
[notebook](../../examples/atari_analysis.ipynb) calls the reusable
[script](../../examples/atari_analysis.py).

Scores use `(score - random) / (human - random)`. The script requires exact
task-set alignment, checks the 100/100/5 run counts, and retains all 5,330 raw
scores. See [data provenance](../../benchmarks/data/PROVENANCE.md).

## Results

Seed 7, 2,000 bootstrap repetitions, nominal 95% percentile intervals, one
thread, and optimality-gap threshold 1:

| Method | IQM | Mean | Median of task means | Optimality gap |
|---|---:|---:|---:|---:|
| SPR | 0.337 | 0.616 | 0.396 | 0.577 |
| DrQ (epsilon) | 0.280 | 0.465 | 0.313 | 0.631 |
| IRIS | 0.501 | 1.046 | 0.289 | 0.512 |

IRIS has the highest IQM and mean. Its mean is much higher than its IQM and
median, showing that the aggregate summaries respond differently to the
observed skew. Lower optimality gap is better.

Primary intervals hold the named 26 tasks fixed and resample runs. They
quantify bootstrap variation within these published runs. The recorded
sensitivity analysis also resamples tasks; those intervals are much
wider because performance varies strongly by game. IRIS has only five runs,
so its run uncertainty is also wider. Run resampling reflects the variation
represented by these published runs; it does not establish the behavior of a
different training or evaluation protocol. Task resampling is a sensitivity
analysis, not a claim about Atari tasks outside this published set.

Pairwise probabilities average taskwise run-pair comparisons; ties count
half. They are **0.610** [0.577, 0.644] for IRIS over SPR, **0.688** [0.654,
0.721] for IRIS over DrQ (epsilon), and **0.383** [0.369, 0.397] for DrQ
(epsilon) over SPR. These answer a different question from aggregate ranks.

Profile bands are pointwise intervals. The individual-score profile weights
every run/task value; the task-mean profile gives each task one thresholded
mean. The displayed 0-3 range does not clip calculations.

![Aggregate estimates](metrics.png)

![Individual-score profiles](profiles.png)

![Task-mean profiles](task-mean-profiles.png)

![Pairwise improvement probabilities](comparison.png)

## Reproduce

Install a wheel downloaded from the matching CI wheel artifact, without
assuming a package-index release:

```sh
python -m venv .venv
.venv/bin/python -m pip install path/to/ferrograd-0.1.0-*.whl numpy matplotlib
.venv/bin/python examples/atari_analysis.py
```

Use `--output DIR` to write the JSON and plots outside the tracked docs tree.

Or execute `examples/atari_analysis.ipynb` with a notebook runner using that
environment. Matplotlib is used directly; rliable is optional. The script
independently verifies every point estimate and writes [results.json](results.json),
including hashes, task order, versions, settings, full values, and intervals.
