# IQM interval coverage study

Finite simulation of nominal 95% percentile-bootstrap coverage for a population-zero IQM. It compares `ferrograd` and `rliable` on identical generated data and bootstrap seeds; it gives no universal coverage guarantee.

Each trial has 5 runs × 26 tasks. There are 200 independent trials per cell, 2,000 bootstrap repetitions, and 16 cells (2 designs × 4 models × 2 backends). The target IQM is zero by symmetry. `normal` is N(0,1), `t3` is Student-t(df=3), and `tied` is symmetric with no task offsets. Only `heterogeneous-task` uses task offsets: `fixed-task` targets the equal-weight mixture of 26 fixed tasks with offsets `±(1.25/13, ..., 1.25)`, while `random-task` targets the corresponding offset superpopulation, drawing offsets from `Normal(0, 0.75²)`; both add `Normal(0,1)` noise. Tied support is `[-2,-1,0,1,2]` with probabilities `[.1,.2,.4,.2,.1]`.

The fixed-task intervals use run bootstrap only (`task_bootstrap=False`); random-task intervals resample both runs and task columns (`task_bootstrap=True`).

Coverage is covered trials / 200. MCSE is binomial Monte Carlo standard error. Wilson is the 95% Wilson interval. Width is mean interval width; width MCSE is its across-trial standard error.

| design | model | backend | coverage | MCSE | Wilson 95% | mean width | width MCSE |
|---|---|---|---:|---:|---:|---:|---:|
| fixed-task | normal | ferrograd | 186/200 (0.930) | 0.0180 | [0.886, 0.958] | 0.3392 | 0.0026 |
| fixed-task | normal | rliable | 186/200 (0.930) | 0.0180 | [0.886, 0.958] | 0.3393 | 0.0025 |
| fixed-task | t3 | ferrograd | 187/200 (0.935) | 0.0174 | [0.892, 0.962] | 0.3828 | 0.0031 |
| fixed-task | t3 | rliable | 187/200 (0.935) | 0.0174 | [0.892, 0.962] | 0.3830 | 0.0030 |
| fixed-task | tied | ferrograd | 187/200 (0.935) | 0.0174 | [0.892, 0.962] | 0.4091 | 0.0038 |
| fixed-task | tied | rliable | 186/200 (0.930) | 0.0180 | [0.886, 0.958] | 0.4089 | 0.0038 |
| fixed-task | heterogeneous-task | ferrograd | 184/200 (0.920) | 0.0192 | [0.874, 0.950] | 0.3477 | 0.0023 |
| fixed-task | heterogeneous-task | rliable | 183/200 (0.915) | 0.0197 | [0.868, 0.946] | 0.3479 | 0.0023 |
| random-task | normal | ferrograd | 198/200 (0.990) | 0.0070 | [0.964, 0.997] | 0.4936 | 0.0033 |
| random-task | normal | rliable | 197/200 (0.985) | 0.0086 | [0.957, 0.995] | 0.4938 | 0.0035 |
| random-task | t3 | ferrograd | 200/200 (1.000) | 0.0000 | [0.981, 1.000] | 0.5678 | 0.0047 |
| random-task | t3 | rliable | 199/200 (0.995) | 0.0050 | [0.972, 0.999] | 0.5683 | 0.0048 |
| random-task | tied | ferrograd | 198/200 (0.990) | 0.0070 | [0.964, 0.997] | 0.5632 | 0.0058 |
| random-task | tied | rliable | 198/200 (0.990) | 0.0070 | [0.964, 0.997] | 0.5642 | 0.0058 |
| random-task | heterogeneous-task | ferrograd | 191/200 (0.955) | 0.0147 | [0.917, 0.976] | 0.7776 | 0.0071 |
| random-task | heterogeneous-task | rliable | 192/200 (0.960) | 0.0139 | [0.923, 0.980] | 0.7753 | 0.0069 |

Paired results are close in every cell. The backends receive the same generated data and numeric bootstrap seed, but use separate RNG implementations, so the seed does not imply identical resampling indices. No paired backend-difference analysis was run, so no observed difference is claimed statistically significant. The higher coverage in homogeneous random-task cells reflects this finite simulation's task-resampling model and does not establish that task bootstrap is generally superior.

## Reproduction

```sh
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/coverage_study.py --trials 200 --reps 2000 --data-seed 20260920 --bootstrap-seed 1107 --output benchmarks/results/coverage-study.jsonl --summary benchmarks/results/coverage-study-summary.json
```

Raw [`coverage-study.jsonl`](results/coverage-study.jsonl): 3,201 lines (metadata + 3,200 records); [`coverage-study-summary.json`](results/coverage-study-summary.json): 16 cells. A stdlib/NumPy recomputation verified all records and cells. Original sanity check remains in [`coverage.py`](coverage.py), with [`package-coverage.json`](results/package-coverage.json): 85/100 rliable and 86/100 ferrograd.

Versions: Python 3.12.14, ferrograd 0.1.0, NumPy 2.5.3, rliable 1.2.0, arch 7.2.0. Native module matches the recorded wheel member.

SHA-256:

```text
benchmarks/coverage_study.py                   f060591542621a060753e58b53f283d04b5bb2874195c6b4ebfac01483df76b5
benchmarks/coverage.py                         5afd06761b3a2e86725b312a90bd4a1703b72073cacd8327f793fa5db262c6be
benchmarks/results/coverage-study.jsonl        7a240ce678fa997786ccaf50001d1e8d0f15bf79a640a8cadaad16f936444490
benchmarks/results/coverage-study-summary.json 0c071c7ee3fe26d58a403d2812e24f2690b1edd31bd74b94ec3c79d8ed346add
benchmarks/results/package-coverage.json       3bca600635048de33b33ef9bce125923c01ccd2c2d10ec17fee25de2b8797118
```
