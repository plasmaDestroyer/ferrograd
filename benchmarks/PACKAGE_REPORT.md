# Final-package IQM benchmark

## Gate: pass

The installed full-package wheel completed the 100-run × 26-game Atari SPR
IQM bootstrap (50,000 repetitions, one thread) in **0.926 s** median, versus
**5.820 s** for pinned `rliable` and **2.117 s** for batched NumPy. The complete
Python-call speedups are **6.28×** and **2.29×**, exceeding the required
3× and 1.5× gates. These are IQM speed results; they do not claim that every
new aggregate, profile, or comparison call has the same speedup.

Each backend ran one untimed warm-up followed by five timed calls. Peak process
RSS was measured in a separate fresh process: 34.7 MiB for Ferrograd,
163.0 MiB for `rliable`, and 99.2 MiB for NumPy. RSS includes interpreter and
imports. The methods use independent RNG implementations, so their percentile
endpoints need not be identical for the same seed.

## Full-package workload matrix

The fresh run has **100 distinct configurations**: 10 workloads × 2 repetition
counts × three single-thread backends, plus Rust at two and four threads. Every
row has five finite positive timing samples, and its stored median and range
match those samples. All 40 parallel Rust rows have exactly the same point
estimate and interval as their one-thread counterpart. The raw samples,
outputs, per-process peak RSS, affinity, package versions, and source hashes
are in [package-gate.jsonl](results/package-gate.jsonl) and
[package-gate.metadata.json](results/package-gate.metadata.json). The earlier
phase-one measurements remain in their original files.

Median complete-call seconds at 50,000 repetitions:

| Workload | rliable 1t | NumPy 1t | Ferrograd 1t | Ferrograd 2t | Ferrograd 4t |
|---|---:|---:|---:|---:|---:|
| Atari SPR 100×26 | 5.820 | 2.117 | 0.926 | 0.480 | 0.256 |
| Normal 5×26 | 3.371 | 0.111 | 0.087 | 0.057 | 0.030 |
| Tied 5×26 | 3.423 | 0.097 | 0.076 | 0.040 | 0.022 |
| Heavy-tailed 5×26 | 3.387 | 0.112 | 0.084 | 0.045 | 0.025 |
| Normal 20×26 | 3.886 | 0.446 | 0.279 | 0.148 | 0.095 |
| Tied 20×26 | 3.967 | 0.392 | 0.270 | 0.152 | 0.088 |
| Heavy-tailed 20×26 | 4.006 | 0.447 | 0.285 | 0.160 | 0.096 |
| Normal 100×100 | 12.819 | 7.910 | 3.513 | 2.039 | 1.245 |
| Tied 100×100 | 11.782 | 6.907 | 3.385 | 1.962 | 1.196 |
| Heavy-tailed 100×100 | 12.786 | 7.943 | 3.498 | 2.026 | 1.237 |

The specified pass/fail gate applies to the Atari one-thread row. Some smaller
synthetic cases did not reach 1.5× against NumPy. Raw 2,000-repetition results
are in the JSONL file; their shorter calls have a larger fixed-overhead share.

## Scope and verification

This measures the final package build, with `src/lib.rs` SHA-256
`84e1c5ecee2d52d481172dba7e27c34ff968e3f73bbd0c81bcf70c21b229fb7a`
and Python wrapper SHA-256
`3026e5a2ac982c32a8c8950158cbb80bb95d8752ba70ce936c6910e6927fa016`.
The pinned reference environment used Python 3.12.14, `rliable==1.2.0`,
`arch==7.2.0`, and NumPy 2.5.3 on an Intel Core i9-12900H. The one-thread
process was pinned to logical CPU 4, selected by the harness's physical-core
policy. BLAS thread counts were set to one. The full setup and Atari source
provenance are documented in [README.md](README.md).

Fixed-index tests checked Rust aggregate statistics, profiles, and comparisons
against the Python reference calculations, including the task bootstrap,
unequal run counts, ties, and percentile endpoints. The final verification
passed 11 Python and four Rust tests, formatting and Clippy checks. A separate
installed-wheel environment ran the public API smoke test with NumPy present
and `rliable`, `arch`, SciPy, and pandas absent. This verifies runtime
independence from the reference stack; cross-platform wheels are checked by
the wheel workflow, not by this local Linux benchmark.

The [coverage sanity check](results/package-coverage.json) used 100
standard-normal 5×26 trials (seed 1107, 2,000 bootstrap repetitions).
Nominal 95% intervals covered the population IQM in 85/100 rliable trials
and 86/100 Ferrograd trials; mean widths were 0.336213 and 0.337032.
Both coverage counts are below 95/100. This small comparison does not
establish calibrated coverage or a general guarantee.

Reproduce the final-package matrix with the same installed wheel and pinned
reference environment:

```sh
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output benchmarks/results/package-gate.jsonl --case atari_spr --reps 50000 --threads 1
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output benchmarks/results/package-gate.jsonl
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/run.py --output benchmarks/results/package-gate.jsonl --scaling
```

Use a new output name when the build or environment changes; the runner
rejects mixing metadata from different builds. Run these commands sequentially
with the same lock to avoid duplicate rows.
