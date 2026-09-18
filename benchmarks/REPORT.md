# Phase-one benchmark report

## Verdict: stop

On the published Atari 100k SPR final-score matrix (100 runs × 26 games),
50,000 stratified bootstrap repetitions, and one thread, Ferrograd took
**2.486 s** median versus **5.729 s** for pinned `rliable` and **2.104 s** for
batched NumPy. Speedups are **2.30×** and **0.85×**, respectively. The required
thresholds are 3× and 1.5×; neither passed. The broader analysis API and
distribution phases should therefore not proceed under this plan.

These are complete Python-call timings, with one warm-up and five measured
calls. The respective process peak RSS measurements were 31.1, 162.6, and
98.7 MiB for Ferrograd, `rliable`, and NumPy. RSS includes interpreter and
imports; the measurements run in separate fresh processes from latency.
Ferrograd stores O(repetitions) metric results plus one score-sized scratch
buffer per worker, rather than all resampled score matrices.

## Measurement setup

- Intel Core i9-12900H, Linux 7.2.5-1-cachyos, Python 3.12.14; one selected
  physical core (CPU 4) for single-thread runs. Affinity and governor are
  recorded in `results/full.metadata.json`.
- Rust 1.96.0; Maturin 1.15.0; release wheel
  `ferrograd-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl`, SHA-256
  `3a89233d9f5d63a6d6c909d6b61ff99b92548f2399ba28d8cede47aae0434b71`.
  No `target-cpu=native` build flag. This is a local Linux wheel, not the
  requested cross-platform distribution artifact.
- Pinned reference: `rliable==1.2.0`, `arch==7.2.0`, NumPy 2.5.3. Full
  installed package versions are in `reference-requirements.txt` and metadata.
- IQM: flatten each (runs, tasks) array, trim `floor(n/4)` values from both
  ends, average the rest. Ordinary bootstrap independently resamples runs
  inside each task; percentile endpoints use linear interpolation. Separate
  RNG implementations mean the three interval samples differ despite equal
  statistical workloads.

## Correctness

`tests/test_iqm.py` compares each Rust metric on fixed resample indices to
`rliable.metrics.aggregate_iqm`, then compares percentile endpoints with
NumPy's default percentile at `rtol=atol=1e-12`. It exercises IQM trim
boundaries, constant/negative/tied scores, both task-bootstrap modes, array
conversion and ownership, deterministic thread ordering, and invalid inputs.
The seeded coverage simulation in `coverage.py` is a separate distributional
check and does not imply nominal coverage for every distribution.

The completed raw matrix has 60 distinct single-thread and 40 distinct Rust
scaling configurations. Every row has five finite positive timings; recorded
median and range match those samples. All 40 multi-thread rows produced the
same point estimate and interval endpoints as their one-thread counterparts,
and the measured source hashes match the metadata.

The phase-one verification run passed four Python unittest methods and four
Rust tests (three unit, one integration), `cargo fmt --check`, and
`cargo clippy --all-targets -- -D warnings`. A clean NumPy-only environment installed and called the measured
wheel without `rliable`, `arch`, SciPy, or pandas. These checks validate the
IQM prototype, not the deferred full API or other platform wheels. The old
starter files and historical specification have not been pivoted because the
gate failed.

## Complete results

The raw timing samples, per-row spread, outputs, memory, and source hashes are
in `results/full.jsonl` and `results/full.metadata.json`.

Each cell is median seconds [minimum, maximum seconds]; peak process RSS in MiB.
Ranges are the spread across five calls, not confidence intervals.

| Case | Reps | rliable 1t | NumPy 1t | Rust 1t | Rust 2t | Rust 4t |
|---|---:|---:|---:|---:|---:|---:|
| atari_spr | 50000 | 5.729 [5.673, 5.758]; 162.6 | 2.104 [2.092, 2.111]; 98.7 | 2.486 [2.483, 2.503]; 31.1 | 1.804 [1.663, 1.898]; 31.6 | 1.043 [0.947, 1.191]; 31.6 |
| atari_spr | 2000 | 0.229 [0.228, 0.237]; 162.3 | 0.084 [0.083, 0.092]; 97.5 | 0.098 [0.097, 0.098]; 31.5 | 0.079 [0.065, 0.124]; 31.6 | 0.046 [0.042, 0.061]; 31.6 |
| normal_5_26 | 2000 | 0.135 [0.134, 0.137]; 161.9 | 0.005 [0.005, 0.005]; 37.1 | 0.006 [0.006, 0.006]; 33.5 | 0.005 [0.004, 0.005]; 33.8 | 0.002 [0.002, 0.004]; 34.0 |
| normal_5_26 | 50000 | 3.342 [3.303, 3.378]; 162.4 | 0.109 [0.109, 0.110]; 97.2 | 0.137 [0.136, 0.156]; 34.2 | 0.102 [0.088, 0.150]; 34.5 | 0.062 [0.047, 0.086]; 33.9 |
| tied_5_26 | 2000 | 0.135 [0.133, 0.137]; 161.3 | 0.004 [0.004, 0.004]; 37.2 | 0.003 [0.003, 0.003]; 34.1 | 0.005 [0.004, 0.005]; 33.5 | 0.002 [0.001, 0.002]; 33.5 |
| tied_5_26 | 50000 | 3.578 [3.375, 3.930]; 163.0 | 0.096 [0.093, 0.098]; 96.8 | 0.083 [0.082, 0.084]; 34.1 | 0.075 [0.067, 0.091]; 33.9 | 0.032 [0.030, 0.039]; 34.6 |
| heavy_5_26 | 2000 | 0.134 [0.133, 0.137]; 161.4 | 0.005 [0.005, 0.006]; 37.1 | 0.005 [0.005, 0.006]; 33.5 | 0.004 [0.004, 0.008]; 34.1 | 0.002 [0.002, 0.004]; 34.1 |
| heavy_5_26 | 50000 | 3.367 [3.334, 3.695]; 163.4 | 0.113 [0.112, 0.116]; 97.3 | 0.138 [0.136, 0.144]; 34.2 | 0.105 [0.088, 0.160]; 34.4 | 0.059 [0.049, 0.072]; 34.6 |
| normal_20_26 | 2000 | 0.152 [0.151, 0.154]; 162.1 | 0.017 [0.017, 0.017]; 49.1 | 0.022 [0.022, 0.022]; 33.7 | 0.024 [0.018, 0.031]; 33.5 | 0.010 [0.010, 0.012]; 33.7 |
| normal_20_26 | 50000 | 3.984 [3.796, 4.472]; 162.7 | 0.453 [0.445, 0.474]; 98.7 | 0.568 [0.565, 0.570]; 33.7 | 0.377 [0.362, 0.415]; 34.3 | 0.214 [0.208, 0.225]; 34.4 |
| tied_20_26 | 2000 | 0.155 [0.154, 0.160]; 161.8 | 0.016 [0.015, 0.017]; 49.4 | 0.011 [0.010, 0.012]; 33.7 | 0.014 [0.008, 0.014]; 34.3 | 0.005 [0.005, 0.006]; 33.9 |
| tied_20_26 | 50000 | 3.918 [3.861, 3.944]; 162.8 | 0.399 [0.394, 0.402]; 98.1 | 0.265 [0.262, 0.270]; 33.9 | 0.164 [0.163, 0.219]; 34.7 | 0.138 [0.119, 0.151]; 34.1 |
| heavy_20_26 | 2000 | 0.230 [0.200, 0.282]; 161.2 | 0.031 [0.029, 0.033]; 49.3 | 0.034 [0.034, 0.035]; 33.5 | 0.023 [0.017, 0.030]; 34.3 | 0.009 [0.008, 0.012]; 33.7 |
| heavy_20_26 | 50000 | 4.389 [4.095, 6.426]; 162.9 | 0.553 [0.494, 0.765]; 98.4 | 0.619 [0.606, 0.678]; 34.0 | 0.425 [0.370, 0.483]; 34.5 | 0.269 [0.223, 0.296]; 34.5 |
| normal_100_100 | 2000 | 0.536 [0.529, 0.588]; 162.1 | 0.334 [0.329, 0.337]; 97.5 | 0.442 [0.425, 0.465]; 33.7 | 0.318 [0.265, 0.397]; 34.4 | 0.173 [0.160, 0.234]; 34.6 |
| normal_100_100 | 50000 | 12.816 [12.758, 14.019]; 162.7 | 8.112 [8.091, 8.261]; 98.4 | 10.710 [10.504, 10.988]; 34.0 | 7.097 [6.882, 9.083]; 34.2 | 4.189 [4.144, 4.255]; 35.1 |
| tied_100_100 | 2000 | 0.506 [0.499, 0.571]; 162.1 | 0.300 [0.293, 0.304]; 97.7 | 0.144 [0.143, 0.159]; 33.8 | 0.110 [0.096, 0.169]; 34.2 | 0.062 [0.048, 0.076]; 34.6 |
| tied_100_100 | 50000 | 12.876 [12.769, 13.395]; 162.0 | 7.280 [6.911, 8.223]; 98.4 | 3.997 [3.941, 4.221]; 34.6 | 2.265 [2.245, 2.296]; 34.5 | 1.377 [1.363, 1.410]; 34.4 |
| heavy_100_100 | 2000 | 0.579 [0.572, 0.586]; 162.6 | 0.354 [0.348, 0.357]; 97.9 | 0.473 [0.465, 0.475]; 34.1 | 0.324 [0.265, 0.499]; 34.4 | 0.185 [0.161, 0.197]; 34.6 |
| heavy_100_100 | 50000 | 14.906 [14.614, 15.289]; 163.5 | 9.641 [8.840, 11.063]; 98.2 | 12.098 [11.847, 12.195]; 34.4 | 7.072 [6.791, 7.540]; 34.0 | 4.036 [4.029, 4.213]; 34.6 |

The 50,000-repetition Atari Rust medians on two and four selected physical
cores were 1.804 s and 1.043 s (1.38× and 2.38× faster than its one-core
median). These scaling observations do not change the specified one-core gate.
The synthetic 100×100 cases show that sorting costs depend strongly on ties:
at 50,000 repetitions Rust took 10.710 s for normal values, 3.997 s for
integer ties, and 12.098 s for heavy-tailed values on one core. Rows were
collected across interrupted sessions with unchanged metadata, but CPU
frequency, temperature, and background load were not held fixed. Treat
cross-session timing differences as descriptive rather than controlled causal
comparisons.

Two simultaneous resumptions produced duplicate heavy_100_100/Rust/50,000/1t
rows at 23.308 s and 23.330 s. Both original rows are preserved in
`results/interrupted-overlap.jsonl`; both were excluded from this table, and
the exact configuration was rerun alone under `flock` at 12.098 s. The main
`full.jsonl` now has exactly 100 distinct configurations. The overlap explains
why those two rows cannot be treated as clean performance measurements; it
also shows the need for the lock in the reproduction commands.

## Component profile

`results/profile.txt` records `cargo run --release --example profile_iqm`
replaying the Rust sampling and IQM loops for the 100×26 Atari matrix and
50,000 repetitions. Seeding ChaCha8, setting each stream, and filling sampled
scores took 0.693 s; sorting all samples and averaging their trimmed middle
took 2.044 s (about 75% of the 2.738 s instrumented total). This standalone
probe was not pinned to a CPU. The profiler copies the two native loops but
excludes Python conversion, point estimate, result collection, and interval
quantile sorting. Timer calls also add overhead, so these components should
not be added to the end-to-end Python-call timings. Sorting is the largest
observed Rust component in this focused probe.

## Distributional sanity check

`results/coverage.json` records 100 seeded standard-normal 5×26 trials with
2,000 repetitions per method. The nominal 95% intervals covered the population
IQM of zero in 85/100 `rliable` trials and 86/100 Ferrograd trials; mean widths
were 0.3362 and 0.3370. This small simulation supports similar behavior on
this one distribution, not general coverage calibration.
