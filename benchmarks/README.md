# Phase-one IQM benchmark

This benchmark is the stop/go gate for the proposed Rust-backed RL analysis
package. It compares complete Python calls computing IQM and 95% percentile
bootstrap intervals. The Rust build measured here is the focused phase-one IQM
prototype; the broader public API is conditional on the speed gate.

## Reproduce

Use Python 3.12 and the pinned packages in `reference-requirements.txt`.
The file records the environment used for this run; its NumPy 2.5.3 version
means the result is not a compatibility claim about `rliable` on all NumPy 2.x
versions.

```sh
uv venv --python 3.12 .venv-reference
uv pip install --python .venv-reference/bin/python -r benchmarks/reference-requirements.txt
.venv-reference/bin/maturin build --release --target x86_64-unknown-linux-gnu --interpreter "$(realpath .venv-reference/bin/python3.12)"
uv pip install --python .venv-reference/bin/python target/wheels/ferrograd-0.1.0-cp312-cp312-*.whl
.venv-reference/bin/python benchmarks/prepare_atari.py
.venv-reference/bin/python -m unittest discover -s tests -p 'test_*.py'
flock benchmarks/results/local.lock .venv-reference/bin/python benchmarks/run.py --output benchmarks/results/local.jsonl
flock benchmarks/results/local.lock .venv-reference/bin/python benchmarks/run.py --output benchmarks/results/local.jsonl --scaling
.venv-reference/bin/python benchmarks/coverage.py
flock benchmarks/results/experiment.lock .venv-reference/bin/python benchmarks/coverage_study.py
cargo run --release --example profile_iqm > benchmarks/results/profile.txt
```

Run one benchmark command at a time. Use the same `flock` path for every
command writing one result file: the runner's resume check alone does not
prevent concurrent processes from appending duplicate rows. This is the Linux x86-64 phase-one harness; multi-platform wheel work is
conditional on passing the gate. The run command resumes completed rows. It rejects mixing rows if the
interpreter, installed versions, source hashes, CPU policy, or power settings
change; use a new output path for a new build or environment. The raw JSONL
contains all five timed calls, their median and range, and a separate process
peak RSS measurement for every workload. The adjacent metadata JSON records
hardware and source hashes. Output allocation, conversions, RNG, sorting, and
percentiles are inside the timed calls. Each child process performs one warm-up
followed by five measured calls. Memory is measured in a separate child to
avoid distorting latency. Imports, compilation, download, and plotting are
outside the calls. NumPy's batch holds roughly 64 MiB or less of simultaneous
resample indices and scores. BLAS thread counts are pinned to one. The script
pins one logical CPU per physical core, selecting faster cores first, and uses
a call-local Rayon pool only for Rust multi-thread rows.

## Published data provenance

`data/SPR.json` contains 100 final scores for each of the 26 Atari 100k games.
The original notebook reads `gs://rl-benchmark-data/atari_100k/SPR.json`; that
bucket was unavailable from this environment. The saved file came from the
[SPR mirror](https://github.com/Eclectic-Mem/SPR/blob/7896a65a1b376e5ed352a43cec38b3991f2fb9c7/result-atari/atari_100k/SPR.json)
and matches the independently pinned
[IRIS mirror](https://github.com/eloialonso/iris/blob/24326aaaa283c527f42b89b44cfdecf2665a7a16/results/data/SPR.json)
byte for byte (SHA-256 `055133e22ddb1e7154bbb48b6d229e3b064ea374911158f5d5c1a92cd59e2ad3`).
Those mirrors support provenance; the inaccessible original bucket prevents
direct byte-level comparison with its object.

The random and human reference scores in
`data/atari_100k_baselines.json` were extracted from the pinned
[rliable notebook](https://github.com/google-research/rliable/blob/3ccd9f4dea577a04d3d2b557f259aac08badbd81/deep_rl_precipice_colab.ipynb)
(notebook SHA-256 `a514c3fa8de302d6b17f00c651332cbc0217da987164cbb44dd2ef776983a349`;
extracted JSON SHA-256 `8f8f34a7fb3ccd28d28cf8e7d35b24521b82ef606105ff31107d4ec685991523`).
`prepare_atari.py` applies `(score - random) / (human - random)` per game, sorts
game names as the notebook does, and stacks the result into a 100×26 float64
array. The generated `data/atari_spr.npy` has SHA-256
`8343c7ce757308399dfbd753ecfbab3e42eda9bb33b69bce891fb71fbcaac5d2`.
The extracted numeric baselines are attributed to the rliable authors; see
[their Apache-2.0 license](https://github.com/google-research/rliable/blob/3ccd9f4dea577a04d3d2b557f259aac08badbd81/LICENSE).

The synthetic cases use a fixed NumPy generator seed, three shapes (5×26,
20×26, 100×100), and normal, integer-tied, and Student-t(df=2) values.
All three backends receive the same input arrays, repetition count, and
nominal interval level. They use their own deterministic bootstrap RNG streams,
so their interval endpoints need not match sample for sample.

See [REPORT.md](REPORT.md) for results and the continuation verdict.

The historical `coverage.py` check remains reproducible as originally run.
The larger symmetric-model study is reported separately in
[COVERAGE_REPORT.md](COVERAGE_REPORT.md); its JSONL writer resumes completed
trial/backend rows and must use the shared `results/experiment.lock` lock.

The focused partition gate is reported in [PARTITION_REPORT.md](PARTITION_REPORT.md),
using [partition-gate.jsonl](results/partition-gate.jsonl) and its
[metadata](results/partition-gate.metadata.json). Install the measured wheel
with `uv pip install --reinstall --no-deps --python .venv-reference/bin/python
target/wheels/ferrograd-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl` so an
existing same-version binary cannot remain installed. The historical sorting
baseline maps to [baseline/lib.rs](baseline/lib.rs) and the archived wheel in
`target/wheels-baseline/` (SHA-256 recorded in the partition report).
