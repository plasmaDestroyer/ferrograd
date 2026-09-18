# Partition gate report

## Verdict: pass

The locked `atari_spr` run used 50,000 repetitions, one thread, one warm-up,
and five measured calls per backend. Both gates pass: rliable/Rust is
**6.14×** (threshold 3×), and NumPy/Rust is **2.27×** (threshold 1.5×).

| Backend | Median seconds | Min–max seconds | Peak RSS |
|---|---:|---:|---:|
| rliable 1t | 6.699625 | 6.560281–6.846804 | 161.9 MiB |
| NumPy 1t | 2.482192 | 2.416723–2.680598 | 98.6 MiB |
| Ferrograd 1t | 1.091677 | 1.048718–1.104969 | 31.7 MiB |

Raw samples: [partition-gate.jsonl](results/partition-gate.jsonl). Metadata:
[partition-gate.metadata.json](results/partition-gate.metadata.json).

Correctness passed: 5 Python tests, 4 Rust tests, and `1e-12` tolerance checks.

Measured wheel: `ferrograd-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl`,
SHA-256 `a8d88fa27793060b7956e8805526d993795c4d75a39b04658755861e24811171`.
The historical sorting baseline remains in [baseline/lib.rs](baseline/lib.rs);
its archived wheel is `target/wheels-baseline/ferrograd-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl`
with SHA-256 `3a89233d9f5d63a6d6c909d6b61ff99b92548f2399ba28d8cede47aae0434b71`.
The old full-run Rust median 2.485657 s is historical and is not a paired
contemporaneous speedup measurement.
