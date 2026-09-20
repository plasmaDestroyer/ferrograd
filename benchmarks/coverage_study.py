"""Reproducible IQM percentile-interval coverage study."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("normal", "t3", "tied", "heterogeneous-task")
DESIGNS = ("fixed-task", "random-task")
TIED_VALUES = np.array([-2., -1., 0., 1., 2.])
TIED_PROBABILITIES = np.array([.1, .2, .4, .2, .1])
_POSITIVE_OFFSETS = np.arange(1, 14, dtype=float) * (1.25 / 13)
FIXED_OFFSETS = np.concatenate((-_POSITIVE_OFFSETS[::-1], _POSITIVE_OFFSETS))


def sample(rng, model, design, runs=5, tasks=26):
    if model == "normal":
        return rng.normal(size=(runs, tasks))
    if model == "t3":
        return rng.standard_t(3, size=(runs, tasks))
    if model == "tied":
        return rng.choice(TIED_VALUES, size=(runs, tasks), p=TIED_PROBABILITIES)
    offsets = (FIXED_OFFSETS if design == "fixed-task" else
               rng.normal(scale=.75, size=tasks))
    return rng.normal(size=(runs, tasks)) + offsets


def wilson(successes, trials, z=1.959963984540054):
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    return [center - radius, center + radius]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def interval(backend, data, reps, seed, task_bootstrap):
    if backend == "rliable":
        from rliable import library, metrics
        np.random.seed(seed)
        point, bounds = library.get_interval_estimates(
            {"algorithm": data}, metrics.aggregate_iqm, reps=reps,
            method="percentile", task_bootstrap=task_bootstrap,
            confidence_interval_size=.95)
    else:
        import ferrograd
        point, bounds = ferrograd.get_interval_estimates(
            {"algorithm": data}, metrics=("iqm",), reps=reps, seed=seed,
            confidence=.95, task_bootstrap=task_bootstrap, threads=1)
    return point["algorithm"], bounds["algorithm"]


def provenance(args):
    import ferrograd
    native = Path(ferrograd._native.__file__)
    wheels = list((ROOT / "target/wheels").glob("ferrograd-*-cp312-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("expected exactly one cp312 wheel")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist()
                   if name.startswith("ferrograd/_native") and name.endswith(".so")]
        if len(members) != 1:
            raise RuntimeError("expected exactly one native module in wheel")
        wheel_native_hash = bytes_sha256(archive.read(members[0]))
    if sha256(native) != wheel_native_hash:
        raise RuntimeError("installed native module does not match cp312 wheel")
    paths = [Path(__file__), ROOT / "benchmarks/coverage.py",
             ROOT / "benchmarks/workflows.py", ROOT / "python/ferrograd/__init__.py",
             ROOT / "src/lib.rs", native, wheel]
    return {
        "type": "metadata",
        "study": "symmetric population-zero IQM coverage",
        "settings": {"runs": 5, "tasks": 26, "trials": args.trials,
                     "reps": args.reps, "confidence": .95,
                     "data_seed": args.data_seed,
                     "bootstrap_seed": args.bootstrap_seed,
                     "designs": list(DESIGNS), "models": list(MODELS),
                     "tied_support": TIED_VALUES.tolist(),
                     "tied_probabilities": TIED_PROBABILITIES.tolist(),
                     "fixed_offsets": FIXED_OFFSETS.tolist(),
                     "random_offset_distribution": "Normal(0, 0.75^2)",
                     "heterogeneous_noise_distribution": "Normal(0, 1)"},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("ferrograd", "numpy", "rliable", "arch")},
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "platform": platform.platform(), "machine": platform.machine(),
                    "logical_cpus": os.cpu_count()},
        "wheel_native_member": members[0],
        "wheel_native_sha256": wheel_native_hash,
        "sha256": {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
                   sha256(path) for path in paths},
    }


def summarize(records, metadata):
    groups = {}
    for row in records:
        key = (row["design"], row["model"], row["backend"])
        groups.setdefault(key, []).append(row)
    results = []
    for (design, model, backend), rows in sorted(groups.items()):
        covered = sum(row["covered"] for row in rows)
        widths = np.array([row["width"] for row in rows])
        n = len(rows)
        results.append({"design": design, "model": model, "backend": backend,
                        "trials": n, "covered": covered, "coverage": covered / n,
                        "coverage_mcse": math.sqrt((covered / n) * (1 - covered / n) / n),
                        "coverage_wilson_95": wilson(covered, n),
                        "mean_width": float(widths.mean()),
                        "mean_width_mcse": float(widths.std(ddof=1) / math.sqrt(n))})
    return {"metadata": metadata, "summary": results}


def run(args):
    output, summary_path = Path(args.output), Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = provenance(args)
    records, completed = [], set()
    if output.exists():
        lines = output.read_text().splitlines()
        if not lines or json.loads(lines[0]).get("type") != "metadata":
            raise RuntimeError("existing output must begin with metadata")
        for index, line in enumerate(lines):
            row = json.loads(line)
            if row["type"] == "metadata":
                if index:
                    raise RuntimeError("metadata may appear only once at the start")
                if row != metadata:
                    raise RuntimeError("existing output metadata differs")
            else:
                key = (row["design"], row["model"], row["trial"], row["backend"])
                if key in completed:
                    raise RuntimeError("duplicate trial/backend row")
                records.append(row)
                completed.add(key)
    else:
        output.write_text(json.dumps(metadata, sort_keys=True) + "\n")

    with output.open("a") as sink:
        for design_index, design in enumerate(DESIGNS):
            for model_index, model in enumerate(MODELS):
                for trial in range(args.trials):
                    sequence = np.random.SeedSequence(
                        [args.data_seed, design_index, model_index, trial])
                    data = sample(np.random.default_rng(sequence), model, design)
                    seed = args.bootstrap_seed + trial
                    ordered = np.sort(data, axis=None)
                    trim = ordered.size // 4
                    expected_point = ordered[trim:ordered.size-trim].mean()
                    for backend in ("rliable", "ferrograd"):
                        key = (design, model, trial, backend)
                        if key in completed:
                            continue
                        point, bounds = interval(backend, data, args.reps, seed,
                                                 design == "random-task")
                        point_value = float(np.asarray(point).reshape(-1)[0])
                        lower, upper = map(float, bounds[:, 0])
                        if (not np.isfinite([point_value, lower, upper]).all() or
                                lower > upper or
                                not np.isclose(point_value, expected_point,
                                               rtol=1e-12, atol=1e-12)):
                            raise RuntimeError("invalid backend result")
                        row = {"type": "trial", "design": design, "model": model,
                               "trial": trial, "backend": backend,
                               "data_seed": [args.data_seed, design_index,
                                             model_index, trial],
                               "bootstrap_seed": seed, "point": point_value,
                               "lower": lower, "upper": upper,
                               "width": upper - lower,
                               "covered": lower <= 0 <= upper}
                        sink.write(json.dumps(row, sort_keys=True) + "\n")
                        sink.flush()
                        records.append(row)
                print(f"completed {design}/{model}", flush=True)
    summary_path.write_text(json.dumps(summarize(records, metadata), indent=2,
                                       sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--reps", type=int, default=2_000)
    parser.add_argument("--data-seed", type=int, default=20260920)
    parser.add_argument("--bootstrap-seed", type=int, default=1107)
    parser.add_argument("--output", default="benchmarks/results/coverage-study.jsonl")
    parser.add_argument("--summary", default="benchmarks/results/coverage-study-summary.json")
    args = parser.parse_args()
    if args.trials < 2 or args.reps < 2:
        parser.error("trials and reps must both be at least 2")
    return args


if __name__ == "__main__":
    run(parse_args())
