"""Published Atari 100k analysis: SPR, DrQ (epsilon), and IRIS."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ferrograd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "data"
OUTPUT = ROOT / "docs" / "atari-analysis"
FILES = {
    "SPR": "SPR.json",
    "DrQ (epsilon)": "DrQ(eps).json",
    "IRIS": "IRIS.json",
}
METRICS = ("iqm", "mean", "median", "optimality_gap")
METRIC_LABELS = ("IQM", "Mean", "Median", "Optimality gap")
CONTROLS = {"reps": 2_000, "confidence": 0.95, "seed": 7, "threads": 1}
THRESHOLDS = np.linspace(0, 3, 121)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(raw, references, games):
    """Align tasks and retain every published run."""
    return np.stack(
        [
            (np.asarray(raw[game], dtype=float) - references["RANDOM_SCORES"][game])
            / (references["HUMAN_SCORES"][game] - references["RANDOM_SCORES"][game])
            for game in games
        ],
        axis=1,
    )


def load_scores(data_dir=DATA):
    references_path = data_dir / "atari_100k_baselines.json"
    references = json.loads(references_path.read_text())
    raw = {
        name: json.loads((data_dir / filename).read_text())
        for name, filename in FILES.items()
    }
    task_sets = [
        set(references["RANDOM_SCORES"]),
        set(references["HUMAN_SCORES"]),
        *(set(values) for values in raw.values()),
    ]
    assert all(tasks == task_sets[0] for tasks in task_sets), "published task sets differ"
    games = sorted(task_sets[0])
    scores = {
        name: normalize(values, references, games) for name, values in raw.items()
    }
    assert {name: values.shape for name, values in scores.items()} == {
        "SPR": (100, 26),
        "DrQ (epsilon)": (100, 26),
        "IRIS": (5, 26),
    }
    assert all(np.isfinite(values).all() for values in scores.values())
    assert sum(values.size for values in scores.values()) == 205 * 26
    return games, scores, references_path


def independent_metrics(values):
    ordered = np.sort(values, axis=None)
    trim = ordered.size // 4
    middle = ordered[trim : -trim or None]
    return np.array(
        [
            middle.mean(),
            values.mean(),
            np.median(values.mean(axis=0)),
            1 - np.minimum(values, 1).mean(),
        ]
    )


def analyze(scores):
    points, intervals = ferrograd.get_interval_estimates(
        scores, metrics=METRICS, **CONTROLS
    )
    _, task_intervals = ferrograd.get_interval_estimates(
        scores, metrics=METRICS, task_bootstrap=True, **CONTROLS
    )
    profiles, profile_intervals = ferrograd.create_performance_profile(
        scores, THRESHOLDS, **CONTROLS
    )
    task_profiles, task_profile_intervals = ferrograd.create_performance_profile(
        scores, THRESHOLDS, use_score_distribution=False, **CONTROLS
    )
    pairs = {
        "IRIS vs SPR": (scores["IRIS"], scores["SPR"]),
        "IRIS vs DrQ (epsilon)": (scores["IRIS"], scores["DrQ (epsilon)"]),
        "DrQ (epsilon) vs SPR": (scores["DrQ (epsilon)"], scores["SPR"]),
    }
    comparisons, comparison_intervals = ferrograd.compare(pairs, **CONTROLS)

    for name, values in scores.items():
        np.testing.assert_allclose(
            points[name], independent_metrics(values), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            profiles[name], [(values > threshold).mean() for threshold in THRESHOLDS]
        )
        np.testing.assert_allclose(
            task_profiles[name],
            [(values.mean(axis=0) > threshold).mean() for threshold in THRESHOLDS],
        )
    for name, (left, right) in pairs.items():
        expected = np.mean(
            [
                (left[:, task, None] > right[None, :, task]).mean()
                + 0.5 * (left[:, task, None] == right[None, :, task]).mean()
                for task in range(left.shape[1])
            ]
        )
        np.testing.assert_allclose(comparisons[name], [expected])
    for bounds in (
        intervals,
        task_intervals,
        profile_intervals,
        task_profile_intervals,
        comparison_intervals,
    ):
        assert all(
            np.isfinite(value).all() and (value[0] <= value[1]).all()
            for value in bounds.values()
        )
    return {
        "points": points,
        "intervals": intervals,
        "task_bootstrap_intervals": task_intervals,
        "profiles": profiles,
        "profile_intervals": profile_intervals,
        "task_mean_profiles": task_profiles,
        "task_mean_profile_intervals": task_profile_intervals,
        "comparisons": comparisons,
        "comparison_intervals": comparison_intervals,
    }


def interval_marks(ax, points, bounds, index=0):
    names = list(points)
    for y, name in enumerate(names):
        low, high = bounds[name][:, index]
        ax.hlines(y, low, high, linewidth=2)
        ax.plot(points[name][index], y, "o")
    ax.set_yticks(range(len(names)), names)


def plot_results(results, output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    for index, ax in enumerate(axes):
        interval_marks(ax, results["points"], results["intervals"], index)
        ax.set_title(METRIC_LABELS[index])
        if index:
            ax.set_yticklabels([])
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for filename, points_key, intervals_key, ylabel in (
        (
            "profiles.png",
            "profiles",
            "profile_intervals",
            "Fraction of individual scores above threshold",
        ),
        (
            "task-mean-profiles.png",
            "task_mean_profiles",
            "task_mean_profile_intervals",
            "Fraction of task means above threshold",
        ),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for name, profile in results[points_key].items():
            ax.plot(THRESHOLDS, profile, label=name)
            ax.fill_between(
                THRESHOLDS,
                results[intervals_key][name][0],
                results[intervals_key][name][1],
                alpha=0.16,
            )
        ax.set(xlabel="Human-normalized score threshold", ylabel=ylabel, ylim=(0, 1))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    interval_marks(ax, results["comparisons"], results["comparison_intervals"])
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 1), xlabel="Probability first method improves on second")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def record_results(games, scores, references_path, results, output=OUTPUT):
    record = {
        "controls": CONTROLS,
        "games": games,
        "shapes": {name: list(values.shape) for name, values in scores.items()},
        "metrics": METRICS,
        "gamma": 1.0,
        "normalization": "(score - random) / (human - random)",
        "primary_resampling": "runs resampled with 26 published tasks fixed",
        "sensitivity_resampling": "runs and tasks resampled for aggregate intervals",
        "input_sha256": {
            **{filename: sha256(DATA / filename) for filename in FILES.values()},
            references_path.name: sha256(references_path),
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("ferrograd", "numpy", "matplotlib")
        },
        "thresholds": THRESHOLDS,
        **results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(record, default=lambda value: value.tolist(), indent=2, allow_nan=False)
        + "\n"
    )
    return record


def main(output=OUTPUT):
    games, scores, references_path = load_scores()
    results = analyze(scores)
    plot_results(results, output)
    record = record_results(games, scores, references_path, results, output)
    print("Shapes:", record["shapes"])
    for name in scores:
        print(name, "points:", results["points"][name])
    print("Comparisons:", results["comparisons"])
    print("Verified independent point estimates; wrote results and four plots to", output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args().output)
