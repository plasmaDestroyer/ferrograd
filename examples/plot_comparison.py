"""Optional plotting example; install rliable and matplotlib to run."""

import numpy as np
import matplotlib.pyplot as plt
from rliable import plot_utils

import ferrograd


scores = {
    "baseline": np.array([[0.2, 0.7], [0.5, 0.9]]),
    "candidate": np.array([[0.4, 0.8], [0.6, 1.1]]),
}
thresholds = np.linspace(0, 1.2, 25)
metrics = ("iqm", "mean", "median", "optimality_gap")

points, intervals = ferrograd.get_interval_estimates(scores, metrics=metrics, reps=2000)
plot_utils.plot_interval_estimates(points, intervals, metric_names=metrics)
plt.savefig("metrics.png", bbox_inches="tight")
plt.close("all")

profiles, profile_intervals = ferrograd.create_performance_profile(
    scores, thresholds, reps=2000)
plot_utils.plot_performance_profiles(profiles, thresholds, profile_intervals)
plt.savefig("profiles.png", bbox_inches="tight")
plt.close("all")

pairs = {"candidate,baseline": (scores["candidate"], scores["baseline"])}
wins, win_intervals = ferrograd.compare(pairs, reps=2000)
plot_utils.plot_probability_of_improvement(wins, win_intervals)
plt.savefig("comparison.png", bbox_inches="tight")
