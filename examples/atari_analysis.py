"""Analyze published SPR scores against fixed human references; no training required."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rliable import metrics as reference_metrics, plot_utils

import ferrograd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs' / 'atari-analysis'
OUTPUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'benchmarks' / 'data'
raw = json.loads((DATA / 'SPR.json').read_text())
baselines = json.loads((DATA / 'atari_100k_baselines.json').read_text())
games = sorted(raw)
assert set(games) == set(baselines['RANDOM_SCORES']) == set(baselines['HUMAN_SCORES'])
spr = np.stack([
    (np.asarray(raw[g]) - baselines['RANDOM_SCORES'][g]) /
    (baselines['HUMAN_SCORES'][g] - baselines['RANDOM_SCORES'][g])
    for g in games
], axis=1)
assert spr.shape == (100, 26) and np.isfinite(spr).all()
np.testing.assert_array_equal(spr, np.load(DATA / 'atari_spr.npy'))
scores = {'SPR': spr, 'Human reference (fixed)': np.ones((1, len(games)))}
metrics = ('iqm', 'mean', 'median', 'optimality_gap')
controls = dict(reps=2000, confidence=.95, seed=7, threads=1)
thresholds = np.linspace(0, 3, 121)

points, intervals = ferrograd.get_interval_estimates(scores, metrics=metrics, **controls)
_, task_intervals = ferrograd.get_interval_estimates(
    scores, metrics=metrics, task_bootstrap=True, **controls)
profiles, profile_intervals = ferrograd.create_performance_profile(
    scores, thresholds, **controls)
task_profiles, task_profile_intervals = ferrograd.create_performance_profile(
    scores, thresholds, use_score_distribution=False, **controls)
comparison, comparison_intervals = ferrograd.compare(
    {'SPR,Human reference (fixed)': (spr, scores['Human reference (fixed)'])}, **controls)

# Check real-data point estimates independently of the Rust implementation.
expected = [reference_metrics.aggregate_iqm(spr), spr.mean(axis=0).mean(),
            np.median(spr.mean(axis=0)), 1 - np.minimum(spr, 1).mean()]
np.testing.assert_allclose(points['SPR'], expected, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(profiles['SPR'], [(spr > t).mean() for t in thresholds])
np.testing.assert_allclose(task_profiles['SPR'], [(spr.mean(axis=0) > t).mean() for t in thresholds])
np.testing.assert_allclose(next(iter(comparison.values())),
                           [(spr > 1).mean() + .5 * (spr == 1).mean()])
for bounds in (intervals, task_intervals, profile_intervals,
               task_profile_intervals, comparison_intervals):
    for value in bounds.values():
        assert np.isfinite(value).all() and (value[0] <= value[1]).all()

plot_utils.plot_interval_estimates(
    points, intervals, metric_names=('IQM', 'Mean', 'Median', 'Optimality gap'),
    row_height=.7, xlabel='', max_ticks=3)
plt.gcf().subplots_adjust(wspace=.45)
plt.savefig(OUTPUT / 'metrics.png', bbox_inches='tight', dpi=150)
plt.close('all')
for name, values, bounds in (
    ('profiles', profiles, profile_intervals),
    ('task-mean-profiles', task_profiles, task_profile_intervals),
):
    ax = plot_utils.plot_performance_profiles(
        values, thresholds, bounds,
        ylabel=('Fraction of run/game scores above threshold' if name == 'profiles'
                else 'Fraction of game means above threshold'))
    ax.legend(fontsize=12)
    plt.savefig(OUTPUT / f'{name}.png', bbox_inches='tight', dpi=150)
    plt.close('all')
ax = plot_utils.plot_probability_of_improvement(
    comparison, comparison_intervals, xticks=np.linspace(0, 1, 5),
    xlabel='Win probability (ties count half)', right_ylabel='Fixed reference')
ax.set_xlim(0, 1)
plt.savefig(OUTPUT / 'comparison.png', bbox_inches='tight', dpi=150)
plt.close('all')

result = dict(
    controls=controls, games=games, shape=list(spr.shape), metrics=metrics,
    gamma=1.0, comparison_reference='Fixed published human scores; no human sampling uncertainty',
    input_sha256={name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                  for name in ('SPR.json', 'atari_100k_baselines.json')},
    points=points, intervals=intervals, task_bootstrap_intervals=task_intervals,
    thresholds=thresholds, profiles=profiles, profile_intervals=profile_intervals,
    task_mean_profiles=task_profiles, task_mean_profile_intervals=task_profile_intervals,
    comparison=comparison, comparison_intervals=comparison_intervals,
)
(OUTPUT / 'results.json').write_text(json.dumps(
    result, default=lambda value: value.tolist(), indent=2, allow_nan=False) + '\n')
print('SPR metric points:', points['SPR'])
print('SPR intervals:', intervals['SPR'])
print('Task-bootstrap intervals:', task_intervals['SPR'])
print('Comparison:', comparison, comparison_intervals)
print('Verified real-data estimates; wrote results.json and four plots to', OUTPUT)
