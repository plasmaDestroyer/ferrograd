"""Bootstrap estimates for aggregate reinforcement-learning scores."""
from collections.abc import Mapping
import operator
import numpy as np
from . import _native


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is out of range")
    return value


def _scores(value):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError("scores must be real numeric arrays")
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("scores must be nonempty 2D arrays")
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("scores must be finite")
    return array


def _mapping(value, name):
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    return value


def _controls(reps, confidence, seed, threads):
    reps = _integer(reps, "reps", 2)
    threads = _integer(threads, "threads", 1)
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    confidence = float(confidence)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    return reps, confidence, seed, threads


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _finite(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _thresholds(value):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.ndim != 1 or not array.size:
        raise ValueError("thresholds must be a nonempty real numeric sequence")
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("thresholds must be finite")
    return array.tolist()


def get_interval_estimates(score_dict, *, metrics=("iqm",), reps=50_000,
                           confidence=0.95, task_bootstrap=False, seed=0,
                           threads=1, gamma=1.0):
    """Return ordered metric estimates and percentile intervals.

    Resample runs independently within tasks; optionally also resample tasks.
    RNG streams restart for each mapping entry. Seeds are reproducible across
    thread counts in a pinned build, but do not match NumPy's RNG.
    """
    _mapping(score_dict, "score_dict")
    if isinstance(metrics, str) or callable(metrics):
        raise TypeError("metrics must be a sequence of metric names")
    try:
        metrics = tuple(metrics)
    except TypeError:
        raise TypeError("metrics must be a sequence of metric names") from None
    if not metrics or any(not isinstance(metric, str) or
                          metric not in ("iqm", "mean", "median", "optimality_gap")
                          for metric in metrics):
        raise ValueError("unsupported metric")
    reps, confidence, seed, threads = _controls(reps, confidence, seed, threads)
    gamma = _finite(gamma, "gamma")
    task_bootstrap = _boolean(task_bootstrap, "task_bootstrap")
    points, intervals = {}, {}
    for name, value in score_dict.items():
        point, bounds = _native.aggregate(_scores(value), list(metrics), reps,
                                          confidence, task_bootstrap, seed, threads, gamma)
        points[name] = np.asarray(point, dtype=np.float64)
        intervals[name] = np.asarray(bounds, dtype=np.float64)
    return points, intervals


def create_performance_profile(score_dict, thresholds, *, reps=2000,
                               confidence=0.95, task_bootstrap=False, seed=0,
                               threads=1, use_score_distribution=True):
    """Return strict-win fractions at each threshold and bootstrap intervals."""
    _mapping(score_dict, "score_dict")
    thresholds = _thresholds(thresholds)
    reps, confidence, seed, threads = _controls(reps, confidence, seed, threads)
    task_bootstrap = _boolean(task_bootstrap, "task_bootstrap")
    use_score_distribution = _boolean(use_score_distribution, "use_score_distribution")
    points, intervals = {}, {}
    for name, value in score_dict.items():
        point, bounds = _native.profile(_scores(value), thresholds, reps, confidence,
                                        task_bootstrap, seed, threads,
                                        use_score_distribution)
        points[name] = np.asarray(point, dtype=np.float64)
        intervals[name] = np.asarray(bounds, dtype=np.float64)
    return points, intervals


def compare(score_pairs, *, reps=2000, confidence=0.95, seed=0, threads=1):
    """Return average taskwise win probabilities, giving ties half credit."""
    _mapping(score_pairs, "score_pairs")
    reps, confidence, seed, threads = _controls(reps, confidence, seed, threads)
    points, intervals = {}, {}
    for name, pair in score_pairs.items():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each score pair must contain two score arrays")
        x, y = map(_scores, pair)
        if x.shape[1] != y.shape[1]:
            raise ValueError("score pair task counts must match")
        point, bounds = _native.compare(x, y, reps, confidence, seed, threads)
        points[name] = np.asarray(point, dtype=np.float64)
        intervals[name] = np.asarray(bounds, dtype=np.float64)
    return points, intervals


def _fixed_iqm(scores, run_indices, task_indices):
    """Private test hook: indices have shapes (reps,runs,tasks)/(reps,tasks)."""
    scores = _scores(scores)
    runs = np.asarray(run_indices)
    tasks = np.asarray(task_indices)
    if runs.dtype.kind not in "iu" or tasks.dtype.kind not in "iu":
        raise TypeError("indices must be integers")
    if runs.ndim != 3 or runs.shape[1:] != scores.shape:
        raise ValueError("run_indices must have shape (reps, runs, tasks)")
    if tasks.shape != (runs.shape[0], scores.shape[1]) or not runs.shape[0]:
        raise ValueError("task_indices must have shape (reps, tasks)")
    if (runs < 0).any() or (runs >= scores.shape[0]).any():
        raise ValueError("run index out of range")
    if (tasks < 0).any() or (tasks >= scores.shape[1]).any():
        raise ValueError("task index out of range")
    return _native.fixed_iqm(scores, np.ascontiguousarray(runs, dtype=np.uint64),
                             np.ascontiguousarray(tasks, dtype=np.uint64))


def _fixed_aggregates(scores, run_indices, task_indices, metrics, gamma=1.0):
    scores = _scores(scores)
    runs, tasks = _fixed_indices(scores, run_indices, task_indices)
    return _native.fixed_aggregates(scores, runs, tasks, list(metrics), _finite(gamma, "gamma"))


def _fixed_profile(scores, run_indices, task_indices, thresholds,
                   use_score_distribution=True):
    scores = _scores(scores)
    runs, tasks = _fixed_indices(scores, run_indices, task_indices)
    return _native.fixed_profile(scores, runs, tasks, _thresholds(thresholds),
                                 _boolean(use_score_distribution, "use_score_distribution"))


def _fixed_compare(x, y, run_indices_x, run_indices_y):
    x, y = _scores(x), _scores(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError("score pair task counts must match")
    rx, _ = _fixed_indices(x, run_indices_x,
                           np.broadcast_to(np.arange(x.shape[1]),
                                           (np.shape(run_indices_x)[0], x.shape[1])))
    ry, _ = _fixed_indices(y, run_indices_y,
                           np.broadcast_to(np.arange(y.shape[1]),
                                           (np.shape(run_indices_y)[0], y.shape[1])))
    if rx.shape[0] != ry.shape[0]:
        raise ValueError("resample counts must match")
    return _native.fixed_compare(x, y, rx, ry)


def _fixed_indices(scores, run_indices, task_indices):
    runs, tasks = np.asarray(run_indices), np.asarray(task_indices)
    if runs.dtype.kind not in "iu" or tasks.dtype.kind not in "iu":
        raise TypeError("indices must be integers")
    if runs.ndim != 3 or runs.shape[1:] != scores.shape or not runs.shape[0]:
        raise ValueError("run_indices must have shape (reps, runs, tasks)")
    if tasks.shape != (runs.shape[0], scores.shape[1]):
        raise ValueError("task_indices must have shape (reps, tasks)")
    if (runs < 0).any() or (runs >= scores.shape[0]).any():
        raise ValueError("run index out of range")
    if (tasks < 0).any() or (tasks >= scores.shape[1]).any():
        raise ValueError("task index out of range")
    return (np.ascontiguousarray(runs, dtype=np.uint64),
            np.ascontiguousarray(tasks, dtype=np.uint64))
