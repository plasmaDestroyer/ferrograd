"""Reference implementations for the phase-B benchmark workflows."""
import numpy as np

METRICS = ("iqm", "mean", "median", "optimality_gap")


def aggregate(values, gamma=1.0):
    """Match rliable's four aggregate definitions."""
    flat = np.sort(np.ravel(values))
    trim = len(flat) // 4
    iqm = flat[trim:len(flat)-trim].mean()
    means = np.asarray(values).mean(axis=0)
    return np.array([iqm, means.mean(), np.median(means),
                     gamma - np.minimum(values, gamma).mean()])


def probability(x, y):
    """Match rliable's taskwise probability of improvement."""
    task_values = []
    for task in range(x.shape[1]):
        ordered = np.sort(y[:, task])
        lower = np.searchsorted(ordered, x[:, task], side="left")
        upper = np.searchsorted(ordered, x[:, task], side="right")
        task_values.append((lower + 0.5 * (upper - lower)).mean() / len(ordered))
    return np.array([np.mean(task_values)])


def _batches(reps, cells):
    # At most 64 MiB of indices and sampled float64 scores live together.
    return max(1, (64 * 1024**2) // (16 * cells))


def _aggregate_batch(sampled, gamma):
    size = sampled.shape[0]
    flat = sampled.reshape(size, -1).copy()
    trim = flat.shape[1] // 4
    flat.partition((trim, flat.shape[1] - trim - 1), axis=1)
    means = sampled.mean(axis=1)
    return np.column_stack((flat[:, trim:flat.shape[1]-trim].mean(axis=1),
                            means.mean(axis=1), np.median(means, axis=1),
                            gamma - np.minimum(sampled, gamma).mean(axis=(1, 2))))


def numpy_workflow(operation, data, *, reps, seed=0, confidence=.95, gamma=1.0,
                   thresholds=None, task_bootstrap=False):
    """Bounded-memory NumPy bootstrap with definitions matching rliable 1.2."""
    rng = np.random.default_rng(seed)
    alpha = 50 * (1 - confidence)
    if operation == "compare":
        x, y = data
        tasks = x.shape[1]
        point = probability(x, y)
        values = np.empty((reps, 1))
        # Broadcast comparison is bounded to 64 MiB and avoids Python loops over reps.
        batch = max(1, (64 * 1024**2) //
                    (16 * tasks * (x.shape[0] + y.shape[0]) +
                     4 * x.shape[0] * y.shape[0]))
        columns = np.arange(tasks)
        for start in range(0, reps, batch):
            size = min(batch, reps - start)
            ix = rng.integers(x.shape[0], size=(size, x.shape[0], tasks))
            iy = rng.integers(y.shape[0], size=(size, y.shape[0], tasks))
            sx, sy = x[ix, columns], y[iy, columns]
            del ix, iy
            total = 0.0
            for task in range(tasks):
                left = sx[:, :, task, None]
                right = sy[:, None, :, task]
                total += ((left > right).mean(axis=(1, 2)) +
                          0.5 * (left == right).mean(axis=(1, 2)))
                del left, right
            values[start:start+size, 0] = total / tasks
            del sx, sy
        return point, np.percentile(values, [alpha, 100-alpha], axis=0)

    scores = data
    runs, tasks = scores.shape
    if operation == "aggregates":
        point = aggregate(scores, gamma)
        width = 4
    else:
        source = scores if operation == "profile-score" else scores.mean(axis=0)
        point = np.array([(source > threshold).mean() for threshold in thresholds])
        width = len(thresholds)
    values = np.empty((reps, width))
    bytes_per_cell = 40 if operation == "aggregates" else 24
    batch = max(1, (64 * 1024**2) // (bytes_per_cell * scores.size))
    columns = np.arange(tasks)
    for start in range(0, reps, batch):
        size = min(batch, reps - start)
        indices = rng.integers(runs, size=(size, runs, tasks))
        task_indices = (rng.integers(tasks, size=(size, tasks)) if task_bootstrap else
                        np.broadcast_to(columns, (size, tasks)))
        sampled = scores[indices, task_indices[:, None, :]]
        del indices, task_indices
        if operation == "aggregates":
            values[start:start+size] = _aggregate_batch(sampled, gamma)
        else:
            source = sampled if operation == "profile-score" else sampled.mean(axis=1)
            for column, threshold in enumerate(thresholds):
                values[start:start+size, column] = (source > threshold).mean(
                    axis=tuple(range(1, source.ndim)))
            del source
        del sampled
    return point, np.percentile(values, [alpha, 100-alpha], axis=0)


def rliable_workflow(operation, data, *, reps, seed=0, confidence=.95, gamma=1.0,
                     thresholds=None, task_bootstrap=False):
    """Pinned rliable 1.2 reference workflow."""
    from rliable import library, metrics
    np.random.seed(seed)
    if operation == "aggregates":
        func = lambda x: np.array([metrics.aggregate_iqm(x), metrics.aggregate_mean(x),
                                   metrics.aggregate_median(x),
                                   metrics.aggregate_optimality_gap(x, gamma)])
        return tuple(group["algorithm"] for group in library.get_interval_estimates(
            {"algorithm": data}, func, reps=reps, method="percentile",
            task_bootstrap=task_bootstrap, confidence_interval_size=confidence))
    if operation.startswith("profile-"):
        return tuple(group["algorithm"] for group in library.create_performance_profile(
            {"algorithm": data}, thresholds,
            use_score_distribution=operation == "profile-score", reps=reps,
            task_bootstrap=task_bootstrap, method="percentile",
            confidence_interval_size=confidence))
    func = lambda x, y: np.array([metrics.probability_of_improvement(x, y)])
    return tuple(group["pair"] for group in library.get_interval_estimates(
        {"pair": list(data)}, func, reps=reps, method="percentile",
        confidence_interval_size=confidence))


def ferrograd_workflow(operation, data, *, reps, seed=0, confidence=.95, gamma=1.0,
                       thresholds=None, task_bootstrap=False, threads=1):
    import ferrograd
    kwargs = dict(reps=reps, seed=seed, confidence=confidence, threads=threads)
    if operation == "aggregates":
        return tuple(group["algorithm"] for group in ferrograd.get_interval_estimates(
            {"algorithm": data}, metrics=METRICS, gamma=gamma,
            task_bootstrap=task_bootstrap, **kwargs))
    if operation.startswith("profile-"):
        return tuple(group["algorithm"] for group in ferrograd.create_performance_profile(
            {"algorithm": data}, thresholds,
            use_score_distribution=operation == "profile-score",
            task_bootstrap=task_bootstrap, **kwargs))
    return tuple(group["pair"] for group in ferrograd.compare({"pair": data}, **kwargs))
