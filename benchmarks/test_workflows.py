"""Small numerical checks for benchmark-only reference code."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from rliable import metrics

from workflows import _aggregate_batch, aggregate, numpy_workflow, probability


def aggregate_oracle(sampled, gamma=1.0):
    """Independent, direct definitions for fixed bootstrap samples."""
    values = []
    for sample in sampled:
        flat = np.sort(sample, axis=None)
        trim = flat.size // 4
        task_means = sample.mean(axis=0)
        values.append([flat[trim:flat.size-trim].mean(), task_means.mean(),
                       np.median(task_means),
                       gamma - np.minimum(sample, gamma).mean()])
    return np.asarray(values)


def probability_oracle(x, y):
    per_task = ((x[:, None, :] > y[None, :, :]).mean(axis=(0, 1)) +
                0.5 * (x[:, None, :] == y[None, :, :]).mean(axis=(0, 1)))
    return np.array([per_task.mean()])


class WorkflowReferences(unittest.TestCase):
    def test_fixed_points(self):
        x = np.array([[0., 1., 3.], [2., 1., -1.]])
        y = np.array([[0., 2., 2.], [1., 0., -1.], [3., 1., 1.]])
        assert_allclose(aggregate(x, 1.5),
                        [metrics.aggregate_iqm(x), metrics.aggregate_mean(x),
                         metrics.aggregate_median(x), metrics.aggregate_optimality_gap(x, 1.5)])
        assert_allclose(probability(x, y), [metrics.probability_of_improvement(x, y)])

    def test_fixed_bootstrap_sample_definitions(self):
        scores = np.array([[0., 2.], [2., 0.]])
        point, _ = numpy_workflow("aggregates", scores, reps=2, seed=3)
        assert_allclose(point, [1., 1., 1., 0.5])
        for operation in ("profile-score", "profile-task"):
            point, bounds = numpy_workflow(operation, scores, reps=5, seed=4,
                                           thresholds=[0., 1.])
            expected = [0.5, 0.5] if operation == "profile-score" else [1., 0.]
            assert_allclose(point, expected)
            self.assertEqual(bounds.shape, (2, 2))

    def test_seeded_replicates_match_fixed_index_oracle(self):
        scores = np.arange(12., dtype=float).reshape(4, 3)
        reps, seed = 9, 17
        rng = np.random.default_rng(seed)
        indices = rng.integers(4, size=(reps, 4, 3))
        sampled = scores[indices, np.arange(3)]
        expected = aggregate_oracle(sampled)
        assert_allclose(_aggregate_batch(sampled, 1.0), expected)
        _, bounds = numpy_workflow("aggregates", scores, reps=reps, seed=seed)
        assert_allclose(bounds, np.percentile(expected, [2.5, 97.5], axis=0))

        rng = np.random.default_rng(seed)
        indices = rng.integers(4, size=(reps, 4, 3))
        tasks = rng.integers(3, size=(reps, 3))
        sampled = np.array([scores[r, t] for r, t in zip(indices, tasks)])
        expected = aggregate_oracle(sampled)
        _, bounds = numpy_workflow("aggregates", scores, reps=reps, seed=seed,
                                   task_bootstrap=True)
        assert_allclose(bounds, np.percentile(expected, [2.5, 97.5], axis=0))

        for operation in ("profile-score", "profile-task"):
            rng = np.random.default_rng(seed)
            indices = rng.integers(4, size=(reps, 4, 3))
            sampled = scores[indices, np.arange(3)]
            source = sampled if operation == "profile-score" else sampled.mean(axis=1)
            expected = np.column_stack([(source > threshold).mean(
                axis=tuple(range(1, source.ndim))) for threshold in (2., 7.)])
            _, bounds = numpy_workflow(operation, scores, reps=reps, seed=seed,
                                       thresholds=[2., 7.])
            assert_allclose(bounds, np.percentile(expected, [2.5, 97.5], axis=0))

        x, y = scores, scores[:2] + 0.5
        rng = np.random.default_rng(seed)
        ix = rng.integers(4, size=(reps, 4, 3))
        iy = rng.integers(2, size=(reps, 2, 3))
        expected = np.array([probability_oracle(x[a, np.arange(3)], y[b, np.arange(3)])
                             for a, b in zip(ix, iy)])
        _, bounds = numpy_workflow("compare", (x, y), reps=reps, seed=seed)
        assert_allclose(bounds, np.percentile(expected, [2.5, 97.5], axis=0))


if __name__ == "__main__":
    unittest.main()
