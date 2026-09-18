import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from rliable import metrics

import ferrograd
from ferrograd import _native


class APIContract(unittest.TestCase):
    def test_fixed_aggregates(self):
        rng = np.random.default_rng(25)
        names = ('iqm', 'mean', 'median', 'optimality_gap')
        for scores in (rng.normal(size=(3, 7)), np.array([[-2., 0., 0., 2.]]),
                       np.full((2, 5), -3.)):
            runs = rng.integers(scores.shape[0], size=(23, *scores.shape))
            for task_bootstrap in (False, True):
                tasks = (rng.integers(scores.shape[1], size=(23, scores.shape[1]))
                         if task_bootstrap else
                         np.broadcast_to(np.arange(scores.shape[1]), (23, scores.shape[1])))
                expected = []
                for r, t in zip(runs, tasks):
                    sampled = scores[r, t]
                    means = sampled.mean(axis=0)
                    expected.append([metrics.aggregate_iqm(sampled), means.mean(),
                                     np.median(means), 1.5 - np.minimum(sampled, 1.5).mean()])
                actual = ferrograd._fixed_aggregates(scores, runs, tasks, names, 1.5)
                assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                for j in range(len(names)):
                    assert_allclose(_native.fixed_percentile(actual[:, j].tolist(), .95),
                                    np.percentile(np.asarray(expected)[:, j], [2.5, 97.5]),
                                    rtol=1e-12, atol=1e-12)

    def test_fixed_profiles_and_compare(self):
        x = np.array([[0., 1., 2.], [2., 1., 0.]])
        y = np.array([[0., 1., 2.], [2., 1., 0.], [1., 1., 1.]])
        rx = np.array([[[0, 0, 0], [1, 1, 1]], [[1, 0, 1], [1, 0, 1]]])
        ry = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                       [[1, 1, 1], [1, 1, 1], [0, 0, 0]]])
        tasks = np.array([[2, 0, 2], [1, 1, 0]])
        thresholds = [2., 0., 1., 0.]
        for distribution in (True, False):
            expected = []
            for r, t in zip(rx, tasks):
                sampled = x[r, t]
                values = sampled if distribution else sampled.mean(axis=0)
                expected.append([(values > threshold).mean() for threshold in thresholds])
            assert_allclose(ferrograd._fixed_profile(x, rx, tasks, thresholds, distribution),
                            expected, rtol=1e-12, atol=1e-12)
        expected = []
        for a, b in zip(rx, ry):
            sx = x[a, np.arange(3)]
            sy = y[b, np.arange(3)]
            wins = (sx[:, None, :] > sy[None, :, :]).mean(axis=(0, 1))
            ties = (sx[:, None, :] == sy[None, :, :]).mean(axis=(0, 1))
            expected.append([metrics.probability_of_improvement(sx, sy)])
            assert_allclose(expected[-1], [(wins + .5 * ties).mean()],
                            rtol=1e-12, atol=1e-12)
        assert_allclose(ferrograd._fixed_compare(x, y, rx, ry), expected,
                        rtol=1e-12, atol=1e-12)

    def test_overflowing_task_mean_is_rejected(self):
        scores = [[np.finfo(float).max], [np.finfo(float).max],
                  [-np.finfo(float).max], [-np.finfo(float).max]]
        with self.assertRaisesRegex(ValueError, 'nonfinite computed task mean'):
            ferrograd.create_performance_profile({'a': scores}, [0.], reps=2,
                                                  use_score_distribution=False)
        with self.assertRaisesRegex(ValueError, 'nonfinite computed task mean'):
            ferrograd.get_interval_estimates({'a': scores}, metrics=('median',), reps=2)

    def test_public_shapes_order_determinism_and_inputs(self):
        x = np.arange(24, dtype=np.int32).reshape(6, 4)[::-1, ::2]
        before = x.copy()
        for task_bootstrap in (False, True):
            kw = dict(reps=37, seed=4, task_bootstrap=task_bootstrap)
            a = ferrograd.get_interval_estimates({'b': x, 'a': x / 3},
                    metrics=('median', 'iqm', 'mean', 'optimality_gap'), **kw)
            b = ferrograd.get_interval_estimates({'a': x / 3, 'b': x},
                    metrics=('median', 'iqm', 'mean', 'optimality_gap'), threads=3, **kw)
            self.assertEqual(list(a[0]), ['b', 'a'])
            for group, other in zip(a, b):
                assert_array_equal(group['b'], other['b'])
            self.assertEqual(a[0]['b'].shape, (4,))
            self.assertEqual(a[1]['b'].shape, (2, 4))
            p = ferrograd.create_performance_profile({'b': x, 'a': x / 3},
                    [0, 3], **kw)
            q = ferrograd.create_performance_profile({'a': x / 3, 'b': x},
                    [0, 3], threads=3, **kw)
            assert_array_equal(p[0]['b'], q[0]['b'])
            assert_array_equal(p[1]['b'], q[1]['b'])
            self.assertEqual(p[0]['b'].shape, (2,))
            self.assertEqual(p[1]['b'].shape, (2, 2))
        c = ferrograd.compare({'b': (x, x / 3), 'a': (x, x)}, reps=37)
        d = ferrograd.compare({'a': (x, x), 'b': (x, x / 3)}, reps=37, threads=3)
        for group, other in zip(c, d):
            assert_array_equal(group['b'], other['b'])
        self.assertEqual(c[0]['b'].shape, (1,))
        self.assertEqual(c[1]['b'].shape, (2, 1))
        assert_array_equal(x, before)

    def test_invalid_inputs(self):
        invalid_scores = ([], [1], [[np.nan]], [[1j]], np.array([[1]], dtype=object))
        for value in invalid_scores:
            for call in (lambda: ferrograd.create_performance_profile({'a': value}, [0], reps=2),
                         lambda: ferrograd.compare({'a': (value, [[1]])}, reps=2)):
                with self.assertRaises((TypeError, ValueError)):
                    call()
        for thresholds in ([], [np.nan], [1j], np.array([1], dtype=object), [[1]]):
            with self.assertRaises((TypeError, ValueError)):
                ferrograd.create_performance_profile({'a': [[1]]}, thresholds, reps=2)
        for call in (lambda: ferrograd.create_performance_profile({}, [0], reps=2),
                     lambda: ferrograd.compare({}, reps=2),
                     lambda: ferrograd.compare({'a': ([[1]], [[1, 2]])}, reps=2),
                     lambda: ferrograd.compare({'a': ([[1]],)}, reps=2),
                     lambda: ferrograd.create_performance_profile({'a': [[1]]}, [0],
                                                                   use_score_distribution=1, reps=2)):
            with self.assertRaises((TypeError, ValueError)):
                call()


if __name__ == '__main__':
    unittest.main()
