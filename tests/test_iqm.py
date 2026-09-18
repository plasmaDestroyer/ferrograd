"""Run: .venv-reference/bin/python -m unittest discover -s tests -p 'test_*.py'."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from rliable import metrics

import ferrograd
from ferrograd import _native


class IQMContract(unittest.TestCase):
    def test_fixed_resamples_and_endpoints(self):
        rng = np.random.default_rng(937)
        for shape in tuple((1, n) for n in range(1, 10)) + ((5, 26),):
            for scores in (rng.normal(size=shape), rng.integers(-2, 3, size=shape),
                           np.full(shape, -3.0)):
                for task_bootstrap in (False, True):
                    runs = rng.integers(shape[0], size=(37, *shape))
                    tasks = (rng.integers(shape[1], size=(37, shape[1]))
                             if task_bootstrap else
                             np.broadcast_to(np.arange(shape[1]), (37, shape[1])))
                    expected = np.array([metrics.aggregate_iqm(scores[r, t])
                                         for r, t in zip(runs, tasks)])
                    actual = ferrograd._fixed_iqm(scores, runs, tasks)
                    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                    for confidence in (0.5, 0.95, 0.999):
                        alpha = (1 - confidence) / 2
                        assert_allclose(_native.fixed_percentile(actual.tolist(), confidence),
                                        np.percentile(expected, [100*alpha, 100*(1-alpha)]),
                                        rtol=1e-12, atol=1e-12)
                    points, _ = ferrograd.get_interval_estimates({'a': scores}, reps=2)
                    assert_allclose(points['a'], [metrics.aggregate_iqm(scores)],
                                    rtol=1e-12, atol=1e-12)

    def test_percentile_extremes(self):
        for values in ([-1e100, 1e100], [-1e100, -1.0, 1.0, 1e100],
                       [-1e10, -2.0, 2.0, 1e10]):
            for confidence in (0.5, 0.95):
                alpha = (1 - confidence) / 2
                assert_allclose(_native.fixed_percentile(values, confidence),
                                np.percentile(values, [100*alpha, 100*(1-alpha)]),
                                rtol=1e-12, atol=1e-12)
        for values in ([], [1.0], [float('inf'), 1.0], [float('nan'), 1.0]):
            with self.assertRaises(ValueError):
                _native.fixed_percentile(values, 0.95)

    def test_fixed_iqm_cancellation(self):
        scores = np.array([[1e6, -1e6, 0.25, 0.5, -0.5, -0.25, 1.0, -1.0]])
        runs = np.zeros((1, 1, scores.shape[1]), dtype=np.uint64)
        tasks = np.arange(scores.shape[1], dtype=np.uint64)[None, :]
        assert_allclose(ferrograd._fixed_iqm(scores, runs, tasks),
                        [metrics.aggregate_iqm(scores)], rtol=1e-12, atol=1e-12)

    def test_threads_order_conversion_and_ownership(self):
        scores = np.arange(48, dtype=np.int32).reshape(6, 8)[::-1, ::2]
        before = scores.copy()
        for task_bootstrap in (False, True):
            one = ferrograd.get_interval_estimates({'b': scores, 'a': scores / 3},
                    metrics=('iqm', 'iqm'), reps=51, task_bootstrap=task_bootstrap)
            many = ferrograd.get_interval_estimates({'a': scores / 3, 'b': scores},
                    metrics=('iqm', 'iqm'), reps=51, threads=3, task_bootstrap=task_bootstrap)
            for group, other in zip(one, many):
                self.assertEqual(list(group), ['b', 'a'])
                for key in group:
                    assert_array_equal(group[key], other[key])
                    self.assertEqual(group[key].dtype, np.float64)
            self.assertEqual(one[0]['a'].shape, (2,))
            self.assertEqual(one[1]['a'].shape, (2, 2))
        assert_array_equal(scores, before)

    def test_invalid_inputs(self):
        for scores in ([], [1, 2], np.empty((0, 2)), [[[1]]], [[np.nan]],
                       [[np.inf]], [[1j]], np.array([[1]], dtype=object), [['1']]):
            with self.subTest(scores=repr(scores)), self.assertRaises((TypeError, ValueError)):
                ferrograd.get_interval_estimates({'a': scores}, reps=2)
        for kwargs in ({'reps': 1}, {'reps': True}, {'threads': 0}, {'threads': 1.5},
                       {'seed': -1}, {'seed': 2**64}, {'confidence': 0},
                       {'confidence': 1}, {'confidence': np.nan}, {'gamma': np.inf},
                       {'metrics': ('bogus',)}, {'metrics': ()}, {'metrics': 'iqm'},
                       {'metrics': lambda x: x}, {'task_bootstrap': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                ferrograd.get_interval_estimates({'a': [[1.0]]}, **kwargs)
        with self.assertRaises(ValueError):
            ferrograd.get_interval_estimates({})
        with self.assertRaises(ValueError):
            ferrograd.get_interval_estimates({'a': np.full((4, 4), np.finfo(float).max)}, reps=2)
        for run, task in (([[[1]]], [[0]]), ([[[-1]]], [[0]]), ([[[0]]], [[1]])):
            with self.assertRaises(ValueError):
                ferrograd._fixed_iqm([[1]], run, task)


if __name__ == '__main__':
    unittest.main()
