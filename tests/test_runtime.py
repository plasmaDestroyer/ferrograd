"""NumPy-only installed-wheel smoke test."""
import unittest

import numpy as np

import ferrograd


class RuntimeSmoke(unittest.TestCase):
    def test_public_api(self):
        scores = np.array([[0., 1.], [1., 2.]])
        estimates, intervals = ferrograd.get_interval_estimates(
            {'a': scores}, metrics=('iqm', 'mean', 'median', 'optimality_gap'), reps=2)
        np.testing.assert_allclose(estimates['a'], [1., 1., 1., .25])
        self.assertEqual(intervals['a'].shape, (2, 4))
        profiles, bounds = ferrograd.create_performance_profile(
            {'a': scores}, [0., 1.], reps=2)
        np.testing.assert_allclose(profiles['a'], [.75, .25])
        self.assertEqual(bounds['a'].shape, (2, 2))
        comparisons, bounds = ferrograd.compare({'a': (scores, scores)}, reps=2)
        np.testing.assert_allclose(comparisons['a'], [.5])
        self.assertEqual(bounds['a'].shape, (2, 1))


if __name__ == '__main__':
    unittest.main()
