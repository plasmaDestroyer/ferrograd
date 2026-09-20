"""Small checks for the coverage-study definitions and summaries."""
import unittest

import numpy as np

from coverage_study import FIXED_OFFSETS, MODELS, sample, wilson


class CoverageStudy(unittest.TestCase):
    def test_models_and_offsets_are_symmetric(self):
        self.assertTrue(np.array_equal(FIXED_OFFSETS, -FIXED_OFFSETS[::-1]))
        for model in MODELS:
            values = sample(np.random.default_rng(3), model, "fixed-task")
            self.assertEqual(values.shape, (5, 26))
            self.assertTrue(np.isfinite(values).all())

    def test_wilson_bounds(self):
        lower, upper = wilson(190, 200)
        self.assertLess(lower, .95)
        self.assertGreater(upper, .95)


if __name__ == "__main__":
    unittest.main()
