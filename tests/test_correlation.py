import unittest

import pandas as pd

from syndat.quality import correlation


class TestCorrelation(unittest.TestCase):

    def test_correlation(self):
        real_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
        synthetic_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
        result = correlation(real_data, synthetic_data)
        self.assertEqual(100, result, "Correlation score should be 100 for identical datasets")

    def test_correlation_zero(self):
        # positive correlation
        real_data = pd.DataFrame({
            'A': [-1, 1, -2, 2, -3],
            'B': [-2, 2, -4, 4, -6]
        })
        # negative correlation
        synthetic_data = pd.DataFrame({
            'A': [1, -1, 2, -2, 3],
            'B': [-1, 1, -2, 2, -3]
        })
        result = correlation(real_data, synthetic_data)
        self.assertEqual(0, result)

    def test_correlation_normalized(self):
        real_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
        synthetic_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
        result = correlation(real_data, synthetic_data, score=False)
        self.assertEqual(0, result, "Normalized correlation score should be 0 for identical datasets")
