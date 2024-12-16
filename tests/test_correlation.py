import unittest

import pandas as pd

from syndat.scores import correlation


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
        self.assertEqual(result, 100, "Correlation score should be 100 for identical datasets")

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
        self.assertEqual(result, 0)

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

    def test_correlation_with_categorical(self):
        real_data = pd.DataFrame({
            'risk': ['low', 'low', 'mid', 'high', 'high'],
            'age': [10, 20, 50, 90, 100]
        })
        synthetic_data = pd.DataFrame({
            'risk': ['low', 'low', 'mid', 'high', 'high'],
            'age': [10, 20, 50, 90, 100]
        })
        result = correlation(real_data, synthetic_data)
        self.assertEqual(result, 100, "Correlation score should be 100 for identical datasets with categorical data")

    def test_correlation_with_categorical_more_columns(self):
        real_data = pd.DataFrame({
            'risk': ['low', 'low', 'mid', 'high', 'high'],
            'age': [10, 20, 50, 90, 100],
            'sex': ['male', 'female', 'male', 'female', 'male'],
            'bmi': [21, 22, 23, 24, 25]
        })
        synthetic_data = pd.DataFrame({
            'risk': ['low', 'low', 'mid', 'high', 'high'],
            'age': [10, 20, 50, 90, 100],
            'sex': ['male', 'female', 'male', 'female', 'male'],
            'bmi': [21, 22, 23, 24, 25]
        })
        result = correlation(real_data, synthetic_data)
        self.assertEqual(result, 100, "Correlation score should be 100 for identical datasets with categorical data")

    def test_correlation_with_categorical_extra_category(self):
        real_data = pd.DataFrame({
            'risk': ['low', 'low', 'mid', 'high', 'high', 'very high'],
            'age': [10, 20, 50, 90, 100, 120],
            'sex': ['male', 'female', 'male', 'female', 'male', 'female']
        })
        synthetic_data = pd.DataFrame({
            'risk': ['low', 'low', 'mid', 'high', 'high', 'high'],
            'age': [10, 20, 50, 90, 100, 100],
            'sex': ['male', 'female', 'male', 'female', 'male', 'female']
        })
        result = correlation(real_data, synthetic_data)
        self.assertLessEqual(result, 100, "Correlation score should be 100 for identical "
                                          "datasets with categorical data")

    def test_correlation_with_different_categorical(self):
        real_data = pd.DataFrame({
            # risk of dying of old age
            'risk': ['low', 'low', 'mid', 'high', 'high'],
            'age': [10, 20, 50, 90, 100]
        })
        synthetic_data = pd.DataFrame({
            # risk of having to live for at least another 30 years
            'risk': ['low', 'low', 'mid', 'high', 'high'],
            'age': [100, 90, 50, 20, 10]
        })
        result = correlation(real_data, synthetic_data)
        self.assertLess(result, 100, "Correlation score should be less than 100 for datasets with different "
                                     "categorical data")

    def test_correlation_with_only_categorical(self):
        real_data = pd.DataFrame({
            'sex': ['male', 'female', 'male', 'female', 'male', 'female'],
            'risk': ['high', 'low', 'high', 'low', 'high', 'low'],
        })
        synthetic_data = pd.DataFrame({
            'sex': ['male', 'female', 'male', 'female', 'male', 'female'],
            'risk': ['high', 'high', 'high', 'low', 'high', 'low'],
        })
        result = correlation(real_data, synthetic_data)
        # Adjust the expected result based on encoding and correlation of categorical data
        self.assertLess(result, 100, "Correlation score should be less than 100 for datasets with different "
                                     "categorical data")

    def test_correlation_constant_column(self):
        real_data = pd.DataFrame({
            'A': [1, 1, 1, 1, 1],
            'B': [2, 3, 4, 5, 6]
        })
        synthetic_data = pd.DataFrame({
            'A': [1, 1, 1, 1, 1],
            'B': [2, 3, 4, 5, 6]
        })
        result = correlation(real_data, synthetic_data)
        # Depending on the implementation, the correlation might return NaN or handle it explicitly
        self.assertFalse(pd.isna(result), "Correlation score should not result in NaN even with a constant column")