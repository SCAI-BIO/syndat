import unittest

import numpy as np
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
        result = correlation(real_data, synthetic_data)
        self.assertEqual(100, result, "Normalized correlation score should be 100 for identical datasets")

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

    def test_correlation_with_sparse_and_tied_data(self):
        # Simulate high missingness, ties, and categorical NA/"Yes" structure
        real_data = pd.DataFrame({
            'cont1': [0, 0, 0, np.nan, np.nan, np.nan, np.nan],  # many ties, few valid
            'cont2': [1, 2, 1, 2, np.nan, np.nan, np.nan],  # low unique count
            'yes_na': ['Yes', np.nan, 'Yes', np.nan, np.nan, 'Yes', np.nan],  # binary categorical
            'group': ['A', 'A', 'B', np.nan, 'C', 'C', 'D'],  # some rare, some missing
        })
        synthetic_data = pd.DataFrame({
            'cont1': [1, 1, 1, 1, 1, 1, 1],  # no missing, all ones
            'cont2': [1, 1, 1, 1, 1, 1, 1],  # constant column
            'yes_na': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'],  # no missing
            'group': ['A', 'B', 'B', 'C', 'C', 'C', 'C'],  # more balanced group
        })

        result = correlation(real_data, synthetic_data)
        self.assertFalse(pd.isna(result), "Correlation score should not return NaN with sparse/tied data")
        self.assertLessEqual(result, 100, "Score should be between 0 and 100")

    def test_correlation_with_high_missingness_return_nan(self):
        # Simulate high missingness
        real_data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, np.nan],
            'B': [np.nan, 2, 3, np.nan, 5]
        })
        synthetic_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
        result = correlation(real_data, synthetic_data)
        # result should be nan as only one valid column remains
        self.assertTrue(pd.isna(result), "Correlation score should return NaN with high missingness")

    def test_correlation_with_high_missingness_successful_filtering(self):
        real_data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, np.nan],
            'B': [np.nan, 2, 3, np.nan, 5],
            'C': [1, 2, 3, 4, 5]
        })
        synthetic_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [1, 2, 3, 4, 5]
        })
        result = correlation(real_data, synthetic_data)
        self.assertLessEqual(result, 100, "Score should be between 0 and 100")

