from unittest import TestCase

import pandas as pd

import syndat


class Test(TestCase):

    def test_jsd_zero_int64(self):
        synthetic = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 11, 12, 13, 14]
        })
        # Create real data
        real = pd.DataFrame({
            'feature1': [6, 7, 8, 9, 10],
            'feature2': [15, 16, 17, 18, 19]
        })
        distribution = syndat.scores.distribution(real, synthetic)
        self.assertEqual(0, distribution)

    def test_jsd_zero_int64_float64(self):
        synthetic = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        # Create real data
        real = pd.DataFrame({
            'feature1': [6, 7, 8, 9, 10],
            'feature2': [0.6, 0.7, 0.8, 0.9, 1.0]
        })
        distribution = syndat.scores.distribution(real, synthetic)
        self.assertEqual(distribution, 0)

    def test_jsd_perfect_int64(self):
        synthetic = pd.DataFrame({
            'feature1': [1, 2, 1, 2, 3],
            'feature2': [11, 12, 13, 14, 15]
        })
        # Create real data
        real = pd.DataFrame({
            'feature1': [1, 2, 1, 2, 3],
            'feature2': [11, 12, 13, 14, 15]
        })
        distribution = syndat.scores.distribution(real, synthetic)
        self.assertEqual(distribution, 100)

    def test_jsd_perfect_int64_and_float64(self):
        synthetic = pd.DataFrame({
            'feature1': [1, 2, 1, 2, 3],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        # Create real data
        real = pd.DataFrame({
            'feature1': [1, 2, 1, 2, 3],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        distribution = syndat.scores.distribution(real, synthetic)
        self.assertEqual(distribution, 100)

    def test_jsd_different_col_types(self):
        synthetic = pd.DataFrame({
            'feature1': [1, 2, 1, 2, 3],
            'feature2': [1.1, 2.1, 3.1, 4.1, 5.1]
        })
        # Create real data
        real = pd.DataFrame({
            'feature1': [1.2, 2.1, 1.1, 2.1, 3.1],
            'feature2': [1, 2, 3, 4, 5]
        })
        distribution = syndat.scores.distribution(real, synthetic, score=False)

    def test_jsd_negative_int64(self):
        synthetic = pd.DataFrame({
            'feature1': [1, 2, 1, 2, 3],
            'feature2': [1.1, 2.1, 3.1, 4.1, 5.1]
        })
        # Create real data
        real = pd.DataFrame({
            'feature1': [-1, 2, 3, 4, 5],
            'feature2': [1, 2, 3, 4, 5]
        })
        distribution = syndat.scores.distribution(real, synthetic)

    def test_jsd_single_outlier(self):
        synthetic = pd.DataFrame({
            'feature1': [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
            'feature2': [1, 1, 1, 1, 1, 1,  2, 3, 4, 5, 6, 7, 8, 9, 9],
        })
        # Create real data
        real = pd.DataFrame({
            'feature1': [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
            'feature2': [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        })
        distribution = syndat.scores.distribution(real, synthetic)
        self.assertTrue(distribution < 100)

    def test_jsd_categorical_equal(self):
        synthetic = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'B', 'C'],
            'feature2': ['X', 'Y', 'Y', 'X', 'Z']
        })
        # Create real data
        real = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'B', 'C'],
            'feature2': ['X', 'Y', 'Y', 'X', 'Z']
        })
        distribution = syndat.scores.distribution(real, synthetic)
        self.assertEqual(distribution, 100)

    def test_jsd_categorical_different(self):
        synthetic = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'B', 'C'],
            'feature2': ['X', 'Y', 'Y', 'X', 'Z']
        })
        # Create real data
        real = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'B', 'D'],
            'feature2': ['X', 'Y', 'Z', 'X', 'W']
        })
        distribution = syndat.scores.distribution(real, synthetic)
        self.assertTrue(distribution < 100)

    def test_jsd_categorical_mixed(self):
        synthetic = pd.DataFrame({
            'feature1': ['A', 'B', 'C', 'D', 'E'],
            'feature2': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        # Create real data
        real = pd.DataFrame({
            'feature1': ['A', 'B', 'C', 'F', 'G'],
            'feature2': [1.0, 2.0, 3.0, 6.0, 7.0]
        })
        distribution = syndat.scores.distribution(real, synthetic)
        self.assertTrue(distribution < 100)

    def test_jsd_categorical_with_numerical(self):
        synthetic = pd.DataFrame({
            'feature1': ['A', 'B', 'C', 'A', 'B'],
            'feature2': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        # Create real data
        real = pd.DataFrame({
            'feature1': ['A', 'B', 'C', 'A', 'D'],
            'feature2': [1.0, 2.0, 3.0, 4.0, 6.0]
        })
        distribution = syndat.scores.distribution(real, synthetic)
        self.assertTrue(distribution < 100)

    def test_jsd_categorical_with_nan(self):
        synthetic = pd.DataFrame({
            'feature1': ['A', 'B', 'C', 'D', 'E'],
            'feature2': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        # Create real data with NaNs
        real = pd.DataFrame({
            'feature1': ['A', 'B', 'C', 'D', None],
            'feature2': [1.0, 2.0, None, 4.0, 5.0]
        })
        distribution = syndat.scores.distribution(real, synthetic)
        self.assertTrue(distribution < 100)

    def test_jsd_categorical_all_nan(self):
        synthetic = pd.DataFrame({
            'feature1': [None, None, None, None, None],
            'feature2': [None, None, None, None, None]
        })
        # Create real data with NaNs
        real = pd.DataFrame({
            'feature1': [None, None, None, None, None],
            'feature2': [None, None, None, None, None]
        })
        distribution = syndat.scores.distribution(real, synthetic)
        self.assertEqual(distribution, 100)
