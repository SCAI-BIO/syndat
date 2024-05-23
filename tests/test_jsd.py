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
        jsd = syndat.quality.jsd(real, synthetic)
        self.assertEqual(0, jsd)

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
        jsd = syndat.quality.jsd(real, synthetic)
        self.assertEqual(jsd, 0)

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
        jsd = syndat.quality.jsd(real, synthetic)
        self.assertEqual(jsd, 100)

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
        jsd = syndat.quality.jsd(real, synthetic)
        self.assertEqual(jsd, 100)

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
        jsd = syndat.quality.jsd(real, synthetic, score=False)

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
        jsd = syndat.quality.jsd(real, synthetic)

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
        jsd = syndat.quality.jsd(real, synthetic)
        self.assertTrue(jsd < 100)
