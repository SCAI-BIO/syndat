import pandas as pd
import numpy as np
import syndat

import unittest


class Test(unittest.TestCase):

    def setUp(self):
        # Create synthetic data
        self.synthetic_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        # Create real data
        self.real_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })

    def test_auc_score(self):
        auc_score = syndat.quality.auc(self.real_data, self.synthetic_data)
        self.assertTrue(isinstance(auc_score, float))
        self.assertGreaterEqual(auc_score, 0.0)
        self.assertLessEqual(auc_score, 100.0)

    def test_auc_score_normalized(self):
        auc_score = syndat.quality.auc(self.real_data, self.synthetic_data)
        self.assertTrue(isinstance(auc_score, float))
        self.assertGreaterEqual(auc_score, 0.0)
        self.assertLessEqual(auc_score, 100.0)

    def test_auc_score_with_missing_values(self):
        # Introduce missing values in real data
        self.real_data.iloc[::10, 0] = np.nan
        auc_score = syndat.quality.auc(self.real_data, self.synthetic_data)
        self.assertTrue(isinstance(auc_score, float))
        self.assertGreaterEqual(auc_score, 0.0)
        self.assertLessEqual(auc_score, 100.0)

    def test_auc_score_with_custom_folds(self):
        auc_score = syndat.quality.auc(self.real_data, self.synthetic_data, n_folds=5)
        self.assertTrue(isinstance(auc_score, float))
        self.assertGreaterEqual(auc_score, 0.0)
        self.assertLessEqual(auc_score, 100.0)