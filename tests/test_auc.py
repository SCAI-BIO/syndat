import pandas as pd
import numpy as np
import syndat

import unittest


class Test(unittest.TestCase):

    def setUp(self):
        # Create synthetic data with numerical features
        self.synthetic_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })

        # Create real data with numerical features
        self.real_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })

        # Create real and synthetic data with categorical features
        self.real_data_cat = pd.DataFrame({
            'feature1': np.random.choice(['A', 'B', 'C'], size=100),
            'feature2': np.random.choice(['X', 'Y'], size=100)
        })

        self.synthetic_data_cat = pd.DataFrame({
            'feature1': np.random.choice(['A', 'B', 'C'], size=100),
            'feature2': np.random.choice(['X', 'Y'], size=100)
        })

    def test_auc_score(self):
        auc_score = syndat.scores.discrimination(self.real_data, self.synthetic_data)
        self.assertTrue(isinstance(auc_score, float))
        self.assertGreaterEqual(auc_score, 0.0)
        self.assertLessEqual(auc_score, 100.0)

    def test_auc_score_normalized(self):
        auc_score = syndat.scores.discrimination(self.real_data, self.synthetic_data)
        self.assertTrue(isinstance(auc_score, float))
        self.assertGreaterEqual(auc_score, 0.0)
        self.assertLessEqual(auc_score, 100.0)

    def test_auc_score_with_missing_values(self):
        # Introduce missing values in real data
        self.real_data.iloc[::10, 0] = np.nan  # 10% missing data
        auc_score = syndat.scores.discrimination(self.real_data, self.synthetic_data)
        self.assertTrue(isinstance(auc_score, float))
        self.assertGreaterEqual(auc_score, 0.0)
        self.assertLessEqual(auc_score, 100.0)

    def test_auc_score_with_missing_values_drop_col(self):
        # Introduce missing values in real data
        self.real_data.iloc[::2, 0] = np.nan  # 50% missing data -> col drop
        auc_score = syndat.scores.discrimination(self.real_data, self.synthetic_data)
        self.assertTrue(isinstance(auc_score, float))
        self.assertGreaterEqual(auc_score, 0.0)
        self.assertLessEqual(auc_score, 100.0)

    def test_auc_score_with_custom_folds(self):
        auc_score = syndat.scores.discrimination(self.real_data, self.synthetic_data, n_folds=5)
        self.assertTrue(isinstance(auc_score, float))
        self.assertGreaterEqual(auc_score, 0.0)
        self.assertLessEqual(auc_score, 100.0)

    def test_auc_score_with_categorical_data(self):
        auc_score = syndat.scores.discrimination(self.real_data_cat, self.synthetic_data_cat)
        self.assertTrue(isinstance(auc_score, float))
        self.assertGreaterEqual(auc_score, 0.0)
        self.assertLessEqual(auc_score, 100.0)

    def test_missing_in_all_cols_raises_error(self):
        with self.assertRaises(ValueError):
            real = pd.DataFrame({
                'feature1': [1, 2, np.nan, np.nan],
                'feature2': ['A', 'B', np.nan, np.nan]
            })
            synthetic = pd.DataFrame({
                'feature1': [1, 2, np.nan, np.nan],
                'feature2': ['X', 'Y', np.nan, np.nan]
            })
            auc_score = syndat.scores.discrimination(real, synthetic)