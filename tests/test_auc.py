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

    def preprocess_categorical_data(self, real_data, synthetic_data):
        # Convert categorical features to one-hot encoding
        real_data_encoded = pd.get_dummies(real_data)
        synthetic_data_encoded = pd.get_dummies(synthetic_data)

        # Align the columns
        real_data_encoded, synthetic_data_encoded = real_data_encoded.align(synthetic_data_encoded, join='left', axis=1,
                                                                            fill_value=0)

        return real_data_encoded, synthetic_data_encoded

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
