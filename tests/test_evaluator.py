import unittest

import pandas as pd
import syndat


class TestEvaluator(unittest.TestCase):
    """
    Test Evaluator class.
    """

    def setUp(self) -> None:
        self.real_data = pd.DataFrame({
            'num_feature1': [1, 2, 3, 4, 5],
            'num_feature2': [5, 4, 3, 2, 1],
            'cat_feature1': ['A', 'B', 'C', 'D', 'E'],
            'cat_feature2': ['X', 'Y', 'Z', 'W', 'V']
        })
        # Similar, but not equal synthetic data
        self.synthetic_data = pd.DataFrame({
            'num_feature1': [1, 2, 3, 4, 4],
            'num_feature2': [5, 4, 3, 2, 2],
            'cat_feature1': ['A', 'B', 'C', 'D', 'D'],
            'cat_feature2': ['X', 'Y', 'Y', 'W', 'V']
        })

    def test_init(self) -> None:
        evaluator = syndat.Evaluator(self.real_data, self.synthetic_data)
        self.assertIsInstance(evaluator, syndat.Evaluator)

    def test_evaluate(self) -> None:
        evaluator = syndat.Evaluator(self.real_data, self.synthetic_data)
        evaluator.evaluate()
        self.assertIsInstance(evaluator.classifier_auc, float)
        self.assertIsInstance(evaluator.correlation_real, pd.DataFrame)
        self.assertIsInstance(evaluator.correlation_synthetic, pd.DataFrame)
        self.assertIsInstance(evaluator.correlation_diff, pd.DataFrame)
        self.assertIsInstance(evaluator.normalized_correlation_quotient, float)
        self.assertIsInstance(evaluator.jsd, dict)

    def test_summary(self):
        evaluator = syndat.Evaluator(self.real_data, self.synthetic_data)
        evaluator.evaluate()
        evaluator.summary_report()
