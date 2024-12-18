import logging
from syndat import metrics

from typing import Dict

import pandas as pd


class Evaluator:
    """
    Evaluate the synthetic data against the real data. Retrieve performance metrics, scores, visualizations and perform
    postprocessing steps.

    :param real_data: The real dataset.
    :param synthetic_data: The synthetic dataset.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> None:
        """
        Initialize the Evaluator with real and synthetic datasets.

        :param real_data: The real dataset
        :param synthetic_data: The synthetic dataset
        """
        # datasets
        self.real_data: pd.DataFrame = real_data
        self.synthetic_data: pd.DataFrame = synthetic_data
        # discriminator performance
        self.classifier_auc: float = None
        # correlation matrices
        self.correlation_real: pd.DataFrame = None
        self.correlation_synthetic: pd.DataFrame = None
        self.correlation_diff: pd.DataFrame = None
        self.normalized_correlation_quotient: float = None
        # Jenson-Shannon Divergence
        self.jsd: Dict[str, float] = None

        self.evaluate()

    def evaluate(self) -> None:
        """
        Evaluate the synthetic data against the real data.
        """
        self.logger.info("Evaluating synthetic data...")
        self.logger.info("Evaluating discriminator performance...")
        self.evaluate_discriminator()
        self.logger.info("Evaluating correlations...")
        self.evaluate_correlations()
        self.logger.info("Evaluating Jenson-Shannon Divergence...")
        self.evaluate_jsd()
        self.logger.info("Evaluation complete.")

    def evaluate_discriminator(self) -> None:
        """
        Evaluate the synthetic data using a Random Forest classifier.
        """
        self.classifier_auc = metrics.evaluate_discriminator(self.real_data, self.synthetic_data)

    def evaluate_correlations(self) -> None:
        """
        Evaluate the correlations between the real and synthetic datasets.
        """
        self.correlation_real, self.correlation_synthetic, self.correlation_diff = metrics.evaluate_correlations(
            self.real_data, self.synthetic_data)

    def evaluate_jsd(self) -> None:
        """
        Evaluate the Jensen-Shannon Divergence between the real and synthetic datasets.
        """
        self.jsd = metrics.evaluate_jsd(self.real_data, self.synthetic_data)

    def get_scores(self) -> Dict[str, float]:
        """
        Retrieve the scores of the synthetic data.

        :return: The scores of the synthetic data.
        """
        scores = {
            "discriminator_score": metrics.discrimination_score(self.classifier_auc),
            "distribution_score": metrics.distribution_score(self.jsd),
            "correlation_score": metrics.correlation_score(self.normalized_correlation_quotient)
        }
        return scores

    def summary_report(self) -> None:
        """
        Print a summary report of the evaluation results.
        """
        print("Discriminator AUC:", self.classifier_auc)
        print("Correlation Real:", self.correlation_real)
        print("Correlation Synthetic:", self.correlation_synthetic)
        print("Correlation Difference:", self.correlation_diff)
        print("Jenson-Shannon Divergence:", self.jsd)
        print("Scores:", self.get_scores())
