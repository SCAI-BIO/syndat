import logging
from typing import Literal

import pandas
import pandas as pd
import numpy as np

import syndat.metrics

logger = logging.getLogger(__name__)


def discrimination(real: pandas.DataFrame, synthetic: pandas.DataFrame, n_folds=5, drop_na_threshold=0.9) -> float:
    """
    Computes the Discrimination Complexity Score (normalized between 0-100) of a classifier trained to differentiate
    between real and synthetic data. The score is calculated based on the classifier performance of a Random Forest
    classifier differentiating between real and synthetic data.

    :param real: The real data.
    :param synthetic: The synthetic data
    :param n_folds: Number of k folds for cross-validation.
    :param drop_na_threshold: Percentage of non-missing values required of any column
    :return: Differentiation Complexity Score
    """
    auc_score = syndat.metrics.discriminator_auc(real, synthetic, n_folds=n_folds, drop_na_threshold=drop_na_threshold)
    return min(0.5, (1 - auc_score)) * 200


def distribution(real: pd.DataFrame, synthetic: pd.DataFrame, n_unique_threshold=10) -> float:
    """
    Computes a normalized score (0-100) quantifying the feature distribution similarity of real and synthetic data
    using the average Jensen-Shannon distance for all features.

    :param real: The real data.
    :param synthetic: The synthetic data.
    :param n_unique_threshold: Threshold to determine at which number of unique values bins will span over several
    values.
    :return: Distribution Similarity Score
    """
    # Initialize a dictionary to store JSD values for each column
    jsd_dict = syndat.metrics.jensen_shannon_distance(real, synthetic, n_unique_threshold=n_unique_threshold)
    jsd_aggregated = np.mean(list(jsd_dict.values()))
    return int((1 - jsd_aggregated) * 100)


def correlation(real: pd.DataFrame, synthetic: pd.DataFrame,
                method: Literal['pearson', 'kendall', 'spearman'] = 'spearman') -> float:
    """
    Computes the Correlation Similarity Score (normalized between 0-100) of real and synthetic data. The score is
    calculated by comparing the correlation matrices of real and synthetic data.

    :param real: The real data.
    :param synthetic: The synthetic data.
    :param method: The correlation method to use. Options are 'pearson', 'kendall', or 'spearman'.
    :return: Correlation score / Norm Quotient
    """
    norm_quotient = syndat.metrics.normalized_correlation_difference(real, synthetic, method)
    # the norm can exceed 1 in case the correlations are mostly negative for real data, but mostly positive for
    # synthetic data. As this is the worst case scenario, we consider this as having a score of zero
    if norm_quotient > 1:
        return 0
    return (1 - max(norm_quotient, 0)) * 100
