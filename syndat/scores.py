import pandas as pd
import numpy as np


def discrimination_score(mean_auc: float) -> float:
    """
    Computes the discrimination complexity score of real and synthetic data.

    :param mean_auc: The mean AUC score across all folds of the Random Forest classifier.
    :return: The discrimination score.
    """
    return min(0.5, (1 - mean_auc)) * 200


def distribution_score(jsd: [], aggregation_method: str = "average") -> float:
    """
    Computes the feature distribution similarity using the Jensen-Shannon distance of real and synthetic data.

    :param jsd: The Jensen-Shannon distance of real and synthetic data.
    :param aggregation_method: The method to aggregate the Jensen-Shannon distances. Either "average" or "median".
    :return: The distribution score.
    """
    if aggregation_method == 'average':
        jsd_aggregated = np.mean(list(jsd.values()))
    elif aggregation_method == 'median':
        jsd_aggregated = np.median(list(jsd.values()))
    else:
        raise ValueError(f'Invalid aggregation method: {aggregation_method}. Use either "average" or "median".')
    return int((1 - jsd_aggregated) * 100)


def correlation_score(normalized_correlation_quotient: float) -> float:
    """
    Computes a score based on the difference between the correlation matrices of real and synthetic data.

    :param normalized_correlation_quotient: The normalized correlation quotient between real and synthetic data.
    :return: The correlation score.
    """
    # this only happens if correlations are too high in synthetic data (hallucinations) -> bad data
    if normalized_correlation_quotient > 1:
        return 0
    else:
        return (1 - max(normalized_correlation_quotient, 0)) * 100
