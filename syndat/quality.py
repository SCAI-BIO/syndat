from typing import Union

import logging
import pandas
import pandas as pd
import numpy as np
import scipy.spatial.distance

from sklearn import ensemble
from sklearn.model_selection import cross_val_score

from syndat.domain import AggregationMethod



def auc(real: pandas.DataFrame, synthetic: pandas.DataFrame, n_folds=10, score: bool = True) -> float:
    """
    Computes the Differentiation Complexity Score / ROC AUC score of a classifier trained to differentiate between real
    and synthetic data.

    :param real: The real data.
    :param synthetic: The synthetic data
    :param n_folds: Number of k folds for cross-validation.
    :param score: Return result in a normalized score in [0,100]. Default is True.
    :return: Differentiation Complexity Score / AUC ROC Score
    """
    # check for missing values in real data
    real_clean = real.dropna(thresh=int(0.8 * len(real)), axis=1)
    real_clean = real_clean.dropna()
    # assert that both real and synthetic have same columns
    synthetic_clean = synthetic[real_clean.columns]
    x = pd.concat([real_clean, synthetic_clean])
    y = np.concatenate((np.zeros(real_clean.shape[0]), np.ones(synthetic_clean.shape[0])), axis=None)
    rfc = ensemble.RandomForestClassifier()
    auc_score = np.average(cross_val_score(rfc, x, y, cv=n_folds, scoring='roc_auc'))
    if score:
        return min(0.5, (1 - auc_score)) * 200
    else:
        return auc_score


def jsd(real: pandas.DataFrame, synthetic: pandas.DataFrame, aggregate_results: bool = True,
        aggregation_method: AggregationMethod = AggregationMethod.MEDIAN, score: bool = True) -> Union[
    list[float], float]:
    """
    Computes the feature distribution similarity using the Jensen-Shannon distance of real and synthetic data.

    :param real: The real data.
    :param synthetic: The synthetic data.
    :param aggregate_results: Compute a single aggregated score for all features. Default is True.
    :param aggregation_method: How the scores are aggregated. Default is using the median of all feature scores.
    :param score: Return result in a normalized score in [0,100]. Default is True.
    :return: Distribution Similarity / JSD
    """
    # load datasets & remove id column
    jsd_dict = {}
    for col in real:
        # delete empty cells
        real_wo_missing = real[col].dropna()
        # binning
        col_dtype_real = real[col].dtype
        col_dtype_synthetic = synthetic[col].dtype
        if col_dtype_real != col_dtype_synthetic:
            logging.warning(f'Real data at col {col} is dtype {col_dtype_real} but synthetic is {col_dtype_synthetic}. '
                            f'Evaluation will be done based on the assumed data type of the real data.')
            for col_synth in synthetic.columns:
                synthetic[col_synth] = synthetic[col_synth].astype(real[col_synth].dtype)
        if col_dtype_real == "int64" or col_dtype_real == "object":
            # handle negative values for bincount -> shift all codes to positive (will yield same result in JSD)
            min_value = min(real[col].min(), synthetic[col].min())
            if min_value < 0:
                real[col] = real[col] + abs(min_value)
                synthetic[col] = synthetic[col] + abs(min_value)
            # categorical column
            real_binned = np.bincount(real[col])
            virtual_binned = np.bincount(synthetic[col])
        else:
            # get optimal amount of bins
            n_bins = np.histogram_bin_edges(real_wo_missing, bins='auto')
            real_binned = np.bincount(np.digitize(real_wo_missing, n_bins))
            virtual_binned = np.bincount(np.digitize(synthetic[col], n_bins))
        # one array might be shorter here then the other, e.g. if real patients contain the categorical
        # encoding 0-3, but virtual patients only contain 0-2
        # in this case -> fill missing bin with zero
        if len(real_binned) != len(virtual_binned):
            padding_size = np.abs(len(real_binned) - len(virtual_binned))
            if len(real_binned) > len(virtual_binned):
                virtual_binned = np.pad(virtual_binned, (0, padding_size))
            else:
                real_binned = np.pad(real_binned, (0, padding_size))
        # compute jsd
        jsd = scipy.spatial.distance.jensenshannon(real_binned, virtual_binned, 2)
        jsd_dict[col] = jsd
    if not aggregate_results:
        return jsd_dict
    if aggregate_results and aggregation_method == AggregationMethod.AVERAGE:
        jsd_aggregated = np.mean(np.array(list(jsd_dict.values())))
    elif aggregate_results and aggregation_method == AggregationMethod.MEDIAN:
        jsd_aggregated = np.median(np.array(list(jsd_dict.values())))
    if not score:
        return jsd_aggregated
    else:
        return int((1 - jsd_aggregated) * 100)


def correlation(real: pandas.DataFrame, synthetic: pandas.DataFrame, score=True) -> float:
    """
    Computes the correlation similarity of real and synthetic data.

    :param real: The real data.
    :param synthetic: The synthetic data.
    :param score: Return result in a normalized score in [0,100]. Default is True.
    :return: Correlation score / Norm Quotient
    """
    corr_real = real.corr()
    corr_synthetic = synthetic.corr()
    norm_diff = np.linalg.norm(corr_real - corr_synthetic)
    norm_real = np.linalg.norm(corr_real)
    norm_quotient = norm_diff / norm_real
    if score:
        # will not happen in a realistic scenario, only if (nearly) all correlations are opposing
        if norm_quotient > 1:
            return 0
        return (1 - max(norm_quotient, 0)) * 100
    else:
        return norm_quotient
