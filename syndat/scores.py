import warnings
from typing import Union, List

import logging
import pandas
import pandas as pd
import numpy as np
import scipy.spatial.distance

from sklearn import ensemble
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from syndat.domain import AggregationMethod

logger = logging.getLogger(__name__)


def auc(real: pandas.DataFrame, synthetic: pandas.DataFrame, n_folds=5,
        drop_na_threshold=0.9, score: bool = True) -> float:
    """
    Computes the Discrimination Complexity Score / ROC AUC score of a classifier trained to differentiate between real
    and synthetic data.

    :param real: The real data.
    :param synthetic: The synthetic data
    :param n_folds: Number of k folds for cross-validation.
    :param drop_na_threshold: Percentage of non-missing values required of any column
    :param score: Return result in a normalized score in [0,100]. Default is True.
    :return: Differentiation Complexity Score / AUC ROC Score
    """

    warnings.warn(
        "auc is deprecated and will be removed in a future version. Please use discrimination instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return discrimination(real, synthetic, n_folds=n_folds, drop_na_threshold=drop_na_threshold, score=score)


def jsd(real: pd.DataFrame, synthetic: pd.DataFrame, aggregate_results: bool = True,
        aggregation_method: AggregationMethod = AggregationMethod.AVERAGE, score: bool = True,
        n_unique_threshold=10) -> Union[List[float], float]:
    """
    Computes the feature distribution similarity using the Jensen-Shannon distance of real and synthetic data.

    :param real: The real data.
    :param synthetic: The synthetic data.
    :param aggregate_results: Compute a single aggregated score for all features. Default is True.
    :param aggregation_method: How the scores are aggregated. Default is using the median of all feature scores.
    :param score: Return result in a normalized score in [0,100]. Default is True.
    :param n_unique_threshold: Threshold to determine at which number of unique values bins will span over several
    values.
    :return: Distribution Similarity / JSD
    """

    warnings.warn(
        "auc is deprecated and will be removed in a future version. Please use discrimination instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return distribution(real, synthetic, aggregate_results, aggregation_method, score, n_unique_threshold)


def discrimination(real: pandas.DataFrame, synthetic: pandas.DataFrame, n_folds=5,
                   drop_na_threshold=0.9, score: bool = True) -> float:
    """
    Computes the Discrimination Complexity Score / ROC AUC score of a classifier trained to differentiate between real
    and synthetic data.

    :param real: The real data.
    :param synthetic: The synthetic data
    :param n_folds: Number of k folds for cross-validation.
    :param drop_na_threshold: Percentage of non-missing values required of any column
    :param score: Return result in a normalized score in [0,100]. Default is True.
    :return: Differentiation Complexity Score / AUC ROC Score
    """
    # filter out entries that are not inclusive for both datasets
    real_filtered, synthetic_filtered = __filter_rows_with_common_categories(real, synthetic)
    # check for missing values in real data
    real_clean = real_filtered.dropna(thresh=int(drop_na_threshold * len(real_filtered)), axis=1)
    logger.info(f'Dropped {real_clean.shape[1] - real_clean.shape[1]} '
                 f'due to high missingness (threshold is {drop_na_threshold}).')
    real_clean = real_clean.dropna()
    logger.info(f'Removed {len(real) - len(real_clean)} entries due to missing values.')
    # assert that both real and synthetic have same columns
    synthetic_clean = synthetic_filtered[real_clean.columns]
    # one-hot-encode categorical columns
    real_clean_encoded = __encode_categorical(real_clean)
    synthetic_clean_encoded = __encode_categorical(synthetic_clean)
    # train & test the classifier
    x = pd.concat([real_clean_encoded, synthetic_clean_encoded])
    y = np.concatenate((np.zeros(real_clean_encoded.shape[0]), np.ones(synthetic_clean_encoded.shape[0])), axis=None)
    rfc = ensemble.RandomForestClassifier()
    skf = StratifiedKFold(n_splits=n_folds)
    # Calculate cross-validated AUC score
    auc_score = np.mean(cross_val_score(rfc, x, y, cv=skf, scoring='roc_auc'))
    if score:
        return min(0.5, (1 - auc_score)) * 200
    else:
        return auc_score


def distribution(real: pd.DataFrame, synthetic: pd.DataFrame, aggregate_results: bool = True,
                 aggregation_method: AggregationMethod = AggregationMethod.AVERAGE, score: bool = True,
                 n_unique_threshold=10) -> Union[List[float], float]:
    """
    Computes the feature distribution similarity using the Jensen-Shannon distance of real and synthetic data.

    :param real: The real data.
    :param synthetic: The synthetic data.
    :param aggregate_results: Compute a single aggregated score for all features. Default is True.
    :param aggregation_method: How the scores are aggregated. Default is using the median of all feature scores.
    :param score: Return result in a normalized score in [0,100]. Default is True.
    :param n_unique_threshold: Threshold to determine at which number of unique values bins will span over several
    values.
    :return: Distribution Similarity / JSD
    """
    # Initialize a dictionary to store JSD values for each column
    jsd_dict = {}
    # Identify categorical columns
    categorical_columns = real.select_dtypes(include=['object', 'category']).columns
    for col in real:
        # Delete empty cells
        real_wo_missing = real[col].dropna()
        # Check and align data types between real and synthetic data
        col_dtype_real = real[col].dtype
        col_dtype_synthetic = synthetic[col].dtype
        if col_dtype_real != col_dtype_synthetic:
            logger.warning(f'Real data at col {col} is dtype {col_dtype_real} but synthetic is {col_dtype_synthetic}. '
                            f'Evaluation will be done based on the assumed data type of the real data.')
            synthetic[col] = synthetic[col].astype(col_dtype_real)
        # categorical column
        if col in categorical_columns:
            # Handle categorical columns
            real_cat = real[col].astype('category')
            synthetic_cat = synthetic[col].astype('category')
            # Compute frequency distributions
            real_counts = real_cat.value_counts().sort_index()
            synthetic_counts = synthetic_cat.value_counts().sort_index()
            # Fill missing categories with zero
            all_categories = real_counts.index.union(synthetic_counts.index)
            real_binned = real_counts.reindex(all_categories, fill_value=0)
            synthetic_binned = synthetic_counts.reindex(all_categories, fill_value=0)
        # ordinal column
        elif col_dtype_real == "int64" and len(real[col].unique()) <= n_unique_threshold:
            # Handle ordinal columns
            min_value = min(real[col].min(), synthetic[col].min())
            if min_value < 0:
                real[col] += abs(min_value)
                synthetic[col] += abs(min_value)
            real_binned = np.bincount(real[col])
            synthetic_binned = np.bincount(synthetic[col])
        # numerical column
        else:
            # Handle numerical columns
            n_bins = np.histogram_bin_edges(real_wo_missing, bins='auto')
            real_binned = np.bincount(np.digitize(real_wo_missing, n_bins))
            synthetic_binned = np.bincount(np.digitize(synthetic[col], n_bins))
        # One array might be shorter than the other; fill missing bins with zero
        if len(real_binned) != len(synthetic_binned):
            padding_size = abs(len(real_binned) - len(synthetic_binned))
            if len(real_binned) > len(synthetic_binned):
                synthetic_binned = np.pad(synthetic_binned, (0, padding_size))
            else:
                real_binned = np.pad(real_binned, (0, padding_size))
        # Compute JSD
        jsd_value = scipy.spatial.distance.jensenshannon(real_binned, synthetic_binned, 2)
        jsd_dict[col] = jsd_value
    if not aggregate_results:
        return jsd_dict
    if aggregation_method == AggregationMethod.AVERAGE:
        jsd_aggregated = np.mean(list(jsd_dict.values()))
    elif aggregation_method == AggregationMethod.MEDIAN:
        jsd_aggregated = np.median(list(jsd_dict.values()))
    if not score:
        return jsd_aggregated
    else:
        return int((1 - jsd_aggregated) * 100)


def correlation(real: pd.DataFrame, synthetic: pd.DataFrame, score=True) -> float:
    """
    Computes the correlation similarity of real and synthetic data.

    Filters out rows with categories not present in both datasets, encodes categorical columns,
    and computes the correlation matrix.

    The goal is to ensure that only comparable data is used in the correlation calculation.

    :param real: The real data.
    :param synthetic: The synthetic data.
    :param score: Return result in a normalized score in [0,100]. Default is True.
    :return: Correlation score / Norm Quotient
    """
    # why do we do this? while it is expected that categorical columns of both data sets contain the same categories,
    # it is entirely possible that this is not the case. If this happens, we omit this extra category from computation
    # to avoid possible crashes in correlation computation.
    real_filtered, synthetic_filtered = __filter_rows_with_common_categories(real, synthetic)
    # Encode categorical columns
    real_encoded = __encode_categorical(real_filtered)
    synthetic_encoded = __encode_categorical(synthetic_filtered)
    # Compute numerical correlation only
    real_numerical = real_encoded.select_dtypes(include=[np.number])
    synthetic_numerical = synthetic_encoded.select_dtypes(include=[np.number])
    # Remove constant columns (zero variance)
    constant_columns = real_numerical.columns[real_numerical.nunique() <= 1]
    if len(constant_columns) > 0:
        logger.warning(f'Removing constant columns {constant_columns} for correlation computation.')
        real_numerical = real_numerical.drop(columns=constant_columns, errors="ignore")
        synthetic_numerical = synthetic_numerical.drop(columns=constant_columns, errors="ignore")
    # Compute correlation matrices
    corr_real = real_numerical.corr(method='spearman')
    corr_synthetic = synthetic_numerical.corr(method='spearman')
    # Remove one-hot-encoded categories from one dimension - otherwise we compute correlations within the same column
    # which would distort results
    one_hot_encoded_columns = list(set(real_encoded.columns) - set(real.columns))
    # beware of special case: only categorical values. In this case we cant drop all columns because the dataframe
    # would be empty afterwards TODO: handle this more gracefully > differentiate between encodings of different cats
    if not corr_real.drop(columns=one_hot_encoded_columns).empty:
        corr_real = corr_real.drop(columns=one_hot_encoded_columns)
        corr_synthetic = corr_synthetic.drop(columns=one_hot_encoded_columns)
    # now compute correlation matrices
    corr_diff = corr_real - corr_synthetic
    norm_diff = np.linalg.norm(corr_diff)
    norm_real = np.linalg.norm(corr_real)
    norm_quotient = norm_diff / norm_real
    if score:
        # Normalize to score in [0, 100]
        if norm_quotient > 1:
            return 0
        return (1 - max(norm_quotient, 0)) * 100
    else:
        return norm_quotient


def __filter_rows_with_common_categories(real: pd.DataFrame, synthetic: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Filters out rows with categories that are not present in both datasets.

    Only keeps rows with categories that are present in both datasets.
    This ensures that both datasets are comparable in terms of categorical data.

    :param real: The real data.
    :param synthetic: The synthetic data.
    :return: Filtered real and synthetic datasets with only common categories in rows.
    """
    # Identify categorical columns
    real_categorical_cols = real.select_dtypes(include=['object', 'category']).columns
    synthetic_categorical_cols = synthetic.select_dtypes(include=['object', 'category']).columns
    # Identify common categorical columns
    common_categorical_cols = set(real_categorical_cols) & set(synthetic_categorical_cols)
    # Filter rows with common categories in each column
    for col in common_categorical_cols:
        real_categories = set(real[col].unique())
        synthetic_categories = set(synthetic[col].unique())
        common_categories = real_categories & synthetic_categories
        if len(real_categories - common_categories) > 0:
            logger.warning(
                f"Categories {real_categories - common_categories} in column '{col}' are in real data but not in "
                f"synthetic data. They will not be considered in the score computation.")
        # Filter rows to keep only common categories
        real = real[real[col].isin(common_categories)]
        synthetic = synthetic[synthetic[col].isin(common_categories)]
    return real, synthetic


def __encode_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns in the dataframe using one-hot encoding.

    Converts categorical variables into a format suitable for correlation computation.

    :param data: The dataframe with potential categorical columns.
    :return: DataFrame with categorical columns one-hot encoded.
    """
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) == 0:
        return data
    # Initialize encoder without dropping any category
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = data.copy()
    for col in categorical_cols:
        encoded_col = encoder.fit_transform(data[[col]])
        encoded_df = pd.DataFrame(encoded_col, columns=encoder.get_feature_names_out([col]), index=data.index)
        encoded_data = encoded_data.drop(col, axis=1)
        encoded_data = pd.concat([encoded_data, encoded_df], axis=1)
    return encoded_data
