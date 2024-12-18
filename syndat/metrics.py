import logging

import pandas as pd
import numpy as np
import scipy
from sklearn import ensemble
from sklearn.model_selection import StratifiedKFold, cross_val_score

from syndat._utils import filter_rows_with_common_categories, encode_categorical, encode_and_get_correlations

logger = logging.getLogger(__name__)


def discriminator_auc(real_data: pd.DataFrame,
                      synthetic_data: pd.DataFrame,
                      n_folds=5,
                      drop_na_threshold=0.9) -> float:
    """
    Evaluate the synthetic data using a Random Forest classifier.

    :param real_data: The real dataset.
    :param synthetic_data: The synthetic dataset.
    :param n_folds: The number of folds for cross-validation.
    :param drop_na_threshold: The threshold for dropping columns with high missingness.

    :return: The mean AUC score across all folds of the Random Forest classifier.
    """
    # filter out entries that are not inclusive for both datasets
    real_filtered, synthetic_filtered = filter_rows_with_common_categories(real_data, synthetic_data)
    # check for missing values in real data
    real_data_clean = real_filtered.dropna(thresh=int(drop_na_threshold * len(real_filtered)), axis=1)
    logger.info(f'Dropped {real_data_clean.shape[1] - real_data_clean.shape[1]} '
                f'due to high missingness (threshold is {drop_na_threshold}).')
    real_data_clean = real_data_clean.dropna()
    logger.info(f'Removed {len(real_data) - len(real_data_clean)} entries due to missing values.')
    # assert that both real and synthetic have same columns
    synthetic_clean = synthetic_filtered[real_data_clean.columns]
    # one-hot-encode categorical columns
    real_clean_encoded = encode_categorical(real_data_clean)
    synthetic_clean_encoded = encode_categorical(synthetic_clean)
    # train & test the classifier
    x = pd.concat([real_clean_encoded, synthetic_clean_encoded])
    y = np.concatenate((np.zeros(real_clean_encoded.shape[0]), np.ones(synthetic_clean_encoded.shape[0])), axis=None)
    rfc = ensemble.RandomForestClassifier()
    skf = StratifiedKFold(n_splits=n_folds)
    # Calculate cross-validated AUC score
    mean_auc = np.mean(cross_val_score(rfc, x, y, cv=skf, scoring='roc_auc'))
    return mean_auc


def jensen_shannon_divergence(real_data: pd.DataFrame,
                              synthetic_data: pd.DataFrame,
                              n_unique_threshold=10) -> dict[str, float]:
    """
    Computes the Jensen-Shannon Divergence between the real and synthetic data.

    :param real_data: The real dataset.
    :param synthetic_data: The synthetic dataset.
    :param n_unique_threshold: The threshold for the number of unique values in a column to be considered ordinal.

    :return: The Jensen-Shannon Divergence for each column.
    """
    # Initialize a dictionary to store JSD values for each column
    jsd_dict = {}
    # Identify categorical columns
    categorical_columns = real_data.select_dtypes(include=['object', 'category']).columns
    for col in real_data:
        # Delete empty cells
        real_wo_missing = real_data[col].dropna()
        # Check and align data types between real and synthetic data
        col_dtype_real = real_data[col].dtype
        col_dtype_synthetic = synthetic_data[col].dtype
        if col_dtype_real != col_dtype_synthetic:
            logger.warning(f'Real data at col {col} is dtype {col_dtype_real} but synthetic is {col_dtype_synthetic}. '
                           f'Evaluation will be done based on the assumed data type of the real data.')
            synthetic_data[col] = synthetic_data[col].astype(col_dtype_real)
        # categorical column
        if col in categorical_columns:
            # Handle categorical columns
            real_cat = real_data[col].astype('category')
            synthetic_cat = synthetic_data[col].astype('category')
            # Compute frequency distributions
            real_counts = real_cat.value_counts().sort_index()
            synthetic_counts = synthetic_cat.value_counts().sort_index()
            # Fill missing categories with zero
            all_categories = real_counts.index.union(synthetic_counts.index)
            real_binned = real_counts.reindex(all_categories, fill_value=0)
            synthetic_binned = synthetic_counts.reindex(all_categories, fill_value=0)
        # ordinal column
        elif col_dtype_real == "int64" and len(real_data[col].unique()) <= n_unique_threshold:
            # Handle ordinal columns
            min_value = min(real_data[col].min(), synthetic_data[col].min())
            if min_value < 0:
                real_data[col] += abs(min_value)
                synthetic_data[col] += abs(min_value)
            real_binned = np.bincount(real_data[col])
            synthetic_binned = np.bincount(synthetic_data[col])
        # numerical column
        else:
            # Handle numerical columns
            n_bins = np.histogram_bin_edges(real_wo_missing, bins='auto')
            real_binned = np.bincount(np.digitize(real_wo_missing, n_bins))
            synthetic_binned = np.bincount(np.digitize(synthetic_data[col], n_bins))
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
    return jsd_dict


def normalized_correlation_quotient(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """
    Computes the normalized correlation quotient between the real and synthetic data.

    :param real_data: The real dataset.
    :param synthetic_data: The synthetic dataset.

    :return: The normalized correlation quotient between the real and synthetic data.
    """
    corr_real, corr_synthetic, corr_diff = encode_and_get_correlations(real_data, synthetic_data)
    norm_diff = np.linalg.norm(corr_diff)
    norm_real = np.linalg.norm(corr_real)
    norm_quotient = norm_diff / norm_real
    return norm_quotient
