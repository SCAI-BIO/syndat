import logging

import pandas as pd
import numpy as np

from typing import Dict, Literal

import scipy
from sklearn import ensemble
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def discriminator_auc(real: pd.DataFrame, synthetic: pd.DataFrame, n_folds: int = 5,
                      drop_na_threshold: float = 0.9) -> float:
    """
    Computes the ROC AUC score of a classifier trained to differentiate between real and synthetic data.

    :param real: The real data.
    :param synthetic: The synthetic data
    :param n_folds: Number of k folds for cross-validation.
    :param drop_na_threshold: Percentage of non-missing values required of any column
    :return: AUC ROC Score
    """
    # filter out entries that are not inclusive for both datasets
    real_filtered, synthetic_filtered = __filter_rows_with_common_categories(real, synthetic)
    # check for missing values in real data
    real_clean = real_filtered.dropna(thresh=int(drop_na_threshold * len(real_filtered)), axis=1)
    n_dropped_columns = real_filtered.shape[1] - real_clean.shape[1]
    logger.info(f'Dropped {n_dropped_columns} columns due to high missingness (threshold is {drop_na_threshold}).')
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
    return auc_score


def jensen_shannon_distance(real: pd.DataFrame, synthetic: pd.DataFrame,
                            n_unique_threshold: int = 10) -> Dict[str, float]:
    """
    Computes the Jensen-Shannon distance (JSD) between the distributions of each feature in the real and synthetic datasets.

    The Jensen-Shannon distance is a symmetric and finite measure of similarity between two probability distributions.
    It ranges from 0 (identical distributions) to 1 (maximally different distributions). This function applies the JSD
    per column, handling categorical, ordinal, and numerical data differently:

    - For categorical columns, JSD is computed from the frequency counts of each category.
    - For ordinal columns (integer dtype with unique values below a threshold), JSD is computed from binned counts.
    - For numerical columns, histograms with automatically determined bins are used to compute JSD.
    - If distributions are disjoint, the JSD is set to 1.

    :param real: DataFrame containing the real data.
    :param synthetic: DataFrame containing the synthetic data.
    :param n_unique_threshold: Maximum number of unique integer values for a column to be treated as ordinal.
    :return: Dictionary mapping each column name to its Jensen-Shannon distance.
    """
    jsd_dict: Dict[str, float] = {}
    categorical_columns = real.select_dtypes(include=['object', 'category']).columns
    for col in real.columns:
        real_col = real[col].dropna().copy()
        synthetic_col = synthetic[col].dropna().copy()
        # Align types
        if real[col].dtype != synthetic[col].dtype:
            logger.warning(f'Dtype mismatch in column "{col}": real={real[col].dtype}, '
                           f'synthetic={synthetic[col].dtype}')
            try:
                synthetic_col = synthetic_col.astype(real[col].dtype)
            except Exception as e:
                logger.warning(f'Could not cast column "{col}": {e}')
                continue
        # Categorical column
        if col in categorical_columns:
            real_counts = real_col.astype('category').value_counts().sort_index()
            synthetic_counts = synthetic_col.astype('category').value_counts().sort_index()
            all_categories = real_counts.index.union(synthetic_counts.index)
            real_binned = real_counts.reindex(all_categories, fill_value=0).to_numpy()
            synthetic_binned = synthetic_counts.reindex(all_categories, fill_value=0).to_numpy()
        # Ordinal column
        elif pd.api.types.is_integer_dtype(real_col) and real_col.nunique() <= n_unique_threshold:
            real_col = real_col.astype(int)
            synthetic_col = synthetic_col.astype(int)
            min_val = min(real_col.min(), synthetic_col.min())
            if min_val < 0:
                shift = abs(min_val)
                real_col += shift
                synthetic_col += shift
            real_binned = np.bincount(real_col)
            synthetic_binned = np.bincount(synthetic_col)
        # Numerical column
        else:
            # rare edge case: both series are completely disjoint -> JSD is 1
            if real_col.max() < synthetic_col.min() or synthetic_col.max() < real_col.min():
                jsd_dict[col] = 1
                continue
            # they are not -> compute optimal (joined) bins
            combined = pd.concat([real_col, synthetic_col])
            bins = np.histogram_bin_edges(combined, bins='auto')
            real_binned, _ = np.histogram(real_col, bins=bins)
            synthetic_binned, _ = np.histogram(synthetic_col, bins=bins)
        # Pad to equal length
        max_len = max(len(real_binned), len(synthetic_binned))
        real_binned = np.pad(real_binned, (0, max_len - len(real_binned)))
        synthetic_binned = np.pad(synthetic_binned, (0, max_len - len(synthetic_binned)))
        # Compute JSD
        jsd = scipy.spatial.distance.jensenshannon(real_binned, synthetic_binned, base=2)
        jsd_dict[col] = float(jsd)
    return jsd_dict


def normalized_correlation_difference(real: pd.DataFrame, synthetic: pd.DataFrame,
                                      method: Literal['pearson', 'kendall', 'spearman'] = 'spearman') -> float:
    """
    Computes the correlation similarity of real and synthetic data by comparing the correlation matrices of both
    datasets. The score is calculated as the norm quotient of the difference between the correlation matrices of real
    and synthetic data. This quotient is 0 for equal correlation matrices and 1 for completely different correlation
    matrices. The norm can in theory exceed 1, in cases of opposing correlation structured (e.g. only negative
    correlations in the real data, but only positive correlations in the synthetic data).

    :param real: The real data.
    :param synthetic: The synthetic data.
    :param method: The method to use for correlation computation. Either of "pearson", "kendall" or "spearman".
    :return: correlation quotient
    """
    # check if both datasets have the same columns
    if set(real.columns) != set(synthetic.columns):
        excluded_columns = list((set(real.columns) ^ set(synthetic.columns)))
        common_cols = list(set(real.columns) & set(synthetic.columns))
        logger.warning(f"Excluding columns from correlation computation that are not present in both datasets: "
                       f"\n {excluded_columns}.")
        real_common_cols = real[common_cols]
        synthetic_common_cols = synthetic[common_cols]
    else:
        real_common_cols = real
        synthetic_common_cols = synthetic
    # filter out rows with categories that are not present in both datasets
    # why do we do this? while it is expected that categorical columns of both data sets contain the same categories,
    # it is entirely possible that this is not the case. If this happens, we omit this extra category from computation
    # to avoid possible crashes in correlation computation.
    real_filtered, synthetic_filtered = __filter_rows_with_common_categories(real_common_cols, synthetic_common_cols)
    # filter out columns with too few valid combinations -> this can happen when columns have a high missingness and
    # have less than 2 valid row combinations without missing values
    real_filtered, synthetic_filtered = __filter_nan_exclusive_combinations(real_filtered, synthetic_filtered)
    # if columns were excluded from computation, log this
    if len(real_filtered.columns) != len(real_common_cols.columns) or len(synthetic_filtered.columns) != len(synthetic_common_cols.columns):
        excluded_columns = list(set(real.columns) - set(real_filtered.columns))
        logger.warning(f'The following columns were excluded from correlation computation due to insufficient '
                       f'or invalid combinations: {excluded_columns}. \nThe result should only be interpreted with '
                       f'regards to the correlations of the remaining columns.')
    # check if both datatsets contain more than one column after filtering
    if real_filtered.shape[1] <= 1 or synthetic_filtered.shape[1] <= 1:
        logger.warning("Not enough columns left for correlation computation after filtering. Returning nan.")
        return np.nan
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
    corr_real = real_numerical.corr(method=method)
    corr_synthetic = synthetic_numerical.corr(method=method)
    # Remove one-hot-encoded categories from one dimension - otherwise we compute correlations within the same column
    # which would distort results
    one_hot_encoded_columns = list(set(real_encoded.columns) - set(real_common_cols.columns))
    # beware of special case: only categorical values. In this case we cant drop all columns because the dataframe
    # would be empty afterwards TODO: handle this more gracefully > differentiate between encodings of different cats
    existing_columns = [col for col in one_hot_encoded_columns if col in corr_real.columns]
    if not corr_real.drop(columns=existing_columns).empty:
        corr_real = corr_real.drop(columns=one_hot_encoded_columns)
        corr_synthetic = corr_synthetic.drop(columns=one_hot_encoded_columns)
    # now compute correlation matrices
    corr_diff = corr_real - corr_synthetic
    norm_diff = np.linalg.norm(corr_diff)
    norm_real = np.linalg.norm(corr_real)
    norm_quotient = norm_diff / norm_real
    return norm_quotient


def __filter_nan_exclusive_combinations(real: pd.DataFrame, synthetic: pd.DataFrame
                                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes columns from both datasets if they have too few valid combinations with other columns,
    either in the real or synthetic dataset.
    """
    real_invalid = __find_invalid_column_combinations(real)
    synthetic_invalid = __find_invalid_column_combinations(synthetic)
    all_invalid = sorted(set(real_invalid).union(synthetic_invalid))
    if all_invalid:
        real = real.drop(columns=all_invalid, errors='ignore')
        synthetic = synthetic.drop(columns=all_invalid, errors='ignore')
    return real, synthetic


def __find_invalid_column_combinations(df: pd.DataFrame) -> list[str]:
    """
    Identifies columns in the DataFrame that are result in invalid correlation computations when paired with another
    column. This may happen either due to:
    - Too few valid (non-NaN) combinations with another column.
    - Either column being constant after dropping NaN values for both pairs.
    :param df: The dataframe to check for valid combinations.
    :return: List of column names that should be dropped due to insufficient valid combinations.
    """
    sorted_columns = df.isna().sum().sort_values(ascending=False).index.tolist()
    columns_to_drop = set()
    for column in sorted_columns:
        if column in columns_to_drop:
            continue
        for other_column in df.columns:
            if column == other_column or other_column in columns_to_drop:
                continue
            non_na_combinations = df[[column, other_column]].dropna()
            if non_na_combinations.shape[0] < 2:
                logger.warning(
                    f'Removing column "{column}" from correlation computation due to insufficient valid (non-nan) '
                    f'combinations with column "{other_column}".'
                )
                columns_to_drop.add(column)
                break
            elif non_na_combinations[column].nunique() <= 1 or non_na_combinations[other_column].nunique() <= 1:
                if non_na_combinations[column].nunique() <= 1:
                    logger.warning(
                        f'Removing column "{column}" from correlation computation due to only constant values '
                        f'(zero-variance) remaining after dropping NaN values in combination with column '
                        f'"{other_column}".'
                    )
                    columns_to_drop.add(column)
                if non_na_combinations[other_column].nunique() <= 1:
                    logger.warning(
                        f'Removing column "{other_column}" from correlation computation due to only constant values '
                        f'(zero-variance) remaining after dropping NaN values in combination with column '
                        f'"{column}".'
                    )
                    columns_to_drop.add(other_column)
                break
    return list(columns_to_drop)


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


