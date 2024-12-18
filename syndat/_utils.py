import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def encode_and_get_correlations(real: pd.DataFrame,
                                synthetic: pd.DataFrame
                                ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Encode categorical columns and compute the correlation matrices of real and synthetic data.

    :param real: The real dataset.
    :param synthetic: The synthetic dataset.
    :return: The correlation matrices of the real and synthetic data, and their difference.
    """
    # why do we do this? while it is expected that categorical columns of both data sets contain the same categories,
    # it is entirely possible that this is not the case. If this happens, we omit this extra category from computation
    # to avoid possible crashes in correlation computation.
    real_filtered, synthetic_filtered = filter_rows_with_common_categories(real, synthetic)
    # Encode categorical columns
    real_encoded = encode_categorical(real_filtered)
    synthetic_encoded = encode_categorical(synthetic_filtered)
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
    corr_real = real_numerical.corr()
    corr_synthetic = synthetic_numerical.corr()
    # Remove one-hot-encoded categories from one dimension - otherwise we compute correlations within the same column
    # which would distort results
    one_hot_encoded_columns = list(set(real_encoded.columns) - set(real.columns))
    # beware of special case: only categorical values. In this case we cant drop all columns because the dataframe
    # would be empty afterwards TODO: handle this more gracefully > differentiate between encodings of different cats
    if not corr_real.drop(columns=one_hot_encoded_columns).empty:
        corr_real = corr_real.drop(columns=one_hot_encoded_columns)
        corr_synthetic = corr_synthetic.drop(columns=one_hot_encoded_columns)
    return corr_real, corr_synthetic, corr_real - corr_synthetic


def encode_categorical(data: pd.DataFrame) -> pd.DataFrame:
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


def filter_rows_with_common_categories(real: pd.DataFrame, synthetic: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
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
