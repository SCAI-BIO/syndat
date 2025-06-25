import pandas as pd
import numpy as np


def normalize_scale(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales the columns in the synthetic DataFrame to match the scale (min and max values) of the corresponding columns
    in the real DataFrame.

    :param real_df: The real dataset used as the scaling reference.
    :param synthetic_df: The synthetic dataset to be scaled.
    :return: The scaled synthetic dataset with columns adjusted to the real dataset's scale.
    """
    # Create a copy of the synthetic dataframe to avoid modifying the original one
    scaled_synthetic_df = synthetic_df.copy()

    # Iterate over each column in the synthetic dataframe
    for column in synthetic_df.columns:
        # Check if the column is of floating-point type in both real and synthetic data
        if np.issubdtype(synthetic_df[column].dtype, np.floating):
            # Find the min and max values in the real data for this column
            real_min = real_df[column].min()
            real_max = real_df[column].max()

            # Find the min and max values in the synthetic data for this column
            synthetic_min = synthetic_df[column].min()
            synthetic_max = synthetic_df[column].max()

            # Scale the synthetic data to match the min/max of the real data
            scaled_synthetic_df[column] = ((synthetic_df[column] - synthetic_min) / (synthetic_max - synthetic_min)) * (
                        real_max - real_min) + real_min

    return scaled_synthetic_df


def assert_minmax(real: pd.DataFrame, synthetic: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
    """
    Postprocess the synthetic data by either deleting records that fall outside the min-max range of the real data,
    or adjusting them to fit within the range. Also normalizes -0.0 to 0.0 to avoid plotting issues.

    :param real: The real dataset.
    :param synthetic: The synthetic dataset.
    :param method: The method to apply. 'delete' to remove records, 'clip' to adjust them.
    :return: The postprocessed synthetic dataset.
    """
    # Normalize -0.0 to 0.0 in synthetic data
    synthetic = synthetic.apply(lambda col: col.map(lambda x: 0.0 if x == -0.0 else x))

    # Iterate over each column in the synthetic DataFrame
    for column in synthetic.columns:
        if column in real.columns:
            # Get the min and max of the column in the real data
            min_val = real[column].min()
            max_val = real[column].max()

            if method == 'delete':
                # Filter the synthetic DataFrame to keep only rows within the min-max range
                synthetic = synthetic[(synthetic[column] >= min_val) & (synthetic[column] <= max_val)]
            elif method == 'clip':
                # Clip the values to be within the min-max range
                synthetic[column] = synthetic[column].clip(lower=min_val, upper=max_val)

    return synthetic


def normalize_float_precision(real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """
    Postprocess the synthetic data to match the precision or step size found in the real data for float columns.

    This function identifies columns in the real dataset that have float data types and determines the precision
    or step size (e.g., 1.0, 0.5, 0.1) used in those columns. It then rounds the corresponding columns in the 
    synthetic dataset to match this detected precision or step size.

    :param real: The real dataset containing float columns.
    :param synthetic: The synthetic dataset that needs to be adjusted to match the precision of the real data.
    :return: The synthetic dataset with float columns rounded to match the precision or step size of the real data.
    """
    # Select float columns from the real dataset
    float_columns = real.select_dtypes(include='float').columns

    for col in float_columns:
        if col in synthetic.columns:
            # Get the unique values from the real data column, excluding NaN
            unique_values = real[col].dropna().unique()

            # Calculate the differences between the unique sorted values
            unique_diffs = np.diff(np.sort(unique_values))

            # If the unique values are all the same, continue to the next column
            if len(unique_diffs) == 0:
                continue

            # Find the smallest non-zero difference (the step size)
            step_size = np.min(unique_diffs[unique_diffs > 0])

            # Round the synthetic column to the nearest multiple of the step size
            synthetic[col] = np.round(synthetic[col] / step_size) * step_size

    return synthetic
