import logging

import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

def convert_to_syndat_scores(
    df: pd.DataFrame,
    only_pos: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts a DataFrame containing observed and predicted (REC_) columns into two separate DataFrames,
    synchronizing values based on MASK columns and optionally clipping predicted values to be non-negative.

    :param df: DataFrame containing columns with prefixes 'OBS_', 'REC_', 'MASK_' and a 'REPI' column.
    :param only_pos: If True, clips negative values in 'REC_' columns to zero.
    :return: Tuple of two DataFrames: (observed_df, predicted_df) with synchronized and filtered values.
    """
    if "REPI" not in df.columns:
        logger.info("Column 'REPI' not found — adding it with ones for library compatibility.")
        df["REPI"] = 1

    df = df[df.REPI == 1]
    for col in df.columns:
        if only_pos and col.startswith("REC_"):
            df[col] = df[col].clip(lower=0)
        if col.startswith("MASK"):
            var_name = "_".join(col.split("_")[1:])
            mask = df[col] == 0
            # Setting values to NAN where  mask is 0
            df[f'OBS_{var_name}'] = df[f'OBS_{var_name}'].mask(mask)
            df[f'REC_{var_name}'] = df[f'REC_{var_name}'].mask(mask)

    observed_df = df[[col for col in df.columns if col.startswith("OBS")]]
    predictions_df = df[[col for col in df.columns if col.startswith("REC")]]
    predictions_df.columns = observed_df.columns
    return observed_df, predictions_df

def get_rp(
    ldt: pd.DataFrame,
    lt: Optional[pd.DataFrame] = None,
    st: Optional[pd.DataFrame] = None) -> dict:
    """
    Creates a dictionary with static and longitudinal variable names categorized by type (categorical, continuous),
    and computes the maximum time value.

    :param ldt: Longitudinal DataFrame containing at least a 'TIME' column.
    :param lt: Longitudinal variables metadata DataFrame with columns 'Variable', 'Type', and 'Cats'.
    :param st: Static variables metadata DataFrame with columns 'Variable' and 'Type'.
    :return: Dictionary with keys: 'Tmax', 'static_vnames', 'static_cat', 'static_cont',
             'long_vnames', 'long_cat', 'long_bin', 'long_cont', each mapping to lists of variable names.
    """

    if lt is None and st is None:
        raise ValueError("At least one of 'lt' or 'st' must be provided.")

    rp = {}
    rp['Tmax'] = ldt.TIME.max()

    if st is not None:
        rp["static_vnames"] = st["Variable"].dropna().unique().tolist()
        rp["static_cat"] = st[st["Type"] == "cat"]["Variable"].dropna().unique().tolist()
        rp["static_bin"] = st[(st["Type"] == "cat") & (st["Cats"] == 2)]["Variable"].dropna().unique().tolist()
        rp["static_cont"] = st[st["Type"] != "cat"]["Variable"].dropna().unique().tolist()
    else:
        rp["static_vnames"] = []
        rp["static_cat"] = []
        rp["static_bin"] = []
        rp["static_cont"] = []

    if lt is not None:
        rp["long_vnames"] = lt["Variable"].dropna().unique().tolist()
        rp["long_cat"] = lt[lt["Type"] == "cat"]["Variable"].dropna().unique().tolist()
        rp["long_bin"] = lt[(lt["Type"] == "cat") & (lt["Cats"] == 2)]["Variable"].dropna().unique().tolist()
        rp["long_cont"] = lt[lt["Type"] != "cat"]["Variable"].dropna().unique().tolist()
    else:
        rp["long_vnames"] = []
        rp["long_cat"] = []
        rp["long_bin"] = []
        rp["long_cont"] = []
    return rp

def convert_long_data_to_tidy(df0: pd.DataFrame, only_pos: bool = False) -> pd.DataFrame:
    """
    Converts a wide-format DataFrame to a tidy-format DataFrame by encoding variables into separate columns,
    separates variable types and names, applies masking, and optionally clips negative values.

    :param df0: Original wide-format DataFrame with encoded columns (e.g., 'OBS_var', 'REC_var', 'MASK_var').
    :param only_pos: If True, clips negative values in the 'DV' column to zero.
    :return: A tidy-format DataFrame with columns: 'SUBJID', 'REPI', 'TIME', 'DRUG', 'TYPE', 'Variable', 'DV', and 'MASK'.
    """

    if "DRUG" not in df0.columns:
        logger.info("Column 'DRUG' not found — adding it with zeros for library compatibility.")
        df0["DRUG"] = 0

    df1 = df0.melt(id_vars=["PTNO", "REPI", "TIME", "DRUG"], 
                   var_name="FullVar", value_name="DV")
    df1[['TYPE', 'Variable']] = df1['FullVar'].str.extract(r'([^_]+)_(.*)')
    df1.drop(columns='FullVar', inplace=True)
    df1["TIME2"] = (df1["TIME"] * 1_000_000).round().astype(int)
    if only_pos:
        df1["DV"] = df1["DV"].clip(lower=0)
    df_main = df1[df1["TYPE"] != "MASK"].copy()
    df_mask = df1[df1["TYPE"] == "MASK"].copy()

    df_mask = df_mask.drop(columns=["TIME", "TYPE", "DRUG"]).rename(columns={"DV": "MASK"})
    df_final = df_main.merge(df_mask, on=["PTNO", "REPI", "Variable", "TIME2"], how="left")
    df_final["TIME"] = df_final["TIME"].round(2)
    df_final.drop(columns="TIME2", inplace=True)
    df_final.rename(columns={"PTNO": "SUBJID"}, inplace=True)
    return df_final

def convert_static_data_to_tidy(df0: pd.DataFrame, only_pos: bool = False) -> pd.DataFrame:
    """
    Converts a wide-format static DataFrame to a tidy-format DataFrame.
    It melts the dataframe, separates variable types and names, applies masking,
    and optionally clips negative values.

    :param df0: Input wide-format DataFrame containing static data with columns including 'PTNO' and 'REPI'.
    :param only_pos: If True, clips negative values in the data to zero.
    :return: A tidy-format DataFrame with columns ['SUBJID', 'REPI', 'TYPE', 'Variable', 'DV', 'MASK'].
    """

    id_vars = ["PTNO", "REPI"]
    cols_ = ["TYPE"]
    if "DRUG" in df0.columns:
        id_vars = id_vars + ["DRUG"]
        cols_ = cols_ + ["DRUG"]
    df1 = df0.melt(id_vars=id_vars, 
                   var_name="FullVar", 
                   value_name="DV")
    df1[["TYPE", "Variable"]] = df1["FullVar"].str.extract(r'([^_]+)_(.*)')
    df1.drop(columns='FullVar', inplace=True)
    if only_pos:
        df1["DV"] = df1["DV"].clip(lower=0)
    df_mask = df1[df1["TYPE"] == "MASK"].drop(columns=cols_).rename(columns={"DV": "MASK"})
    df_final = df1[df1["TYPE"] != "MASK"].merge(df_mask, on=["PTNO", "REPI", "Variable"], how="left")
    df_final = df_final.rename(columns={"PTNO": "SUBJID"})
    return df_final

def convert_data_to_tidy(df0: pd.DataFrame, type: str, only_pos: bool = False, only_realtimes: bool = False) -> pd.DataFrame:
    """
    Converts a DataFrame from wide to tidy format, depending on whether it is longitudinal or static data.
    It uses `convert_long_data` for longitudinal data and `convert_static_data` for static data.
    It filters the observations to keep only observed values where the mask is 1, and renames type categories.

    :param df0: Input wide-format DataFrame.
    :param type: Type of data to convert; 'long' for longitudinal or other values for static data.
    :param only_pos: If True, clips negative values in the data to zero.
    :return: A tidy-format DataFrame with standardized TYPE column ('Observed', 'Reconstructed', 'Simulations').
    """
    if "REPI" not in df0.columns:
        logger.info("Column 'REPI' not found — adding it with ones for library compatibility.")
        df0["REPI"] = 1

    if type=='long':
        df_final = convert_long_data_to_tidy(df0,only_pos=only_pos)
    else:
        df_final = convert_static_data_to_tidy(df0,only_pos=only_pos)

    if only_realtimes:
        df_final = df_final[df_final["MASK"] == 1]
    else:
        df_final = df_final[
            ((df_final["TYPE"] == "OBS") & (df_final["MASK"] == 1)) | (df_final["TYPE"] != "OBS")]

    df_final = df_final.drop(columns=["MASK"])

    df_final["TYPE"] = df_final["TYPE"].replace({
        "OBS": "Observed",
        "REC": "Reconstructed",
        "SIM": "Simulations"})
    return df_final