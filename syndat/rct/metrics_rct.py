import logging
import numpy as np
import pandas as pd

from typing import Optional, List, Dict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

logger = logging.getLogger(__name__)

def compute_continuous_error_metrics(
    rp0: dict,
    dt: pd.DataFrame,
    mode: str = "Reconstructed",
    strat_vars: Optional[List[str]] = None,
    static: bool = False,
    per_time_mean: bool = False,
    per_variable_mean: bool = False,
    epsilon: float = 1e-8) -> pd.DataFrame:
    """
    Computes MAE, RMSE, and MAPE stratified by variables like DRUG, per TIME and Variable.

    :param rp0: Dictionary with key 'long_cont' containing a list of continuous variable names.
    :param dt: Main DataFrame containing data for "Observed" and `mode`.
    :param mode: String, usually "Reconstructed", used for filtering TYPE.
    :param strat_vars: List of column names to stratify by (e.g., ['DRUG']).
    :param static: If True, metrics for static variables will be calculated.
    :param per_time_mean: If True, returns mean error per time point (averaged across variables).
    :param per_variable_mean: If True, returns mean error per variable (averaged across time).
    :param epsilon: Small value added to denominator for MAPE to avoid division by zero.
    :return: Dict of dataframes with keys:
             - 'full': full stratified error metrics
             - 'per_time': (optional) mean per time point
             - 'per_variable': (optional) mean per variable
             - 'overall': overall mean values
    """

    strat_vars = strat_vars or []

    if static:
        dt = (dt[(dt["REPI"] == 1 if mode == "Reconstructed" else True) &
                (dt['TYPE'].isin(["Observed", mode])) &
                (dt['Variable'].isin(rp0['static_cont']))])
        TIME_v = []
    else:
        dt = (dt[(dt["REPI"] == 1 if mode == "Reconstructed" else True) &
                (dt['TYPE'].isin(["Observed", mode])) &
                (dt['Variable'].isin(rp0['long_cont']))])
        TIME_v = ["TIME"]

    dt = dt.pivot_table(
        index=strat_vars + ["SUBJID", "Variable"] + TIME_v,
        columns="TYPE",
        values="DV").dropna(subset=["Observed", mode])

    # Compute error components
    dt["abs_error"] = np.abs(dt["Observed"] - dt[mode])
    dt["sq_error"] = (dt["Observed"] - dt[mode]) ** 2
    dt["pct_error"] = np.where(dt["Observed"] != 0,
                                    np.abs((dt["Observed"] - dt[mode]) / dt["Observed"]) * 100,
                                    np.nan)

    dt["pct_error"] = np.abs((dt["Observed"] - dt[mode]) / (dt["Observed"] + epsilon)) * 100

    # Aggregate per time point and variable
    full_df = dt.groupby(strat_vars + TIME_v + ["Variable"]).agg(
        MAE=("abs_error", "mean"),
        RMSE=("sq_error", lambda x: np.sqrt(np.mean(x))),
        MAPE=("pct_error", "mean")
    ).reset_index()
    result = {"full": full_df}

    # Per time point mean (across variables)
    if per_time_mean and not static:
        time_group = strat_vars + ["TIME"]
        per_time_df = full_df.groupby(time_group).agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            MAPE=("MAPE", "mean")
        ).reset_index()
        result["per_time"] = per_time_df

    # Per variable mean (across time)
    if per_variable_mean:
        var_group = strat_vars + ["Variable"]
        per_var_df = full_df.groupby(var_group).agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            MAPE=("MAPE", "mean")
        ).reset_index()
        result["per_variable"] = per_var_df

    # Overall
    if strat_vars:
        overall_df = full_df.groupby(strat_vars).agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            MAPE=("MAPE", "mean")
        ).reset_index()
    else:
        overall_df = pd.DataFrame([{
            "MAE": full_df["MAE"].mean(),
            "RMSE": full_df["RMSE"].mean(),
            "MAPE": full_df["MAPE"].mean()
        }])

    result["overall"] = overall_df
    for key in ["full", "per_time", "per_variable", "overall"]:
        if key in result:
            result[key][["MAE", "RMSE", "MAPE"]] = result[key][["MAE", "RMSE", "MAPE"]].round(3)

    return result


def compute_categorical_error_metrics(
    rp0: dict,
    dt: pd.DataFrame,
    mode: str = "Reconstructed",
    average: str = "weighted",  # "macro", "micro", or "weighted"
    strat_vars: Optional[List[str]] = None,
    static: bool = False,
    per_time_mean: bool = False,
    per_variable_mean: bool = False) -> Dict[str, pd.DataFrame]:

    """
    Computes classification metrics for long-format categorical variables.
    Returns a dict with full metrics, per-time mean, per-variable mean, and overall mean.

    :param rp0: Dictionary with key 'long_cat' and/or 'long_bin' with list of categorical variable names.
    :param dt: Main DataFrame containing data for "Observed" and `mode`.
    :param mode: String, usually "Reconstructed", used for filtering TYPE.
    :param average: what to calculate the average. Macro micro or weighted.
    :param strat_vars: List of column names to stratify by (e.g., ['DRUG']).
    :param static: If True, metrics for static variables will be calculated.
    :param per_time_mean: If True, returns mean error per time point (averaged across variables).
    :param per_variable_mean: If True, returns mean error per variable (averaged across time).
    :return: Dict of dataframes with keys:
             - 'full': full stratified error metrics
             - 'per_time': (optional) mean per time point
             - 'per_variable': (optional) mean per variable
             - 'overall': overall mean values
    """
    if average not in ["weighted", "macro", "micro"]:
        logger.info("`average` should be one of: 'weighted', 'macro', or 'micro'")
        raise AssertionError("Invalid value for `average`.")

    strat_vars = strat_vars or []
    if static:
        dt = (dt[(dt["REPI"] == 1 if mode == "Reconstructed" else True) &
                (dt['TYPE'].isin(["Observed", mode])) &
                (dt['Variable'].isin(rp0['static_cat'] + rp0['static_bin']))])
        TIME_v = []
    else:
        dt = (dt[(dt["REPI"] == 1 if mode == "Reconstructed" else True) &
            (dt['TYPE'].isin(["Observed", mode])) &
            (dt['Variable'].isin(rp0['long_cat'] + rp0['long_bin']))])
        TIME_v = ["TIME"]


    dt = dt.pivot_table(
        index=strat_vars + ["SUBJID", "Variable"] + TIME_v,
        columns="TYPE",
        values="DV").dropna(subset=["Observed", mode])

    dt["Observed"] = dt["Observed"].astype(int)
    dt[mode] = dt[mode].astype(int)
    variable_max_categories = dt.groupby('Variable')['Observed'].max().to_dict()
    records = []
    group_cols = strat_vars + TIME_v + ["Variable"]
    for group_keys, group_df in dt.groupby(group_cols):
        variable_name = group_keys[-1]  # last in group_cols

        max_cat = variable_max_categories.get(variable_name)
        if max_cat is None:
            logger.warning(f"Max categories not found for variable '{variable_name}', skipping group")
            continue

        try:
            y_true = group_df["Observed"]
            y_pred = group_df[mode]

            classes = list(range(max_cat + 1))

            # Sanity check for label variability
            if len(set(y_true)) <= 1 or len(set(y_pred)) <= 1:
                raise ValueError("Only one class present")

            record = dict(zip(group_cols, group_keys))
            record["F1"] = round(f1_score(y_true, y_pred, average=average, labels=classes, zero_division=0), 3)
            record["Accuracy"] = round(accuracy_score(y_true, y_pred), 3)
            record["Precision"] = round(precision_score(y_true, y_pred, average=average, labels=classes, zero_division=0), 3)
            record["Recall"] = round(recall_score(y_true, y_pred, average=average, labels=classes, zero_division=0), 3)
        except Exception as e:
            record = dict(zip(group_cols, group_keys))
            record["F1"] = np.nan
            record["Accuracy"] = np.nan
            record["Precision"] = np.nan
            record["Recall"] = np.nan
            logger.info(f"For group {group_keys}, metrics are set to NaN: {str(e)}")

        records.append(record)

    full_df = pd.DataFrame(records)
    result = {"full": full_df}

    # Per time point mean (across variables)
    if per_time_mean and not static:
        time_group = strat_vars + ["TIME"]
        per_time_df = full_df.groupby(time_group).agg({
            "F1": "mean", "Accuracy": "mean", "Precision": "mean", "Recall": "mean"
        }).reset_index()
        result["per_time"] = per_time_df

        if per_time_df[["F1", "Accuracy", "Precision", "Recall"]].isna().any().any():
            logger.info(
                "There are NaN values in per_time mean because some metrics could not be calculated. "
                "If for a given TIME, all variables have NaN metrics, the mean across variables will be NaN. "
                "If at least one variable has a non-NaN metric for a TIME, the mean will be non-NaN."
            )

    # Per variable mean (across time)
    if per_variable_mean:
        var_group = strat_vars + ["Variable"]
        per_variable_df = full_df.groupby(var_group).agg({
            "F1": "mean", "Accuracy": "mean", "Precision": "mean", "Recall": "mean"
        }).reset_index()
        result["per_variable"] = per_variable_df

        if per_variable_df[["F1", "Accuracy", "Precision", "Recall"]].isna().any().any():
            logger.info(
                "There are NaN values in per_variable mean because some metrics could not be calculated. "
                "If for a given Variable, all instances have NaN metrics, the mean across instances will be NaN. "
                "If at least one instance has a non-NaN metric for a Variable, the mean will be non-NaN."
            )

    # Overall
    if strat_vars:
        overall_df = full_df.groupby(strat_vars).agg({
            "F1": "mean", "Accuracy": "mean", "Precision": "mean", "Recall": "mean"
        }).reset_index()
    else:
        overall_df = pd.DataFrame([{
            "F1": full_df["F1"].mean(),
            "Accuracy": full_df["Accuracy"].mean(),
            "Precision": full_df["Precision"].mean(),
            "Recall": full_df["Recall"].mean()
        }])
    result["overall"] = overall_df
    for key in ["full", "per_time", "per_variable", "overall"]:
        if key in result:
            result[key][["F1", "Accuracy", "Precision", "Recall"]] = result[key][["F1", "Accuracy", "Precision", "Recall"]].round(3)

    return result
