"""Utility report skeleton for synthetic data evaluation.

This module provides the architecture for the utility report described in
issue #142. It is a skeleton only: the report classes are defined here, but
the actual utility computation is not implemented yet and the compute methods
raise ``NotImplementedError``.

The utility report is intended to be combined later with a ``FidelityReport``
into a ``SyndatReport``.
"""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictionTargetInference:
    """Prediction target inferred for a utility evaluation.

    :param target_col: Name of the target column the utility model predicts.
    :param target_justification: Justification for choosing this target column.
    """

    target_col: str
    target_justification: str


@dataclass
class UtilityReport:
    """Skeleton of a report on the utility of synthetic data.

    Utility is evaluated by training a model on real data and on synthetic
    data and comparing how well each model predicts the target column
    (see issue #139). This class only defines the report's structure; the
    computation itself is not implemented yet.

    :param model: The model used for the utility evaluation, e.g. a
        scikit-learn estimator.
    :param accuracy: Accuracy achieved by the model.
    :param target_col: Name of the target column the utility model predicts.
    :param target_justification: Justification for choosing this target column.
    """

    model: Any = None
    accuracy: float = float("nan")
    target_col: str = ""
    target_justification: str = ""

    def compute(
        self,
        inference: PredictionTargetInference,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> "UtilityReport":
        """Compute the utility report from real and synthetic data.

        :param inference: The prediction target inference describing the
            target column and its justification.
        :param real_data: DataFrame containing the real data.
        :param synthetic_data: DataFrame containing the synthetic data.
        :return: A populated ``UtilityReport``.
        """
        raise NotImplementedError("Utility computation is not implemented yet (see #142).")

    def to_dict(self) -> dict:
        """Return the report as a dictionary.

        :return: Dictionary with the report fields.
        """
        raise NotImplementedError("UtilityReport.to_dict is not implemented yet (see #142).")

    def to_dataframe(self) -> pd.DataFrame:
        """Return the report as a DataFrame.

        :return: DataFrame representation of the report.
        """
        raise NotImplementedError("UtilityReport.to_dataframe is not implemented yet (see #142).")
