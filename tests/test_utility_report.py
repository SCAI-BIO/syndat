"""Tests for the utility report skeleton (issue #142).

The utility report is a skeleton only: the classes exist with the fields
described in #142, but the computation methods raise ``NotImplementedError``
until the utility evaluation is implemented.
"""

from dataclasses import fields

import pandas as pd
import pytest

from syndat.utility import PredictionTargetInference, UtilityReport


def test_utility_report_imports():
    """The skeleton module can be imported."""
    assert UtilityReport is not None
    assert PredictionTargetInference is not None


def test_prediction_target_inference_fields():
    """PredictionTargetInference carries the target column and justification."""
    inference = PredictionTargetInference(
        target_col="outcome", target_justification="Primary trial endpoint."
    )
    assert inference.target_col == "outcome"
    assert inference.target_justification == "Primary trial endpoint."


def test_utility_report_fields():
    """UtilityReport has exactly the fields described in #142."""
    field_names = [f.name for f in fields(UtilityReport)]
    assert field_names == ["model", "accuracy", "target_col", "target_justification"]


def test_utility_report_stores_values():
    """UtilityReport stores the model, accuracy and target description."""
    report = UtilityReport(
        model=None,
        accuracy=0.85,
        target_col="outcome",
        target_justification="Primary trial endpoint.",
    )
    assert report.model is None
    assert report.accuracy == 0.85
    assert report.target_col == "outcome"
    assert report.target_justification == "Primary trial endpoint."


def test_utility_report_compute_raises_not_implemented():
    """compute() is part of the skeleton only and raises NotImplementedError."""
    inference = PredictionTargetInference(
        target_col="outcome", target_justification="Primary trial endpoint."
    )
    real_data = pd.DataFrame({"outcome": [0, 1]})
    synthetic_data = pd.DataFrame({"outcome": [1, 0]})
    report = UtilityReport()
    with pytest.raises(NotImplementedError):
        report.compute(inference, real_data, synthetic_data)


def test_utility_report_to_dict_raises_not_implemented():
    """to_dict() is part of the skeleton only and raises NotImplementedError."""
    report = UtilityReport()
    with pytest.raises(NotImplementedError):
        report.to_dict()


def test_utility_report_to_dataframe_raises_not_implemented():
    """to_dataframe() is part of the skeleton only and raises NotImplementedError."""
    report = UtilityReport()
    with pytest.raises(NotImplementedError):
        report.to_dataframe()
