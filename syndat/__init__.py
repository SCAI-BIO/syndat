from syndat import domain
from syndat import scores
from syndat import visualization
from syndat import postprocessing

from .visualization import (
    plot_distributions,
    plot_correlations,
    plot_shap_discrimination,
    plot_categorical_feature,
    plot_numerical_feature
)

from .scores import (
    auc,
    jsd,
    discrimination,
    distribution,
    correlation,
)

from .postprocessing import (
    normalize_scale,
    normalize_float_precision,
    assert_minmax,
) 

__all__ = [
    # visualization
    'plot_distributions',
    'plot_correlations',
    'plot_shap_discrimination',
    'plot_categorical_feature',
    'plot_numerical_feature',
    # scores
    'auc',
    'jsd',
    'discrimination',
    'distribution',
    'correlation',
    # postprocessing
    'normalize_scale',
    'normalize_float_precision',
    'assert_minmax',
]
