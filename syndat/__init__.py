from syndat import scores
from syndat import metrics
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
    discrimination,
    distribution,
    correlation,
)

from .postprocessing import (
    normalize_scale,
    normalize_float_precision,
    assert_minmax,
)

from .metrics import (
    jensen_shannon_distance,
    correlation_quotient,
    discriminator_auc,
)

__all__ = [
    # visualization
    'plot_distributions',
    'plot_correlations',
    'plot_shap_discrimination',
    'plot_categorical_feature',
    'plot_numerical_feature',
    # metrics
    'jensen_shannon_distance',
    'correlation_quotient',
    'discriminator_auc',
    # scores
    'discrimination',
    'distribution',
    'correlation',
    # postprocessing
    'normalize_scale',
    'normalize_float_precision',
    'assert_minmax',
]
