from syndat import scores
from syndat import metrics
from syndat import core
from syndat import visualization
from syndat import postprocessing

from .visualization import (
    plot_distributions,
    plot_correlations,
    plot_shap_discrimination,
    plot_categorical_feature,
    plot_numerical_feature
)

from .postprocessing import (
    normalize_scale,
    normalize_float_precision,
    assert_minmax,
)

from .scores import (
    discrimination_score,
    distribution_score,
    correlation_score,
)

from .metrics import (
    discriminator_auc,
    jensen_shannon_divergence,
    normalized_correlation_quotient
)

from .core import (
    Evaluator
)

__all__ = [
    # visualization
    'plot_distributions',
    'plot_correlations',
    'plot_shap_discrimination',
    'plot_categorical_feature',
    'plot_numerical_feature',
    # metrics
    'discriminator_auc',
    'jensen_shannon_divergence',
    'normalized_correlation_quotient',
    # core
    'Evaluator',
    # scores
    'discrimination_score',
    'distribution_score',
    'correlation_score',
    # postprocessing
    'normalize_scale',
    'normalize_float_precision',
    'assert_minmax',
]
