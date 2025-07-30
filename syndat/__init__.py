from syndat import scores
from syndat import metrics
from syndat import visualization
from syndat import postprocessing
from syndat import preprocessing_tidy_format
from syndat import visualization_clical_trials

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
    normalized_correlation_difference,
    discriminator_auc,
)

from .preprocessing_tidy_format import (
    convert_to_syndat_scores,
    get_rp,
    convert_long_data_to_tidy,
    convert_static_data_to_tidy,
    convert_data_to_tidy
)

from .visualization_clical_trials import (
    gof_continuous,
    gof_continuous_list,
    gof_binary_list,
    bar_categorical,
    bar_categorical_list,
    bin_traj_time_list,
    assign_visit_absolute,
    trajectory_plot,
    trajectory_plot_list,
    raincloud_plot,
    raincloud_continuous_list
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
    'normalized_correlation_difference',
    'discriminator_auc',
    # scores
    'discrimination',
    'distribution',
    'correlation',
    # postprocessing
    'normalize_scale',
    'normalize_float_precision',
    'assert_minmax',
    # Preprocessing tidy format
    'convert_to_syndat_scores',
    'get_rp',
    'convert_long_data_to_tidy',
    'convert_static_data_to_tidy',
    'convert_data_to_tidy',
    # visualization clinical trials
    'gof_continuous',
    'gof_continuous_list',
    'gof_binary_list',
    'bar_categorical',
    'bar_categorical_list',
    'bin_traj_time_list',
    'assign_visit_absolute',
    'trajectory_plot',
    'trajectory_plot_list',
    'raincloud_plot',
    'raincloud_continuous_list'
]
