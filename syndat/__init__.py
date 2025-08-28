from syndat import scores
from syndat import metrics
from syndat import visualization
from syndat import postprocessing
from .rct import preprocessing_tidy_format, visualization_clical_trials, metrics_rct

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
    discriminator_auc
)
from .rct.metrics_rct import (
    compute_continuous_error_metrics,
    compute_categorical_error_metrics
)

from syndat.rct.preprocessing_tidy_format import (
    convert_to_syndat_scores,
    get_rp,
    convert_long_data_to_tidy,
    convert_static_data_to_tidy,
    convert_data_to_tidy
)

from syndat.rct.visualization_clical_trials import (
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
    'compute_continuous_error_metrics',
    'compute_categorical_error_metrics',
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
