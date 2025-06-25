Quickstart
==========

Installation
------------

Install via pip:

.. code-block:: bash

   pip install syndat

Basic Usage
-----------

.. code-block:: python

    import pandas as pd
    from syndat.metrics import (
        jensen_shannon_distance,
        normalized_correlation_difference,
        discriminator_auc
    )

    real = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'A', 'B', 'C']
    })

    synthetic = pd.DataFrame({
        'feature1': [1, 2, 2, 3, 3],
        'feature2': ['A', 'B', 'A', 'C', 'C']
    })

    print(jensen_shannon_distance(real, synthetic))
    >> {'feature1': 0.4990215421876156, 'feature2': 0.22141025172133794}

    print(normalized_correlation_difference(real, synthetic))
    >> 0.24571345029108108

    print(discriminator_auc(real, synthetic))
    >> 0.6

    # JSD score is being aggregated over all features
    distribution_similarity_score = syndat.scores.distribution(real, synthetic)
    discrimination_score = syndat.scores.discrimination(real, synthetic)
    correlation_score = syndat.scores.correlation(real, synthetic)

    # plot *all* feature distribution and store image files
    syndat.visualization.plot_distributions(real, synthetic, store_destination="results/plots")
    syndat.visualization.plot_correlations(real, synthetic, store_destination="results/plots")

    # plot and display specific feature distribution plot
    syndat.visualization.plot_numerical_feature("feature_xy", real, synthetic)
    syndat.visualization.plot_numerical_feature("feature_xy", real, synthetic)

    # plot a shap plot of differentiating feature for real and synthetic data
    syndat.visualization.plot_shap_discrimination(real, synthetic)


    # postprocess synthetic data
    synthetic_post = syndat.postprocessing.assert_minmax(real, synthetic)
    synthetic_post = syndat.postprocessing.normalize_float_precision(real, synthetic)

Modules
-------

- ``syndat.metrics``: Metrics for the evaluation of synthetic data fidelity.
- ``syndat.scores``: Scoring functions that normalize metrics for easier comparison.
- ``syndat.visualization``: Visualize feature distributions correlation and SHAP analysis.
- ``syndat.postprocessing``: Optional cleaning and formatting helpers.

Examples
--------

See the `examples/` folder on GitHub for end-to-end demos:
https://github.com/SCAI-BIO/syndat/tree/main/examples
