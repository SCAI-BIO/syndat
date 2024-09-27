# Syndat
![tests](https://github.com/SCAI-BIO/syndat/actions/workflows/tests.yaml/badge.svg) ![docs](https://readthedocs.org/projects/syndat/badge/?version=latest&style=flat) ![version](https://img.shields.io/github/v/release/SCAI-BIO/syndat)

Syndat is a software package that provides basic functionalities for the evaluation and visualizsation of synthetic data. Quality scores can be computed on 3 base metrics (Discrimation, Correlation and Distribution) and data may be visualized to inspect correlation structures or statistical distribution plots.

# Installation

Install via pip:

```bash
pip install syndat
```

# Usage

## Quality metrics

Compute data quality metrics by comparing real and synthetic data in terms of their separation complexity, 
distribution similarity or pairwise feature correlations:

```python
import pandas as pd
import syndat

real = pd.read_csv("real.csv")
synthetic = pd.read_csv("synthetic.csv")

# How similar are the statistical distributions of real and synthetic features 
distribution_similarity_score = syndat.scores.distribution(real, synthetic)

# How hard is it for a classifier to discriminate real and synthetic data
discrimination_score = syndat.scores.discrimination(real, synthetic)

# How well are pairwise feature correlations preserved
correlation_score = syndat.scores.correlation(real, synthetic)
```

Scores are defined in a range of 0-100, with a higher score corresponding to better data fidelity.

## Visualization

Visualize real vs. synthetic data distributions, summary statistics and discriminating features:

```python
import pandas as pd
import syndat

real = pd.read_csv("real.csv")
synthetic = pd.read_csv("synthetic.csv")

# plot *all* feature distribution and store image files
syndat.visualization.plot_distributions(real, synthetic, store_destination="results/plots")
syndat.visualization.plot_correlations(real, synthetic, store_destination="results/plots")

# plot and display specific feature distribution plot
syndat.visualization.plot_numerical_feature("feature_xy", real, synthetic)
syndat.visualization.plot_numerical_feature("feature_xy", real, synthetic)

# plot a shap plot of differentiating feature for real and synthetic data
syndat.visualization.plot_shap_discrimination(real, synthetic)
```


## Postprocessing

Postprocess synthetic data to improve data fidelity:

```python
import pandas as pd
import syndat

real = pd.read_csv("real.csv")
synthetic = pd.read_csv("synthetic.csv")

# postprocess synthetic data
synthetic_post = syndat.postprocessing.assert_minmax(real, synthetic)
synthetic_post = syndat.postprocessing.normalize_float_precision(real, synthetic)
```

