# Syndat
[![DOI](https://zenodo.org/badge/734391183.svg)](https://doi.org/10.5281/zenodo.15791976)
![tests](https://github.com/SCAI-BIO/syndat/actions/workflows/tests.yaml/badge.svg) [![codecov](https://codecov.io/gh/SCAI-BIO/syndat/branch/main/graph/badge.svg)](https://codecov.io/gh/SCAI-BIO/syndat) ![docs](https://readthedocs.org/projects/syndat/badge/?version=latest&style=flat) ![version](https://img.shields.io/pypi/v/syndat)

Syndat is a software package that provides basic functionalities for the evaluation and visualisation of synthetic data. Quality scores can be computed on 3 base metrics (Discrimation, Correlation and Distribution) and data may be visualized to inspect correlation structures or statistical distribution plots.

Syndat also allows users to generate stratified and interpretable visualisations, including raincloud plots, GOF plots, and trajectory comparisons, offering deeper insights into the quality of synthetic clinical data across different subgroups.

# Installation

Install via pip:

```bash
pip install syndat
```

# Usage

## Fidelity metrics

### Jenson-Shannon Distance

The Jenson-Shannon distance is a measure of similarity between two probability distributions. In our case, we compute
probability distributions for each feature in the datasets and compute and can thus compare the statistic feature 
similarity of two dataframes. 

It is bounded between 0 and 1, with 0 indicating identical distributions. 

### (Normalized) Correlation Difference

In addition to statistical similarity between the same features, we also want to make sure to preserve the correlations
across different features. The normalized correlation difference measures the similarity of the correlation matrix of 
two dataframes.

A low correlation difference near zero indicates that the correlation structure of the synthetic data is similar to the 
real data.

### Discriminator AUC

A classifier is trained to discriminate between real and synthetic data. Based on the Receiver Operating Characteristic 
(ROC) curve, we compute the area under the curve (AUC) as a measure of how well the classifier can distinguish between 
the two datasets. 

An AUC of 0.5 indicates that the classifier is unable to distinguish between the two datasets, while an AUC of 1.0 
indicates perfect discrimination.

Exemplary usage:

```python
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
```

### Scoring Functions

For convenience and easier interpretation, a normalized score can be computed for each of the 
metrics instead:

```python
# JSD score is being aggregated over all features
distribution_similarity_score = syndat.scores.distribution(real, synthetic)
discrimination_score = syndat.scores.discrimination(real, synthetic)
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

# Evaluation and Visualization of Synthetic Clinical Trial Data

An example demonstrating how to compute distribution, discrimination, and correlation scores, as well as how to generate stratified visualizations (gof, raincloud and other plots), is available in `examples/rct_example.py`.
