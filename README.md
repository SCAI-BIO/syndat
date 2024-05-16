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

jsd = syndat.quality.jsd(real, synthetic)
auc = syndat.quality.auc(real, synthetic)
norm = syndat.quality.correlation(real, synthetic)
```

## Visualization

Visualize real vs. synthetic data distributions and summary statistics for each feature:

```python
import pandas as pd
import syndat

real = pd.read_csv("real.csv")
synthetic = pd.read_csv("synthetic.csv")

syndat.visualization.plot_distributions(real, synthetic, store_destination="results/plots")
syndat.visualization.plot_correlations(real, synthetic, store_destination="results/plots")
```

