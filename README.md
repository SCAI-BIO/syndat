# Syndat
Synthetic data quality evaluation &amp; visualization 

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
from syndat import quality

real = pd.read_csv("real.csv")
synthetic = pd.read_csv("synthetic.csv")

jsd = quality.get_jsd(real, synthetic)
auc = quality.get_auc(real, synthetic)
norm = quality.get_correlation_quotient(real, synthetic)
```

## Visualization

Visualize real vs. synthetic data distributions and summary statistics for each feature:

```python
import pandas as pd
from syndat import visualization

real = pd.read_csv("real.csv")
synthetic = pd.read_csv("synthetic.csv")

visualization.plot_distributions(real, synthetic, store_destination="results/plots")
```

