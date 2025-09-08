import pandas as pd

import syndat
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

# METRICS
jsd = jensen_shannon_distance(real, synthetic)
norm = normalized_correlation_difference(real, synthetic)
auc = discriminator_auc(real, synthetic)

print(jsd)
print(norm)
print(auc)

# SCORES
distribution_similarity_score = syndat.scores.distribution(real, synthetic)
discrimination_score = syndat.scores.discrimination(real, synthetic)
correlation_score = syndat.scores.correlation(real, synthetic)

print(distribution_similarity_score)
print(discrimination_score)
print(correlation_score)

# VISUALIZATION
syndat.visualization.plot_numerical_feature("feature1", real, synthetic)
syndat.visualization.plot_categorical_feature("feature2", real, synthetic)