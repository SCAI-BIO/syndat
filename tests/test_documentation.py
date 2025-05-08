import unittest

import pandas as pd

from syndat import jensen_shannon_distance, normalized_correlation_difference, discriminator_auc


class TestDocumentation(unittest.TestCase):

    def test_metrics_readme(self):
        real = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'A', 'B', 'C']
        })

        synthetic = pd.DataFrame({
            'feature1': [1, 2, 2, 3, 3],
            'feature2': ['A', 'B', 'A', 'C', 'C']
        })

        jsd = jensen_shannon_distance(real, synthetic)
        print(jsd)
        norm_dist = normalized_correlation_difference(real, synthetic)
        print(norm_dist)
        auc = discriminator_auc(real, synthetic)
        print(auc)