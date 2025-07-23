import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch

from syndat import (
    plot_distributions,
    plot_correlations,
    plot_categorical_feature,
    plot_numerical_feature
)

class TestPlotFunctions(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.real = pd.DataFrame({
            "num_feature": np.random.normal(0, 1, 100),
            "cat_feature": np.random.choice(["A", "B", "C"], 100)
        })
        self.synthetic = pd.DataFrame({
            "num_feature": np.random.normal(0, 1, 100),
            "cat_feature": np.random.choice(["A", "B", "C"], 100)
        })

        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plot_distributions_creates_plots(self):
        plot_distributions(self.real, self.synthetic, store_destination=self.store_path)
        output_files = os.listdir(self.store_path)
        self.assertIn("num_feature.png", output_files)
        self.assertIn("cat_feature.png", output_files)

    def test_plot_correlations_creates_heatmaps(self):
        plot_correlations(self.real[["num_feature"]], self.synthetic[["num_feature"]], self.store_path)
        output_files = os.listdir(self.store_path)
        self.assertIn("real_corr.png", output_files)
        self.assertIn("syntehtic_corr.png", output_files)

    @patch("matplotlib.pyplot.show")
    def test_plot_categorical_feature_runs(self, mock_show):
        try:
            plot_categorical_feature("cat_feature", self.real, self.synthetic)
        except Exception as e:
            self.fail(f"plot_categorical_feature raised an exception: {e}")

    @patch("matplotlib.pyplot.show")
    def test_plot_numerical_feature_runs(self, mock_show):
        try:
            plot_numerical_feature("num_feature", self.real, self.synthetic)
        except Exception as e:
            self.fail(f"plot_numerical_feature raised an exception: {e}")
