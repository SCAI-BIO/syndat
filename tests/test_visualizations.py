import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import syndat.visualization as visualization_module
from syndat import (
    plot_categorical_feature,
    plot_correlations,
    plot_distributions,
    plot_numerical_feature,
)


class TestPlotFunctions(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.real = pd.DataFrame(
            {
                "num_feature": np.random.normal(0, 1, 100),
                "cat_feature": np.random.choice(["A", "B", "C"], 100),
            }
        )
        self.synthetic = pd.DataFrame(
            {
                "num_feature": np.random.normal(0, 1, 100),
                "cat_feature": np.random.choice(["A", "B", "C"], 100),
            }
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plot_distributions_creates_plots(self):
        plot_distributions(self.real, self.synthetic, store_destination=self.store_path)
        output_files = os.listdir(self.store_path)
        self.assertIn("num_feature.png", output_files)
        self.assertIn("cat_feature.png", output_files)

    def test_plot_distributions_relative_frequency_creates_plots(self):
        synthetic_larger = pd.concat(
            [self.synthetic, self.synthetic.iloc[:20]], ignore_index=True
        )
        plot_distributions(
            self.real,
            synthetic_larger,
            store_destination=self.store_path,
            categorical_y_scale="relative",
        )
        output_files = os.listdir(self.store_path)
        self.assertIn("num_feature.png", output_files)
        self.assertIn("cat_feature.png", output_files)

    @patch("syndat.visualization.sns.barplot")
    def test_plot_distributions_auto_relative_frequency_when_size_differs(
        self, mock_barplot
    ):
        synthetic_larger = pd.concat(
            [self.synthetic, self.synthetic.iloc[:20]], ignore_index=True
        )
        plot_distributions(
            self.real, synthetic_larger, store_destination=self.store_path
        )
        self.assertTrue(
            any(
                call.kwargs.get("y") == "frequency"
                for call in mock_barplot.call_args_list
            )
        )

    @patch("syndat.visualization.sns.barplot")
    def test_plot_distributions_auto_relative_frequency_not_triggered_when_size_close(
        self, mock_barplot
    ):
        synthetic_close = self.synthetic.iloc[:100].reset_index(drop=True)
        plot_distributions(
            self.real, synthetic_close, store_destination=self.store_path
        )
        mock_barplot.assert_not_called()

    def test_plot_correlations_creates_heatmaps(self):
        plot_correlations(
            self.real[["num_feature"]], self.synthetic[["num_feature"]], self.store_path
        )
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
    def test_plot_categorical_feature_relative_frequency(self, mock_show):
        real = pd.DataFrame({"cat_feature": ["A"] * 8 + ["B"] * 2})
        synthetic = pd.DataFrame({"cat_feature": ["A"] * 2 + ["B"] * 2 + ["C"] * 6})

        plot_categorical_feature("cat_feature", real, synthetic, y_scale="relative")
        axes = plt.gcf().axes
        self.assertEqual(axes[0].get_ylabel(), "Relative Frequency (%)")
        self.assertEqual(axes[1].get_ylabel(), "Relative Frequency (%)")
        self.assertAlmostEqual(
            sum(patch.get_height() for patch in axes[0].patches), 100.0, places=5
        )
        self.assertAlmostEqual(
            sum(patch.get_height() for patch in axes[1].patches), 100.0, places=5
        )
        plt.close(plt.gcf())

    def test_plot_distributions_invalid_categorical_y_scale_raises(self):
        with self.assertRaises(ValueError):
            plot_distributions(
                self.real,
                self.synthetic,
                store_destination=self.store_path,
                categorical_y_scale="invalid",
            )

    @patch("matplotlib.pyplot.show")
    def test_plot_categorical_feature_invalid_y_scale_raises(self, mock_show):
        with self.assertRaises(ValueError):
            plot_categorical_feature(
                "cat_feature", self.real, self.synthetic, y_scale="invalid"
            )

    @patch("matplotlib.pyplot.show")
    def test_plot_categorical_feature_auto_relative_frequency_when_size_differs(
        self, mock_show
    ):
        real = pd.DataFrame({"cat_feature": ["A"] * 8 + ["B"] * 2})
        synthetic = pd.DataFrame({"cat_feature": ["A"] * 2 + ["B"] * 2 + ["C"] * 8})

        plot_categorical_feature("cat_feature", real, synthetic)
        axes = plt.gcf().axes
        self.assertEqual(axes[0].get_ylabel(), "Relative Frequency (%)")
        self.assertEqual(axes[1].get_ylabel(), "Relative Frequency (%)")
        plt.close(plt.gcf())

    @patch("matplotlib.pyplot.show")
    def test_plot_categorical_feature_auto_count_when_size_close(self, mock_show):
        real = pd.DataFrame({"cat_feature": ["A"] * 50 + ["B"] * 50})
        synthetic = pd.DataFrame({"cat_feature": ["A"] * 48 + ["B"] * 48})

        plot_categorical_feature("cat_feature", real, synthetic)
        axes = plt.gcf().axes
        self.assertEqual(axes[0].get_ylabel(), "Relative Frequency (%)")
        self.assertEqual(axes[1].get_ylabel(), "Relative Frequency (%)")
        plt.close(plt.gcf())

    @patch("matplotlib.pyplot.show")
    def test_plot_numerical_feature_runs(self, mock_show):
        try:
            plot_numerical_feature("num_feature", self.real, self.synthetic)
        except Exception as e:
            self.fail(f"plot_numerical_feature raised an exception: {e}")
