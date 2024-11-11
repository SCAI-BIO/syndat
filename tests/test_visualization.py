import unittest
import pandas as pd
import numpy as np
import os

from syndat import plot_shap_discrimination


class TestPlotShapDiscrimination(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.real = pd.DataFrame(np.random.normal(size=(100, 5)), columns=[f"feature_{i}" for i in range(5)])
        self.synthetic = pd.DataFrame(np.random.normal(size=(100, 5)), columns=[f"feature_{i}" for i in range(5)])

        # Define the path where the plot will be temporarily saved
        self.save_path = "test_shap_plot.png"

    def test_plot_shap_discrimination(self):
        # Call the function with test data and save_path
        plot_shap_discrimination(self.real, self.synthetic, save_path=self.save_path)

        # Check if the plot file was created
        self.assertTrue(os.path.exists(self.save_path), "SHAP plot file was not created.")

    def tearDown(self):
        # Remove the file if it exists after the test
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

