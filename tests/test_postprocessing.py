import unittest
import pandas as pd
import numpy as np

from syndat import normalize_scale, assert_minmax, normalize_float_precision


class TestPostprocessing(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.real = pd.DataFrame({
            "float1": np.linspace(10.0, 20.0, 100),
            "float2": np.arange(0.0, 50.0, 0.5)
        })

        self.synthetic = pd.DataFrame({
            "float1": np.linspace(0.0, 1.0, 100),
            "float2": np.random.normal(25.0, 5.0, 100)
        })

    def test_normalize_scale_matches_range(self):
        scaled = normalize_scale(self.real, self.synthetic)
        for col in self.synthetic.columns:
            real_min, real_max = self.real[col].min(), self.real[col].max()
            scaled_min, scaled_max = scaled[col].min(), scaled[col].max()

            self.assertAlmostEqual(scaled_min, real_min, places=5)
            self.assertAlmostEqual(scaled_max, real_max, places=5)

    def test_assert_minmax_clip_within_range(self):
        clipped = assert_minmax(self.real, self.synthetic, method="clip")
        for col in clipped.columns:
            self.assertGreaterEqual(clipped[col].min(), self.real[col].min())
            self.assertLessEqual(clipped[col].max(), self.real[col].max())

    def test_normalize_float_precision_rounding(self):
        # Make float2 rounded to 0.5 steps in real
        real = pd.DataFrame({
            "float2": np.arange(0, 10, 0.5)
        })
        synthetic = pd.DataFrame({
            "float2": [0.1, 0.9, 1.4, 2.6, 3.7]
        })

        adjusted = normalize_float_precision(real, synthetic)
        expected = pd.Series([0.0, 1.0, 1.5, 2.5, 3.5], name="float2")

        pd.testing.assert_series_equal(adjusted["float2"], expected)


if __name__ == "__main__":
    unittest.main()
