import os

import pandas as pd

import syndat

TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


real = pd.read_csv(os.path.join(TEST_DIR_PATH, "resources", "real.csv"))
synthetic = pd.read_csv(os.path.join(TEST_DIR_PATH, "resources", "synthetic.csv"))


def test_get_jsd_avg():
    syndat.quality.get_jsd(real=real, synthetic=synthetic)
    assert True
