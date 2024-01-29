import os
from unittest import TestCase

import pandas as pd

from syndat.quality import get_jsd


class Test(TestCase):

    TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

    real = pd.read_csv(os.path.join(TEST_DIR_PATH, "resources", 'real.csv'))
    synthetic = pd.read_csv(os.path.join(TEST_DIR_PATH, "resources", 'synthetic.csv'))

    def test_get_jsd_avg(self):
        jsd = get_jsd(self.real, self.synthetic)
        self.assertEqual(0.1602252096223203, jsd)