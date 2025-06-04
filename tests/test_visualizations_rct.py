import unittest
import pandas as pd
import numpy as np
import random
import os
from syndat import *

def generate_mock_rp_and_df(n_patients=10, n_reps=5, times=[0.0, 6.0, 12.0, 18.0]):
    rp = {
        'Tmax': np.float64(max(times)),
        'static_vnames': [f'VarStat{i}' for i in range(1, 5)],
        'static_cat': [f'VarStat{i}' for i in [1, 3]],
        'static_cont': [f'VarStat{i}' for i in [2, 4]],
        'long_vnames': [f'VarLong{i}' for i in range(1, 6)],
        'long_cat': [f'VarLong{i}' for i in [1, 3, 4]],
        'long_bin': [f'VarLong{i}' for i in [3, 4]],
        'long_cont': [f'VarLong{i}' for i in [2, 5]],
    }

    rows = []
    for subj in range(1, n_patients + 1):
        drug = f'DRUG {random.choice([0, 1])}'
        for repi in range(1, n_reps + 1):
            for var in rp['long_vnames']:
                for t in times:
                    dv_value = (
                        np.random.randint(0, 2) if var in rp['long_bin']
                        else np.random.normal(50, 10) if var in rp['long_cont']
                        else np.random.randint(0, 5)
                    )
                    rows.append({
                        'SUBJID': subj,
                        'REPI': repi,
                        'TIME': t,
                        'DRUG': drug,
                        'DV': dv_value,
                        'TYPE': random.choice(['Observed', 'Reconstructed']),
                        'Variable': var,
                    })

    df = pd.DataFrame(rows)
    return rp, df


class TestPlotsRCT(unittest.TestCase):
    def setUp(self):
        self.rp, self.df = generate_mock_rp_and_df()
        self.save_path = "./examples/"
        self.strat_vars=["DRUG"]

    def test_gof_continuous_list(self):
        gof_continuous_list(self.rp, self.df, strat_vars=["DRUG"], save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('gof_plot.png')]
        self.assertTrue(len(png_files) > 0, "GOF plot files were not created.")

    def test_gof_binary_list(self):
        gof_binary_list(self.rp, self.df, strat_vars=["DRUG"], save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('gof_bin_plot.png')]
        self.assertTrue(len(png_files) > 0, "Binary GOF plot files were not created.")

    def test_gof_categorical_list(self):
        gof_categorical_list(self.rp, self.df, strat_vars=["DRUG"], save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('gof_cat_perc_plot.png')]
        self.assertTrue(len(png_files) > 0, "Categorical GOF plot files were not created.")

    def test_gof_categorical_list2(self):
        gof_categorical_list(self.rp, self.df, strat_vars=["DRUG"], type_="Subjects", save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('gof_cat_subj_plot.png')]
        self.assertTrue(len(png_files) > 0, "Categorical GOF plot files were not created.")

    def test_trajectory_plot_list(self):
        trajectory_plot_list(self.rp, self.df, strat_vars=["DRUG"], save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('trajectory_plot.png')]
        self.assertTrue(len(png_files) > 0, "Trajectory plot files were not created.")

    def test_raincloud_continuous_list(self):
        raincloud_continuous_list(self.rp, self.df, strat_vars=["DRUG"], type="longitudinal", save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('raincloud_plot.png')]
        self.assertTrue(len(png_files) > 0, "Raincloud plot files were not created.")

    def tearDown(self):
        if os.path.exists(self.save_path):
            for f in os.listdir(self.save_path):
                if f.endswith('.png'):
                    os.remove(os.path.join(self.save_path, f))