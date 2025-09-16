import unittest
import pandas as pd
import numpy as np
import random
import os
from syndat import *
from syndat import compute_continuous_error_metrics, compute_categorical_error_metrics


def generate_mock_rp_and_df(n_patients=10, n_reps=5, times=[0.0, 6.0, 12.0, 18.0]):
    rp = {
        'Tmax': np.float64(max(times)),
        'static_vnames': [f'VarStat{i}' for i in range(1, 5)],
        'static_cat': [f'VarStat{i}' for i in [1, 3]],
        'static_bin': [f'VarStat{i}' for i in [3]],
        'static_cont': [f'VarStat{i}' for i in [2, 4]],
        'long_vnames': [f'VarLong{i}' for i in range(1, 6)],
        'long_cat': [f'VarLong{i}' for i in [1, 3, 4]],
        'long_bin': [f'VarLong{i}' for i in [3, 4]],
        'long_cont': [f'VarLong{i}' for i in [2, 5]],
    }

    long_rows = []
    static_rows = []
    for subj in range(1, n_patients + 1):
        drug = f'DRUG {random.choice([0, 1])}'
        for repi in range(1, n_reps + 1):
            for var in rp['static_vnames']:
                for type_ in ['Observed', 'Reconstructed']:
                    dv_value = (
                                np.random.randint(0, 2) if var in rp['static_bin']
                                else np.random.normal(50, 10) if var in rp['static_cont']
                                else np.random.randint(0, 5)
                            )
                    static_rows.append({
                        'SUBJID': subj,
                        'REPI': repi,
                        'DRUG': drug,
                        'DV': dv_value,
                        'TYPE': type_,
                        'Variable': var,
                    })

            for var in rp['long_vnames']:
                for t in times:
                    for type_ in ['Observed', 'Reconstructed']:
                        dv_value = (
                            np.random.randint(0, 2) if var in rp['long_bin']
                            else np.random.normal(50, 10) if var in rp['long_cont']
                            else np.random.randint(0, 5)
                        )
                        long_rows.append({
                            'SUBJID': subj,
                            'REPI': repi,
                            'TIME': t,
                            'DRUG': drug,
                            'DV': dv_value,
                            'TYPE': type_,
                            'Variable': var,
                        })

    df_static = pd.DataFrame(static_rows)
    df_long = pd.DataFrame(long_rows)

    return rp, df_static, df_long


class TestPlotsRCT(unittest.TestCase):
    def setUp(self):
        self.rp, self.sdf, self.df = generate_mock_rp_and_df()
        self.save_path = "./examples/"
        self.strat_vars=["DRUG"]

        self.pbo = self.df[self.df.DRUG == 0]
        self.dt_cs = self.df[self.df.DRUG == 1]
        self.dt_cs["DRUG"] = 0        

    def test_exceptions_pre_processing(self):
        with self.assertRaises(AssertionError):
            compute_categorical_error_metrics(
                self.rp, self.df, strat_vars=self.strat_vars, average="WEIGHTED"
            )

    def test_exceptions_cat(self):
        with self.assertRaises(AssertionError):
            compute_categorical_error_metrics(
                self.rp, self.df, strat_vars=self.strat_vars, average="WEIGHTED"
            )

    def test_cat_error_metrics(self):
        result = compute_categorical_error_metrics(
            self.rp, self.df, strat_vars=self.strat_vars, per_time_mean=True, per_variable_mean=True)
        expected_keys = {"full", "per_time", "per_variable", "overall"}
        self.assertEqual(set(result.keys()), expected_keys)
        # Check that each value is a non-empty DataFrame
        for key in expected_keys:
            self.assertIsInstance(result[key], pd.DataFrame, f"{key} is not a DataFrame")
            self.assertFalse(result[key].empty, f"{key} DataFrame is empty")

        result = compute_categorical_error_metrics(
            self.rp, self.df, per_time_mean=True, per_variable_mean=True)
        expected_keys = {"full", "per_time", "per_variable", "overall"}
        self.assertEqual(set(result.keys()), expected_keys)
        # Check that each value is a non-empty DataFrame
        for key in expected_keys:
            self.assertIsInstance(result[key], pd.DataFrame, f"{key} is not a DataFrame")
            self.assertFalse(result[key].empty, f"{key} DataFrame is empty")

        result = compute_categorical_error_metrics(
            self.rp, self.sdf, strat_vars=self.strat_vars, static=True, per_variable_mean=True)
        expected_keys = {"full", "per_variable", "overall"}
        self.assertEqual(set(result.keys()), expected_keys)
        # Check that each value is a non-empty DataFrame
        for key in expected_keys:
            self.assertIsInstance(result[key], pd.DataFrame, f"{key} is not a DataFrame")
            self.assertFalse(result[key].empty, f"{key} DataFrame is empty")

        result = compute_categorical_error_metrics(
            self.rp, self.sdf, static=True, per_variable_mean=True)
        expected_keys = {"full", "per_variable", "overall"}
        self.assertEqual(set(result.keys()), expected_keys)
        # Check that each value is a non-empty DataFrame
        for key in expected_keys:
            self.assertIsInstance(result[key], pd.DataFrame, f"{key} is not a DataFrame")
            self.assertFalse(result[key].empty, f"{key} DataFrame is empty")

    def test_cont_error_metrcis(self):
        result = compute_continuous_error_metrics(
            self.rp, self.df, strat_vars=self.strat_vars, per_time_mean=True, per_variable_mean=True)
        expected_keys = {"full", "per_time", "per_variable", "overall"}
        self.assertEqual(set(result.keys()), expected_keys)
        # Check that each value is a non-empty DataFrame
        for key in expected_keys:
            self.assertIsInstance(result[key], pd.DataFrame, f"{key} is not a DataFrame")
            self.assertFalse(result[key].empty, f"{key} DataFrame is empty")

        result = compute_continuous_error_metrics(
            self.rp, self.df, per_time_mean=True, per_variable_mean=True)
        expected_keys = {"full", "per_time", "per_variable", "overall"}
        self.assertEqual(set(result.keys()), expected_keys)
        # Check that each value is a non-empty DataFrame
        for key in expected_keys:
            self.assertIsInstance(result[key], pd.DataFrame, f"{key} is not a DataFrame")
            self.assertFalse(result[key].empty, f"{key} DataFrame is empty")

        result = compute_continuous_error_metrics(
            self.rp, self.sdf, strat_vars=self.strat_vars, static=True, per_variable_mean=True)
        expected_keys = {"full", "per_variable", "overall"}
        self.assertEqual(set(result.keys()), expected_keys)
        # Check that each value is a non-empty DataFrame
        for key in expected_keys:
            self.assertIsInstance(result[key], pd.DataFrame, f"{key} is not a DataFrame")
            self.assertFalse(result[key].empty, f"{key} DataFrame is empty")

        result = compute_continuous_error_metrics(
            self.rp, self.sdf, static=True, per_variable_mean=True)
        expected_keys = {"full", "per_variable", "overall"}
        self.assertEqual(set(result.keys()), expected_keys)
        # Check that each value is a non-empty DataFrame
        for key in expected_keys:
            self.assertIsInstance(result[key], pd.DataFrame, f"{key} is not a DataFrame")
            self.assertFalse(result[key].empty, f"{key} DataFrame is empty")

    def test_gof_continuous_list(self):
        gof_continuous_list(self.rp, self.df, strat_vars=["DRUG"], save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('gof_plot.png')]
        self.assertTrue(len(png_files) > 0, "GOF plot files were not created.")

        gof_continuous_list(self.rp, self.sdf, strat_vars=["DRUG"], static=True, save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('gof_plot.png')]
        self.assertTrue(len(png_files) > 0, "GOF plot files were not created.")

    def test_log_gof_continuous_list(self):
        gof_continuous_list(self.rp, self.df, strat_vars=["DRUG"], log_trans=True, save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('Loggof_plot.png')]
        self.assertTrue(len(png_files) > 0, "GOF plot files were not created.")

        gof_continuous_list(self.rp, self.sdf, strat_vars=["DRUG"], static=True, log_trans=True, save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('Loggof_plot.png')]
        self.assertTrue(len(png_files) > 0, "GOF plot files were not created.")

    def test_gof_binary_list(self):
        gof_binary_list(self.rp, self.df, strat_vars=["DRUG"], save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('gof_bin_plot.png')]
        self.assertTrue(len(png_files) > 0, "Binary GOF plot files were not created.")

        gof_binary_list(self.rp, self.sdf, strat_vars=["DRUG"], static=True, save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('gof_bin_plot.png')]
        self.assertTrue(len(png_files) > 0, "Binary GOF plot files were not created.")

    def test_bar_time_binary_list(self):
        bin_traj_time_list(self.rp, self.df, strat_vars=["DRUG"], save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('bin_time_plot.png')]
        self.assertTrue(len(png_files) > 0, "Binary time plot files were not created.")

        bin_traj_time_list(self.rp, self.pbo, dt_cs=self.dt_cs, save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('bin_time_plot.png')]
        self.assertTrue(len(png_files) > 0, "Counterfactuak binary time plot files were not created.")

    def test_gof_categorical_list(self):

        with self.assertRaises(AssertionError):
            bar_categorical_list(self.rp, self.df, type_='all')

        with self.assertRaises(AssertionError):
            bar_categorical_list(self.rp, self.df, type_='Subjects', dt_cs=self.df)

        bar_categorical_list(self.rp, self.pbo, dt_cs=self.dt_cs, save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('bar_cat_perc_plot.png')]
        self.assertTrue(len(png_files) > 0, "Counterfactual Categorical time plot files were not created.")

        bar_categorical_list(self.rp, self.df, strat_vars=["DRUG"], save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('bar_cat_perc_plot.png')]
        self.assertTrue(len(png_files) > 0, "Categorical plot files were not created.")

        bar_categorical_list(self.rp, self.df, strat_vars=["DRUG","TIME"], save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('bar_cat_perc_plot.png')]
        self.assertTrue(len(png_files) > 0, "Categorical plot files were not created.")

        bar_categorical_list(self.rp, self.df, save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('bar_cat_perc_plot.png')]
        self.assertTrue(len(png_files) > 0, "Categorical plot files were not created.")

        bar_categorical_list(self.rp, self.sdf, strat_vars=["DRUG"], static=True, save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('bar_cat_perc_plot.png')]
        self.assertTrue(len(png_files) > 0, "Categorical plot files were not created.")

        bar_categorical_list(self.rp, self.sdf, static=True, save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('bar_cat_perc_plot.png')]
        self.assertTrue(len(png_files) > 0, "Categorical plot files were not created.")

    def test_gof_categorical_list2(self):
        bar_categorical_list(self.rp, self.df, type_="Subjects", strat_vars=["DRUG","TIME"], save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('bar_cat_subj_plot.png')]
        self.assertTrue(len(png_files) > 0, "Categorical plot files were not created.")

        bar_categorical_list(self.rp, self.sdf, type_="Subjects", strat_vars=["DRUG"], static=True, save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('bar_cat_subj_plot.png')]
        self.assertTrue(len(png_files) > 0, "Categorical plot files were not created.")

    def test_trajectory_plot_list(self):
        trajectory_plot_list(self.rp, self.pbo, dt_cs=self.dt_cs, save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('trajectory_plot.png')]
        self.assertTrue(len(png_files) > 0, "Counterfactual Trajectory plot files were not created.")

        trajectory_plot_list(self.rp, self.df, strat_vars=["DRUG"], save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('trajectory_plot.png')]
        self.assertTrue(len(png_files) > 0, "Trajectory plot files were not created.")

    def test_raincloud_continuous_list(self):
        raincloud_continuous_list(self.rp, self.df, strat_vars=["DRUG"], save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('raincloud_plot.png')]
        self.assertTrue(len(png_files) > 0, "Raincloud plot files were not created.")

        raincloud_continuous_list(self.rp, self.sdf, strat_vars=["DRUG"], static=True, save_path=self.save_path)
        png_files = [f for f in os.listdir(self.save_path) if f.endswith('raincloud_plot.png')]
        self.assertTrue(len(png_files) > 0, "Raincloud plot files were not created.")

    def tearDown(self):
        if os.path.exists(self.save_path):
            for f in os.listdir(self.save_path):
                if f.endswith('.png'):
                    os.remove(os.path.join(self.save_path, f))