import unittest
import pandas as pd
import numpy as np
import random
from syndat import *


def sim_data(n_patients = 20, n_repi = 5, n_timepoints = 4, Tmax = 18):
    timepoints = np.arange(0, n_timepoints * 6, 6)
    # Define the variable types
    lt = pd.DataFrame({
        'Variable': ['Long_Var1', 'Long_Var2', 'Long_Var3'],
        'Type': ['cat', 'cat','pos'],
        'Cats': [4, 2, 1]})

    st = pd.DataFrame({
        'Variable': ['Stat_Var1', 'Stat_Var2', 'Stat_Var3', 'Stat_Var4'],
        'Type': ['cat', 'pos', 'cat','pos'],
        'Cats': [5, 1, 2, 1]})

    rows = []
    for ptno in range(1, n_patients + 1):
        drug = random.choice(["Placebo", "Treated"])
        mask_lt = {var: {time: np.random.choice([0, 1]) for time in timepoints}
                for var in lt['Variable']}
        mask_st = {var: np.random.choice([0, 1]) for var in st['Variable']}

        for repi in range(1, n_repi + 1):
            static_values = {}
            for _, row_st in st.iterrows():
                var = row_st['Variable']
                if row_st['Type'] == 'cat':
                    static_values[f'OBS_{var}'] = np.random.randint(0, 2)
                    static_values[f'REC_{var}'] = np.random.randint(0, 2)
                else:
                    static_values[f'OBS_{var}'] = np.random.normal(loc=50, scale=10)
                    static_values[f'REC_{var}'] = np.random.normal(loc=50, scale=10)
                static_values[f'MASK_{var}'] = mask_st[var]

            for time in timepoints:
                row = {
                    'PTNO': ptno,
                    'REPI': repi,
                    'TIME': float(time),
                    'DRUG': drug
                }

                # Add OBS and REC for longitudinal vars
                for _, row_lt in lt.iterrows():
                    var = row_lt['Variable']
                    if row_lt['Type'] == 'cat':
                        row[f'OBS_{var}'] = np.random.randint(0, 4)  # simulate categories
                        row[f'REC_{var}'] = np.random.randint(0, 4)
                    else:
                        row[f'OBS_{var}'] = np.random.normal(loc=50, scale=10)
                        row[f'REC_{var}'] = np.random.normal(loc=50, scale=10)  
                    row[f'MASK_{var}'] = mask_lt[var][time]

            row.update(static_values)
            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    return lt, st, df


class TestTidyFormat(unittest.TestCase):
    def setUp(self):
        self.lt, self.st, df = sim_data()

        self.long_vars = self.lt['Variable'].tolist()
        self.static_vars = self.st['Variable'].tolist()
        long_cols = ['OBS_' + v for v in self.long_vars] + ['REC_' + v for v in self.long_vars] + ['MASK_' + v for v in self.long_vars]
        static_cols = ['OBS_' + v for v in self.static_vars] + ['REC_' + v for v in self.static_vars] + ['MASK_' + v for v in self.static_vars]
        self.id_cols = ['PTNO', 'REPI']
        self.long_base_cols = self.id_cols + ['TIME', 'DRUG']

        # Build longitudinal and static DataFrames
        ldt = df[self.long_base_cols + long_cols].copy()
        sdt = df[self.id_cols + static_cols].drop_duplicates().reset_index(drop=True)
        self.sdt = sdt
        self.ldt = ldt
        self.save_path = "./examples/"
        self.strat_vars=["DRUG"]

    def test_merge_data(self):
        real_ldf = self.ldt[self.long_base_cols + ['OBS_' + v for v in self.long_vars] + ['MASK_' + v for v in self.long_vars]].copy()
        synthetic_ldf = self.ldt[self.long_base_cols + ['REC_' + v for v in self.long_vars]].copy()
        real_ldf = real_ldf.rename(
            columns={col: col.replace("OBS_", "") for col in real_ldf.columns 
                    if col.startswith("OBS_")})

        synthetic_ldf = synthetic_ldf.rename(
            columns={col: col.replace("REC_", "") for col in synthetic_ldf.columns 
                    if col.startswith("REC_")})

        real_sdf = self.sdt[self.id_cols + ['OBS_' + v for v in self.static_vars] + ['MASK_' + v for v in self.static_vars]].copy()
        synthetic_sdf = self.sdt[self.id_cols + ['REC_' + v for v in self.static_vars]].copy()
        real_sdf = real_sdf.rename(
            columns={col: col.replace("OBS_", "") for col in real_sdf.columns 
                    if col.startswith("OBS_")})

        synthetic_sdf = synthetic_sdf.rename(
            columns={col: col.replace("REC_", "") for col in synthetic_sdf.columns 
                    if col.startswith("REC_")})
    
        ldt = merge_real_synthetic(real_ldf, synthetic_ldf, patient_identifier='PTNO', type='longitudinal')
        self.assertIsInstance(ldt, pd.DataFrame, "ldt is not a Dataframe")

        sdt = merge_real_synthetic(real_sdf, synthetic_sdf, patient_identifier='PTNO', type='static')
        self.assertIsInstance(sdt, pd.DataFrame, "sdt is not a Dataframe")

        # Without DRUG and REPI
        real_ldf_ = real_ldf.drop(columns=["DRUG", "REPI"])
        synthetic_ldf_ = synthetic_ldf.drop(columns=["DRUG", "REPI"])

        ldt = merge_real_synthetic(real_ldf_, synthetic_ldf_, patient_identifier='PTNO', type='longitudinal')
        self.assertIsInstance(ldt, pd.DataFrame, "ldt is not a Dataframe")

        # Without mask
        real_ldf = self.ldt[self.long_base_cols + ['OBS_' + v for v in self.long_vars]].copy()
        ldt = merge_real_synthetic(real_ldf, synthetic_ldf, patient_identifier='PTNO', type='longitudinal')
        self.assertIsInstance(ldt, pd.DataFrame, "ldt is not a Dataframe")

        real_ldf_no_time = real_ldf.drop(columns=["TIME"])
        synthetic_ldf_no_time = synthetic_ldf.drop(columns=["TIME"])

        with self.assertRaises(AssertionError):
            merge_real_synthetic(real_ldf_no_time, synthetic_ldf_no_time,
                                 patient_identifier='PTNO', type='longitudinal')

        real_sdf_time = real_ldf.copy()
        real_sdf_time['TIME'] = 0
        synthetic_sdf_time = synthetic_ldf.copy()
        synthetic_sdf_time['TIME'] = 0
        with self.assertRaises(AssertionError):
            merge_real_synthetic(real_sdf_time, synthetic_sdf_time,
                                 patient_identifier='PTNO', type='static')

        with self.assertRaises(AssertionError):
            merge_real_synthetic(real_sdf_time, synthetic_sdf_time,
                                 patient_identifier='PTNO', type='random')


    def test_get_rp(self):
        rp_ = get_rp(self.ldt, self.lt, self.st)
        self.assertIsNotNone(rp_)

    def test_convert_data_to_syndat_scores(self):
        ldt_ = convert_to_syndat_scores(self.ldt)
        self.assertIsNotNone(ldt_)

    def test_convert_data_to_tidy_long(self):
        ldt_ = convert_data_to_tidy(self.ldt,'long',only_pos=True)
        self.assertIsNotNone(ldt_)

    def test_convert_data_to_tidy_static(self):
        std_ = convert_data_to_tidy(self.ldt,'static',only_pos=True)
        self.assertIsNotNone(std_)