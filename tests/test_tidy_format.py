import unittest
import pandas as pd
import numpy as np
from syndat import *


def sim_data(n_patients = 20, n_repi = 5, n_timepoints = 4, Tmax = 18):
    timepoints = np.arange(0, n_timepoints * 6, 6)
    # Define the variable types
    lt = pd.DataFrame({
        'Variable': ['Long_Var1', 'Long_Var2', 'Long_Var3'],
        'Type': ['cat', 'cat','pos'],
        'Cats': [4, 2, 1]
    })

    st = pd.DataFrame({
        'Variable': ['Stat_Var1', 'Stat_Var2', 'Stat_Var3'],
        'Type': ['cat', 'pos', 'pos'],
        'Cats': [5, 1, 1]
    })

    rows = []
    for ptno in range(1, n_patients + 1):
        for repi in range(1, n_repi + 1):
            for time in timepoints:
                row = {
                    'PTNO': ptno,
                    'REPI': repi,
                    'TIME': float(time),
                    'DRUG': -1  # constant for now
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
                    row[f'MASK_{var}'] = np.random.choice([0, 1])

                # Add OBS and REC for static vars — same across timepoints
                for _, row_st in st.iterrows():
                    var = row_st['Variable']
                    if row_st['Type'] == 'cat':
                        value_obs = np.random.randint(0, 2)
                        value_rec = np.random.randint(0, 2)
                    else:
                        value_obs = np.random.normal(loc=50, scale=10)
                        value_rec = np.random.normal(loc=50, scale=10)

                    row[f'OBS_{var}'] = value_obs
                    row[f'REC_{var}'] = value_rec
                    row[f'MASK_{var}'] = np.random.choice([0, 1])

                rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    long_vars = lt['Variable'].tolist()
    static_vars = st['Variable'].tolist()
    long_cols = ['OBS_' + v for v in long_vars] + ['REC_' + v for v in long_vars] + ['MASK_' + v for v in long_vars]
    static_cols = ['OBS_' + v for v in static_vars] + ['REC_' + v for v in static_vars] + ['MASK_' + v for v in static_vars]
    id_cols = ['PTNO', 'REPI']
    long_base_cols = id_cols + ['TIME', 'DRUG']

    # Build longitudinal and static DataFrames
    ldt = df[long_base_cols + long_cols].copy()
    sdt = df[id_cols + static_cols].drop_duplicates().reset_index(drop=True)

    return lt, st, ldt, sdt


class TestTidyFormat(unittest.TestCase):
    def setUp(self):
        self.lt, self.st, self.ldt, self.sdt = sim_data()
        self.save_path = "./examples/"
        self.strat_vars=["DRUG"]

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