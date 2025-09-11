import warnings
import pandas as pd
import numpy as np
import random
from syndat.metrics import *
from syndat.scores import *
from syndat.rct.metrics_rct import compute_continuous_error_metrics, compute_categorical_error_metrics
from syndat.rct.preprocessing_tidy_format import *
from syndat.rct.visualization_clinical_trials import *
from tests.test_visualizations_rct import generate_mock_rp_and_df
warnings.filterwarnings('ignore')

# Define the variable types
# 'cat' refers to categorical values, please define the number of categories
#       in the 'Cats' vector 
# For binary variables use also cat as the type
# For any other continuous value please use the following values
#       positive value: 'pos' and 1 as number of categories,
#       real value i.e positive or negative and 1 as number of categories,
#       probabilities also 1 as number of categories
# Names can be assigned according to user preferences
lt = pd.DataFrame({
    'Variable': ['Long_Var1', 'Long_Var2', 'Long_Var3'],
    'Type': ['cat', 'cat','pos'],
    'Cats': [4, 2, 1]
})

st = pd.DataFrame({
    'Variable': ['Stat_Var1', 'Stat_Var2', 'Stat_Var3', 'Stat_Var4'],
    'Type': ['cat', 'pos', 'cat','pos'],
    'Cats': [5, 1, 2, 1]
})

# Configuration
n_patients = 20
n_repi = 5
n_timepoints = 4
timepoints = np.arange(0, n_timepoints * 6, 6)  # [0, 6, 12, 18]
Tmax = 18
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

long_vars = lt['Variable'].tolist()
static_vars = st['Variable'].tolist()
long_cols = ['OBS_' + v for v in long_vars] + ['REC_' + v for v in long_vars] + ['MASK_' + v for v in long_vars]
static_cols = ['OBS_' + v for v in static_vars] + ['REC_' + v for v in static_vars] + ['MASK_' + v for v in static_vars]
id_cols = ['PTNO', 'REPI']
long_base_cols = id_cols + ['TIME', 'DRUG']

# In case you have two dataframes (real and synthetic one). You can also
# use our function to merge them and organie them correctly
# If the real dataframe does not include a MASK_x column for each variable, then
# the column will be created assuming all time points are observed
real_ldf = df[long_base_cols + ['OBS_' + v for v in long_vars] + ['MASK_' + v for v in long_vars]].copy()
synthetic_ldf = df[long_base_cols + ['REC_' + v for v in long_vars]].copy()
real_ldf = real_ldf.rename(
    columns={col: col.replace("OBS_", "") for col in real_ldf.columns 
             if col.startswith("OBS_")})

synthetic_ldf = synthetic_ldf.rename(
    columns={col: col.replace("REC_", "") for col in synthetic_ldf.columns 
             if col.startswith("REC_")})

real_sdf = df[id_cols + ['OBS_' + v for v in static_vars] + ['MASK_' + v for v in static_vars]].copy()
synthetic_sdf = df[id_cols + ['REC_' + v for v in static_vars]].copy()
real_sdf = real_sdf.rename(
    columns={col: col.replace("OBS_", "") for col in real_sdf.columns 
             if col.startswith("OBS_")})

synthetic_sdf = synthetic_sdf.rename(
    columns={col: col.replace("REC_", "") for col in synthetic_sdf.columns 
             if col.startswith("REC_")})

ldt = merge_real_synthetic(real_ldf, synthetic_ldf, patient_identifier='PTNO', type='longitudinal')
sdt = merge_real_synthetic(real_sdf, synthetic_sdf, patient_identifier='PTNO', type='static')

# In this case, some time points were not observed for some patients,
# therefore we are building the dataframe with the Mask directly
ldt = df[long_base_cols + long_cols].copy()
sdt = df[id_cols + static_cols].drop_duplicates().reset_index(drop=True)

# converting to SYNDAT format
ldt_obs, ldt_rec = convert_to_syndat_scores(ldt)
sdt_obs, sdt_rec = convert_to_syndat_scores(sdt, only_pos=True)

distribution_similarity = jensen_shannon_distance(sdt_obs, sdt_rec)
correlation_scores = {'Stat':correlation(sdt_obs, sdt_rec)}
distribution_similarity.update(jensen_shannon_distance(ldt_obs, ldt_rec))
correlation_scores.update({'Long': correlation(ldt_obs, ldt_rec)})
 
# Here begins the new part
rp = get_rp(ldt, lt, st)
ldt = convert_data_to_tidy(ldt,'long',only_pos=True)
sdt = convert_data_to_tidy(sdt,'static',only_pos=True)

# Metrics
long_cont_metrics = compute_continuous_error_metrics(
    rp,ldt,strat_vars=["DRUG"],
    per_time_mean=True,
    per_variable_mean=True)

long_cat_metrics_w = compute_categorical_error_metrics(
    rp,ldt,strat_vars=["DRUG"],
    average="weighted",
    per_time_mean=True,
    per_variable_mean=True)

long_cat_metrics_macro = compute_categorical_error_metrics(
    rp,ldt,strat_vars=["DRUG"],
    average="macro",
    per_time_mean=True,
    per_variable_mean=True)

long_cat_metrics_micro = compute_categorical_error_metrics(
    rp,ldt,strat_vars=["DRUG"],
    average="micro",
    per_time_mean=True,
    per_variable_mean=True)

# Plots
results_path='./'
gof_continuous_list(rp, ldt, strat_vars=["DRUG"], save_path=results_path)
gof_binary_list(rp, ldt, strat_vars=["DRUG"], save_path=results_path)
bar_categorical_list(rp, ldt, strat_vars=["DRUG"], save_path=results_path)
bar_categorical_list(rp, ldt, strat_vars=["DRUG"], type_="Subjects", save_path=results_path)
trajectory_plot_list(rp, ldt, strat_vars=["DRUG"], save_path=results_path) 
#raincloud_continuous_list(rp, ldt,strat_vars=["DRUG"], save_path=results_path) 
print('static')
raincloud_continuous_list(rp, sdt,static=True, save_path=results_path)

# Assume counterfactual simulations for placebo are done
pbo = ldt[ldt.DRUG=="Placebo"]
dt_cs = ldt[ldt.DRUG=="Treated"]
dt_cs["DRUG"] = "Placebo"
bar_categorical_list(rp, pbo, dt_cs=dt_cs, type_='Percentage', strat_vars=["DRUG"], save_path=results_path)
bin_traj_time_list(rp, pbo, dt_cs=dt_cs, save_path=results_path)
trajectory_plot_list(rp, pbo, dt_cs=dt_cs, strat_vars=["DRUG"], save_path=results_path)