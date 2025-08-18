import warnings

from syndat import compute_long_continuous_error_metrics, compute_long_categorical_error_metrics
from syndat.metrics import *
from syndat.scores import *
from syndat.rct.preprocessing_tidy_format import *
from syndat.rct.visualization_clical_trials import *

warnings.filterwarnings('ignore')

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

# Configuration
n_patients = 20
n_repi = 5
n_timepoints = 4
timepoints = np.arange(0, n_timepoints * 6, 6)  # [0, 6, 12, 18]
Tmax = 18
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

unique_subjids = ldt["SUBJID"].unique()
subjid_to_drug = {subjid: f'DRUG {np.random.choice([0, 1])}' for subjid in unique_subjids}
ldt["DRUG"] = ldt["SUBJID"].map(subjid_to_drug)

# Metrics
long_cont_metrics = compute_long_continuous_error_metrics(
    rp,ldt,strat_vars=["DRUG"],
    per_time_mean=True,
    per_variable_mean=True)

long_cat_metrics_w = compute_long_categorical_error_metrics(
    rp,ldt,strat_vars=["DRUG"],
    average="weighted",
    per_time_mean=True,
    per_variable_mean=True)

long_cat_metrics_macro = compute_long_categorical_error_metrics(
    rp,ldt,strat_vars=["DRUG"],
    average="macro",
    per_time_mean=True,
    per_variable_mean=True)

long_cat_metrics_micro = compute_long_categorical_error_metrics(
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
raincloud_continuous_list(rp, ldt,type='longitudinal',strat_vars=["DRUG"], save_path=results_path) 
raincloud_continuous_list(rp, sdt,type='static', save_path=results_path)

# Assume counterfactual simulations for placebo are done
pbo = ldt[ldt.DRUG==0]
dt_cs = ldt[ldt.DRUG==1]
dt_cs["DRUG"] = 0
bar_categorical_list(rp, pbo, dt_cs=dt_cs, type_='Percentage', strat_vars=["DRUG"], save_path=results_path)
bin_traj_time_list(rp, pbo, dt_cs=dt_cs, save_path=results_path)
trajectory_plot_list(rp, pbo, dt_cs=dt_cs, strat_vars=["DRUG"], save_path=results_path)