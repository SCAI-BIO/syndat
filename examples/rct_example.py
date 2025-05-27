import os
import pandas as pd
import numpy as np
import syndat
import warnings
from syndat.preprocessing_tidy_format import *
from syndat.visualization_clical_trials import *
warnings.filterwarnings('ignore')

# Expected to be done by the user cause we do not know how does the data look like
results_path = '/home/valderramanino/SYNTHIA/Development_SDG/SYNDAT/raw_output/GERAS_JAPAN/CurrentBest_NoFall_model_fit/Val_Imgs_Sampling_PSD/Raw_Output'
data_path = '/home/valderramanino/SYNTHIA/Development_SDG/SYNDAT/data/GERAS_JAPAN'
 
ldt = pd.read_csv(os.path.join(results_path, 'Sims_Long_Enc0_EP1833.csv'), na_values='.')
sdt = pd.read_csv(os.path.join(results_path, 'Sims_Stat_EP1833.csv'), na_values='.')
Tmax = 18
ldt['TIME'] = ldt['TIME'] * Tmax
 
lt = pd.read_csv(os.path.join(data_path, 'long_types_nofall.csv'), na_values='.')
st = pd.read_csv(os.path.join(data_path, 'static_types.csv'), na_values='.')
 
 
# converting to SYNDAT format
ldt_obs, ldt_rec = convert_to_syndat_scores(ldt)
sdt_obs, sdt_rec = convert_to_syndat_scores(ldt, only_pos=True)
 
distribution_similarity = syndat.metrics.jensen_shannon_distance(sdt_obs, sdt_rec)
discrimination_scores = {'Stat':syndat.scores.discrimination(sdt_obs, sdt_rec)}
correlation_scores = {'Stat':syndat.scores.correlation(sdt_obs, sdt_rec)}

distribution_similarity.update(syndat.metrics.jensen_shannon_distance(ldt_obs, ldt_rec))
try:
    discrimination_scores.update({'Long': syndat.scores.discrimination(ldt_obs, ldt_rec)})
except:
    discrimination_scores.update({'Long': np.nan})
correlation_scores.update({'Long': syndat.scores.correlation(ldt_obs, ldt_rec)})
 
# Here begins the new part
rp = get_rp(ldt, lt, st)
#ldt = convert_data_to_tidy(ldt,'long',only_pos=True)
#sdt = convert_data_to_tidy(sdt,'static',only_pos=True)
 
# For debugging
#ldt.to_pickle(os.path.join(results_path, 'ldt_debug.pkl'))
#sdt.to_pickle(os.path.join(results_path, 'sdt_debug.pkl'))
 
ldt = pd.read_pickle(os.path.join(results_path, 'ldt_debug.pkl'))
sdt = pd.read_pickle(os.path.join(results_path, 'sdt_debug.pkl'))

unique_subjids = ldt["SUBJID"].unique()
subjid_to_drug = {subjid: f'DRUG {np.random.choice([0, 1])}' for subjid in unique_subjids}
ldt_dummy = ldt.copy()
ldt_dummy["DRUG"] = ldt_dummy["SUBJID"].map(subjid_to_drug)

gof_continuous_list(rp, ldt_dummy, strat_vars=["DRUG"], save_path=results_path)
gof_binary_list(rp, ldt_dummy, strat_vars=["DRUG"], save_path=results_path)
gof_categorical_list(rp, ldt_dummy, strat_vars=["DRUG"], save_path=results_path)
gof_categorical_list(rp, ldt_dummy, strat_vars=["DRUG"], type_="Subjects", save_path=results_path)
trajectory_plot_list(rp, ldt_dummy, strat_vars=["DRUG"], save_path=results_path) 
raincloud_continuous_list(rp, ldt_dummy,type='longitudinal',strat_vars=["DRUG"], save_path=results_path) 
raincloud_continuous_list(rp, sdt,type='static', save_path=results_path) 