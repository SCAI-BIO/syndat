import os
import pandas as pd
import numpy as np
import syndat
import warnings
from plotnine import ggsave
from helper_dataloading_functions import *
from helper_plot_functions import *
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
 
distribution_similarity = syndat.scores.distribution(sdt_obs, sdt_rec, aggregate_results=False)
discrimination_scores = {'Stat':syndat.scores.discrimination(sdt_obs, sdt_rec)}
correlation_scores = {'Stat':syndat.scores.correlation(sdt_obs, sdt_rec)}

distribution_similarity.update(syndat.scores.distribution(ldt_obs, ldt_rec, aggregate_results=False))
try:
    discrimination_scores.update({'Long': syndat.scores.discrimination(ldt_obs, ldt_rec)})
except:
    discrimination_scores.update({'Long': np.nan})
correlation_scores.update({'Long': syndat.scores.correlation(ldt_obs, ldt_rec)})
 
# Here begins the new part
rp = get_rp(ldt, lt, st)
#ldt = convert_data(ldt,'long',only_pos=True)
#sdt = convert_data(sdt,'static',only_pos=True)
 
# For debugging
#ldt.to_pickle(os.path.join(results_path, 'ldt_debug.pkl'))
#sdt.to_pickle(os.path.join(results_path, 'sdt_debug.pkl'))
 
ldt = pd.read_pickle(os.path.join(results_path, 'ldt_debug.pkl'))
sdt = pd.read_pickle(os.path.join(results_path, 'sdt_debug.pkl'))
import ipdb; ipdb.set_trace() 
unique_subjids = ldt["SUBJID"].unique()
subjid_to_drug = {subjid: f'DRUG {np.random.choice([0, 1])}' for subjid in unique_subjids}
ldt_dummy = ldt.copy()
ldt_dummy["DRUG"] = ldt_dummy["SUBJID"].map(subjid_to_drug)

gof_list = gof_con_list(rp, ldt_dummy, strat_vars=["DRUG"])
for var_name, plot in gof_list.items():
    filename = os.path.join(results_path, '%s_gof_plot.png'%(var_name))
    ggsave(plot, filename, width=8, height=6, dpi=300)

gof_list = gof_bin_list(rp, ldt_dummy, strat_vars=["DRUG"])
for var_name, plot in gof_list.items():
    filename = os.path.join(results_path, '%s_gof_bin_plot.png'%(var_name))
    ggsave(plot, filename, width=8, height=6, dpi=300)
 
 
gof_list = gof_cat_list(rp, ldt_dummy, strat_vars=["DRUG"])
for var_name, plot in gof_list.items():
    filename = os.path.join(results_path, '%s_gof_cat_perc_plot.png'%(var_name))
    ggsave(plot, filename, width=8, height=6, dpi=300)
 
gof_list = gof_cat_list(rp, ldt_dummy, strat_vars=["DRUG"], type_="Subjects")
for var_name, plot in gof_list.items():
    filename = os.path.join(results_path, '%s_gof_cat_subj_plot.png'%(var_name))
    ggsave(plot, filename, width=8, height=6, dpi=300)
 
traj_list = var_plot_list(rp, ldt_dummy, strat_vars=["DRUG"])
for var_name, plot in traj_list.items():
    filename = os.path.join(results_path, '%s_traj_plot.png'%(var_name))
    ggsave(plot, filename, width=8, height=6, dpi=300)
 
raincloud_list = raincloud_con_list(rp, ldt_dummy,type='longitudinal',strat_vars=["DRUG"])
for var_name, plot in raincloud_list.items():
    filename = os.path.join(results_path, '%s_raincloud_plot.png'%(var_name))
    ggsave(plot, filename, width=8, height=6, dpi=300)
 
raincloud_list = raincloud_con_list(rp, sdt,type='static')
for var_name, plot in raincloud_list.items():
    filename = os.path.join(results_path, '%s_raincloud_plot.png'%(var_name))
    ggsave(plot, filename, width=8, height=6, dpi=300)
 