import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from syndat.metrics import *
from syndat.scores import *
from syndat.preprocessing_tidy_format import *
from syndat.visualization_clical_trials import *

fold_path = '/home/valderramanino/INDIA'
placebo_flag = pd.read_csv(os.path.join(fold_path,'PlaceboFlag.csv'), na_values='.')
sdt_Enc0 = pd.read_csv(os.path.join(fold_path,'Sims_Stat_S_Enc0_EP21.csv'), na_values='.')
ldt_Enc0 = pd.read_csv(os.path.join(fold_path,'Sims_Long_Enc0_EP21.csv'), na_values='.')

Tmin = 1
Tmax = 500

ldt_Enc0['TIME'] = (ldt_Enc0['Delta_days'] * (Tmax - Tmin) + Tmin).round().astype(int)
placebo_flag['TIME'] = (placebo_flag['Delta_days'] * (Tmax - Tmin) + Tmin).round().astype(int)
ids_all_zero = placebo_flag.groupby('subject_id')['Placebo_Flag'].sum() == 0
ids_all_zero = ids_all_zero[ids_all_zero].index.tolist()

ldt_Enc0['DRUG'] = ldt_Enc0['subject_id'].apply(lambda x: 0 if x in ids_all_zero else 1)
ldt_Enc0 = ldt_Enc0.rename(columns={'subject_id': 'PTNO'}).drop(columns=['Delta_days'])
placebo_flag=placebo_flag.drop(columns=['Delta_days'])

lt = pd.read_csv(os.path.join(fold_path,'TEST_L_Types.csv'), na_values='.')
st = pd.read_csv(os.path.join(fold_path,'TEST_S_Types.csv'), na_values='.')

lt_Enc0 =  lt[lt['Enc'] == 1]
st_Enc0 =  st[st['Enc'] == 1]

# converting to SYNDAT format
ldt_Enc0_obs, ldt_Enc0_rec = convert_to_syndat_scores(ldt_Enc0)
sdt_Enc0_obs, sdt_Enc0_rec = convert_to_syndat_scores(sdt_Enc0, only_pos=True) 

# distribution_similarity = jensen_shannon_distance(sdt_Enc0_obs, sdt_Enc0_rec)
# discrimination_scores = {'Stat_Enc0':discrimination(sdt_Enc0_obs, sdt_Enc0_rec)}
# correlation_scores = {'Stat_Enc0':correlation(sdt_Enc0_obs, sdt_Enc0_rec)}

# distribution_similarity.update(jensen_shannon_distance(ldt_Enc0_obs, ldt_Enc0_rec))
# try:
#     discrimination_scores.update({'Long_Enc0': discrimination(ldt_Enc0_obs, ldt_Enc0_rec)})
# except:
#     discrimination_scores.update({'Long_Enc0': np.nan})

# correlation_scores.update({'Long_Enc0': correlation(ldt_Enc0_obs, ldt_Enc0_rec)})

rp_Enc0 = get_rp(ldt_Enc0, lt_Enc0, st_Enc0)

rp_Enc0['long_cont'] = ['FVC@FVC_Subject_Liters',
                        'VITALSIGNS@Blood_Pressure_Diastolic',
                        'VITALSIGNS@Blood_Pressure_Systolic',
                        'VITALSIGNS@Pulse']
rp_Enc0['long_cat'] = ['ALSFRS@Q1_Speech', 'ALSFRS@Q2_Salivation',
                       'ALSFRS@Q3_Swallowing', 'ALSFRS@Q4_Handwriting']
ldt_Enc = ldt_Enc0.copy()
ldt_Enc = ldt_Enc[(ldt_Enc.PTNO < 30) & (ldt_Enc.REPI < 5)]
# ldt_Enc = ldt_Enc[(ldt_Enc.PTNO < 100)]
print('Converting to tidy format')
ldt = convert_data_to_tidy(ldt_Enc,'long',only_pos=True)
# trajectory_plot_list(rp_Enc0, ldt, strat_vars=["DRUG"], save_path=fold_path)
bins = list(range(1, 501, 10))
pbo = ldt[ldt.DRUG==0]
dt_cs = ldt[ldt.DRUG==1]
dt_cs["DRUG"] = 0

bar_categorical_list(rp_Enc0, pbo, dt_cs=dt_cs, type_='Percentage', strat_vars=["DRUG"], save_path=fold_path)
trajectory_plot_list(rp_Enc0, pbo, dt_cs=dt_cs, strat_vars=["DRUG"], bins=bins, save_path=fold_path)
