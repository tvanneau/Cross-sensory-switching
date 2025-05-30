# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:57:38 2024

@author: theov
"""

#============================================
#               AVSRT experiment
#============================================

#loading needed toolboxes 
import mne
import numpy as np
import copy
import tkinter as tk
import pandas as pd
from tkinter import filedialog
import logging
import os
from matplotlib import pyplot as plt

#%% Reading BDF files 

Subjects = os.listdir('//data2.einsteinmed.edu/users/CNL Lab/Data_new/SFARI_DATA_2024/AVSRT/AV-SRT_CLEAN_SFARI_ONLY/')

#%% 

Subject = Subjects[0] 

file_path = '//data2.einsteinmed.edu/users/CNL Lab/Data_new/SFARI_DATA_2024/AVSRT/AV-SRT_CLEAN_SFARI_ONLY/' + Subject

subject_path = f'//data2.einsteinmed.edu/users/CNL Lab/Data_new/SFARI_DATA_2024/AVSRT/AV-SRT_CLEAN_SFARI_ONLY/{Subject}'

EEG_files = os.listdir('//data2.einsteinmed.edu/users/CNL Lab/Data_new/SFARI_DATA_2024/AVSRT/AV-SRT_CLEAN_SFARI_ONLY/' + Subject)

# For subject 1211 no raw 3 / For subject 12113(6) no EEG files / 

#%%

raw_EEG = []
Events = []
for i in range(0, len(EEG_files),1):
    if (Subject=='1211') and (i==2):
        continue
    raw_tmp = mne.io.read_raw_bdf(file_path + "/" + EEG_files[1], eog=None, misc = None, stim_channel='auto',
                        exclude = (), infer_types = True, preload= True)
    
    events_tmp = mne.find_events(raw_tmp, stim_channel="Status", shortest_event = 1)
    
    raw_EEG.append(raw_tmp)
    Events.append(events_tmp)
    
raw, events= mne.concatenate_raws(raw_EEG, preload=True, 
                                    events_list=Events)
    
raw_no_PREP = raw.copy()
# raw_PREP = raw.copy()

print("Number of channels:", len(raw.ch_names))

montage = mne.channels.make_standard_montage('biosemi160')
raw_no_PREP = raw_no_PREP.set_montage(montage, on_missing='ignore')

raw_no_PREP.set_channel_types({"EXG1":"emg"})
raw_no_PREP.set_channel_types({"EXG2":"emg"})
raw_no_PREP.set_channel_types({"EXG3":"emg"})
raw_no_PREP.set_channel_types({"EXG4":"emg"})
raw_no_PREP.set_channel_types({"EXG5":"emg"})
raw_no_PREP.set_channel_types({"EXG6":"emg"})
raw_no_PREP.set_channel_types({"EXG7":"emg"})
raw_no_PREP.set_channel_types({"EXG8":"emg"})


# Test pyprep - Noisy Channels to detect bad channels in raw object

from pyprep.find_noisy_channels import NoisyChannels

nd = NoisyChannels(raw_no_PREP, random_state=None) 

nd.find_all_bads(ransac = True, channel_wise = True) # Call all the functions to detect bad channels

# Functions = 
# # # NOTE: Bad-by-NaN/flat is already run during init, no need to re-run here
#         self.find_bad_by_deviation()
#         self.find_bad_by_hfnoise()
#         self.find_bad_by_correlation()
#         self.find_bad_by_SNR()
#         if ransac:
#             self.find_bad_by_ransac(
#                 channel_wise=channel_wise, max_chunk_size=max_chunk_size
#             )

nd.get_bads()

#%% interpolate bad s1

raw_no_PREP.info["bads"] = nd.get_bads()
raw_no_PREP = raw_no_PREP.interpolate_bads()

#%% Filtration

lowpass_epochs = 40
highpass_epochs = 0.05

raw_no_PREP = raw_no_PREP.filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)

#%% Delete trial without response from the subject

events_news = events.copy()
for i in range(len(events)-1,-1, -1):
    if ((events_news[i,2] == 3) or (events_news[i,2] == 4) or (events_news[i,2] == 5)) and (events_news[i+1,2] != 1):
        events_news = np.delete(events_news, i,0)

# Delete consecutive 1
for i in range (len(events_news)-2,-1,-1):
    if events_news[i,2]==1 and events_news[i+1,2]==1:
        events_news = np.delete(events_news, i+1,0)
        
if events_news[0,2]==1:
    events_news = np.delete(events_news, 0,0)
        
events_news = events_news[np.where((events_news[:,2]==3) | (events_news[:,2]==4) | (events_news[:,2]==5) | (events_news[:,2]==1))]
events_epochs = events_news[np.where((events_news[:,2]==3) | (events_news[:,2]==4) | (events_news[:,2]==5))]

#%% Epoching EEG data with all trials after PREP

event_dict = {
    "AV": 3,
    "A": 4,
    "V": 5,
}

tmin_epochs = -0.5
tmax_epochs = 0.8

#Create epoch file without baseline correction and decim=4 to resample the data at 128Hz
epochs_all_no_PREP = mne.Epochs(raw_no_PREP, events_epochs, tmin=tmin_epochs, tmax=tmax_epochs, reject=None,event_id = event_dict, baseline=None,detrend=1, preload=True, decim=4)
epochs_all_no_PREP.drop_bad()

#%% Calculate reaction time for each stimulation

# RT all trial
RT_all = np.zeros((int(len(events_news)/2),1))
Stim_type_all = []
d=0
for i in range(0,len(events_news),2):
    RT_all[d,0] = (events_news[i+1,0] - events_news[i,0]) / raw.info['sfreq']
    d = d+1
    
    if events_news[i,2] == 3:
        Stim_type_all.append('AV')
    elif events_news[i,2] == 4:
        Stim_type_all.append('A')
    elif events_news[i,2] == 5:
        Stim_type_all.append('V')

Reaction_time_all = pd.DataFrame(data = RT_all,
                    columns = ['Reaction time'])

Reaction_time_all.insert(1,'Type of stimulation', Stim_type_all)

#Calculate ISI for each trial
sampling_rate = 512
events_times = events_epochs[:,0]

events_time_secs = events_times / sampling_rate
isis = np.diff(events_time_secs)
isis_nan = np.insert(isis, 0, np.nan)

Reaction_time_all['ISI'] = isis_nan

#Report preceding stimulus type
preceding_stim = Reaction_time_all['Type of stimulation'][0:]

nan_row = pd.Series([np.nan], name=preceding_stim.name)

pred_stim = pd.concat([nan_row, preceding_stim], ignore_index=True)

Reaction_time_all['Preceding stim']= pred_stim[:-1]


#%% Drop trial in reaction time files

dropped_epochs_all = [n for n, dl in enumerate(epochs_all_no_PREP.drop_log) if len(dl)]  # result is a list
Reaction_time_all = Reaction_time_all.drop(dropped_epochs_all)

Reaction_time_all = Reaction_time_all.set_index(pd.Index(np.arange(0,len(Reaction_time_all),1)))


#%%#INDEPENDENT COMPONENTS ANALYSIS (ICA) for PREP epochs files

epochs_ica = epochs_all_no_PREP.filter(l_freq=1., h_freq=None)
epochs_ica.set_eeg_reference('average')
#ICA est sensible aux dérives basse fréquence donc 1Hz + charge données

#Parameters ICA
n_components = None #0.99 # % de vraiance expliquée ou alors le nombre d'électrodes
max_pca_components = 160 # disparaitra dans une future version de MNE, nombre de PCA a faire avant ICA
random_state = 42 #* pour que l'ICA donne la même chose sur les mêmes données
method = 'fastica' # méthode de l'ICA (si 'picard' nécessite pip install python-picard)
fit_params = None # fastica_it=5 paramètre lié à la methode picard
max_iter = 1000 # nombre d'iterations de l'ICA

ica = mne.preprocessing.ICA(n_components=n_components, method = method, max_iter = max_iter, fit_params= fit_params, random_state=random_state)

ica.fit(epochs_ica)
ica.plot_sources(epochs_all_no_PREP)
ica.plot_components()

#%%APPLICATION ICA COMMUN AVERAGE REFERENCE
ica.apply(epochs_all_no_PREP)

#%% Baseline correction after ICA

baseline = (-0.2,0)

epochs_all_no_PREP.apply_baseline(baseline=baseline)


#%% Use of autoreject package after ICA to remove bads epochs for no PREP epochs
import autoreject

ar = autoreject.AutoReject(n_interpolate= [1, 4, 32],
                           n_jobs = 1,
                           verbose = True)

ar.fit(epochs_all_no_PREP)
epochs_ar_all_no_PREP, reject_log_all_no_PREP = ar.transform(epochs_all_no_PREP, return_log=True)

ar = autoreject.AutoReject(n_interpolate= [1, 4, 32],
                           n_jobs = 1,
                           verbose = True)

#%% Visualisation tools all no PREP

# All
# Visualize the dropped epochs
epochs_all_no_PREP[reject_log_all_no_PREP.bad_epochs].plot(scalings=dict(eeg=100e-6))

# Visualize the rejected log
reject_log_all_no_PREP.plot('horizontal')

#%% Drop RT corresponding to dropped epoch for PREP

RT_all_no_PREP = Reaction_time_all.copy().drop(np.where(reject_log_all_no_PREP.bad_epochs)[0])

RT_all_no_PREP = RT_all_no_PREP.set_index(pd.Index(np.arange(0,len(RT_all_no_PREP),1)))

#%% Plot ERP NO PREP

epochs_ar_all_no_PREP.set_eeg_reference(ref_channels='average') # For CAR: ref_channels='average' ; for frontal: ref_channels=['D8']

# epochs_ar_all_no_PREP.set_eeg_reference(ref_channels=["D6"]) # For CAR: ref_channels='average' ; for frontal: ref_channels=['D8']

ERP_AV = epochs_ar_all_no_PREP['AV'].average()
ERP_A = epochs_ar_all_no_PREP['A'].average()
ERP_V = epochs_ar_all_no_PREP['V'].average()

ERP_V.plot()

#%%

ERP_V.plot(picks=['Cz'])

#%% Saved epochs file with corresponding RT files

file_path = 'C://Users/tvanneau/Dropbox (EinsteinMed)/AVSRT-Analyzes/Adults - 160 channels - epochs files/All ISI'

# Epochs files
saving_file_all = file_path+'/'+Subject+'_AVSRT_no_PREP_ar_all_epo.fif'
epochs_ar_all_no_PREP.save(saving_file_all)


# Reaction time files
csv_path_all = file_path+"/"+Subject+"_no_PREP_ar_RT_all"
RT_all_no_PREP.to_csv(csv_path_all,  sep='\t', encoding='utf-8')

