# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:09:45 2024

@author: tvanneau
"""

import os
import pickle
import mne
import matplotlib.pyplot as plt

# Define the directory where the pickle files are stored
directory = '//data2.einsteinmed.edu/users/CNL Lab/Analysis/Theo - AVSRT/Results - 6-14/Power - Audio ISI'

# Short ISI
tfr_NPL_v_short_list = []
# Loop through each subject's pickle file and load the data
for subject_id in range(1, 55):
    if subject_id==11:
        continue
    # filename = f'Power_Visual_Short_ISI_Subject_{subject_id}.pkl'
    filename = f'Power_Audio_Short_ISI_Subject_{subject_id}.pkl'

    file_path = os.path.join(directory, filename)
    
    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
        # Append the loaded variables to the respective lists
        tfr_NPL_v_short_list.append(data[1])
M_Power_V_short = mne.grand_average(tfr_NPL_v_short_list)

# Medium ISI
tfr_NPL_v_medium_list = []
# Loop through each subject's pickle file and load the data
for subject_id in range(1, 55):
    if subject_id==11:
        continue
    # filename = f'Power_Visual_Medium_ISI_Subject_{subject_id}.pkl'
    filename = f'Power_Audio_Medium_ISI_Subject_{subject_id}.pkl'

    file_path = os.path.join(directory, filename)
    
    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
        # Append the loaded variables to the respective lists
        tfr_NPL_v_medium_list.append(data[1])
M_Power_V_medium = mne.grand_average(tfr_NPL_v_medium_list)


# Medium ISI
tfr_NPL_v_long_list = []
# Loop through each subject's pickle file and load the data
for subject_id in range(1, 55):
    if subject_id==11:
        continue
    # filename = f'Power_Visual_Long_ISI_Subject_{subject_id}.pkl'
    filename = f'Power_Audio_Long_ISI_Subject_{subject_id}.pkl'

    file_path = os.path.join(directory, filename)
    
    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
        # Append the loaded variables to the respective lists
        tfr_NPL_v_long_list.append(data[1])
M_Power_V_long = mne.grand_average(tfr_NPL_v_long_list)


#%% Plot topomap of specific frequencies

# Create a figure with a grid of subplots (4 rows, 2 columns)
fig, axes = plt.subplots(3, 3, figsize=(10, 8))

fmin=13
fmax=30

tmin1=0.0
tmax1=0.1

tmin2=0.1
tmax2=0.2

tmin3=0.2
tmax3=0.3

# Theta visual
# vlim1 = -2.6
# vlim2= 0.3

# Alpha visual
# vlim1 = -2.8
# vlim2= 0.0

# Beta visual
# vlim1 = -3.2
# vlim2= 0.0

# Theta audio
# vlim1 = -2.2
# vlim2= 0.6

# Alpha visual
# vlim1 = -1.8
# vlim2= -0.2

# Beta visual
vlim1 = -2.4
vlim2= 0.0

# # Plot each pair of topomaps side by side

M_Power_V_short.plot_topomap(tmin=tmin1, tmax=tmax1, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), axes=axes[0, 0], show=False)
M_Power_V_short.plot_topomap(tmin=tmin2, tmax=tmax2, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), axes=axes[0,1], show=False)
M_Power_V_short.plot_topomap(tmin=tmin3, tmax=tmax3, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), axes=axes[0,2], show=False)

M_Power_V_medium.plot_topomap(tmin=tmin1, tmax=tmax1, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), axes=axes[1, 0], show=False)
M_Power_V_medium.plot_topomap(tmin=tmin2, tmax=tmax2, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), axes=axes[1,1], show=False)
M_Power_V_medium.plot_topomap(tmin=tmin3, tmax=tmax3, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), axes=axes[1,2], show=False)

M_Power_V_long.plot_topomap(tmin=tmin1, tmax=tmax1, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), axes=axes[2, 0], show=False)
M_Power_V_long.plot_topomap(tmin=tmin2, tmax=tmax2, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), axes=axes[2,1], show=False)
M_Power_V_long.plot_topomap(tmin=tmin3, tmax=tmax3, fmin=fmin, fmax=fmax, baseline=None, cmap='jet', vlim=(vlim1, vlim2), axes=axes[2,2], show=False)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#%% Alpha suppression
import numpy as np

freqs = np.linspace(*np.array([2, 40]), num=200) #num = step / bin allows a precision of the frequency resolution

# Occipital
# channels = ['E31','E30','E29','A11','A14','A24','A27','B8',
#             'A10','A15','A23','A28','B7',
#             'A9','A16','A22','A29','B6','B13','B12','B11']

# FINAL AUDITORY CLUSTER
# channels = ['D1','C1','D15','D2','C25']

# Frontal cluster
# channels = ['D10','D11','D7','D6','C31','C30']

# Parieto-temporal cluster
# channels = ['E27','A7','E26','A6']

# channels = ['A9','E29','E19','E20','E28','A8','E27','A7']

channels = ['E12','E4','E14','E13','E21','E22',
            'E20','E27','E26','E25',
            'E28','A7','A6']

# channels = ['E12','E4','E13','E21','E27','E26']


# FCz channels
# channels = ['D3','D14','C26','D15','D2','C25']

tmin = -0.2
tmax = 0.7

Alpha_min = 27 # 7Hz
Alpha_max = 58 # 13Hz

# Alpha_min = 6 # 3Hz
# Alpha_max = 27 # 7Hz

# Alpha_min = 58 # 7Hz
# Alpha_max = 147 # 13Hz

Alpha_suppresion_short = []
Alpha_suppresion_med = []
Alpha_suppresion_long = []

for i in range(0,len(tfr_NPL_v_short_list),1):
    
    Aph_short = tfr_NPL_v_short_list[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data
    Aph_med = tfr_NPL_v_medium_list[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data
    Aph_long = tfr_NPL_v_long_list[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data
    
    Fa_short = np.mean(np.mean(Aph_short, axis=0)[Alpha_min:Alpha_max,:],axis=0)
    Fa_med = np.mean(np.mean(Aph_med, axis=0)[Alpha_min:Alpha_max,:],axis=0)
    Fa_long = np.mean(np.mean(Aph_long, axis=0)[Alpha_min:Alpha_max,:],axis=0)
    
    Alpha_suppresion_short.append(Fa_short)
    Alpha_suppresion_med.append(Fa_med)
    Alpha_suppresion_long.append(Fa_long)

fig, ax = plt.subplots()

Mean_Q0 =   np.mean(np.array( Alpha_suppresion_short),axis=0)
Mean_Q1 =   np.mean(np.array( Alpha_suppresion_med),axis=0)
Mean_Q2 =   np.mean(np.array( Alpha_suppresion_long),axis=0)

SEM_Q0 = np.std(np.array(Alpha_suppresion_short), axis=0) / np.sqrt(len(Alpha_suppresion_short))
SEM_Q1 = np.std(np.array(Alpha_suppresion_med), axis=0) / np.sqrt(len(Alpha_suppresion_med))
SEM_Q2 = np.std(np.array(Alpha_suppresion_long), axis=0) / np.sqrt(len(Alpha_suppresion_long))

# x = np.linspace(-0.05,0.3,45)
x = np.linspace(-0.2,0.7,117)

ax.plot(x, Mean_Q0, label = 'Short ISI')
ax.fill_between(x, Mean_Q0 - SEM_Q0, Mean_Q0 + SEM_Q0, alpha=0.2)

ax.plot(x, Mean_Q1, label = 'Medium ISI')
ax.fill_between(x, Mean_Q1 - SEM_Q1, Mean_Q1 + SEM_Q1, alpha=0.2)

ax.plot(x, Mean_Q2, label = 'Long ISI')
ax.fill_between(x, Mean_Q2 - SEM_Q2, Mean_Q2 + SEM_Q2, alpha=0.2)

ax.legend()
ax.set_title('Alpha supression in response to a visual stimulation')

#%% Peak to Peak

import numpy as np

freqs = np.linspace(*np.array([2, 40]), num=200) #num = step / bin allows a precision of the frequency resolution

# Occipital
# channels = ['E31','E30','E29','A11','A14','A24','A27','B8',
#             'A10','A15','A23','A28','B7',
#             'A9','A16','A22','A29','B6','B13','B12','B11']

# FCz channels
# channels = ['D3','D14','C26','D15','D2','C25']

# channels = ['E12','E4','E13','E21','E27','E26']

# channels = ['A9','E29','E19','E20','E28','A8','E27','A7']

channels = ['E14','E13','E21','E22',
            'E20','E27','E26','E25',
            'E28','A7','A6']


tmin = 0.0
tmax = 0.3

# Alpha_min = 6 # 3Hz
# Alpha_max = 27 # 7Hz

# Alpha_min = 27 # 7Hz
# Alpha_max = 58 # 13Hz

Alpha_min = 58 # 7Hz
Alpha_max = 147 # 13Hz

Alpha_suppression_short = []
Alpha_suppression_med = []
Alpha_suppression_long = []

for i in range(len(tfr_NPL_v_short_list)):
    
    Aph_short = tfr_NPL_v_short_list[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data
    Aph_med = tfr_NPL_v_medium_list[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data
    Aph_long = tfr_NPL_v_long_list[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data
    
    Fa_short = np.mean(np.mean(Aph_short, axis=0)[Alpha_min:Alpha_max,:],axis=0)
    Fa_med = np.mean(np.mean(Aph_med, axis=0)[Alpha_min:Alpha_max,:],axis=0)
    Fa_long = np.mean(np.mean(Aph_long, axis=0)[Alpha_min:Alpha_max,:],axis=0)
    
    # Calculate peak-to-peak difference for the alpha band
    # peak_to_peak_short = np.ptp(Fa_short)
    # peak_to_peak_med = np.ptp(Fa_med)
    # peak_to_peak_long = np.ptp(Fa_long)

    # Calculate peak-to-peak difference for the alpha band
    peak_to_peak_short = np.mean(Fa_short)
    peak_to_peak_med = np.mean(Fa_med)
    peak_to_peak_long = np.mean(Fa_long)

    
    # Append the results to the lists
    Alpha_suppression_short.append(peak_to_peak_short)
    Alpha_suppression_med.append(peak_to_peak_med)
    Alpha_suppression_long.append(peak_to_peak_long)

fig, ax = plt.subplots()


plt.bar([1,2,3], [np.mean(Alpha_suppression_short), np.mean(Alpha_suppression_med),np.mean(Alpha_suppression_long)],
        tick_label=['short', 'med', 'long'])
plt.errorbar([1,2,3], [np.mean(Alpha_suppression_short), np.mean(Alpha_suppression_med),np.mean(Alpha_suppression_long)], 
             yerr = [np.std(Alpha_suppression_short)/np.sqrt(53), 
                     np.std(Alpha_suppression_med)/np.sqrt(53),
                     np.std(Alpha_suppression_long)/np.sqrt(53)],fmt='o')

plt.ylim([-1.8,-1.45])


#%%
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

# Create DataFrame for analysis
subjects = np.tile(np.arange(1, len(Alpha_suppression_short) + 1), 3)
isi_conditions = np.repeat(['short', 'med', 'long'], len(Alpha_suppression_short))
amplitudes = np.concatenate((Alpha_suppression_short,Alpha_suppression_med,Alpha_suppression_long))

data = {
    'Subject': subjects,
    'ISI': isi_conditions,
    'Amplitude': amplitudes,
}

df = pd.DataFrame(data)

# Perform repeated measures ANOVA for amplitude
rm_anova = AnovaRM(df, 'Amplitude', 'Subject', within=['ISI']).fit()
print(rm_anova)

# Perform Tukey's HSD post-hoc test for amplitude
tukey_amplitude = pairwise_tukeyhsd(df['Amplitude'], df['ISI'])
print(tukey_amplitude)