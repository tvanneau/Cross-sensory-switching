# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:11:12 2024

@author: tvanneau
"""

import matplotlib as mpl

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

#============================================
#               AVSRT Analysis
#============================================

#loading needed toolboxes 
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
import tkinter
import pandas as pd
from os.path import isfile
from tkinter import filedialog
import os
from pyprep.find_noisy_channels import NoisyChannels
from pyprep.prep_pipeline import PrepPipeline

# list_file = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/AVSRT-Analyzes/Adults - 160 channels - epochs files/No PREP')
# file_path = 'C://Users/tvanneau/Dropbox (EinsteinMed)/AVSRT-Analyzes/Adults - 160 channels - epochs files/No PREP'

list_file = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/AVSRT-Analyzes/Adults - 160 channels - epochs files/All ISI')
file_path = 'C://Users/tvanneau/Dropbox (EinsteinMed)/AVSRT-Analyzes/Adults - 160 channels - epochs files/All ISI'

# list_files_epochs = [s for s in list_file if '_no_PREP_ar_1500_epo' in s]
# list_files_RT = [s for s in list_file if '_no_PREP_ar_RT_1500' in s]

list_files_epochs = [s for s in list_file if '_no_PREP_ar_all_epo' in s]
list_files_RT = [s for s in list_file if '_no_PREP_ar_RT_all' in s]

# list_files_RT = [s for s in list_file if '_no_PREP_ar_RT_2000' in s]

# Load epochs files for all subjects
epochs_lst = []
RT_lst = []
for i in range(0,len(list_files_epochs),1):
    f_name = file_path + "/" + list_files_epochs[i]
    epochs_lst.append(mne.read_epochs(f_name))
    RT_lst.append(pd.read_csv(file_path + "/" + list_files_RT[i], sep='\t', encoding='utf-8'))

del RT_lst[10] 
del epochs_lst[10]

#%% Filter both the reaction time values and the associated epochs file to keep only the 95th percentile RT trials

filtered_RT_lst = []  # To store filtered DataFrames
filtered_epochs_lst = []  # To store filtered MNE Epochs objects
upper_lst = []
lower_lst = []

for df, epochs in zip(RT_lst, epochs_lst):
    indices_to_keep = []

    for stim_type in ['A', 'V', 'AV']:
        condition_df = df[df['Type of stimulation'] == stim_type]
        
        # Calculate the 95th percentile for the condition
        lower_bound = condition_df['Reaction time'].quantile(0.025)
        upper_bound = condition_df['Reaction time'].quantile(0.975)
        
        # lower_bound = 0.15
        # upper_bound = 0.5
        
        upper_lst.append(upper_bound)
        lower_lst.append(lower_bound)
        # Filter and retain indices of the trials within the 95th percentile
        filtered_condition_indices = condition_df[(condition_df['Reaction time'] >= lower_bound) & (condition_df['Reaction time'] <= upper_bound)].index
        indices_to_keep.extend(filtered_condition_indices)

    indices_to_keep.sort()
    # Filter the DataFrame using the retained indices
    filtered_df = df.loc[indices_to_keep].copy()
    filtered_df.sort_index(inplace=True)  # Sort to maintain original order
    filtered_df.index = range(0,len(filtered_df))
    filtered_RT_lst.append(filtered_df)
    
    # Ensure that epochs are selected based on the DataFrame's index to maintain order
    # This requires that epochs.events[:, 2] (event_id) aligns with the DataFrame's index
    filtered_epochs = epochs[indices_to_keep]
    filtered_epochs_lst.append(filtered_epochs)
    
#%% Calculate the average RT for each type of stimulation
import seaborn as sns
# List to hold results
results = []
# Iterate over each DataFrame
for i, df in enumerate(filtered_RT_lst):
    # Calculate the mean reaction time for each type of stimulation
    means = df.groupby('Type of stimulation')['Reaction time'].mean().reset_index()
    means['Subject'] = f'Subject_{i+1}'
    results.append(means)

# Combine all results into a single DataFrame
combined_results = pd.concat(results)

plt.figure(figsize=(10, 6))
sns.violinplot(x='Type of stimulation', y='Reaction time', data=combined_results, inner='quartile', order=['V', 'A', 'AV'])
plt.title('Reaction Time by Type of Stimulation')
plt.xlabel('Type of Stimulation')
plt.ylabel('Reaction Time (ms)')
plt.show()

combined_results.to_csv('Ave_RT_53Subject_Stats.csv', index=False)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Process each dataframe in the list
normalized_dfs = []
for i, df in enumerate(filtered_RT_lst, start=1):
    # Filter for 'A' type of stimulation
    df_a = df[df['Type of stimulation'] == 'A'].copy()
    
    # Normalize reaction time
    df_a['Normalized RT'] = (df_a['Reaction time'] - df_a['Reaction time'].mean()) / df_a['Reaction time'].std()
    
    # Create ISI_category
    df_a['ISI_category'] = pd.cut(df_a['ISI'], bins=[1, 2, 3], labels=['short', 'long'], right=False)
    
    # Add subject_number
    df_a['subject_number'] = i
    
    normalized_dfs.append(df_a)

# Concatenate all normalized dataframes
combined_df_normalized = pd.concat(normalized_dfs)
combined_df_normalized.columns = combined_df_normalized.columns.str.replace(' ', '_').str.replace('-', '_')

#%% Plot groups of RT separated with short and long ISI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

combined_df_normalized = combined_df_normalized[combined_df_normalized['Preceding_stim'].isin(['V','A'])]

# Extract the column names
column_names = combined_df_normalized.columns
i = 5

# Calculate the average visual alpha phase stim for each combination of preceding stim type, subject, and ISI category
avg_alpha_phase_stim = combined_df_normalized.groupby(['Preceding_stim', 'subject_number'])[column_names[i]].mean().reset_index()

# Plot the results using a violin plot
plt.figure(figsize=(14, 8))
sns.violinplot(x='Preceding_stim', y=column_names[i],  data=avg_alpha_phase_stim)
plt.xlabel('Preceding Stim Type')
plt.ylabel(column_names[i])
plt.show()

# Ensure the column name does not have spaces for the formula
column_to_analyze = column_names[i].replace(" ", "_")
column_to_analyze = column_names[i].replace("-", "_")
avg_alpha_phase_stim = avg_alpha_phase_stim.rename(columns={column_names[i]: column_to_analyze})

# Define the mixed-effects model formula
formula = f"{column_to_analyze} ~ C(Preceding_stim) + (1|subject_number)"

# Define the mixed-effects model
model = smf.mixedlm(f"{column_to_analyze} ~ C(Preceding_stim)", avg_alpha_phase_stim, groups=avg_alpha_phase_stim["subject_number"])

# Fit the model
result = model.fit()

# Print the summary of the mixed-effects model
print(result.summary())

# Check the residuals
residuals = result.resid

# Plot the residuals to check for normality
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Q-Q plot to check normality
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()

# Separate data into 'short' and 'long' ISI categories
short_isi = avg_alpha_phase_stim[avg_alpha_phase_stim['ISI_category'] == 'short']
long_isi = avg_alpha_phase_stim[avg_alpha_phase_stim['ISI_category'] == 'long']

# Conduct pairwise comparisons for 'short' ISI
post_hoc_short = pg.pairwise_ttests(dv=column_to_analyze, between='Preceding_stim', padjust='bonf', data=short_isi)
print("Pairwise comparisons for 'short' ISI category:")
print(post_hoc_short)

# Conduct pairwise comparisons for 'long' ISI
post_hoc_long = pg.pairwise_ttests(dv=column_to_analyze, between='Preceding_stim', padjust='bonf', data=long_isi)
print("Pairwise comparisons for 'long' ISI category:")
print(post_hoc_long)

# Conduct pairwise comparisons between short and long ISI for each type of preceding stimuli
post_hoc_between_isi = pg.pairwise_ttests(dv=column_to_analyze, between='ISI_category', within='Preceding_stim', padjust='bonf', data=avg_alpha_phase_stim)
print("Pairwise comparisons between short and long ISI for each type of preceding stimuli:")
print(post_hoc_between_isi)


#%% Plot standard deviation for each subject 
# List to hold results
results = []

# Iterate over each DataFrame (each subject)
for i, df in enumerate(filtered_RT_lst):
    # Calculate the standard deviation of reaction time for each type of stimulation
    std_devs = df.groupby('Type of stimulation')['Reaction time'].std().reset_index()
    std_devs['Subject'] = f'Subject_{i+1}'
    results.append(std_devs)

# Combine all results into a single DataFrame
combined_results = pd.concat(results)

# Plotting
plt.figure(figsize=(10, 6))
sns.violinplot(x='Type of stimulation', y='Reaction time', data=combined_results, inner='quartile', order=['V', 'A', 'AV'])
plt.title('Reaction Time by Type of Stimulation')
plt.xlabel('Type of Stimulation')
plt.ylabel('Reaction Time (ms)')
plt.show()

combined_results.to_csv('Ave_RT_53Subject_Stats_STD.csv', index=False)

#%% Plot distribution of RT for one subject as an example
df = filtered_RT_lst[22] # 7 ok
# Plotting the KDE plot
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df[df['Type of stimulation'] == 'V'], x='Reaction time', label='V', fill=True)
sns.kdeplot(data=df[df['Type of stimulation'] == 'A'], x='Reaction time', label='A', fill=True)
sns.kdeplot(data=df[df['Type of stimulation'] == 'AV'], x='Reaction time', label='AV', fill=True)
plt.title('Distribution of Reaction Time by Type of Stimulation')
plt.xlabel('Reaction Time (ms)')
plt.ylabel('Density')
plt.legend(title='Type of Stimulation')
plt.show()

#%% Calculate ERP for each type of preceding stimuli

Evoked_Audio_0 = []
Evoked_Audio_1 = []
Evoked_Audio_2 = []

stim_type = 'V'
s=0
for rt_df, epochs in zip(filtered_RT_lst, filtered_epochs_lst):
    
    s=s+1
    print('Subject number:',s)
    # Filter DataFrame for current stimulation type
    stim_df = rt_df[rt_df['Type of stimulation'] == stim_type]
    
    # Get indices of trials in this quartile
    indices_0 = stim_df[stim_df['Preceding stim'] == 'V'].index
    indices_1 = stim_df[stim_df['Preceding stim'] == 'A'].index
    indices_2 = stim_df[stim_df['Preceding stim'] == 'AV'].index
    
    # epochs.set_eeg_reference(['E17','B18']) # Temporal ref

    # epochs.set_eeg_reference(['D4','D6','C30','D11']) # Frontal ref

    # Use indices to extract corresponding epochs
    quartile_epochs_0 = epochs[indices_0]
    quartile_epochs_1 = epochs[indices_1]
    quartile_epochs_2 = epochs[indices_2]
    
    # Extract the Evoked for that subject for each quartile
    Evoked_Audio_0.append(quartile_epochs_0.average())
    Evoked_Audio_1.append(quartile_epochs_1.average())
    Evoked_Audio_2.append(quartile_epochs_2.average())
    
    
#%% Plot topomap

Evoked_Audio_0_GA = mne.grand_average(Evoked_Audio_0)
Evoked_Audio_1_GA = mne.grand_average(Evoked_Audio_1)
# Evoked_Audio_2_GA = mne.grand_average(Evoked_Audio_2)
# Evoked_Audio_3_GA = mne.grand_average(Evoked_Audio_3)

csd_Evoked_Audio_0_GA = mne.preprocessing.compute_current_source_density(Evoked_Audio_0_GA)
csd_Evoked_Audio_1_GA = mne.preprocessing.compute_current_source_density(Evoked_Audio_1_GA)
# csd_Evoked_Audio_2_GA = mne.preprocessing.compute_current_source_density(Evoked_Audio_2_GA)


# Define your time points of interest, e.g., 100 ms and 200 ms post-stimulus
# times = [-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Time in seconds

times = [-0.1,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3]  # Time in seconds

# times = [-0.3,-0.25,-0.2,-0.15,-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Time in seconds

# Plot the topographical maps at these time points
Evoked_Audio_0_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')
Evoked_Audio_1_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')
# Evoked_Audio_2_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')
# Evoked_Audio_3_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')

csd_Evoked_Audio_0_GA.plot_topomap(times=times, size=1, vlim=(-2, 2), cmap='jet')
csd_Evoked_Audio_1_GA.plot_topomap(times=times, size=1, vlim=(-2, 2), cmap='jet')
# csd_Evoked_Audio_2_GA.plot_topomap(times=times, size=1, vlim=(-2, 2), cmap='jet')

#%%

plt.savefig('CSD_Visual_Preceded_Audio.svg')

#%% Plot ERP for a specific time range and a specific cluster of electrode

# FULL ERP

Audio_0 = []
Audio_1 = []
Audio_2 = []

# Cluster electrodes - Visual stim - 100ms timing
# channels = ['E17','E31','A11','E18','E30','A10','E19','E29','A9',
#             'B18','B11','B8','B17','B12','B7','B16','B13','B6']

# Cluster electrodes - Visual stim - 150ms timing
# channels = ['A15','A23','A28','A16','A22','A29']

# Cluster electrodes - Visual stim - 200ms timing
# channels = ['D23','D32','D24','D31','C17','C12','C16','C13']

# Cluster electrodes - Visual stim - 300ms timing 
# channels = ['E27','E26','A7','A6','E13','E21','E25','A5','A18','E20','E28','A8','E22']

tmin = -0.1
tmax = 0.3

for i in range(0,len(epochs_lst),1):
    Audio_0.append(Evoked_Audio_0[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_1.append(Evoked_Audio_1[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_2.append(Evoked_Audio_2[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    # Audio_3.append(Evoked_Audio_3[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)

x = Evoked_Audio_0[0].times[51:103]
fig, ax = plt.subplots()

# Audio Q0
Mean_Audio_0 = np.mean(np.mean(np.array(Audio_0), axis=0), axis=0)
SEM_Audio_0 = np.std(np.mean(np.array(Audio_0), axis=1), axis=0) / np.sqrt(len(Audio_0))
ax.plot(x, Mean_Audio_0, label = 'Preceded Visual')
ax.fill_between(x, Mean_Audio_0 - SEM_Audio_0, Mean_Audio_0 + SEM_Audio_0, color='blue', alpha=0.2)

# Audio Q1
Mean_Audio_1 = np.mean(np.mean(np.array(Audio_1), axis=0), axis=0)
SEM_Audio_1 = np.std(np.mean(np.array(Audio_1), axis=1), axis=0) / np.sqrt(len(Audio_1))
ax.plot(x, Mean_Audio_1, label = 'Preceded Audio')
ax.fill_between(x, Mean_Audio_1 - SEM_Audio_1, Mean_Audio_1 + SEM_Audio_1, color='orange', alpha=0.2)

# Audio Q2
# Mean_Audio_2 = np.mean(np.mean(np.array(Audio_2), axis=0), axis=0)
# SEM_Audio_2 = np.std(np.mean(np.array(Audio_2), axis=1), axis=0) / np.sqrt(len(Audio_2))
# ax.plot(x, Mean_Audio_2, label = 'Preceded Audiovisual')
# ax.fill_between(x, Mean_Audio_2 - SEM_Audio_2, Mean_Audio_2 + SEM_Audio_2, color='green', alpha=0.2)

# Audio Q3
# Mean_Audio_3 = np.mean(np.mean(np.array(Audio_3), axis=0), axis=0)
# SEM_Audio_3 = np.std(np.mean(np.array(Audio_3), axis=1), axis=0) / np.sqrt(len(Audio_3))
# ax.plot(x, Mean_Audio_3, label = 'Q4')
# ax.fill_between(x, Mean_Audio_3 - SEM_Audio_3, Mean_Audio_3 + SEM_Audio_3, color='red', alpha=0.2)


# Shading the SEM area
plt.axvline(x=x[13], color='black', linestyle='--')
ax.set_title('Occipital cluster - Visual')
ax.legend()

#%% Plot ERP for a specific time range and a specific cluster of electrode

# ZOOMED ERP

Audio_0 = []
Audio_1 = []
Audio_2 = []

# Visual fauciles
# channels = ['E31','E30','E29','A11','A10','A9',
#             'B11','B12','B13','B8','B7','B6']

# Auditory fauciles
channels = ['C2','C3','C4','C5','C6','C24',
            'D16','E2','E3','D28','D29','D17']

tmin = 0.0
tmax = 0.15

for i in range(0,len(epochs_lst),1):
    Audio_0.append(Evoked_Audio_0[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_1.append(Evoked_Audio_1[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_2.append(Evoked_Audio_2[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)

x = Evoked_Audio_0[0].times[64:84]
fig, ax = plt.subplots()

# Audio Q0
Mean_Audio_0 = np.mean(np.mean(np.array(Audio_0), axis=0), axis=0)
SEM_Audio_0 = np.std(np.mean(np.array(Audio_0), axis=1), axis=0) / np.sqrt(len(Audio_0))
ax.plot(x, Mean_Audio_0, label = 'Q1')
ax.fill_between(x, Mean_Audio_0 - SEM_Audio_0, Mean_Audio_0 + SEM_Audio_0, color='blue', alpha=0.2)

# Audio Q1
Mean_Audio_1 = np.mean(np.mean(np.array(Audio_1), axis=0), axis=0)
SEM_Audio_1 = np.std(np.mean(np.array(Audio_1), axis=1), axis=0) / np.sqrt(len(Audio_1))
ax.plot(x, Mean_Audio_1, label = 'Q2')
ax.fill_between(x, Mean_Audio_1 - SEM_Audio_1, Mean_Audio_1 + SEM_Audio_1, color='orange', alpha=0.2)

# Audio Q2
Mean_Audio_2 = np.mean(np.mean(np.array(Audio_2), axis=0), axis=0)
SEM_Audio_2 = np.std(np.mean(np.array(Audio_2), axis=1), axis=0) / np.sqrt(len(Audio_2))
ax.plot(x, Mean_Audio_2, label = 'Q3')
ax.fill_between(x, Mean_Audio_2 - SEM_Audio_2, Mean_Audio_2 + SEM_Audio_2, color='green', alpha=0.2)

# Shading the SEM area
plt.axvline(x=x[0], color='black', linestyle='--')
ax.set_title('Occipital cluster - Visual')
ax.legend()

#%% Bar graph for mean amplitude over a specific period

Audio_0 = []
Audio_1 = []
Audio_2 = []


for i in range(0,len(epochs_lst),1):
    
    # Cluster electrodes - Visual stim - 100ms timing
    channels = ['E17','E31','A11','E18','E30','A10','E19','E29','A9',
                'B18','B11','B8','B17','B12','B7','B16','B13','B6']
    tmin = 0.1
    tmax = 0.15
    
    Audio_0.append(Evoked_Audio_0[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_1.append(Evoked_Audio_1[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_2.append(Evoked_Audio_2[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    # Audio_3.append(Evoked_Audio_3[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)


plt.bar([1,2], [np.mean(Audio_0), np.mean(Audio_1)],legend=['V','A'],tick_label=['V', 'A'])

plt.errorbar([1,2], [np.mean(Audio_0), np.mean(Audio_1)], yerr = [np.std(Audio_0)/np.sqrt(55), np.std(Audio_1)/np.sqrt(55)],fmt='o')

# Stats = np.concatenate((np.mean(np.mean(Audio_0, axis=1), axis=1),np.mean(np.mean(Audio_1, axis=1), axis=1),
#                          np.mean(np.mean(Audio_2, axis=1), axis=1)))

#%% t-test for amplitude values

from scipy.stats import ttest_ind

# Example data: two groups of data
group1 = np.mean(np.mean(Audio_0, axis=1), axis=1)
group2 = np.mean(np.mean(Audio_1, axis=1), axis=1)

# Perform the t-test
t_stat, p_value = ttest_ind(group1, group2)

# Print the results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")


#%% Calculate ERP for short vs long ISI for visual trials for each individual

Evoked_Audio_0 = []
Evoked_Audio_1 = []
Evoked_Audio_2 = []

stim_type = 'V'
s=0
for rt_df, epochs in zip(filtered_RT_lst, filtered_epochs_lst):
    
    s=s+1
    print('Subject number:',s)
    # Filter DataFrame for current stimulation type
    stim_df = rt_df[rt_df['Type of stimulation'] == stim_type]
    # stim_df = stim_df[stim_df['Preceding stim'] == 'V']
    
    stim_df['ISI category'] = pd.cut(stim_df['ISI'], bins=[1, 1.66, 2.33,3], labels=['short','med', 'long'], right=False)
    
    stim_df['ISI category'] = pd.cut(stim_df['ISI'], bins=[1, 1.66, 2.33,3], labels=['short','med', 'long'], right=False)

    # Get indices of trials in this quartile
    indices_0 = stim_df[stim_df['ISI category'] == 'short'].index
    indices_1 = stim_df[stim_df['ISI category'] == 'med'].index
    indices_2 = stim_df[stim_df['ISI category'] == 'long'].index

    # Use indices to extract corresponding epochs
    quartile_epochs_0 = epochs[indices_0]
    quartile_epochs_1 = epochs[indices_1]
    quartile_epochs_2 = epochs[indices_2]

    # Extract the Evoked for that subject for each quartile
    Evoked_Audio_0.append(quartile_epochs_0.average())
    Evoked_Audio_1.append(quartile_epochs_1.average())
    Evoked_Audio_2.append(quartile_epochs_2.average())

#%% Plot topomap

Evoked_Audio_0_GA = mne.grand_average(Evoked_Audio_0)
Evoked_Audio_1_GA = mne.grand_average(Evoked_Audio_1)
Evoked_Audio_2_GA = mne.grand_average(Evoked_Audio_2)
# Evoked_Audio_3_GA = mne.grand_average(Evoked_Audio_3)

csd_Evoked_Audio_0_GA = mne.preprocessing.compute_current_source_density(Evoked_Audio_0_GA)
csd_Evoked_Audio_1_GA = mne.preprocessing.compute_current_source_density(Evoked_Audio_1_GA)
csd_Evoked_Audio_2_GA = mne.preprocessing.compute_current_source_density(Evoked_Audio_2_GA)


# Define your time points of interest, e.g., 100 ms and 200 ms post-stimulus
# times = [-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Time in seconds

times = [0.0,0.075,0.1,0.125,0.15]  # Time in seconds

# times = [-0.3,-0.25,-0.2,-0.15,-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Time in seconds

# Plot the topographical maps at these time points
Evoked_Audio_0_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')
Evoked_Audio_1_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')
Evoked_Audio_2_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')
# Evoked_Audio_3_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')

csd_Evoked_Audio_0_GA.plot_topomap(times=times, size=1, vlim=(-2.5, 2.5), cmap='jet')
csd_Evoked_Audio_1_GA.plot_topomap(times=times, size=1, vlim=(-2.5, 2.5), cmap='jet')
csd_Evoked_Audio_2_GA.plot_topomap(times=times, size=1, vlim=(-2.5, 2.5), cmap='jet')

#%%

plt.savefig('CSD_Audiovisual_Preceded_Visual.svg')

#%% Plot ERP for a specific time range and a specific cluster of electrode

# FULL ERP

Audio_0 = []
Audio_1 = []
Audio_2 = []


# channels = ['A23', 'A10', 'A15', 'A28', 'B7', 
#            'A9', 'A16', 'A22', 'A29', 'B6',
#            'A11','A14','A24','A27','B8']

channels = ['E31','E30','E29','A11','A10','A9',
            'B11','B12','B13','B8','B7','B6']

tmin = -0.2
tmax = 0.4

for i in range(0,len(epochs_lst),1):
    Audio_0.append(Evoked_Audio_0[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_1.append(Evoked_Audio_1[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_2.append(Evoked_Audio_2[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)

x = Evoked_Audio_0[0].times[38:116]
fig, ax = plt.subplots()

# Audio Q0
Mean_Audio_0 = np.mean(np.mean(np.array(Audio_0), axis=0), axis=0)
SEM_Audio_0 = np.std(np.mean(np.array(Audio_0), axis=1), axis=0) / np.sqrt(len(Audio_0))
ax.plot(x, Mean_Audio_0, label = 'Short ISI')
ax.fill_between(x, Mean_Audio_0 - SEM_Audio_0, Mean_Audio_0 + SEM_Audio_0, color='blue', alpha=0.2)

# Audio Q1
Mean_Audio_1 = np.mean(np.mean(np.array(Audio_1), axis=0), axis=0)
SEM_Audio_1 = np.std(np.mean(np.array(Audio_1), axis=1), axis=0) / np.sqrt(len(Audio_1))
ax.plot(x, Mean_Audio_1, label = 'Medium ISI')
ax.fill_between(x, Mean_Audio_1 - SEM_Audio_1, Mean_Audio_1 + SEM_Audio_1, color='orange', alpha=0.2)

# Audio Q2
Mean_Audio_2 = np.mean(np.mean(np.array(Audio_2), axis=0), axis=0)
SEM_Audio_2 = np.std(np.mean(np.array(Audio_2), axis=1), axis=0) / np.sqrt(len(Audio_2))
ax.plot(x, Mean_Audio_2, label = 'Long ISI')
ax.fill_between(x, Mean_Audio_2 - SEM_Audio_2, Mean_Audio_2 + SEM_Audio_2, color='orange', alpha=0.2)

# Shading the SEM area
plt.axvline(x=x[26], color='black', linestyle='--')
ax.set_title('Occipital cluster - Visual')
ax.legend()

#%% Plot ERP for a specific time range and a specific cluster of electrode

# FULL ERP

Audio_0 = []
Audio_1 = []
Audio_2 = []


channels = ['E31','E30','E29','A11','A10','A9',
            'B11','B12','B13','B8','B7','B6']

tmin = 0.0
tmax = 0.15

for i in range(0,len(epochs_lst),1):
    Audio_0.append(Evoked_Audio_0[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_1.append(Evoked_Audio_1[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_2.append(Evoked_Audio_2[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)

x = Evoked_Audio_0[0].times[64:84]
fig, ax = plt.subplots()

# Audio Q0
Mean_Audio_0 = np.mean(np.mean(np.array(Audio_0), axis=0), axis=0)
SEM_Audio_0 = np.std(np.mean(np.array(Audio_0), axis=1), axis=0) / np.sqrt(len(Audio_0))
ax.plot(x, Mean_Audio_0, label = 'Short ISI')
ax.fill_between(x, Mean_Audio_0 - SEM_Audio_0, Mean_Audio_0 + SEM_Audio_0, color='blue', alpha=0.2)

# Audio Q1
Mean_Audio_1 = np.mean(np.mean(np.array(Audio_1), axis=0), axis=0)
SEM_Audio_1 = np.std(np.mean(np.array(Audio_1), axis=1), axis=0) / np.sqrt(len(Audio_1))
ax.plot(x, Mean_Audio_1, label = 'Long ISI')
ax.fill_between(x, Mean_Audio_1 - SEM_Audio_1, Mean_Audio_1 + SEM_Audio_1, color='orange', alpha=0.2)

# Audio Q2
Mean_Audio_2 = np.mean(np.mean(np.array(Audio_2), axis=0), axis=0)
SEM_Audio_2 = np.std(np.mean(np.array(Audio_2), axis=1), axis=0) / np.sqrt(len(Audio_2))
ax.plot(x, Mean_Audio_2, label = 'Long ISI')
ax.fill_between(x, Mean_Audio_2 - SEM_Audio_2, Mean_Audio_2 + SEM_Audio_2, color='orange', alpha=0.2)

# Shading the SEM area
plt.axvline(x=x[0], color='black', linestyle='--')
ax.set_title('Occipital cluster - Visual')
ax.legend()

#%%
Audio_0 = []
Audio_1 = []
Audio_2 = []


for i in range(0,len(epochs_lst),1):
    
    channels = ['E31','E30','E29','A11','A10','A9',
                'B11','B12','B13','B8','B7','B6']

    tmin = 0.08
    tmax = 0.15
    
    Audio_0.append(Evoked_Audio_0[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_1.append(Evoked_Audio_1[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_2.append(Evoked_Audio_2[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    # Audio_3.append(Evoked_Audio_3[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)


plt.bar([1,2,3], [np.mean(Audio_0), np.mean(Audio_1),np.mean(Audio_2)])

plt.errorbar([1,2,3], [np.mean(Audio_0), np.mean(Audio_1),np.mean(Audio_2)], yerr = [np.std(Audio_0)/np.sqrt(55), np.std(Audio_1)/np.sqrt(55), np.std(Audio_2)/np.sqrt(55)],fmt='o')

Stats = np.concatenate((np.mean(np.mean(Audio_0, axis=1), axis=1),np.mean(np.mean(Audio_1, axis=1), axis=1),
                         np.mean(np.mean(Audio_2, axis=1), axis=1)))

#%% Bar graph for mean amplitude over a specific period

Audio_0 = []
Audio_1 = []

for i in range(0,len(epochs_lst),1):

    tmin = 0.0
    tmax = 0.15
    
    Audio_0.append(Evoked_Audio_0[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_1.append(Evoked_Audio_1[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)


plt.bar([1,2], [np.mean(Audio_0), np.mean(Audio_1)])

plt.errorbar([1,2], [np.mean(Audio_0), np.mean(Audio_1)], yerr = [np.std(Audio_0)/np.sqrt(53), np.std(Audio_1)/np.sqrt(53)],fmt='o')

Stats = np.concatenate((np.mean(np.mean(Audio_0, axis=1), axis=1),np.mean(np.mean(Audio_1, axis=1), axis=1)))

#%% t-test for amplitude values

from scipy.stats import ttest_ind

# Example data: two groups of data
group1 = np.mean(np.mean(Audio_0, axis=1), axis=1)
group2 = np.mean(np.mean(Audio_2, axis=1), axis=1)

# Perform the t-test
t_stat, p_value = ttest_ind(group1, group2)

# Print the results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
#%% Plot topomap

Evoked_Audio_0_GA = mne.grand_average(Evoked_Audio_0)
Evoked_Audio_1_GA = mne.grand_average(Evoked_Audio_1)

Evoked_Audio_0_GA.set_montage('biosemi160')
# Define your time points of interest, e.g., 100 ms and 200 ms post-stimulus
# times = [-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Time in seconds

times = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,0.11,0.12,0.13,0.14,0.15]  # Time in seconds

# times = [-0.3,-0.25,-0.2,-0.15,-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Time in seconds

# Plot the topographical maps at these time points
Evoked_Audio_0_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')
Evoked_Audio_1_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')

#%%

plt.savefig('Topomap_Visual_Preceded_Audiovisual.svg')


#%% Calculate ERP for fast versus slow repeated visual trials for each individual

Evoked_Audio_0 = []
Evoked_Audio_1 = []

stim_type = 'V'
s=0
for rt_df, epochs in zip(filtered_RT_lst, filtered_epochs_lst):
    
    s=s+1
    print('Subject number:',s)
    # Filter DataFrame for current stimulation type
    stim_df = rt_df[rt_df['Type of stimulation'] == stim_type]
    stim_df = stim_df[stim_df['Preceding stim'] == 'V']
    
    stim_df['Quartile'] = stim_df['Reaction time'].transform(
        lambda x: pd.qcut(x, 2, labels=False) + 1  # +1 so the quartiles are labeled 1 to 4 instead of 0 to 3
    )
    
    # Get indices of trials in this quartile
    indices_0 = stim_df[stim_df['Quartile'] == 1].index
    indices_1 = stim_df[stim_df['Quartile'] == 2].index

    # Use indices to extract corresponding epochs
    quartile_epochs_0 = epochs[indices_0]
    quartile_epochs_1 = epochs[indices_1]

    # Extract the Evoked for that subject for each quartile
    Evoked_Audio_0.append(quartile_epochs_0.average())
    Evoked_Audio_1.append(quartile_epochs_1.average())
    
#%% Plot ERP for a specific time range and a specific cluster of electrode

# FULL ERP

Audio_0 = []
Audio_1 = []


channels = ['E31','E30','E29','A11','A10','A9',
            'B11','B12','B13','B8','B7','B6']

tmin = -0.2
tmax = 0.4

for i in range(0,len(epochs_lst),1):
    Audio_0.append(Evoked_Audio_0[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_1.append(Evoked_Audio_1[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)

x = Evoked_Audio_0[0].times[38:116]
fig, ax = plt.subplots()

# Audio Q0
Mean_Audio_0 = np.mean(np.mean(np.array(Audio_0), axis=0), axis=0)
SEM_Audio_0 = np.std(np.mean(np.array(Audio_0), axis=1), axis=0) / np.sqrt(len(Audio_0))
ax.plot(x, Mean_Audio_0, label = 'Fast RT')
ax.fill_between(x, Mean_Audio_0 - SEM_Audio_0, Mean_Audio_0 + SEM_Audio_0, color='blue', alpha=0.2)

# Audio Q1
Mean_Audio_1 = np.mean(np.mean(np.array(Audio_1), axis=0), axis=0)
SEM_Audio_1 = np.std(np.mean(np.array(Audio_1), axis=1), axis=0) / np.sqrt(len(Audio_1))
ax.plot(x, Mean_Audio_1, label = 'Slow RT')
ax.fill_between(x, Mean_Audio_1 - SEM_Audio_1, Mean_Audio_1 + SEM_Audio_1, color='orange', alpha=0.2)

# Shading the SEM area
plt.axvline(x=x[26], color='black', linestyle='--')
ax.set_title('Occipital cluster - Visual')
ax.legend()


#%% Plot ERP for a specific time range and a specific cluster of electrode

# FULL ERP

Audio_0 = []
Audio_1 = []


channels = ['E31','E30','E29','A11','A10','A9',
            'B11','B12','B13','B8','B7','B6']
tmin = 0.0
tmax = 0.15

for i in range(0,len(epochs_lst),1):
    Audio_0.append(Evoked_Audio_0[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_1.append(Evoked_Audio_1[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)

x = Evoked_Audio_0[0].times[64:84]
fig, ax = plt.subplots()

# Audio Q0
Mean_Audio_0 = np.mean(np.mean(np.array(Audio_0), axis=0), axis=0)
SEM_Audio_0 = np.std(np.mean(np.array(Audio_0), axis=1), axis=0) / np.sqrt(len(Audio_0))
ax.plot(x, Mean_Audio_0, label = 'Fast RT')
ax.fill_between(x, Mean_Audio_0 - SEM_Audio_0, Mean_Audio_0 + SEM_Audio_0, color='blue', alpha=0.2)

# Audio Q1
Mean_Audio_1 = np.mean(np.mean(np.array(Audio_1), axis=0), axis=0)
SEM_Audio_1 = np.std(np.mean(np.array(Audio_1), axis=1), axis=0) / np.sqrt(len(Audio_1))
ax.plot(x, Mean_Audio_1, label = 'Slow RT')
ax.fill_between(x, Mean_Audio_1 - SEM_Audio_1, Mean_Audio_1 + SEM_Audio_1, color='orange', alpha=0.2)

# Shading the SEM area
plt.axvline(x=x[0], color='black', linestyle='--')
ax.set_title('Occipital cluster - Visual')
ax.legend()

#%% Bar graph for mean amplitude over a specific period

Audio_0 = []
Audio_1 = []

for i in range(0,len(epochs_lst),1):

    tmin = 0.05
    tmax = 0.15
    
    Audio_0.append(Evoked_Audio_0[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_1.append(Evoked_Audio_1[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)


plt.bar([1,2], [np.mean(Audio_0), np.mean(Audio_1)])

plt.errorbar([1,2], [np.mean(Audio_0), np.mean(Audio_1)], yerr = [np.std(Audio_0)/np.sqrt(53), np.std(Audio_1)/np.sqrt(53)],fmt='o')

Stats = np.concatenate((np.mean(np.mean(Audio_0, axis=1), axis=1),np.mean(np.mean(Audio_1, axis=1), axis=1)))
#%% Plot topomap

Evoked_Audio_0_GA = mne.grand_average(Evoked_Audio_0)
Evoked_Audio_1_GA = mne.grand_average(Evoked_Audio_1)


# Define your time points of interest, e.g., 100 ms and 200 ms post-stimulus
# times = [-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Time in seconds

times = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,0.11,0.12,0.13,0.14,0.15]  # Time in seconds

# times = [-0.3,-0.25,-0.2,-0.15,-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Time in seconds

# Plot the topographical maps at these time points
Evoked_Audio_0_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')
Evoked_Audio_1_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')

#%%

plt.savefig('Topomap_Visual_Preceded_Audiovisual.svg')


#%% Calculate ERP for each type of preceding stimuli and for each type of ISI

Evoked_visual_short = []
Evoked_visual_long = []

Evoked_audio_short = []
Evoked_audio_long = []

Evoked_audiovisual_short = []
Evoked_audiovisual_long = []

stim_type = 'A'
s=0
for rt_df, epochs in zip(filtered_RT_lst, filtered_epochs_lst):
    
    s=s+1
    print('Subject number:',s)
    # Filter DataFrame for current stimulation type
    stim_df = rt_df[rt_df['Type of stimulation'] == stim_type]
    stim_df['ISI category'] = pd.cut(stim_df['ISI'], bins=[1, 2,3], labels=['short','long'], right=False)

    # Get indices of trials in this quartile
    idx_visual_short = stim_df[(stim_df['Preceding stim'] == 'V') & (stim_df['ISI category'] == 'short')].index
    idx_visual_long = stim_df[(stim_df['Preceding stim'] == 'V') & (stim_df['ISI category'] == 'long')].index

    idx_audio_short = stim_df[(stim_df['Preceding stim'] == 'A') & (stim_df['ISI category'] == 'short')].index
    idx_audio_long = stim_df[(stim_df['Preceding stim'] == 'A') & (stim_df['ISI category'] == 'long')].index

    idx_audiovisual_short = stim_df[(stim_df['Preceding stim'] == 'AV') & (stim_df['ISI category'] == 'short')].index
    idx_audiovisual_long = stim_df[(stim_df['Preceding stim'] == 'AV') & (stim_df['ISI category'] == 'long')].index

    # epochs.set_eeg_reference(['E17','B18']) # Temporal ref

    # epochs.set_eeg_reference(['D4','D6','C30','D11']) # Frontal ref

    # Use indices to extract corresponding epochs
    epochs_visual_short = epochs[idx_visual_short]
    epochs_visual_long = epochs[idx_visual_long]
    
    epochs_audio_short = epochs[idx_audio_short]
    epochs_audio_long = epochs[idx_audio_long]
    
    epochs_audiovisual_short = epochs[idx_audiovisual_short]
    epochs_audiovisual_long = epochs[idx_audiovisual_long]
    
    # Extract the Evoked for that subject for each quartile
    Evoked_visual_short.append(epochs_visual_short.average())
    Evoked_visual_long.append(epochs_visual_long.average())

    Evoked_audio_short.append(epochs_audio_short.average())
    Evoked_audio_long.append(epochs_audio_long.average())

    Evoked_audiovisual_short.append(epochs_audiovisual_short.average())
    Evoked_audiovisual_long.append(epochs_audiovisual_long.average())


#%% Plot topomap

Evoked_Audio_0_GA = mne.grand_average(Evoked_visual_short)
Evoked_Audio_1_GA = mne.grand_average(Evoked_visual_long)

Evoked_Audio_2_GA = mne.grand_average(Evoked_audio_short)
Evoked_Audio_3_GA = mne.grand_average(Evoked_audio_long)

Evoked_Audio_4_GA = mne.grand_average(Evoked_audiovisual_short)
Evoked_Audio_5_GA = mne.grand_average(Evoked_audiovisual_long)

csd_Evoked_Audio_0_GA = mne.preprocessing.compute_current_source_density(Evoked_Audio_0_GA)
csd_Evoked_Audio_1_GA = mne.preprocessing.compute_current_source_density(Evoked_Audio_1_GA)
csd_Evoked_Audio_2_GA = mne.preprocessing.compute_current_source_density(Evoked_Audio_2_GA)
csd_Evoked_Audio_3_GA = mne.preprocessing.compute_current_source_density(Evoked_Audio_3_GA)
csd_Evoked_Audio_4_GA = mne.preprocessing.compute_current_source_density(Evoked_Audio_4_GA)
csd_Evoked_Audio_5_GA = mne.preprocessing.compute_current_source_density(Evoked_Audio_5_GA)

# Define your time points of interest, e.g., 100 ms and 200 ms post-stimulus
# times = [-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Time in seconds

times = [-0.1,0.1,0.15,0.2,0.25,0.3]  # Time in seconds

# times = [-0.3,-0.25,-0.2,-0.15,-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Time in seconds

# Plot the topographical maps at these time points
Evoked_Audio_0_GA.plot_topomap(times=times, size=1, vlim=(-2.5, 2.5), cmap='jet')
Evoked_Audio_2_GA.plot_topomap(times=times, size=1, vlim=(-2.5, 2.5), cmap='jet')
Evoked_Audio_2_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')
# Evoked_Audio_3_GA.plot_topomap(times=times, size=1, vlim=(-1, 1), cmap='jet')

csd_Evoked_Audio_0_GA.plot_topomap(times=times, size=1, vlim=(-2.3, 2.3), cmap='jet')
csd_Evoked_Audio_1_GA.plot_topomap(times=times, size=1, vlim=(-2.3, 2.3), cmap='jet')
csd_Evoked_Audio_2_GA.plot_topomap(times=times, size=1, vlim=(-2.3, 2.3), cmap='jet')
csd_Evoked_Audio_3_GA.plot_topomap(times=times, size=1, vlim=(-2.3, 2.3), cmap='jet')
csd_Evoked_Audio_4_GA.plot_topomap(times=times, size=1, vlim=(-2.3, 2.3), cmap='jet')
csd_Evoked_Audio_5_GA.plot_topomap(times=times, size=1, vlim=(-2.3, 2.3), cmap='jet')

#%%

plt.savefig('Topomap_CSD_Audio_Preceded_Audiovisual_Long_ISI.svg')

#%% FULL ERP

Visual_short = []
Visual_long = []

Audio_short = []
Audio_long = []

Audiovisual_short = []
Audiovisual_long = []

# Centro-frontal
# channels = ['D4','D3','D2','D1','C1','D15','C25','D14','C26','C28','D13']

# Motor 
# channels = ['E13','E21','E22','E27','E26','E25','E20','E28','A7','A6']

# Occipital
# channels = [ 'A11', 'A14', 'A24', 'A27', 'B8',
#               'A10', 'A15', 'A23', 'A28', 'B7',
#             'A9', 'A16', 'A22', 'A29', 'B6']

# Centro-parietal
channels = [ 'A3','A4','A19','A5','A32','E24','B2']

# Visual fauciles
# channels = [ 'E31','E30','E29','A11','A10','A9',
#             'B11','B12','B13','B8','B7','B6']

tmin = -0.1
tmax = 0.6

for i in range(0,len(epochs_lst),1):
    Visual_short.append(Evoked_visual_short[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Visual_long.append(Evoked_visual_long[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    
    Audio_short.append(Evoked_audio_short[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audio_long.append(Evoked_audio_long[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    
    Audiovisual_short.append(Evoked_audiovisual_short[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)
    Audiovisual_long.append(Evoked_audiovisual_long[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data)

time_array = Evoked_visual_short[0].times
tmin_idx = np.where(time_array >= tmin)[0][0]
tmax_idx = np.where(time_array <= tmax)[0][-1]
x = Evoked_visual_short[0].times[tmin_idx:tmax_idx+3]

# x = Evoked_visual_short[0].times[64:84]

fig, ax = plt.subplots()

# Audio Q0
# Mean_Audio_0 = np.mean(np.mean(np.array(Visual_short), axis=0), axis=0)
# SEM_Audio_0 = np.std(np.mean(np.array(Visual_short), axis=1), axis=0) / np.sqrt(len(Visual_short))
# ax.plot(x, Mean_Audio_0, label = 'Visual_short')
# ax.fill_between(x, Mean_Audio_0 - SEM_Audio_0, Mean_Audio_0 + SEM_Audio_0, color='blue', alpha=0.2)

# # Audio Q0
Mean_Audio_1 = np.mean(np.mean(np.array(Visual_long), axis=0), axis=0)
SEM_Audio_1 = np.std(np.mean(np.array(Visual_long), axis=1), axis=0) / np.sqrt(len(Visual_long))
ax.plot(x, Mean_Audio_1, label = 'Visual_long')
ax.fill_between(x, Mean_Audio_1 - SEM_Audio_1, Mean_Audio_1 + SEM_Audio_1, color='blue', alpha=0.2)

# Audio Q0
# Mean_Audio_2 = np.mean(np.mean(np.array(Audio_short), axis=0), axis=0)
# SEM_Audio_2 = np.std(np.mean(np.array(Audio_short), axis=1), axis=0) / np.sqrt(len(Audio_short))
# ax.plot(x, Mean_Audio_2, label = 'Audio_short')
# ax.fill_between(x, Mean_Audio_2 - SEM_Audio_2, Mean_Audio_2 + SEM_Audio_2, color='orange', alpha=0.2)

# # Audio Q0
Mean_Audio_3 = np.mean(np.mean(np.array(Audio_long), axis=0), axis=0)
SEM_Audio_3 = np.std(np.mean(np.array(Audio_long), axis=1), axis=0) / np.sqrt(len(Audio_long))
ax.plot(x, Mean_Audio_3, label = 'Audio_long')
ax.fill_between(x, Mean_Audio_3 - SEM_Audio_3, Mean_Audio_3 + SEM_Audio_3, color='orange', alpha=0.2)

# Audio Q0
# Mean_Audio_4 = np.mean(np.mean(np.array(Audiovisual_short), axis=0), axis=0)
# SEM_Audio_4 = np.std(np.mean(np.array(Audiovisual_short), axis=1), axis=0) / np.sqrt(len(Audiovisual_short))
# ax.plot(x, Mean_Audio_4, label = 'Audiovisual_short')
# ax.fill_between(x, Mean_Audio_4 - SEM_Audio_4, Mean_Audio_4 + SEM_Audio_4, color='green', alpha=0.2)

# # Audio Q0
Mean_Audio_5 = np.mean(np.mean(np.array(Audiovisual_long), axis=0), axis=0)
SEM_Audio_5 = np.std(np.mean(np.array(Audiovisual_long), axis=1), axis=0) / np.sqrt(len(Audiovisual_long))
ax.plot(x, Mean_Audio_5, label = 'Audiovisual_long')
ax.fill_between(x, Mean_Audio_5 - SEM_Audio_5, Mean_Audio_5 + SEM_Audio_5, color='green', alpha=0.2)

# Shading the SEM area
plt.axvline(x=x[12], color='black', linestyle='--')
# plt.axvline(x=x[0], color='black', linestyle='--')
# ax.set_ylim([-0.5e-6,2.5e-6])
ax.set_ylim([-1.0e-6,2.2e-6])
ax.set_xlim([-0.03,0.3])

ax.set_title('Occipital cluster - Visual')
ax.legend()

#%% Bar graph for mean amplitude over a specific period

Audio_0 = []
Audio_1 = []
Audio_2 = []
Audio_3 = []
Audio_4 = []
Audio_5 = []

for i in range(0,len(epochs_lst),1):
    
    # Centro-frontal
    # channels = ['D4','D3','D2','D1','C1','D15','C25','D14','C26','C28','D13']
    
    # Motor 
    channels = ['E13','E21','E22','E27','E26','E25','E20','E28','A7','A6']
    
    # Occipital
    # channels = [ 'A11', 'A14', 'A24', 'A27', 'B8',
    #               'A10', 'A15', 'A23', 'A28', 'B7',
    #             'A9', 'A16', 'A22', 'A29', 'B6']
    
    # Centro-parietal
    # channels = [ 'A3','A4','A19','A5','A32','E24','B2']

    tmin = 0.15
    tmax = 0.30
    
    Audio_0.append(np.mean(Evoked_visual_short[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data))
    Audio_1.append(np.mean(Evoked_visual_long[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data))
    
    Audio_2.append(np.mean(Evoked_audio_short[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data))
    Audio_3.append(np.mean(Evoked_audio_long[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data))
    
    Audio_4.append(np.mean(Evoked_audiovisual_short[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data))
    Audio_5.append(np.mean(Evoked_audiovisual_long[i].copy().pick(channels).crop(tmin=tmin, tmax=tmax).data))


# Average the data within each list
mean_Audio_0 = np.mean(Audio_0, axis=0)
mean_Audio_1 = np.mean(Audio_1, axis=0)
mean_Audio_2 = np.mean(Audio_2, axis=0)
mean_Audio_3 = np.mean(Audio_3, axis=0)
mean_Audio_4 = np.mean(Audio_4, axis=0)
mean_Audio_5 = np.mean(Audio_5, axis=0)

std_Audio_0 = np.std(Audio_0, axis=0) / np.sqrt(53)
std_Audio_1 = np.std(Audio_1, axis=0)/ np.sqrt(53)
std_Audio_2 = np.std(Audio_2, axis=0)/ np.sqrt(53)
std_Audio_3 = np.std(Audio_3, axis=0)/ np.sqrt(53)
std_Audio_4 = np.std(Audio_4, axis=0)/ np.sqrt(53)
std_Audio_5 = np.std(Audio_5, axis=0)/ np.sqrt(53)

species = ("Short", "Long")
penguin_means = {
    'Preceded Audio': (mean_Audio_2, mean_Audio_3),
    'Preceded Audiovisual': (mean_Audio_4, mean_Audio_5),
    'Preceded Visual': (mean_Audio_0, mean_Audio_1)}
penguin_stds = {
    'Preceded Audio': (std_Audio_2, std_Audio_3),
    'Preceded Audiovisual': (std_Audio_4, std_Audio_5),
    'Preceded Visual': (std_Audio_0, std_Audio_1)}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(12, 8))

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    error = penguin_stds[attribute]
    rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=error, capsize=5)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Amplitude')
ax.set_title('Grouped bar graph of Audio conditions with error bars')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=3)

# ax.set_ylim([-2.80e-6,-1.0e-6])

plt.show()

#%%
import pingouin as pg

subjects = np.arange(len(Audio_0))

# Prepare the data in long format
data = {
    'Subject': np.tile(subjects, 6),
    'Value': np.concatenate([Audio_0, Audio_1, Audio_2, Audio_3, Audio_4, Audio_5]),
    'Species': np.concatenate([['Short'] * len(Audio_0), ['Long'] * len(Audio_1), ['Short'] * len(Audio_2), ['Long'] * len(Audio_3), ['Short'] * len(Audio_4), ['Long'] * len(Audio_5)]),
    'Preceding_Stim': np.concatenate([['Preceded Visual'] * len(Audio_0), ['Preceded Visual'] * len(Audio_1), ['Preceded Audio'] * len(Audio_2), ['Preceded Audio'] * len(Audio_3), ['Preceded Audiovisual'] * len(Audio_4), ['Preceded Audiovisual'] * len(Audio_5)])
}

df = pd.DataFrame(data)

# Run the repeated measures ANOVA
aov = pg.rm_anova(dv='Value', within=['Species', 'Preceding_Stim'], subject='Subject', data=df, detailed=True)
print(aov)

post_hoc = pg.pairwise_tests(dv='Value', within=['Species', 'Preceding_Stim'], subject='Subject', data=df, padjust='bonf')
print(post_hoc)

#%% Spatial cluster based permutation

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:22:32 2024

@author: tvanneau
"""

import mne
from mne.stats import spatio_temporal_cluster_test
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import ttest_ind

# Assuming Evoked_audio_long and Evoked_visual_long are already defined
# Convert lists of Evoked objects to a 3D array (subjects, channels, time points)
data_audio = np.array([evoked.data for evoked in Evoked_visual_short])
data_visual = np.array([evoked.data for evoked in Evoked_audio_short])

# Ensure the data shapes are correct
assert data_audio.shape == data_visual.shape, "Shape mismatch between conditions"

# Define the time window for the analysis
time_min = -0.05
time_max = 0.3

# Find the indices for the specified time window
times = Evoked_audio_long[0].times
time_mask = (times >= time_min) & (times <= time_max)
selected_times = times[time_mask]

# Subset the data to the specified time window
data_audio = data_audio[:, :, time_mask]
data_visual = data_visual[:, :, time_mask]

n_subjects, n_channels, n_times = data_audio.shape

# Define adjacency matrix for clustering over spatial dimensions (electrodes)
adjacency, ch_names = mne.channels.find_ch_adjacency(Evoked_audio_long[0].info, ch_type='eeg')

# Prepare the data for the test
X = np.concatenate([data_audio, data_visual], axis=0)
X = X[:, :, :, np.newaxis]  # Add a singleton dimension for compatibility with spatio_temporal_cluster_test

# Create temporal adjacency (each time point is only adjacent to itself)
temporal_adjacency = sparse.eye(n_times, format='csr')

# Create full spatio-temporal adjacency matrix
spatial_adjacency = sparse.kron(sparse.eye(n_times), adjacency, format='csr')
full_adjacency = spatial_adjacency + sparse.kron(temporal_adjacency, sparse.eye(n_channels, format='csr'), format='csr')

# Perform the cluster-based permutation test
T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_test(
    [X[:n_subjects], X[n_subjects:]], threshold=None, n_permutations=500, adjacency=full_adjacency, tail=0, n_jobs=1
)

# Create a mask for significant clusters (e.g., p < 0.05)
significant_mask = np.zeros(T_obs.shape, dtype=bool)
for cluster, p_value in zip(clusters, cluster_p_values):
    if p_value < 0.05:
        significant_mask[cluster] = True

# Plot the results
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(T_obs.squeeze(), aspect='auto', origin='lower', cmap='Reds', extent=[selected_times[0], selected_times[-1], 0, n_channels],vmax=15)
plt.colorbar(im, ax=ax, label='F-obs')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Electrodes')

# Overlay significant clusters
for i in range(significant_mask.shape[0]):
    for j in range(significant_mask.shape[1]):
        if significant_mask[i, j]:
            ax.scatter(selected_times[j], i, color='k', s=10)

ax.set_title('Statistical Differences Between Audio and Visual Conditions')

# Add topomaps
num_segments = 4
segment_size = n_channels // num_segments
topomap_axs = []
for segment in range(num_segments):
    start_idx = segment * segment_size
    end_idx = (segment + 1) * segment_size
    ax_topo = fig.add_axes([0.85, 0.1 + segment * 0.2, 0.1, 0.2])
    mask = np.zeros(n_channels, dtype=bool)
    mask[start_idx:end_idx] = True
    mask_params = dict(marker='o', markerfacecolor='r', markeredgecolor='r', linewidth=0)
    mne.viz.plot_topomap(np.zeros(n_channels), Evoked_audio_long[0].info, axes=ax_topo, show=False, mask=mask, mask_params=mask_params)
    ax_topo.set_title(f'{start_idx}-{end_idx}', fontsize=8)

plt.show()
