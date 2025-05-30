# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:28:54 2024

@author: tvanneau
"""

#loading needed toolboxes 
import mne
import numpy as np
import pandas as pd
import os
import gc
import pickle
from fooof import FOOOF
from fooof.utils import trim_spectrum

# list_file = os.listdir('C://Users/tvanneau/Dropbox (EinsteinMed)/AVSRT-Analyzes/Adults - 160 channels - epochs files/No PREP')
# file_path = 'C://Users/tvanneau/Dropbox (EinsteinMed)/AVSRT-Analyzes/Adults - 160 channels - epochs files/No PREP'

list_file = os.listdir('C://Users/theov/Dropbox (EinsteinMed)/AVSRT-Analyzes/Adults - 160 channels - epochs files/All ISI')
file_path = 'C://Users/theov/Dropbox (EinsteinMed)/AVSRT-Analyzes/Adults - 160 channels - epochs files/All ISI'

# list_files_epochs = [s for s in list_file if '_no_PREP_ar_1500_epo' in s]
# list_files_RT = [s for s in list_file if '_no_PREP_ar_RT_1500' in s]

list_files_epochs = [s for s in list_file if '_no_PREP_ar_all_epo' in s]
list_files_RT = [s for s in list_file if '_no_PREP_ar_RT_all' in s]

# list_files_RT = [s for s in list_file if '_no_PREP_ar_RT_2000' in s]

# Load epochs files for all subjects

for ix in range(0,len(list_files_epochs),1):
    
    Subject = ix+1
    print('Subject number:',ix)
    
    f_name = file_path + "/" + list_files_epochs[ix]
    epochs = mne.read_epochs(f_name)
    df = pd.read_csv(file_path + "/" + list_files_RT[ix], sep='\t', encoding='utf-8')
    
    if ix==10:
        continue
    
    indices_to_keep = []

    for stim_type in ['A', 'V', 'AV']:
        condition_df = df[df['Type of stimulation'] == stim_type]
        
        # Calculate the 95th percentile for the condition
        lower_bound = condition_df['Reaction time'].quantile(0.025)
        upper_bound = condition_df['Reaction time'].quantile(0.975)
        
        # Filter and retain indices of the trials within the 95th percentile
        filtered_condition_indices = condition_df[(condition_df['Reaction time'] >= lower_bound) & (condition_df['Reaction time'] <= upper_bound)].index
        indices_to_keep.extend(filtered_condition_indices)

    indices_to_keep.sort()
    # Filter the DataFrame using the retained indices
    filtered_df = df.loc[indices_to_keep].copy()
    filtered_df.sort_index(inplace=True)  # Sort to maintain original order
    filtered_df.index = range(0,len(filtered_df))
    
    # Ensure that epochs are selected based on the DataFrame's index to maintain order
    # This requires that epochs.events[:, 2] (event_id) aligns with the DataFrame's index
    filtered_epochs = epochs[indices_to_keep]
    
    # Select the type of stimulation 
    stim_type = 'V'

    # Filter DataFrame for current stimulation type
    stim_df = filtered_df[filtered_df['Type of stimulation'] == stim_type]
    
    # stim_df['ISI category'] = pd.cut(stim_df['ISI'], bins=[1, 2, 3], labels=['short', 'long'], right=False)
    # stim_df['ISI category2'] = pd.cut(stim_df['ISI'], bins=[1, 1.66, 2.33,3], labels=['short', 'med','long'], right=False)

    # Select only the visual stimulation
    indices_stim = stim_df.index
    epochs_stim = filtered_epochs[indices_stim]
    stim_df.index = range(0,len(stim_df))
    
    # Select only the stim with preceding stim = 'V' or 'A' (get rid of 'AV')
    stim_df = stim_df[stim_df['Preceding stim'].isin(['V','A'])]
    indices_stim = stim_df.index
    epochs_stim = epochs_stim[indices_stim]
    stim_df.index = range(0,len(stim_df))

    # Extract amplitude for windows of 0-200, 200-400 and 400-600 ms
    # and maximum peak value as latency of the peak for specific cluster of electrodes 
    
    epochs_stim.set_eeg_reference(['D2','D3'])
    
    # Define the clusters of electrodes
    
    # Occipital visual cluster
    cluster_1 = ['E31','E30','E29','A11','A14','A24','A27','B8',
                'A10','A15','A23','A28','B7',
                'A9','A16','A22','A29','B6','B13','B12','B11']
    
    # Define a function to get channel indices
    def get_channel_indices(channel_names, cluster):
        return [channel_names.index(ch) for ch in cluster]
    
    # Assuming `epochs_stim` is a NumPy array of shape (n_epochs, n_channels, n_times)
    # and `channel_names` is a list of all channel names in the correct order
    channel_names = epochs_stim.ch_names
    # Get channel indices for each cluster
    cluster_1_indices = get_channel_indices(channel_names, cluster_1)

    # Initialize new columns in the dataframe for ERP parameters
    columns = ['ERP_MinPeak_160_240ms_Occi','ERP_MinPeakLatency_160_240ms_Occi',
               'ERP_MaxPeak_220_300ms_Occi','ERP_MaxPeakLatency_220_300ms_Occi']
    
    for col in columns:
        stim_df[col] = np.nan
    
    sfreq = epochs_stim.info['sfreq']  # Use the sampling frequency from the epochs object

    # Loop through each epoch
    for idx in range(len(epochs_stim)):
        epoch = epochs_stim[idx]
        
        # Extract the maximum peak and latency for the specified window and clusters
        data_1_160_240ms = epoch.copy().crop(tmin=0.16, tmax=0.24).pick_channels(cluster_1).get_data().mean(axis=1)
        max_peak_1 = data_1_160_240ms.min()
        max_peak_latency_1 = (data_1_160_240ms.argmin() / sfreq) + 0.16
        stim_df.loc[idx, 'ERP_MinPeak_160_240ms_Occi'] = max_peak_1
        stim_df.loc[idx, 'ERP_MinPeakLatency_160_240ms_Occi'] = max_peak_latency_1
        
        # Extract the minimum peak and latency for the specified window and cluster
        data_1_220_300ms = epoch.copy().crop(tmin=0.22, tmax=0.30).pick_channels(cluster_1).get_data().mean(axis=1)
        min_peak_1 = data_1_220_300ms.max()
        min_peak_latency_1 = (data_1_220_300ms.argmax() / sfreq) + 0.22
        stim_df.loc[idx, 'ERP_MaxPeak_220_300ms_Occi'] = min_peak_1
        stim_df.loc[idx, 'ERP_MaxPeakLatency_220_300ms_Occi'] = min_peak_latency_1
        
    epochs_stim.set_eeg_reference('average')

    # Time-frequency analysis for each quartile 
    freqs = np.linspace(*np.array([2, 40]), num=200) #num = step / bin allows a precision of the frequency resolution
    n_cycles = freqs / 2  # different number of cycle per frequency
    
    # Total power for each quartile
    power = mne.time_frequency.tfr_morlet(epochs_stim, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            output = 'complex', return_itc=False, decim=1, n_jobs=1, average=False)
    
    # Del data to save memory
    del epochs, filtered_epochs, df, filtered_df
    gc.collect()
    
    # Define clusters of electrodes for each cortical area
    electrode_clusters = {
        'Occipital': ['E31','E30','E29','A11','A14','A24','A27','B8',
                    'A10','A15','A23','A28','B7',
                    'A9','A16','A22','A29','B6','B13','B12','B11'],
        'Central': ['D1','C1','D15','D2','C25'],
        'Frontal': ['D10','D11','D7','D6','C31','C30'],
        'Motor': ['E27','A7','E26','A6'],
        'Fronto-central':['D3','D14','C26','D15','D2','C25']
    }
    
    # Define frequency bands of interest
    freq_bands = {
        'Theta': (3, 7),
        'Alpha': (7, 13),
        'Beta': (13, 30)
    }
    
    # Define the time windows
    time_windows = {
        'baseline': (-0.3, 0.0),
        'post_0_100ms': (0.0, 0.1),
        'post_100_150ms': (0.1, 0.15),
        'post_150_200ms': (0.15, 0.2),
        'post_200_250ms': (0.2, 0.25),
        'post_0_150ms': (0.0, 0.15),
        'post_150_300ms': (0.15, 0.3),
    }
    
    # Reset the index of the DataFrame
    stim_df.reset_index(drop=True, inplace=True)
    
    # Verify that all electrode names are present in the channel names
    for region, electrodes in electrode_clusters.items():
        for electrode in electrodes:
            if electrode not in power.info['ch_names']:
                raise ValueError(f"Electrode {electrode} in region {region} not found in channel names")
    
    baseline_period = (-0.3, 0.0)
    baseline_indices = np.where((power.times >= baseline_period[0]) & (power.times <= baseline_period[1]))[0]
    
    # Initialize columns in stim_df for phase values at t=0
    new_columns = []
    for region in electrode_clusters:
        for band in freq_bands:
            new_columns.append(f'{region}_{band}_Phase_Stim_Total')
    
    # Initialize new columns in the DataFrame
    stim_df = pd.concat([stim_df, pd.DataFrame(columns=new_columns)], axis=1)
    stim_df[new_columns] = np.nan
    
    ######### Calculate average phase values at t=0 #########
    stim_time = 0
    stim_index = np.where(power.times == stim_time)[0][0]
    
    for i in range(len(power)):
        epoch = power[i].data[0]  # Remove extra dimension
    
        # Convert to standard complex numpy array
        epoch_complex = np.asarray(epoch, dtype=np.complex128)
    
        for region, electrodes in electrode_clusters.items():
            electrode_indices = [power.info['ch_names'].index(e) for e in electrodes]
    
            if not electrode_indices:
                print(f"No valid electrodes found for {region}. Skipping.")
                continue
    
            for band, (fmin, fmax) in freq_bands.items():
                # Select the frequency band
                freq_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
                # print(f"Processing {region} for band {band}: freq_indices {freq_indices}")
    
                # Extract phase information for total power at t=0
                total_phase = np.angle(epoch_complex[np.ix_(electrode_indices, freq_indices, [stim_index])])
                avg_total_phase = np.mean(total_phase)
    
                # Store the extracted phase values in the DataFrame
                stim_df.at[i, f'{region}_{band}_Phase_Stim_Total'] = avg_total_phase
        
    ########## Calculate phase coherence (PLV) between regions for each #########
    #########  frequency band during baseline period  #########
    
    # Initialize new columns in the dataframe for each region, band, and time window
    for region in electrode_clusters:
        for band in freq_bands:
            for window in time_windows:
                stim_df[f'{region}_PLV_{band}_{window}'] = np.nan
    
    for region1 in electrode_clusters:
        for region2 in electrode_clusters:
            if region1 != region2:
                for band in freq_bands:
                    for window in time_windows:
                        stim_df[f'{region1}_{region2}_PLV_{band}_{window}'] = np.nan
    
    for i in range(len(power)):
        epoch = power[i].data[0]  # Remove extra dimension
    
        # Convert to standard complex numpy array
        epoch_complex = np.asarray(epoch, dtype=np.complex128)
    
        for region, electrodes in electrode_clusters.items():
            electrode_indices = [power.info['ch_names'].index(e) for e in electrodes]
    
            for band, (fmin, fmax) in freq_bands.items():
                freq_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    
                for window, (tmin, tmax) in time_windows.items():
                    time_indices = np.where((power.times >= tmin) & (power.times <= tmax))[0]
    
                    plvs_within = []
                    for e1 in range(len(electrode_indices)):
                        for e2 in range(e1 + 1, len(electrode_indices)):
                            phase_diff = np.angle(epoch_complex[electrode_indices[e1], freq_indices, :][:, time_indices]) - \
                                         np.angle(epoch_complex[electrode_indices[e2], freq_indices, :][:, time_indices])
                            plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=1)).mean()
                            plvs_within.append(plv)
    
                    avg_plv_within = np.mean(plvs_within)
                    stim_df.at[i, f'{region}_PLV_{band}_{window}'] = avg_plv_within
    
        for region1, electrodes1 in electrode_clusters.items():
            for region2 in electrode_clusters:
                if region1 != region2:
                    electrode_indices1 = [power.info['ch_names'].index(e) for e in electrodes1]
                    electrode_indices2 = [power.info['ch_names'].index(e) for e in electrode_clusters[region2]]
    
                    for band, (fmin, fmax) in freq_bands.items():
                        freq_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    
                        for window, (tmin, tmax) in time_windows.items():
                            time_indices = np.where((power.times >= tmin) & (power.times <= tmax))[0]
    
                            total_phase1 = np.angle(epoch_complex[np.ix_(electrode_indices1, freq_indices, time_indices)])
                            total_phase2 = np.angle(epoch_complex[np.ix_(electrode_indices2, freq_indices, time_indices)])
    
                            avg_total_phase1 = np.mean(total_phase1, axis=0)
                            avg_total_phase2 = np.mean(total_phase2, axis=0)
    
                            phase_diff_total = avg_total_phase1 - avg_total_phase2
                            plv_total = np.abs(np.mean(np.exp(1j * phase_diff_total), axis=1)).mean()
    
                            stim_df.at[i, f'{region1}_{region2}_PLV_{band}_{window}'] = plv_total

    # Del data to save memory
    del power
    gc.collect()

    # Total power for each quartile
    power = mne.time_frequency.tfr_morlet(epochs_stim, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=False, decim=1, n_jobs=1, average=False)
    
    # Non-phase locked power
    erp = epochs_stim.average()
    NPL = epochs_stim.copy()
    for i in np.arange(0,np.size(epochs_stim._data,0),1): #pour chaque epoch
        NPL._data[i,:160,:] = NPL._data[i,:160,:] - erp._data[:,:]
    power_NPL = mne.time_frequency.tfr_morlet(NPL, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=False, decim=1, n_jobs=1, average=False)
    
    # Calculate aperiodic values for the pre-stim period using FOOOF
    # Initialize columns in the dataframe
    for cluster in electrode_clusters:
        stim_df[f'{cluster}_prestim_offset'] = np.nan
        stim_df[f'{cluster}_prestim_exponent'] = np.nan
        
    # Function to compute FOOOF for each epoch and cluster
    def compute_fooof(power, epochs, df, clusters, freq_range):
        for i, epoch in enumerate(epochs):
            for cluster, electrodes in clusters.items():
                # Intersect electrodes with actual channel names to avoid dimension mismatch
                valid_electrodes = list(set(electrodes) & set(epochs.info['ch_names']))
                if not valid_electrodes:
                    continue
    
                # Get the power for the specified electrodes and epoch
                idx = [epochs.info['ch_names'].index(ch) for ch in valid_electrodes]
                psds = power.data[i, idx, :, :].mean(axis=0)
                avg_psds = psds.mean(axis=1)  # Average across time

                trimmed_freqs, trimmed_psds = trim_spectrum(freqs, avg_psds, freq_range)
                
                # Initialize and fit the FOOOF model
                fm = FOOOF()
                fm.fit(trimmed_freqs, trimmed_psds)
    
                # Store the aperiodic parameters in the dataframe
                df.loc[i, f'{cluster}_prestim_offset'] = fm.aperiodic_params_[0]
                df.loc[i, f'{cluster}_prestim_exponent'] = fm.aperiodic_params_[1]
                
    # Compute FOOOF parameters for total power
    compute_fooof(power, epochs_stim, stim_df, electrode_clusters, [2,40])
    
    # Generate list of new column names
    new_columns = []
    # Initialize columns for peak and weighted peak frequencies
    for region in electrode_clusters:
        for band in freq_bands:
            new_columns.append(f'{region}_Peak_{band}_Freq_Total')
            new_columns.append(f'{region}_Weighted_{band}_Freq_Total')
            
    # Initialize new columns in the DataFrame
    stim_df = pd.concat([stim_df, pd.DataFrame(columns=new_columns)], axis=1)
    stim_df[new_columns] = np.nan
    
    ########## Calculate peak and weighted peak frequencies for the baseline period #########
    
    for i in range(len(power)):
        epoch = power[i].data[0]
        for region, electrodes in electrode_clusters.items():
            electrode_indices = [power.info['ch_names'].index(e) for e in electrodes]
            for band, (fmin, fmax) in freq_bands.items():
                # Select the frequency band
                freq_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
                
                # Extract power information for total power
                total_power = epoch[np.ix_(electrode_indices, freq_indices, baseline_indices)]
                baseline_total_power = total_power.mean(axis=2).mean(axis=0)
                
                # Peak frequency for total power
                peak_total_freq = freqs[freq_indices][np.argmax(baseline_total_power)]
                
                # Weighted peak frequency for total power
                weighted_total_freq = np.sum(freqs[freq_indices] * baseline_total_power) / np.sum(baseline_total_power)
                
                # Store in DataFrame
                stim_df.at[i, f'{region}_Peak_{band}_Freq_Total'] = peak_total_freq
                stim_df.at[i, f'{region}_Weighted_{band}_Freq_Total'] = weighted_total_freq
    
    ########## Trial-by-trial normalisation #########
    
    # # For trial by trial classic norm = using trial baseline
    power.apply_baseline(baseline=(-0.2, 0.0), mode='logratio')
    power_NPL.apply_baseline(baseline=(-0.2, 0.0), mode='logratio')

    # # For decibels
    power.data *=10
    power_NPL.data *=10
    
    # Generate list of new column names
    new_columns = []
    for region in electrode_clusters:
        for band in freq_bands:
            for period in time_windows:
                new_columns.append(f'{region}_{band}_{period}_power_Total')
                new_columns.append(f'{region}_{band}_{period}_power_NPL')
    
    # Initialize new columns in the DataFrame
    stim_df = pd.concat([stim_df, pd.DataFrame(columns=new_columns)], axis=1)
    stim_df[new_columns] = np.nan
    
    ########## Calculate power for each frequency band during specified periods #########
    for i in range(len(power)):
        if i >= len(stim_df):
            break  # Ensure we do not exceed the size of stim_df
    
        epoch = power[i].data[0]  # Remove extra dimension
        epoch_npl = power_NPL[i].data[0]  # Remove extra dimension
    
        for region, electrodes in electrode_clusters.items():
            electrode_indices = [power.info['ch_names'].index(e) for e in electrodes]
    
            if not electrode_indices:
                print(f"No valid electrodes found for {region}. Skipping.")
                continue
    
            for band, (fmin, fmax) in freq_bands.items():
                # Select the frequency band
                freq_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
                # print(f"Processing {region} for band {band}: freq_indices {freq_indices}")
    
                for period, (tmin, tmax) in time_windows.items():
                    # Select the time window
                    time_indices = np.where((power.times >= tmin) & (power.times <= tmax))[0]
    
                    # Extract power information for total power during specified period
                    total_power = epoch[np.ix_(electrode_indices, freq_indices, time_indices)]
                    power_avg_total = total_power.mean()
    
                    # Extract power information for non-phase locked power during specified period
                    induced_power = epoch_npl[np.ix_(electrode_indices, freq_indices, time_indices)]
                    power_avg_induced = induced_power.mean()
    
                    # Store the extracted power in the DataFrame
                    stim_df.at[i, f'{region}_{band}_{period}_power_Total'] = power_avg_total
                    stim_df.at[i, f'{region}_{band}_{period}_power_NPL'] = power_avg_induced

    stim_df['Theta/(Alpha+Beta)'] = stim_df['Fronto-central_Theta_post_150_300ms_power_NPL'] / ( stim_df['Occipital_Alpha_post_150_300ms_power_NPL'] + stim_df['Occipital_Beta_post_150_300ms_power_NPL'])
    # Calculation of average for figures 
    # For total power
    # Get the indices for each type of preceding stimulus
    indices_v = stim_df[stim_df['Preceding stim'] == 'V'].index
    indices_a = stim_df[stim_df['Preceding stim'] == 'A'].index
    
    # Select the epochs for each category
    tfr_v = power[indices_v]
    tfr_a = power[indices_a]
    
    # Calculate the average TFR for each category
    tfr_v_avg = tfr_v.average()
    tfr_a_avg = tfr_a.average()
    
    del tfr_v, tfr_a
    gc.collect()
    
    # For non-phase locked power
    # Select the epochs for each category
    tfr_NPL_v = power_NPL[indices_v]
    tfr_NPL_a = power_NPL[indices_a]
    
    # Calculate the average TFR for each category
    tfr_NPL_v_avg = tfr_NPL_v.average()
    tfr_NPL_a_avg = tfr_NPL_a.average()
    
    del tfr_NPL_v, tfr_NPL_a
    gc.collect()
    
    Subject = str(Subject)
    
    # Add a new column 'subject_number' and set its value to ix for all rows
    stim_df['subject_number'] = Subject

    stim_df.to_csv('Data_trials_Visual_Subject_'+Subject+'.csv', index=False)

    with open('Power_Visual_Preceded_Visual_Subject_'+Subject+'.pkl', 'wb') as fp:
        pickle.dump([tfr_v_avg, tfr_NPL_v_avg], fp)    
    with open('Power_Visual_Preceded_Audio_Subject_'+Subject+'.pkl', 'wb') as fp:
        pickle.dump([tfr_a_avg,tfr_NPL_a_avg], fp)

    del tfr_v_avg, tfr_NPL_v_avg, tfr_a_avg, tfr_NPL_a_avg
    gc.collect()
        
    # # Average TFR between 'fast' and 'slow' RT for Preceding stim = 'V'
    # # Step 1: Separate the trials by 'Preceding stim' categories
    # category_df = stim_df[stim_df['Preceding stim'] == 'V']
    # category_df['Type'] = pd.qcut(category_df['Reaction time'], q=2, labels=['fast', 'slow'])

    # index_fast = category_df[(category_df['Preceding stim'] == 'V') & (category_df['Type'] == 'fast')].index
    # index_slow = category_df[(category_df['Preceding stim'] == 'V') & (category_df['Type'] == 'slow')].index

    # tfr_fast = power[index_fast]
    # tfr_slow = power[index_slow]
    
    # tfr_NPL_fast = power_NPL[index_fast]
    # tfr_NPL_slow = power_NPL[index_slow]
    
    # tfr_fast_avg = tfr_fast.average()
    # tfr_slow_avg = tfr_slow.average()
    
    # tfr_NPL_fast_avg = tfr_NPL_fast.average()
    # tfr_NPL_slow_avg = tfr_NPL_slow.average()
    
    # with open('Power_Visual_Visual_Fast_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_fast_avg, tfr_NPL_fast_avg], fp)    
    # with open('Power_Visual_Visual_Slow_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_slow_avg,tfr_NPL_slow_avg], fp)
        
    # del tfr_fast, tfr_slow, tfr_NPL_fast, tfr_NPL_slow
    # gc.collect()
        
    # # Average between long and short ISI for repeated trials
    
    # index_short = category_df[(category_df['Preceding stim'] == 'V') & (category_df['ISI category'] == 'short')].index
    # index_long = category_df[(category_df['Preceding stim'] == 'V') & (category_df['ISI category'] == 'long')].index
    
    # tfr_short = power[index_short]
    # tfr_long = power[index_long]
    
    # tfr_NPL_short = power_NPL[index_short]
    # tfr_NPL_long = power_NPL[index_long]
    
    # tfr_short_avg = tfr_short.average()
    # tfr_long_avg = tfr_long.average()
    
    # tfr_NPL_short_avg = tfr_NPL_short.average()
    # tfr_NPL_long_avg = tfr_NPL_long.average()
    
    # with open('Power_Visual_Visual_ShortISI_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_short_avg, tfr_NPL_short_avg], fp)    
    # with open('Power_Visual_Visual_LongISI_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_long_avg,tfr_NPL_long_avg], fp)
        

    # # Average TFR between quartile for all visual stim
    # # Step 1: Separate the trials by 'Preceding stim' categories
    # stim_df['Quartile'] = pd.qcut(stim_df['Reaction time'], q=4, labels=['Q1', 'Q2','Q3','Q4'])

    # index_Q1 = stim_df[stim_df['Quartile'] == 'Q1'].index
    # index_Q2 = stim_df[stim_df['Quartile'] == 'Q2'].index
    # index_Q3 = stim_df[stim_df['Quartile'] == 'Q3'].index
    # index_Q4 = stim_df[stim_df['Quartile'] == 'Q4'].index

    # tfr_Q1 = power[index_Q1]
    # tfr_Q2 = power[index_Q2]
    # tfr_Q3 = power[index_Q3]
    # tfr_Q4 = power[index_Q4]
    
    # tfr_NPL_Q1 = power_NPL[index_Q1]
    # tfr_NPL_Q2 = power_NPL[index_Q2]
    # tfr_NPL_Q3 = power_NPL[index_Q3]
    # tfr_NPL_Q4 = power_NPL[index_Q4]
    
    # tfr_Q1_avg = tfr_Q1.average()
    # tfr_Q2_avg = tfr_Q2.average()
    # tfr_Q3_avg = tfr_Q3.average()
    # tfr_Q4_avg = tfr_Q4.average()
    
    # tfr_NPL_Q1_avg = tfr_NPL_Q1.average()
    # tfr_NPL_Q2_avg = tfr_NPL_Q2.average()
    # tfr_NPL_Q3_avg = tfr_NPL_Q3.average()
    # tfr_NPL_Q4_avg = tfr_NPL_Q4.average()
    
    # with open('Power_Visual_Q1_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q1_avg, tfr_NPL_Q1_avg], fp)    
    # with open('Power_Visual_Q2_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q2_avg,tfr_NPL_Q2_avg], fp)
    # with open('Power_Visual_Q3_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q3_avg, tfr_NPL_Q3_avg], fp)    
    # with open('Power_Visual_Q4_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q4_avg,tfr_NPL_Q4_avg], fp)
        
    # # Average TFR between ISI type all visual stim
    # # Step 1: Separate the trials by 'Preceding stim' categories

    # index_Q1 = stim_df[stim_df['ISI category2'] == 'short'].index
    # index_Q2 = stim_df[stim_df['ISI category2'] == 'med'].index
    # index_Q3 = stim_df[stim_df['ISI category2'] == 'long'].index

    # tfr_Q1 = power[index_Q1]
    # tfr_Q2 = power[index_Q2]
    # tfr_Q3 = power[index_Q3]
    
    # tfr_NPL_Q1 = power_NPL[index_Q1]
    # tfr_NPL_Q2 = power_NPL[index_Q2]
    # tfr_NPL_Q3 = power_NPL[index_Q3]
    
    # tfr_Q1_avg = tfr_Q1.average()
    # tfr_Q2_avg = tfr_Q2.average()
    # tfr_Q3_avg = tfr_Q3.average()
    
    # tfr_NPL_Q1_avg = tfr_NPL_Q1.average()
    # tfr_NPL_Q2_avg = tfr_NPL_Q2.average()
    # tfr_NPL_Q3_avg = tfr_NPL_Q3.average()
    
    # with open('Power_Visual_Short_ISI_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q1_avg, tfr_NPL_Q1_avg], fp)    
    # with open('Power_Visual_Medium_ISI_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q2_avg,tfr_NPL_Q2_avg], fp)
    # with open('Power_Visual_Long_ISI_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q3_avg, tfr_NPL_Q3_avg], fp)    
        
    # # Average TFR for each type of preceding stim depending on the ISI type
    # # Step 1: Separate the trials by 'Preceding stim' categories

    # index_visual_short = stim_df[(stim_df['Preceding stim'] == 'V') & (stim_df['ISI category'] == 'short')].index
    # index_visual_long = stim_df[(stim_df['Preceding stim'] == 'V') & (stim_df['ISI category'] == 'long')].index

    # index_audio_short = stim_df[(stim_df['Preceding stim'] == 'A') & (stim_df['ISI category'] == 'short')].index
    # index_audio_long = stim_df[(stim_df['Preceding stim'] == 'A') & (stim_df['ISI category'] == 'long')].index

    # index_audiovisual_short = stim_df[(stim_df['Preceding stim'] == 'AV') & (stim_df['ISI category'] == 'short')].index
    # index_audiovisual_long = stim_df[(stim_df['Preceding stim'] == 'AV') & (stim_df['ISI category'] == 'long')].index

    # tfr_Q1 = power[index_visual_short]
    # tfr_Q2 = power[index_visual_long]
    
    # tfr_Q3 = power[index_audio_short]
    # tfr_Q4 = power[index_audio_long]
    
    # tfr_Q5 = power[index_audiovisual_short]
    # tfr_Q6 = power[index_audiovisual_long]
        
    # tfr_NPL_Q1 = power_NPL[index_visual_short]
    # tfr_NPL_Q2 = power_NPL[index_visual_long]
    
    # tfr_NPL_Q3 = power_NPL[index_audio_short]
    # tfr_NPL_Q4 = power_NPL[index_audio_long]
    
    # tfr_NPL_Q5 = power_NPL[index_audiovisual_short]
    # tfr_NPL_Q6 = power_NPL[index_audiovisual_long]
    
    # tfr_Q1_avg = tfr_Q1.average()
    # tfr_Q2_avg = tfr_Q2.average()
    # tfr_Q3_avg = tfr_Q3.average()
    # tfr_Q4_avg = tfr_Q4.average()
    # tfr_Q5_avg = tfr_Q5.average()
    # tfr_Q6_avg = tfr_Q6.average()
    
    # tfr_NPL_Q1_avg = tfr_NPL_Q1.average()
    # tfr_NPL_Q2_avg = tfr_NPL_Q2.average()
    # tfr_NPL_Q3_avg = tfr_NPL_Q3.average()
    # tfr_NPL_Q4_avg = tfr_NPL_Q4.average()
    # tfr_NPL_Q5_avg = tfr_NPL_Q5.average()
    # tfr_NPL_Q6_avg = tfr_NPL_Q6.average()
    
    # with open('Power_Visual_Visual_Short_ISI_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q1_avg, tfr_NPL_Q1_avg], fp)    
    # with open('Power_Visual_Visual_Long_ISI_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q2_avg,tfr_NPL_Q2_avg], fp)
        
    # with open('Power_Audio_Visual_Short_ISI_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q3_avg, tfr_NPL_Q3_avg], fp) 
    # with open('Power_Audio_Visual_Long_ISI_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q4_avg, tfr_NPL_Q4_avg], fp)    
        
    # with open('Power_Audiovisual_Visual_Short_ISI_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q5_avg,tfr_NPL_Q5_avg], fp)
    # with open('Power_Audiovisual_Visual_Long_ISI_Subject_'+Subject+'.pkl', 'wb') as fp:
    #     pickle.dump([tfr_Q6_avg, tfr_NPL_Q6_avg], fp) 

        
    #%%
    
    
    
    

    
    
