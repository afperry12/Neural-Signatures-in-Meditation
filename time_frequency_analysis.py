from pathlib import Path
import numpy as np
import mne
from mne.time_frequency import tfr_morlet
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
import pandas as pd

bids_root = Path(__file__).parent / "ds001787-download"

# Load participant information to separate experts from novices
participants_file = bids_root / "participants.tsv"
participants_df = pd.read_csv(participants_file, sep='\t')
expert_subs = participants_df[participants_df['group'] == 'expert']['participant_id'].str.split('-').str[1].tolist()
novice_subs = participants_df[participants_df['group'] == 'novice']['participant_id'].str.split('-').str[1].tolist()

print(f"Found {len(expert_subs)} expert subjects: {expert_subs}")
print(f"Found {len(novice_subs)} novice subjects: {novice_subs}")

# Define ROIs according to the Brandmeyer & Delorme (2016) paper
roi_fm_theta = ['Fz', 'FCz', 'Cz']  # Frontal-midline theta ROI
roi_central_alpha = ['C3', 'C4', 'CP3', 'CP4']  # Somatosensory/central alpha ROI
roi_posterior_alpha = ['P3', 'P4', 'PO3', 'PO4', 'O1', 'O2', 'Pz', 'POz', 'Oz']  # Additional posterior ROI

# Define frequency bands
freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Define non-EEG channel patterns to exclude
def filter_eeg_channels(raw):
    """
    Filter out non-EEG channels and convert data to microvolts
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw MNE object
        
    Returns:
    --------
    raw_eeg : mne.io.Raw
        Filtered raw object with only EEG channels, scaled to microvolts
    """
    print("Filtering channels and scaling to microvolts...")
    
    # Ensure data is loaded
    if not raw.preload:
        print("Loading raw data...")
        raw.load_data()
    
    # Define patterns for non-EEG channels
    bad_patterns = ['GSR', 'EXG', 'ACC', 'Temp', 'PLET', 'RESP', 'Status', 'EOG', 'Erg']
    
    # Create a list of indices for EEG channels only
    eeg_idx = [i for i, ch in enumerate(raw.ch_names) 
              if not any(pattern.lower() in ch.lower() for pattern in bad_patterns)]
    
    if len(eeg_idx) == 0:
        print("Warning: No EEG channels found after filtering!")
        # Fall back to just using the pick_types approach
        raw_eeg = raw.copy().pick_types(eeg=True, stim=False, exclude='bads')
    else:
        # Pick only the EEG channels
        raw_eeg = raw.copy().pick(eeg_idx)
        print(f"Kept {len(eeg_idx)} EEG channels after filtering out auxiliary channels")
    
    # Convert all data to microvolts
    print("Converting all channels to microvolts...")
    raw_eeg.apply_function(lambda x: x * 1e6, picks='all', channel_wise=False)
    
    return raw_eeg

def run_time_frequency_analysis(output_dir):
    """
    Perform time-frequency analysis on meditation data to visualize
    how brain rhythms change over time, especially before responses.
    Separates analysis by expertise group.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save output figures
    """
    # Create main output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Define frequencies for wavelet analysis
    freqs = np.logspace(np.log10(4), np.log10(40), 30)  # Log spaced frequencies from 4-40 Hz
    n_cycles = freqs / 2.  # Use fewer cycles for lower frequencies
    
    # Process each expertise group separately
    for expertise, subjects in [('expert', expert_subs), ('novice', novice_subs)]:
        print(f"\n=== Processing {expertise} subjects ===")
        
        # Create expertise-specific output directory
        expertise_dir = output_path / expertise
        expertise_dir.mkdir(exist_ok=True)
        
        # Initialize dictionaries to store average power values
        conc_power_values = {band: [] for band in freq_bands}
        wander_power_values = {band: [] for band in freq_bands}
        
        # Track successful analyses for grand averages
        successful_subjects = 0
        
        # Process each subject
        for sub in subjects:
            print(f"\nProcessing sub-{sub}...")
            ses_paths = sorted((bids_root / f"sub-{sub}").glob('ses-*'))
            sessions = [p.name.split('-')[1] for p in ses_paths] or [None]
            
            for ses in sessions:
                print(f"Processing sub-{sub}, ses-{ses}...")
                bp = BIDSPath(
                    root=bids_root,
                    subject=sub,
                    session=ses,
                    task='meditation',
                    datatype='eeg',
                    suffix='eeg'
                )
                
                try:
                    # Read the raw data
                    raw = read_raw_bids(bp, verbose=False)
                    raw.load_data()
                    
                    # Apply proper filtering and scaling
                    raw = filter_eeg_channels(raw)
                    
                    # Pick only EEG channels for analysis
                    raw.pick_types(eeg=True, exclude='bads')
                    print(f"   ↳ Using {len(raw.ch_names)} EEG channels for analysis.")
                    
                    # Skip if we don't have enough EEG channels
                    if len(raw.ch_names) < 5:
                        print(f"   ↳ Not enough EEG channels (need at least 5, found {len(raw.ch_names)})")
                        continue
                    
                    # Apply a bandpass filter
                    raw.filter(l_freq=1, h_freq=45)
                    
                    # Read events
                    events, event_id = mne.events_from_annotations(raw)
                    
                    # Find concentration (low values 1-2) and mind wandering (high values 4-8) events
                    concentration_events = events[events[:, 2] <= 2]  # 1-2 → Concentration
                    mind_wandering_events = events[events[:, 2] >= 4]  # 4-8 → Mind-wandering
                    
                    if len(concentration_events) < 3 or len(mind_wandering_events) < 3:
                        print(f"   ↳ skipping: not enough concentration or mind wandering events (need at least 3 of each)")
                        continue
                    
                    print(f"   ↳ Found {len(concentration_events)} concentration and {len(mind_wandering_events)} mind wandering events")
                    
                    # Create epochs with longer time window
                    conc_epochs = mne.Epochs(raw, concentration_events, tmin=-90.0, tmax=-0.1, 
                                            baseline=None, preload=True)
                                            
                    wander_epochs = mne.Epochs(raw, mind_wandering_events, tmin=-90.0, tmax=-0.1, 
                                              baseline=None, preload=True)
                    
                    print(f"   ↳ Created epochs with time range {conc_epochs.times[0]:.1f}s to {conc_epochs.times[-1]:.1f}s")
                    
                    # For time-frequency analysis, we may need to downsample extremely long epochs
                    # to avoid excessive memory usage
                    epochs_tf_conc = conc_epochs.copy().resample(128, npad='auto')
                    epochs_tf_wander = wander_epochs.copy().resample(128, npad='auto')
                    
                    # Apply time-frequency decomposition
                    conc_power = tfr_morlet(epochs_tf_conc, freqs=freqs, n_cycles=n_cycles, 
                                           use_fft=True, return_itc=False, decim=4, 
                                           n_jobs=1, average=True)
                                       
                    wander_power = tfr_morlet(epochs_tf_wander, freqs=freqs, n_cycles=n_cycles, 
                                             use_fft=True, return_itc=False, decim=4, 
                                             n_jobs=1, average=True)
                    
                    # Convert power to dB (10*log10)
                    # Add a small epsilon to avoid log of zero
                    epsilon = 1e-10
                    conc_power_data = 10 * np.log10(conc_power.data + epsilon)
                    wander_power_data = 10 * np.log10(wander_power.data + epsilon)
                    
                    # Use the converted data
                    conc_power.data = conc_power_data
                    wander_power.data = wander_power_data
                    
                    # Apply baseline normalization
                    # For very long epochs, use a different approach
                    time_mask = (conc_power.times >= -90) & (conc_power.times <= -80)
                    if np.any(time_mask):
                        print(f"   ↳ Using baseline period from -90s to -80s")
                        conc_power.apply_baseline(baseline=(-90, -80), mode='logratio')
                        wander_power.apply_baseline(baseline=(-90, -80), mode='logratio')
                    else:
                        print(f"   ↳ Could not apply baseline - using first 10s of epoch instead")
                        conc_power.apply_baseline(baseline=(None, None), mode='mean')
                        wander_power.apply_baseline(baseline=(None, None), mode='mean')
                    
                    # Store data for each frequency band for this subject
                    for band_name, (fmin, fmax) in freq_bands.items():
                        # Extract data for this frequency band
                        freq_mask = (freqs >= fmin) & (freqs <= fmax)
                        band_freqs = freqs[freq_mask]
                        
                        if len(band_freqs) > 0:
                            # Average over frequencies in the band and over channels
                            conc_band_power = conc_power.data[:, freq_mask, :].mean(axis=(0, 1))
                            wander_band_power = wander_power.data[:, freq_mask, :].mean(axis=(0, 1))
                            
                            conc_power_values[band_name].append(conc_band_power)
                            wander_power_values[band_name].append(wander_band_power)
                    
                    # Individual time-frequency plots for this subject
                    # Generate plots for ROIs of interest
                    for roi_name, roi_channels in [('frontal_midline', roi_fm_theta), 
                                                  ('central', roi_central_alpha),
                                                  ('posterior', roi_posterior_alpha)]:
                        
                        # Get available channels from the ROI
                        available_channels = [ch for ch in roi_channels if ch in conc_epochs.ch_names]
                        
                        if not available_channels:
                            print(f"   ↳ No channels available for {roi_name} ROI")
                            continue
                            
                        print(f"   ↳ Generating TF plots for {roi_name} ROI with channels: {available_channels}")
                        
                        # Get channel indices
                        ch_indices = [conc_epochs.ch_names.index(ch) for ch in available_channels]
                        
                        # For each relevant band for this ROI
                        for band_name, band_range in freq_bands.items():
                            # Skip non-relevant bands
                            if (roi_name == 'frontal_midline' and band_name != 'theta') or \
                               (roi_name in ['central', 'posterior'] and band_name != 'alpha'):
                                continue
                            
                            # Create time-frequency plots
                            fig, ax = plt.subplots(figsize=(12, 6))
                            conc_power.plot([ch_indices[0]], baseline=None, mode='mean', 
                                          tmin=-90, tmax=0, 
                                          fmin=band_range[0], fmax=band_range[1],
                                          title=f'Sub-{sub} Concentration {band_name.capitalize()} Power - {roi_name}',
                                          axes=ax, show=False)
                            plt.tight_layout()
                            plt.savefig(expertise_dir / f'sub-{sub}_{roi_name}_{band_name}_concentration_tfr.png', dpi=150)
                            plt.close()
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            wander_power.plot([ch_indices[0]], baseline=None, mode='mean', 
                                            tmin=-90, tmax=0, 
                                            fmin=band_range[0], fmax=band_range[1],
                                            title=f'Sub-{sub} Mind Wandering {band_name.capitalize()} Power - {roi_name}',
                                            axes=ax, show=False)
                            plt.tight_layout()
                            plt.savefig(expertise_dir / f'sub-{sub}_{roi_name}_{band_name}_mind_wandering_tfr.png', dpi=150)
                            plt.close()
                    
                    successful_subjects += 1
                    print(f"   ↳ Time-frequency analysis completed successfully")
                    
                except Exception as e:
                    print(f"   ↳ Error processing: {str(e)}")
                    continue
        
        # Check if we have enough data for grand average plots
        if successful_subjects == 0:
            print(f"No successful subject analyses for {expertise} group. Cannot create grand average plots.")
            # Create placeholder files to prevent dashboard errors
            for band_name in freq_bands:
                # Create empty plots with error message
                plt.figure(figsize=(12, 6))
                plt.text(0.5, 0.5, f"No data available for {band_name} band analysis - {expertise} group", 
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.axis('off')
                plt.savefig(expertise_dir / f'{expertise}_grand_average_{band_name}_power.png', dpi=300)
                plt.savefig(expertise_dir / f'{expertise}_difference_{band_name}_power.png', dpi=300)
                plt.savefig(expertise_dir / f'{expertise}_t_values_{band_name}_power.png', dpi=300)
                plt.close()
                
                # Also save to the main output directory
                plt.figure(figsize=(12, 6))
                plt.text(0.5, 0.5, f"No data available for {band_name} band analysis - {expertise} group", 
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.axis('off')
                plt.savefig(output_path / f'{expertise}_grand_average_{band_name}_power.png', dpi=300)
                plt.close()
                
                # Create empty CSV files
                df = pd.DataFrame({
                    'Time': np.linspace(-90.0, 0.0, 20),
                    'Note': ['No data available' for _ in range(20)]
                })
                df.to_csv(expertise_dir / f'{expertise}_{band_name}_power_results.csv', index=False)
                df.to_csv(output_path / f'{expertise}_{band_name}_power_results.csv', index=False)
            
            continue
        
        # Now create grand average plots for each frequency band
        for band_name in freq_bands:
            if not conc_power_values[band_name]:
                print(f"No data for {band_name} band in {expertise} group, creating placeholder")
                # Create placeholder files
                plt.figure(figsize=(12, 6))
                plt.text(0.5, 0.5, f"No data available for {band_name} band analysis - {expertise} group", 
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.axis('off')
                plt.savefig(expertise_dir / f'{expertise}_grand_average_{band_name}_power.png', dpi=300)
                plt.savefig(expertise_dir / f'{expertise}_difference_{band_name}_power.png', dpi=300)
                plt.savefig(expertise_dir / f'{expertise}_t_values_{band_name}_power.png', dpi=300)
                plt.close()
                
                # Also save to the main output directory
                plt.figure(figsize=(12, 6))
                plt.text(0.5, 0.5, f"No data available for {band_name} band analysis - {expertise} group", 
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.axis('off')
                plt.savefig(output_path / f'{expertise}_grand_average_{band_name}_power.png', dpi=300)
                plt.close()
                
                # Create empty CSV files
                df = pd.DataFrame({
                    'Time': np.linspace(-90.0, 0.0, 20),
                    'Note': ['No data available' for _ in range(20)]
                })
                df.to_csv(expertise_dir / f'{expertise}_{band_name}_power_results.csv', index=False)
                df.to_csv(output_path / f'{expertise}_{band_name}_power_results.csv', index=False)
                continue
            
            # Get the time points from the first subject's data
            first_data = conc_power_values[band_name][0]
            times = np.linspace(-90.0, 0.0, first_data.shape[0])
            
            # Convert lists to arrays for easier manipulation
            conc_data = np.array(conc_power_values[band_name])
            wander_data = np.array(wander_power_values[band_name])
            
            # Calculate mean and standard error across subjects
            conc_mean = np.mean(conc_data, axis=0)
            conc_sem = np.std(conc_data, axis=0) / np.sqrt(conc_data.shape[0])
            
            wander_mean = np.mean(wander_data, axis=0)
            wander_sem = np.std(wander_data, axis=0) / np.sqrt(wander_data.shape[0])
            
            # Plot the time course for this frequency band
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Print min/max values to verify data range
            print(f"DEBUG: {expertise} group - {band_name} band data ranges:")
            print(f"  Concentration: min={conc_mean.min():.4f}, max={conc_mean.max():.4f}")
            print(f"  Mind Wandering: min={wander_mean.min():.4f}, max={wander_mean.max():.4f}")
            
            # Concentration
            ax.plot(times, conc_mean, 'b-', linewidth=2, label='Concentration')
            ax.fill_between(times, conc_mean - conc_sem, conc_mean + conc_sem, 
                            color='blue', alpha=0.2)
            
            # Mind wandering
            ax.plot(times, wander_mean, 'r-', linewidth=2, label='Mind Wandering')
            ax.fill_between(times, wander_mean - wander_sem, wander_mean + wander_sem, 
                            color='red', alpha=0.2)
            
            # Add details to plot
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Response')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Power (dB)')
            ax.set_title(f'{expertise.capitalize()}: Grand Average {band_name.capitalize()} Power')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate data range with a bit of padding
            data_min = min(conc_mean.min(), wander_mean.min())
            data_max = max(conc_mean.max(), wander_mean.max())

            # Check for NaN or Inf values
            if np.isnan(data_min) or np.isnan(data_max) or np.isinf(data_min) or np.isinf(data_max):
                print(f"  Warning: NaN or Inf detected in data range. Using default plot limits.")
                y_min, y_max = -1.0, 1.0  # Default values
            else:
                # Add some buffer to make the plot look better
                y_range = data_max - data_min
                y_buffer = y_range * 0.1  # 10% padding
                
                if y_range < 1:  # If range is very small
                    y_buffer = 1  # Use fixed buffer
                
                y_min = data_min - y_buffer
                y_max = data_max + y_buffer

            # Set y-axis limits explicitly to ensure data is visible
            ax.set_ylim(y_min, y_max)

            # Print y-axis limits
            print(f"  Plot y-axis limits: {y_min:.4f} to {y_max:.4f}")
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(expertise_dir / f'{expertise}_grand_average_{band_name}_power.png', dpi=300)
            plt.savefig(output_path / f'{expertise}_grand_average_{band_name}_power.png', dpi=300)
            plt.close()
            
            # Also create a difference plot (concentration - mind wandering)
            fig, ax = plt.subplots(figsize=(12, 6))
            diff_mean = conc_mean - wander_mean
            
            # Calculate standard error of the difference
            diff_sem = np.sqrt(conc_sem**2 + wander_sem**2)
            
            # DEBUG: Print diff min/max values
            print(f"  Difference: min={diff_mean.min():.4f}, max={diff_mean.max():.4f}")
            
            ax.plot(times, diff_mean, 'g-', linewidth=2)
            ax.fill_between(times, diff_mean - diff_sem, diff_mean + diff_sem, 
                            color='green', alpha=0.2)
            
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Response')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Power Difference (dB)')
            ax.set_title(f'{expertise.capitalize()}: Difference in {band_name.capitalize()} Power (Conc-Wand)')
            ax.grid(True, alpha=0.3)
            
            # Calculate better y limits for difference plot
            diff_min = diff_mean.min()
            diff_max = diff_mean.max()

            # Check for NaN or Inf values
            if np.isnan(diff_min) or np.isnan(diff_max) or np.isinf(diff_min) or np.isinf(diff_max):
                print(f"  Warning: NaN or Inf detected in difference data. Using default plot limits.")
                diff_y_min, diff_y_max = -1.0, 1.0  # Default values
            else:
                diff_range = diff_max - diff_min
                diff_buffer = max(diff_range * 0.1, 0.5)  # At least 0.5 dB buffer
                
                diff_y_min = diff_min - diff_buffer
                diff_y_max = diff_max + diff_buffer

            # Set y-axis limits for difference plot
            ax.set_ylim(diff_y_min, diff_y_max)

            # Debug - print diff y-axis limits
            print(f"  Diff plot y-axis limits: {diff_y_min:.4f} to {diff_y_max:.4f}")
            
            plt.tight_layout()
            plt.savefig(expertise_dir / f'{expertise}_difference_{band_name}_power.png', dpi=300)
            plt.savefig(output_path / f'{expertise}_difference_{band_name}_power.png', dpi=300)
            plt.close()
            
            print(f"Created grand average plots for {expertise} - {band_name} band")
            
            # Perform statistical analysis
            # Calculate t-values at each time point
            t_values = []
            p_values = []
            from scipy import stats
            
            for t in range(conc_data.shape[1]):
                t_val, p_val = stats.ttest_rel(conc_data[:, t], wander_data[:, t])
                t_values.append(t_val)
                p_values.append(p_val)
            
            t_values = np.array(t_values)
            p_values = np.array(p_values)
            
            # Plot t-values with significance markers
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Print t-values min/max
            print(f"  T-values: min={t_values.min():.4f}, max={t_values.max():.4f}")
            
            ax.plot(times, t_values, 'k-', linewidth=2)
            
            # Mark significant time points (p < 0.05)
            significant = p_values < 0.05
            if np.any(significant):
                ax.scatter(times[significant], t_values[significant], color='r', s=50, zorder=3)
            
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Response')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('t-value')
            ax.set_title(f'{expertise.capitalize()}: T-values for {band_name.capitalize()} Power Difference')
            ax.grid(True, alpha=0.3)
            
            # Calculate t-value range with padding
            t_min = t_values.min()
            t_max = t_values.max()

            # Check for NaN or Inf values in t-values
            if np.isnan(t_min) or np.isnan(t_max) or np.isinf(t_min) or np.isinf(t_max):
                print(f"  Warning: NaN or Inf detected in t-values. Using default plot limits.")
                t_y_min, t_y_max = -3.0, 3.0  # Default values
            else:
                t_range = t_max - t_min
                t_buffer = max(t_range * 0.1, 0.5)  # At least 0.5 buffer
                
                t_y_min = t_min - t_buffer
                t_y_max = t_max + t_buffer

            # Set y-axis limits for t-values plot
            ax.set_ylim(t_y_min, t_y_max)

            # Debug - print t-value y-axis limits
            print(f"  T-value plot y-axis limits: {t_y_min:.4f} to {t_y_max:.4f}")
            
            plt.tight_layout()
            plt.savefig(expertise_dir / f'{expertise}_t_values_{band_name}_power.png', dpi=300)
            plt.savefig(output_path / f'{expertise}_t_values_{band_name}_power.png', dpi=300)
            plt.close()
            
            # Save the data to CSV
            result_df = pd.DataFrame({
                'Time': times,
                'Concentration_Mean': conc_mean,
                'Concentration_SEM': conc_sem,
                'Mind_Wandering_Mean': wander_mean,
                'Mind_Wandering_SEM': wander_sem,
                'Difference_Mean': diff_mean,
                'Difference_SEM': diff_sem,
                't_value': t_values,
                'p_value': p_values
            })
            
            result_df.to_csv(expertise_dir / f'{expertise}_{band_name}_power_results.csv', index=False)
            result_df.to_csv(output_path / f'{expertise}_{band_name}_power_results.csv', index=False)
            
            print(f"Statistical analysis for {expertise} - {band_name} band completed")
        
        print(f"Time-frequency analysis completed for {expertise} group")
    
    # Create a combined visualization comparing expert vs novice
    # Focus on alpha and theta bands which are the most relevant according to the paper
    try:
        plt.figure(figsize=(14, 10))
        
        for band_idx, band_name in enumerate(['alpha', 'theta']):
            for exp_idx, expertise in enumerate(['expert', 'novice']):
                # Load data
                try:
                    result_df = pd.read_csv(output_path / f'{expertise}_{band_name}_power_results.csv')
                    
                    # Create subplot
                    ax = plt.subplot(2, 2, exp_idx * 2 + band_idx + 1)
                    
                    # Plot data
                    ax.plot(result_df['Time'], result_df['Concentration_Mean'], 'b-', linewidth=2, label='Concentration')
                    ax.fill_between(result_df['Time'], 
                                    result_df['Concentration_Mean'] - result_df['Concentration_SEM'],
                                    result_df['Concentration_Mean'] + result_df['Concentration_SEM'],
                                    color='blue', alpha=0.2)
                    
                    ax.plot(result_df['Time'], result_df['Mind_Wandering_Mean'], 'r-', linewidth=2, label='Mind Wandering')
                    ax.fill_between(result_df['Time'], 
                                    result_df['Mind_Wandering_Mean'] - result_df['Mind_Wandering_SEM'],
                                    result_df['Mind_Wandering_Mean'] + result_df['Mind_Wandering_SEM'],
                                    color='red', alpha=0.2)
                    
                    # Add details
                    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Power (dB)')
                    ax.set_title(f'{expertise.capitalize()}: {band_name.capitalize()} Power')
                    ax.grid(True, alpha=0.3)
                    
                    # Add legend only once
                    if exp_idx == 0 and band_idx == 0:
                        ax.legend()
                    
                except Exception as e:
                    print(f"Could not create combined plot for {expertise} - {band_name}: {e}")
                    ax = plt.subplot(2, 2, exp_idx * 2 + band_idx + 1)
                    ax.text(0.5, 0.5, f"No data for {expertise} - {band_name}", 
                            ha='center', va='center', fontsize=14)
                    ax.set_title(f'{expertise.capitalize()}: {band_name.capitalize()} Power')
                    ax.axis('off')
        
        plt.suptitle('Expert vs Novice: Alpha and Theta Power during Meditation', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'expert_vs_novice_tf_comparison.png', dpi=300)
        plt.close()
        print("Created expert vs novice comparison visualization")
        
    except Exception as e:
        print(f"Error creating combined visualization: {e}")
    
    print("Time-frequency analysis completed")
    return output_path

if __name__ == '__main__':
    # Output directory
    output_dir = "meditation_analysis_output/time_frequency"
    
    # Run time-frequency analysis
    output_path = run_time_frequency_analysis(output_dir)
    print(f"Time-frequency analysis complete. Results saved to: {output_path}") 