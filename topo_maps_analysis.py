from pathlib import Path
import re
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import pandas as pd

bids_root = Path(__file__).parent / "ds001787-download"

# Load participant information to separate experts from novices
participants_file = bids_root / "participants.tsv"
participants_df = pd.read_csv(participants_file, sep='\t')
expert_subs = participants_df[participants_df['group'] == 'expert']['participant_id'].str.split('-').str[1].tolist()
novice_subs = participants_df[participants_df['group'] == 'novice']['participant_id'].str.split('-').str[1].tolist()

print(f"Found {len(expert_subs)} expert subjects: {expert_subs}")
print(f"Found {len(novice_subs)} novice subjects: {novice_subs}")

# Define ROIs according to the Brandmeyer & Delorme (2018) paper
roi_fm_theta = ['Fz', 'FCz', 'Cz']  # Frontal-midline theta ROI
roi_central_alpha = ['C3', 'C4', 'CP3', 'CP4']  # Somatosensory/central alpha ROI
roi_posterior_alpha = ['P3', 'P4', 'PO3', 'PO4', 'O1', 'O2', 'Pz', 'POz', 'Oz']  # Additional posterior ROI

# Try to read channel name information from the dataset's channel file
def create_channel_mapping(bids_root):
    """Load standard EEG channel names from channels.tsv"""
    channel_file = bids_root / "task-meditation_channels.tsv"
    if channel_file.exists():
        print(f"Reading channel information from {channel_file}")
        channels_df = pd.read_csv(channel_file, sep='\t')
        std_channels = channels_df[channels_df['type'] == 'EEG']['name'].tolist()
        print(f"Found {len(std_channels)} standard EEG channels")
        return std_channels
    else:
        print(f"No channel file found at {channel_file}")
    return None

freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

def filter_eeg_channels(raw):
    """Filter out non-EEG channels and convert data to microvolts"""
    print("Filtering channels and scaling to microvolts...")
    if not raw.preload:
        print("Loading raw data...")
        raw.load_data()
    bad_patterns = ['GSR', 'EXG', 'ACC', 'Temp', 'PLET', 'RESP', 'Status', 'EOG', 'Erg']
    eeg_idx = [i for i, ch in enumerate(raw.ch_names) if not any(pattern.lower() in ch.lower() for pattern in bad_patterns)]
    if len(eeg_idx) == 0:
        print("Warning: No EEG channels found after filtering!")
        raw_eeg = raw.copy().pick_types(eeg=True, stim=False, exclude='bads')
    else:
        raw_eeg = raw.copy().pick(eeg_idx)
        print(f"Kept {len(eeg_idx)} EEG channels after filtering out auxiliary channels")
    print("Converting all channels to microvolts...")
    raw_eeg.apply_function(lambda x: x * 1e6, picks='all', channel_wise=False)
    return raw_eeg

def create_topo_maps(output_dir, debug=True):
    std_channel_names = create_channel_mapping(bids_root)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Store results by expertise group
    results = {
        'expert': {'conc_data': [], 'wander_data': [], 'info_all': None, 'freqs_all': None},
        'novice': {'conc_data': [], 'wander_data': [], 'info_all': None, 'freqs_all': None}
    }
    
    # Process each expertise group separately
    for expertise, subjects in [('expert', expert_subs), ('novice', novice_subs)]:
        print(f"\n=== Processing {expertise} subjects ===")
        
        for sub in subjects:
            print(f"\nProcessing sub-{sub}...")
            ses_paths = sorted((bids_root / f"sub-{sub}").glob('ses-*'))
            sessions = [p.name.split('-')[1] for p in ses_paths] or [None]
            
            for ses in sessions:
                print(f"Processing sub-{sub}, ses-{ses}...")
                try:
                    for extension in ['.bdf', '.edf', '.set', None]:  # Prioritize .bdf
                        bp = BIDSPath(root=bids_root, subject=sub, session=ses, task='meditation', datatype='eeg', suffix='eeg', extension=extension)
                        if debug:
                            print(f"   ↳ Trying path: {bp}")
                        try:
                            raw = read_raw_bids(bp, verbose=False)
                            if debug:
                                print(f"   ↳ Successfully read data with extension {extension}")
                            break
                        except Exception as e:
                            if debug:
                                print(f"   ↳ Failed with extension {extension}: {str(e)}")
                            continue
                    else:
                        print(f"   ↳ Could not read data for sub-{sub}, ses-{ses}")
                        continue
                    
                    raw.load_data()
                    raw = filter_eeg_channels(raw)
                    raw.pick_types(eeg=True, exclude='bads')
                    
                    if std_channel_names and len(raw.ch_names) >= 64:
                        rename_dict = {raw.ch_names[i]: std_channel_names[i] for i in range(min(64, len(raw.ch_names)))}
                        raw.rename_channels(rename_dict)
                        if debug:
                            print(f"   ↳ Renamed first 64 channels to: {raw.ch_names[:10]}...")
                        # Pick only the 64 renamed channels for further processing
                        raw = raw.pick_channels(list(rename_dict.values()), ordered=True)
                        if debug:
                            print(f"   ↳ Picked only the 64 renamed channels: {raw.ch_names[:10]}...")
                    
                    # Apply standard montage
                    montage = mne.channels.make_standard_montage('standard_1020')
                    raw.set_montage(montage, on_missing='ignore')
                    if debug:
                        print(f"   ↳ Standard 10-20 montage applied")
                    
                    current_ch_names = raw.ch_names
                    current_info = raw.info
                    if results[expertise]['info_all'] is None:
                        results[expertise]['info_all'] = current_info.copy()
                        if debug:
                            print(f"   ↳ Set info_all with ch_names: {results[expertise]['info_all']['ch_names'][:10]}...")
                    
                    if debug:
                        print(f"   ↳ Selected {len(current_ch_names)} EEG channels for analysis.")
                    
                    if raw.info['sfreq'] != 256:
                        if debug:
                            print(f"   ↳ Resampling from {raw.info['sfreq']} Hz to 256 Hz")
                        raw.resample(256)
                    
                    annotations = raw.annotations
                    pattern = re.compile(r'response/(\d+)')
                    response_values = [int(match.group(1)) for desc in annotations.description if (match := pattern.match(str(desc)))]
                    
                    if not response_values:
                        print(f"   ↳ No response events found")
                        continue
                    
                    if debug:
                        print(f"   ↳ Found response values: {set(response_values)}")
                    
                    events = np.zeros((len(response_values), 3), dtype=int)
                    for i, idx in enumerate([i for i, desc in enumerate(annotations.description) if pattern.match(str(desc))]):
                        sample = int(annotations.onset[idx] * raw.info['sfreq'])
                        events[i, 0] = sample
                        events[i, 2] = response_values[i]
                    
                    conc_events = events[events[:, 2] <= 2]  # 1-2 → Concentration
                    wander_events = events[events[:, 2] >= 4]  # 4-8 → Mind-wandering
                    
                    if len(conc_events) == 0 or len(wander_events) == 0:
                        print(f"   ↳ skipping: no concentration or mind wandering events")
                        continue
                    
                    print(f"   ↳ Found {len(conc_events)} concentration and {len(wander_events)} mind wandering events")
                    
                    # Create longer epochs - 90 seconds before probe
                    conc_epochs = mne.Epochs(raw, conc_events, tmin=-90.0, tmax=-0.1, baseline=None, preload=True)
                    wander_epochs = mne.Epochs(raw, wander_events, tmin=-90.0, tmax=-0.1, baseline=None, preload=True)
                    
                    if len(conc_epochs) == 0 or len(wander_epochs) == 0:
                        print(f"   ↳ All epochs were dropped")
                        continue
                    
                    print(f"   ↳ Processing {len(conc_epochs)} concentration and {len(wander_events)} mind wandering epochs, each covering {conc_epochs.times[0]:.1f}s to {conc_epochs.times[-1]:.1f}s")
                    
                    n_fft = 256
                    conc_psd = conc_epochs.compute_psd(method='welch', fmin=1, fmax=45, n_fft=n_fft)
                    wander_psd = wander_epochs.compute_psd(method='welch', fmin=1, fmax=45, n_fft=n_fft)
                    
                    freqs = conc_psd.freqs
                    if results[expertise]['freqs_all'] is None:
                        results[expertise]['freqs_all'] = freqs
                    
                    conc_psd_uV2Hz = conc_psd.get_data().mean(axis=0)
                    wander_psd_uV2Hz = wander_psd.get_data().mean(axis=0)
                    epsilon = 1e-15
                    conc_psd_dB = 10 * np.log10(conc_psd_uV2Hz + epsilon)
                    wander_psd_dB = 10 * np.log10(wander_psd_uV2Hz + epsilon)
                    
                    results[expertise]['conc_data'].append(conc_psd_dB)
                    results[expertise]['wander_data'].append(wander_psd_dB)
                    
                    print(f"   ↳ Successfully extracted data from {len(current_ch_names)} channels")
                    
                except Exception as e:
                    print(f"   ↳ Error processing: {str(e)}")
                    continue
    
    # Process each expertise group separately for visualization
    for expertise in ['expert', 'novice']:
        print(f"\n=== Processing results for {expertise} meditators ===")
        
        # Create directory for this expertise group
        expertise_dir = output_path / expertise
        expertise_dir.mkdir(exist_ok=True)
        
        conc_data = results[expertise]['conc_data']
        wander_data = results[expertise]['wander_data']
        info_all = results[expertise]['info_all']
        freqs_all = results[expertise]['freqs_all']
        
        if not conc_data or freqs_all is None:
            print(f"No data could be processed for {expertise} group. Skipping.")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"No data could be processed for {expertise} group", ha='center', va='center', fontsize=20)
            plt.axis('off')
            plt.savefig(expertise_dir / f'{expertise}_error_no_data.png', dpi=300)
            plt.close()
            continue
        
        conc_data = np.array(conc_data)
        wander_data = np.array(wander_data)
        avg_conc = np.mean(conc_data, axis=0)
        avg_wander = np.mean(wander_data, axis=0)
        sem_conc = np.std(conc_data, axis=0) / np.sqrt(conc_data.shape[0]) if conc_data.shape[0] > 0 else np.zeros_like(avg_conc)
        sem_wander = np.std(wander_data, axis=0) / np.sqrt(wander_data.shape[0]) if wander_data.shape[0] > 0 else np.zeros_like(avg_wander)
        diff_data = avg_conc - avg_wander
        diff_sem = np.sqrt(sem_conc**2 + sem_wander**2)
        
        if info_all is not None:
            info_plot = info_all.copy()
            can_plot_topo = True
        else:
            print(f"No valid info structure found for {expertise} group. Cannot generate topomaps.")
            can_plot_topo = False
        
        for band_name, (fmin, fmax) in freq_bands.items():
            idx_band = np.logical_and(freqs_all >= fmin, freqs_all <= fmax)
            band_conc_dB = np.mean(avg_conc[:, idx_band], axis=1)
            band_wander_dB = np.mean(avg_wander[:, idx_band], axis=1)
            band_diff_dB = np.mean(diff_data[:, idx_band], axis=1)
            
            # Print mean power values for each condition
            print(f"--- {expertise} - {band_name} (dB) ---")
            print(f"  Concentration Power mean: {np.mean(band_conc_dB):.2f} dB")
            print(f"  Mind Wandering Power mean: {np.mean(band_wander_dB):.2f} dB")
            print(f"  Difference Power mean: {np.mean(band_diff_dB):.2f} dB (Conc-Wand)")
            
            # More detailed debugging for the alpha band
            if band_name == 'alpha':
                print(f"\nDetailed comparison of {expertise} - Concentration vs Mind Wandering data:")
                print(f"  Shape of band_conc_dB: {band_conc_dB.shape}")
                print(f"  Shape of band_wander_dB: {band_wander_dB.shape}")
                print(f"  First 5 values - Concentration: {band_conc_dB[:5]}")
                print(f"  First 5 values - Mind Wandering: {band_wander_dB[:5]}")
                print(f"  Max difference: {np.max(np.abs(band_conc_dB - band_wander_dB)):.4f} dB")
                print(f"  Mean absolute difference: {np.mean(np.abs(band_conc_dB - band_wander_dB)):.4f} dB")
            
            # Check if data is identical
            if np.allclose(band_conc_dB, band_wander_dB):
                print(f"{expertise} WARNING: Concentration and Mind Wandering data are identical!")
            else:
                print(f"{expertise} Data differs between Concentration and Mind Wandering—good!")
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'{expertise.capitalize()}: {band_name.capitalize()} Band Power (dB relative to 1 µV²/Hz)', fontsize=16)
            
            plot_titles = ['Concentration', 'Mind Wandering', 'Difference (Conc-Wand)']
            plot_data = [band_conc_dB, band_wander_dB, band_diff_dB]
            
            # Robust Percentile Scaling
            combined_power_data = np.concatenate((band_conc_dB, band_wander_dB))
            vmin_power = np.percentile(combined_power_data, 5)
            vmax_power = np.percentile(combined_power_data, 95)
            print(f"  Robust power scale for {expertise} {band_name}: vmin={vmin_power:.2f} dB, vmax={vmax_power:.2f} dB (5th-95th percentiles)")
            
            # For difference map, use symmetric scaling
            vmax_diff = np.max(np.abs(band_diff_dB))
            vmin_diff = -vmax_diff
            
            cmaps = ['viridis', 'viridis', 'RdBu_r']
            vmins = [vmin_power, vmin_power, vmin_diff]
            vmaxs = [vmax_power, vmax_power, vmax_diff]
            
            if can_plot_topo:
                for i, (title, data, ax, cmap, vmin, vmax) in enumerate(zip(plot_titles, plot_data, axes, cmaps, vmins, vmaxs)):
                    ax.set_title(title, fontsize=14)
                    # Use shared vmin and vmax for Concentration and Mind Wandering
                    if i < 2:  # For Concentration and Mind Wandering
                        im, _ = mne.viz.plot_topomap(plot_data[i], info_plot, cmap=cmap, vlim=(vmin_power, vmax_power), axes=ax, show=False, contours=6)
                    else:  # For Difference
                        im, _ = mne.viz.plot_topomap(data, info_plot, cmap=cmap, vlim=(vmin_diff, vmax_diff), axes=ax, show=False, contours=6)
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Power (dB)')
                print(f"Created topomap for {expertise} {band_name} band.")
            else:
                print(f"Cannot generate topomap for {expertise} {band_name}, falling back to bar chart.")
                for i, (title, data, ax, cmap, vmin, vmax) in enumerate(zip(plot_titles, plot_data, axes, cmaps, vmins, vmaxs)):
                    bars = ax.bar(range(len(data)), data)
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)
                    colors = plt.get_cmap(cmap)(norm(data))
                    for bar_idx, bar in enumerate(bars):
                        bar.set_color(colors[bar_idx])
                    ax.set_title(title, fontsize=14)
                    plot_ch_names = info_plot.ch_names if info_plot else range(len(data))
                    tick_indices = np.linspace(0, len(plot_ch_names) - 1, num=10, dtype=int)
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels([plot_ch_names[idx] for idx in tick_indices], rotation=45, ha="right")
                    ax.set_xlabel('Channel Index / Name', fontsize=12)
                    ax.set_ylabel('Power (dB)', fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.set_xlim(-1, len(data))
                    ax.set_ylim(vmin, vmax)
                    if i == 2: ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(expertise_dir / f'{expertise}_{band_name}_topomap.png', dpi=300)
            plt.savefig(output_path / f'{expertise}_{band_name}_topomap.png', dpi=300)
            plt.close()
        
        # Statistical Analysis for alpha band
        idx_alpha = np.logical_and(freqs_all >= freq_bands['alpha'][0], freqs_all <= freq_bands['alpha'][1])
        t_values = np.zeros(avg_conc.shape[0])
        p_values = np.zeros(avg_conc.shape[0])
        
        if conc_data.shape[0] >= 2:
            for ch_idx in range(avg_conc.shape[0]):
                ch_conc_db = np.mean(conc_data[:, ch_idx, idx_alpha], axis=1)
                ch_wander_db = np.mean(wander_data[:, ch_idx, idx_alpha], axis=1)
                if np.var(ch_conc_db) > 1e-10 and np.var(ch_wander_db) > 1e-10:
                    t, p = ttest_rel(ch_conc_db, ch_wander_db, nan_policy='omit')
                    t_values[ch_idx] = t if not np.isnan(t) else 0
                    p_values[ch_idx] = p if not np.isnan(p) else 1
                else:
                    t_values[ch_idx] = 0
                    p_values[ch_idx] = 1
            
            reject_fdr, pval_fdr = mne.stats.fdr_correction(p_values, alpha=0.05)
            print(f"{expertise} Alpha band: Found {reject_fdr.sum()} significant channels after FDR correction.")
            
            if can_plot_topo:
                fig_t, ax_t = plt.subplots(figsize=(8, 8))
                mask = reject_fdr
                im, _ = mne.viz.plot_topomap(t_values, info_plot, mask=mask, cmap='RdBu_r', axes=ax_t, show=False, contours=6,
                                            mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', markersize=4))
                ax_t.set_title(f'{expertise.capitalize()}: Alpha Power T-values (Conc vs. Wand, FDR p<0.05)', fontsize=14)
                cbar = fig_t.colorbar(im, ax=ax_t, fraction=0.046, pad=0.04)
                cbar.set_label('t-value')
                plt.tight_layout()
                plt.savefig(expertise_dir / f'{expertise}_alpha_ttest_topomap.png', dpi=300)
                plt.savefig(output_path / f'{expertise}_alpha_ttest_topomap.png', dpi=300)
                plt.close()
            else:
                print(f"Cannot generate t-value topomap for {expertise}.")
                fig_t, ax_t = plt.subplots(figsize=(12, 8))
                bars = ax_t.bar(range(len(t_values)), t_values)
                plot_ch_names = info_plot.ch_names if info_plot else range(len(t_values))
                for i, bar in enumerate(bars):
                    bar.set_color('red' if t_values[i] > 0 else 'blue')
                    if reject_fdr[i]:
                        bar.set_edgecolor('black')
                        bar.set_linewidth(1.5)
                ax_t.set_title(f'{expertise.capitalize()}: T-values for Alpha Power (Concentration vs. Mind Wandering)', fontsize=14)
                tick_indices = np.linspace(0, len(plot_ch_names) - 1, num=15, dtype=int)
                ax_t.set_xticks(tick_indices)
                ax_t.set_xticklabels([plot_ch_names[idx] for idx in tick_indices], rotation=45, ha="right")
                ax_t.set_xlabel('Channel', fontsize=12)
                ax_t.set_ylabel('t-value', fontsize=12)
                ax_t.grid(True, linestyle='--', alpha=0.6)
                ax_t.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax_t.set_xlim(-1, len(t_values))
                plt.tight_layout()
                plt.savefig(expertise_dir / f'{expertise}_alpha_ttest_bars.png', dpi=300)
                plt.savefig(output_path / f'{expertise}_alpha_ttest_bars.png', dpi=300)
                plt.close()
        
        else:
            print(f"Not enough subjects for paired t-test in {expertise} group.")
        
        # Band Results
        band_results = {}
        for band_name, (fmin, fmax) in freq_bands.items():
            idx_band = np.logical_and(freqs_all >= fmin, freqs_all <= fmax)
            mean_conc = np.mean(np.mean(avg_conc[:, idx_band], axis=1))
            mean_wander = np.mean(np.mean(avg_wander[:, idx_band], axis=1))
            band_results[band_name] = (mean_conc, mean_wander)
        
        # Alpha Power Comparison
        plt.figure(figsize=(10, 6))
        alpha_conc, alpha_wander = band_results['alpha']
        alpha_conc_sem = np.mean(sem_conc[:, idx_alpha], axis=1).mean()
        alpha_wander_sem = np.mean(sem_wander[:, idx_alpha], axis=1).mean()
        x = np.arange(2)
        plt.bar(x, [alpha_conc, alpha_wander], width=0.6, color=['blue', 'red'], 
                yerr=[alpha_conc_sem, alpha_wander_sem], capsize=5)
        plt.ylabel('Alpha Power (dB)', fontsize=12)
        plt.title(f'{expertise.capitalize()}: Alpha Power during Meditation States', fontsize=14)
        plt.xticks(x, ['Concentration', 'Mind-wandering'], fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        diff_db = alpha_conc - alpha_wander
        plt.text(0.5, max(alpha_conc, alpha_wander) * 0.9, 
                 f'Difference: {diff_db:.2f} dB (Conc-Wand)', ha='center', fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(expertise_dir / f'{expertise}_alpha_power_comparison_dB.png', dpi=300)
        plt.savefig(output_path / f'{expertise}_alpha_power_comparison_dB.png', dpi=300)
        plt.close()
        
        # All Bands Comparison
        plt.figure(figsize=(12, 8))
        bands = list(band_results.keys())
        x = np.arange(len(bands))
        width = 0.35
        plt.bar(x - width/2, [band_results[band][0] for band in bands], width, label='Concentration', color='blue')
        plt.bar(x + width/2, [band_results[band][1] for band in bands], width, label='Mind Wandering', color='red')
        plt.ylabel('Power (dB)', fontsize=12)
        plt.title(f'{expertise.capitalize()}: Power Comparison Across Frequency Bands', fontsize=14)
        plt.xticks(x, [b.capitalize() for b in bands], fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(expertise_dir / f'{expertise}_all_bands_comparison_dB.png', dpi=300)
        plt.savefig(output_path / f'{expertise}_all_bands_comparison_dB.png', dpi=300)
        plt.close()
        
        # Save Results to CSV
        rows = []
        for band, (fmin, fmax) in freq_bands.items():
            idx_band = np.logical_and(freqs_all >= fmin, freqs_all <= fmax)
            conc_power_avg = np.mean(avg_conc[:, idx_band], axis=1)
            wander_power_avg = np.mean(avg_wander[:, idx_band], axis=1)
            conc_sem_avg = np.mean(sem_conc[:, idx_band], axis=1)
            wander_sem_avg = np.mean(sem_wander[:, idx_band], axis=1)
            diff_power_avg = conc_power_avg - wander_power_avg
            diff_sem_avg = np.sqrt(conc_sem_avg**2 + wander_sem_avg**2)
            
            rows.append({'Group': expertise, 'Band': band, 'Condition': 'Concentration', 'Power_Mean_dB': np.mean(conc_power_avg), 'Power_SEM_dB': np.mean(conc_sem_avg)})
            rows.append({'Group': expertise, 'Band': band, 'Condition': 'Mind Wandering', 'Power_Mean_dB': np.mean(wander_power_avg), 'Power_SEM_dB': np.mean(wander_sem_avg)})
            rows.append({'Group': expertise, 'Band': band, 'Condition': 'Difference', 'Power_Mean_dB': np.mean(diff_power_avg), 'Power_SEM_dB': np.mean(diff_sem_avg)})
        
        final_df = pd.DataFrame(rows)
        final_df.to_csv(expertise_dir / f'{expertise}_frequency_power_results_dB.csv', index=False)
        final_df.to_csv(output_path / f'{expertise}_frequency_power_results_dB.csv', index=False)
        
        alpha_df = final_df[(final_df['Band'] == 'alpha')].copy()
        alpha_df = alpha_df[['Group', 'Band', 'Condition', 'Power_Mean_dB']].rename(columns={'Power_Mean_dB': 'Power_dB'})
        alpha_df.to_csv(expertise_dir / f'{expertise}_alpha_power_results_dB.csv', index=False)
        alpha_df.to_csv(output_path / f'{expertise}_alpha_power_results_dB.csv', index=False)
        
        print(f"Power comparison results for {expertise} saved to CSV and visualizations created")
    
    # Create a combined visualization comparing expert vs novice
    # Include all five frequency bands
    plt.figure(figsize=(18, 10))
    
    # Create data for the plot
    experts_data = pd.read_csv(output_path / 'expert_frequency_power_results_dB.csv')
    novices_data = pd.read_csv(output_path / 'novice_frequency_power_results_dB.csv')
    
    # Get all bands, and concentration + mind wandering (exclude difference)
    bands = ['alpha', 'theta', 'beta', 'delta', 'gamma']
    
    # Collect data for each band
    expert_values = []
    novice_values = []
    
    for band in bands:
        # Expert values
        expert_conc = experts_data[(experts_data['Band'] == band) & (experts_data['Condition'] == 'Concentration')]['Power_Mean_dB'].values[0]
        expert_wander = experts_data[(experts_data['Band'] == band) & (experts_data['Condition'] == 'Mind Wandering')]['Power_Mean_dB'].values[0]
        expert_values.extend([expert_conc, expert_wander])
        
        # Novice values
        novice_conc = novices_data[(novices_data['Band'] == band) & (novices_data['Condition'] == 'Concentration')]['Power_Mean_dB'].values[0]
        novice_wander = novices_data[(novices_data['Band'] == band) & (novices_data['Condition'] == 'Mind Wandering')]['Power_Mean_dB'].values[0]
        novice_values.extend([novice_conc, novice_wander])
    
    # Set up positions for the bars
    x = np.arange(len(bands) * 2)
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, expert_values, width, label='Experts', color='blue')
    plt.bar(x + width/2, novice_values, width, label='Novices', color='green')
    
    # Add labels and title
    plt.ylabel('Power (dB)', fontsize=14)
    plt.title('Expert vs Novice Meditation States: Power Across Frequency Bands', fontsize=16)
    
    # Create x-labels for each pair of bars
    labels = []
    for band in bands:
        labels.extend([f'{band.capitalize()}\nConc', f'{band.capitalize()}\nMW'])
    plt.xticks(x, labels, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add vertical separators between bands
    for i in range(1, len(bands)):
        plt.axvline(x=i*2 - 0.5, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path / 'expert_vs_novice_comparison.png', dpi=300)
    plt.close()
    
    print(f"Expert vs novice comparison visualization created")
    
    return output_path

if __name__ == '__main__':
    output_dir = "meditation_analysis_output/topo_maps"
    created_dir = create_topo_maps(output_dir)
    print("Topographic map analysis complete. Results saved to:", output_dir)