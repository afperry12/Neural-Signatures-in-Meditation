from pathlib import Path
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
from mne.decoding import CSP
import argparse

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

def extract_features(expertise_group, subjects, sessions=None, debug=False):
    """
    Extract features from EEG data for machine learning classification
    
    Parameters:
    -----------
    expertise_group : str
        'expert' or 'novice'
    subjects : list
        List of subject IDs
    sessions : list or None
        List of session IDs
    debug : bool
        Enable debug mode
    
    Returns:
    --------
    features : np.array
        Extracted features
    labels : np.array
        Labels for the features
    groups : np.array
        Subject identifiers for the features
    ch_names : list
        List of channel names
    """
    all_features = []
    all_labels = []
    all_groups = []
    ch_names = None
    
    # Process only the subjects from the specified expertise group
    for sub in subjects:
        print(f"\nProcessing {expertise_group} sub-{sub}...")
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
                
                # Pick only EEG channels
                raw.pick_types(eeg=True, stim=False, exclude='bads')
                if debug:
                    print(f"   ↳ Picked {len(raw.ch_names)} EEG channels after pick_types.")
                
                # Ensure we actually have some EEG channels left
                if len(raw.ch_names) == 0:
                    print(f"   ↳ No EEG channels left after picking for sub-{sub}, ses-{ses}. Skipping.")
                    continue

                # Store channel names from the first successful file that HAS EEG channels
                if ch_names is None:
                    ch_names = raw.ch_names
                    if debug:
                        print(f"   -> Stored {len(ch_names)} channel names from EEG data.")
                
                # Apply band-pass filter
                raw.filter(l_freq=1, h_freq=45)
                
                # Read events
                events, event_id = mne.events_from_annotations(raw)
                
                # Extract responses (value 1-2 = concentration, value 4-8 = mind wandering)
                concentration_events = events[events[:, 2] <= 2]  # 1-2 → Concentration
                mind_wandering_events = events[events[:, 2] >= 4]  # 4-8 → Mind-wandering
                
                if len(concentration_events) == 0 or len(mind_wandering_events) == 0:
                    print(f"   ↳ skipping: no concentration or mind wandering events")
                    continue
                
                # Create epochs with longer time window
                # for concentration and mind wandering
                conc_epochs = mne.Epochs(raw, concentration_events, tmin=-90.0, tmax=-0.1, 
                                       baseline=None, preload=True)
                wander_epochs = mne.Epochs(raw, mind_wandering_events, tmin=-90.0, tmax=-0.1, 
                                         baseline=None, preload=True)
                
                print(f"   ↳ Created epochs with time range {conc_epochs.times[0]:.1f}s to {conc_epochs.times[-1]:.1f}s")
                print(f"   ↳ Found {len(conc_epochs)} concentration and {len(wander_epochs)} mind wandering epochs")
                
                # Extract features from epochs
                features_conc = []
                features_wander = []
                
                # Get PSD for each epoch and extract band power
                for epoch_idx in range(len(conc_epochs)):
                    epoch_features = []
                    
                    try:
                        # Calculate PSD using the Epochs method
                        spectrum = conc_epochs[epoch_idx].compute_psd(
                            method='welch', fmin=1, fmax=45, n_fft=512, 
                            n_overlap=256, verbose=False
                        )
                        psds, freqs = spectrum.get_data(return_freqs=True)
                        
                        if psds.ndim == 3 and psds.shape[0] == 1:
                            psds_ch_freq = psds[0] # Remove the first dimension if it's 1 (single epoch)
                        elif psds.ndim == 2:
                            psds_ch_freq = psds
                        else:
                            print(f"    ↳ Unexpected PSD shape: {psds.shape}. Skipping epoch.")
                            # Attempt to create zero features for this epoch to maintain structure
                            num_expected_features_per_band = len(ch_names) if ch_names else 0
                            epoch_features.extend(np.zeros(num_expected_features_per_band * len(freq_bands)))
                            features_conc.append(epoch_features)
                            if debug: print(f"    DEBUG [Conc Epoch {epoch_idx}]: Added zeros due to PSD shape error. len(epoch_features) = {len(epoch_features)}")
                            continue # to next epoch

                        # Convert to uV^2/Hz then to dB
                        psds_uV2Hz = psds_ch_freq
                        epsilon = 1e-15
                        psds_dB = 10 * np.log10(psds_uV2Hz + epsilon)

                        # Extract band power for each channel and frequency band (from psds_dB)
                        for band_name, (fmin, fmax) in freq_bands.items():
                            freq_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                            
                            if debug and epoch_idx == 0:
                                print(f"    DEBUG [{band_name}]: psds_dB.shape={psds_dB.shape}, freq_idx.sum()={np.sum(freq_idx)}")

                            if np.any(freq_idx) and psds_dB.size > 0:
                                band_power = np.mean(psds_dB[:, freq_idx], axis=1)
                            else:
                                band_power = np.zeros(psds_dB.shape[0]) # n_channels zeros
                            
                            if debug and epoch_idx == 0:
                                print(f"     DEBUG [{band_name}]: band_power.shape = {band_power.shape}")
                                
                            epoch_features.extend(band_power.ravel())
                        
                        # Check length
                        if debug and epoch_idx == 0:
                            print(f"    DEBUG [Conc Epoch {epoch_idx}]: len(epoch_features) = {len(epoch_features)}, expected {len(ch_names) * len(freq_bands)}")
                        
                        features_conc.append(epoch_features)
                        
                    except Exception as psd_err:
                        print(f"    ↳ Skipping concentration epoch {epoch_idx} due to PSD error: {psd_err}")
                        continue # Skip this epoch if PSD fails
                
                # Do the same for mind wandering epochs
                for epoch_idx in range(len(wander_epochs)):
                    epoch_features = []
                    
                    try:
                        # Calculate PSD using the Epochs method
                        spectrum = wander_epochs[epoch_idx].compute_psd(
                            method='welch', fmin=1, fmax=45, n_fft=512, 
                            n_overlap=256, verbose=False
                        )
                        psds, freqs = spectrum.get_data(return_freqs=True)
                        
                        if psds.ndim == 3 and psds.shape[0] == 1:
                            psds_ch_freq = psds[0]
                        elif psds.ndim == 2:
                            psds_ch_freq = psds
                        else:
                            print(f"    ↳ Unexpected PSD shape for wander: {psds.shape}. Skipping epoch.")
                            num_expected_features_per_band = len(ch_names) if ch_names else 0
                            epoch_features.extend(np.zeros(num_expected_features_per_band * len(freq_bands)))
                            features_wander.append(epoch_features)
                            if debug: print(f"    DEBUG [Wand Epoch {epoch_idx}]: Added zeros due to PSD shape error. len(epoch_features) = {len(epoch_features)}")
                            continue

                        # Convert to uV^2/Hz then to dB
                        psds_uV2Hz = psds_ch_freq
                        epsilon = 1e-15
                        psds_dB = 10 * np.log10(psds_uV2Hz + epsilon)

                        # Extract band power for each channel and frequency band (from psds_dB)
                        for band_name, (fmin, fmax) in freq_bands.items():
                            freq_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                            
                            if np.any(freq_idx) and psds_dB.size > 0:
                                band_power = np.mean(psds_dB[:, freq_idx], axis=1)
                            else:
                                band_power = np.zeros(psds_dB.shape[0])
                                
                            epoch_features.extend(band_power.ravel())
                        
                        # Check length
                        if debug and epoch_idx == 0:
                            print(f"    DEBUG [Wand Epoch {epoch_idx}]: len(epoch_features) = {len(epoch_features)}, expected {len(ch_names) * len(freq_bands)}")

                        features_wander.append(epoch_features)
                        
                    except Exception as psd_err:
                        print(f"    ↳ Skipping wandering epoch {epoch_idx} due to PSD error: {psd_err}")
                        continue # Skip this epoch if PSD fails
                
                # Convert to arrays and combine data
                features_conc = np.array(features_conc)
                features_wander = np.array(features_wander)
                
                # Create labels (0 for concentration, 1 for mind wandering)
                labels_conc = np.zeros(len(features_conc))
                labels_wander = np.ones(len(features_wander))
                
                # Add to overall dataset
                X_sub = np.vstack((features_conc, features_wander))
                y_sub = np.hstack((labels_conc, labels_wander))
                
                # Add subject identifiers for later analysis
                subject_ids = np.repeat(sub, len(X_sub))
                
                all_features.append(X_sub)
                all_labels.append(y_sub)
                all_groups.append(subject_ids)
                
                print(f"   ↳ Extracted features from {len(features_conc)} concentration and {len(features_wander)} mind wandering epochs")
                
            except Exception as e:
                print(f"   ↳ Error processing: {str(e)}")
                continue
    
    # Check if we have any valid features
    if not all_features:
        print(f"No valid features extracted for {expertise_group} subjects.")
        return np.array([]), np.array([]), np.array([]), ch_names
    
    # Combine all data
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    groups = np.hstack(all_groups)
    
    print(f"Total {expertise_group} dataset shape: Features={X.shape}, Labels={y.shape}, Groups={groups.shape}")
    
    return X, y, groups, ch_names

def run_ml_classification(output_dir, debug=False):
    """
    Perform machine learning classification of meditation states
    (concentration vs. mind wandering) for experts and novices separately,
    as well as expert vs. novice trait classification.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save output files
    debug : bool
        Enable debug mode
    """
    output_path = Path(output_dir) / "ml_classification"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directories for each expertise group
    expert_dir = output_path / "expert"
    expert_dir.mkdir(exist_ok=True)
    
    novice_dir = output_path / "novice"
    novice_dir.mkdir(exist_ok=True)
    
    # Create directory for expertise classification
    expertise_dir = output_path / "expertise_classification"
    expertise_dir.mkdir(exist_ok=True)
    
    print("\n--- Starting ML Classification ---")
    
    # Store data for later expert vs. novice classification
    all_expert_features = []
    all_novice_features = []
    all_expert_subjects = []
    all_novice_subjects = []
    all_ch_names = None
    
    # Process each expertise group separately
    for expertise, subjects, output_dir in [
        ('expert', expert_subs, expert_dir),
        ('novice', novice_subs, novice_dir)
    ]:
        print(f"\n=== ML Classification for {expertise} meditators ===")
        
        # 1. Extract Features
        print(f"Extracting features for {expertise} subjects...")
        features, labels, groups, ch_names = extract_features(expertise, subjects, debug=debug)
        
        # Store channel names for later use
        if all_ch_names is None and ch_names is not None:
            all_ch_names = ch_names
            
        # Store features for expert vs. novice classification
        if expertise == 'expert' and features.size > 0:
            all_expert_features.append(features)
            all_expert_subjects.extend(groups)
        elif expertise == 'novice' and features.size > 0:
            all_novice_features.append(features)
            all_novice_subjects.extend(groups)
        
        # Handle case where no features were extracted
        if features.size == 0 or ch_names is None:
            print(f"Error: No features extracted or channel names not found for {expertise} group. Skipping ML.")
            # Create dummy empty files to prevent dashboard errors
            Path(output_dir / "classification_report.json").touch()
            Path(output_dir / "feature_importance.png").touch()
            Path(output_dir / "confusion_matrix.png").touch()
            Path(output_dir / "roc_curve.png").touch()
            
            # Also create copies in the main output directory
            Path(output_path / f"{expertise}_classification_report.json").touch()
            
            continue
            
        print(f"Total {expertise} dataset: {features.shape[0]} samples, {features.shape[1]} features")
        
        # Perform within-subject classification
        print(f"\nPerforming within-subject classification for {expertise} group...")
        
        # Store results
        within_subject_results = []
        
        for subject in np.unique(groups):
            # Select data for this subject
            mask = groups == subject
            X_subject = features[mask]
            y_subject = labels[mask]
            
            if len(np.unique(y_subject)) < 2:
                print(f"Skipping sub-{subject}: insufficient classes")
                continue
            
            # Make sure we have enough samples
            if len(X_subject) < 10:
                print(f"Skipping sub-{subject}: too few samples ({len(X_subject)})")
                continue
            
            # Create a classification pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(f_classif, k=min(20, X_subject.shape[1]))),
                ('classifier', SVC(kernel='rbf', probability=True))
            ])
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, X_subject, y_subject, cv=cv, scoring='accuracy')
            
            print(f"{expertise} Subject {subject}: Mean accuracy = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            
            # Store results
            within_subject_results.append({
                'Subject': subject,
                'Mean_Accuracy': np.mean(scores),
                'Std_Accuracy': np.std(scores),
                'Num_Samples': len(X_subject)
            })
        
        # Save within-subject results
        within_df = pd.DataFrame(within_subject_results)
        within_df.to_csv(output_dir / f'{expertise}_within_subject_classification.csv', index=False)
        within_df.to_csv(output_path / f'{expertise}_within_subject_classification.csv', index=False)
        
        # Plot within-subject accuracies
        if len(within_subject_results) > 0:
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Subject', y='Mean_Accuracy', data=within_df)
            
            # Add chance level line
            plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Chance level')
            
            # Customize plot
            plt.ylim(0, 1.0)
            plt.title(f'{expertise.capitalize()} Within-Subject Classification Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Subject')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Save figure
            plt.savefig(output_dir / f'{expertise}_within_subject_accuracy.png', dpi=300)
            plt.savefig(output_path / f'{expertise}_within_subject_accuracy.png', dpi=300)
            plt.close()
        
        # Perform cross-subject classification
        print(f"\nPerforming cross-subject classification for {expertise} group...")
        
        # Train classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True)
        }
        
        # Results container
        cross_subj_results = []
        
        # Use LeaveOneGroupOut cross-validation to test generalization across subjects
        cv = LeaveOneGroupOut()
        
        for name, classifier in classifiers.items():
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(f_classif, k=min(50, features.shape[1]))),
                ('classifier', classifier)
            ])
            
            # Cross-validation scores
            try:
                scores = []
                for train_idx, test_idx in cv.split(features, labels, groups):
                    X_train, X_test = features[train_idx], features[test_idx]
                    y_train, y_test = labels[train_idx], labels[test_idx]
                    
                    pipeline.fit(X_train, y_train)
                    score = pipeline.score(X_test, y_test)
                    scores.append(score)
                
                print(f"{expertise} {name}: Mean accuracy = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
                
                # Store results
                cross_subj_results.append({
                    'Classifier': name,
                    'Mean_Accuracy': np.mean(scores),
                    'Std_Accuracy': np.std(scores)
                })
            except Exception as e:
                print(f"Error with {name} classifier for {expertise} group: {e}")
                continue
        
        # Save cross-subject results
        if cross_subj_results:
            cross_df = pd.DataFrame(cross_subj_results)
            cross_df.to_csv(output_dir / f'{expertise}_cross_subject_classification.csv', index=False)
            cross_df.to_csv(output_path / f'{expertise}_cross_subject_classification.csv', index=False)
            
            # Plot cross-subject accuracies
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='Classifier', y='Mean_Accuracy', data=cross_df)
            
            # Add chance level line
            plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Chance level')
            
            # Customize plot
            plt.ylim(0, 1.0)
            plt.title(f'{expertise.capitalize()} Cross-Subject Classification Accuracy')
            plt.ylabel('Accuracy')
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_dir / f'{expertise}_cross_subject_accuracy.png', dpi=300)
            plt.savefig(output_path / f'{expertise}_cross_subject_accuracy.png', dpi=300)
            plt.close()
        
        # Feature importance analysis
        print(f"\nAnalyzing feature importance for {expertise} group...")
        
        # Use Random Forest for feature importance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, labels)
        
        # Calculate feature importances
        importances = rf.feature_importances_
        
        # Generate feature names (using stored ch_names)
        feature_names = []
        if ch_names is not None:
            for band in freq_bands.keys():
                for ch_name in ch_names:
                    feature_names.append(f'{band}_dB_{ch_name}')
        else:
            num_actual_features = features.shape[1] if features.ndim == 2 and features.shape[1] > 0 else 0
            feature_names = [f'feature_dB_{i}' for i in range(num_actual_features)]
        
        # Ensure we have the right number of feature names
        if len(feature_names) != features.shape[1] and features.size > 0:
            print(f"Warning: Mismatch creating feature names. Expected based on ch_names: {len(feature_names)}, Actual from data: {features.shape[1]}")
            feature_names = [f'feature_dB_{i}' for i in range(features.shape[1])]

        # Create a DataFrame with features and importances
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importances)],
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Save to CSV
        importance_df.to_csv(output_dir / f'{expertise}_feature_importance.csv', index=False)
        importance_df.to_csv(output_path / f'{expertise}_feature_importance.csv', index=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'{expertise.capitalize()}: Top 20 Important Features')
        plt.tight_layout()
        plt.savefig(output_dir / f'{expertise}_top_feature_importance.png', dpi=300)
        plt.savefig(output_path / f'{expertise}_top_feature_importance.png', dpi=300)
        plt.close()
        
        # Generate detailed report
        # Train final model on all data for confusion matrix and ROC curve
        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=min(50, features.shape[1]))),
            ('classifier', SVC(kernel='rbf', probability=True))
        ])
        
        # Use cross-validation to get predictions
        cv_predictions = np.zeros_like(labels)
        cv_probas = np.zeros((len(labels), 2))
        
        # Use simple StratifiedKFold for final evaluation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, test_idx in skf.split(features, labels):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            final_pipeline.fit(X_train, y_train)
            cv_predictions[test_idx] = final_pipeline.predict(X_test)
            cv_probas[test_idx] = final_pipeline.predict_proba(X_test)
        
        # Confusion Matrix
        cm = confusion_matrix(labels, cv_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Concentration', 'Mind Wandering'],
                    yticklabels=['Concentration', 'Mind Wandering'])
        plt.title(f'{expertise.capitalize()}: Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / f'{expertise}_confusion_matrix.png', dpi=300)
        plt.savefig(output_path / f'{expertise}_confusion_matrix.png', dpi=300)
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(labels, cv_probas[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{expertise.capitalize()}: Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(output_dir / f'{expertise}_roc_curve.png', dpi=300)
        plt.savefig(output_path / f'{expertise}_roc_curve.png', dpi=300)
        plt.close()
        
        # Classification Report
        report = classification_report(labels, cv_predictions, 
                                      target_names=['Concentration', 'Mind Wandering'],
                                      output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(output_dir / f'{expertise}_classification_report.csv')
        report_df.to_csv(output_path / f'{expertise}_classification_report.csv')
        
        print(f"Machine learning classification for {expertise} group complete. Results saved to: {output_dir}")
    
    # Add Expert vs Novice trait classification
    print("\n=== Expert vs Novice Trait Classification ===")
    
    # Check if we have both expert and novice data
    if all_expert_features and all_novice_features:
        # Combine expert and novice features
        expert_features = np.vstack(all_expert_features)
        novice_features = np.vstack(all_novice_features)
        
        # Create labels (0 for expert, 1 for novice)
        expert_labels = np.zeros(len(expert_features))
        novice_labels = np.ones(len(novice_features))
        
        # Combine all data
        X = np.vstack((expert_features, novice_features))
        y = np.hstack((expert_labels, novice_labels))
        groups = np.hstack((all_expert_subjects, all_novice_subjects))
        
        print(f"Expert vs Novice dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Expert samples: {len(expert_features)}, Novice samples: {len(novice_features)}")
        
        # Leave-one-subject-out classification
        # Use LeaveOneGroupOut for cross-validation (leave one subject out)
        logo = LeaveOneGroupOut()
        
        # Initialize classifiers
        classifiers = {
            'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
            'Logistic Regression': LogisticRegression(class_weight='balanced')
        }
        
        # Store results
        loso_results = {name: {'accuracies': [], 'predictions': [], 'true_labels': []} 
                       for name in classifiers.keys()}
        
        print("Performing leave-one-subject-out cross-validation...")
        
        # Track best model
        best_acc = 0
        best_clf_name = None
        
        # For each classifier
        for clf_name, clf in classifiers.items():
            # For each fold (leave one subject out)
            for train_idx, test_idx in logo.split(X, y, groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Feature selection (select top k features)
                k = min(50, X_train.shape[1])
                fs = SelectKBest(f_classif, k=k)
                X_train_fs = fs.fit_transform(X_train_scaled, y_train)
                X_test_fs = fs.transform(X_test_scaled)
                
                # Train and predict
                clf.fit(X_train_fs, y_train)
                y_pred = clf.predict(X_test_fs)
                
                # Calculate accuracy
                acc = accuracy_score(y_test, y_pred)
                loso_results[clf_name]['accuracies'].append(acc)
                loso_results[clf_name]['predictions'].extend(y_pred)
                loso_results[clf_name]['true_labels'].extend(y_test)
                
            # Calculate mean accuracy
            mean_acc = np.mean(loso_results[clf_name]['accuracies'])
            std_acc = np.std(loso_results[clf_name]['accuracies'])
            
            print(f"{clf_name}: Mean Accuracy = {mean_acc:.3f} ± {std_acc:.3f}")
            
            # Track best model
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_clf_name = clf_name
        
        # Generate visualizations
        # Classification accuracy plot
        plt.figure(figsize=(10, 6))
        
        # Prepare data for plotting
        clf_names = list(classifiers.keys())
        mean_accs = [np.mean(loso_results[name]['accuracies']) for name in clf_names]
        std_accs = [np.std(loso_results[name]['accuracies']) for name in clf_names]
        
        # Create bar plot
        plt.bar(range(len(clf_names)), mean_accs, yerr=std_accs, capsize=10)
        plt.xticks(range(len(clf_names)), clf_names)
        plt.axhline(0.5, color='r', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.0)
        plt.ylabel('Classification Accuracy')
        plt.title('Expert vs Novice Classification Accuracy')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(expertise_dir / 'expertise_classification_accuracy.png', dpi=300)
        plt.savefig(output_path / 'expertise_classification_accuracy.png', dpi=300)
        plt.close()
        
        # Confusion matrix for best classifier
        true_labels = loso_results[best_clf_name]['true_labels']
        predictions = loso_results[best_clf_name]['predictions']
        
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Expert', 'Novice'],
                    yticklabels=['Expert', 'Novice'])
        plt.title(f'Confusion Matrix ({best_clf_name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(expertise_dir / 'expertise_confusion_matrix.png', dpi=300)
        plt.savefig(output_path / 'expertise_confusion_matrix.png', dpi=300)
        plt.close()
        
        # Feature importance analysis
        print("\nAnalyzing feature importance for expertise classification...")
        
        # Train Random Forest on entire dataset for feature importance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_scaled, y)
        
        # Calculate feature importances
        importances = rf.feature_importances_
        
        # Generate feature names
        feature_names = []
        if all_ch_names is not None:
            for band in freq_bands.keys():
                for ch_name in all_ch_names:
                    feature_names.append(f'{band}_dB_{ch_name}')
        else:
            feature_names = [f'feature_dB_{i}' for i in range(X.shape[1])]
        
        # Ensure we have the right number of feature names
        if len(feature_names) != X.shape[1]:
            feature_names = [f'feature_dB_{i}' for i in range(X.shape[1])]
        
        # Create a DataFrame with features and importances
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importances)],
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Save to CSV
        importance_df.to_csv(expertise_dir / 'expertise_feature_importance.csv', index=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 20 Important Features for Expert vs Novice Classification')
        plt.tight_layout()
        plt.savefig(expertise_dir / 'expertise_feature_importance.png', dpi=300)
        plt.savefig(output_path / 'expertise_feature_importance.png', dpi=300)
        plt.close()
        
        print("Expert vs Novice trait classification completed.")
    else:
        print("Not enough data to perform expert vs novice classification.")
    
    # Continue with the existing code to create expert vs novice comparison visualization
    try:
        # Load results
        expert_within = pd.read_csv(expert_dir / "expert_within_subject_classification.csv")
        novice_within = pd.read_csv(novice_dir / "novice_within_subject_classification.csv")
        expert_cross = pd.read_csv(expert_dir / "expert_cross_subject_classification.csv")
        novice_cross = pd.read_csv(novice_dir / "novice_cross_subject_classification.csv")
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        # Within-subject comparison
        plt.subplot(1, 2, 1)
        expert_acc = expert_within['Mean_Accuracy'].mean()
        novice_acc = novice_within['Mean_Accuracy'].mean()
        expert_std = expert_within['Mean_Accuracy'].std() / np.sqrt(len(expert_within))
        novice_std = novice_within['Mean_Accuracy'].std() / np.sqrt(len(novice_within))
        
        plt.bar([0, 1], [expert_acc, novice_acc], 
                yerr=[expert_std, novice_std],
                color=['blue', 'green'], width=0.6, capsize=10)
        plt.xticks([0, 1], ['Expert', 'Novice'])
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.7)
        plt.title('Within-Subject Classification')
        plt.ylabel('Mean Accuracy')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        # Cross-subject comparison
        plt.subplot(1, 2, 2)
        svm_expert = expert_cross[expert_cross['Classifier'] == 'SVM']['Mean_Accuracy'].values[0]
        svm_novice = novice_cross[novice_cross['Classifier'] == 'SVM']['Mean_Accuracy'].values[0]
        std_expert = expert_cross[expert_cross['Classifier'] == 'SVM']['Std_Accuracy'].values[0]
        std_novice = novice_cross[novice_cross['Classifier'] == 'SVM']['Std_Accuracy'].values[0]
        
        plt.bar([0, 1], [svm_expert, svm_novice], 
                yerr=[std_expert, std_novice],
                color=['blue', 'green'], width=0.6, capsize=10)
        plt.xticks([0, 1], ['Expert', 'Novice'])
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.7)
        plt.title('Cross-Subject Classification (SVM)')
        plt.ylabel('Mean Accuracy')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Expert vs Novice Classification Performance', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'expert_vs_novice_classification.png', dpi=300)
        plt.close()
        
        print("Created expert vs novice comparison visualization")
    except Exception as e:
        print(f"Could not create expert vs novice comparison: {e}")
    
    print("\nMachine learning classification complete for all groups.")
    return output_path

if __name__ == '__main__':
    # Add argument parsing
    parser = argparse.ArgumentParser(description="Machine Learning Classification for Meditation States")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()
    
    # Define output directory
    output_dir = Path(__file__).parent / "meditation_analysis_output"
    
    # Run ML classification separately for expert and novice groups
    run_ml_classification(output_dir, debug=args.debug) 