# Meditation EEG Analysis Suite

A comprehensive set of analytical tools for studying brain activity during meditation, focusing on the neural correlates of concentration and mind wandering. Please note that the dataset and dashboard are not in this code but can be found here: https://openneuro.org/datasets/ds001787/versions/1.1.1

and here:
https://gofile.io/d/u7wWRX

respectively.

## Overview

This project provides a suite of advanced EEG analysis scripts specifically designed for the ds001787-download meditation dataset. The suite includes tools for:

1. **Topographic Brain Mapping** - Visualizing spatial distribution of different frequency bands, including alpha and theta power analysis
2. **Time-Frequency Analysis** - Examining how brain rhythms evolve over time before response events
3. **Functional Connectivity Analysis** - Investigating brain network dynamics during meditation
4. **Machine Learning Classification** - Using EEG features to predict meditation states
5. **Interactive Dashboard** - Combining all results into a comprehensive visualization

## Dataset

This suite is designed to work with the ds001787-download meditation dataset, which contains EEG recordings of expert and novice meditation practitioners during meditation sessions. As described in Brandmeyer & Delorme (2016), subjects were interrupted approximately every 2 minutes to report their current state (concentrated or mind wandering).

The analysis follows the methodology of the original paper, separating expert meditators (≥10,000 hours of practice) from novices (100-300 hours of practice) and using 90-second epochs before each report to capture the entire meditation episode.

## Installation

### Prerequisites

- Python 3.7+
- MNE-Python (`pip install mne`)
- MNE-BIDS (`pip install mne-bids`)
- NumPy, SciPy, pandas, matplotlib
- scikit-learn
- NetworkX (for connectivity analysis)
- Seaborn (for visualization)

### Setup

1. Clone this repository to your local machine
2. Ensure the ds001787-download dataset is placed in the same directory as the scripts
3. Install required dependencies: `pip install -r requirements.txt`

## Usage

### Individual Analysis Scripts

Each script can be run independently:

```bash
# Run topographic brain mapping
python topo_maps_analysis.py

# Run time-frequency analysis
python time_frequency_analysis.py

# Run connectivity analysis 
python connectivity_analysis.py

# Run machine learning classification
python ml_classification.py
```

### All-in-One Dashboard Generation

To run all analyses and generate a comprehensive dashboard:

```bash
python meditation_dashboard.py --run-all
```

Or to generate a dashboard from existing results:

```bash
python meditation_dashboard.py
```

## Script Descriptions

### topo_maps_analysis.py

Creates topographic brain maps showing the spatial distribution of power across different frequency bands (delta, theta, alpha, beta, gamma) for both expert and novice meditators. Also analyzes alpha and theta power between concentration and mind wandering states, focusing on specific regions of interest (ROIs) defined in the Brandmeyer & Delorme paper:
- Frontal-midline theta ROI (Fz, FCz, Cz)
- Somatosensory/central alpha ROI (C3, C4, CP3, CP4)
- Posterior alpha ROI (P3, P4, PO3, PO4, O1, O2, Pz, POz, Oz)

Generates visualizations of:
- Power distribution during concentration
- Power distribution during mind wandering
- Power differences between states
- Statistical t-maps of significant differences

### time_frequency_analysis.py

Performs time-frequency decomposition to examine how brain rhythms evolve over time (90 seconds) before the participant reports either concentration or mind wandering. Analyzes expert and novice meditators separately, comparing power in different frequency bands between states.

### connectivity_analysis.py

Investigates functional connectivity between brain regions during different meditation states for expert and novice meditators. Produces:
- Connectivity matrices for different frequency bands
- Network graphs showing key connections
- Comparisons between expert and novice connectivity patterns

### ml_classification.py

Uses machine learning to classify concentration versus mind wandering states from EEG features. Analyses are performed separately for expert and novice meditators:
- Within-subject classification (personalized models)
- Cross-subject classification (generalized models)
- Feature importance analysis
- Performance metrics and visualizations

### meditation_dashboard.py

Combines all analysis results into an interactive HTML dashboard, with separate sections for expert and novice meditators, allowing easy exploration and comparison of findings.

## Output Structure

Results are organized in the following directory structure:

```
meditation_analysis_output/
├── topo_maps/
│   ├── expert_alpha_topomap.png
│   ├── novice_alpha_topomap.png
│   └── ...
├── time_frequency/
│   ├── expert/
│   │   ├── expert_grand_average_alpha_power.png
│   │   └── ...
│   ├── novice/
│   │   ├── novice_grand_average_alpha_power.png
│   │   └── ...
│   └── ...
├── connectivity/
│   ├── expert/
│   │   ├── alpha_connectivity_matrices.png
│   │   └── ...
│   ├── novice/
│   │   ├── alpha_connectivity_matrices.png
│   │   └── ...
│   └── ...
├── ml_classification/
│   ├── expert/
│   │   ├── expert_within_subject_accuracy.png
│   │   └── ...
│   ├── novice/
│   │   ├── novice_within_subject_accuracy.png
│   │   └── ...
│   └── ...
└── dashboard/
    └── meditation_dashboard.html
```

## Scientific Background

This analysis suite is built on established neuroscientific methods for analyzing EEG data during meditation, following the methodology of Brandmeyer & Delorme (2016):

1. **Frontal-Midline Theta**: Expert meditators show increased theta power (4-7 Hz) during concentration compared to mind-wandering, particularly in frontal-midline regions.

2. **Somatosensory Alpha**: Increased alpha power (8-13 Hz) in central regions during concentration compared to mind-wandering in expert meditators.

3. **Expertise Differences**: Experts show stronger and more consistent neural patterns than novices due to their extensive meditation practice.

4. **Network Dynamics**: Meditation involves changes in brain network connectivity, with different patterns for concentration vs. mind-wandering states.

## Reference

Brandmeyer, T., & Delorme, A. (2018). Reduced mind wandering in experienced meditators and associated EEG correlates. Experimental Brain Research, 234(7), 1-10. https://doi.org/10.1007/s00221-016-4811-5

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Brandmeyer & Delorme for their original research and dataset
- The ds001787-download dataset creators
- MNE-Python and MNE-BIDS development teams

## Connectivity Analysis

This module computes and visualizes the functional connectivity between brain regions during different meditation states:

- Generates connectivity matrices for different frequency bands (alpha, beta, theta, delta, gamma)
- Provides network visualizations showing the strongest connections between brain regions
- Compares connectivity patterns between concentration and mind wandering states
- Visualizes differences between the two mental states

Connectivity is calculated using weighted phase lag index (WPLI) and other metrics to minimize the effects of volume conduction 
