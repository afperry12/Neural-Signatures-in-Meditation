from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import base64
from io import BytesIO
import argparse
import os
import subprocess

def create_dashboard(output_dir):
    """
    Create an HTML dashboard that combines all analysis results
    
    Parameters:
    -----------
    output_dir : str
        Directory where analysis results are stored
    """
    # Define analysis directories
    topo_dir = Path(output_dir) / "topo_maps"
    tf_dir = Path(output_dir) / "time_frequency"
    connectivity_dir = Path(output_dir) / "connectivity"
    ml_dir = Path(output_dir) / "ml_classification"
    
    # Create a directory for the dashboard
    dashboard_dir = Path(output_dir) / "dashboard"
    dashboard_dir.mkdir(exist_ok=True, parents=True)
    
    # Debug: Print available image files in the topo_maps directory
    print("Available topo map files:")
    if topo_dir.exists():
        for file in topo_dir.glob("*.png"):
            print(f"  {file.name}")
    else:
        print(f"  Warning: {topo_dir} directory not found")
    
    # Function to encode images as base64 for HTML embedding
    def get_image_base64(image_path):
        try:
            if not os.path.exists(image_path):
                print(f"Error: Image file does not exist at {image_path}")
                return None
            
            with open(image_path, "rb") as img_file:
                try:
                    encoded = base64.b64encode(img_file.read()).decode('utf-8')
                    if not encoded:
                        print(f"Warning: Empty image file at {image_path}")
                    return encoded
                except Exception as e:
                    print(f"Error encoding image {image_path}: {str(e)}")
                    return None
        except Exception as e:
            print(f"Error accessing image {image_path}: {str(e)}")
            return None
    
    # Start building HTML
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Meditation EEG Analysis Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1, h2, h3, h4 {
                color: #2c3e50;
            }
            h1 {
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
                margin-top: 30px;
            }
            .section {
                background-color: white;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .img-container {
                margin: 20px 0;
                text-align: center;
            }
            .img-container img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .img-caption {
                font-size: 0.9em;
                color: #7f8c8d;
                margin-top: 10px;
            }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .flex-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
                gap: 20px;
                margin: 20px 0;
            }
            .flex-item {
                flex: 1;
                min-width: 300px;
                max-width: 100%;
                text-align: center;
            }
            .flex-item img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .tabs {
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
                border-radius: 5px 5px 0 0;
            }
            .tabs button {
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                font-size: 17px;
            }
            .tabs button:hover {
                background-color: #ddd;
            }
            .tabs button.active {
                background-color: #3498db;
                color: white;
            }
            .tabcontent {
                display: none;
                padding: 6px 12px;
                border: 1px solid #ccc;
                border-top: none;
                border-radius: 0 0 5px 5px;
                animation: fadeEffect 1s;
            }
            @keyframes fadeEffect {
                from {opacity: 0;}
                to {opacity: 1;}
            }
        </style>
    </head>
    <body>
        <header>
            <div class="container">
                <h1>Meditation EEG Analysis Dashboard</h1>
                <p>Analytical results from ds001787-download meditation dataset</p>
            </div>
        </header>
        
        <div class="container">
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This dashboard presents the results of a comprehensive EEG analysis of meditation data, focusing on the neural correlates of concentration and mind wandering states during meditation. The analysis includes topographic brain mapping, time-frequency analysis, functional connectivity, and machine learning classification.</p>
                <p>Based on the Brandmeyer & Delorme (2016) paper, we analyzed expert (≥10,000 hours) and novice (100-300 hours) meditators separately, as they show different patterns of brain activity. Analysis was performed using longer epochs (90 seconds before each probe) to capture the full meditation episode as described in the dataset information.</p>
                <p><strong>Note:</strong> Power values are reported in decibels (dB) relative to 1 µV²/Hz.</p>
            </div>
    """
    
    # Create tabs for expert vs novice analysis
    html += """
            <div class="section">
                <h2>Expert vs Novice Comparison</h2>
                <p>This section compares the key brain activity patterns observed in expert and novice meditators during concentration and mind-wandering states.</p>
    """
    
    # Expert vs Novice comparison images
    comparison_images = [
        (topo_dir / "expert_vs_novice_comparison.png", "Expert vs Novice: Band Power Comparison"), 
        (ml_dir / "expert_vs_novice_classification.png", "Expert vs Novice: Classification Performance")
    ]
    
    for img_path, caption in comparison_images:
        img_data = get_image_base64(img_path)
        if img_data:
            html += f"""
                <div class="img-container">
                    <img src="data:image/png;base64,{img_data}" alt="{caption}">
                    <div class="img-caption">{caption}</div>
                </div>
            """
    
    # Close the Expert vs Novice section
    html += """
            </div>
    """
    
    # Create tabs for expert and novice data
    html += """
            <div class="section">
                <h2>Detailed Analysis by Expertise Level</h2>
                <div class="tabs">
                    <button class="tablinks" onclick="openTab(event, 'ExpertTab')" id="defaultOpen">Expert Meditators</button>
                    <button class="tablinks" onclick="openTab(event, 'NoviceTab')">Novice Meditators</button>
                    <button class="tablinks" onclick="openTab(event, 'TraitClassificationTab')">Expert vs Novice Classification</button>
                </div>
    """
    
    # Expert Tab Content
    html += """
                <div id="ExpertTab" class="tabcontent" style="display: block;">
    """
    
    # Add Expert Topographic Maps Section
    html += """
                    <h3>Topographic Brain Maps (Expert Meditators)</h3>
                    <p>Spatial distribution of brain activity during concentration and mind wandering states across different frequency bands for expert meditators.</p>
    """
    
    for band in ['alpha', 'theta', 'beta', 'delta', 'gamma']:
        topo_img_path = topo_dir / f"expert_{band}_topomap.png"
        img_data = get_image_base64(topo_img_path)
        if img_data:
            html += f"""
                    <h4>{band.capitalize()} Band</h4>
                    <div class="img-container">
                        <img src="data:image/png;base64,{img_data}" alt="Expert {band} Topomap">
                        <div class="img-caption">Expert meditators: {band.capitalize()} power distribution (dB). Concentration & Mind Wandering maps (left, middle) share a common color scale for direct comparison. The Difference map (right, Conc-Wand) shows the contrast between states.</div>
                    </div>
            """
    
    # Add Expert Time-Frequency Analysis Section
    html += """
                    <h3>Time-Frequency Analysis (Expert Meditators)</h3>
                    <p>Temporal dynamics of brain rhythms before experts reported concentration or mind wandering states.</p>
                    <div class="grid">
    """
    
    # Add expert grand average plots for all frequency bands
    for band in ['alpha', 'theta', 'beta', 'delta', 'gamma']:
        ga_img = tf_dir / f"expert_grand_average_{band}_power.png"
        img_data = get_image_base64(ga_img)
        if img_data:
            html += f"""
                        <div>
                            <h4>{band.capitalize()} Power</h4>
                            <div class="img-container">
                                <img src="data:image/png;base64,{img_data}" alt="Expert {band} Power">
                                <div class="img-caption">Expert meditators: {band.capitalize()} power over time before responses.</div>
                            </div>
                        </div>
            """
    
    html += """
                    </div>
                    
                    <h4>Statistical Time Course (Expert Meditators)</h4>
                    <div class="grid">
    """
    
    # Add expert t-value plots for all frequency bands
    for band in ['alpha', 'theta', 'beta', 'delta', 'gamma']:
        t_img = tf_dir / f"expert_t_values_{band}_power.png"
        img_data = get_image_base64(t_img)
        if img_data:
            html += f"""
                        <div>
                            <div class="img-container">
                                <img src="data:image/png;base64,{img_data}" alt="Expert {band} T-values">
                                <div class="img-caption">Expert meditators: T-values for {band} power differences over time.</div>
                            </div>
                        </div>
            """
            
    # Also add expert difference plots
    html += """
                    </div>
                    
                    <h4>Difference Between Meditation States (Expert Meditators)</h4>
                    <div class="grid">
    """
    
    for band in ['alpha', 'theta', 'beta', 'delta', 'gamma']:
        diff_img = tf_dir / f"expert_difference_{band}_power.png"
        img_data = get_image_base64(diff_img)
        if img_data:
            html += f"""
                        <div>
                            <div class="img-container">
                                <img src="data:image/png;base64,{img_data}" alt="Expert {band} Difference">
                                <div class="img-caption">Expert meditators: Difference in {band} power (Concentration-Mind Wandering) over time.</div>
                            </div>
                        </div>
            """
    
    html += """
                    </div>
    """
    
    # Add Expert Connectivity Analysis Section
    html += """
                    <h3>Connectivity Analysis (Expert Meditators)</h3>
                    <p>Functional connectivity between brain regions in concentration vs. mind wandering states for expert meditators.</p>
                    
                    <h4>Connectivity Matrices (Expert Meditators)</h4>
                    <p>These matrices show the strength of connectivity between different brain regions during concentration (left), mind wandering (middle), and their difference (right).</p>
    """
    
    # Add expert connectivity matrices for all frequency bands
    for band in ['alpha', 'theta', 'beta', 'delta', 'gamma']:
        matrix_img_path = connectivity_dir / "expert" / f"{band}_connectivity_matrices.png"
        matrix_img_data = get_image_base64(matrix_img_path)
        
        if matrix_img_data:
            html += f"""
                    <div class="img-container">
                        <img src="data:image/png;base64,{matrix_img_data}" alt="Expert {band} connectivity matrices">
                        <div class="img-caption">Expert meditators: {band.capitalize()} band connectivity matrices showing concentration (left), mind wandering (middle), and their difference (right).</div>
                    </div>
            """
    
    # Add Expert Network Visualizations
    html += """
                    <h4>Brain Network Visualizations (Expert Meditators)</h4>
                    <p>These network graphs show the strongest connections between brain regions during concentration and mind wandering states for expert meditators.</p>
    """
    
    # Add expert network visualizations for all frequency bands
    for band in ['alpha', 'theta', 'beta', 'delta', 'gamma']:
        conc_network_path = connectivity_dir / "expert" / f"{band}_concentration_network.png"
        wander_network_path = connectivity_dir / "expert" / f"{band}_wandering_network.png"
        
        conc_img_data = get_image_base64(conc_network_path)
        wander_img_data = get_image_base64(wander_network_path)
        
        if conc_img_data and wander_img_data:
            html += f"""
                    <h4>{band.capitalize()} Band Networks (Expert Meditators)</h4>
                    <div class="flex-container">
                        <div class="flex-item">
                            <img src="data:image/png;base64,{conc_img_data}" alt="Expert {band} concentration network">
                            <div class="img-caption">Concentration state</div>
                        </div>
                        <div class="flex-item">
                            <img src="data:image/png;base64,{wander_img_data}" alt="Expert {band} wandering network">
                            <div class="img-caption">Mind wandering state</div>
                        </div>
                    </div>
            """
    
    # Add Expert Machine Learning Section
    html += """
                    <h3>Machine Learning Classification (Expert Meditators)</h3>
                    <p>Automated classification of concentration vs. mind wandering states using EEG features from expert meditators.</p>
    """
    
    # Expert Within-subject classification results
    within_img = ml_dir / "expert" / "expert_within_subject_accuracy.png"
    img_data = get_image_base64(within_img)
    if img_data:
        html += f"""
                    <h4>Within-Subject Classification (Expert Meditators)</h4>
                    <div class="img-container">
                        <img src="data:image/png;base64,{img_data}" alt="Expert Within-Subject Classification">
                        <div class="img-caption">Classification accuracy for each individual expert subject.</div>
                    </div>
        """
    
    # Expert Cross-subject classification results
    cross_img = ml_dir / "expert" / "expert_cross_subject_accuracy.png"
    img_data = get_image_base64(cross_img)
    if img_data:
        html += f"""
                    <h4>Cross-Subject Classification (Expert Meditators)</h4>
                    <div class="img-container">
                        <img src="data:image/png;base64,{img_data}" alt="Expert Cross-Subject Classification">
                        <div class="img-caption">Comparison of different classifiers for cross-subject classification of expert meditators.</div>
                    </div>
        """
    
    # Expert Feature importance
    feat_img = ml_dir / "expert" / "expert_top_feature_importance.png"
    img_data = get_image_base64(feat_img)
    if img_data:
        html += f"""
                    <h4>Feature Importance (Expert Meditators)</h4>
                    <div class="img-container">
                        <img src="data:image/png;base64,{img_data}" alt="Expert Feature Importance">
                        <div class="img-caption">Most important EEG features for classifying meditation states in expert meditators.</div>
                    </div>
        """
    
    # Close the Expert Tab
    html += """
                </div>
    """
    
    # Novice Tab Content
    html += """
                <div id="NoviceTab" class="tabcontent">
    """
    
    # Add Novice Topographic Maps Section
    html += """
                    <h3>Topographic Brain Maps (Novice Meditators)</h3>
                    <p>Spatial distribution of brain activity during concentration and mind wandering states across different frequency bands for novice meditators.</p>
    """
    
    for band in ['alpha', 'theta', 'beta', 'delta', 'gamma']:
        topo_img_path = topo_dir / f"novice_{band}_topomap.png"
        img_data = get_image_base64(topo_img_path)
        if img_data:
            html += f"""
                    <h4>{band.capitalize()} Band</h4>
                    <div class="img-container">
                        <img src="data:image/png;base64,{img_data}" alt="Novice {band} Topomap">
                        <div class="img-caption">Novice meditators: {band.capitalize()} power distribution (dB). Concentration & Mind Wandering maps (left, middle) share a common color scale for direct comparison. The Difference map (right, Conc-Wand) shows the contrast between states.</div>
                    </div>
            """
    
    # Add Novice Time-Frequency Analysis Section
    html += """
                    <h3>Time-Frequency Analysis (Novice Meditators)</h3>
                    <p>Temporal dynamics of brain rhythms before novices reported concentration or mind wandering states.</p>
                    <div class="grid">
    """
    
    # Add novice grand average plots for all frequency bands
    for band in ['alpha', 'theta', 'beta', 'delta', 'gamma']:
        ga_img = tf_dir / f"novice_grand_average_{band}_power.png"
        img_data = get_image_base64(ga_img)
        if img_data:
            html += f"""
                        <div>
                            <h4>{band.capitalize()} Power</h4>
                            <div class="img-container">
                                <img src="data:image/png;base64,{img_data}" alt="Novice {band} Power">
                                <div class="img-caption">Novice meditators: {band.capitalize()} power over time before responses.</div>
                            </div>
                        </div>
            """
    
    html += """
                    </div>
                    
                    <h4>Statistical Time Course (Novice Meditators)</h4>
                    <div class="grid">
    """
    
    # Add novice t-value plots for all frequency bands
    for band in ['alpha', 'theta', 'beta', 'delta', 'gamma']:
        t_img = tf_dir / f"novice_t_values_{band}_power.png"
        img_data = get_image_base64(t_img)
        if img_data:
            html += f"""
                        <div>
                            <div class="img-container">
                                <img src="data:image/png;base64,{img_data}" alt="Novice {band} T-values">
                                <div class="img-caption">Novice meditators: T-values for {band} power differences over time.</div>
                            </div>
                        </div>
            """
            
    # Also add novice difference plots
    html += """
                    </div>
                    
                    <h4>Difference Between Meditation States (Novice Meditators)</h4>
                    <div class="grid">
    """
    
    for band in ['alpha', 'theta', 'beta', 'delta', 'gamma']:
        diff_img = tf_dir / f"novice_difference_{band}_power.png"
        img_data = get_image_base64(diff_img)
        if img_data:
            html += f"""
                        <div>
                            <div class="img-container">
                                <img src="data:image/png;base64,{img_data}" alt="Novice {band} Difference">
                                <div class="img-caption">Novice meditators: Difference in {band} power (Concentration-Mind Wandering) over time.</div>
                            </div>
                        </div>
            """
    
    html += """
                    </div>
    """
    
    # Add Novice Connectivity Analysis Section
    html += """
                    <h3>Connectivity Analysis (Novice Meditators)</h3>
                    <p>Functional connectivity between brain regions in concentration vs. mind wandering states for novice meditators.</p>
                    
                    <h4>Connectivity Matrices (Novice Meditators)</h4>
                    <p>These matrices show the strength of connectivity between different brain regions during concentration (left), mind wandering (middle), and their difference (right).</p>
    """
    
    # Add novice connectivity matrices for all frequency bands
    for band in ['alpha', 'theta', 'beta', 'delta', 'gamma']:
        matrix_img_path = connectivity_dir / "novice" / f"{band}_connectivity_matrices.png"
        matrix_img_data = get_image_base64(matrix_img_path)
        
        if matrix_img_data:
            html += f"""
                    <div class="img-container">
                        <img src="data:image/png;base64,{matrix_img_data}" alt="Novice {band} connectivity matrices">
                        <div class="img-caption">Novice meditators: {band.capitalize()} band connectivity matrices showing concentration (left), mind wandering (middle), and their difference (right).</div>
                    </div>
            """
    
    # Add Novice Network Visualizations
    html += """
                    <h4>Brain Network Visualizations (Novice Meditators)</h4>
                    <p>These network graphs show the strongest connections between brain regions during concentration and mind wandering states for novice meditators.</p>
    """
    
    # Add novice network visualizations for all frequency bands
    for band in ['alpha', 'theta', 'beta', 'delta', 'gamma']:
        conc_network_path = connectivity_dir / "novice" / f"{band}_concentration_network.png"
        wander_network_path = connectivity_dir / "novice" / f"{band}_wandering_network.png"
        
        conc_img_data = get_image_base64(conc_network_path)
        wander_img_data = get_image_base64(wander_network_path)
        
        if conc_img_data and wander_img_data:
            html += f"""
                    <h4>{band.capitalize()} Band Networks (Novice Meditators)</h4>
                    <div class="flex-container">
                        <div class="flex-item">
                            <img src="data:image/png;base64,{conc_img_data}" alt="Novice {band} concentration network">
                            <div class="img-caption">Concentration state</div>
                        </div>
                        <div class="flex-item">
                            <img src="data:image/png;base64,{wander_img_data}" alt="Novice {band} wandering network">
                            <div class="img-caption">Mind wandering state</div>
                        </div>
                    </div>
            """
    
    # Add Novice Machine Learning Section
    html += """
                    <h3>Machine Learning Classification (Novice Meditators)</h3>
                    <p>Automated classification of concentration vs. mind wandering states using EEG features from novice meditators.</p>
    """
    
    # Novice Within-subject classification results
    within_img = ml_dir / "novice" / "novice_within_subject_accuracy.png"
    img_data = get_image_base64(within_img)
    if img_data:
        html += f"""
                    <h4>Within-Subject Classification (Novice Meditators)</h4>
                    <div class="img-container">
                        <img src="data:image/png;base64,{img_data}" alt="Novice Within-Subject Classification">
                        <div class="img-caption">Classification accuracy for each individual novice subject.</div>
                    </div>
        """
    
    # Novice Cross-subject classification results
    cross_img = ml_dir / "novice" / "novice_cross_subject_accuracy.png"
    img_data = get_image_base64(cross_img)
    if img_data:
        html += f"""
                    <h4>Cross-Subject Classification (Novice Meditators)</h4>
                    <div class="img-container">
                        <img src="data:image/png;base64,{img_data}" alt="Novice Cross-Subject Classification">
                        <div class="img-caption">Comparison of different classifiers for cross-subject classification of novice meditators.</div>
                    </div>
        """
    
    # Novice Feature importance
    feat_img = ml_dir / "novice" / "novice_top_feature_importance.png"
    img_data = get_image_base64(feat_img)
    if img_data:
        html += f"""
                    <h4>Feature Importance (Novice Meditators)</h4>
                    <div class="img-container">
                        <img src="data:image/png;base64,{img_data}" alt="Novice Feature Importance">
                        <div class="img-caption">Most important EEG features for classifying meditation states in novice meditators.</div>
                    </div>
        """
    
    # Close the Novice Tab
    html += """
                </div>
    """
    
    # Add the Expert vs Novice Trait Classification Tab
    html += """
                <div id="TraitClassificationTab" class="tabcontent">
                    <h3>Expert vs Novice Classification</h3>
                    <p>Machine learning classification of meditation expertise (expert vs. novice) based on brain activity patterns, regardless of meditation state.</p>
                    
                    <h4>Classification Performance</h4>
                    <p>The accuracy of different machine learning algorithms in distinguishing expert from novice meditators.</p>
    """
    
    # Add expertise classification accuracy
    accuracy_img = ml_dir / "expertise_classification" / "expertise_classification_accuracy.png"
    img_data = get_image_base64(accuracy_img)
    if img_data:
        html += f"""
                    <div class="img-container">
                        <img src="data:image/png;base64,{img_data}" alt="Expert vs Novice Classification Accuracy">
                        <div class="img-caption">Accuracy of different classifiers in distinguishing expert from novice meditators based on EEG patterns.</div>
                    </div>
        """
    
    # Add confusion matrix
    conf_img = ml_dir / "expertise_classification" / "expertise_confusion_matrix.png"
    img_data = get_image_base64(conf_img)
    if img_data:
        html += f"""
                    <h4>Confusion Matrix</h4>
                    <div class="img-container">
                        <img src="data:image/png;base64,{img_data}" alt="Expert vs Novice Confusion Matrix">
                        <div class="img-caption">Confusion matrix showing how often the classifier correctly or incorrectly identified expert and novice meditators.</div>
                    </div>
        """
    
    # Add feature importance
    feat_img = ml_dir / "expertise_classification" / "expertise_feature_importance.png"
    img_data = get_image_base64(feat_img)
    if img_data:
        html += f"""
                    <h4>Feature Importance</h4>
                    <div class="img-container">
                        <img src="data:image/png;base64,{img_data}" alt="Expert vs Novice Feature Importance">
                        <div class="img-caption">Most important EEG features (channels and frequency bands) for distinguishing expert from novice meditators.</div>
                    </div>
        """
        
    # Add channel-specific or band-specific models section
    html += """
                    <h4>Channel-Specific and Frequency Band-Specific Models</h4>
                    <p>Comparison of classification performance when using only specific brain regions or frequency bands.</p>
                    <ul>
                        <li><strong>Frontal-Midline Theta:</strong> Classification using only theta power from frontal-midline electrodes (Fz, FCz, Cz)</li>
                        <li><strong>Central/Somatosensory Alpha:</strong> Classification using only alpha power from central/somatosensory electrodes (C3, C4, CP3, CP4)</li>
                        <li><strong>Posterior Alpha:</strong> Classification using only alpha power from posterior electrodes (P3, Pz, P4, O1, Oz, O2)</li>
                        <li><strong>RF-Selected Features:</strong> Classification using only the top features identified by Random Forest feature importance</li>
                    </ul>
    """
    
    # Close the Trait Classification Tab
    html += """
                </div>
            </div>
    """
    
    # Add conclusions
    html += """
            <div class="section">
                <h2>Conclusions</h2>
                <ul>
                    <li><strong>Expert vs. Novice Differences:</strong> Expert meditators show stronger and more consistent neural signatures between concentration and mind-wandering states, particularly in frontal-midline theta and central/posterior alpha power.</li>
                    <li><strong>Topographic Analysis:</strong> Spatial patterns of alpha and theta power show distinct differences between concentration and mind wandering states, with patterns differing between expert and novice meditators.</li>
                    <li><strong>Time-Frequency Analysis:</strong> Temporal dynamics reveal how brain rhythms evolve before participants report different meditation states, with experts showing more sustained changes.</li>
                    <li><strong>Connectivity Analysis:</strong> Expert meditators show stronger frontal connectivity during concentration, while novices show less differentiation between states.</li>
                    <li><strong>Machine Learning:</strong> Classification accuracy is generally higher for expert meditators, suggesting more reliable neural markers of meditation states with increased meditation experience.</li>
                </ul>
                <p>These findings are consistent with the Brandmeyer & Delorme (2016) paper, which reported increased frontal midline theta and somatosensory alpha rhythms during meditation compared to mind wandering, particularly in expert practitioners.</p>
            </div>
            
            <div class="footer">
                <p>Generated from meditation dataset ds001787-download | NYU Brain-Mind-Consciousness Lab</p>
            </div>
        </div>

        <script>
        function openTab(evt, tabName) {
          var i, tabcontent, tablinks;
          tabcontent = document.getElementsByClassName("tabcontent");
          for (i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
          }
          tablinks = document.getElementsByClassName("tablinks");
          for (i = 0; i < tablinks.length; i++) {
            tablinks[i].className = tablinks[i].className.replace(" active", "");
          }
          document.getElementById(tabName).style.display = "block";
          evt.currentTarget.className += " active";
        }
        </script>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(dashboard_dir / "meditation_dashboard.html", "w") as f:
        f.write(html)
    
    print(f"Dashboard created: {dashboard_dir / 'meditation_dashboard.html'}")
    
    return dashboard_dir / "meditation_dashboard.html"

def run_analysis(run_all=False, output_base_dir="meditation_analysis_output", debug=False):
    # Create the base output directory if it doesn't exist
    base_dir = Path(output_base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if run_all:
        print("Running all analysis scripts...")
        for script in analysis_scripts:
            script_path = Path(__file__).parent / script
            if script_path.exists():
                print(f"\n--- Running {script} ---")
                # Construct command with debug flag if set
                cmd = ["python", str(script_path)]
                if debug:
                    cmd.append("--debug")
                
                try:
                    # Run the script and capture output
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(result.stdout)
                    if result.stderr:
                        print("--- STDERR ---")
                        print(result.stderr)
                except subprocess.CalledProcessError as e:
                    print(f"Error running {script}: {e}")
                    print("--- STDOUT ---:")
                    print(e.stdout)
                    print("--- STDERR ---:")
                    print(e.stderr)
            else:
                print(f"Script {script} not found.")
        print("\n--- All analyses complete ---")
    else:
        print("Skipping analysis runs. Use --run-all to execute scripts.")
    
    # --- Dashboard Generation ---
    print("Generating HTML dashboard...")
    dashboard_path = create_dashboard(output_base_dir)
    
    return dashboard_path

# Define analysis scripts
analysis_scripts = [
    "topo_maps_analysis.py",
    "time_frequency_analysis.py", 
    "connectivity_analysis.py",
    "ml_classification.py"
]

# Define output directories
output_dirs = {
    "topo_maps_analysis.py": "topo_maps",
    "time_frequency_analysis.py": "time_frequency",
    "connectivity_analysis.py": "connectivity",
    "ml_classification.py": "ml_classification"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Meditation EEG Analysis Dashboard")
    parser.add_argument("--run-all", action="store_true", help="Run all analysis scripts before generating the dashboard.")
    parser.add_argument("--output-dir", default="meditation_analysis_output", help="Base directory for analysis outputs and dashboard.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for analysis scripts.")
    args = parser.parse_args()
    
    # Pass the debug flag to the run_analysis function
    run_analysis(run_all=args.run_all, output_base_dir=args.output_dir, debug=args.debug)
    
    print(f"\nAll done! Open the dashboard at: {Path(args.output_dir) / 'dashboard' / 'meditation_dashboard.html'}")
    print("Use: --run-all flag to run all analyses before generating dashboard")
    print("Example: python meditation_dashboard.py --run-all") 