import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def ensure_dir_exists(dir_path):
    """Ensure directory exists"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    
def create_placeholder_image(filepath, title="No Data Available", width=12, height=6):
    """Create a placeholder image with error message"""
    plt.figure(figsize=(width, height))
    plt.text(0.5, 0.5, title, 
            horizontalalignment='center', verticalalignment='center', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Created placeholder image: {filepath}")

def create_placeholder_csv(filepath):
    """Create a placeholder CSV file with dummy data"""
    times = np.linspace(-90.0, 0.0, 20)
    df = pd.DataFrame({
        'Time': times,
        'Concentration_Mean': np.zeros(20),
        'Concentration_SEM': np.zeros(20),
        'Mind_Wandering_Mean': np.zeros(20),
        'Mind_Wandering_SEM': np.zeros(20),
        'Difference_Mean': np.zeros(20),
        'Difference_SEM': np.zeros(20),
        't_value': np.zeros(20),
        'p_value': np.ones(20) * 0.5
    })
    df.to_csv(filepath, index=False)
    print(f"Created placeholder CSV: {filepath}")

def fix_time_frequency_files():
    """Create missing time-frequency analysis files"""
    base_dir = "meditation_analysis_output"
    tf_dir = Path(base_dir) / "time_frequency"
    ensure_dir_exists(tf_dir)
    
    # Create expert and novice subdirectories
    expert_dir = tf_dir / "expert"
    novice_dir = tf_dir / "novice"
    ensure_dir_exists(expert_dir)
    ensure_dir_exists(novice_dir)
    
    # Define frequency bands
    bands = ['alpha', 'theta', 'beta', 'delta', 'gamma']
    
    # Create expert vs novice comparison visualization
    create_placeholder_image(tf_dir / "expert_vs_novice_tf_comparison.png", 
                            "Expert vs Novice: Alpha and Theta Power during Meditation")
    
    # Create files for each expertise group
    for expertise, out_dir in [('expert', expert_dir), ('novice', novice_dir)]:
        for band in bands:
            # Grand average plots
            grand_avg_path = tf_dir / f"{expertise}_grand_average_{band}_power.png"
            if not os.path.exists(grand_avg_path):
                create_placeholder_image(grand_avg_path, 
                                        f"{expertise.capitalize()}: Grand Average {band.capitalize()} Power")
            
            # T-values plots
            t_values_path = tf_dir / f"{expertise}_t_values_{band}_power.png"
            if not os.path.exists(t_values_path):
                create_placeholder_image(t_values_path, 
                                        f"{expertise.capitalize()}: T-values for {band.capitalize()} Power")
            
            # Difference plots
            diff_path = tf_dir / f"{expertise}_difference_{band}_power.png"
            if not os.path.exists(diff_path):
                create_placeholder_image(diff_path, 
                                        f"{expertise.capitalize()}: Difference in {band.capitalize()} Power")
            
            # CSV files
            csv_path = tf_dir / f"{expertise}_{band}_power_results.csv"
            if not os.path.exists(csv_path):
                create_placeholder_csv(csv_path)
            
            # Also create in expertise directory
            # Grand average plots in expertise dir
            grand_avg_path = out_dir / f"{expertise}_grand_average_{band}_power.png"
            if not os.path.exists(grand_avg_path):
                create_placeholder_image(grand_avg_path, 
                                        f"{expertise.capitalize()}: Grand Average {band.capitalize()} Power")
            
            # T-values plots in expertise dir
            t_values_path = out_dir / f"{expertise}_t_values_{band}_power.png"
            if not os.path.exists(t_values_path):
                create_placeholder_image(t_values_path, 
                                        f"{expertise.capitalize()}: T-values for {band.capitalize()} Power")
            
            # Difference plots in expertise dir
            diff_path = out_dir / f"{expertise}_difference_{band}_power.png"
            if not os.path.exists(diff_path):
                create_placeholder_image(diff_path, 
                                        f"{expertise.capitalize()}: Difference in {band.capitalize()} Power")
            
            # CSV files in expertise dir
            csv_path = out_dir / f"{expertise}_{band}_power_results.csv"
            if not os.path.exists(csv_path):
                create_placeholder_csv(csv_path)
                
    print("Time-frequency analysis files created successfully")

def fix_ml_classification_files():
    """Create missing machine learning classification files"""
    base_dir = "meditation_analysis_output"
    ml_dir = Path(base_dir) / "ml_classification"
    ensure_dir_exists(ml_dir)
    
    # Create subdirectories
    expert_dir = ml_dir / "expert"
    novice_dir = ml_dir / "novice"
    expertise_dir = ml_dir / "expertise_classification"
    ensure_dir_exists(expert_dir)
    ensure_dir_exists(novice_dir)
    ensure_dir_exists(expertise_dir)
    
    # Create expert vs novice comparison
    create_placeholder_image(ml_dir / "expert_vs_novice_classification.png",
                           "Expert vs Novice Classification Performance")
    
    # Create files for each expertise group
    for expertise, out_dir in [('expert', expert_dir), ('novice', novice_dir)]:
        # Within-subject accuracy
        within_path = ml_dir / f"{expertise}_within_subject_accuracy.png"
        if not os.path.exists(within_path):
            create_placeholder_image(within_path, 
                                   f"{expertise.capitalize()} Within-Subject Classification Accuracy")
        
        within_path = out_dir / f"{expertise}_within_subject_accuracy.png"
        if not os.path.exists(within_path):
            create_placeholder_image(within_path, 
                                   f"{expertise.capitalize()} Within-Subject Classification Accuracy")
        
        # Cross-subject accuracy
        cross_path = ml_dir / f"{expertise}_cross_subject_accuracy.png"
        if not os.path.exists(cross_path):
            create_placeholder_image(cross_path, 
                                   f"{expertise.capitalize()} Cross-Subject Classification Accuracy")
        
        cross_path = out_dir / f"{expertise}_cross_subject_accuracy.png"
        if not os.path.exists(cross_path):
            create_placeholder_image(cross_path, 
                                   f"{expertise.capitalize()} Cross-Subject Classification Accuracy")
        
        # Feature importance
        feat_path = ml_dir / f"{expertise}_top_feature_importance.png"
        if not os.path.exists(feat_path):
            create_placeholder_image(feat_path, 
                                   f"{expertise.capitalize()} Top Feature Importance")
        
        feat_path = out_dir / f"{expertise}_top_feature_importance.png"
        if not os.path.exists(feat_path):
            create_placeholder_image(feat_path, 
                                   f"{expertise.capitalize()} Top Feature Importance")
        
        # Confusion matrix
        conf_path = ml_dir / f"{expertise}_confusion_matrix.png"
        if not os.path.exists(conf_path):
            create_placeholder_image(conf_path, 
                                   f"{expertise.capitalize()} Confusion Matrix")
        
        conf_path = out_dir / f"{expertise}_confusion_matrix.png"
        if not os.path.exists(conf_path):
            create_placeholder_image(conf_path, 
                                   f"{expertise.capitalize()} Confusion Matrix")
        
        # ROC curve
        roc_path = ml_dir / f"{expertise}_roc_curve.png"
        if not os.path.exists(roc_path):
            create_placeholder_image(roc_path, 
                                   f"{expertise.capitalize()} ROC Curve")
        
        roc_path = out_dir / f"{expertise}_roc_curve.png"
        if not os.path.exists(roc_path):
            create_placeholder_image(roc_path, 
                                   f"{expertise.capitalize()} ROC Curve")
        
        # CSV files
        csv_path = ml_dir / f"{expertise}_classification_report.csv"
        if not os.path.exists(csv_path):
            df = pd.DataFrame({
                'precision': [0.0, 0.0, 0.0],
                'recall': [0.0, 0.0, 0.0],
                'f1-score': [0.0, 0.0, 0.0],
                'support': [0, 0, 0]
            }, index=['Concentration', 'Mind Wandering', 'accuracy'])
            df.to_csv(csv_path)
        
        csv_path = out_dir / f"{expertise}_classification_report.csv"
        if not os.path.exists(csv_path):
            df = pd.DataFrame({
                'precision': [0.0, 0.0, 0.0],
                'recall': [0.0, 0.0, 0.0],
                'f1-score': [0.0, 0.0, 0.0],
                'support': [0, 0, 0]
            }, index=['Concentration', 'Mind Wandering', 'accuracy'])
            df.to_csv(csv_path)
    
    # Expertise classification files
    expertise_paths = [
        expertise_dir / "expertise_classification_accuracy.png",
        expertise_dir / "expertise_confusion_matrix.png",
        expertise_dir / "expertise_feature_importance.png",
        ml_dir / "expertise_classification_accuracy.png",
        ml_dir / "expertise_confusion_matrix.png",
        ml_dir / "expertise_feature_importance.png"
    ]
    
    for path in expertise_paths:
        if not os.path.exists(path):
            title = os.path.basename(path).replace(".png", "").replace("_", " ").title()
            create_placeholder_image(path, title)
    
    print("Machine learning classification files created successfully")

if __name__ == "__main__":
    print("Creating missing time-frequency and ML classification files...")
    fix_time_frequency_files()
    fix_ml_classification_files()
    print("All missing files have been created. The dashboard should now load properly.")
    print("Note: These are placeholder images. To generate actual data visualizations, run:")
    print("  python time_frequency_analysis.py")
    print("  python ml_classification.py") 