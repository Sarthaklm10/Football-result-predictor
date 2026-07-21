"""
Step 13: Model Explainability (SHAP)
Generates SHAP summary plots to explain which features drive predictions.
"""

import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'

def main():
    print("=" * 60)
    print("STEP 13: SHAP Explainability")
    print("=" * 60)
    
    print("Loading model and data...")
    model = joblib.load(MODELS_DIR / 'best_model.joblib')
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
    
    # SHAP on Random Forest can be slow, so we use a random sample of 1,000 rows
    print("Calculating SHAP values (using 1,000 samples for speed)...")
    X_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
    
    # TreeExplainer is heavily optimized for tree-based models (RF, XGBoost)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Multiclass SHAP output formats vary by version. We want Class 2 (Home Win).
    print("Generating plot...")
    plt.figure(figsize=(10, 8))
    
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[2], X_sample, show=False)
    elif len(shap_values.shape) == 3:
        shap.summary_plot(shap_values[:, :, 2], X_sample, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, show=False)

    plt.title("SHAP Feature Importance (Predicting 'Home Win')", y=1.05)
    plt.tight_layout()
    
    out_path = REPORTS_DIR / '16_shap_summary.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP summary plot saved to: {out_path}")
    print("\n" + "=" * 60)
    print("STEP 13 COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()