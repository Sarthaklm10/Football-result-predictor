"""
Steps 9 & 10: Baseline Model + Train Multiple Models
Trains LR, Decision Tree, Random Forest, XGBoost.
Compares all using accuracy, precision, recall, F1, confusion matrix.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
import joblib
import json
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'

# ── Load Preprocessed Data ───────────────────────────────────
print("Loading preprocessed data...")
X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv').values.ravel()
y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv').values.ravel()
label_encoder = joblib.load(MODELS_DIR / 'label_encoder.joblib')

print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
print(f"  Test:  {X_test.shape[0]:,} samples")
print(f"  Classes: {list(label_encoder.classes_)}")


# ══════════════════════════════════════════════════════════════
# HELPER: Evaluate a Model
# ══════════════════════════════════════════════════════════════

def evaluate_model(model, X_train, X_test, y_train, y_test,
                   model_name: str, label_encoder) -> dict:
    """Train, predict, and evaluate a model. Returns dict of metrics."""
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Cross-validation on training data (5-fold)
    # This checks if the model generalizes or just memorizes
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='f1_macro', n_jobs=-1)

    # Print results
    class_names = list(label_encoder.classes_)
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} (macro)")
    print(f"  Recall:    {rec:.4f} (macro)")
    print(f"  F1 Score:  {f1:.4f} (macro)")
    print(f"  CV F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=class_names, digits=4))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(f"  {'Predicted ->':>15} {class_names[0]:>10} {class_names[1]:>10} {class_names[2]:>10}")
    for i, cls in enumerate(class_names):
        print(f"  {'Actual ' + cls:>15} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")

    return {
        'model_name': model_name,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_macro': round(f1, 4),
        'cv_f1_mean': round(cv_scores.mean(), 4),
        'cv_f1_std': round(cv_scores.std(), 4),
    }


# ── Baseline: Logistic Regression ────────────────────────────

print("\n" + "#" * 60)
print("# STEP 9: Baseline Model — Logistic Regression")
print("#" * 60)

lr_model = LogisticRegression(
    max_iter=1000,       # Allow enough iterations to converge
    random_state=42,     # Reproducibility
    class_weight='balanced',  # Handle class imbalance
    solver='lbfgs'       # Optimizer that works with multiclass
)

lr_results = evaluate_model(
    lr_model, X_train, X_test, y_train, y_test,
    "Logistic Regression (Baseline)", label_encoder
)


# ══════════════════════════════════════════════════════════════
# STEP 10: Train Multiple Models
# ══════════════════════════════════════════════════════════════

print("\n" + "#" * 60)
print("# STEP 10: Training Multiple Models")
print("#" * 60)


# ── Decision Tree ────────────────────────────────────────────

dt_model = DecisionTreeClassifier(
    max_depth=10,           # Control overfitting
    min_samples_split=20,   # Minimum samples for split
    min_samples_leaf=10,    # Minimum samples per leaf
    random_state=42,
    class_weight='balanced'
)

dt_results = evaluate_model(
    dt_model, X_train, X_test, y_train, y_test,
    "Decision Tree", label_encoder
)


# ── Random Forest ────────────────────────────────────────────

rf_model = RandomForestClassifier(
    n_estimators=200,       # Number of trees
    max_depth=15,           # Tree depth limit
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_results = evaluate_model(
    rf_model, X_train, X_test, y_train, y_test,
    "Random Forest", label_encoder
)


# ── XGBoost ──────────────────────────────────────────────────

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',  # Multiclass log loss
    use_label_encoder=False,
    verbosity=0              # Suppress warnings
)

xgb_results = evaluate_model(
    xgb_model, X_train, X_test, y_train, y_test,
    "XGBoost", label_encoder
)


# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

all_results = [lr_results, dt_results, rf_results, xgb_results]
comparison_df = pd.DataFrame(all_results)
comparison_df = comparison_df.sort_values('f1_macro', ascending=False)

print(f"\n{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'CV F1':>10}")
print("-" * 80)
for _, row in comparison_df.iterrows():
    print(f"{row['model_name']:<30} {row['accuracy']:>10.4f} {row['precision']:>10.4f} "
          f"{row['recall']:>8.4f} {row['f1_macro']:>8.4f} {row['cv_f1_mean']:>7.4f}+/-{row['cv_f1_std']:.4f}")

best_model_name = comparison_df.iloc[0]['model_name']
best_f1 = comparison_df.iloc[0]['f1_macro']
print(f"\nBest model: {best_model_name} (F1 = {best_f1:.4f})")
print(f"Dummy baseline (always Home Win): ~49% accuracy")

# Save comparison
comparison_df.to_csv(REPORTS_DIR / 'model_comparison.csv', index=False)

# Save the best model (we'll tune it in Step 12)
# For now, save all models
joblib.dump(lr_model, MODELS_DIR / 'logistic_regression.joblib')
joblib.dump(dt_model, MODELS_DIR / 'decision_tree.joblib')
joblib.dump(rf_model, MODELS_DIR / 'random_forest.joblib')
joblib.dump(xgb_model, MODELS_DIR / 'xgboost.joblib')

print(f"\nAll models saved to models/ directory")
print(f"Comparison saved to reports/model_comparison.csv")

print("\n" + "=" * 60)
print("STEPS 9-10 COMPLETE")
print("=" * 60)
