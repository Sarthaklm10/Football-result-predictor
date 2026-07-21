"""
Steps 11 & 12: Model Comparison + Hyperparameter Tuning
Tunes Random Forest and XGBoost using RandomizedSearchCV,
then picks the best overall model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'

# ── Load Data ────────────────────────────────────────────────
print("Loading preprocessed data...")
X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv').values.ravel()
y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv').values.ravel()
label_encoder = joblib.load(MODELS_DIR / 'label_encoder.joblib')

print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
print(f"  Test:  {X_test.shape[0]:,} x {X_test.shape[1]}")


# ── TimeSeriesSplit for CV ───────────────────────────────────
# Regular KFold shuffles data randomly. TimeSeriesSplit respects
# chronological order — each fold trains on earlier data and
# validates on later data. This matches our real use case.
tscv = TimeSeriesSplit(n_splits=5)


# ══════════════════════════════════════════════════════════════
# TUNE: Random Forest
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Tuning Random Forest...")
print("=" * 60)

rf_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [8, 12, 15, 20, None],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [3, 5, 10],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'class_weight': ['balanced', 'balanced_subsample'],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_grid,
    n_iter=40,               # Try 40 random combinations
    cv=tscv,
    scoring='f1_macro',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train, y_train)
print(f"\n  Best RF params: {rf_search.best_params_}")
print(f"  Best RF CV F1:  {rf_search.best_score_:.4f}")

rf_best = rf_search.best_estimator_


# ══════════════════════════════════════════════════════════════
# TUNE: XGBoost
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Tuning XGBoost...")
print("=" * 60)

xgb_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2],
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False,
        verbosity=0
    ),
    param_distributions=xgb_param_grid,
    n_iter=50,               # Try 50 random combinations
    cv=tscv,
    scoring='f1_macro',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)
print(f"\n  Best XGB params: {xgb_search.best_params_}")
print(f"  Best XGB CV F1:  {xgb_search.best_score_:.4f}")

xgb_best = xgb_search.best_estimator_


# ══════════════════════════════════════════════════════════════
# EVALUATE TUNED MODELS ON TEST SET
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Evaluating tuned models on test set...")
print("=" * 60)

class_names = list(label_encoder.classes_)
results = []

for name, model in [("Random Forest (tuned)", rf_best),
                     ("XGBoost (tuned)", xgb_best)]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n  {name}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1 Macro: {f1:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=class_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  {'Predicted ->':>15} {class_names[0]:>10} {class_names[1]:>10} {class_names[2]:>10}")
    for i, cls in enumerate(class_names):
        print(f"  {'Actual ' + cls:>15} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")

    results.append({
        'model': name,
        'accuracy': round(acc, 4),
        'f1_macro': round(f1, 4),
    })

# Also load and eval the previous untuned models for comparison
lr_model = joblib.load(MODELS_DIR / 'logistic_regression.joblib')
y_pred_lr = lr_model.predict(X_test)
results.append({
    'model': 'Logistic Regression (baseline)',
    'accuracy': round(accuracy_score(y_test, y_pred_lr), 4),
    'f1_macro': round(f1_score(y_test, y_pred_lr, average='macro'), 4),
})


# ══════════════════════════════════════════════════════════════
# COMPARISON: Before vs After Tuning
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

# Load old untuned results
old_results = pd.read_csv(REPORTS_DIR / 'model_comparison.csv')
old_rf = old_results[old_results['model_name'] == 'Random Forest'].iloc[0]
old_xgb = old_results[old_results['model_name'] == 'XGBoost'].iloc[0]

print(f"\n{'Model':<35} {'Accuracy':>10} {'F1 Macro':>10}")
print("-" * 58)
print(f"{'LR Baseline':<35} {results[2]['accuracy']:>10.4f} {results[2]['f1_macro']:>10.4f}")
print(f"{'RF (before tuning)':<35} {old_rf['accuracy']:>10.4f} {old_rf['f1_macro']:>10.4f}")
print(f"{'RF (after tuning)':<35} {results[0]['accuracy']:>10.4f} {results[0]['f1_macro']:>10.4f}")
print(f"{'XGB (before tuning)':<35} {old_xgb['accuracy']:>10.4f} {old_xgb['f1_macro']:>10.4f}")
print(f"{'XGB (after tuning)':<35} {results[1]['accuracy']:>10.4f} {results[1]['f1_macro']:>10.4f}")

# Pick the best overall model
best = max(results, key=lambda x: x['f1_macro'])
print(f"\nBest overall model: {best['model']} (F1 = {best['f1_macro']:.4f})")


# ══════════════════════════════════════════════════════════════
# SAVE BEST MODEL + FEATURE IMPORTANCE PLOT
# ══════════════════════════════════════════════════════════════

# Determine which tuned model is best
if results[0]['f1_macro'] >= results[1]['f1_macro']:
    best_model = rf_best
    best_name = "Random Forest (tuned)"
else:
    best_model = xgb_best
    best_name = "XGBoost (tuned)"

# Save as the final model
joblib.dump(best_model, MODELS_DIR / 'best_model.joblib')
print(f"\nSaved best model ({best_name}) to models/best_model.joblib")

# Feature importance plot
feature_names = list(X_train.columns)
importances = best_model.feature_importances_
sorted_idx = np.argsort(importances)[-15:]  # Top 15

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(sorted_idx)), importances[sorted_idx], color='#3498db', edgecolor='white')
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx])
ax.set_title(f'Top 15 Feature Importances ({best_name})', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(REPORTS_DIR / '15_feature_importance.png', dpi=150)
plt.close()
print(f"Feature importance plot saved to reports/15_feature_importance.png")

# Save tuning results
pd.DataFrame(results).to_csv(REPORTS_DIR / 'tuned_model_comparison.csv', index=False)

print("\n" + "=" * 60)
print("STEPS 11-12 COMPLETE")
print("=" * 60)
