"""
===============================================================
STEPS 7 & 8: Target Variable & Preprocessing
===============================================================
International Football Match Outcome Predictor

Step 7: Target variable is already created ('result' column).
        This script formalizes it with label encoding.

Step 8: Preprocessing pipeline:
  1. Select features (drop raw columns the model shouldn't see)
  2. Chronological train/test split
  3. Feature scaling (StandardScaler)
  4. Save processed X_train, X_test, y_train, y_test

WHY chronological split instead of random?
  - Sports data is time-dependent
  - Random split leaks future information into training
  - We train on past, test on future = realistic simulation
===============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / 'data' / 'processed' / 'featured_matches.csv'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
# STEP 7: Target Variable
# ══════════════════════════════════════════════════════════════

def encode_target(y: pd.Series) -> tuple:
    """Encode string labels to integers using LabelEncoder.

    ML algorithms need numbers, not strings.
    LabelEncoder converts:
        'Away Win' -> 0
        'Draw'     -> 1
        'Home Win' -> 2

    We also save the encoder so we can convert predictions
    BACK to human-readable labels later.

    Why LabelEncoder and not One-Hot for the target?
      - One-Hot is for INPUT features (creates separate columns)
      - For the TARGET, algorithms expect a single column of integers
      - Multiclass classifiers internally handle the rest
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("--- Step 7: Target Encoding ---")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"  Distribution:")
    for cls in le.classes_:
        count = (y == cls).sum()
        print(f"    {cls}: {count:,} ({count/len(y)*100:.1f}%)")
    return y_encoded, le


# ══════════════════════════════════════════════════════════════
# STEP 8: Preprocessing
# ══════════════════════════════════════════════════════════════

def select_features(df: pd.DataFrame) -> tuple:
    """Separate features (X) from target (y).

    Columns we DROP from X:
      - date:       Not a feature (used for splitting only)
      - home_team:  Already captured by rolling stats
      - away_team:  Already captured by rolling stats
      - home_score: DATA LEAKAGE (contains the answer)
      - away_score: DATA LEAKAGE (contains the answer)
      - tournament: Replaced by tournament_importance
      - result:     This IS the target variable (y)

    What remains as features (X):
      - 26 rolling stats (win rate, goals, goal diff for both teams)
      - 2 days_since_last (home + away)
      - tournament_importance
      - neutral
      - h2h_home_win_rate
      - h2h_total_matches
      = 30 features total
    """
    cols_to_drop = ['date', 'home_team', 'away_team',
                    'home_score', 'away_score', 'tournament', 'result']

    X = df.drop(columns=cols_to_drop)
    y = df['result']

    print("\n--- Step 8a: Feature Selection ---")
    print(f"  Dropped from X: {cols_to_drop}")
    print(f"  X shape: {X.shape} ({X.shape[1]} features)")
    print(f"  y shape: {y.shape}")
    print(f"  Features: {list(X.columns)}")
    return X, y, df['date']


def chronological_split(X: pd.DataFrame, y: np.ndarray,
                        dates: pd.Series, test_ratio: float = 0.2) -> tuple:
    """Split data chronologically: train on past, test on future.

    WHY NOT random split?
      Random: Model trains on 2024 data, tested on 2015 data.
              It already "knows" how teams evolved. Unfair.

      Chronological: Model trains on pre-2020 data, tested on 2020+.
                     It must predict the future from the past. Fair.

    We use the last 20% of matches (by date) as the test set.
    This is approximately matches from 2018-2019 onwards.
    """
    split_idx = int(len(X) * (1 - test_ratio))
    split_date = dates.iloc[split_idx]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n--- Step 8b: Chronological Split ---")
    print(f"  Split date: {split_date}")
    print(f"  Train: {len(X_train):,} matches (before {split_date})")
    print(f"  Test:  {len(X_test):,} matches (from {split_date} onwards)")
    print(f"  Train ratio: {len(X_train)/len(X)*100:.1f}%")
    print(f"  Test ratio:  {len(X_test)/len(X)*100:.1f}%")

    # Verify no data leakage: all train dates < all test dates
    train_max = dates.iloc[:split_idx].max()
    test_min = dates.iloc[split_idx:].min()
    assert train_max <= test_min, "DATA LEAKAGE: train dates overlap test dates!"
    print(f"  Leakage check: PASSED (train max={train_max}, test min={test_min})")

    # Check class distribution in both splits
    print(f"\n  Class distribution (train):")
    for val in sorted(set(y_train)):
        count = (y_train == val).sum()
        print(f"    Class {val}: {count:,} ({count/len(y_train)*100:.1f}%)")
    print(f"  Class distribution (test):")
    for val in sorted(set(y_test)):
        count = (y_test == val).sum()
        print(f"    Class {val}: {count:,} ({count/len(y_test)*100:.1f}%)")

    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Standardize features to have mean=0 and std=1.

    WHY scale?
      - 'days_since_last' ranges from 0 to 5000+
      - 'win_rate_last5' ranges from 0 to 1
      - Without scaling, the model thinks 'days_since_last'
        is more important just because it has bigger numbers

    StandardScaler transforms each feature:
      scaled_value = (value - mean) / std_deviation

    CRITICAL: We fit the scaler on TRAINING data only, then
    apply it to BOTH train and test. Why?
      - If we fit on test data too, test statistics leak into
        the transformation = subtle data leakage
      - In production, you only have training data when building
        the model

    Common mistake: Fitting scaler on ALL data before splitting.
    This leaks test set statistics into training.
    """
    scaler = StandardScaler()

    # Fit on train ONLY, transform both
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"\n--- Step 8c: Feature Scaling ---")
    print(f"  Scaler: StandardScaler (mean=0, std=1)")
    print(f"  Fitted on: training data only ({len(X_train):,} rows)")
    print(f"  Applied to: train + test")
    print(f"  Sample (first 3 features, train mean after scaling):")
    for col in X_train_scaled.columns[:3]:
        print(f"    {col}: mean={X_train_scaled[col].mean():.6f}, "
              f"std={X_train_scaled[col].std():.6f}")

    return X_train_scaled, X_test_scaled, scaler


# ── Main Pipeline ────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STEPS 7 & 8: Target Variable & Preprocessing")
    print("=" * 60)

    # Load featured data
    df = pd.read_csv(FEATURES_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Loaded {len(df):,} featured matches\n")

    # Step 7: Encode target
    y_raw = df['result']
    y_encoded, label_encoder = encode_target(y_raw)

    # Step 8a: Select features
    X, _, dates = select_features(df)

    # Step 8b: Chronological split
    X_train, X_test, y_train, y_test = chronological_split(
        X, y_encoded, dates, test_ratio=0.2
    )

    # Step 8c: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save everything for model training
    print(f"\n--- Saving Preprocessed Data ---")

    # Save as CSV for inspection
    X_train_scaled.to_csv(PROCESSED_DIR / 'X_train.csv', index=False)
    X_test_scaled.to_csv(PROCESSED_DIR / 'X_test.csv', index=False)
    pd.Series(y_train).to_csv(PROCESSED_DIR / 'y_train.csv', index=False)
    pd.Series(y_test).to_csv(PROCESSED_DIR / 'y_test.csv', index=False)

    # Save scaler and label encoder for later use (Streamlit app)
    joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')
    joblib.dump(label_encoder, MODELS_DIR / 'label_encoder.joblib')

    # Save feature names for reference
    feature_names = list(X_train.columns)
    joblib.dump(feature_names, MODELS_DIR / 'feature_names.joblib')

    print(f"  X_train: {X_train_scaled.shape}")
    print(f"  X_test:  {X_test_scaled.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")
    print(f"  Scaler saved:  models/scaler.joblib")
    print(f"  Encoder saved: models/label_encoder.joblib")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE -- Ready for model training!")
    print("=" * 60)


if __name__ == '__main__':
    main()
