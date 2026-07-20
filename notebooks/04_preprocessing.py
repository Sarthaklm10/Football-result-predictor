"""
Steps 7 & 8: Target Encoding & Preprocessing
Label encodes the target, selects features, does chronological
train/test split, and scales features with StandardScaler.
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
    """Encode string labels (Away Win, Draw, Home Win) to integers."""
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
    """Separate features (X) from target (y), dropping raw/leaky columns."""
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
    """Split data by date: train on past, test on future (last 20%)."""
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
    """Standardize features (mean=0, std=1). Fit on train only to avoid leakage."""
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
