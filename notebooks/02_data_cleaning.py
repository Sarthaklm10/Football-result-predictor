"""
Step 5: Data Cleaning
Cleans the raw dataset and saves processed output to data/processed/.
"""

# ── Imports ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / 'data' / 'raw' / 'results.csv'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
PROCESSED_DIR.mkdir(exist_ok=True)
SAVE_PATH = PROCESSED_DIR / 'cleaned_matches.csv'


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load raw CSV with date parsing."""
    df = pd.read_csv(filepath, parse_dates=['date'])
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Log missing values per column."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(4)
    
    print("\n--- Missing Values ---")
    for col in df.columns:
        if null_counts[col] > 0:
            print(f"  {col}: {null_counts[col]} missing ({null_pct[col]}%)")
    
    total_nulls = null_counts.sum()
    if total_nulls == 0:
        print("  No missing values!")
    else:
        print(f"  Total: {total_nulls} missing values across {len(df):,} rows")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN values. Safe here since only ~5 rows are affected."""
    before = len(df)
    df = df.dropna()
    after = len(df)
    dropped = before - after
    print(f"\n--- Handling Missing Values ---")
    print(f"  Dropped {dropped} rows with missing values")
    print(f"  Remaining: {after:,} rows ({after/before*100:.2f}%)")
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert score columns from float to int (safe after dropna)."""
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    
    print("\n--- Fixed Data Types ---")
    print(f"  home_score: float64 -> int64")
    print(f"  away_score: float64 -> int64")
    print(f"  date: {df['date'].dtype}")
    print(f"  neutral: {df['neutral'].dtype}")
    return df


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Check for exact and logical duplicates (same date + same teams)."""
    # Exact duplicates
    exact_dupes = df.duplicated().sum()
    print(f"\n--- Duplicate Check ---")
    print(f"  Exact duplicates: {exact_dupes}")
    
    # Logical duplicates: same date + same matchup
    logical_dupes = df.duplicated(
        subset=['date', 'home_team', 'away_team'], keep=False
    )
    logical_count = logical_dupes.sum()
    print(f"  Logical duplicates (same date + teams): {logical_count}")
    
    if logical_count > 0:
        print("  Showing first few:")
        print(df[logical_dupes].sort_values('date').head(6).to_string())
        # Keep first occurrence, drop rest
        before = len(df)
        df = df.drop_duplicates(
            subset=['date', 'home_team', 'away_team'], keep='first'
        )
        print(f"  Dropped {before - len(df)} logical duplicates")
    
    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Derive target column (Home Win / Draw / Away Win) from scores."""
    df['result'] = np.where(
        df['home_score'] > df['away_score'], 'Home Win',
        np.where(df['away_score'] > df['home_score'], 'Away Win', 'Draw')
    )
    
    print("\n--- Target Variable Created ---")
    counts = df['result'].value_counts()
    for label in ['Home Win', 'Draw', 'Away Win']:
        count = counts.get(label, 0)
        pct = count / len(df) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop city and country — high cardinality and redundant with other columns."""
    cols_to_drop = ['city', 'country']
    df = df.drop(columns=cols_to_drop)
    
    print(f"\n--- Dropped Columns ---")
    print(f"  Removed: {cols_to_drop}")
    print(f"  Remaining columns: {list(df.columns)}")
    return df


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort matches by date for chronological feature engineering."""
    df = df.sort_values('date').reset_index(drop=True)
    print(f"\n--- Sorted by Date ---")
    print(f"  First match: {df['date'].iloc[0].date()}")
    print(f"  Last match:  {df['date'].iloc[-1].date()}")
    return df


def save_cleaned_data(df: pd.DataFrame, filepath: Path) -> None:
    """Save cleaned DataFrame to CSV."""
    df.to_csv(filepath, index=False)
    size_mb = filepath.stat().st_size / 1e6
    print(f"\n--- Saved ---")
    print(f"  Path: {filepath}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")


# ── Main Pipeline ────────────────────────────────────────────
def main():
    """Run the full data cleaning pipeline."""
    print("=" * 60)
    print("STEP 5: Data Cleaning Pipeline")
    print("=" * 60)
    
    # 1. Load
    df = load_raw_data(RAW_PATH)
    
    # 2. Inspect missing values
    df = check_missing_values(df)
    
    # 3. Drop missing values
    df = handle_missing_values(df)
    
    # 4. Fix data types (must come AFTER dropping NaNs)
    df = fix_data_types(df)
    
    # 5. Check and handle duplicates
    df = check_duplicates(df)
    
    # 6. Create target variable
    df = create_target_variable(df)
    
    # 7. Drop unnecessary columns
    df = drop_unnecessary_columns(df)
    
    # 8. Sort chronologically
    df = sort_by_date(df)
    
    # 9. Save
    save_cleaned_data(df, SAVE_PATH)
    
    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
