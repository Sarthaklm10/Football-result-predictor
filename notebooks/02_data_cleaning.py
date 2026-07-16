"""
===============================================================
STEP 5: Data Cleaning
===============================================================
International Football Match Outcome Predictor

This script cleans the raw dataset and prepares it for
feature engineering. Cleaned data is saved to data/processed/.

What we do:
  1. Handle missing values (drop — only 5 rows affected)
  2. Fix data types (date to datetime, scores to int)
  3. Remove invalid/incomplete rows
  4. Create the target variable
  5. Drop columns not needed for modeling
  6. Save cleaned dataset
===============================================================
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
    """Load raw CSV and return a DataFrame.
    
    We use parse_dates=['date'] to tell Pandas to automatically
    convert the 'date' column from string to datetime during loading.
    This is cleaner than converting it afterwards.
    """
    df = pd.read_csv(filepath, parse_dates=['date'])
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Inspect and report missing values.
    
    Intuition: Missing values are like holes in your data.
    Before filling or dropping them, you need to understand
    WHY they're missing — is it random, or systematic?
    
    Three strategies for handling missing values:
      1. DROP the rows    — when very few rows are affected (<1%)
      2. FILL with a value — when you can make a reasonable guess
         (mean, median, mode, or a domain-specific default)
      3. IMPUTE using ML   — when missingness is complex (overkill here)
    
    Our case: Only 5 rows out of 49,509 have missing values.
    That's 0.01%. Dropping is the clear choice — we lose
    almost nothing.
    """
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
    """Drop rows with any missing values.
    
    We use dropna() which removes any row that has at least one NaN.
    
    Why drop instead of fill?
    - Only 5 rows are affected (0.01% of data)
    - We can't guess what the missing team name or score was
    - Filling scores with mean/median would create fake data
    - Losing 5 rows out of 49,509 has zero impact on our model
    """
    before = len(df)
    df = df.dropna()
    after = len(df)
    dropped = before - after
    print(f"\n--- Handling Missing Values ---")
    print(f"  Dropped {dropped} rows with missing values")
    print(f"  Remaining: {after:,} rows ({after/before*100:.2f}%)")
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to their correct data types.
    
    Why fix types?
    - home_score/away_score are float64 (because of NaN values).
      Now that NaNs are gone, we can safely convert to int.
      Goals are always whole numbers (you can't score 2.5 goals!)
    
    - date is already datetime (we used parse_dates in read_csv)
    
    - neutral is already boolean — correct
    
    Common mistake: Trying to convert to int BEFORE dropping NaNs.
    NaN can only exist in float columns, so int conversion would
    crash if NaNs remain.
    """
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    
    print("\n--- Fixed Data Types ---")
    print(f"  home_score: float64 -> int64")
    print(f"  away_score: float64 -> int64")
    print(f"  date: {df['date'].dtype}")
    print(f"  neutral: {df['neutral'].dtype}")
    return df


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Check for and report duplicate rows.
    
    Why check duplicates?
    - Duplicates inflate certain teams' statistics
    - They bias the model by overrepresenting certain matches
    - Real-world data often has accidental duplicates from
      multiple data sources being merged
    
    We check exact duplicates (all columns match) AND
    logical duplicates (same date + same teams).
    """
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
    """Create the target column: Home Win / Draw / Away Win.
    
    This is what we're trying to PREDICT. It's the 'y' in
    our ML equation: y = f(X).
    
    We use np.where() instead of a for loop because:
    - np.where processes the ENTIRE column at once in C
    - A for loop processes one row at a time in Python
    - On 49K rows, np.where is ~50x faster
    
    IMPORTANT: After creating the target, we DROP home_score
    and away_score as features. Using them directly would be
    DATA LEAKAGE — the model would just learn
    "if home_score > away_score -> Home Win" (100% accuracy
    in training, useless in real life).
    """
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
    """Remove columns that won't be used in modeling.
    
    Columns we DROP and why:
    
    - city: Too many unique values (2,089). High cardinality
      categorical features create sparse, noisy encodings.
      The information is partially captured by 'country' already.
    
    - country: Highly correlated with home_team (if Brazil plays
      at home, country is almost always Brazil). Keeping both
      adds redundancy without new information.
    
    Columns we KEEP:
    - date:       Needed for chronological splitting + feature engineering
    - home_team:  Core input
    - away_team:  Core input
    - home_score: Needed for feature engineering (historical stats)
    - away_score: Needed for feature engineering (historical stats)
    - tournament: Match importance indicator
    - neutral:    Home advantage modifier
    - result:     Target variable
    
    NOTE: home_score and away_score are kept at this stage
    because we need them for feature engineering (computing
    rolling averages, historical stats). They will NOT be
    passed as features to the model — that happens in Step 8.
    """
    cols_to_drop = ['city', 'country']
    df = df.drop(columns=cols_to_drop)
    
    print(f"\n--- Dropped Columns ---")
    print(f"  Removed: {cols_to_drop}")
    print(f"  Remaining columns: {list(df.columns)}")
    return df


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the entire dataset chronologically.
    
    Why? Feature engineering in Step 6 computes rolling
    statistics like 'last 5 games.' If data isn't sorted
    by date, 'last 5 games' would be meaningless.
    
    Also, chronological order is required for our train/test
    split — we train on the past and test on the future.
    """
    df = df.sort_values('date').reset_index(drop=True)
    print(f"\n--- Sorted by Date ---")
    print(f"  First match: {df['date'].iloc[0].date()}")
    print(f"  Last match:  {df['date'].iloc[-1].date()}")
    return df


def save_cleaned_data(df: pd.DataFrame, filepath: Path) -> None:
    """Save the cleaned DataFrame to CSV.
    
    index=False prevents Pandas from saving the row numbers
    as an extra column (a common beginner mistake).
    """
    df.to_csv(filepath, index=False)
    size_mb = filepath.stat().st_size / 1e6
    print(f"\n--- Saved ---")
    print(f"  Path: {filepath}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")


# ── Main Pipeline ────────────────────────────────────────────
def main():
    """Run the complete data cleaning pipeline.
    
    Each function does ONE thing (Single Responsibility Principle).
    This makes the code easy to read, test, and debug.
    """
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
