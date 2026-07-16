# Viva Q&A — Step 5: Data Cleaning

## What The Cleaning Pipeline Did

| Step | Action | Before | After |
|------|--------|--------|-------|
| 1 | Load raw data | — | 49,509 rows |
| 2 | Drop missing values | 8 nulls | 49,506 rows (lost 3) |
| 3 | Fix dtypes | scores=float64 | scores=int64 |
| 4 | Remove logical duplicates | 4 found | 49,504 rows (lost 2) |
| 5 | Create target variable | — | Home Win/Draw/Away Win |
| 6 | Drop city, country | 9 columns | 7 columns + target |
| 7 | Sort by date | unsorted | chronological order |
| 8 | Save to processed/ | — | cleaned_matches.csv (3.28 MB) |

**Total data lost: 5 rows out of 49,509 (0.01%) — negligible.**

---

## Viva Questions

### Q1: Why is data cleaning necessary?

**Answer:** Raw data is messy — missing values, wrong types, duplicates. If you feed dirty data to an ML model, it learns wrong patterns ("garbage in, garbage out"). Cleaning ensures the data accurately represents reality before the model sees it.

**Interview-Ready:** *"Data cleaning ensures data quality. I dropped 5 rows with missing values (0.01%), fixed score columns from float to int, removed 2 logical duplicates, and dropped high-cardinality columns — losing less than 0.01% of data."*

---

### Q2: What are the three strategies for handling missing values? When do you use each?

**Answer:**

| Strategy | When to Use | Example |
|----------|------------|---------|
| **Drop rows** | Very few rows affected (<1%) | Our case: 5 out of 49,509 |
| **Fill with value** | Moderate missingness, reasonable guess exists | Fill missing salary with median salary |
| **ML imputation** | Complex patterns, large % missing | Using KNNImputer or IterativeImputer |

We used **drop** because only 0.01% of rows were affected. You can't guess a missing team name or score, so filling doesn't make sense here.

**Interview-Ready:** *"With only 5 missing rows out of 49,509, I chose to drop rather than impute. The missing values were in critical columns (team names, scores) where imputation would create fabricated data."*

---

### Q3: Why convert scores from float64 to int AFTER dropping NaNs?

**Answer:** In Pandas, integer columns **cannot contain NaN values** (in standard dtypes). When a column has even one NaN, Pandas automatically stores it as float64. So the conversion order must be:

1. First: `dropna()` — remove NaNs
2. Then: `astype(int)` — safe because no NaNs remain

If you reverse the order, `astype(int)` crashes with `IntCastingNaNError`.

**Interview-Ready:** *"I converted scores from float to int after dropping NaN values because Pandas standard integer dtypes cannot represent NaN. The float type was an artifact of the missing values, not the data itself."*

---

### Q4: What's the difference between exact duplicates and logical duplicates?

**Answer:**
- **Exact duplicate:** Every single column matches between two rows (copy-paste error)
- **Logical duplicate:** Key columns match but minor details differ (same match recorded from two sources)

We found 4 logical duplicates — same date + same teams but different cities (e.g., Gibraltar vs Europa Point are both in Gibraltar). We kept the first occurrence and dropped the rest.

**Interview-Ready:** *"I checked both exact and logical duplicates. I found 4 logical duplicates where the same match appeared with slightly different city names, likely from different data sources. I kept the first occurrence."*

---

### Q5: Why did you drop `city` and `country` columns?

**Answer:**

| Column | Problem | Reason to Drop |
|--------|---------|---------------|
| `city` | 2,089 unique values | **High cardinality** — one-hot encoding would create 2,089 columns of mostly zeros. Too sparse and noisy. |
| `country` | Highly correlated with `home_team` | **Redundancy** — if Brazil plays at home, country is almost always Brazil. Adds no new information. |

**Interview-Ready:** *"I dropped city (2,089 unique values creating sparse encodings) and country (redundant with home_team). The useful location information is captured by the neutral venue flag instead."*

---

### Q6: Why must the data be sorted chronologically?

**Answer:** Two critical reasons:
1. **Feature engineering** (Step 6): We'll compute "last 5 game win rate." If data isn't sorted by date, "last 5" would grab random matches, not recent ones.
2. **Train/test split** (Step 8): We train on past matches and test on future matches. Without chronological order, we can't do this correctly.

**Interview-Ready:** *"I sorted data chronologically because our feature engineering computes backward-looking rolling statistics, and our train/test split uses a time-based cutoff. Both require date-ordered data."*

---

### Q7: Why keep home_score and away_score after cleaning if they cause data leakage?

**Answer:** We keep them at this stage because they're needed for **feature engineering** in Step 6 — computing historical statistics like "Brazil's average goals in the last 10 games." They will be **dropped as direct features** before model training in Step 8. The key distinction:
- **Direct feature**: `home_score=3` → LEAKAGE (model sees the answer)
- **Historical stat**: `avg_home_goals_last_10=1.8` → SAFE (computed from past matches only)

**Interview-Ready:** *"I retained scores for computing backward-looking historical features but explicitly excluded them as direct model inputs. Only aggregated historical statistics derived from past matches are used as features."*

---

### Q8: What does "Single Responsibility Principle" mean in your code?

**Answer:** Each function does ONE thing: `handle_missing_values()` only handles nulls, `fix_data_types()` only fixes types, etc. This makes the code:
- **Easy to read:** function name tells you exactly what it does
- **Easy to debug:** if types are wrong, you know exactly which function to check
- **Easy to test:** you can test each function independently
- **Easy to modify:** changing duplicate logic doesn't risk breaking null handling

**Interview-Ready:** *"I followed the Single Responsibility Principle — each function handles one cleaning task. This makes the pipeline modular, testable, and easy to modify without side effects."*

---

## Quick-Fire Viva Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | How many rows did you lose in cleaning? | 5 out of 49,509 (0.01%) |
| 2 | Why are scores stored as float initially? | Because NaN can only exist in float columns in Pandas |
| 3 | What does `dropna()` do? | Removes any row containing at least one NaN value |
| 4 | What does `parse_dates=['date']` do? | Tells read_csv to auto-convert the date column to datetime |
| 5 | Why `index=False` in `to_csv()`? | Prevents saving row numbers as an extra unnamed column |
| 6 | What columns remain after cleaning? | date, home_team, away_team, home_score, away_score, tournament, neutral, result |
| 7 | Why not fill missing scores with 0? | A 0 score is a real result (0-0 draw). Filling with 0 creates fake matches. |
| 8 | What does `reset_index(drop=True)` do? | Resets row numbers to 0,1,2... without adding old index as a column |
