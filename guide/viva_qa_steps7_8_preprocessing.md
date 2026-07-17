# Viva Q&A -- Steps 7 & 8: Target Variable & Preprocessing

## What Was Done

| Step | Action | Result |
|------|--------|--------|
| 7 | Target encoding | Away Win=0, Draw=1, Home Win=2 |
| 8a | Feature selection | 30 features (dropped date, teams, scores, tournament name) |
| 8b | Chronological split | Train: 39,602 (pre-2016) / Test: 9,901 (2016+) |
| 8c | Feature scaling | StandardScaler (mean=0, std=1), fit on train only |
| Save | Artifacts | X_train.csv, X_test.csv, y_train.csv, y_test.csv, scaler.joblib, label_encoder.joblib |

---

## Viva Questions

### Q1: Why use LabelEncoder for the target instead of One-Hot Encoding?

**Answer:** One-Hot Encoding creates separate columns (e.g., is_home_win, is_draw, is_away_win). This is for INPUT features. For the TARGET variable, classifiers expect a single column of integers (0, 1, 2). The algorithm handles multiclass logic internally.

**Interview-Ready:** *"I used LabelEncoder for the target because multiclass classifiers expect integer labels in a single column. One-Hot Encoding is for input features where each category becomes a separate binary column."*

---

### Q2: Why chronological split instead of random train_test_split?

**Answer:** Sports data is time-dependent. Random split mixes future and past data:
- Training on a 2024 match, then "predicting" a 2018 match = cheating
- The model already knows how teams evolved

Chronological split: train on everything before 2016, test on 2016+. This simulates real deployment where you only know the past.

Our split: Train = 39,602 matches (pre-2016), Test = 9,901 matches (2016+).

**Interview-Ready:** *"I used chronological splitting because sports outcomes are time-dependent. Training on future data to predict past results would give unrealistically high accuracy. My model trains on pre-2016 data and is evaluated on 2016+ matches."*

---

### Q3: What does StandardScaler do mathematically?

**Answer:** For each feature, it computes:
```
scaled_value = (original_value - mean) / standard_deviation
```

After scaling: every feature has mean=0 and std=1.

**Why?** Without scaling:
- `days_since_last` ranges 0 to 5000
- `win_rate_last5` ranges 0 to 1

The model might think days_since_last is 5000x more important just because its numbers are bigger. Scaling puts all features on equal footing.

**Interview-Ready:** *"StandardScaler normalizes features to mean=0, std=1. This prevents features with larger numeric ranges from dominating the model's learning process."*

---

### Q4: Why fit the scaler on training data ONLY?

**Answer:** If we fit on ALL data (train + test), the scaler's mean and std would include test data statistics. When we then transform training data, test information has leaked in.

Correct order:
1. `scaler.fit(X_train)` -- learn mean/std from train only
2. `scaler.transform(X_train)` -- apply to train
3. `scaler.transform(X_test)` -- apply to test using train's mean/std

This simulates production: when you deploy the model, you only have the training data's statistics saved.

**Interview-Ready:** *"I fit the scaler exclusively on training data to prevent information leakage. In production, test data statistics are unknown, so the scaler must use only training-derived parameters."*

---

### Q5: What 30 features does the model actually see?

**Answer:** All derived features, grouped:

| Group | Count | Examples |
|-------|-------|---------|
| Home team rolling stats | 12 | win_rate, goals scored/conceded, goal diff (x2 windows) |
| Away team rolling stats | 12 | same as above, for away team |
| Home days since last | 1 | fatigue/rustiness indicator |
| Away days since last | 1 | same for away |
| Tournament importance | 1 | 0.3 (friendly) to 1.0 (World Cup) |
| Neutral venue | 1 | 0 or 1 |
| Head-to-head win rate | 1 | historical rivalry |
| Head-to-head matches | 1 | how many times they've met |

The model NEVER sees: team names, raw scores, dates, or tournament names.

---

### Q6: This is where scores finally get dropped, right?

**Answer:** Yes! Here's the complete journey of `home_score` and `away_score`:

| Step | What happens to scores |
|------|----------------------|
| Step 2 | Loaded from CSV |
| Step 5 | Used to create target variable (Home Win/Draw/Away Win) |
| Step 6 | Used to compute rolling features (avg_goals_scored_last5, etc.) |
| **Step 8** | **DROPPED from feature matrix X** -- model never sees raw scores |

The derived features like `home_avg_goals_scored_last5 = 1.8` go in. The raw `home_score = 3` does not.

---

### Q7: How did you verify no data leakage occurred?

**Answer:** Three checks:
1. **Time check:** Asserted that max training date <= min test date
2. **Feature check:** Confirmed home_score/away_score are NOT in X columns
3. **Class distribution check:** Verified train and test have similar class proportions (~49% Home Win, ~28% Away Win, ~23% Draw)

If class distributions were wildly different between train and test, it could indicate the data changed dramatically over time (which is worth knowing but not necessarily leakage).

---

### Q8: What files were saved and why?

| File | Purpose |
|------|---------|
| `X_train.csv` | Training features (30 columns, scaled) |
| `X_test.csv` | Test features (30 columns, scaled) |
| `y_train.csv` | Training labels (0, 1, 2) |
| `y_test.csv` | Test labels (0, 1, 2) |
| `scaler.joblib` | Saved StandardScaler for Streamlit app |
| `label_encoder.joblib` | Saved LabelEncoder to decode predictions back to text |
| `feature_names.joblib` | List of 30 feature names for reference |

**Interview-Ready:** *"I saved the preprocessing artifacts (scaler, encoder) alongside the data so the Streamlit app can apply the same transformations to new user inputs at prediction time."*

---

## Quick-Fire Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | What's the train/test split date? | March 29, 2016 |
| 2 | How many training samples? | 39,602 |
| 3 | How many test samples? | 9,901 |
| 4 | What does LabelEncoder produce? | Away Win=0, Draw=1, Home Win=2 |
| 5 | Why not fit scaler on all data? | Leaks test statistics into training |
| 6 | What's the baseline accuracy? | ~49% (always predict Home Win) |
| 7 | How many features does the model see? | 30 |
| 8 | Where are scores dropped? | Step 8a (select_features function) |

---

## Project Progress

| Step | Topic | Status |
|------|-------|--------|
| 1 | Project Setup | ✅ |
| 2 | Load Dataset | ✅ |
| 3 | Data Understanding | ✅ |
| 4 | EDA (14 plots) | ✅ |
| 5 | Data Cleaning | ✅ |
| 6 | Feature Engineering (30 features) | ✅ |
| 7 | Target Variable | ✅ |
| 8 | Preprocessing | ✅ |
| **9** | **Baseline Model** | **Next** |
| 10 | Train Multiple Models | -- |
| 11 | Model Comparison | -- |
| 12 | Hyperparameter Tuning | -- |
| 13 | SHAP Explainability | -- |
| 14 | Model Saving | -- |
| 15 | Streamlit App | -- |
