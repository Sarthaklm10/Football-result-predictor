# 🎓 Viva Q&A — Steps 1, 2, 3

> [!TIP]
> Read each question, try to answer it mentally, then read the answer. The "Interview-Ready" answers are short, confident responses you can use word-for-word in a viva.

---

## STEP 1: Project Setup

### Q1: Why did you separate `data/raw/` and `data/processed/`?

**Answer:**
`data/raw/` holds the original, untouched CSV files exactly as downloaded from Kaggle. `data/processed/` holds the cleaned, feature-engineered data ready for modeling.

**Why it matters:** If my cleaning or feature engineering code has a bug and I accidentally overwrite the original data, I've lost it forever. By keeping raw data untouched, I can always reprocess from scratch.

**Interview-Ready:** *"I follow the principle of data immutability — raw data is never modified. All transformations produce new files in `data/processed/`, so I can always reproduce my pipeline from the original source."*

---

### Q2: What is `__init__.py` and why is it in every `src/` subfolder?

**Answer:**
`__init__.py` is a special Python file that tells Python: "This folder is a package — you can import from it." Without it, Python doesn't recognize the folder as importable.

With `__init__.py` in `src/features/`:
```python
from src.features import build_features  # ✅ Works
```

Without it:
```python
from src.features import build_features  # ❌ ModuleNotFoundError
```

**Interview-Ready:** *"I added `__init__.py` to make each subdirectory an importable Python package, enabling clean modular imports across the project."*

---

### Q3: Why use a virtual environment instead of installing libraries globally?

**Answer:**
A virtual environment creates an **isolated** Python environment for this project. Without it:
- Project A needs `pandas==1.5`, Project B needs `pandas==2.0` → they conflict
- Installing globally can break system tools that depend on specific versions
- Someone cloning your project might get different library versions → different behavior

With a virtual environment, each project has its own independent set of packages.

**Interview-Ready:** *"I used a virtual environment to isolate project dependencies and ensure reproducibility. Anyone can recreate my exact setup using `requirements.txt`."*

---

### Q4: Why is each library in your `requirements.txt`?

**Answer:**

| Library | One-Line Purpose |
|---------|-----------------|
| **pandas** | Loading, cleaning, and manipulating tabular data (DataFrames) |
| **numpy** | Fast numerical operations on arrays (vectorized math) |
| **matplotlib** | Creating static charts and visualizations |
| **plotly** | Creating interactive charts (hover, zoom, click) |
| **seaborn** | Statistical visualizations built on top of matplotlib |
| **scikit-learn** | ML algorithms, preprocessing, evaluation metrics, train/test split |
| **xgboost** | Gradient boosting — often the best algorithm for tabular data |
| **shap** | Explaining *why* a model made a specific prediction |
| **joblib** | Saving and loading trained models to/from disk |
| **streamlit** | Building web apps with pure Python (no HTML/JS needed) |
| **jupyter + notebook** | Interactive coding environment with inline outputs |
| **ipykernel** | Connecting the virtual environment to Jupyter as a kernel |

**Interview-Ready:** *"Every dependency in my requirements file serves a specific purpose in the ML pipeline — from data manipulation and visualization through model training, explainability, and deployment."*

---

### Q5: What is `.gitignore` and why did you exclude certain files?

**Answer:**
`.gitignore` tells Git which files to **not track**. We exclude:

| Excluded | Why |
|----------|-----|
| `venv/` | Virtual environment is machine-specific — 200MB+ of binaries |
| `__pycache__/` | Python's compiled bytecode cache — auto-generated |
| `models/*.joblib` | Trained models are large binary files (50MB+) |
| `data/processed/*.csv` | Generated files — can be recreated by running the pipeline |
| `.ipynb_checkpoints/` | Jupyter's autosave folder — clutter |

**Important:** We **do NOT** exclude `data/raw/*.csv` because our CSVs are small (~7MB) and anyone cloning the repo needs them to run anything.

**Interview-Ready:** *"I configured `.gitignore` to exclude environment-specific files, auto-generated caches, and large binary artifacts while keeping the raw data tracked so the project is immediately runnable after cloning."*

---

## STEP 2: Loading the Dataset

### Q6: What is `pd.read_csv()` and what does it return?

**Answer:**
`pd.read_csv()` reads a CSV (Comma-Separated Values) file and converts it into a **Pandas DataFrame** — a 2-dimensional labeled table with rows and columns.

What happens internally:
1. Opens the file
2. Reads the first line as **column headers**
3. Reads all subsequent lines as **data rows**
4. **Infers data types** for each column (string, number, boolean)
5. Returns a DataFrame object

```python
df = pd.read_csv('data/raw/results.csv')
print(df.shape)  # (49509, 9) → 49,509 rows, 9 columns
```

**Interview-Ready:** *"`pd.read_csv()` parses a CSV file and returns a DataFrame — a 2D labeled data structure with automatic type inference. Our dataset has 49,509 rows and 9 columns."*

---

### Q7: What is a DataFrame?

**Answer:**
A DataFrame is Pandas' primary data structure — think of it as:
- An **Excel spreadsheet** with named columns
- A **SQL table** you can query with Python
- A **dictionary of Series**, where each column is a Series (1D array with labels)

Key properties:
- `df.shape` → `(rows, columns)` → `(49509, 9)`
- `df.columns` → list of column names
- `df.dtypes` → data type of each column
- `df.memory_usage(deep=True).sum()` → how much RAM it uses

**Interview-Ready:** *"A DataFrame is a 2D labeled data structure in Pandas — essentially a table with rows indexed by integers and columns indexed by names. Each column is a Pandas Series."*

---

## STEP 3: Data Understanding

### Q8: What are the 9 columns in your dataset and what do they mean?

**Answer:**

| Column | Type | Meaning | Example |
|--------|------|---------|---------|
| `date` | string (should be datetime) | Match date | `2022-11-22` |
| `home_team` | categorical | Team playing at home | `Brazil` |
| `away_team` | categorical | Visiting team | `Argentina` |
| `home_score` | numerical (float) | Goals by home team | `3.0` |
| `away_score` | numerical (float) | Goals by away team | `1.0` |
| `tournament` | categorical | Competition type | `FIFA World Cup` |
| `city` | categorical | Match city | `Doha` |
| `country` | categorical | Match country | `Qatar` |
| `neutral` | boolean | Neutral venue? | `True` |

**Interview-Ready:** *"The dataset has 9 columns: match date, both team names, both scores, tournament type, match location (city + country), and a neutral venue flag. It covers 49,509 matches from 1872 to 2024."*

---

### Q9: ⭐ What is data leakage and how does it affect your project? (VERY COMMON VIVA QUESTION)

**Answer:**
**Data leakage** means your model accidentally gets access to information that **would not be available at prediction time**. It makes your model appear highly accurate during evaluation but completely useless in the real world.

**In our project:**
`home_score` and `away_score` literally **contain the answer**. If we use them as input features:

```
Input: home_score=3, away_score=1 → Model predicts: "Home Win" ← duh!
```

The model just learns "if home_score > away_score → Home Win." It gets ~100% accuracy — but it's useless because **you don't know the score before the match happens**.

**How we handle it:**
- Use `home_score` and `away_score` **only** to create the target variable and compute **historical statistics** (like "Brazil's average goals in the last 10 games")
- Never pass them directly as features
- Always compute features using **only data from past matches** (never future matches)

**Interview-Ready:** *"Data leakage occurs when the model sees information at training time that wouldn't be available at prediction time. In my project, `home_score` and `away_score` cannot be used as features because they contain the outcome itself. I use them only to derive the target variable and to compute backward-looking historical statistics."*

---

### Q10: ⭐ Why is this a multiclass classification problem? (VERY COMMON VIVA QUESTION)

**Answer:**

| Term | Meaning | Example |
|------|---------|---------|
| **Regression** | Predicting a continuous number | "How many goals will Brazil score?" → 2.3 |
| **Binary Classification** | Predicting one of 2 outcomes | "Will Brazil win? Yes / No" |
| **Multiclass Classification** | Predicting one of 3+ outcomes | "Home Win / Draw / Away Win" |

Our target variable has **3 possible values** → it's a **multiclass classification** problem.

This affects:
1. **Algorithm choice** — not all algorithms handle 3+ classes natively
2. **Evaluation metrics** — we need metrics that work across all 3 classes (macro F1, weighted F1)
3. **Output format** — the model outputs **probabilities for each class** (e.g., Home Win: 55%, Draw: 25%, Away Win: 20%)

**Interview-Ready:** *"This is a multiclass classification problem because the target variable has three discrete categories: Home Win, Draw, and Away Win. I use algorithms that support multiclass outputs and evaluate using macro-averaged F1 score to ensure balanced performance across all three classes."*

---

### Q11: ⭐ Why is chronological splitting better than random splitting for sports data?

**Answer:**

**Random split (`train_test_split(test_size=0.2)`):**
- Randomly puts 80% of matches in training and 20% in testing
- A match from **2024** might be in training, while a match from **2018** is in testing
- The model "learns from the future to predict the past" — that never happens in real life

**Chronological split:**
- Train on matches **before** a cutoff date (e.g., before 2020)
- Test on matches **after** that date (e.g., 2020–2024)
- Simulates real-world usage: you only know the past, you predict the future

**Why random splitting gives overly optimistic results:**
If training data includes 2024 matches, the model learns current team strengths. When it then "predicts" a 2020 match, it already knows how strong those teams are in 2024. That's information leakage through time — **temporal leakage**.

**Interview-Ready:** *"I use chronological splitting instead of random splitting because sports data is time-dependent. Random splitting allows temporal leakage — the model could train on 2024 data and be tested on 2020 data, which is unrealistic. Chronological splitting simulates real-world deployment where we only have past data to predict future outcomes."*

---

### Q12: Why is `np.where()` faster than a `for` loop with `iterrows()`?

**Answer:**

**`for` loop with `iterrows()` (Method A):**
- Python processes **one row at a time**
- For each row, Python must: look up the row, check the condition, append to a list
- 49,509 rows = 49,509 separate Python operations
- Python is an **interpreted language** — each operation is slow

**`np.where()` (Method B):**
- NumPy sends the **entire column** to a C function at once
- The C function processes all 49,509 comparisons in one batch
- One Python call → one C function → done

**Analogy:** Imagine shipping 50,000 packages.
- `iterrows()` = driving one package at a time to the post office (50,000 trips)
- `np.where()` = loading all packages into a truck and making one trip

The speed difference is typically **10–100x** on datasets this size.

**Interview-Ready:** *"Vectorized operations like `np.where()` delegate computation to optimized C code that processes entire arrays at once, while `iterrows()` uses slow Python-level iteration. On our 49K-row dataset, vectorized operations are approximately 50x faster."*

---

## 📝 Quick-Fire Viva Questions (Rapid Response Practice)

These might come up. Practice answering each in **one sentence**:

| # | Question | Your Answer |
|---|----------|-------------|
| 1 | What dataset did you use? | International Football Results from 1872–2024, containing 49,509 matches from Kaggle. |
| 2 | How many features does the raw data have? | 9 columns: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral. |
| 3 | What are you predicting? | The outcome: Home Win, Draw, or Away Win — a multiclass classification problem. |
| 4 | Why didn't you use home_score as a feature? | It causes data leakage — it contains the answer we're trying to predict. |
| 5 | What's a DataFrame? | A 2D labeled data structure in Pandas — like an Excel spreadsheet with named columns. |
| 6 | Why use a virtual environment? | To isolate project dependencies and ensure reproducibility across machines. |
| 7 | What is .gitignore for? | To exclude machine-specific files, caches, and large binaries from version control. |
| 8 | Why chronological split over random? | Sports data is time-dependent — random splitting causes temporal leakage. |
| 9 | What Python data types are in your dataset? | String (dates, team names), float64 (scores), and boolean (neutral). |
| 10 | How much memory does the dataset use? | Approximately 6.17 MB — small enough to fit entirely in RAM. |
