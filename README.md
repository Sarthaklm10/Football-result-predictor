# ⚽ International Football Match Outcome Predictor
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

## 📌 Objective

Predict the outcome of international football matches (**Home Win**, **Draw**, or **Away Win**) using historical match data and machine learning.

## 📊 Dataset

- **Source**: [International Football Results from 1872 to 2024](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- **Primary file**: `results.csv` (~47,000 matches)

## 🏗️ Project Structure

```
ResultPredictor/
│
├── data/
│   ├── raw/                  ← Original CSV (never modified)
│   └── processed/            ← Cleaned & feature-engineered data
│
├── notebooks/                ← Pipeline scripts (run in order)
│   ├── 01_eda.py             ← Exploratory Data Analysis
│   ├── 02_data_cleaning.py   ← Missing values, dtypes, target creation
│   ├── 03_feature_engineering.py ← Rolling stats, H2H, tournament weights
│   ├── 04_preprocessing.py   ← Encoding, splitting, scaling
│   ├── 05_model_training.py  ← LR, DT, RF, XGBoost training
│   ├── 06_tuning.py          ← Hyperparameter tuning (RandomizedSearchCV)
│   └── 07_shap_analysis.py   ← SHAP explainability
│
├── models/                   ← Saved models & artifacts (.joblib)
│
├── reports/                  ← EDA plots, feature importance, SHAP
│
├── app.py                    ← Streamlit web application
├── requirements.txt
├── README.md
└── .gitignore
```

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML | Scikit-Learn, XGBoost |
| Explainability | SHAP |
| Web App | Streamlit |
| Version Control | Git, GitHub |

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Sarthaklm10/Football-result-predictor.git
cd Football-result-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Run the Pipeline

Run each script in order from the project root:

```bash
python notebooks/01_eda.py
python notebooks/02_data_cleaning.py
python notebooks/03_feature_engineering.py
python notebooks/04_preprocessing.py
python notebooks/05_model_training.py
python notebooks/06_tuning.py
python notebooks/07_shap_analysis.py
```

## 🌐 Launch the App

```bash
streamlit run app.py
```

## 📈 Results

| Model | Accuracy | F1 (macro) |
|-------|----------|------------|
| Logistic Regression | 51.00% | 0.4491 |
| Random Forest | 50.50% | 0.4613 |
| **Random Forest (tuned)** | **50.16%** | **0.4679** |
| XGBoost | 52.98% | 0.3897 |
| XGBoost (tuned) | 49.30% | 0.4003 |

Best model: **Random Forest (tuned)** — selected on F1-macro to balance all 3 classes.