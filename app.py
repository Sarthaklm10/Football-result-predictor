"""
Streamlit App — International Football Match Outcome Predictor
Loads the trained model and historical data, computes features
for any matchup, and predicts Home Win / Draw / Away Win.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'


@st.cache_resource
def load_artifacts():
    """Load model, scaler, encoder, and feature names once."""
    model = joblib.load(MODELS_DIR / 'best_model.joblib')
    scaler = joblib.load(MODELS_DIR / 'scaler.joblib')
    label_encoder = joblib.load(MODELS_DIR / 'label_encoder.joblib')
    feature_names = joblib.load(MODELS_DIR / 'feature_names.joblib')
    return model, scaler, label_encoder, feature_names


@st.cache_data
def load_historical_data():
    """Load featured matches to compute rolling stats from."""
    df = pd.read_csv(DATA_DIR / 'featured_matches.csv', parse_dates=['date'])
    return df


def get_team_stats(df, team, role='home'):
    """Get the latest rolling stats for a given team."""
    prefix = f'{role}_'

    # Get all matches where this team played (as home or away)
    home_matches = df[df['home_team'] == team].copy()
    away_matches = df[df['away_team'] == team].copy()

    # Use the most recent match for this team's stats
    all_matches = []

    if not home_matches.empty:
        latest_home = home_matches.iloc[-1]
        all_matches.append(('home', latest_home, latest_home['date']))

    if not away_matches.empty:
        latest_away = away_matches.iloc[-1]
        all_matches.append(('away', latest_away, latest_away['date']))

    if not all_matches:
        return None

    # Pick the most recent match overall
    all_matches.sort(key=lambda x: x[2], reverse=True)
    match_role, latest, _ = all_matches[0]

    # Map the stats from whatever role they played to the requested role
    stat_suffixes = [
        'win_rate_last5', 'draw_rate_last5', 'loss_rate_last5',
        'avg_goals_scored_last5', 'avg_goals_conceded_last5', 'avg_goal_diff_last5',
        'win_rate_last10', 'draw_rate_last10', 'loss_rate_last10',
        'avg_goals_scored_last10', 'avg_goals_conceded_last10', 'avg_goal_diff_last10',
        'days_since_last'
    ]

    stats = {}
    for suffix in stat_suffixes:
        source_col = f'{match_role}_{suffix}'
        target_col = f'{prefix}{suffix}'
        stats[target_col] = latest.get(source_col, 0)

    return stats


def get_h2h_stats(df, home_team, away_team):
    """Get head-to-head stats from the most recent meeting."""
    meetings = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
        ((df['home_team'] == away_team) & (df['away_team'] == home_team))
    ]

    if meetings.empty:
        return {'h2h_home_win_rate': 0.5, 'h2h_total_matches': 0}

    latest = meetings.iloc[-1]
    return {
        'h2h_home_win_rate': latest.get('h2h_home_win_rate', 0.5),
        'h2h_total_matches': latest.get('h2h_total_matches', 0)
    }


def get_tournament_importance(tournament_type):
    """Map tournament type to importance weight."""
    mapping = {
        'Major Tournament (World Cup, Euros, Copa America, etc.)': 1.0,
        'Qualification Match': 0.7,
        'Regional / Minor Tournament': 0.5,
        'Friendly': 0.3
    }
    return mapping.get(tournament_type, 0.5)


def get_team_elo(df, team):
    """Get the latest ELO rating for a team from featured data."""
    # Check if team played as home
    home_matches = df[df['home_team'] == team]
    away_matches = df[df['away_team'] == team]

    latest_elo = 1500  # default

    candidates = []
    if not home_matches.empty:
        row = home_matches.iloc[-1]
        candidates.append((row['date'], row.get('home_elo', 1500)))
    if not away_matches.empty:
        row = away_matches.iloc[-1]
        candidates.append((row['date'], row.get('away_elo', 1500)))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        latest_elo = candidates[0][1]

    return latest_elo


# ══════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .prob-bar {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .stSelectbox label, .stCheckbox label {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align:center;'>⚽ Football Match Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Predict international football match outcomes using ML</p>", unsafe_allow_html=True)
st.divider()

# Load everything
model, scaler, label_encoder, feature_names = load_artifacts()
df = load_historical_data()

teams = sorted(df['home_team'].unique().tolist())

# ── Team Selection ───────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox("Select home team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)

with col2:
    st.subheader("✈️ Away Team")
    away_team = st.selectbox("Select away team", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)

# ── Match Settings ───────────────────────────────────────────
st.divider()
col3, col4 = st.columns(2)

with col3:
    tournament = st.selectbox("🏆 Tournament Type", [
        'Major Tournament (World Cup, Euros, Copa America, etc.)',
        'Qualification Match',
        'Regional / Minor Tournament',
        'Friendly'
    ])

with col4:
    neutral_venue = st.checkbox("🏟️ Neutral Venue", value=False)

# ── Predict Button ───────────────────────────────────────────
st.divider()

if home_team == away_team:
    st.warning("⚠️ Please select two different teams.")
else:
    if st.button("🔮 Predict Outcome", use_container_width=True, type="primary"):

        # Build feature vector
        home_stats = get_team_stats(df, home_team, 'home')
        away_stats = get_team_stats(df, away_team, 'away')
        h2h = get_h2h_stats(df, home_team, away_team)

        if home_stats is None or away_stats is None:
            st.error("❌ Not enough historical data for one of the selected teams.")
        else:
            features = {
                'neutral': int(neutral_venue),
                'tournament_importance': get_tournament_importance(tournament),
            }
            features.update(home_stats)
            features.update(away_stats)
            features.update(h2h)

            # ELO features
            home_elo = get_team_elo(df, home_team)
            away_elo = get_team_elo(df, away_team)
            features['home_elo'] = home_elo
            features['away_elo'] = away_elo
            features['elo_diff'] = home_elo - away_elo

            # Create DataFrame in correct column order
            X = pd.DataFrame([features])[feature_names]

            # Scale and predict
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            predicted_label = label_encoder.inverse_transform([prediction])[0]

            # Class order from label encoder
            classes = label_encoder.classes_  # ['Away Win', 'Draw', 'Home Win']

            # Display prediction
            st.markdown("---")

            # Result colors
            color_map = {
                'Home Win': '#27ae60',
                'Draw': '#f39c12',
                'Away Win': '#e74c3c'
            }
            result_color = color_map.get(predicted_label, '#3498db')

            st.markdown(
                f"<div style='text-align:center; padding:1.5rem; "
                f"background: linear-gradient(135deg, {result_color}22, {result_color}44); "
                f"border: 2px solid {result_color}; border-radius:12px;'>"
                f"<h2 style='margin:0; color:{result_color};'>🏆 {predicted_label}</h2>"
                f"<p style='margin:0.5rem 0 0 0; color:gray;'>{home_team} vs {away_team}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            st.markdown("")

            # Probability bars
            st.subheader("📊 Prediction Probabilities")
            for i, cls in enumerate(classes):
                prob = probabilities[i]
                bar_color = color_map.get(cls, '#3498db')

                col_label, col_bar, col_pct = st.columns([2, 6, 1])
                with col_label:
                    st.markdown(f"**{cls}**")
                with col_bar:
                    st.progress(prob)
                with col_pct:
                    st.markdown(f"**{prob*100:.1f}%**")

            # Team stats preview
            with st.expander("📋 Feature Values Used"):
                st.dataframe(X.T.rename(columns={0: 'Value'}).style.format("{:.4f}"))

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
    "Built with Streamlit • Model: Random Forest (tuned) • F1 Score: 0.5272"
    "</p>",
    unsafe_allow_html=True
)
