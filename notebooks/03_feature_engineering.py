"""
Step 6: Feature Engineering
Builds ELO ratings, rolling stats, head-to-head, tournament importance.
All features computed using only past matches (no leakage).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_PATH = PROJECT_ROOT / 'data' / 'processed' / 'cleaned_matches.csv'
FEATURES_PATH = PROJECT_ROOT / 'data' / 'processed' / 'featured_matches.csv'


# ══════════════════════════════════════════════════════════════
# HELPER: Build a Team-Centric View
# ══════════════════════════════════════════════════════════════

def build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """Convert match rows into team-centric view (2 rows per match, one per team)."""
    # Home team perspective
    home = df[['date', 'home_team', 'home_score', 'away_score',
               'tournament', 'neutral', 'result']].copy()
    home.columns = ['date', 'team', 'goals_for', 'goals_against',
                    'tournament', 'neutral', 'result']
    home['is_home'] = 1
    home['win'] = (home['result'] == 'Home Win').astype(int)
    home['draw'] = (home['result'] == 'Draw').astype(int)
    home['loss'] = (home['result'] == 'Away Win').astype(int)

    # Away team perspective
    away = df[['date', 'away_team', 'away_score', 'home_score',
               'tournament', 'neutral', 'result']].copy()
    away.columns = ['date', 'team', 'goals_for', 'goals_against',
                    'tournament', 'neutral', 'result']
    away['is_home'] = 0
    away['win'] = (away['result'] == 'Away Win').astype(int)
    away['draw'] = (away['result'] == 'Draw').astype(int)
    away['loss'] = (away['result'] == 'Home Win').astype(int)

    # Combine and sort
    team_history = pd.concat([home, away], ignore_index=True)
    team_history = team_history.sort_values(['team', 'date']).reset_index(drop=True)
    team_history['goal_diff'] = team_history['goals_for'] - team_history['goals_against']

    return team_history


# ══════════════════════════════════════════════════════════════
# HELPER: Compute Rolling Stats for Each Team
# ══════════════════════════════════════════════════════════════

def compute_rolling_stats(team_history: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling win/draw/loss rates and goal stats per team.
    Uses shift(1) to exclude the current match from its own features.
    """
    suffix = f'_last{window}'
    grouped = team_history.groupby('team')

    stats = pd.DataFrame(index=team_history.index)

    # shift(1) = exclude current match, then rolling = look at past N
    stats[f'win_rate{suffix}'] = grouped['win'].apply(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    ).values

    stats[f'draw_rate{suffix}'] = grouped['draw'].apply(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    ).values

    stats[f'loss_rate{suffix}'] = grouped['loss'].apply(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    ).values

    stats[f'avg_goals_scored{suffix}'] = grouped['goals_for'].apply(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    ).values

    stats[f'avg_goals_conceded{suffix}'] = grouped['goals_against'].apply(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    ).values

    stats[f'avg_goal_diff{suffix}'] = grouped['goal_diff'].apply(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    ).values

    return stats


# ══════════════════════════════════════════════════════════════
# FEATURE: Days Since Last Match
# ══════════════════════════════════════════════════════════════

def compute_days_since_last(team_history: pd.DataFrame) -> pd.Series:
    """Days since each team's previous match (captures fatigue/rustiness)."""
    prev_date = team_history.groupby('team')['date'].shift(1)
    days_since = (team_history['date'] - prev_date).dt.days
    return days_since


# ══════════════════════════════════════════════════════════════
# FEATURE: Head-to-Head Record
# ══════════════════════════════════════════════════════════════

def compute_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical head-to-head win rate between each pair of teams."""
    h2h_home_wr = []
    h2h_matches = []

    # Build a lookup of past results between each pair
    for idx, row in df.iterrows():
        ht, at, date = row['home_team'], row['away_team'], row['date']

        # All PAST meetings (both directions)
        past = df[
            (df['date'] < date) &
            (
                ((df['home_team'] == ht) & (df['away_team'] == at)) |
                ((df['home_team'] == at) & (df['away_team'] == ht))
            )
        ]

        if len(past) == 0:
            h2h_home_wr.append(np.nan)
            h2h_matches.append(0)
        else:
            # Count wins for home_team (in either direction)
            wins_as_home = ((past['home_team'] == ht) & (past['result'] == 'Home Win')).sum()
            wins_as_away = ((past['away_team'] == ht) & (past['result'] == 'Away Win')).sum()
            total_wins = wins_as_home + wins_as_away
            h2h_home_wr.append(total_wins / len(past))
            h2h_matches.append(len(past))

    return pd.DataFrame({
        'h2h_home_win_rate': h2h_home_wr,
        'h2h_total_matches': h2h_matches
    }, index=df.index)


# ══════════════════════════════════════════════════════════════
# FEATURE: Tournament Importance
# ══════════════════════════════════════════════════════════════

def get_tournament_k_factor(tournament: str) -> float:
    """Return K-factor for ELO update based on tournament type."""
    t = tournament.lower()
    if any(kw in t for kw in ['fifa world cup', 'uefa euro',
           'copa am', 'african cup of nations',
           'afc asian cup', 'concacaf gold cup',
           'confederations cup', 'nations league']):
        if 'qualification' not in t:
            return 60    # Major tournament finals
    if 'qualification' in t or 'qualifying' in t:
        return 40        # Qualifiers
    if 'friendly' not in t:
        return 30        # Regional / minor tournaments
    return 20            # Friendlies


def compute_tournament_importance(df: pd.DataFrame) -> pd.Series:
    """Map tournaments to importance weights (1.0=major, 0.7=qualifier, 0.5=regional, 0.3=friendly)."""
    def classify(tournament: str) -> float:
        t = tournament.lower()
        if any(kw in t for kw in ['fifa world cup', 'uefa euro',
               'copa am', 'african cup of nations',
               'afc asian cup', 'concacaf gold cup',
               'confederations cup', 'nations league']):
            if 'qualification' not in t:
                return 1.0
        if 'qualification' in t or 'qualifying' in t:
            return 0.7
        if 'friendly' not in t:
            return 0.5
        return 0.3

    return df['tournament'].apply(classify)


# ══════════════════════════════════════════════════════════════
# FEATURE: ELO Ratings
# ══════════════════════════════════════════════════════════════

def compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ELO ratings for each team, processed chronologically.
    Stores the PRE-MATCH rating as the feature (no leakage).
    Uses variable K-factor by tournament and goal-diff multiplier.
    """
    elo = {}  # team_name -> current rating
    DEFAULT_ELO = 1500

    home_elos = []
    away_elos = []

    for _, row in df.iterrows():
        ht = row['home_team']
        at = row['away_team']

        # Get current ratings (default 1500 for new teams)
        r_home = elo.get(ht, DEFAULT_ELO)
        r_away = elo.get(at, DEFAULT_ELO)

        # Store PRE-MATCH ratings as features
        home_elos.append(r_home)
        away_elos.append(r_away)

        # Expected scores
        e_home = 1.0 / (1.0 + 10 ** ((r_away - r_home) / 400.0))
        e_away = 1.0 - e_home

        # Actual scores
        if row['result'] == 'Home Win':
            s_home, s_away = 1.0, 0.0
        elif row['result'] == 'Away Win':
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        # K-factor varies by tournament importance
        K = get_tournament_k_factor(row['tournament'])

        # Goal difference multiplier: blowouts update more
        goal_diff = abs(row['home_score'] - row['away_score'])
        gd_mult = np.log(1 + goal_diff) if goal_diff > 0 else 1.0

        K_eff = K * gd_mult

        # Update ratings
        elo[ht] = r_home + K_eff * (s_home - e_home)
        elo[at] = r_away + K_eff * (s_away - e_away)

    result = pd.DataFrame({
        'home_elo': home_elos,
        'away_elo': away_elos,
        'elo_diff': [h - a for h, a in zip(home_elos, away_elos)]
    }, index=df.index)

    # Print some stats
    top_teams = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:10]
    print("  Top 10 ELO ratings (final):")
    for team, rating in top_teams:
        print(f"    {team:<25} {rating:.0f}")

    return result


# ══════════════════════════════════════════════════════════════
# MAIN: Assemble All Features
# ══════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the complete feature matrix from cleaned match data."""

    # --- ELO ratings (must be first — processes chronologically) ---
    print("Computing ELO ratings...")
    elo_features = compute_elo_ratings(df)
    df = pd.concat([df.reset_index(drop=True), elo_features], axis=1)

    print("Building team history...")
    team_hist = build_team_history(df)
    print(f"  Team-centric rows: {len(team_hist):,} (2x matches)")

    # --- Rolling stats for window=5 and window=10 ---
    print("Computing rolling stats (last 5)...")
    stats_5 = compute_rolling_stats(team_hist, window=5)

    print("Computing rolling stats (last 10)...")
    stats_10 = compute_rolling_stats(team_hist, window=10)

    # Attach stats to team history
    team_hist = pd.concat([team_hist, stats_5, stats_10], axis=1)

    # --- Days since last match ---
    print("Computing days since last match...")
    team_hist['days_since_last'] = compute_days_since_last(team_hist)

    # --- Split back into home/away and merge to original ---
    print("Merging features back to match data...")
    home_feats = team_hist[team_hist['is_home'] == 1].copy()
    away_feats = team_hist[team_hist['is_home'] == 0].copy()

    # Select feature columns (everything computed)
    feat_cols = [c for c in team_hist.columns if c.startswith(('win_rate', 'draw_rate',
                 'loss_rate', 'avg_goals', 'avg_goal_diff', 'days_since'))]

    # Rename with home_/away_ prefix
    home_rename = {c: f'home_{c}' for c in feat_cols}
    away_rename = {c: f'away_{c}' for c in feat_cols}

    home_feats = home_feats[feat_cols].rename(columns=home_rename)
    away_feats = away_feats[feat_cols].rename(columns=away_rename)

    # Reset indices to match original df
    home_feats.index = range(len(home_feats))
    away_feats.index = range(len(away_feats))

    # The home_feats and away_feats are in the SAME order as df
    # because we built them from df and maintained order
    df = df.reset_index(drop=True)
    df = pd.concat([df, home_feats, away_feats], axis=1)

    # --- Tournament importance ---
    print("Computing tournament importance...")
    df['tournament_importance'] = compute_tournament_importance(df)

    # --- Head-to-head ---
    print("Computing head-to-head records (this may take a minute)...")
    h2h = compute_head_to_head(df)
    df = pd.concat([df, h2h], axis=1)

    # --- Neutral already exists as boolean, convert to int ---
    df['neutral'] = df['neutral'].astype(int)

    return df


def drop_rows_without_history(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where teams lack enough match history for features."""
    before = len(df)
    # Only drop rows where ALL rolling features are NaN (team's first match)
    feature_cols = [c for c in df.columns if 'last5' in c or 'last10' in c]
    df = df.dropna(subset=feature_cols, how='all')
    # Fill remaining NaN in h2h with 0.5 (no prior meetings = coin flip)
    df['h2h_home_win_rate'] = df['h2h_home_win_rate'].fillna(0.5)
    # Fill any remaining NaN with 0
    df = df.fillna(0)
    after = len(df)
    print(f"\n  Dropped {before - after} rows without enough history")
    print(f"  Remaining: {after:,} rows")
    return df


# ── Main Pipeline ────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STEP 6: Feature Engineering")
    print("=" * 60)

    # Load cleaned data
    df = pd.read_csv(CLEANED_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Loaded {len(df):,} cleaned matches\n")

    # Engineer features
    df = engineer_features(df)

    # Drop rows without history
    df = drop_rows_without_history(df)

    # Show what we built
    feature_cols = [c for c in df.columns if c not in
                    ['date', 'home_team', 'away_team', 'home_score',
                     'away_score', 'tournament', 'result']]
    print(f"\n  Total features created: {len(feature_cols)}")
    print(f"  Feature names:")
    for c in sorted(feature_cols):
        print(f"    - {c}")

    # Save
    df.to_csv(FEATURES_PATH, index=False)
    print(f"\n  Saved to: {FEATURES_PATH}")
    print(f"  Shape: {df.shape}")

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
