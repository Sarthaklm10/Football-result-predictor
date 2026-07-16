"""
===============================================================
STEP 6: Feature Engineering
===============================================================
International Football Match Outcome Predictor

THE most important step. Raw columns like 'home_team' don't
help a model predict outcomes. But 'home_team_win_rate_last10'
tells the model HOW GOOD a team has been recently.

GOLDEN RULE: Every feature is computed using ONLY past matches.
             Never peek into the future. This prevents data leakage.

Features we create:
  For each team (home and away):
    - Win/Draw/Loss rate (last 5, last 10 matches)
    - Avg goals scored (last 5, last 10)
    - Avg goals conceded (last 5, last 10)
    - Goal difference (last 5, last 10)
    - Days since last match
  Match-level:
    - Head-to-head win rate
    - Tournament importance weight
    - Neutral venue flag (already exists)
===============================================================
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
    """Convert match data into a team-centric view.

    Problem: In each row, a team can be 'home_team' OR 'away_team'.
    Brazil might be home in row 5 and away in row 12. To compute
    'Brazil's last 5 games', we need ALL of Brazil's matches in
    one view.

    Solution: Create two copies of each match — one from each
    team's perspective — then combine them.

    Example:
      Original row:  Brazil 3 - 1 Argentina  (Home Win)

      Becomes two rows:
        team=Brazil,    goals_for=3, goals_against=1, win=1, is_home=1
        team=Argentina, goals_for=1, goals_against=3, win=0, is_home=0
    """
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
    """Compute rolling statistics per team using a given window.

    CRITICAL: We use .shift(1) before rolling. This ensures we
    only use PAST matches, never the current match.

    Without shift:  rolling includes the current match = DATA LEAKAGE
    With shift(1):  rolling uses only previous matches = SAFE

    Example (window=3, for Brazil):
      Match 1: Win   → stats = NaN (not enough history)
      Match 2: Loss  → stats = NaN (not enough history)
      Match 3: Win   → stats = NaN (not enough history)
      Match 4: Draw  → stats based on matches 1,2,3 ← SAFE
      Match 5: Win   → stats based on matches 2,3,4 ← SAFE
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
    """Calculate days since a team's previous match.

    Intuition: A team that played 3 days ago might be tired.
    A team that hasn't played in 60 days might be rusty.
    Both extremes can affect performance.

    We use .shift(1) to get the PREVIOUS match date, then
    subtract from current date.
    """
    prev_date = team_history.groupby('team')['date'].shift(1)
    days_since = (team_history['date'] - prev_date).dt.days
    return days_since


# ══════════════════════════════════════════════════════════════
# FEATURE: Head-to-Head Record
# ══════════════════════════════════════════════════════════════

def compute_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical head-to-head win rate between two teams.

    For each match, we look at ALL previous meetings between
    home_team and away_team, and compute:
      - h2h_home_win_rate: how often the home team won in past meetings
      - h2h_total_matches: how many times they've played before

    This captures rivalries and historical dominance.
    Example: Brazil vs Argentina have played 100+ times.
    If Brazil won 60% historically, that's a strong signal.
    """
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

def compute_tournament_importance(df: pd.DataFrame) -> pd.Series:
    """Assign an importance weight to each tournament.

    Intuition: Teams try harder in World Cup than in friendlies.
    A World Cup match is more predictive of true team strength
    than a friendly where squads are rotated.

    We group ~200 tournament names into 4 tiers:
      Tier 1 (weight 1.0): Major tournaments (World Cup, Euros, etc.)
      Tier 2 (weight 0.7): Qualification matches
      Tier 3 (weight 0.5): Regional/minor tournaments
      Tier 4 (weight 0.3): Friendlies
    """
    def classify(tournament: str) -> float:
        t = tournament.lower()
        # Tier 1: Major finals
        if any(kw in t for kw in ['fifa world cup', 'uefa euro',
               'copa am', 'african cup of nations',
               'afc asian cup', 'concacaf gold cup',
               'confederations cup', 'nations league']):
            if 'qualification' not in t:
                return 1.0
        # Tier 2: Qualifiers
        if 'qualification' in t or 'qualifying' in t:
            return 0.7
        # Tier 3: Other named tournaments
        if 'friendly' not in t:
            return 0.5
        # Tier 4: Friendlies
        return 0.3

    return df['tournament'].apply(classify)


# ══════════════════════════════════════════════════════════════
# MAIN: Assemble All Features
# ══════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the complete feature matrix.

    Steps:
    1. Build team-centric view (each match -> 2 rows, one per team)
    2. Compute rolling stats (last 5, last 10) for each team
    3. Map these stats back to the original match rows
    4. Add head-to-head, tournament importance, days since last
    """
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
    """Remove rows where teams don't have enough history.

    Teams' first few matches have NaN features because there's
    no prior history to compute rolling stats from. We drop these.

    We require at least 1 past match (min_periods=1 in rolling),
    so mainly the very first match of each team will have NaN.
    """
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
