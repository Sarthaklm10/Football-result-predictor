# Viva Q&A -- Step 6: Feature Engineering

## What Was Built

**30 features** from 8 raw columns. Here's the full list grouped by category:

| Category | Features (for both home & away) |
|----------|-------------------------------|
| **Win/Draw/Loss Rate** | win_rate_last5, win_rate_last10, draw_rate_last5, draw_rate_last10, loss_rate_last5, loss_rate_last10 |
| **Goals** | avg_goals_scored_last5, avg_goals_scored_last10, avg_goals_conceded_last5, avg_goals_conceded_last10 |
| **Goal Difference** | avg_goal_diff_last5, avg_goal_diff_last10 |
| **Recency** | days_since_last |
| **Head-to-Head** | h2h_home_win_rate, h2h_total_matches |
| **Match Context** | tournament_importance, neutral |

Each "for both home & away" means we compute it twice: once for the home team's history, once for the away team's history. That's how 13 base stats become 26 + 4 match-level = 30 total.

---

## Viva Questions

### Q1: What is feature engineering and why is it the most important step?

**Answer:** Feature engineering is transforming raw data into meaningful numerical inputs that help the ML model learn patterns. Raw data like `home_team = "Brazil"` means nothing to a model. But `home_win_rate_last10 = 0.80` tells the model Brazil has been dominant recently.

The model can only learn from what you give it. Better features = better predictions, regardless of which algorithm you use.

**Interview-Ready:** *"Feature engineering transforms raw columns into predictive signals. I created 30 features capturing recent form, goal-scoring patterns, head-to-head history, and match context. This is the most impactful step -- better features matter more than algorithm choice."*

---

### Q2: What is the "team-centric view" and why do you need it?

**Answer:** In the raw data, Brazil appears sometimes as `home_team` and sometimes as `away_team`. To compute "Brazil's last 5 games," I need ALL of Brazil's matches in one place.

The solution: create two rows per match (one from each team's perspective), then sort by team + date. Now I can use `groupby('team').rolling(5)` to get each team's last 5 games.

**Interview-Ready:** *"I created a team-centric view by melting each match into two rows -- one per team's perspective. This allowed me to use Pandas groupby and rolling operations to compute per-team historical statistics efficiently."*

---

### Q3: What does `.shift(1)` do and why is it critical?

**Answer:** `.shift(1)` moves all values DOWN by one row. The current row sees the PREVIOUS row's value instead of its own.

Without shift: `rolling(5).mean()` includes the CURRENT match = **data leakage**
With shift(1): `rolling(5).mean()` uses only the 5 PREVIOUS matches = **safe**

```
Match    Score    shift(1)    rolling(3).mean()
  1       Win      NaN         NaN
  2       Loss     Win         NaN
  3       Win      Loss        NaN
  4       Draw     Win         Win,Loss,Win = 0.67  <-- uses matches 1,2,3 only
```

**Interview-Ready:** *"I used shift(1) before every rolling computation to exclude the current match from its own features. Without this, the model would see the current result in its input features -- a form of data leakage."*

---

### Q4: What is a rolling window and why use 5 and 10?

**Answer:** A rolling window of size N means "look at the last N matches." It captures RECENT form, not all-time history.

- **Window=5**: Captures very recent form (last month or so). Reacts quickly to hot/cold streaks.
- **Window=10**: Captures medium-term form. Smoother, less noisy.

Using both gives the model two perspectives: short-term momentum and medium-term consistency.

**Interview-Ready:** *"I used both 5-match and 10-match rolling windows to capture short-term momentum and medium-term consistency separately. The model can learn which time horizon is more predictive."*

---

### Q5: How does head-to-head work and why include it?

**Answer:** For each match, I look at ALL previous meetings between those two teams and compute the home team's historical win rate against that specific opponent.

Example: If Brazil has beaten Argentina in 6 out of their 10 previous meetings, `h2h_home_win_rate = 0.6`. This captures rivalries and psychological edges that general team form can't.

When teams have never met before, I fill with 0.5 (50% -- no advantage either way).

**Interview-Ready:** *"Head-to-head captures matchup-specific dynamics that general form statistics miss. Some teams consistently perform better against certain opponents due to tactical matchups or psychological factors."*

---

### Q6: How did you handle tournament importance?

**Answer:** I grouped ~200 tournament names into 4 tiers:

| Tier | Weight | Examples |
|------|--------|---------|
| 1 (Major finals) | 1.0 | FIFA World Cup, UEFA Euro, Copa America |
| 2 (Qualifiers) | 0.7 | World Cup qualification, Euro qualification |
| 3 (Regional) | 0.5 | CECAFA Cup, Merdeka Tournament |
| 4 (Friendlies) | 0.3 | Friendly |

This converts a categorical column with 200 values into a single meaningful number.

**Interview-Ready:** *"I encoded tournament type as an importance weight from 0.3 (friendlies) to 1.0 (major finals). This reduces 200 categorical values to one numeric feature that captures match stakes."*

---

### Q7: Why include "days since last match"?

**Answer:** It captures two effects:
- **Fatigue:** A team playing 3 days after their last match may be tired
- **Rustiness:** A team that hasn't played in 90 days may be out of rhythm

Both extremes can hurt performance. The model can learn the optimal rest period.

**Interview-Ready:** *"Days since last match captures fatigue (too few days) and rustiness (too many days). It provides match-specific context that rolling form statistics don't capture."*

---

### Q8: Why fill missing h2h with 0.5 instead of 0?

**Answer:** 0.5 means "50% win rate" -- a neutral prior. If two teams have never met, we assume neither has an advantage. Filling with 0 would mean "the home team has NEVER won against them," which is wrong -- they've just never played.

**Interview-Ready:** *"I used 0.5 as a neutral prior for teams with no head-to-head history, representing equal chances rather than zero wins."*

---

## Quick-Fire Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | How many features did you create? | 30 features from 8 raw columns |
| 2 | What prevents data leakage in rolling stats? | `.shift(1)` excludes the current match |
| 3 | Why two window sizes (5 and 10)? | Short-term momentum vs medium-term consistency |
| 4 | What is the team-centric view? | Each match split into 2 rows (one per team's perspective) |
| 5 | What does tournament_importance encode? | Match stakes: 1.0 (World Cup) to 0.3 (Friendly) |
| 6 | Why keep scores during feature engineering? | Needed to compute rolling avg goals -- dropped as direct features later |
| 7 | What does h2h_home_win_rate = 0.5 mean? | Teams have never met, or are evenly matched historically |
| 8 | How many rows survived? | 49,503 out of 49,504 (lost 1 with no history) |
