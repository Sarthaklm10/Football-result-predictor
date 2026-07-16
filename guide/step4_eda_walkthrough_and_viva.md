# STEP 4: Exploratory Data Analysis (EDA) - Walkthrough & Viva Q&A

## What is EDA?

**Intuition:** EDA is like a detective investigating a crime scene before forming any theories. You observe everything, take notes, measure things, look for patterns, and spot anything unusual. You do this BEFORE jumping to conclusions.

In ML, EDA means **understanding your data deeply** before building any models. Models are only as good as the data and features you give them.

**Rule of thumb:** Spend 20-30% of your total project time on EDA.

---

## The 14 Visualizations We Created

All plots are saved in `reports/` folder. Here's what each one teaches us:

### Plot 1: Missing Values
- **Why:** Missing data can bias analysis and crash ML algorithms
- **Finding:** Only 11 nulls in 49,509 rows (0.02%) -- negligible
- **ML action:** We can safely drop these few rows instead of imputing

### Plot 2: Target Variable Distribution
- **Why:** We need to check if classes are balanced
- **Finding:** Home Win (49%), Away Win (28.3%), Draw (22.7%)
- **ML action:** Classes are moderately imbalanced. Home Win dominates. A dummy model predicting "Home Win always" would get 49% accuracy -- our model must beat this baseline

### Plot 3: Matches Per Year
- **Why:** Data from 1872 isn't representative of modern football
- **Finding:** Massive growth post-1990. Peak: 2024 (1,231 matches)
- **ML action:** We may want to weight recent data more heavily, or filter to post-2000 era

### Plot 4: Most Active Countries
- **Why:** Teams with few matches produce unreliable statistics
- **Finding:** Sweden (1,105), England (1,097), Argentina (1,076) are most active
- **ML action:** Features like "last 10 game win %" are unreliable for teams with <50 total matches

### Plot 5: Win Percentage by Country
- **Why:** Raw win counts are misleading without context of total games
- **Finding:** Brazil (63.4%), Spain (59%), Germany (58%) lead among teams with 100+ matches
- **ML action:** Historical win percentage is a strong predictive feature

### Plot 6: Goals Distribution
- **Why:** Understanding scoring patterns helps design features and detect outliers
- **Finding:** Most teams score 0-3 goals per match. Max: 31 goals in one match!
- **ML action:** Rolling average goals will be useful; need to handle outliers carefully

### Plot 7: Average Goals per Decade
- **Why:** Football has changed over 150 years
- **Finding:** 1880s: 5.6 goals/match. 2020s: 2.7 goals/match. Huge decline.
- **ML action:** Old data has very different patterns. Features should be computed from recent history

### Plot 8: Tournament Distribution
- **Why:** Tournament type affects match intensity
- **Finding:** Friendlies = 37% of all matches. FIFA World Cup = only 2.2%
- **ML action:** Tournament type is a feature indicating match importance (World Cup > Friendly)

### Plot 9: Home Advantage
- **Why:** Home advantage is the most studied phenomenon in sports analytics
- **Finding:** All matches: 49% home wins. Non-neutral venues: 50.7% home wins
- **ML action:** The "is home team?" signal is a strong baseline feature

### Plot 10: Neutral Venue Impact
- **Why:** Neutral venues should reduce home advantage
- **Finding:** Home win drops from 50.7% to 44.2% on neutral ground. Away win rises from 26.4% to 33.4%
- **ML action:** The `neutral` column is a **critical feature** -- confirmed by data

### Plot 11: Draw Percentage Over Time
- **Why:** Understanding draw patterns helps set realistic expectations
- **Finding:** Draw % is relatively stable at 22-26% across decades
- **ML action:** Draws are inherently hardest to predict (~23% base rate). Our model will likely struggle most with this class

### Plot 12: Correlation Heatmap
- **Why:** Shows linear relationships between numerical variables
- **Finding:** home_score and away_score are weakly negatively correlated (-0.145). Neutral has slight positive correlation with away_score (0.081)
- **ML action:** Low correlations mean features carry independent information (good!)

### Plot 13: Outlier Box Plots
- **Why:** Extreme values distort statistics and model training
- **Finding:** Most scores 0-5. Anything above 8 is a statistical outlier
- **ML action:** When computing rolling averages, outlier scores (like 31-0) could skew team statistics

### Plot 14: Home Advantage Over Decades
- **Why:** Has home advantage changed over time?
- **Finding:** Home advantage has slightly decreased in recent decades
- **ML action:** Model should learn from recent data to capture current trends

---

## 10 Key Findings Summary

| # | Finding | ML Implication |
|---|---------|---------------|
| 1 | 49,509 matches, 1872-2026 | Large dataset, but old data may hurt |
| 2 | Only 11 missing values | Safe to drop, no imputation needed |
| 3 | 0 duplicates | Clean data |
| 4 | Home Win 49%, Away 28%, Draw 23% | Moderately imbalanced -- need class weights |
| 5 | Home advantage: 51% non-neutral | Neutral flag is critical feature |
| 6 | Goals declining over decades | Recent data is more representative |
| 7 | 200 tournaments, 327 teams | High cardinality -- need smart encoding |
| 8 | Friendlies = 37% | Tournament type indicates match importance |
| 9 | 477 matches with 10+ goals | Outliers need handling in feature engineering |
| 10 | Brazil 63% win rate leads | Historical win rate is strong feature |

---

## Viva Q&A for Step 4

### Q1: What is EDA and why is it important?

**Answer:** EDA (Exploratory Data Analysis) is the process of analyzing datasets to summarize their main characteristics using statistics and visualizations. It's done BEFORE building models to understand data quality, find patterns, detect outliers, and inform feature engineering decisions.

**Interview-Ready:** *"I spent significant time on EDA to understand data quality, discover patterns like home advantage, detect outliers, and validate that my features would be meaningful for the model."*

---

### Q2: What is class imbalance and how does it affect your model?

**Answer:** Class imbalance means one class has significantly more samples than others. Our target has Home Win (49%), Away Win (28%), Draw (23%). A model could achieve 49% accuracy by always predicting "Home Win" without learning anything.

**How to handle it:**
- Use **class weights** to penalize misclassifying minority classes more
- Use **stratified sampling** to maintain class proportions in train/test split
- Evaluate with **F1 score** (not accuracy) since F1 accounts for both precision and recall

**Interview-Ready:** *"Our classes are moderately imbalanced with Home Win at 49%. A naive baseline achieves 49% by always predicting Home Win. I used class weights and evaluated with macro F1 to ensure the model performs well across all three classes."*

---

### Q3: What does home advantage look like in your data?

**Answer:** Home teams win 49% of all matches, but 50.7% when excluding neutral venues. On neutral ground, home wins drop to 44.2% and away wins rise to 33.4%. This 6.5 percentage point difference proves the `neutral` flag is a meaningful feature.

**Interview-Ready:** *"Home teams win 50.7% on home soil but only 44.2% on neutral venues -- a 6.5 point drop. This confirmed that the neutral venue flag is a significant predictor of match outcomes."*

---

### Q4: Why is the neutral venue flag important for your model?

**Answer:** It captures whether a match is played at a neutral venue (like a World Cup in a third country). On neutral ground, neither team has the crowd, climate, or travel advantage. Our data shows home win rate drops from 50.7% to 44.2% on neutral venues -- a statistically significant difference that the model should learn.

**Interview-Ready:** *"Neutral venue eliminates home advantage effects like crowd support and travel fatigue. My EDA confirmed a 6.5% drop in home win rate on neutral ground, validating it as a key feature."*

---

### Q5: Why are draws hardest to predict?

**Answer:** Draws are the least common class (22.7%) so the model has fewer examples to learn from. Also, draws are inherently unpredictable -- they represent matches where teams are very evenly matched, which is hard to distinguish from matches that could go either way.

**Interview-Ready:** *"Draws are both the minority class (23%) and inherently uncertain events. Even professional sports prediction models struggle with draws because they represent situations where team strengths are nearly equal."*

---

### Q6: What outliers did you find and how would you handle them?

**Answer:** 477 matches had 10+ total goals, including Australia 31-0 American Samoa. These are real results but statistically extreme.

**Handling approaches:**
- **Don't remove them** -- they're real data
- **Cap/clip** values when computing rolling averages (e.g., cap at 10 goals)
- **Use median** instead of mean for rolling statistics (median is robust to outliers)

**Interview-Ready:** *"I found 477 extreme-scoring matches. Rather than removing real data, I used median-based statistics and capping when computing rolling features to prevent outliers like the 31-0 result from distorting team statistics."*

---

### Q7: Why has average goals per match decreased over time?

**Answer:** Football has evolved toward more organized defense, tactical sophistication, and professional training. The 1880s averaged 5.6 goals/match; the 2020s average 2.7. This means features computed from old matches (like "average goals scored") would misrepresent a team's current attacking ability.

**Interview-Ready:** *"Goals have declined from 5.6 per match in the 1880s to 2.7 in the 2020s due to tactical evolution. This is why I compute features using only recent matches rather than all-time history."*

---

### Q8: Why is tournament type a useful feature?

**Answer:** Different tournaments have different levels of competitiveness:
- **World Cup**: Maximum effort, best players, high stakes
- **Friendly**: Experimental squads, less motivation, more rotation

Teams behave differently depending on stakes. A World Cup match between Brazil and Argentina is very different from a friendly.

**Interview-Ready:** *"Tournament type captures match importance. Teams field stronger squads and play more defensively in World Cup matches versus friendlies, making it a valuable predictive signal."*

---

### Q9: What does a box plot tell you that a histogram doesn't?

**Answer:** A box plot shows the **five-number summary** in one visual: minimum, Q1 (25th percentile), median, Q3 (75th percentile), and maximum. It also explicitly marks outliers as dots beyond the whiskers. Histograms show the full distribution shape but make outliers harder to spot.

**Interview-Ready:** *"I used box plots alongside histograms because box plots make the five-number summary and outliers immediately visible, while histograms better show the overall distribution shape."*

---

### Q10: What's the difference between correlation and causation?

**Answer:** Correlation means two variables move together. Causation means one CAUSES the other. Our heatmap shows neutral venues correlate with higher away scores (0.081). But neutral venues don't CAUSE away teams to score more -- the correlation exists because neutral venues remove the home team's defensive advantage.

**Interview-Ready:** *"Correlation measures linear association but doesn't imply causation. The slight correlation between neutral venues and away goals exists because neutral venues remove home advantage, not because the venue directly causes more away goals."*
