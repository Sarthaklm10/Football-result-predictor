# EDA Code Explained (Brief)

## Setup (Lines 12-55)

```python
import pandas as pd          # Data manipulation
import numpy as np           # Numerical operations
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns        # Statistical plots
from pathlib import Path     # File path handling
```

**Config:** `plt.style.use('seaborn-v0_8-darkgrid')` sets a clean visual theme. `plt.rcParams.update({...})` sets default figure size, font, and DPI (image quality).

**Path setup:** `Path(__file__).resolve().parent.parent` means "go up from this file's folder (notebooks/) to the project root." This way the script works regardless of where you run it from.

**Derived columns we create immediately after loading:**
- `df['date'] = pd.to_datetime(...)` — convert string to datetime (enables `.dt.year`, `.dt.month`)
- `df['year']`, `df['month']`, `df['decade']` — extracted from date for grouping
- `df['result']` — target variable via `np.where()` (Home Win / Draw / Away Win)
- `df['total_goals']` — home_score + away_score

---

## The 14 Plots — Key Code Patterns

### Plot 1: Missing Values (bar chart)
`df.isnull().sum()` counts nulls per column. Colors bars green (0 nulls) or red (has nulls). Simple `ax.bar()` call.

### Plot 2: Target Distribution (bar + pie)
`df['result'].value_counts()` counts each class. `axes[0].bar()` makes bar chart, `axes[1].pie()` makes pie chart. `plt.subplots(1, 2)` creates two side-by-side plots.

### Plot 3: Matches Per Year (area chart)
`df.groupby('year').size()` counts matches per year. `ax.fill_between()` creates the shaded area. `ax.axvline(x=2000)` draws a vertical reference line.

### Plot 4: Most Active Countries (horizontal bar)
`home_c.add(away_c, fill_value=0)` combines home + away match counts per team. `.sort_values(ascending=False).head(15)` gets top 15. `ax.barh()` draws horizontal bars. `ax.invert_yaxis()` puts #1 at top.

### Plot 5: Win Percentage (horizontal bar with gradient)
Counts home wins per team + away wins per team, divides by total matches. `total_matches >= 100` filters out tiny teams. `plt.cm.RdYlGn()` creates a green-to-red color gradient.

### Plot 6: Goals Distribution (histogram)
`axes[0].hist(df['home_score'], bins=range(0,15))` creates histogram. Two overlapping histograms (home blue, away red) with `alpha=0.5` for transparency.

### Plot 7: Goals per Decade (bar + line combo)
`df.groupby('decade')['total_goals'].mean()` computes average per decade. Bar chart shows magnitude, line chart shows trend.

### Plot 8: Tournament Distribution (horizontal bar)
`df['tournament'].value_counts().head(12)` gets top 12 tournaments. `plt.cm.viridis()` creates a purple-to-yellow color gradient.

### Plot 9: Home Advantage (two pie charts)
First pie: all matches. Second pie: `df[df['neutral'] == False]` (non-neutral only). Side-by-side comparison shows home advantage increases when excluding neutral venues.

### Plot 10: Neutral vs Non-Neutral (grouped bar)
`value_counts(normalize=True) * 100` gives percentages. Two `ax.bar()` calls offset by `width/2` create the grouped effect (blue = non-neutral, orange = neutral).

### Plot 11: Draw % Over Time (line + rolling average)
`groupby('year').apply(lambda x: (x['result']=='Draw').mean()*100)` calculates draw % per year. `.rolling(window=10).mean()` smooths the noisy yearly data into a 10-year moving average.

### Plot 12: Correlation Heatmap
`df[['home_score','away_score','neutral','total_goals']].corr()` computes pairwise correlation matrix. `sns.heatmap(annot=True, cmap='RdBu_r')` visualizes it with red (negative) to blue (positive) colors.

### Plot 13: Outlier Box Plots
`ax.boxplot(data, patch_artist=True)` creates a box-and-whisker plot. The box shows Q1-Q3 range, the line is median, whiskers extend to 1.5x IQR, and dots beyond are outliers.

### Plot 14: Home Advantage Over Decades (stacked bar)
`value_counts(normalize=True).unstack()` pivots results into columns. Three `ax.bar()` calls with `bottom=` parameter stack Home Win → Draw → Away Win on top of each other to always total 100%.

---

## Key Pandas Functions Used

| Function | What It Does |
|----------|-------------|
| `groupby('col').size()` | Count rows per group |
| `groupby('col')['other'].mean()` | Average of a column per group |
| `value_counts()` | Count unique values |
| `value_counts(normalize=True)` | Same but as proportions (0-1) |
| `.add(other, fill_value=0)` | Add two Series, treating missing as 0 |
| `.unstack()` | Pivot a multi-index into columns |
| `.rolling(window=10).mean()` | 10-period moving average |
| `.corr()` | Pairwise correlation matrix |
