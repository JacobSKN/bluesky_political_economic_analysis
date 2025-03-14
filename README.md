# German Political Economic Discourse Analysis

This repository contains scripts and analysis of economic discourse on Bluesky around the 2025 German federal election (held on February 23, 2025). The analysis focuses on how economic topics were discussed before and after the election.

## Overview

This project analyzes economic language and topics in Bluesky posts around the 2025 German federal election through:
- Filtering posts containing economic terms
- Categorizing economic terms into thematic groups
- Analyzing changes in economic discourse before and after the election
- Applying regression discontinuity design (RDD) to identify significant shifts in topic prevalence

## Key Findings

- **Overall increase in economic discussion**: Economic content increased from 4.74% of posts before the election to 7.56% after (59.49% relative increase)
- **Shift in economic topics**: Significant changes in topic distribution after the election
  - Fiscal/Monetary topics increased by 18.38% (p<0.001)
  - Labor/Social topics decreased by 11.55% (p<0.001)
  - Environment & Energy topics decreased by 6.82% (p<0.05)
  - Housing & Property topics decreased by 2.91% (p<0.01)
- **Topic distribution**: Before the election, Labor/Social (22%), Environment/Energy (21%), and Digital Infrastructure (19%) dominated discussions. After the election, Fiscal/Monetary topics became dominant (38%), with Digital Infrastructure remaining stable (20%).

## Scripts

### 1. `filtering_economic_posts.py`
Main script for filtering and analyzing economic content in Bluesky posts.

**Key components:**
- `COMBINED_ECONOMIC_TERMS`: Comprehensive list of German economic terms
- `EnhancedPoliticalEconomicFilter`: Filters posts for economic content and categorizes terms
- `EconomicContentPreprocessor`: Preprocesses text for term extraction
- Various visualization functions:
  - `generate_category_pie_charts()`: Visualizes topic distribution before/after election
  - `economic_content_over_time_windows()`: Shows topic evolution over time
  - `analyze_economic_percentage_over_time()`: Tracks prevalence of economic content
  - `calculate_election_period_stats()`: Compares statistics between periods

### 2. `economic_RDD.py`
Implements regression discontinuity design analysis to detect causal effects of the election on economic discourse.

**Key functions:**
- `run_rdd_analysis()`: Main RDD analysis function
- `run_single_topic_rdd()`: Performs RDD for individual topics
- `analyze_economic_topic_volumes()`: Prepares data for RDD analysis
- `run_rdd_with_parameters()`: Wrapper for testing different RDD parameters

## Data Files

- `economic_percentage_stats.json`: Summary statistics of economic content prevalence
- `election_period_stats.json`: Detailed statistics of economic discourse before and after election

## Visualizations

The analysis generates several key visualizations:

1. **Topic distribution before and after election**
   - Bar chart showing percentage distribution of economic topics
   - Notable shifts in Fiscal/Monetary (increased) and Environment & Energy (decreased)

2. **Daily topic distribution over time**
   - Stacked percentage chart showing daily evolution of economic topics
   - Clear visualization of shifting discourse priorities around election day

## RDD Analysis Results

The following table shows the Regression Discontinuity Design results calculated using a 3-day rolling average and 14-day bandwidth:

| Category | RDD Results | p-value |
|-----------|----------------|---------|
| Fiskal/Monetär | 18.38 % | 0.0002 (***) |
| Arbeit/Soziales | -11.55 % | 0.0002 (***) |
| Marktwettbewerb | 0.47 % | 0.6067 |
| Umwelt & Energie | -6.82 % | 0.0250 (**) |
| Digitale Infrastruktur | -0.02 % | 0.9923 |
| Industrie | 1.43 % | 0.3200 |
| Finanzmärkte | -0.42 % | 0.4199 |
| Wohnen & Immobilien | -2.91 % | 0.0050 (***) |
| Öffentliche Politik | 1.45 % | 0.4681 |

Significance levels: *** p<0.01, ** p<0.05, * p<0.1

## Usage

```python
# Example: Filter posts and generate analysis
from filtering_economic_posts import EnhancedPoliticalEconomicFilter

# Initialize filter
economic_filter = EnhancedPoliticalEconomicFilter()

# Filter posts
economic_posts, analysis = economic_filter.filter_posts(posts_df)

# Run RDD analysis
from economic_RDD import run_rdd_with_parameters

results = run_rdd_with_parameters(
    economic_posts=economic_posts,
    analysis=analysis,
    event_date="2025-02-23",  # Election day
    bandwidth_days=14,
    polynomial_order=1,
    rolling_window=3
)
```

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scipy
- spacy (with `de_core_news_sm` model)

## Citation

If you use this code or analysis in your work, please cite:

```
@misc{german_economic_discourse_2025,
  author = {Jacob Schildknecht},
  title = {German Political Economic Discourse Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/JacobSKN/german-economic-discourse}
}
```
