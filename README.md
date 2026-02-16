# Trader Performance vs Market Sentiment Analysis

**PrimeTrade.ai — Data Science Intern Assignment (Round-0)**

## Objective

Analyze how Bitcoin market sentiment (Fear/Greed Index) relates to trader behavior and performance on Hyperliquid, uncovering actionable patterns for smarter trading strategies.

## Datasets

| Dataset | Records | Source |
|---------|---------|--------|
| Bitcoin Fear & Greed Index | 2,644 daily records (2018–2025) | [Google Drive](https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf) |
| Hyperliquid Trader Data | 211,224 trades (32 accounts) | [Google Drive](https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs) |

## Project Structure

```
DS_Intern_PrimeTrade_Ass/
├── analysis.ipynb          # Main analysis notebook (all parts A/B/C + Bonus)
├── dashboard.py            # Streamlit dashboard (bonus)
├── data/
│   ├── sentiment_data.csv  # Fear/Greed index
│   └── trader_data.csv     # Hyperliquid trades
├── output/                 # Saved charts
│   ├── trade_size_distribution.png
│   ├── ls_ratio_trade_freq.png
│   ├── performance_fear_vs_greed.png
│   ├── behavioral_shifts.png
│   ├── segment_leverage.png
│   ├── segment_frequency.png
│   ├── segment_consistency.png
│   ├── insight1_pnl_segments.png
│   ├── insight2_ls_sentiment.png
│   ├── insight3_winrate_profitability.png
│   ├── bonus_feature_importance.png
│   ├── bonus_cluster_elbow.png
│   ├── bonus_cluster_pca.png
│   └── bonus_cluster_radar.png
├── requirements.txt
└── README.md
```

## Setup & How to Run

### Prerequisites
- Python 3.10+
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the notebook
```bash
jupyter notebook analysis.ipynb
# OR
jupyter lab analysis.ipynb
```
Run all cells sequentially (Kernel → Restart & Run All).

### Run the Streamlit dashboard (bonus)
```bash
streamlit run dashboard.py
```

## Key Findings

### Insight 1: Fear days → higher mean PnL but greater variance
Traders earn \$5,185/day on Fear vs \$3,973 on Greed (mean), but median PnL is lower (\$123 vs \$243). A few large wins during Fear skew the average — most traders actually perform worse.

### Insight 2: Counterintuitive long bias persists during Fear
Position sizes are **27% larger** and long/short ratio is **20% higher** during Fear. Traders lean into longs despite bearish sentiment, amplifying drawdown risk.

### Insight 3: Consistent winners ignore sentiment
The top-performing trader maintains ~90% profitable days regardless of Fear/Greed. Inconsistent traders see profitability drop on Fear days.

## Strategy Recommendations

1. **Sentiment-Adaptive Position Sizing:** Reduce position sizes by 20–30% during Fear days, especially for high-leverage traders. The data shows traders _increase_ exposure during Fear, which amplifies risk without proportional returns.

2. **Reduce Long Bias During Fear:** Shift toward neutral or slightly short positioning during Fear periods. The persistent long bias during Fear (L/S ratio 2.24 vs 1.78 on Greed days) is a contrarian signal that exposes traders to downside.

## Methodology

1. **Data Cleaning:** Zero duplicates or missing values in either dataset. 211,218 trades aligned with sentiment data.
2. **Feature Engineering:** Daily PnL, win rate, trade direction, position size, L/S ratio, drawdown proxy.
3. **Segmentation:** By position size (leverage proxy), trade frequency, and consistency (win rate).
4. **Statistical Tests:** Mann-Whitney U tests for all Fear vs Greed comparisons.
5. **Bonus — Predictive Model:** Random Forest (88% accuracy) using lagged behavioral features + sentiment.
6. **Bonus — Clustering:** KMeans (K=2, silhouette=0.54) identified mainstream traders (n=30) vs whale traders (n=2).

## Evaluation Criteria Addressed

| Criterion | Coverage |
|-----------|----------|
| Data cleaning + merge correctness | ✓ Zero missing/duplicates, proper date alignment, documented merges |
| Strength of reasoning | ✓ Statistical tests (Mann-Whitney U), percentage changes, segment comparisons |
| Quality of insights | ✓ 3 actionable insights with supporting evidence, not generic observations |
| Clarity of communication | ✓ Structured notebook with clear section headers, summary write-up |
| Reproducibility | ✓ requirements.txt, sequential notebook, saved outputs |
