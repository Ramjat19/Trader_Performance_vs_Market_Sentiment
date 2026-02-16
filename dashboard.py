"""
Streamlit Dashboard: Trader Performance vs Market Sentiment
PrimeTrade.ai — Data Science Intern Assignment

Run: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Trader vs Sentiment", layout="wide", page_icon="📊")

# ── Load & Prepare Data ──────────────────────────────────────────────
@st.cache_data
def load_data():
    sentiment_df = pd.read_csv('data/sentiment_data.csv')
    trader_df = pd.read_csv('data/trader_data.csv')

    # Sentiment processing
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    sentiment_df['sentiment_binary'] = sentiment_df['classification'].apply(
        lambda x: 'Fear' if 'Fear' in x else 'Greed'
    )

    # Trader processing
    trader_df['Timestamp IST'] = pd.to_datetime(trader_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
    trader_df['trade_date'] = pd.to_datetime(trader_df['Timestamp IST'].dt.date)
    trader_df['Closed PnL'] = pd.to_numeric(trader_df['Closed PnL'], errors='coerce').fillna(0)
    trader_df['Size USD'] = pd.to_numeric(trader_df['Size USD'], errors='coerce').fillna(0)
    trader_df['Fee'] = pd.to_numeric(trader_df['Fee'], errors='coerce').fillna(0)
    trader_df['is_long'] = (trader_df['Side'].str.upper() == 'BUY').astype(int)
    trader_df['is_win'] = (trader_df['Closed PnL'] > 0).astype(int)

    # Merge
    df = trader_df.merge(
        sentiment_df[['date', 'value', 'classification', 'sentiment_binary']],
        left_on='trade_date', right_on='date', how='left'
    )
    df = df[df['sentiment_binary'].notna()].copy()

    # Daily aggregation
    daily = df.groupby(['Account', 'trade_date', 'sentiment_binary']).agg(
        daily_pnl=('Closed PnL', 'sum'),
        num_trades=('Closed PnL', 'count'),
        num_wins=('is_win', 'sum'),
        avg_size_usd=('Size USD', 'mean'),
        num_longs=('is_long', 'sum'),
        sentiment_value=('value', 'first')
    ).reset_index()
    daily['win_rate'] = daily['num_wins'] / daily['num_trades']
    daily['num_shorts'] = daily['num_trades'] - daily['num_longs']
    daily['ls_ratio'] = daily['num_longs'] / daily['num_shorts'].replace(0, np.nan)
    daily['is_profitable'] = (daily['daily_pnl'] > 0).astype(int)

    return df, daily, sentiment_df


df, daily, sentiment_df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("🔧 Filters")

accounts = ['All'] + sorted(daily['Account'].unique().tolist())
selected_account = st.sidebar.selectbox("Trader Account", accounts)

sentiment_filter = st.sidebar.multiselect(
    "Sentiment", ['Fear', 'Greed'], default=['Fear', 'Greed']
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(daily['trade_date'].min(), daily['trade_date'].max()),
    min_value=daily['trade_date'].min(),
    max_value=daily['trade_date'].max()
)

# Apply filters
filtered = daily.copy()
if selected_account != 'All':
    filtered = filtered[filtered['Account'] == selected_account]
filtered = filtered[filtered['sentiment_binary'].isin(sentiment_filter)]
if len(date_range) == 2:
    filtered = filtered[
        (filtered['trade_date'] >= pd.Timestamp(date_range[0])) &
        (filtered['trade_date'] <= pd.Timestamp(date_range[1]))
    ]

# ── Header ───────────────────────────────────────────────────────────
st.title("📊 Trader Performance vs Market Sentiment")
st.markdown("**Hyperliquid traders × Bitcoin Fear/Greed Index**")

# ── KPIs ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trades", f"{filtered['num_trades'].sum():,.0f}")
col2.metric("Unique Traders", f"{filtered['Account'].nunique()}")
col3.metric("Total PnL", f"${filtered['daily_pnl'].sum():,.0f}")
col4.metric("Avg Win Rate", f"{filtered['win_rate'].mean():.1%}")
col5.metric("Avg Daily PnL", f"${filtered['daily_pnl'].mean():,.0f}")

st.divider()

# ── Tab Layout ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🔄 Behavior", "👥 Segments", "🧠 Insights"])

with tab1:
    st.subheader("Performance: Fear vs Greed")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(filtered, x='sentiment_binary', y='daily_pnl',
                     color='sentiment_binary',
                     color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                     title='Daily PnL Distribution',
                     labels={'daily_pnl': 'Daily PnL ($)', 'sentiment_binary': 'Sentiment'})
        # Clip for visibility
        fig.update_yaxes(range=[-2000, 5000])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.box(filtered, x='sentiment_binary', y='win_rate',
                      color='sentiment_binary',
                      color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                      title='Win Rate Distribution',
                      labels={'win_rate': 'Win Rate', 'sentiment_binary': 'Sentiment'})
        st.plotly_chart(fig2, use_container_width=True)

    # Daily PnL over time
    daily_agg = filtered.groupby(['trade_date', 'sentiment_binary']).agg(
        total_pnl=('daily_pnl', 'sum')).reset_index()
    fig3 = px.scatter(daily_agg, x='trade_date', y='total_pnl',
                      color='sentiment_binary',
                      color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                      title='Daily Aggregate PnL Over Time',
                      labels={'total_pnl': 'Total PnL ($)', 'trade_date': 'Date'})
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Behavioral Shifts by Sentiment")

    metrics_agg = filtered.groupby('sentiment_binary').agg(
        avg_trades=('num_trades', 'mean'),
        avg_size=('avg_size_usd', 'mean'),
        avg_ls_ratio=('ls_ratio', 'mean'),
        avg_win_rate=('win_rate', 'mean')
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(metrics_agg, x='sentiment_binary', y='avg_trades',
                     color='sentiment_binary',
                     color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                     title='Avg Trades per Day',
                     text_auto='.1f')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(metrics_agg, x='sentiment_binary', y='avg_size',
                     color='sentiment_binary',
                     color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                     title='Avg Position Size (USD)',
                     text_auto='$.0f')
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(metrics_agg, x='sentiment_binary', y='avg_ls_ratio',
                     color='sentiment_binary',
                     color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                     title='Long/Short Ratio',
                     text_auto='.3f')
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        # Trade direction pie
        if selected_account != 'All':
            trade_df = df[df['Account'] == selected_account]
        else:
            trade_df = df
        trade_df = trade_df[trade_df['sentiment_binary'].isin(sentiment_filter)]
        side_counts = trade_df['Side'].value_counts().reset_index()
        side_counts.columns = ['Side', 'Count']
        fig = px.pie(side_counts, values='Count', names='Side',
                     title='Trade Direction Split',
                     color_discrete_sequence=['#3498db', '#e67e22'])
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Trader Segments")

    seg_type = st.radio("Segment by:", ["Position Size (Leverage Proxy)", "Trade Frequency", "Consistency"],
                        horizontal=True)

    # Build segments
    if seg_type == "Position Size (Leverage Proxy)":
        trader_metric = df.groupby('Account')['Size USD'].mean().reset_index()
        trader_metric.columns = ['Account', 'metric']
        med = trader_metric['metric'].median()
        trader_metric['segment'] = np.where(trader_metric['metric'] >= med, 'High Leverage', 'Low Leverage')
    elif seg_type == "Trade Frequency":
        trader_metric = df.groupby('Account').size().reset_index(name='metric')
        med = trader_metric['metric'].median()
        trader_metric['segment'] = np.where(trader_metric['metric'] >= med, 'Frequent', 'Infrequent')
    else:
        trader_wr = daily.groupby('Account').agg(
            wr=('win_rate', 'mean'), days=('trade_date', 'nunique')).reset_index()
        trader_wr['segment'] = np.where(
            (trader_wr['wr'] > 0.5) & (trader_wr['days'] >= 3),
            'Consistent Winner', 'Inconsistent')
        trader_metric = trader_wr[['Account', 'segment']]

    seg_daily = filtered.merge(trader_metric[['Account', 'segment']], on='Account', how='left')

    seg_perf = seg_daily.groupby(['segment', 'sentiment_binary']).agg(
        avg_pnl=('daily_pnl', 'mean'),
        avg_win_rate=('win_rate', 'mean'),
        pct_profitable=('is_profitable', 'mean'),
        count=('daily_pnl', 'count')
    ).reset_index()

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.bar(seg_perf, x='segment', y='avg_pnl', color='sentiment_binary',
                     barmode='group', color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                     title='Mean Daily PnL', text_auto='$.0f')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(seg_perf, x='segment', y='avg_win_rate', color='sentiment_binary',
                     barmode='group', color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                     title='Win Rate', text_auto='.1%')
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.bar(seg_perf, x='segment', y='pct_profitable', color='sentiment_binary',
                     barmode='group', color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                     title='% Profitable Days', text_auto='.1%')
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(seg_perf.round(4), use_container_width=True)

with tab4:
    st.subheader("Key Insights & Strategy Recommendations")

    st.markdown("""
    ### Insight 1: Fear days → higher mean PnL but greater variance
    Traders earn **\$5,185/day** on Fear vs **\$3,973** on Greed (mean), but **median PnL is lower**
    (\$123 vs \$243). This indicates a few large wins during Fear skew the average — most traders
    actually perform worse on Fear days.

    ### Insight 2: Counterintuitive long bias persists during Fear
    Position sizes are **27% larger** and long/short ratio is **20% higher** during Fear.
    Traders lean into longs despite bearish sentiment, amplifying drawdown risk.

    ### Insight 3: Consistent winners ignore sentiment
    Top-performing traders maintain **~90% profitable days** regardless of sentiment regime,
    while inconsistent traders see profitability drop from 62% to 59% during Fear.

    ---
    ### Strategy Recommendations

    **1. Sentiment-Adaptive Position Sizing:**
    During Fear days, reduce position sizes by 20–30%, especially for high-leverage traders.

    **2. Reduce Long Bias During Fear:**
    Shift toward neutral or slightly short positioning during Fear periods.
    The persistent long bias (L/S ratio 2.24 on Fear vs 1.78 on Greed) is a
    contrarian signal exposing traders to unnecessary downside.
    """)

    # Sentiment timeline
    st.subheader("Sentiment Timeline")
    sent_timeline = sentiment_df[sentiment_df['date'] >= '2023-05-01'].copy()
    fig = px.scatter(sent_timeline, x='date', y='value',
                     color='sentiment_binary',
                     color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                     title='Bitcoin Fear & Greed Index Over Time',
                     labels={'value': 'Fear/Greed Value', 'date': 'Date'})
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Neutral (50)")
    st.plotly_chart(fig, use_container_width=True)
