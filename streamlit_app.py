# streamlit_app.py
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import plotly.express as px

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")
st.title("📈 Live Stock Sentiment (News + FinBERT)")

SNAP = "data/processed/sentiment/latest.parquet"

if not os.path.exists(SNAP):
    st.warning("No snapshot found yet. Start the stream loop first.")
    st.stop()

# Read snapshot
df = pd.read_parquet(SNAP)

# Guard rails
required = {"ticker","text","sent_label","sent_score","fetched_at","source"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing columns in snapshot: {missing}")
    st.stop()

# Clean + order
df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce")
df = df.dropna(subset=["fetched_at"]).sort_values("fetched_at")

# Controls
tickers = sorted(df["ticker"].dropna().unique().tolist() or ["AAPL"])
selected_tickers = st.multiselect("Select ticker(s):", tickers, default=tickers[:1])
view = df[df["ticker"].isin(selected_tickers)].copy()

if view.empty:
    st.warning("No sentiment data available for selected tickers.")
    st.stop()

# Header metrics
col1, col2, col3 = st.columns(3)
col1.metric("Avg Sentiment", f"{view['sent_score'].mean():.2f}")
col2.metric("Positive %", f"{(view['sent_label'].eq('positive').mean()*100):.1f}%")
col3.metric("Latest Headlines", len(view))

st.caption(f"Last updated: {view['fetched_at'].max():%b %d, %H:%M UTC}")

# Aggregate for smoother time visualization
hourly = (
    view.groupby([pd.Grouper(key="fetched_at", freq="1H"), "ticker"])["sent_score"]
    .mean()
    .reset_index()
)

# Layout
tab1, tab2, tab3 = st.tabs(["📊 Sentiment Trends", "🧩 Label Distribution", "📰 Headlines"])

with tab1:
    st.subheader("Average Sentiment Over Time")
    fig = px.line(hourly, x="fetched_at", y="sent_score", color="ticker",
                  title="Sentiment Trends by Ticker", markers=True)
    fig.update_layout(xaxis_title="Time", yaxis_title="Sentiment Score (0-1)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Sentiment Label Counts")
    label_counts = view.groupby(["ticker", "sent_label"]).size().reset_index(name="count")
    fig2 = px.bar(label_counts, x="ticker", y="count", color="sent_label", barmode="group",
                  title="Sentiment Label Distribution")
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Latest Headlines")
    show_cols = ["fetched_at", "source", "sent_label", "sent_score", "text", "url"]
    if "url" not in view.columns:
        show_cols.remove("url")
    st.dataframe(view[show_cols].sort_values("fetched_at", ascending=False).head(25), use_container_width=True)

# Quick insights
try:
    avg_by_ticker = view.groupby("ticker")["sent_score"].mean()
    most_pos = avg_by_ticker.idxmax()
    most_neg = avg_by_ticker.idxmin()
    st.markdown(f"📈 **{most_pos}** currently has the highest average sentiment.")
    st.markdown(f"⚠️ **{most_neg}** has the lowest sentiment among selected tickers.")
except Exception:
    pass

# Footer
st.caption("Data source: NewsAPI | Model: FinBERT (yiyanghkust/finbert-tone)")