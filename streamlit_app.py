# streamlit_app.py
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load env (optional for future extensions)
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
ticker = st.selectbox("Ticker", tickers, index=0)

view = df[df["ticker"] == ticker].copy()

# Layout
col_left, col_right = st.columns([2,1], gap="large")

with col_left:
    st.subheader("Sentiment over time")
    if len(view):
        st.line_chart(view.set_index("fetched_at")["sent_score"])
    else:
        st.info("No rows for this ticker yet.")

with col_right:
    st.subheader("Label counts")
    st.bar_chart(view["sent_label"].value_counts())

st.markdown("### Latest Headlines")
show_cols = ["fetched_at","source","sent_label","sent_score","text","url"]
if "url" not in view.columns:  # older runs may not have it
    show_cols.remove("url")
st.dataframe(view[show_cols].tail(20), use_container_width=True)

# Small footer
st.caption("Data source: NewsAPI | Model: FinBERT (yiyanghkust/finbert-tone)")