# scripts/stream_loop.py
import os, time, uuid, datetime as dt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT / ".env", override=True)

from src.models.finbert_batch import FinBertScorer
try:
    from src.data_ingest.newsapi import fetch_news as _fetch_news
    HAS_NEWSAPI = True
except Exception:
    _fetch_news = None
    HAS_NEWSAPI = False

DATA_DIR = "data/processed/sentiment"
TICKERS = ["AAPL", "TSLA", "AMZN", "NVDA"]  # Add more tickers here
INTERVAL = 3600  # 1 hour between cycles (adjust as needed)


def ensure_partition_dir(ts: dt.datetime) -> str:
    part = ts.strftime("date=%Y-%m-%d")
    path = os.path.join(DATA_DIR, part)
    os.makedirs(path, exist_ok=True)
    return path

def ingest(ticker: str) -> pd.DataFrame:
    """Fetch news for a given ticker. Skip if no data returned."""
    if not HAS_NEWSAPI or _fetch_news is None:
        raise RuntimeError("NewsAPI not available or import failed.")

    try:
        df = _fetch_news(ticker)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        if df.empty:
            print(f"No headlines returned for {ticker}.")
            return pd.DataFrame()

        now_iso = dt.datetime.now(dt.UTC).isoformat()
        required_defaults = {
            "id": lambda i: str(uuid.uuid4()),
            "source": "newsapi",
            "ticker": ticker,
            "text": "",
            "created_at": now_iso,
            "fetched_at": now_iso,
            "author": "",
            "url": "",
            "lang": "en",
        }

        for col, default in required_defaults.items():
            if col not in df.columns:
                df[col] = [default(i) if callable(default) else default for i in range(len(df))]

        df["text"] = df["text"].fillna("")
        return df[list(required_defaults.keys())]

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def append_parquet(df: pd.DataFrame, ts: dt.datetime):
    path = ensure_partition_dir(ts)
    filename = os.path.join(path, f"part-{int(ts.timestamp())}.parquet")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filename)


def write_snapshot(df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, os.path.join(DATA_DIR, "latest.parquet"))


def main():
    scorer = FinBertScorer()
    print("Starting multi-ticker loop. Ctrl+C to stop.")
    while True:
        ts = dt.datetime.now(dt.UTC)
        all_results = []

        for ticker in TICKERS:
            print(f"Fetching headlines for {ticker}...")
            df = ingest(ticker)

            if df.empty:
                print(f"No data for {ticker}")
                continue

            # Score sentiment
            scores = scorer.score_texts(df["text"].fillna("").tolist())
            out = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
            all_results.append(out)

        if not all_results:
            print("No data fetched for any ticker; sleeping.")
            time.sleep(INTERVAL)
            continue

        combined = pd.concat(all_results, ignore_index=True)
        append_parquet(combined, ts)
        write_snapshot(combined)

        print(f"Wrote {len(combined)} rows across {len(TICKERS)} tickers at {ts.isoformat()}")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()