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

def ensure_partition_dir(ts: dt.datetime) -> str:
    part = ts.strftime("date=%Y-%m-%d")
    path = os.path.join(DATA_DIR, part)
    os.makedirs(path, exist_ok=True)
    return path

def mock_ingest_aapl() -> pd.DataFrame:
    # Replace this with real Twitter/NewsAPI later
    now = dt.datetime.now(dt.UTC)
    samples = [
        {"id": str(uuid.uuid4()), "source": "mock_news", "ticker": "AAPL",
         "text": "Apple shares rise as analysts boost iPhone revenue estimates.",
         "created_at": now.isoformat(), "fetched_at": now.isoformat(),
         "author": "wire", "url": "", "lang": "en"},
        {"id": str(uuid.uuid4()), "source": "mock_tweet", "ticker": "AAPL",
         "text": "Not impressed by the new iPhone event tbh.", 
         "created_at": now.isoformat(), "fetched_at": now.isoformat(),
         "author": "user123", "url": "", "lang": "en"},
    ]
    return pd.DataFrame(samples)

def ingest() -> pd.DataFrame:
    """Try real news ingest; fallback to mock if unavailable or failing."""
    if HAS_NEWSAPI and _fetch_news is not None:
        try:
            df = _fetch_news()
            # Normalize to DataFrame
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            # Ensure required schema
            now_iso = dt.datetime.now(dt.UTC).isoformat()
            required_defaults = {
                "id": lambda i: str(uuid.uuid4()),
                "source": "newsapi",
                "ticker": "AAPL",  # default if your fetcher doesn't set it
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
            # Minimal cleanup
            df["text"] = df["text"].fillna("")
            return df[list(required_defaults.keys())]
        except Exception as e:
            print(f"NewsAPI ingest failed: {e}; falling back to mock.")
    return mock_ingest_aapl()

def append_parquet(df: pd.DataFrame, ts: dt.datetime):
    path = ensure_partition_dir(ts)
    filename = os.path.join(path, f"part-{int(ts.timestamp())}.parquet")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filename)

def main():
    scorer = FinBertScorer()
    print("Starting loop. Ctrl+C to stop.")
    while True:
        ts = dt.datetime.now(dt.UTC)
        raw = ingest()
        if raw.empty:
            time.sleep(60); continue

        # score
        scores = scorer.score_texts(raw["text"].fillna("").tolist())
        out = pd.concat([raw.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

        # write
        append_parquet(out, ts)
        print(f"Wrote {len(out)} rows at {ts.isoformat()}")
        time.sleep(60)

if __name__ == "__main__":
    main()