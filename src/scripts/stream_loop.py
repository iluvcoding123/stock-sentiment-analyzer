# scripts/stream_loop.py
import os, time, uuid, datetime as dt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.models.finbert_batch import FinBertScorer

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
        raw = mock_ingest_aapl()
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