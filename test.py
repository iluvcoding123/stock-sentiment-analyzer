from dotenv import load_dotenv
load_dotenv()
from src.data_ingest.newsapi import fetch_news
for t in ["AAPL", "TSLA", "AMZN", "NVDA"]:
    df = fetch_news(t)
    print(t, len(df))