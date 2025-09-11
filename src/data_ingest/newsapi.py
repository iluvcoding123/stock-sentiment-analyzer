import os, datetime as dt, requests, pandas as pd

NEWS_API = "https://newsapi.org/v2/everything"
API_KEY = os.getenv("NEWSAPI_KEY")

def fetch_news(ticker: str = "AAPL", page_size: int = 20) -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError("NEWSAPI_KEY not set")
    now = dt.datetime.now(dt.UTC)
    q = f'{ticker} OR "Apple Inc"'
    params = {
        "q": q, "language": "en", "sortBy": "publishedAt",
        "pageSize": page_size, "apiKey": API_KEY,
    }
    r = requests.get(NEWS_API, params=params, timeout=20)
    r.raise_for_status()
    arts = r.json().get("articles", [])
    rows = []
    for a in arts:
        rows.append({
            "id": a.get("url"),  # stable id
            "source": "newsapi",
            "ticker": ticker,
            "text": f'{a.get("title","")} {a.get("description","")}'.strip(),
            "created_at": a.get("publishedAt", now.isoformat()),
            "fetched_at": now.isoformat(),
            "author": a.get("source",{}).get("name",""),
            "url": a.get("url",""),
            "lang": "en",
        })
    return pd.DataFrame(rows)