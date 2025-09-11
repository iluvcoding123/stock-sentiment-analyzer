# stock-sentiment-analyzer

End-to-end ML pipeline for real-time stock sentiment analysis from tweets, news, and forums.

## Phase 1 – Baseline (VADER)
	•	Implemented data ingestion of sample tweets/news.
	•	Preprocessed text (basic cleaning, tokenization).
	•	Applied VADER sentiment scoring (neg, neu, pos, compound).
	•	Stored results in a DataFrame for exploration.
	•	Found that VADER struggles with financial text (accuracy ~42% on synthetic labels).

## Phase 1.5 – Domain Model (FinBERT)
	•	Integrated FinBERT, a transformer model tuned for financial text.
	•	Created finbert.py helper to load the model and run batch scoring.
	•	Evaluated FinBERT on the same dataset and confirmed stronger alignment with finance sentiment.
	•	Established FinBERT as the default model moving forward.

## Next Steps – Phase 2 (Pipeline + Dashboard)
	•	Implement a processing pipeline: clean text → detect tickers → sentiment scoring.
	•	Store results in a structured format (Parquet or SQLite).
	•	Build a Streamlit dashboard with:
	•	Line chart of rolling sentiment by ticker.
	•	Bar chart of sentiment counts.
	•	Table of recent posts with filters.
	•	Toggle between VADER and FinBERT.

## Quickstart
    1) Create .env with `NEWSAPI_KEY=...`
    2) Start the stream loop:
       python -m src.scripts.stream_loop
    3) Launch the dashboard:
       streamlit run streamlit_app.py