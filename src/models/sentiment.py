# src/models/sentiment.py
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# lazy-ensure the lexicon exists
def _ensure_vader():
    try:
        import nltk
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        import nltk
        nltk.download("vader_lexicon")

def score_vader(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    _ensure_vader()
    sia = SentimentIntensityAnalyzer()

    # compute scores
    scores = df[text_col].fillna("").apply(sia.polarity_scores)
    scores_df = pd.DataFrame(list(scores))
    # merge with original
    out = df.reset_index(drop=True).join(scores_df[["neg", "neu", "pos", "compound"]])
    return out