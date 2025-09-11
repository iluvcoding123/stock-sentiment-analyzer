# src/models/finbert_batch.py
from typing import List
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

FINBERT_MODEL_ID = "yiyanghkust/finbert-tone"  # POSITIVE / NEGATIVE / NEUTRAL

class FinBertScorer:
    def __init__(self, model_id: str = FINBERT_MODEL_ID, device: int = -1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.pipe = TextClassificationPipeline(
            model=self.model, tokenizer=self.tokenizer, framework="pt", device=device, return_all_scores=True
        )

    def score_texts(self, texts: List[str]) -> pd.DataFrame:
        if not texts:
            return pd.DataFrame(columns=["sent_label","sent_score","sent_pos","sent_neu","sent_neg"])
        outputs = self.pipe(texts, truncation=True, max_length=256)
        # outputs is list of list[dict(label, score)]
        rows = []
        for scores in outputs:
            m = {d["label"].lower(): d["score"] for d in scores}  # {'positive':..., 'negative':..., 'neutral':...}
            # pick label with max score
            top = max(scores, key=lambda d: d["score"])
            rows.append({
                "sent_label": top["label"].lower(),
                "sent_score": float(top["score"]),
                "sent_pos": float(m.get("positive", 0.0)),
                "sent_neu": float(m.get("neutral", 0.0)),
                "sent_neg": float(m.get("negative", 0.0)),
            })
        return pd.DataFrame(rows)