# src/models/finbert.py
from typing import List, Optional
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

FINBERT_MODEL_ID = "yiyanghkust/finbert-tone"  # POSITIVE / NEGATIVE / NEUTRAL

def _device(device_preference: str = "auto") -> str:
    if device_preference == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device_preference == "cpu":
        return "cpu"
    return "mps" if torch.backends.mps.is_available() else "cpu"

def load_finbert_pipeline(model_id: str = FINBERT_MODEL_ID, device_preference: str = "auto", batch_size: int = 16):
    dev = _device(device_preference)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(dev)
    pipe = TextClassificationPipeline(
        model=model, tokenizer=tok,
        device=0 if dev == "mps" else -1,  # -1 = CPU
        return_all_scores=True, truncation=True, max_length=256, batch_size=batch_size,
    )
    return pipe

def _scores_to_row(scores: List[dict]) -> dict:
    row = {s["label"].upper(): float(s["score"]) for s in scores}
    for k in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
        row.setdefault(k, 0.0)
    pred = max(row, key=row.get)
    row["pred_label"] = pred
    row["pred_score"] = row[pred]
    return row

def score_finbert_df(df: pd.DataFrame, text_col: str = "text", pipe: Optional[TextClassificationPipeline] = None, batch_size: int = 16) -> pd.DataFrame:
    assert text_col in df.columns, f"Missing column: {text_col}"
    if pipe is None:
        pipe = load_finbert_pipeline(batch_size=batch_size)
    texts = df[text_col].fillna("").astype(str).tolist()
    out_rows = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        outputs = pipe(batch)
        for o in outputs:
            out_rows.append(_scores_to_row(o))
    scores = pd.DataFrame(out_rows)
    return df.reset_index(drop=True).join(scores[["POSITIVE","NEUTRAL","NEGATIVE","pred_label","pred_score"]])

