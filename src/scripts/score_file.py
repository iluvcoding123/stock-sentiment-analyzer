# src/scripts/score_file.py
import argparse
import pandas as pd
from pathlib import Path


from src.models.finbert import load_finbert_pipeline

def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}. Create it or check the path.")
    if p.suffix.lower() in [".csv"]:
        return pd.read_csv(p)
    if p.suffix.lower() in [".jsonl", ".json"]:
        # jsonl (one json per line) or a json array
        try:
            return pd.read_json(p, lines=True)
        except ValueError:
            return pd.read_json(p)
    raise ValueError(f"Unsupported file type: {p.suffix}")

def _write_any(df: pd.DataFrame, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    elif p.suffix.lower() in [".parquet"]:
        try:
            df.to_parquet(p, index=False)
        except Exception as e:
            raise RuntimeError(
                "Writing Parquet requires a Parquet engine (pyarrow or fastparquet). "
                "Install one with `pip install pyarrow` or output to .csv instead."
            ) from e
    elif p.suffix.lower() in [".jsonl"]:
        df.to_json(p, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported output type: {p.suffix}")

def main():
    ap = argparse.ArgumentParser(
        description="Score text sentiment with FinBERT.",
        epilog="""Examples:
    python -m src.scripts.score_file --in data/raw/sample.csv --out data/processed/scored.csv --text-col text
    python -m src.scripts.score_file --in data/raw/sample.jsonl --out data/processed/scored.parquet --text-col body --device mps --batch-size 32
      """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Input file (.csv, .json, .jsonl)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output file (.csv, .parquet, .jsonl)")
    ap.add_argument("--text-col", dest="text_col", default="text", help="Column containing text (default: text)")
    ap.add_argument("--device", dest="device", default="auto", choices=["auto", "cpu", "mps"], help="Compute device")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=16, help="Batch size for inference")
    args = ap.parse_args()

    df = _read_any(args.in_path)
    if args.text_col not in df.columns:
        raise KeyError(f"Column '{args.text_col}' not found in: {args.in_path}")

    pipe = load_finbert_pipeline(device_preference=args.device, batch_size=args.batch_size)

    texts = df[args.text_col].fillna("").astype(str).tolist()
    # FinBERT returns list of dicts with 'label' and 'score' per input when top_k=None
    preds = pipe(texts, truncation=True, top_k=None)

    # Convert predictions to columns: label + score for each class
    # Expect POSITIVE / NEGATIVE / NEUTRAL
    def to_row(ps):
        row = {"label": None, "score_positive": 0.0, "score_neutral": 0.0, "score_negative": 0.0}
        # choose argmax as label
        if isinstance(ps, list) and ps:
            best = max(ps, key=lambda x: x["score"])
            row["label"] = best["label"].upper()
            for d in ps:
                key = d["label"].lower()
                if key in ["positive", "neutral", "negative"]:
                    row[f"score_{key}"] = float(d["score"])
        return row

    scored_rows = [to_row(p) for p in preds]
    scored_df = pd.DataFrame(scored_rows)

    out = pd.concat([df.reset_index(drop=True), scored_df], axis=1)
    _write_any(out, args.out_path)

if __name__ == "__main__":
    main()