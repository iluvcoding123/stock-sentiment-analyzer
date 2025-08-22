# src/preprocess/text_cleaner.py
import re

URL_RE   = re.compile(r"https?://\S+|www\.\S+")
MENTION  = re.compile(r"@\w+")
HASHTAG  = re.compile(r"#(\w+)")
EMOJI    = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+",
    flags=re.UNICODE,
)
NONALNUM = re.compile(r"[^a-z0-9\s]+")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = URL_RE.sub(" ", s)
    s = MENTION.sub(" ", s)
    # keep hashtag word but drop '#'
    s = HASHTAG.sub(r"\1", s)
    s = EMOJI.sub(" ", s)
    s = NONALNUM.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s