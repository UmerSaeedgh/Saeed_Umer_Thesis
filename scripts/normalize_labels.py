"""
Adds normalized <Model>_label_norm columns to the 3 master experiment CSVs, fixing mechanical
formatting artifacts in the raw model output columns without touching the raw columns themselves:

  - markdown bold wrapping: '**joy**' -> 'joy'
  - numbered-list answers: some models answer with the numbered list's digit or "N. word" instead
    of the plain word (e.g. Qwen3 answering '8' or '7. relief'). Mapped back to the word using the
    same 11-label order the numbered prompts used (see build_prompts.py's LABELS).
  - casing: 'Anger' -> 'anger'

Genuinely off-list answers (invented emotions like "boredom"/"frustration", refusals, echoed
system prompts, non-English text, etc.) are intentionally left as their cleaned-but-non-matching
string rather than force-mapped to a label -- that's real model failure, not a formatting bug, and
matches the existing is_valid_label policy in run_classification.py of keeping unparseable output
for analysis rather than discarding or guessing it.
"""
import re
import pandas as pd
from pathlib import Path

BASE = Path.home() / 'Desktop' / 'thesis' / 'thesis final'

LABELS = ["anger", "disgust", "fear", "guilt", "joy", "pride", "relief", "sadness", "shame", "surprise", "trust"]

MODEL_COLS = ['ChatGPT_label', 'Claude_label', 'Gemini_label', 'Qwen3_label', 'Gemma3_label', 'Ministral_label', 'Llama31_label']

FILES = [
    BASE / 'experiment_results_unambiguous.csv',
    BASE / 'experiment_results_author_independent.csv',
    BASE / 'experiment_results_author_relevant.csv',
]

NUM_PREFIX_RE = re.compile(r'^\s*(\d{1,2})\s*[.\):]\s*(.+?)\s*$')
BARE_NUM_RE = re.compile(r'^\s*(\d{1,2})\s*[.\):]?\s*$')


def normalize_label(raw):
    if pd.isna(raw):
        return raw
    s = str(raw).strip()
    s = s.strip('*').strip()

    m = BARE_NUM_RE.match(s)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= len(LABELS):
            return LABELS[idx - 1]
        return s.lower()

    m = NUM_PREFIX_RE.match(s)
    if m:
        s = m.group(2).strip('*').strip()

    return s.lower()


total_changed = 0
for path in FILES:
    df = pd.read_csv(path)
    for col in MODEL_COLS:
        if col not in df.columns:
            continue
        norm_col = col + '_norm'
        df[norm_col] = df[col].apply(normalize_label)
        changed = (df[norm_col].astype(str) != df[col].astype(str)).sum()
        total_changed += changed
        print(f"{path.name}: {col} -> {norm_col}, {changed} values changed by normalization")
    df.to_csv(path, index=False)
    print(f"  saved {path.name}\n")

print(f"Total normalized values across all files/models: {total_changed}")
