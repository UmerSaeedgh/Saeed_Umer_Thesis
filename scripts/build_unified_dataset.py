"""
Concatenates the three pool-specific LLM classification result files (kept separate during data
collection so run_classification.py could track per-pool progress independently) into a single
unified dataset, experiment_results_all.csv, covering all 300 texts x 24 rows = 7,200
classifications across all 7 models. The existing 'classification' column already carries the pool
label (Unambiguous / Author-Independent Ambiguous / Author-Relevant Ambiguous), so no new column is
needed to tell the pools apart in the unified file.
"""
import pandas as pd
from pathlib import Path

BASE = Path.home() / 'Desktop' / 'thesis' / 'thesis final'

POOL_FILES = [
    BASE / 'experiment_results_unambiguous.csv',
    BASE / 'experiment_results_author_independent.csv',
    BASE / 'experiment_results_author_relevant.csv',
]

dfs = [pd.read_csv(p) for p in POOL_FILES]
for p, df in zip(POOL_FILES, dfs):
    print(f"{p.name}: {len(df)} rows, {df['text_id'].nunique()} texts")

unified = pd.concat(dfs, ignore_index=True)

assert len(unified) == sum(len(d) for d in dfs), "row count mismatch after concat"
assert unified['text_id'].nunique() == 300, f"expected 300 unique texts, got {unified['text_id'].nunique()}"
dupe_check = unified.duplicated(subset=['text_id', 'framing_condition', 'prompt_variant', 'label_format'])
assert not dupe_check.any(), f"{dupe_check.sum()} duplicate (text_id, framing_condition, prompt_variant, label_format) rows"

model_cols = []
for m in ['ChatGPT', 'Claude', 'Gemini', 'Qwen3', 'Gemma3', 'Ministral', 'Llama31']:
    model_cols += [f'{m}_label', f'{m}_reason']
empties = sum(unified[c].isna().sum() + (unified[c] == '').sum() for c in model_cols)
assert empties == 0, f"{empties} empty model cells in unified dataset"

out_path = BASE / 'experiment_results_all.csv'
unified.to_csv(out_path, index=False)
print(f"\nWrote {len(unified)} rows ({unified['text_id'].nunique()} texts x 24 rows) to {out_path.name}")
print(f"Pool breakdown:")
print(unified.groupby('classification')['text_id'].nunique())
