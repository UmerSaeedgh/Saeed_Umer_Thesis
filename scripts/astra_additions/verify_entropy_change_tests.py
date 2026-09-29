"""Canonical six-test entropy-change table for Table 5.15 / Figure 5.9 (Section 5.14, res:entropy).

Added 2026-09-26 in response to an external review that found the printed p-values in
Table 5.15 were not reproducible from the packaged notebooks or any script in this
project -- the six-test entropy hypothesis-test calculation existed only in the
superseded scripts/run_analysis.py (entropy_change_summary()) and was never carried
into the current canonical notebook pipeline. This script formalises that exact,
already-documented method against the current comparison_audit.csv so the table has a
real, current, reproducible source.

Method (unchanged from the original run_analysis.py, verified line-for-line against it):
for each (source, category) cell, average each text's entropy_change to exactly one
value per text (avoiding the pseudoreplication a per-comparison test would introduce --
each text otherwise contributes 2 rows for humans or 12 for the model ensemble), then
run a one-sample Wilcoxon signed-rank test of that per-text array against zero.
Benjamini-Hochberg FDR correction is applied jointly across all six tests (three
categories times two sources), not separately within each source.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

root = Path(__file__).resolve().parents[2]  # thesis final/ (patched from original sibling-of-script layout)

CATEGORY_ORDER = ['unambiguous', 'author_independent', 'author_relevant']
POOL_LABEL = {'unambiguous': 'Unambiguous', 'author_independent': 'Author-Independent',
              'author_relevant': 'Author-Relevant'}


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_min = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        val = min(pvals[idx] * n / (rank + 1), 1.0)
        running_min = min(running_min, val)
        adj[idx] = running_min
    return adj


def entropy_change_summary(df, label):
    df = df.groupby(['category', 'text_id'], as_index=False)['entropy_change'].mean()
    rows = []
    for c in CATEGORY_ORDER:
        sub = df[df['category'] == c]['entropy_change'].dropna().values
        if len(sub) >= 3:
            try:
                _, p = stats.wilcoxon(sub, np.zeros_like(sub))
            except ValueError:
                p = 1.0
        else:
            p = 1.0
        rows.append({'source': label, 'category': POOL_LABEL[c], 'n': len(sub),
                     'mean_entropy_change': sub.mean(), 'wilcoxon_p_raw': p})
    return pd.DataFrame(rows)


audit = pd.read_csv(root / "results/comparison_audit.csv")
human = audit[audit['source'] == 'Human']
llm = audit[audit['source'] == 'LLM']

ec_df = pd.concat([entropy_change_summary(human, 'Human'), entropy_change_summary(llm, 'Model')],
                   ignore_index=True)
ec_df['p_bh_fdr'] = bh_fdr(ec_df['wilcoxon_p_raw'].values)

out = root / "results/entropy_change_by_category_verified.csv"
ec_df.to_csv(out, index=False)
print(ec_df.to_string(index=False))
print("saved", out)
