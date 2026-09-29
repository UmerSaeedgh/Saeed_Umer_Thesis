"""Canonical source for the H4-magnitude paired Wilcoxon test quoted in result.tex (Section
res:h4magnitude): "paired Wilcoxon signed-rank test over 300 text-level pairs gives a mean paired
difference of 34.86pp ... p = ..., with matched-pairs rank-biserial r = ...".

Added 2026-09-26 in response to an external review that found this exact p-value and r had no
canonical source anywhere in the project -- results/vector_H4_magnitude.csv records the mean
difference and its bootstrap CI, but not the Wilcoxon statistic itself, so the printed p-value
could only be reproduced approximately, and depended on which scipy zero_method setting was used
(the exact digits differ slightly between 'wilcox', 'pratt' and 'zsplit' handling of the 5 exactly-
tied pairs). This script fixes a specific, documented choice (scipy's default, zero_method=
'wilcox', which excludes exact ties from the ranking) so the number has one reproducible source.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

root = Path(__file__).resolve().parents[2]  # thesis final/ (patched from original sibling-of-script layout)

audit = pd.read_csv(root / "results/comparison_audit.csv")
human = audit[audit['source'] == 'Human'].groupby('text_id')['tv'].mean()
llm = audit[audit['source'] == 'LLM'].groupby('text_id')['tv'].mean()
paired = pd.DataFrame({'human': human, 'llm': llm}).dropna()
diff = (paired['human'] - paired['llm']).to_numpy()

stat, p = stats.wilcoxon(paired['human'], paired['llm'], zero_method='wilcox')

nz = diff[diff != 0]
ranks = stats.rankdata(np.abs(nz))
pos_rank_sum = ranks[nz > 0].sum()
neg_rank_sum = ranks[nz < 0].sum()
r_rb = (pos_rank_sum - neg_rank_sum) / (len(nz) * (len(nz) + 1) / 2)

out = pd.DataFrame([{
    'n_texts': len(paired),
    'n_positive': int((diff > 0).sum()),
    'n_negative': int((diff < 0).sum()),
    'n_tied': int((diff == 0).sum()),
    'wilcoxon_statistic': stat,
    'wilcoxon_p_value': p,
    'zero_method': 'wilcox',
    'rank_biserial_r': r_rb,
}])
out.to_csv(root / "results/h4_magnitude_wilcoxon_verified.csv", index=False)
print(out.to_string(index=False))
