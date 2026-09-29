"""Exploratory direct A-minus-B comparison in author-independent human texts."""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2]  # thesis final/ (patched from original sibling-of-script layout)
audit = pd.read_csv(root / "results/comparison_audit.csv")
rows = audit[(audit.source == "Human") & (audit.category == "author_independent")]
paired = rows.pivot(index="text_id", columns="side", values="tv")
assert paired.shape == (100, 2) and set(paired.columns) == {"a", "b"}
diff = 100 * (paired.a - paired.b).to_numpy()
rng = np.random.default_rng(42)
n = 20000
bootstrap = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
lo, hi = np.quantile(bootstrap, [.025, .975])
signflips = rng.choice([-1, 1], size=(n, len(diff)))
null = (diff * signflips).mean(axis=1)
p = (1 + np.count_nonzero(np.abs(null) >= abs(diff.mean()))) / (n + 1)
result = pd.DataFrame([{"category": "author_independent", "n_texts": len(diff),
                        "mean_raw_a_minus_b_pp": diff.mean(), "bootstrap_ci_lo_pp": lo,
                        "bootstrap_ci_hi_pp": hi, "paired_signflip_p_exploratory": p,
                        "texts_a_greater": int((diff > 0).sum()),
                        "texts_b_greater": int((diff < 0).sum()),
                        "texts_equal": int((diff == 0).sum())}])
result.to_csv(root / "results/author_independent_persona_side_contrast_verified.csv", index=False)
print(result.to_string(index=False))
