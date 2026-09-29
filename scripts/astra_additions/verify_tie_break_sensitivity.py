"""Does the selected-label H1 verdict (0 of 6 significant) depend on the tie-break convention?

Added 2026-09-26 in response to an external review. The selected-label method picks the top
emotion by np.argmax over vote shares, which silently breaks ties alphabetically (first emotion
in E wins). results/tie_rule_sensitivity.csv already shows the raw change/no-change *decision*
flips for 125 of 600 human comparisons under reverse-alphabetical tie-breaking -- but nobody had
checked whether the actual H1 *significance test* verdict (how many of the six category x persona
tests reject the null after Holm correction) is itself sensitive to this choice, as opposed to just
the individual-comparison decisions.

This script reruns the exact same permutation-test machinery as
build_readable_notebooks.py's test_h1_h2() (same seeds, same N_RANDOM, same Holm correction),
but substitutes reverse-alphabetical tie-breaking for BOTH the observed statistic and the
permutation null -- consistency between the two is essential: naively comparing a reverse-tie
observed value against an alphabetical-tie null would spuriously manufacture significance. Both
sides of the comparison use the same convention here.

Requires distribution_analysis_v1/01_majority_labels_then_vectors.ipynb to already exist (reads
`study.h`/`study.l`, which that notebook builds); run via `jupyter nbconvert --to script` and
`runpy` to get a `study` object, exactly as the canonical notebook does.
"""
from pathlib import Path
from itertools import combinations, product
import subprocess
import runpy
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2]  # thesis final/ (patched from original sibling-of-script layout)
nb_path = root / "distribution_analysis_v1" / "01_majority_labels_then_vectors.ipynb"

E = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'pride',
     'relief', 'sadness', 'shame', 'surprise', 'trust']
CATS = ['unambiguous', 'author_independent', 'author_relevant']
N_RANDOM = 10000

# --- Build `study` by running the notebook's own setup code (up to the `study = Study()` line),
#     via a scratch script, so this reuses the exact same data-loading and comparison logic. ---
import tempfile
script_text = subprocess.run(
    ["jupyter", "nbconvert", "--to", "script", str(nb_path), "--stdout"],
    check=True, capture_output=True, text=True,
).stdout
cut = script_text.index("study = Study()") + len("study = Study()")
script_text = script_text[:cut].replace("%matplotlib inline", "#%matplotlib inline")
script_text = script_text.replace(
    "ROOT = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent",
    f"ROOT = Path({str(root)!r})",
)
script_text = script_text.replace(
    "RESULTS = ROOT / 'results'",
    "import tempfile as _tempfile; RESULTS = Path(_tempfile.mkdtemp())  # this run only checks significance; nothing here is saved to the real results/",
)
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
    f.write(script_text)
    tmp_path = f.name
ns = runpy.run_path(tmp_path)
study = ns["study"]
h, l = study.h, study.l


def rev_argmax(p):
    return p.shape[-1] - 1 - np.argmax(p[..., ::-1], axis=-1)


null_cache = {}


def chance_shuffles_reverse(source, index):
    key = (source, index)
    if key in null_cache:
        return null_cache[key]
    r = (h if source == 'Human' else l).iloc[index]
    a, b = r.neutral_votes, r.persona_votes
    if source == 'Human':
        pool = np.concatenate([a, b])
        n = len(a)
        total = np.bincount(pool, minlength=len(E))
        counts = np.array([np.bincount(pool[list(c)], minlength=len(E))
                            for c in combinations(range(len(pool)), n)])
        pa = counts / n
        pb = (total - counts) / len(b)
    else:
        swaps = np.array(list(product([False, True], repeat=len(a))))
        av = np.where(swaps, b, a)
        bv = np.where(swaps, a, b)
        pa = np.eye(len(E))[av].mean(axis=1)
        pb = np.eye(len(E))[bv].mean(axis=1)
    result = {'flip_reverse': (rev_argmax(pa) != rev_argmax(pb)).astype(float)}
    null_cache[key] = result
    return result


def holm_correct(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((n - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


rows = []
for source, d_ in [('Human', h), ('LLM', l)]:
    for side in ['a', 'b']:
        for cat in (CATS if source == 'Human' else ['all']):
            sub = d_[(d_.side == side) & ((d_.category == cat) if cat != 'all' else True)]
            rng = np.random.default_rng(4200 + len(rows))
            null = np.zeros(N_RANDOM)
            expect = 0
            for index in sub.index:
                values = chance_shuffles_reverse(source, index)['flip_reverse']
                null += values[rng.integers(0, len(values), N_RANDOM)]
                expect += values.mean()
            null /= len(sub)
            expect /= len(sub)
            obs = sub['flip_reverse'].mean()
            p = (1 + np.count_nonzero(null >= obs - 1e-12)) / (N_RANDOM + 1)
            rows.append(dict(hypothesis='H1' if source == 'Human' else 'H2', source=source,
                              side=side, category=cat, n_cells=len(sub), observed_pct=100 * obs,
                              conditional_null_pct=100 * expect, p_raw=p))

out = pd.DataFrame(rows)
out['p_holm'] = np.nan
for hh in ['H1', 'H2']:
    use = out.hypothesis == hh
    out.loc[use, 'p_holm'] = holm_correct(out.loc[use, 'p_raw'].values)
out['reject_05'] = out.p_holm < .05

out.to_csv(root / "results/tie_break_h1_significance_verified.csv", index=False)
print(out.to_string(index=False))
print()
print("H1 significant count under REVERSE-alphabetical tie-break:", int(out[out.hypothesis == 'H1'].reject_05.sum()), "of 6")
print("(compare to 0 of 6 under the alphabetical convention used throughout the rest of this thesis)")
