# Guided thesis analysis — start here

Open `distribution_analysis_v1/01_majority_labels_then_vectors.ipynb` first.

It completes the old one-label analysis and H1–H4 before explaining its losses,
introducing vectors, repeating the questions and tests, and comparing the answers.
Every plot has an explanation and every event count states its denominator.

Then read `02_understanding_distribution_changes.ipynb` for deeper analyses and
real examples. `00_corpus_and_sampling.ipynb` documents the corpus and sample.

## What is included

- Three executed notebooks, with embedded figures and outputs.
- Original data, preserved without modifying votes or sample membership.
- The source calculations in `scripts/guided_analysis.py`.
- All plotted tables as CSV and 25 scientific figures as PNG in `results/`.
- Execution reports, verification checks and input hashes.

## The new analysis is more than a narrative reorder

The one-label method now has its own complete hypothesis analysis. The new
comparisons use the same observations and common valid models in both conditions.
They include explicit tie rules, broad and strict proportion-only changes,
appearance/disappearance counts, distinct-text incidence, matched human/model
event profiles, count-standardised entropy, persona-specific directional cosines,
and a generic-author control using a three-way model intersection.

These are retrospective analyses. The old method's H4 statistics are categorical
proxies; one selected label cannot recover distributional magnitude or direction.
Inference retains the participant-dependence and model-exchangeability limitations.

## Why some values differ from earlier notebooks

The primary model comparisons now consistently retain the same valid models on
both sides. Earlier all-valid-per-condition descriptive percentages are a different
analysis and must not be substituted into these tables. For example, same observed
emotions with changed shares occur in 543/3,600 common-model comparisons here.

Human conditional null expectations are now calculated by exhaustive enumeration
within each cell. Aggregate p-values still use 10,000 simulations. This removes
Monte Carlo noise from each cell's expected value and changes some final decimals.
H3 now uses permutation tests of mean differences for both representations;
earlier rank-based tests and bootstrap-tail tests target different statistics.
Directional cosines retain persona sides separately and match model subsets;
the earlier cosine of averaged delta vectors is a different estimand.

The results in this package are authoritative for these new notebooks. Earlier
thesis prose/PDFs need reconciliation before quoting these newly defined results.
No thesis PDF is presented as a final submission artifact in this package.

## Run the analysis

Python 3.12 was used. Install the packages in `requirements.txt`, then use Jupyter
to restart and run each notebook from top to bottom. The setup works from the
project root or the notebook directory. Paths are relative to this project.

Alternatively, from the project root:

```bash
python scripts/execute_verified_notebooks.py
python scripts/verify_guided_analysis.py
```

The supplied execution used a fresh Python process with IPython for each notebook.
The analysis code ran in source order. Only `%matplotlib inline` was replaced by
explicit figure capture, because a networked Jupyter kernel was unavailable.

## Source of truth for future edits (updated 23 Sep 2026)

**All calculation code now lives inside the notebooks.** No notebook imports anything from
`scripts/`; each notebook runs on its own, and every function appears just before the first
result that uses it. The explanations were also rewritten in plain language.

- `scripts/build_readable_notebooks.py` is the canonical source for notebooks 01 and 02, and it
  patches the text and setup cell of notebook 00. To change a notebook: edit this script, run it,
  then run `scripts/execute_verified_notebooks.py` and `scripts/verify_guided_analysis.py`.
- `scripts/create_guided_notebooks.py.superseded` is the old generator. **Do not run it**: it would
  bring back the old notebooks that import `guided_analysis.py`.
- `scripts/guided_analysis.py` is kept only as the independent reference used by
  `verify_guided_analysis.py`. The notebook code is a line-by-line readable copy of it: same
  formulas, seeds and table names. After the change, all 44 tables in `results/` were
  byte-identical to the previous run and 30/30 verification checks passed.
- If you change a calculation in the notebooks, make the same change in `guided_analysis.py`,
  or the verification script will no longer match.
- Backup of the previous state: `backup_pre_readable_notebooks_20260923/`.

Never re-run sampling or model collection just to view results.

The source helper `build_prompts.py` is included only because the sampling-rule
module imports its fixed example IDs. It is not part of the analysis run.
