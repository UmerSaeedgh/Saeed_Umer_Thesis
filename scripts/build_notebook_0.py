"""
Builds distribution_analysis_v1/00_corpus_and_sampling.ipynb -- the dataset-construction notebook
that was previously missing from the analysis folder.

This documents the steps that happen BEFORE 01_foundations_and_baseline.ipynb picks up: corpus
inspection, measurement of annotator disagreement, the ambiguity threshold, the author-relevance
typing rule, and the draw of the balanced 300-text sample. Those steps existed only as
scripts/classify_ambiguity_pools.py and as notebooks/01_april_corpus_exploration.ipynb outside the
analysis folder, so the sampling decisions that every later result depends on were not visible
alongside the results themselves.

Run: python build_notebook_0.py
Then: jupyter nbconvert --to notebook --execute --inplace <output path>
"""
import nbformat as nbf
from pathlib import Path

OUT_PATH = Path(r"C:\Users\umers\Desktop\thesis\thesis final\distribution_analysis_v1\00_corpus_and_sampling.ipynb")

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Part 0: Corpus, Ambiguity Threshold and Sampling

This notebook documents how the 300-text experimental sample was constructed, and is the
precondition for everything in Parts 1 and 2. It covers the corpus, how disagreement was measured,
the threshold that separates unambiguous from ambiguous texts, the rule that splits ambiguous
texts into author-relevant and author-independent, and the draw of the balanced sample.

It exists because these decisions were previously recorded only in `scripts/classify_ambiguity_pools.py`
and in an exploration notebook outside the analysis folder. Every result in Parts 1 and 2 is
conditional on the choices made here, so they belong next to the results rather than upstream of
them.

**Labeling convention:** this notebook is **[DESIGN]** throughout -- it constructs the dataset and
tests no hypothesis.""")

code("""import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
sys.path.insert(0, str(Path.cwd().parent / 'scripts'))
from plot_style import setup_style, COLORS, CATEGORY_LABELS
from emotion_vectors import EMOTIONS as CATEGORY_EMOTIONS
setup_style()

BASE = Path.home() / 'Desktop' / 'thesis' / 'thesis final'
CORPUS = Path.home() / 'Desktop' / 'thesis' / 'data' / 'corpus'

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 140)

# The rule's parameters, imported from the script that applies them so this notebook cannot
# drift out of sync with the pipeline that actually produced the sample.
import classify_ambiguity_pools as cap
THRESHOLD = cap.THRESHOLD
SEED = cap.SEED
OFF_LABELS = cap.OFF_LABELS
EXAMPLE_IDS = cap.EXAMPLE_IDS
print(f"THRESHOLD = {THRESHOLD}   SEED = {SEED}")
print(f"Excluded labels: {sorted(OFF_LABELS)}")
print(f"Reserved prompt-example texts (never sampled): {sorted(EXAMPLE_IDS)}")
""")

# ===========================================================================
md("""## 1. The corpus

The source is crowd-enVent (Troiano et al., 2023): short self-authored descriptions of emotional
events. Its value for this study is the **dual annotation layer** -- each text carries the emotion
its author intended, and independent labels from five readers who never saw the author's label.
That makes disagreement measurable at two levels: author against readers, and readers among
themselves.""")

code("""gen = pd.read_csv(CORPUS / 'crowd-enVent_generation.tsv', sep='\\t')
val = pd.read_csv(CORPUS / 'crowd-enVent_validation.tsv', sep='\\t')

texts = gen[['text_id', 'generated_text', 'emotion']].rename(
    columns={'generated_text': 'text', 'emotion': 'author_emotion'})
readers = val[['text_id', 'emotion']].rename(columns={'emotion': 'reader_emotion'})

print(f"generation rows : {len(gen):,}  ({texts['text_id'].nunique():,} unique texts)")
print(f"validation rows : {len(val):,}")
rpt = readers.groupby('text_id').size()
print(f"readers per text: {rpt.value_counts().to_dict()}")
print()
print("The validation layer covers a 1,200-text subset at exactly 5 readers each, which is the")
print("pool this study samples from. Five is simply the number of independent judgments this")
print("corpus provides per text; disagreement is observable with as few as two. What five buys is")
print("resolution -- it distinguishes unanimity, a 4-1 split, a 3-2 split and a tie, which is the")
print("granularity the ambiguity threshold in Section 3 relies on.")
""")

code("""ac = pd.read_csv(BASE / 'ambiguity_classification_1200.csv')
ac['text_id'] = ac['text_id'].astype(int)
print(f"Annotated candidate pool: {len(ac):,} texts")
print(f"Columns: {list(ac.columns)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
ac['author_emotion'].value_counts().plot.bar(ax=axes[0], color=COLORS['human'])
axes[0].set_title('Author-intended emotion', pad=12); axes[0].set_ylabel('# texts')
axes[0].tick_params(axis='x', rotation=45)
ac['reader_majority'].value_counts().plot.bar(ax=axes[1], color=COLORS['llm'])
axes[1].set_title('Reader-majority emotion', pad=12); axes[1].set_ylabel('# texts')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.show()

wc = ac['text'].astype(str).str.split().str.len()
print(f"Text length: median {wc.median():.0f} words, IQR [{wc.quantile(.25):.0f}, {wc.quantile(.75):.0f}], max {wc.max()}")
""")

md("""> **Observation.** The author-intended and reader-majority distributions differ in shape, which
> is the first indication that the two layers do not simply agree. Note also that the corpus
> contains labels outside the eleven-emotion set used in this study (`boredom`, `no-emotion`);
> their treatment is decided in Section 3.""")

# ===========================================================================
md("""## 2. Measuring disagreement

Two quantities are computed per text, and they capture different things.

**Author--reader agreement** (`author_matches_readers`): does the selected reader-majority label
coincide with the emotion the author reports having intended? A mismatch records that the two
labels differ. It does **not** establish that the text failed to communicate: readers may be
responding to a genuinely available alternative reading, which is the phenomenon this thesis
studies rather than an error to be explained away.

**Top-label share** (stored as `majority_share`): what fraction of the five readers chose the
most common emotion? The column name says *majority*, but the quantity is the share of the **top
label**, which is not always a majority. It takes the values 0.2, 0.4, 0.6, 0.8 and 1.0, and only
the last three exceed half the readers:

- **0.2** -- all five readers chose *different* emotions. There is no majority and no unique top
  label at all.
- **0.4** -- two readers share the most common answer. That may be a unique plurality (2-1-1-1) or
  a tie at the top (2-2-1).
- **0.6 and above** -- a strict majority, and necessarily a unique winner.

The measure is coarse, which directly constrains where a threshold can sit: only at one of those
five points.""")

code("""print("Author-reader agreement:")
print(ac['author_matches_readers'].value_counts().to_string())
print(f"  -> readers recover the author's label for {ac['author_matches_readers'].mean()*100:.1f}% of texts")
print()
print("Top-label share (stored as `majority_share`), 5 readers so only 5 values are attainable:")
share_tab = ac['majority_share'].value_counts().sort_index()
display(pd.DataFrame({'n_texts': share_tab, 'pct': (share_tab / len(ac) * 100).round(1)}))

fig, ax = plt.subplots(figsize=(8.5, 5))
x = share_tab.index.astype(float)
bars = ax.bar(x, share_tab.values, width=0.13,
              color=['#C44E52' if v < THRESHOLD else '#4C72B0' for v in x])
ax.axvline(THRESHOLD - 0.1, color='black', linestyle='--', lw=1.8)
ax.text(THRESHOLD - 0.09, share_tab.max()*0.92, f'  THRESHOLD = {THRESHOLD}', fontsize=10, fontweight='bold')
ax.set_xlabel('Top-label share among the 5 readers'); ax.set_ylabel('# texts')
ax.set_title('Reader agreement is coarse: only five values are attainable\\nBlue = at or above threshold, red = below', pad=14)
for b, v in zip(bars, share_tab.values):
    ax.text(b.get_x()+b.get_width()/2, v+6, str(v), ha='center', fontsize=9)
plt.tight_layout(); plt.show()
""")

md("""### 2.1 Are these fields trustworthy? Reconstructing them from the raw votes

The two quantities above are read from `ambiguity_classification_1200.csv` rather than derived
here. That file is an input, so this section rebuilds both fields directly from the raw corpus
votes and checks whether they agree. It also pins down the tie-breaking rule, which the file
itself does not document.

**Scope.** This reconstruction covers the **full candidate pool: 1,200 texts, 5 readers each,
6,000 votes.** It is not the 300-text sample. The sample's own vote counts are reported separately
in Section 3.2, where the relevant figure is 1,500 votes across 300 texts; the two should not be
conflated.""")

code("""sample_ids_early = set(pd.read_csv(BASE / 'sampled_texts.csv')['text_id'].astype(int))

recon = []
for tid, g in val[val['text_id'].isin(set(ac['text_id']))].groupby('text_id'):
    vc = g['emotion'].value_counts()
    top = vc.max()
    winners = sorted(vc[vc == top].index)
    recon.append({'text_id': tid, 'recon_share': top / len(g),
                  'n_top': len(winners), 'recon_winners': winners})
R = pd.DataFrame(recon).merge(
    ac[['text_id', 'reader_majority', 'majority_share', 'author_emotion', 'author_matches_readers']],
    on='text_id', how='inner', validate='one_to_one')
print(f"Reconstruction scope: {len(val[val['text_id'].isin(set(ac['text_id']))]):,} votes "
      f"across {len(R):,} candidate texts (5 readers each)")

share_ok = bool(np.allclose(R['recon_share'], R['majority_share']))
print(f"majority_share reconstructs exactly from raw votes : {share_ok}")

uniq = R[R['n_top'] == 1]
maj_ok = bool((uniq['recon_winners'].str[0] == uniq['reader_majority']).all())
print(f"reader_majority matches where the top label is unique: {maj_ok}  (n={len(uniq)})")

tied = R[R['n_top'] > 1]
tie_ok = bool(tied.apply(lambda r: r['reader_majority'] in r['recon_winners'], axis=1).all())
print(f"where the top is TIED, reader_majority is one of the tied winners: {tie_ok}  (n={len(tied)})")
print()
print(f"-> {len(tied)} of {len(R)} texts have a tie at the top, so a tie-break was applied.")
print("   The file does not record which rule; what is verifiable is that the stored label is")
print("   always one of the tied winners.")
print()

# Which definition does author_matches_readers use?
R['in_top_set'] = R.apply(lambda r: r['author_emotion'] in r['recon_winners'], axis=1)
R['eq_chosen'] = R['author_emotion'] == R['reader_majority']
print("author_matches_readers is defined as ...")
print(f"   author label within the tied TOP SET      : {(R['in_top_set'] == R['author_matches_readers']).mean()*100:.1f}% agreement")
print(f"   author label == the single chosen majority: {(R['eq_chosen'] == R['author_matches_readers']).mean()*100:.1f}% agreement")
assert bool((R['eq_chosen'] == R['author_matches_readers']).all()), \
    'author_matches_readers is not equality against the chosen reader_majority'
print()
print("So the field compares the author's label against the single TIE-BROKEN label, not against")
print("the set of tied winners. On a tied text whose author label is among the winners but is not")
print("the one selected, it records False.")
""")

md("""> **Does the tie-break affect the outcome?** It cannot affect the *threshold* decision, but
> it **can** affect *eligibility*, and the two need separating.
>
> **It cannot affect the threshold.** The unambiguous test requires a top-label share of at least
> 0.8, which with five readers means 4-1 or 5-0 --- both of which have a unique top label by
> construction. A tie can never arise at or above the threshold, so no text's position relative to
> it depends on how a tie was resolved.
>
> **It can affect Step 0.** Step 0 excludes a text when the *selected* reader-majority label is
> `boredom` or `no-emotion`. If a tied top set contains both an allowed and an excluded emotion,
> then which label the tie-break happened to pick decides whether that text is eligible at all.
> A tie of `anger, anger, boredom, boredom, sadness` is retained if `anger` is selected and
> dropped if `boredom` is. This is a real dependency and is measured below rather than reasoned
> about.""")

code("""# Which texts, if any, have eligibility that hinges on the tie-break?
ALLOWED = set(CATEGORY_EMOTIONS)
crit = R[(R['n_top'] > 1)
         & R['recon_winners'].apply(lambda w: bool(set(w) & ALLOWED) and bool(set(w) & OFF_LABELS))
         & R['author_emotion'].isin(ALLOWED)].copy()
crit['retained'] = ~crit['reader_majority'].isin(OFF_LABELS)
crit['in_sample'] = crit['text_id'].isin(sample_ids_early)

print(f"Candidate texts whose ELIGIBILITY hinges on the tie-break: {len(crit)}")
print(f"   retained, because an allowed emotion was selected : {int(crit['retained'].sum())}")
print(f"   excluded, because an off-label emotion was selected: {int((~crit['retained']).sum())}")
print()
print(f"How many of these reached the final 300-text sample?   {int(crit['in_sample'].sum())}")
print()
print(f"Top-label shares among them: {sorted(crit['majority_share'].unique())}")
print("All sit at or below 0.4, so none could ever have met the 0.8 unambiguous threshold -- had")
print("any been retained and sampled, it could only have entered an AMBIGUOUS pool.")
print()
if len(crit):
    display(crit[['text_id', 'recon_winners', 'reader_majority', 'author_emotion',
                  'majority_share', 'retained', 'in_sample']])
assert not crit['in_sample'].any(), (
    'a text with tie-break-dependent eligibility entered the sample; its effect must be documented')
print("No text with tie-break-dependent eligibility entered the analysed sample, so the 300 texts")
print("used in Parts 1 and 2 are unaffected. The candidate POOL was affected, by at most the count")
print("above, which is recorded here rather than claimed to be immaterial.")
""")

md("""> **Key observation [DESIGN].** Reader agreement is not concentrated at the top. A top-label
> share of 0.6 -- three readers of five -- is the single most common outcome, and a further group
> sits at 0.4 or below, where no emotion reaches half the readers at all. Disagreement is the
> normal case in this corpus, not an edge case, which is the empirical basis for the
> distributional representation used throughout Parts 1 and 2.""")

# ===========================================================================
md("""## 3. The classification rule

Three decisions are applied in a fixed order. All three are parameters of
`scripts/classify_ambiguity_pools.py` rather than manual judgments, so the sample is reproducible.

**Step 0 -- exclude off-label texts.** Texts are dropped when **either the author's own label or
the selected reader-majority label** is `boredom` or `no-emotion`. These fall outside the
eleven-emotion set, and forcing them into the taxonomy produces persona-to-persona flips that are
a labelling artefact rather than a real interpretive shift.

**This filter operates on those two labels only.** It does not require that every individual
reader vote falls inside the eleven-emotion set, and retained texts still contain some
off-inventory votes. Section 3.2 quantifies how many survive into the final sample, because a
check named "no off-label text" could otherwise be read as a stronger guarantee than it is.

**Step 1 -- threshold first.** A text is **Unambiguous** if its reader agreement is strong
(`majority_share >= THRESHOLD`) *and* the readers recover the author's own label
(`author_matches_readers`). This is decided purely from the original corpus's reader data.

**Step 2 -- flip test, among the remainder only.** Of the texts not classified as unambiguous,
those where the two candidate personas produce different judged emotions (`judge_a != judge_b`)
are **Author-Relevant Ambiguous**; the rest are **Author-Independent Ambiguous**.

The ordering matters and is deliberate. A persona flip does **not** override strong,
author-matching reader agreement: a text readers already agree on stays unambiguous even if a
persona pair can be found that moves a judge. Reversing the order would let the flip test recruit
high-agreement texts into the author-relevant group.

**What this ordering does and does not protect.** It protects the *unambiguous control category*
from being depleted by the flip test, which is what makes it usable as a control. It does **not**
remove the dependence of the two *ambiguous* categories on the pilot flip judgments: the split
between author-relevant and author-independent is made entirely by that test. H3 therefore remains
a transfer question rather than an independent discovery, for the reason set out in Section 4.1.""")

code("""def classify(full, threshold=THRESHOLD):
    off = full['reader_majority'].isin(OFF_LABELS) | full['author_emotion'].isin(OFF_LABELS)
    pool = pd.Series(None, index=full.index, dtype=object)
    not_off = ~off
    is_unamb = not_off & (full['majority_share'] >= threshold) & (full['author_matches_readers'] == True)
    pool.loc[is_unamb] = 'Unambiguous'
    remaining = not_off & ~is_unamb
    flip_true = remaining & (full['flip'] == True)
    pool.loc[flip_true] = 'Author-Relevant Ambiguous'
    pool.loc[remaining & ~flip_true] = 'Author-Independent Ambiguous'
    return pool

ac['pool'] = classify(ac)
n_off = ac['pool'].isna().sum()
print(f"Step 0: {n_off} texts dropped as off-label ({sorted(OFF_LABELS)})")
print()
print("Candidate pool sizes after classification:")
counts = ac['pool'].value_counts().reindex(CATEGORY_LABELS)
display(pd.DataFrame({'n_candidates': counts, 'pct_of_eligible': (counts/counts.sum()*100).round(1)}))
print(f"Total eligible: {int(counts.sum())} of {len(ac)}")
print()
# The 1,200-text file also carries a `new_classification` column. It is HISTORICAL: it predates
# both the off-label exclusion and the threshold-first ordering, so it does not agree with the
# rule above. Quantified here rather than left as a silent discrepancy for a reader to trip over.
_cmp = pd.DataFrame({'rule': ac['pool'].fillna('OFF'),
                     'stored': ac['new_classification'].fillna('OFF')})
_dis = _cmp[_cmp['rule'] != _cmp['stored']]
print(f"Rows where the current rule differs from the file's stored `new_classification` column: {len(_dis)}")
display(_dis.groupby(['stored', 'rule']).size().to_frame('n'))
print("That column is a HISTORICAL artefact predating the off-label exclusion and the")
print("threshold-first ordering. It is NOT used by this pipeline: the authoritative category for")
print("every sampled text is `sampled_texts.csv.classification`, verified against the rule below.")
""")

md("""### 3.2 What the off-label filter does *not* remove

Step 0 screens two labels per text: the author's own, and the selected reader-majority label. It
does not screen the individual reader votes. A text can therefore pass the filter while still
containing reader votes outside the eleven-emotion set, and some do.

This is quantified here because the sample check later in this notebook is named "no off-label
text", which could otherwise be read as guaranteeing something stronger.""")

code("""EMO11 = set(CATEGORY_EMOTIONS)
sample_ids = set(pd.read_csv(BASE / 'sampled_texts.csv')['text_id'].astype(int))
sv = val[val['text_id'].isin(sample_ids)]
off_votes = sv[~sv['emotion'].isin(EMO11)]

print(f"Reader votes across the 300 sampled texts : {len(sv):,}")
print(f"Votes OUTSIDE the eleven-emotion set      : {len(off_votes)}"
      f"  ({len(off_votes)/len(sv)*100:.2f}%), across {off_votes['text_id'].nunique()} texts")
print(f"Breakdown: {off_votes['emotion'].value_counts().to_dict()}")
print()
print("These texts are legitimately in the sample: neither their author label nor their selected")
print("reader-majority label is off-inventory. What carries through is that a handful of the")
print("underlying reader votes are.")
print()
print("Consequence for Parts 1 and 2: such votes are EXCLUDED when the original-corpus vectors are")
print("built, so those vectors rest on fewer than five votes for the affected texts. That is why")
print("the vector construction tracks n per vector rather than assuming five throughout.")
retained_all_five = sv.groupby('text_id').apply(
    lambda g: bool(g['emotion'].isin(EMO11).all()), include_groups=False)
print(f"\\nSampled texts retaining all five in-inventory votes: {int(retained_all_five.sum())} of 300")
""")

md("""### 3.1 How sensitive is the split to the threshold?

`THRESHOLD = 0.8` means four of five readers must agree *and* match the author. Because
`majority_share` takes only five values, the only other defensible settings are 0.6 and 1.0. The
consequences of each are shown below, so the choice is visible rather than buried.""")

code("""rows = []
for t in (0.6, 0.8, 1.0):
    p = classify(ac, threshold=t)
    c = p.value_counts().reindex(CATEGORY_LABELS).fillna(0).astype(int)
    rows.append({'threshold': t, **{lab: c[lab] for lab in CATEGORY_LABELS},
                 'enough_for_100_each': bool((c >= 100).all())})
sens = pd.DataFrame(rows)
display(sens)

fig, ax = plt.subplots(figsize=(9.5, 5))
x = np.arange(len(sens)); w = 0.26
for i, lab in enumerate(CATEGORY_LABELS):
    ax.bar(x + (i-1)*w, sens[lab], width=w, label=lab, color=COLORS[lab])
ax.axhline(100, color='black', linestyle='--', lw=1.5)
ax.text(len(sens)-0.45, 108, 'n = 100 needed per pool', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f'threshold = {t}' for t in sens['threshold']])
ax.set_ylabel('# candidate texts'); ax.legend(fontsize=8)
ax.set_title('Effect of the ambiguity threshold on candidate pool sizes', pad=14)
plt.tight_layout(); plt.show()

print("At 0.6 the unambiguous pool absorbs texts where only three of five readers agreed, which")
print("weakens the control category the design depends on. At 1.0 it admits only unanimous texts.")
print(f"Every setting shown leaves at least 100 candidates per pool, so {THRESHOLD} is not forced by")
print("availability -- it is chosen as the stricter of the two defensible middle options.")
""")

# ===========================================================================
md("""## 4. Arriving at the balanced sample

The sample was not produced by a single fresh draw, and describing it that way would misstate what
is reproducible. `classify_ambiguity_pools.py` performs a **reconciliation** against an existing
sample:

1. **Retain** the current `sampled_texts.csv`.
2. **Reclassify** every retained text under the rule in Section 3. A text may move pool, or become
   ineligible entirely if it is off-label.
3. **Trim** any pool now holding more than 100 texts back to 100, dropping the excess
   reproducibly with the fixed seed.
4. **Replace** shortfalls by drawing from eligible candidates not already used, excluding the
   three texts reserved as one-shot and few-shot prompt examples so that no sampled text can
   appear inside its own prompt.
5. **Assert** the result is 300 texts, 100 per pool, with no duplicates and no off-label
   reader-majority label, then write it back.

**What this means for reproducibility.** The rule in Section 3 is fully reproducible and is
verified against all 300 texts below. The *specific membership* of the sample is not recoverable
from the seed alone: it depends on the prior sample's contents and on the candidate ordering at
the time each replacement was drawn. `sampled_texts.csv` is therefore the authoritative record of
which texts were used, not something this notebook regenerates.""")

code("""sampled = pd.read_csv(BASE / 'sampled_texts.csv')
sampled['text_id'] = sampled['text_id'].astype(int)

print(f"Final sample: {len(sampled)} texts")
final_counts = sampled['classification'].value_counts().reindex(CATEGORY_LABELS)
display(final_counts.to_frame('n_texts'))

checks = []
checks.append(('exactly 300 texts', len(sampled) == 300))
checks.append(('balanced at 100 per category', bool((final_counts == 100).all())))
checks.append(('no reserved example text sampled', not set(sampled['text_id']) & EXAMPLE_IDS))
checks.append(('no duplicate text_id', sampled['text_id'].duplicated().sum() == 0))
merged_chk = sampled.merge(ac[['text_id', 'pool']], on='text_id', how='left', validate='one_to_one')
checks.append(('every sampled text has an eligible pool', merged_chk['pool'].notna().all()))
checks.append(('sample classification matches the rule',
               bool((merged_chk['classification'] == merged_chk['pool']).all())))
off_in_sample = merged_chk['text_id'].isin(
    ac.loc[ac['pool'].isna(), 'text_id']).sum()
checks.append(('no off-label text in the sample', off_in_sample == 0))

# Candidate-file integrity, checked before anything is concluded from it.
assert ac['text_id'].is_unique, 'duplicate text_id in the candidate file'
assert sampled['text_id'].is_unique, 'duplicate text_id in the sample'

for name, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print()
print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed")

# These are assertions, not advisory prints: a FAIL must stop the notebook rather than leave a
# red line in the output for a reader to notice or miss.
failed = [name for name, ok in checks if not ok]
assert not failed, f"sample verification failed: {failed}"
print("All sample checks are assertions -- execution stops here on any failure.")
""")

code("""fig, ax = plt.subplots(figsize=(9.5, 5))
x = np.arange(len(CATEGORY_LABELS)); w = 0.38
cand = ac['pool'].value_counts().reindex(CATEGORY_LABELS)
ax.bar(x - w/2, cand.values, width=w, label='Eligible candidates', color='#B0B0B0')
ax.bar(x + w/2, final_counts.values, width=w, label='Sampled', color=[COLORS[l] for l in CATEGORY_LABELS])
for i, (c, s) in enumerate(zip(cand.values, final_counts.values)):
    ax.text(i - w/2, c + 8, str(c), ha='center', fontsize=9)
    ax.text(i + w/2, s + 8, str(s), ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(CATEGORY_LABELS, rotation=12, ha='right')
ax.set_ylabel('# texts'); ax.legend(fontsize=9)
ax.set_title('Candidate pools and the balanced 300-text sample drawn from them', pad=14)
plt.tight_layout(); plt.show()

print("Sampling fractions differ by pool because the pools differ in size. This is intended: the")
print("design prioritises equal n per category over proportional representation of the corpus, so")
print("that category comparisons are not confounded with unequal sample size.")
""")

md("""### 4.1 What the persona pairs look like

Each sampled text carries two personas and a rationale for the pairing. The judged emotions under
each persona, and whether they differ (`flip`), are the inputs to Step 2 above.""")

code("""cols = ['text_id', 'classification', 'author_emotion', 'reader_majority', 'majority_share',
        'persona_a', 'judge_a', 'persona_b', 'judge_b', 'flip']
ex = sampled[sampled['classification'] == CATEGORY_LABELS[2]].head(3)[cols]
for _, r in ex.iterrows():
    print(f"text {r['text_id']}  [{r['classification']}]")
    print(f"  author said: {r['author_emotion']}   readers said: {r['reader_majority']} ({r['majority_share']})")
    print(f"  persona A: {r['persona_a']}  -> judged {r['judge_a']}")
    print(f"  persona B: {r['persona_b']}  -> judged {r['judge_b']}")
    print(f"  flip = {r['flip']}")
    print()

print("Flip rate within the final sample, by category:")
display((sampled.groupby('classification')['flip'].mean() * 100).reindex(CATEGORY_LABELS).round(1).to_frame('flip_rate_pct'))
""")

md("""> **A property that constrains H3, recorded here rather than in the results.** The
> author-relevant category is defined by `flip = True`, so its flip rate is 100% by construction
> and the unambiguous category's is whatever the corpus happens to give. H3 later asks whether
> framing effects are larger for author-relevant texts -- that is, whether a grouping built partly
> from pilot flip judgments predicts framing effects in **newly collected** human and model data.
> That is a transfer question, and a legitimate one, but it is not an independent discovery that
> author-relevance is the operative principle. Part 2 words the H3 verdict accordingly.""")

# ===========================================================================
md("""## 5. Summary of what this notebook fixes

| Parameter | Value | Where it is set |
|---|---|---|
| Ambiguity threshold | `majority_share >= 0.8` **and** author-matching | `classify_ambiguity_pools.THRESHOLD` |
| Category order of operations | threshold first, then flip test | `classify_ambiguity_pools.classify()` |
| Excluded labels | `boredom`, `no-emotion` | `classify_ambiguity_pools.OFF_LABELS` |
| Reserved prompt examples | 3 texts, never sampled | `classify_ambiguity_pools.EXAMPLE_IDS` |
| Random seed | 42 | `classify_ambiguity_pools.SEED` |
| Final sample | 300 texts, 100 per category | `sampled_texts.csv` |

Parts 1 and 2 take `sampled_texts.csv` as given. Every result there is conditional on the choices
recorded above, which is why they are documented here rather than only in the script that applies
them.""")

nb['cells'] = cells
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
OUT_PATH.write_text(nbf.writes(nb), encoding='utf-8')
print(f"Wrote {OUT_PATH} ({len(cells)} cells)")
