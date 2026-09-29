"""Build the three readable analysis notebooks with ALL calculation code inside them.

This script is the canonical source for notebooks 01 and 02 (it replaces
create_guided_notebooks.py) and patches notebook 00's text in place. The calculation code
below is a readable, line-by-line copy of scripts/guided_analysis.py: same formulas, same
random seeds, same table names. After building, run execute_verified_notebooks.py and compare
results/*.csv with the previous run to confirm every number is unchanged.

Usage (from the project folder):  python scripts/build_readable_notebooks.py
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / 'distribution_analysis_v1'


def md(text):
    return new_markdown_cell(text.strip('\n'))


def code(text):
    return new_code_cell(text.strip('\n'))


# =============================================================================================
# Shared calculation code (placed in the notebooks as code cells)
# =============================================================================================

SETUP = r'''
# ---- Housekeeping: libraries, folders and display helpers. Nothing is calculated here. ----
from pathlib import Path
from itertools import combinations, product
import hashlib
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
%matplotlib inline

# The project folder is the one that contains data/. This works whether Jupyter was started
# in the project folder or inside distribution_analysis_v1/.
ROOT = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'          # every table and graph below is also saved here
RESULTS.mkdir(exist_ok=True)

plt.rcParams.update({'figure.dpi': 115, 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.titlesize': 13,
                     'figure.figsize': (10, 5), 'savefig.bbox': 'tight'})
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 60)
pd.set_option('display.width', 150)


def explain(text):
    """Show a short explanation under a result."""
    display(Markdown(text))


def table(df, name, digits=3):
    """Save a table as results/<name>.csv and show it, rounded."""
    df.to_csv(RESULTS / (name + '.csv'), index=False)
    shown = df.round(digits).copy()
    # p-values keep 4 significant digits, so a very small p never shows as 0.
    for c in df.columns:
        if str(c).startswith('p_') or c in ['p_value', 'p_BH']:
            shown[c] = df[c].map(lambda x: f'{x:.4g}' if pd.notna(x) else '')
    display(shown)


def finish(fig, name):
    """Save a figure as results/<name>.png and show it."""
    fig.tight_layout()
    fig.savefig(RESULTS / (name + '.png'), dpi=170)
    plt.show()
'''

SETTINGS = r'''
# The 11 emotions, always in this alphabetical order. Every vote-share vector uses this order.
E = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'pride',
     'relief', 'sadness', 'shame', 'surprise', 'trust']
EI = {e: i for i, e in enumerate(E)}      # emotion -> position, e.g. EI['joy'] is 4

# The three text groups: short code names and display names.
CATS = ['unambiguous', 'author_independent', 'author_relevant']
CAT_NAMES = ['Unambiguous', 'Author-independent', 'Author-relevant']
CAT_NAME = dict(zip(CATS, CAT_NAMES))
POOL_TO_KEY = {'Unambiguous': 'unambiguous',
               'Author-Independent Ambiguous': 'author_independent',
               'Author-Relevant Ambiguous': 'author_relevant'}
COLORS = {'Human': '#177e89', 'LLM': '#d66b35'}

# How many random shuffles or resamples every test and error bar uses.
N_RANDOM = 10000

# Where each model's answer is stored in experiment_results_all.csv.
MODEL_COLS = {'OpenAI': 'ChatGPT_label_norm', 'Claude': 'Claude_label_norm',
              'Gemini': 'Gemini_label_norm', 'Qwen3': 'Qwen3_label_norm',
              'Gemma3': 'Gemma3_label_norm', 'Ministral': 'Ministral_label_norm',
              'Llama31': 'Llama31_label_norm'}
MODEL_RAW_COLS = {m: col.replace('_label_norm', '_label') for m, col in MODEL_COLS.items()}
'''

BUILDING_BLOCKS = r'''
def vec(votes):
    """Votes -> vote shares over the 11 emotions.
    votes is a list of emotion positions: anger, anger, sadness = [0, 0, 7]
    -> 0.667 anger, 0.333 sadness, 0 for the other nine emotions."""
    return np.bincount(votes, minlength=len(E)).astype(float) / len(votes)


def entropy(v):
    """How spread out a vote-share vector is (in 'nats').
    0 = everyone chose the same emotion; bigger = votes spread over more emotions."""
    v = np.asarray(v)
    safe = np.where(v > 0, v, 1)          # log(1) = 0, so emotions with no votes add nothing
    return -(safe * np.log(safe)).sum(axis=-1)


def cosine(a, b):
    """Direction similarity of two shift vectors: +1 same direction, 0 unrelated, -1 opposite.
    It is undefined (NaN) when either vector is all zeros, i.e. that side did not move."""
    lengths = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide((a * b).sum(axis=-1), lengths,
                     out=np.full_like(lengths, np.nan, dtype=float), where=lengths > 1e-12)


def compare_votes(a, b):
    """Compare the NEUTRAL votes a with the PERSONA votes b for one text.
    Returns every quantity the notebooks use for that one comparison."""
    p, q = vec(a), vec(b)                  # neutral shares, persona shares
    ps, qs = p > 0, q > 0                  # which emotions received at least one vote
    tp, tq = p == p.max(), q == q.max()    # the top emotion(s); more than one means a tie

    tv = np.abs(q - p).sum() / 2           # SHIFT (total variation): share of votes that must move
    gain = qs & ~ps                        # emotions that APPEAR under the persona
    loss = ps & ~qs                        # emotions that DISAPPEAR under the persona
    same_support = np.array_equal(ps, qs)  # exactly the same emotions present on both sides?
    same_top = np.array_equal(tp, tq)      # exactly the same top emotion(s) on both sides?

    # OLD METHOD: the selected label is the top emotion; a tie goes to the alphabetically first.
    w0, w1 = int(np.argmax(p)), int(np.argmax(q))
    # The same, but with ties broken by reverse alphabet (used to show that the tie rule matters).
    wr0 = len(E) - 1 - int(np.argmax(p[::-1]))
    wr1 = len(E) - 1 - int(np.argmax(q[::-1]))

    # The five kinds of change. Every comparison gets exactly one.
    if tv < 1e-12:
        change = 'No distribution change'
    elif same_support:
        change = 'Same emotions; proportions change'
    elif gain.any() and loss.any():
        change = 'Emotions appear AND disappear'
    elif gain.any():
        change = 'Emotions appear only'
    else:
        change = 'Emotions disappear only'

    moved = tv > 1e-12
    return dict(p=p, q=q, delta=q - p, neutral_votes=a, persona_votes=b,
                neutral_n=len(a), persona_n=len(b), winner0=w0, winner1=w1,
                flip=float(w0 != w1),                  # old method: did the label change?
                flip_reverse=float(wr0 != wr1), tv=tv,
                neutral_tie=tp.sum() > 1, persona_tie=tq.sum() > 1,
                same_top=same_top, same_support=same_support, gained=gain, lost=loss,
                gain_any=bool(gain.any()), loss_any=bool(loss.any()),
                n_gained=int(gain.sum()), n_lost=int(loss.sum()),
                same_support_change=bool(same_support and moved),
                hidden_label=bool(w0 == w1 and moved),         # moved, but same selected label
                hidden_top=bool(same_top and moved),           # moved, but same top emotion(s)
                strict_proportion=bool(same_top and same_support and moved),
                entropy_change=float(entropy(q) - entropy(p)), change=change)
'''

BUILDING_BLOCKS_DEMO = r'''
# A tiny worked example: neutral = anger, anger, sadness; persona = anger, fear, fear.
demo = compare_votes(np.array([EI['anger'], EI['anger'], EI['sadness']]),
                     np.array([EI['anger'], EI['fear'], EI['fear']]))
print('neutral shares:', {e: round(float(s), 3) for e, s in zip(E, demo['p']) if s})
print('persona shares:', {e: round(float(s), 3) for e, s in zip(E, demo['q']) if s})
print('shift (TV)    :', round(demo['tv'], 3))
print('label changed :', bool(demo['flip']), f"({E[demo['winner0']]} -> {E[demo['winner1']]})")
print('kind of change:', demo['change'])
'''

LOADING = r'''
# Some model answers were stored only in raw form (e.g. '5. joy', '**joy**' or just '5').
# This turns them into a plain emotion name. It is only used where the cleaned column is blank.
_NUMBERED = re.compile(r'^\s*(\d{1,2})\s*[.\):]\s*(.+?)\s*$')
_BARE_NUMBER = re.compile(r'^\s*(\d{1,2})\s*[.\):]?\s*$')


def clean_model_answer(raw):
    if pd.isna(raw):
        return raw
    s = str(raw).strip().strip('*').strip()
    m = _BARE_NUMBER.match(s)
    if m:                                   # just a number: the n-th emotion in the list
        idx = int(m.group(1))
        return E[idx - 1] if 1 <= idx <= len(E) else s.lower()
    m = _NUMBERED.match(s)
    if m:                                   # '5. joy' -> 'joy'
        s = m.group(2).strip('*').strip()
    return s.lower()


def original_corpus_shares(ids):
    """Vote shares of the 5 ORIGINAL corpus readers per text (no persona).
    Votes outside the 11 emotions (no-emotion, boredom) are left out."""
    votes = pd.read_csv(DATA / 'corpus' / 'crowd-enVent_validation.tsv', sep='\t',
                        usecols=['text_id', 'emotion'])
    votes['text_id'] = votes.text_id.astype(str)
    votes = votes[votes.text_id.isin(set(ids))]
    shares = {}
    for tid, g in votes.groupby('text_id'):
        cleaned = [v.strip().lower() for v in g.emotion.astype(str) if v.strip() != '']
        valid = [v for v in cleaned if v in EI]
        shares[tid] = (np.array([valid.count(e) / len(valid) for e in E]) if valid
                       else np.full(len(E), np.nan))
    return shares


class Study:
    """Loads every vote once and builds all neutral-vs-persona comparisons."""

    def __init__(self):
        # 1. The 300 sampled texts and their group.
        self.master = pd.read_csv(DATA / 'sampled_texts.csv')
        self.master['text_id'] = self.master.text_id.astype(str)
        assert self.master.text_id.is_unique and len(self.master) == 300
        self.ids = sorted(self.master.text_id, key=int)
        by_id = self.master.set_index('text_id')
        self.category = by_id.classification.map(POOL_TO_KEY).to_dict()
        assert pd.Series(self.category).value_counts().reindex(CATS).eq(100).all()
        self.text = by_id.text.to_dict()

        # 2. Human votes: one row per text and condition, votes stored like 'joy|joy|pride'.
        self.human_raw = pd.read_csv(DATA / 'human_annotations.csv')
        self.human_raw['text_id'] = self.human_raw.text_id.astype(str)
        assert not self.human_raw.duplicated(['text_id', 'side']).any()
        self.hvotes, self.personas = {}, {}
        for _, r in self.human_raw.iterrows():
            labels = [x.strip().lower() for x in str(r.raters_raw).split('|')]
            self.hvotes[(r.text_id, r.side)] = np.array([EI[x] for x in labels if x in EI])
            self.personas[(r.text_id, r.side)] = str(r.persona)
        assert len(self.hvotes) == 900 and all(len(v) >= 3 for v in self.hvotes.values())
        assert set(self.hvotes) == set(product(self.ids, ['neutral', 'a', 'b']))

        # 3. Model answers: 7,200 rows = 300 texts x 4 framings x 6 prompt set-ups; 7 models per row.
        raw = pd.read_csv(DATA / 'experiment_results_all.csv')
        raw['text_id'] = raw.text_id.astype(str)
        keys = ['text_id', 'framing_condition', 'prompt_variant', 'label_format']
        assert len(raw) == 7200 and not raw.duplicated(keys).any()
        self.configs = sorted(set(zip(raw.prompt_variant, raw.label_format)))
        assert len(self.configs) == 6
        self.mvotes = {}
        for _, r in raw.iterrows():
            votes = {}
            for model, col in MODEL_COLS.items():
                answer = r[col]
                if pd.isna(answer) or not str(answer).strip():
                    answer = clean_model_answer(r[MODEL_RAW_COLS[model]])
                answer = str(answer).strip().lower()
                if answer in EI:                # answers outside the 11 emotions are left out
                    votes[model] = EI[answer]
            self.mvotes[(r.text_id, r.framing_condition, r.prompt_variant, r.label_format)] = votes
        assert set(self.mvotes) == {(tid, fr, pv, lf) for tid in self.ids
                                    for fr in ['neutral', 'generic_human', 'persona_a', 'persona_b']
                                    for pv, lf in self.configs}

        # 4. Build every comparison: one text, neutral vs ONE persona.
        #    Humans: 300 texts x 2 personas = 600. Models: also x 6 prompt set-ups = 3,600.
        #    For models, only models with a valid answer on BOTH sides are used.
        h, l, individual = [], [], []
        for tid in self.ids:
            for side in ['a', 'b']:
                h.append(dict(text_id=tid, category=self.category[tid], source='Human', side=side,
                              **compare_votes(self.hvotes[(tid, 'neutral')], self.hvotes[(tid, side)])))
                for pv, lf in self.configs:
                    n = self.mvotes[(tid, 'neutral', pv, lf)]
                    p = self.mvotes[(tid, 'persona_' + side, pv, lf)]
                    common = sorted(n.keys() & p.keys())
                    assert len(common) >= 3
                    a = np.array([n[m] for m in common])
                    b = np.array([p[m] for m in common])
                    l.append(dict(text_id=tid, category=self.category[tid], source='LLM', side=side,
                                  config=pv + '/' + lf, models='|'.join(common), **compare_votes(a, b)))
                    individual.extend(dict(text_id=tid, category=self.category[tid], side=side,
                                           config=pv + '/' + lf, model=m, changed=int(n[m] != p[m]))
                                      for m in common)
        self.h = pd.DataFrame(h)                      # the 600 human comparisons
        self.l = pd.DataFrame(l)                      # the 3,600 model comparisons
        self.individual = pd.DataFrame(individual)    # every single model's answer pair
        self.cells = pd.concat([self.h, self.l], ignore_index=True)
        assert len(self.h) == 600 and len(self.l) == 3600
        assert self.l.groupby(['text_id', 'side']).size().eq(6).all()

        # 5. The original corpus readers (used for the before-persona comparison).
        self.original = original_corpus_shares(self.ids)
        self.null_cache, self.test_cache = {}, {}
        audit = self.cells.drop(columns=['p', 'q', 'delta', 'neutral_votes', 'persona_votes',
                                         'gained', 'lost'])
        audit.to_csv(RESULTS / 'comparison_audit.csv', index=False)


def record_hashes():
    """Fingerprint every input file (results/input_sha256.csv) to prove the data were not changed."""
    rows = []
    for p in sorted(DATA.rglob('*')):
        if p.is_file():
            rows.append(dict(file=str(p.relative_to(ROOT)),
                             sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    pd.DataFrame(rows).to_csv(RESULTS / 'input_sha256.csv', index=False)


def design_overview(study):
    rows = []
    for src, d in [('Human', study.h), ('LLM', study.l)]:
        rows.append(dict(source=src, texts=d.text_id.nunique(), neutral_persona_comparisons=len(d),
                         observations_per_text=2 if src == 'Human' else 12,
                         minimum_votes=int(min(d.neutral_n.min(), d.persona_n.min())),
                         maximum_votes=int(max(d.neutral_n.max(), d.persona_n.max()))))
    return pd.DataFrame(rows)
'''

LOAD_RUN = r'''
study = Study()
record_hashes()
table(design_overview(study), 'design_overview')
'''

BASELINE = r'''
def baseline(study):
    """Before any persona. For each text compare (1) the original corpus readers with our new
    human raters, and (2) our human raters with the model ensemble in each prompt set-up."""
    rows = []
    for tid in study.ids:
        h = vec(study.hvotes[(tid, 'neutral')])
        o = study.original[tid]
        rows.append(dict(text_id=tid, category=study.category[tid],
                         comparison='Original vs new human neutral',
                         label_agrees=np.argmax(h) == np.argmax(o),
                         tv=np.abs(h - o).sum() / 2, cosine=cosine(h, o)))
        for pv, lf in study.configs:
            m = vec(list(study.mvotes[(tid, 'neutral', pv, lf)].values()))
            rows.append(dict(text_id=tid, category=study.category[tid],
                             comparison='Human vs LLM neutral',
                             label_agrees=np.argmax(h) == np.argmax(m),
                             tv=np.abs(h - m).sum() / 2, cosine=cosine(h, m)))
    return pd.DataFrame(rows)
'''

AVERAGES = r'''
def boot_mean(x, seed=42, n=N_RANDOM):
    """Mean plus a 95% error bar. The error bar: redraw the list of values (one per TEXT) with
    replacement 10,000 times, recompute the mean each time, keep the middle 95%."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(x.mean()), *np.quantile(means, [.025, .975])


def summary(study, metric, by_category=True):
    """Average of a measure, first per text (so each text counts once), then over texts."""
    groups = ['source'] + (['category'] if by_category else [])
    rows = []
    for key, g in study.cells.groupby(groups, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        per_text = g.groupby('text_id')[metric].mean().to_numpy(float)
        mean, lo, hi = boot_mean(per_text)
        rows.append(dict(zip(groups, key)) | dict(comparisons=len(g), texts=len(per_text),
                                                  value_pct=100 * mean, lo_pct=100 * lo, hi_pct=100 * hi))
    return pd.DataFrame(rows)


def bar(study, metric, name, title, ylabel):
    """Bar chart of summary() by group, humans next to models, with 95% error bars."""
    d = summary(study, metric)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, src in enumerate(['Human', 'LLM']):
        q = d[d.source == src].set_index('category').reindex(CATS)
        x = np.arange(3) + (j - .5) * .35
        ax.bar(x, q.value_pct, .33, label=src, color=COLORS[src])
        ax.errorbar(x, q.value_pct, yerr=[q.value_pct - q.lo_pct, q.hi_pct - q.value_pct],
                    fmt='none', color='#303030', capsize=4)
        for xx, yy in zip(x, q.value_pct):
            ax.text(xx, yy + 1.5, f'{yy:.1f}%', ha='center', fontsize=10)
    ax.set_xticks(range(3), CAT_NAMES)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, min(110, max(d.hi_pct) * 1.22 + 3))
    finish(fig, name)
    table(d, name)
    return d
'''

TRANSITIONS = r'''
def transitions(study):
    """Count how often each winning neutral emotion (rows) becomes each persona emotion (columns)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (src, d) in zip(axes, [('Human', study.h), ('LLM', study.l)]):
        mat = pd.crosstab(d.winner0, d.winner1).reindex(index=range(11), columns=range(11), fill_value=0)
        arr = mat.to_numpy()
        ax.imshow(arr, cmap='Blues')
        for i in range(11):
            for j in range(11):
                if arr[i, j]:
                    ax.text(j, i, str(arr[i, j]), ha='center', va='center', fontsize=7,
                            color='white' if arr[i, j] > arr.max() / 2 else 'black')
        ax.set_xticks(range(11), E, rotation=60, ha='right', fontsize=8)
        ax.set_yticks(range(11), E, fontsize=8)
        ax.set_title(f'{src}: {len(d):,} comparisons')
        ax.set_xlabel('Selected persona label')
        ax.set_ylabel('Selected neutral label')
    finish(fig, 'old_transitions')
'''

TESTS = r'''
def chance_shuffles(study, source, index):
    """For ONE comparison, list every way the votes could have come out if the persona did nothing.
    Humans (different people per version): pool the neutral and persona votes, then try every way
      of splitting them back into two groups of the original sizes (3 + 3 votes = 20 ways).
    Models (the same model answered both versions): each model's own two answers are either kept
      or swapped, giving 2 x 2 x ... possibilities (128 for 7 models).
    Returns, for every possibility, whether the label changed and how big the shift is."""
    key = (source, index)
    if key in study.null_cache:
        return study.null_cache[key]
    r = (study.h if source == 'Human' else study.l).iloc[index]
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
    result = {'flip': (pa.argmax(axis=1) != pb.argmax(axis=1)).astype(float),
              'tv': np.abs(pa - pb).sum(axis=1) / 2}
    study.null_cache[key] = result
    return result


def holm_correct(pvals):
    """Holm correction: sort the p-values, multiply the smallest by the number of tests, the next
    by one less, and so on; never let an adjusted value drop below the one before it; cap at 1."""
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


def test_h1_h2(study, metric):
    """H1 (humans: 3 groups x 2 personas = 6 tests) and H2 (models: 2 personas = 2 tests).
    metric is 'flip' (old method: did the label change?) or 'tv' (vector method: shift size).
    For each test:
      1. observed = average of the metric over the comparisons in that group;
      2. chance = 10,000 times, give every comparison one random outcome from chance_shuffles()
         and average them -> 10,000 averages that could happen if the persona did nothing;
      3. p = share of those chance averages that are at least as big as the observed one."""
    if metric in study.test_cache:
        return study.test_cache[metric].copy()
    rows = []
    for source, d in [('Human', study.h), ('LLM', study.l)]:
        for side in ['a', 'b']:
            for cat in (CATS if source == 'Human' else ['all']):
                sub = d[(d.side == side) & ((d.category == cat) if cat != 'all' else True)]
                rng = np.random.default_rng(4200 + len(rows))
                null = np.zeros(N_RANDOM)
                expect = 0
                for index in sub.index:
                    values = chance_shuffles(study, source, index)[metric]
                    null += values[rng.integers(0, len(values), N_RANDOM)]
                    expect += values.mean()
                null /= len(sub)
                expect /= len(sub)
                obs = sub[metric].mean()
                p = (1 + np.count_nonzero(null >= obs - 1e-12)) / (N_RANDOM + 1)
                mean, lo, hi = boot_mean(sub.groupby('text_id')[metric].mean())
                rows.append(dict(hypothesis='H1' if source == 'Human' else 'H2', source=source,
                                 side=side, category=cat, n_cells=len(sub), observed_pct=100 * obs,
                                 conditional_null_pct=100 * expect, excess_pp=100 * (obs - expect),
                                 ci_lo=100 * lo, ci_hi=100 * hi, p_raw=p))
    out = pd.DataFrame(rows)
    out['p_holm'] = np.nan
    for h in ['H1', 'H2']:                 # correct H1's 6 tests together, H2's 2 tests together
        use = out.hypothesis == h
        out.loc[use, 'p_holm'] = holm_correct(out.loc[use, 'p_raw'])
    out['reject_05'] = out.p_holm < .05
    study.test_cache[metric] = out
    return out.copy()


def plot_h1_h2(study, metric, name):
    """Dot = observed value with 95% error bar; x = what chance alone gives; label = Holm p."""
    d = test_h1_h2(study, metric)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [2, 1]})
    for ax, h in zip(axes, ['H1', 'H2']):
        q = d[d.hypothesis == h].reset_index(drop=True)
        y = np.arange(len(q))
        ax.errorbar(q.observed_pct, y, xerr=[q.observed_pct - q.ci_lo, q.ci_hi - q.observed_pct],
                    fmt='o', color='#177e89', capsize=3, label='Observed; 95% text CI')
        ax.scatter(q.conditional_null_pct, y, marker='x', color='#d66b35', label='Conditional-null mean')
        ax.set_yticks(y, [('All texts' if r.category == 'all' else CAT_NAME[r.category]) + ' / '
                          + r.side.upper() for _, r in q.iterrows()])
        for i, r in q.iterrows():
            ax.annotate(f'p={r.p_holm:.4f}', (r.observed_pct, i), xytext=(0, 10),
                        textcoords='offset points', fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_title(h + ': ' + ('one-label changes' if metric == 'flip' else 'distribution distances'))
        ax.set_xlabel('Changed comparisons (%)' if metric == 'flip' else 'TV × 100')
        ax.set_ylim(-.7, len(q) - .3)
    axes[0].legend(fontsize=8, loc='lower left')
    finish(fig, name)
    table(d, name)
    return d
'''

H3 = r'''
def test_h3(study, metric):
    """H3: do author-relevant texts change more than author-independent ones (humans)?
    Each text gets one number = its average over the two personas.
    'Raw' compares those numbers directly.
    'Conditional-null referenced' first subtracts what chance alone would give for that same
      text (the average of chance_shuffles), then compares what is left (the 'excess').
    p-value: shuffle which group each text belongs to 10,000 times and see how often the
      difference between the group averages is at least as big as the real one (either sign).
    Error bar: resample the texts within each group 10,000 times."""
    d = study.h.copy()
    d['expected'] = [chance_shuffles(study, 'Human', i)[metric].mean() for i in d.index]
    d['excess'] = d[metric] - d.expected
    t = d.groupby(['text_id', 'category'])[[metric, 'expected', 'excess']].mean().reset_index()
    rows = []
    for mode, col in [('Raw', metric), ('Conditional-null referenced', 'excess')]:
        for a, b in [('author_relevant', 'author_independent'), ('author_independent', 'unambiguous'),
                     ('author_relevant', 'unambiguous')]:
            x = t.loc[t.category == a, col].to_numpy()
            y = t.loc[t.category == b, col].to_numpy()
            obs = x.mean() - y.mean()
            rng = np.random.default_rng(102 + len(rows))
            pooled = np.r_[x, y]
            indices = np.argsort(rng.random((N_RANDOM, len(pooled))), axis=1)
            perms = pooled[indices]
            null = perms[:, :len(x)].mean(axis=1) - perms[:, len(x):].mean(axis=1)
            p = (1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (N_RANDOM + 1)
            boots = (x[rng.integers(0, len(x), (N_RANDOM, len(x)))].mean(axis=1)
                     - y[rng.integers(0, len(y), (N_RANDOM, len(y)))].mean(axis=1))
            lo, hi = np.quantile(boots, [.025, .975])
            rows.append(dict(analysis=mode, contrast=CAT_NAME[a] + ' minus ' + CAT_NAME[b],
                             difference_pp=100 * obs, ci_lo=100 * lo, ci_hi=100 * hi, p_raw=p))
    out = pd.DataFrame(rows)
    out['p_holm'] = np.nan
    for mode in out.analysis.unique():     # Holm over the 3 contrasts, separately per analysis
        use = out.analysis == mode
        out.loc[use, 'p_holm'] = holm_correct(out.loc[use, 'p_raw'])
    return out, t
'''

H4_OLD = r'''
def h4_size(study, metric):
    """Human minus model, per text (each text's average over its comparisons), then averaged."""
    h = study.h.groupby('text_id')[metric].mean().reindex(study.ids)
    l = study.l.groupby('text_id')[metric].mean().reindex(study.ids)
    point, lo, hi = boot_mean(h - l)
    return pd.DataFrame([dict(metric=metric, n_texts=300, human_mean_pct=100 * h.mean(),
                              llm_mean_pct=100 * l.mean(), human_minus_llm_pp=100 * point,
                              ci_lo=100 * lo, ci_hi=100 * hi)])


def label_agreement(study):
    """Match every model comparison with the human comparison for the same text and persona,
    and record who changed their selected label."""
    h = study.h[['text_id', 'side', 'flip', 'winner0', 'winner1']].rename(
        columns={c: 'h_' + c for c in ['flip', 'winner0', 'winner1']})
    j = study.l.merge(h, on=['text_id', 'side'], validate='many_to_one')
    j['status'] = np.select([(j.h_flip == 0) & (j.flip == 0), (j.h_flip == 1) & (j.flip == 0),
                             (j.h_flip == 0) & (j.flip == 1)],
                            ['Both keep their label', 'Only humans change label', 'Only LLMs change label'],
                            default='Both change label')
    both = (j.h_flip == 1) & (j.flip == 1)
    agree = (j.winner0 == j.h_winner0) & (j.winner1 == j.h_winner1)
    return j, pd.DataFrame([dict(comparisons=len(j), both_changed=int(both.sum()),
                                 same_transition_among_both_changed=int((both & agree).sum()),
                                 same_transition_pct=100 * agree[both].mean(),
                                 change_status_agreement_pct=100 * (j.h_flip == j.flip).mean())])


def label_direction(study):
    """Old-method stand-in for direction: when BOTH humans and models change their label, how often
    is it the same switch (same start AND end emotion)? Chance: pair each text's human results
    with another random text's model results, 10,000 times."""
    j, _ = label_agreement(study)
    j = j.sort_values(['text_id', 'side', 'config'], key=lambda x: x.astype(int) if x.name == 'text_id' else x)
    h0 = j.h_winner0.to_numpy().reshape(300, 12)
    h1 = j.h_winner1.to_numpy().reshape(300, 12)
    l0 = j.winner0.to_numpy().reshape(300, 12)
    l1 = j.winner1.to_numpy().reshape(300, 12)
    eligible = (h0 != h1) & (l0 != l1)
    same = (h0 == l0) & (h1 == l1) & eligible
    sums = same.sum(axis=1)
    counts = eligible.sum(axis=1)
    obs = sums.sum() / counts.sum()
    rng = np.random.default_rng(302)
    idx = rng.integers(0, 300, (N_RANDOM, 300))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boots, [.025, .975])
    null = np.empty(N_RANDOM)
    for b in range(N_RANDOM):
        ix = rng.permutation(300)
        ok = (h0 != h1) & (l0[ix] != l1[ix])
        match = (h0 == l0[ix]) & (h1 == l1[ix]) & ok
        null[b] = match.sum() / ok.sum() if ok.any() else np.nan
    p = (1 + np.count_nonzero(null >= obs - 1e-12)) / (1 + np.isfinite(null).sum())
    return pd.DataFrame([dict(eligible_comparisons=int(counts.sum()), same_selected_transition=int(sums.sum()),
                              agreement_pct=100 * obs, ci_lo=100 * lo, ci_hi=100 * hi,
                              null_mean_pct=100 * np.nanmean(null), p_permutation=p)])
'''

TIES = r'''
def ties(study):
    """How much does the arbitrary tie rule matter? Compare alphabetical vs reverse-alphabetical."""
    rows = []
    for src, g in study.cells.groupby('source', sort=False):
        unique = ~g.neutral_tie & ~g.persona_tie
        rows.append(dict(source=src, comparisons=len(g), either_condition_tied=int((~unique).sum()),
                         alphabetic_change_pct=100 * g.flip.mean(),
                         reverse_order_change_pct=100 * g.flip_reverse.mean(),
                         changed_decision=int((g.flip != g.flip_reverse).sum()),
                         unique_winners_only_n=int(unique.sum()),
                         unique_winners_only_change_pct=100 * g.loc[unique, 'flip'].mean()))
    return pd.DataFrame(rows)
'''

EXAMPLE = r'''
def example(study, event, source, name):
    """Plot the comparison with the LARGEST shift among those that match `event`.
    Chosen because it is the clearest illustration, not because it is typical."""
    d = study.h if source == 'Human' else study.l
    choose = d[d[event]].sort_values(['tv', 'text_id'], ascending=[False, True])
    if choose.empty:
        explain(f'No {source} comparison meets this definition.')
        return
    r = choose.iloc[0]
    p, q = r.p, r.q
    active = np.flatnonzero((p + q) > 0)
    x = np.arange(len(active))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - .18, p[active] * 100, .35, label='Neutral', color='#586f7c')
    ax.bar(x + .18, q[active] * 100, .35, label='Persona', color=COLORS[source])
    for xx, pp, qq in zip(x, p[active] * 100, q[active] * 100):
        ax.text(xx - .18, pp + 1, f'{pp:.0f}%', ha='center', fontsize=9)
        ax.text(xx + .18, qq + 1, f'{qq:.0f}%', ha='center', fontsize=9)
    ax.set_xticks(x, [E[i] for i in active])
    ax.set_ylim(0, 110)
    ax.set_ylabel('Vote share (%)')
    ax.legend()
    ax.set_title(f'{source}, text {r.text_id}, persona {r.side.upper()} — TV {r.tv * 100:.1f}%')
    finish(fig, name)
    explain(f'**Text:** {study.text[r.text_id]}\n\n**Persona:** {study.personas[(r.text_id, r.side)]}\n\n'
            f'**Winning label:** {E[r.winner0]} → {E[r.winner1]}. **Votes:** {r.neutral_n} → {r.persona_n}. '
            f'**Emotions added:** {", ".join(np.array(E)[r.gained]) or "none"}; '
            f'**removed:** {", ".join(np.array(E)[r.lost]) or "none"}. '
            + (f'**Prompt set-up:** {r.config}. ' if source == 'LLM' else '')
            + 'This is the example with the largest shift of its kind, chosen to illustrate, not to be typical.')
'''

EVENTS = r'''
EVENT_TYPES = ['No distribution change', 'Same emotions; proportions change', 'Emotions appear only',
               'Emotions disappear only', 'Emotions appear AND disappear']


def events(study):
    """Count the five kinds of change, per source."""
    rows = []
    for src, d in [('Human', study.h), ('LLM', study.l)]:
        for change in EVENT_TYPES:
            flag = d.change == change
            rows.append(dict(source=src, event=change, comparisons=int(flag.sum()), denominator=len(d),
                             percent=100 * flag.mean(),
                             texts_with_event=int(d.loc[flag, 'text_id'].nunique()), text_denominator=300))
    return pd.DataFrame(rows)


def events_plot(study):
    d = events(study)
    fig, ax = plt.subplots(figsize=(11, 5))
    names = d.event.drop_duplicates().tolist()
    y = np.arange(len(names))
    for j, src in enumerate(['Human', 'LLM']):
        q = d[d.source == src]
        ax.barh(y + (j - .5) * .36, q.percent, .34, label=src, color=COLORS[src])
        for yy, (_, r) in zip(y + (j - .5) * .36, q.iterrows()):
            ax.text(r.percent + .5, yy, f'{r.percent:.1f}% ({r.comparisons:,}/{r.denominator:,})',
                    va='center', fontsize=9)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, min(100, d.percent.max() + 26))
    ax.set_xlabel('Share of neutral–persona comparisons (%)')
    ax.set_title('What changed inside the distribution?')
    ax.legend()
    finish(fig, 'vector_change_types')
    table(d, 'vector_change_types')
    return d
'''

DIRECTION = r'''
def direction(study, n_perm=N_RANDOM):
    """H4 direction. For each text, persona and prompt set-up: cosine between the human shift
    vector and the model shift vector (persona sides kept separate, never averaged).
    Comparisons where either side did not move have no direction and are left out.
    Chance: pair each text's human shifts with another random text's model shifts, 10,000 times."""
    h_index = study.h.set_index(['text_id', 'side'])
    l_index = study.l.set_index(['text_id', 'side', 'config'])
    hd = np.stack([h_index.loc[(tid, side), 'delta'] for tid in study.ids
                   for side in ['a', 'b']]).reshape(300, 2, 11)
    ld = np.stack([l_index.loc[(tid, side, pv + '/' + lf), 'delta'] for tid in study.ids
                   for side in ['a', 'b'] for pv, lf in study.configs]).reshape(300, 2, 6, 11)
    hs = np.repeat(hd[:, :, None, :], 6, axis=2)       # same human shift for all 6 set-ups
    cos = cosine(hs, ld)
    mask = np.isfinite(cos)
    sums = np.nansum(cos, axis=(1, 2))
    counts = mask.sum(axis=(1, 2))
    obs = sums.sum() / counts.sum()
    rng = np.random.default_rng(301)
    idx = rng.integers(0, 300, (N_RANDOM, 300))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(300)
        null[b] = np.nanmean(cosine(hs, ld[perm]))
    p = (1 + (null >= obs - 1e-12).sum()) / (n_perm + 1)
    return pd.DataFrame([dict(mean_cosine=obs, ci_lo=lo, ci_hi=hi, defined_comparisons=int(mask.sum()),
                              all_comparisons=3600, texts_with_any_defined=int((counts > 0).sum()),
                              null_mean=float(null.mean()), p_permutation=p)]), cos, null
'''

EMOTION_EVENTS = r'''
def emotion_events(study):
    """For each emotion: how often it appears (0 votes -> some) or disappears (some -> 0)."""
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (src, d) in zip(axes, [('Human', study.h), ('LLM', study.l)]):
        gained = np.stack(d.gained).mean(axis=0) * 100
        lost = np.stack(d.lost).mean(axis=0) * 100
        y = np.arange(11)
        ax.barh(y - .18, gained, .35, color='#177e89', label='Appears (zero → positive)')
        ax.barh(y + .18, lost, .35, color='#d66b35', label='Disappears (positive → zero)')
        ax.set_yticks(y, E)
        ax.invert_yaxis()
        ax.set_title(src)
        ax.set_xlabel('Comparisons (%)')
        ax.legend(fontsize=8)
        for e, g, l in zip(E, gained, lost):
            rows.append(dict(source=src, emotion=e, appears_pct=g, disappears_pct=l))
    finish(fig, 'emotion_appearance_disappearance')
    out = pd.DataFrame(rows)
    table(out, 'emotion_appearance_disappearance')
    return out


def bh_fdr(pvals):
    """Benjamini-Hochberg correction for many exploratory tests: sort the p-values; the k-th
    smallest is multiplied by (number of tests / k); keep the running minimum from the top; cap at 1."""
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
'''

COUNTS = r'''
def count_sensitivity(study, draws=200):
    """Give both sides the same number of votes: 3 randomly chosen human votes per version, and
    the SAME 3 randomly chosen models on both sides of a model comparison. Repeat 200 times and
    average the shift and the spread change."""
    rng = np.random.default_rng(42)
    rows = []
    for src, d in [('Human', study.h), ('LLM', study.l)]:
        for _, r in d.iterrows():
            a, b = r.neutral_votes, r.persona_votes
            ia = np.argsort(rng.random((draws, len(a))), axis=1)[:, :3]
            ib = np.argsort(rng.random((draws, len(b))), axis=1)[:, :3] if src == 'Human' else ia
            p = np.eye(11)[a[ia]].mean(axis=1)
            q = np.eye(11)[b[ib]].mean(axis=1)
            rows.append(dict(source=src, text_id=r.text_id, category=r.category,
                             tv_raw=r.tv, tv_standardised=np.abs(q - p).sum(axis=1).mean() / 2,
                             entropy_raw=r.entropy_change,
                             entropy_standardised=(entropy(q) - entropy(p)).mean()))
    out = pd.DataFrame(rows)
    summary_rows = []
    for src, g in out.groupby('source', sort=False):
        for col in ['tv_raw', 'tv_standardised', 'entropy_raw', 'entropy_standardised']:
            point, lo, hi = boot_mean(g.groupby('text_id')[col].mean())
            summary_rows.append(dict(source=src, quantity=col, mean=point, ci_lo=lo, ci_hi=hi))
    return out, pd.DataFrame(summary_rows)


def prompt_test(study, metric):
    """Friedman test: within each text, rank the 6 prompt set-ups by the metric; if prompt format
    did not matter, every set-up would get similar average ranks."""
    d = study.l.groupby(['text_id', 'config'])[metric].mean().unstack('config').reindex(study.ids)
    stat, p = stats.friedmanchisquare(*[d[c] for c in d])
    return d, pd.DataFrame([dict(metric=metric, texts=300, configurations=6, friedman_stat=stat, p_value=p)])
'''

# =============================================================================================
# Notebook 01
# =============================================================================================


def notebook_01():
    c = []
    c.append(md(r'''
# Notebook 1: From one winning label to full vote distributions

**The question.** If the words of a text stay exactly the same but we say who wrote it, does the
emotion people read into it change? And do LLMs react the way people do?

**How to read this notebook**
- Read the text cells first. The code cells can be skipped on a first read: every result is
  explained right under it.
- **All the code is here.** Nothing is hidden in another file. Each piece of code appears just
  before the first result that uses it.
- New to the terms (vote shares, shift, chance shuffles, p-value, Holm)? Read Part 1 of
  `Presentation_Script_Guided_Analysis.pdf` first.

**The plan**

| Part | What happens |
|---|---|
| A | Analyse everything the **old way**: keep only the winning emotion per text |
| B | Show what that old way **throws away** |
| C | Redo the same questions **keeping all the votes** (vote-share vectors) |
| D | **Compare** the two sets of answers |

**The four hypotheses (plain words)**

| | Prediction |
|---|---|
| H1 | People read ambiguous texts differently when told who wrote them; clear texts hardly change |
| H2 | LLMs change their answer when told who wrote the text |
| H3 | The author matters more for author-relevant texts than for author-independent texts |
| H4 | LLMs move partly in the same direction as people, but by a different amount |

Not every hypothesis has to be confirmed; the point is to test them.
'''))
    c.append(md(r'''
## Step 0: Setup

Libraries, folders and three small helpers (`explain`, `table`, `finish`) that show and save
results. Nothing is calculated here.
'''))
    c.append(code(SETUP))
    c.append(md(r'''
## Step 1: The building blocks

Two ideas are used everywhere in this notebook.

**Vote shares.** Votes become percentages over the 11 emotions.
*anger, anger, sadness* → 67% anger, 33% sadness, 0% for everything else.

**Comparing two sets of votes** (`compare_votes`): for one text, the neutral votes against the
votes under one persona. It records:
- the **shift** (total variation, TV): how much of the vote share has to move to turn one set into
  the other. 0 = identical, 1 = completely different emotions. It is half of the summed
  differences, so nothing is counted twice;
- the **old-method label change**: did the winning emotion change? (ties go to the alphabetically
  first emotion);
- which emotions **appear** or **disappear**, and which of **five kinds of change** it is;
- the change in **spread** (entropy).

The small example at the end shows all of this on three made-up votes per side.
'''))
    c.append(code(SETTINGS))
    c.append(code(BUILDING_BLOCKS))
    c.append(code(BUILDING_BLOCKS_DEMO))
    c.append(md(r'''
**Reading the example.** Anger falls from 67% to 33%, sadness from 33% to 0% and fear rises from
0% to 67%. The differences add up to 33 + 33 + 67 = 133 points; half of that is **67%**, the
shift. The winning label changed (anger → fear), fear appeared and sadness disappeared.

## Step 2: Load all the data

- **300 texts** (100 per group) from `sampled_texts.csv`.
- **Human votes**: about 3 people per version (neutral, persona A, persona B). Nobody saw two
  versions of the same text.
- **Model answers**: 7 models × 4 versions (neutral, "written by a person", persona A, persona B)
  × 6 prompt set-ups.
- Then every **comparison** is built: one text, neutral against one persona.
  Humans: 300 × 2 = **600**. Models: 300 × 2 × 6 = **3,600**. These are still only 300 different
  texts, so every error bar and test treats the **text** as the unit.
- For each model comparison, only models with a valid answer on **both** sides are used, so the
  group of models never changes between the two sides.
'''))
    c.append(code(LOADING))
    c.append(code(LOAD_RUN))

    # ------------------------------------------------------------------ Part A
    c.append(md(r'''
# Part A: The old method (one winning label per text)

The old method keeps only the most-voted emotion, the **selected label**. When there is a tie,
the alphabetically first emotion wins. That is an arbitrary rule; Part B shows it matters.

## A1. From votes to one label
'''))
    c.append(code(r'''
preview = []
for tid in study.ids[:6]:
    a = study.hvotes[(tid, 'neutral')]
    counts = np.bincount(a, minlength=11)
    top = counts.max() / counts.sum()
    preview.append({'text_id': tid, 'neutral_votes': ', '.join(E[i] for i in a),
                    'selected_label': E[counts.argmax()], 'top_share_pct': 100 * top,
                    'strict_majority': bool(top > .5), 'tied_top': bool((counts == counts.max()).sum() > 1)})
table(pd.DataFrame(preview), 'old_vote_to_label_examples', 1)
explain('**Read this table:** each row shows the votes and the single label the old method keeps. '
        'The first two rows are three-way ties: "anger" wins only because it comes first in the alphabet.')
'''))
    c.append(md(r'''
## A2. Before any persona: do the labels agree?

Two comparisons, both with **no author** given:
- the **original** corpus readers vs **our new** human raters;
- our human raters vs the **model ensemble** (the 7 models' answers together).
'''))
    c.append(code(BASELINE))
    c.append(code(r'''
baseline_rows = baseline(study)
base_label = baseline_rows.groupby(['comparison', 'category']).agg(
    comparisons=('text_id', 'size'), agreement_pct=('label_agrees', lambda x: 100 * x.mean())).reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
for j, comp in enumerate(base_label.comparison.unique()):
    sub = base_label[base_label.comparison == comp].set_index('category').reindex(CATS)
    ax.bar(np.arange(3) + (j - .5) * .35, sub.agreement_pct, .33, label=comp)
    for i, v in enumerate(sub.agreement_pct):
        ax.text(i + (j - .5) * .35, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
ax.set_xticks(range(3), CAT_NAMES)
ax.set_ylim(0, 105)
ax.set_ylabel('Selected labels agreeing (%)')
ax.set_title('Old method: agreement before personas')
ax.legend(fontsize=9)
finish(fig, 'old_baseline_agreement')
table(base_label, 'old_baseline_agreement', 1)
explain('**What it means:** even with no author at all, two groups of readers pick the same winning '
        'emotion only about half the time on ambiguous texts. Any persona effect has to beat this '
        'background disagreement. A match or mismatch also hides *how* different the votes are.')
'''))
    c.append(md(r'''
## A3. Under a persona, how often does the winning emotion change?

For each comparison: 1 if the winning emotion changed, 0 if not. The bars show the percentage
that changed.

**How the averages and error bars work** (`summary`, `boot_mean`): each text first gets its own
average (humans have 2 comparisons per text, models 12), so every text counts once. The black
error bar is the **95% range**: redraw the 100 texts with replacement 10,000 times, recompute the
average each time, and keep the middle 95%.
'''))
    c.append(code(AVERAGES))
    c.append(code(r'''
old_rates = bar(study, 'flip', 'old_label_change_rates', 'Old method: how often does the selected emotion change?',
                'Comparisons changing selected label (%)')
old_total = summary(study, 'flip', False)
table(old_total, 'old_overall_change_rates', 2)
for _, r in old_total.iterrows():
    explain(f'**{r.source}:** the winning emotion changes in {r.value_pct:.1f}% of comparisons. '
            'Every other comparison counts as "unchanged", whatever happened to the other votes.')
explain('**Not yet evidence:** people change 39% of the time even on clear texts, partly just because '
        'different people rated each version. Whether this beats chance is tested in A5.')
'''))
    c.append(md(r'''
## A4. Which emotion turns into which?

Rows = winning emotion without a persona, columns = with a persona. The diagonal means no change.
'''))
    c.append(code(TRANSITIONS))
    c.append(code(r'''
transitions(study)
explain('**What it adds:** now we can see *which* switches happen (models mostly move from joy to pride, '
        'relief or surprise). **Still missing:** how much of the vote moved, and any change that does '
        'not flip the winner.')
'''))
    c.append(md(r'''
## A5. H1 and H2, the old way: are these changes more than chance?

**The idea: a chance shuffle** (`chance_shuffles`). If the persona did nothing, it would not matter
which votes came from which version.
- **Humans:** different people rated each version, so pool the neutral and persona votes of a text
  and try every way of splitting them back into two groups of the same sizes. With 3 + 3 votes
  there are exactly 20 ways.
- **Models:** the same model answered both versions, so each model's own two answers are either
  kept or swapped. With 7 models that is 128 possibilities.

**The test** (`test_h1_h2`): the observed average is compared with 10,000 averages built from these
chance possibilities. **p** = the share of chance averages at least as big as the observed one.

**Holm correction** (`holm_correct`): H1 is 6 tests (3 groups × 2 personas). Running 6 tests raises
the odds of a lucky pass, so the p-values are adjusted upwards before comparing with 0.05.

In the graph, the dot is the observed value, the × is what chance gives on average, and the
label is the corrected p-value.
'''))
    c.append(code(TESTS))
    c.append(code(r'''
old_h12 = plot_h1_h2(study, 'flip', 'old_H1_H2')
for hyp in ['H1', 'H2']:
    q = old_h12[old_h12.hypothesis == hyp]
    explain(f'**{hyp}, old method:** {q.reject_05.sum()} of {len(q)} tests are significant after correction.')
explain('**What it means:** counting only the winning emotion, **no human framing effect is detected**. '
        'The model effect is detected. "Not detected" is not the same as "no effect": Part C tests '
        'the same question with all the votes.')
'''))
    c.append(md(r'''
## A6. H3, the old way: do author-relevant texts change more?

H3 is specifically **author-relevant minus author-independent**. Each text gets one number (its
average over the two personas); the groups are compared, and the p-value comes from shuffling
which group each text belongs to (`test_h3`).
'''))
    c.append(code(H3))
    c.append(code(r'''
old_h3, old_h3_text = test_h3(study, 'flip')
old_h3_raw = old_h3[old_h3.analysis == 'Raw']
table(old_h3_raw, 'old_H3_raw')
r = old_h3_raw.iloc[0]
fig, ax = plt.subplots(figsize=(9, 4))
ax.errorbar(old_h3_raw.difference_pp, np.arange(3),
            xerr=[old_h3_raw.difference_pp - old_h3_raw.ci_lo, old_h3_raw.ci_hi - old_h3_raw.difference_pp],
            fmt='o', capsize=4)
ax.axvline(0, color='grey', ls='--')
ax.set_yticks(range(3), old_h3_raw.contrast)
ax.set_xlabel('Difference in selected-label-change rate (percentage points)')
ax.set_title('H3, old method: category differences')
finish(fig, 'old_H3_contrasts')
explain(f'**H3, old method:** author-relevant texts change their label {r.difference_pp:.2f} points more often '
        f'than author-independent ones; 95% range [{r.ci_lo:.2f}, {r.ci_hi:.2f}], corrected p = {r.p_holm:.4f}. '
        + ('That is significant.' if r.p_holm < .05 else
           'The range includes zero, so no difference is detected.'))
'''))
    c.append(md(r'''
## A7. H4, the old way: what can one label say?

Two rough stand-ins, because one label cannot measure the size or direction of a full shift:
1. **Size stand-in:** do people change their label more often than models?
2. **Direction stand-in:** when *both* change, do they make the *same* switch (e.g. both sadness →
   guilt)? Chance: pair each text's human results with a random other text's model results.
'''))
    c.append(code(H4_OLD))
    c.append(code(r'''
old_h4 = h4_size(study, 'flip')
table(old_h4, 'old_H4_frequency')
old_pairs, old_agreement = label_agreement(study)
table(old_agreement, 'old_H4_change_agreement')
old_direction = label_direction(study)
table(old_direction, 'old_H4_transition_proxy')
status = old_pairs.status.value_counts().reindex(
    ['Both keep their label', 'Only humans change label', 'Only LLMs change label', 'Both change label'],
    fill_value=0)
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(status.index, status.values / len(old_pairs) * 100, color=['#8795a1', '#177e89', '#d66b35', '#7359a3'])
ax.invert_yaxis()
for i, n in enumerate(status):
    ax.text(n / len(old_pairs) * 100 + .5, i, f'{n:,}/3,600 ({n / len(old_pairs) * 100:.1f}%)', va='center')
ax.set_xlim(0, max(status) / len(old_pairs) * 100 + 22)
ax.set_xlabel('Matched comparisons (%)')
ax.set_title('Old method: do humans and models change their labels together?')
finish(fig, 'old_H4_status')
r = old_h4.iloc[0]
dr = old_direction.iloc[0]
explain(f'**Size stand-in:** people change their label {r.human_minus_llm_pp:.2f} points more often than models '
        f'(range [{r.ci_lo:.2f}, {r.ci_hi:.2f}]). **Direction stand-in:** in {dr.agreement_pct:.1f}% of the '
        f'{int(dr.eligible_comparisons):,} cases where both changed, they made the same switch; chance gives '
        f'{dr.null_mean_pct:.1f}% (p = {dr.p_permutation:.4f}).')
'''))
    c.append(md('## A8. Summary of the old method'))
    c.append(code(r'''
n_h1_old = int(old_h12.query("hypothesis == 'H1'").reject_05.sum())
n_h2_old = int(old_h12.query("hypothesis == 'H2'").reject_05.sum())
old_summary = pd.DataFrame([
    {'hypothesis': 'H1', 'old_method_answer': f'{n_h1_old}/6 adjusted tests reject; category/side-specific',
     'scope': 'Selected-label-change rate; non-rejection is not absence'},
    {'hypothesis': 'H2', 'old_method_answer': f'{n_h2_old}/2 adjusted tests reject',
     'scope': 'Ensemble selected-label changes, preserving model pairs'},
    {'hypothesis': 'H3', 'old_method_answer': f'AR−AI {old_h3_raw.iloc[0].difference_pp:.2f}pp; adjusted p={old_h3_raw.iloc[0].p_holm:.4f}',
     'scope': 'Raw category difference in label-change frequency'},
    {'hypothesis': 'H4', 'old_method_answer': f'Human−model label-change frequency {old_h4.iloc[0].human_minus_llm_pp:.2f}pp; transition agreement {old_direction.iloc[0].agreement_pct:.1f}%',
     'scope': 'Only categorical proxies; full magnitude/direction unavailable'}])
table(old_summary, 'old_hypothesis_summary')
'''))

    # ------------------------------------------------------------------ Part B
    c.append(md(r'''
# Part B: What does the old method throw away?

## B1. Ties make the answer depend on an arbitrary rule

Repeat the label-change count with ties broken by the **reverse** alphabet. If the result
changes, the old method depends on a rule that has nothing to do with the data.
'''))
    c.append(code(TIES))
    c.append(code(r'''
tie_report = ties(study)
table(tie_report, 'tie_rule_sensitivity', 2)
explain('**What it means:** `changed_decision` counts comparisons whose changed/unchanged verdict flips just '
        'by reversing the tie rule. For people that is a large share, because 3-vote ties are common.')
'''))
    c.append(md(r'''
## B2. A label that stays the same can hide real movement

Two ways of counting hidden movement:
- **Same selected label:** the old method says "no change", but the votes moved.
- **Same complete top set:** even keeping all tied winners says "no change", but the votes moved.

Both are shown as a share of **all** comparisons. The table also gives the share of the
"unchanged" comparisons that actually moved (`hidden_pct_of_unchanged_labels`). Do not mix up
the two denominators.
'''))
    c.append(code(r'''
hidden = []
for src, d in [('Human', study.h), ('LLM', study.l)]:
    unchanged = (d.flip == 0).sum()
    hidden.append({'source': src, 'all_comparisons': len(d), 'selected_label_unchanged': int(unchanged),
                   'moved_with_same_selected_label': int(d.hidden_label.sum()),
                   'hidden_pct_of_all': 100 * d.hidden_label.mean(),
                   'hidden_pct_of_unchanged_labels': 100 * d.hidden_label.sum() / unchanged,
                   'moved_with_same_complete_top_set': int(d.hidden_top.sum()),
                   'same_top_set_hidden_pct': 100 * d.hidden_top.mean()})
hidden = pd.DataFrame(hidden)
table(hidden, 'information_lost_by_labels', 2)
fig, ax = plt.subplots(figsize=(9, 5))
for j, col in enumerate(['hidden_pct_of_all', 'same_top_set_hidden_pct']):
    ax.bar(np.arange(2) + (j - .5) * .34, hidden[col], .32, label=['Same selected label', 'Same complete top set'][j])
    for i, v in enumerate(hidden[col]):
        ax.text(i + (j - .5) * .34, v + 1, f'{v:.1f}%', ha='center')
ax.set_xticks(range(2), hidden.source)
ax.set_ylabel('All comparisons with concealed movement (%)')
ax.set_ylim(0, hidden.hidden_pct_of_all.max() + 15)
ax.set_title('What the categorical summaries missed')
ax.legend(fontsize=9)
finish(fig, 'hidden_distribution_movement')
explain('**What it means:** a recorded "unchanged" label does not mean the votes stayed the same. This is '
        'lost information; it does not prove every hidden movement is caused by the persona, because '
        'small vote samples also vary by chance.')
'''))
    c.append(md(r'''
## B3. What exactly is thrown away?

| Lost by keeping one label | Why it matters |
|---|---|
| The vote percentages | A 100% winner and a 34% winner look identical |
| The other emotions | A new minority reading can appear without winning |
| Ties | One label makes a split vote look decisive |
| How far the votes move | A tiny lead change and a complete change both count as one "flip" |
| Direction across all emotions | Several emotions can gain or lose at once |

# Part C: Keep all the votes, then ask the same questions again

From here on, every set of votes is kept as **vote shares** over the 11 emotions (`vec`), and a
change is measured as the **shift** (TV). The texts, people and models are exactly the same as in
Part A; only the way the answers are summarised changes.

## C1. A real example of hidden change
'''))
    c.append(code(EXAMPLE))
    c.append(code(r'''
example(study, 'hidden_top', 'Human', 'human_hidden_change_example')
explain('**Compare with the old method:** the winner is the same on both sides, so the old method records '
        '"no change", yet most of the votes moved.')
'''))
    c.append(md(r'''
## C2. Before any persona: how far apart are the vote shares?

The same two comparisons as A2, now measured as a shift instead of match or no match.
'''))
    c.append(code(r'''
base_vectors = baseline_rows.groupby(['comparison', 'category']).agg(
    comparisons=('text_id', 'size'), mean_TV_pct=('tv', lambda x: 100 * x.mean()),
    mean_cosine=('cosine', 'mean')).reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
for j, comp in enumerate(base_vectors.comparison.unique()):
    q = base_vectors[base_vectors.comparison == comp].set_index('category').reindex(CATS)
    ax.bar(np.arange(3) + (j - .5) * .35, q.mean_TV_pct, .33, label=comp)
    for i, v in enumerate(q.mean_TV_pct):
        ax.text(i + (j - .5) * .35, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
ax.set_xticks(range(3), CAT_NAMES)
ax.set_ylabel('Mean TV × 100')
ax.set_ylim(0, 100)
ax.legend(fontsize=9)
ax.set_title('Vector method: how far apart are the neutral distributions?')
finish(fig, 'vector_baseline_distances')
table(base_vectors, 'vector_baseline_distances', 2)
explain('**What it means:** two groups of readers with no persona already differ by a third to a half of '
        'their votes. This is background disagreement between rater groups (they also differ in who they '
        'are and when they rated), so it is context, not a number to subtract.')
'''))
    c.append(md('## C3. Under a persona, how far do the votes move?'))
    c.append(code(r'''
vector_rates = bar(study, 'tv', 'vector_TV_by_category', 'Vector method: how far do the emotion shares move?',
                   'Mean TV × 100 (probability mass moved)')
vector_total = summary(study, 'tv', False)
table(vector_total, 'vector_overall_TV', 2)
explain('**Compare with A3:** a label change is yes/no; the shift measures *how much* moved, whether or not '
        'the winner changed. These averages still include ordinary rater-to-rater disagreement; the tests '
        'in C5 account for it.')
'''))
    c.append(md(r'''
## C4. What kind of change happened?

Every comparison falls into exactly one of five kinds: no change; the same emotions with
different percentages; new emotions appear; emotions disappear; both. These describe the
**group's** vote distribution, not individual people changing their minds (different people rated
each version).
'''))
    c.append(code(EVENTS))
    c.append(code(r'''
event_counts = events_plot(study)
for src in ['Human', 'LLM']:
    r = event_counts[(event_counts.source == src) & (event_counts.event == 'Same emotions; proportions change')].iloc[0]
    explain(f'**{src}:** {int(r.comparisons):,} of {int(r.denominator):,} comparisons ({r.percent:.1f}%) keep exactly '
            f'the same emotions but change their percentages (in {int(r.texts_with_event)} of 300 texts).')
explain('**Careful with counts:** the five kinds add up to 100% of comparisons, but text counts do not, because '
        'one text can show different kinds under different personas or prompt set-ups.')
'''))
    c.append(md(r'''
## C5. H1 and H2 again, now with all the votes

Exactly the same test as A5 (same comparisons, same chance shuffles, same Holm correction), but
measuring the **shift** instead of "did the label change".

For H2 this test picks up changes that **several models share**. If only one model changes in a
comparison, every chance shuffle gives the same shift, so that comparison cannot count as
evidence. Notebook 2 therefore also reports how often each individual model changed.
'''))
    c.append(code(r'''
vector_h12 = plot_h1_h2(study, 'tv', 'vector_H1_H2')
for hyp in ['H1', 'H2']:
    q = vector_h12[vector_h12.hypothesis == hyp]
    explain(f'**{hyp}, vector method:** {q.reject_05.sum()} of {len(q)} tests are significant after correction.')
explain('**What it means:** with all the votes, human framing effects are detected for author-relevant texts '
        '(both personas) and for one persona on author-independent texts; none for clear texts. The old '
        'method detected none of these. Models are affected under both personas.')
'''))
    c.append(md(r'''
## C6. H3 again: raw, and after allowing for chance

Ambiguous texts have raters who disagree more to begin with, so even with no persona effect their
votes would shift more between versions. `test_h3` therefore reports two versions:
- **Raw:** compare the observed shifts (the main test).
- **Conditional-null referenced:** first subtract what chance alone gives for that same text,
  then compare what is left. This was added after the raw result was known, so it is a
  **sensitivity check**, reported next to the raw test and not instead of it.
'''))
    c.append(code(r'''
vector_h3, vector_h3_text = test_h3(study, 'tv')
table(vector_h3, 'vector_H3_raw_and_referenced')
reference = vector_h3_text.groupby('category')[['tv', 'expected', 'excess']].mean().reindex(CATS) * 100
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for j, col in enumerate(['tv', 'expected']):
    axes[0].bar(np.arange(3) + (j - .5) * .33, reference[col], .31, label=['Observed', 'Conditional-null mean'][j])
axes[0].set_xticks(range(3), CAT_NAMES, rotation=15)
axes[0].set_ylabel('TV × 100')
axes[0].set_title('Much raw TV is expected under the reference')
axes[0].legend(fontsize=9)
axes[1].bar(CAT_NAMES, reference.excess, color='#177e89')
axes[1].tick_params(axis='x', rotation=15)
axes[1].axhline(0, color='grey')
axes[1].set_ylabel('Observed minus expected (pp)')
axes[1].set_title('Reference-adjusted distance is a different quantity')
finish(fig, 'vector_H3_reference')
table(reference.reset_index(), 'vector_H3_category_reference', 2)
for mode in ['Raw', 'Conditional-null referenced']:
    r = vector_h3[vector_h3.analysis == mode].iloc[0]
    explain(f'**{mode}, author-relevant minus author-independent:** {r.difference_pp:.2f} points, '
            f'range [{r.ci_lo:.2f}, {r.ci_hi:.2f}], corrected p = {r.p_holm:.4f}. '
            + ('Significant.' if r.p_holm < .05 else 'Not significant.'))
explain('**What it means:** the main test says author-relevant texts shift more, but the sensitivity check '
        'does not. So the H3 result depends on the reference used and should not be called simply '
        '"supported". "Not significant" also does not prove the groups are equal.')
'''))
    c.append(md(r'''
## C7. H4 again: size and direction, now measured separately

- **Size:** for each text, the average human shift minus the average model shift.
- **Direction** (`direction`): the **shift vector** is persona shares minus neutral shares (for
  example fear +67, anger −33, sadness −33). The cosine compares the human and model shift vectors
  for the same text, persona and prompt set-up: +1 same direction, 0 unrelated, −1 opposite. If
  either side did not move there is no direction, and that comparison is left out (never counted
  as zero). Chance: pair each text's human shifts with a random other text's model shifts.
'''))
    c.append(code(DIRECTION))
    c.append(code(r'''
vector_h4 = h4_size(study, 'tv')
table(vector_h4, 'vector_H4_magnitude')
direction_result, cosines, direction_null = direction(study)
table(direction_result, 'vector_H4_direction')
hv = study.h.groupby('text_id').tv.mean().reindex(study.ids) * 100
lv = study.l.groupby('text_id').tv.mean().reindex(study.ids) * 100
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for cat in CATS:
    mask = np.array([study.category[t] == cat for t in study.ids])
    axes[0].scatter(hv[mask], lv[mask], s=20, alpha=.55, label=CAT_NAME[cat])
axes[0].plot([0, 100], [0, 100], '--', color='grey')
axes[0].set(xlim=(0, 100), ylim=(0, 100), xlabel='Human mean TV × 100', ylabel='Model mean TV × 100',
            title='Below diagonal: humans move further')
axes[0].legend(fontsize=8)
axes[1].hist(cosines[np.isfinite(cosines)], bins=25, color='#177e89', alpha=.8)
axes[1].axvline(direction_result.iloc[0].mean_cosine, color='#d66b35', label='Observed mean')
axes[1].axvline(direction_result.iloc[0].null_mean, color='black', ls='--', label='Shuffled-reference mean')
axes[1].set(xlabel='Delta-vector cosine', ylabel='Defined comparison count',
            title='Direction: alignment varies across comparisons')
axes[1].legend(fontsize=8)
finish(fig, 'vector_H4_magnitude_direction')
r = vector_h4.iloc[0]
dr = direction_result.iloc[0]
explain(f'**Size:** people shift {r.human_minus_llm_pp:.2f} points more than models (range [{r.ci_lo:.2f}, {r.ci_hi:.2f}]). '
        f'**Direction:** average cosine {dr.mean_cosine:.3f} (range [{dr.ci_lo:.3f}, {dr.ci_hi:.3f}]) against about '
        f'{dr.null_mean:.3f} by chance, p = {dr.p_permutation:.4f}; measurable in {int(dr.defined_comparisons):,} of 3,600 '
        'comparisons. Models lean the same way as people more than chance, but only slightly. Notebook 2 checks '
        'whether the size gap survives giving both sides the same number of votes.')
'''))

    # ------------------------------------------------------------------ Part D
    c.append(md(r'''
# Part D: Old method vs vector method

The old method counts label switches; the vector method measures how far and where the votes
move. Their percentages mean different things, so they are compared side by side, never
subtracted.
'''))
    c.append(code(r'''
comparison = old_h12[['hypothesis', 'category', 'side', 'observed_pct', 'p_holm', 'reject_05']].merge(
    vector_h12[['hypothesis', 'category', 'side', 'observed_pct', 'p_holm', 'reject_05']],
    on=['hypothesis', 'category', 'side'], suffixes=('_label', '_vector'), validate='one_to_one')
table(comparison, 'old_vs_vector_H1_H2')
n_h1_vec = int(vector_h12.query("hypothesis == 'H1'").reject_05.sum())
n_h2_vec = int(vector_h12.query("hypothesis == 'H2'").reject_05.sum())
final_summary = pd.DataFrame([
    {'question': 'H1', 'old': 'Selected-label switch frequency', 'vector': 'Full-distribution TV against conditional null',
     'conclusion': f'{n_h1_vec}/6 vector tests reject; interpret by category and side; controls not proven null'},
    {'question': 'H2', 'old': 'Selected ensemble-label switches', 'vector': 'Net redistribution under paired model swaps',
     'conclusion': f'{n_h2_vec}/2 vector tests reject; coordination and individual changes remain distinct'},
    {'question': 'H3', 'old': f'Raw AR−AI label-rate p={old_h3_raw.iloc[0].p_holm:.4f}',
     'vector': f'Raw AR−AI p={vector_h3.iloc[0].p_holm:.4f}; referenced p={vector_h3.iloc[3].p_holm:.4f}',
     'conclusion': 'Evaluate the direct contrast under both references; raw ordering alone is insufficient'},
    {'question': 'H4 magnitude', 'old': f'Frequency gap {old_h4.iloc[0].human_minus_llm_pp:.2f}pp',
     'vector': f'TV gap {vector_h4.iloc[0].human_minus_llm_pp:.2f}pp',
     'conclusion': 'Different constructs; count and sampling-process sensitivity still matters'},
    {'question': 'H4 direction', 'old': 'Exact selected-label transition proxy',
     'vector': f'Delta cosine {direction_result.iloc[0].mean_cosine:.3f}, with explicit coverage',
     'conclusion': 'Partial resemblance is conditional; no averaging away opposing personas'},
    {'question': 'What was hidden?', 'old': 'A stable label is recorded as no change',
     'vector': 'Separates unchanged, proportion changes, appearances and disappearances',
     'conclusion': 'The information-loss and event tables provide the direct counts'}])
table(final_summary, 'final_method_comparison')
plain_verdicts = pd.DataFrame([
    {'hypothesis': 'H1 — humans', 'old_method': 'No adjusted selected-label tests reject in this run',
     'vector_method': 'Qualified partial support: author-relevant A and B, and author-independent A reject',
     'meaning': 'Full distributions detect differences the selected-label statistic misses; control non-rejection is not absence'},
    {'hypothesis': 'H2 — models', 'old_method': 'Both paired selected-label tests reject',
     'vector_method': 'Both paired-TV coordination tests reject',
     'meaning': 'Conditional support; single generations and joint exchangeability limit attribution'},
    {'hypothesis': 'H3 — author relevance', 'old_method': 'Direct AR−AI label-frequency contrast does not reject',
     'vector_method': 'Raw contrast rejects; conditional-null-referenced contrast does not',
     'meaning': 'The stronger category-sensitivity claim is not robust to the reference'},
    {'hypothesis': 'H4 — resemblance', 'old_method': 'Frequency differs; categorical transition proxy exceeds shuffled pairing',
     'vector_method': 'Larger human TV; small positive directional correspondence',
     'meaning': 'Qualified partial resemblance; count sensitivity follows in Notebook 2'}])
# These checks stop the notebook if a future data change would make the verdict text above wrong.
assert old_h12[old_h12.hypothesis == 'H1'].reject_05.sum() == 0
assert vector_h12[vector_h12.hypothesis == 'H1'].reject_05.sum() == 3
assert old_h12[old_h12.hypothesis == 'H2'].reject_05.all() and vector_h12[vector_h12.hypothesis == 'H2'].reject_05.all()
assert old_h3_raw.iloc[0].p_holm >= .05 and vector_h3.iloc[0].p_holm < .05 and vector_h3.iloc[3].p_holm >= .05
assert old_direction.iloc[0].p_permutation < .05 and direction_result.iloc[0].p_permutation < .05
assert old_h4.iloc[0].ci_lo > 0 and vector_h4.iloc[0].ci_lo > 0
table(plain_verdicts, 'plain_language_hypothesis_verdicts')
'''))
    c.append(md(r'''
## D2. What this notebook shows

| | Old method (one label) | Vector method (all votes) | Verdict |
|---|---|---|---|
| **H1** humans | 0 of 6 tests significant | 3 of 6: author-relevant A and B, author-independent A | Partially supported |
| **H2** models | 2 of 2 | 2 of 2 | Supported at the ensemble level |
| **H3** | no difference detected | raw test significant; chance-adjusted check not | Weak / not robust |
| **H4** | people change far more often; same switch more than chance | people shift far more; direction slightly aligned | Qualified support |

**The main point.** Keeping all the votes does two things. It **detects** human framing effects
that a one-label analysis misses, and it **shows** where an apparently strong result (H3) is not
robust once chance is taken into account.

**Limits to keep in mind**
- Each human version was rated by different people; each model answered both versions itself.
- Each model gave one answer per condition, so a model's own randomness is not measured.
- People rated about 30 texts each, but who rated what was not kept.
- The author-relevant group was defined with a pilot persona test.
- The extra analyses (H3 sensitivity check, Notebook 2) were added after the main tests.
  "Not significant" never proves "no effect".

**Next:** Notebook 2 looks at *what* moved inside the vote distributions.
'''))
    return c


# =============================================================================================
# Notebook 02
# =============================================================================================


def notebook_02():
    c = []
    c.append(md(r'''
# Notebook 2: What changed inside the vote distributions?

Notebook 1 answered the hypotheses. This notebook looks closer at **what** moved.

**Route:** the five kinds of change → how many texts → same emotions with new percentages →
which emotions appear or disappear → by group → humans vs models → average gain or loss per emotion
→ spread and vote counts → individual models → prompt set-ups → generic author vs specific persona
→ the H3 check for the old method.

**About the code.** The first section repeats the calculation code from Notebook 1, so this
notebook runs on its own. The code is explained in Notebook 1; you can skip it here.

As in Notebook 1, every model comparison uses only the models that answered validly on both
sides, and every percentage is shown with its count.
'''))
    c.append(md('## Setup: the same calculation code as Notebook 1'))
    c.append(code(SETUP))
    c.append(code(SETTINGS))
    c.append(code(BUILDING_BLOCKS))
    c.append(code(LOADING))
    c.append(code(AVERAGES))
    c.append(code(TESTS))
    c.append(code(H3))
    c.append(code(EVENTS))
    c.append(code(EXAMPLE))
    c.append(md(r'''
Two more pieces of code are used only in this notebook: which emotions appear or disappear
(`emotion_events`) with the Benjamini-Hochberg correction for many tests (`bh_fdr`), and the
equal-vote-count check (`count_sensitivity`) with the prompt test (`prompt_test`).
'''))
    c.append(code(EMOTION_EVENTS))
    c.append(code(COUNTS))
    c.append(code(LOAD_RUN))

    c.append(md(r'''
## 1. How many comparisons show each kind of change?

The five kinds are exclusive: no change; same emotions with different percentages; emotions appear
only; emotions disappear only; both. A **comparison** is one text, neutral vs one persona (and, for
models, one prompt set-up).
'''))
    c.append(code(r'''
event_counts = events_plot(study)
assert event_counts.groupby('source').percent.sum().round(8).eq(100).all()
explain('**Read the plot:** bars give percentages; the fractions give exact counts. A new emotion does not '
        'have to become the winner. Models have more comparisons per text (12 vs 2), not more texts.')
'''))
    c.append(md(r'''
## 2. In how many different texts does each event happen?

"Text incidence" = the event happens in at least one comparison of that text. One text can show
several events, so these counts overlap and must not be added up. People have 2 chances per text
and models 12, so the second table also counts model texts within each prompt set-up (2 chances
per text).
'''))
    c.append(code(r'''
rows = []
for src, d in [('Human', study.h), ('LLM', study.l)]:
    for event in ['gain_any', 'loss_any', 'same_support_change', 'hidden_top', 'strict_proportion']:
        for cat in CATS:
            q = d[d.category == cat]
            any_text = q.groupby('text_id')[event].any()
            rows.append({'source': src, 'category': CAT_NAME[cat], 'event': event,
                         'texts_any_event': int(any_text.sum()), 'text_denominator': 100,
                         'comparisons_with_event': int(q[event].sum()), 'comparison_denominator': len(q),
                         'opportunities_per_text': 2 if src == 'Human' else 12})
incidence = pd.DataFrame(rows)
table(incidence, 'event_text_incidence')
config_incidence = []
for config, d in study.l.groupby('config'):
    for event in ['gain_any', 'loss_any', 'same_support_change', 'hidden_top', 'strict_proportion']:
        config_incidence.append({'configuration': config, 'event': event,
                                 'texts_with_event': int(d.groupby('text_id')[event].any().sum()), 'out_of': 300})
config_incidence = pd.DataFrame(config_incidence)
table(config_incidence, 'model_event_text_incidence_by_config')
table(config_incidence.groupby('event').texts_with_event.agg(['mean', 'min', 'max']).reset_index(),
      'model_event_incidence_config_summary', 2)
explain('**Event names:** gain_any = at least one emotion appears; loss_any = at least one disappears; '
        'same_support_change = same emotions, different percentages; hidden_top = same top emotion(s) but '
        'the votes moved; strict_proportion = same emotions AND same top, only the percentages changed.')
'''))
    c.append(md(r'''
## 3. Same emotions, different percentages

Two definitions:
- **Broad:** exactly the same emotions are present, their percentages change, and the winner may
  change.
- **Strict:** the same emotions **and** the same top emotion(s); only the percentages change. Even an
  analysis that keeps ties would miss this.

The two examples are the largest shifts of the strict kind, one for people and one for models.
'''))
    c.append(code(r'''
rows = []
for src, d in [('Human', study.h), ('LLM', study.l)]:
    for event in ['same_support_change', 'strict_proportion']:
        n = int(d[event].sum())
        rows.append({'source': src, 'definition': event, 'comparisons': n, 'out_of': len(d),
                     'percent': 100 * n / len(d), 'unique_texts': d.loc[d[event], 'text_id'].nunique()})
proportions = pd.DataFrame(rows)
table(proportions, 'proportion_only_definitions', 2)
example(study, 'strict_proportion', 'Human', 'strict_proportions_human')
example(study, 'strict_proportion', 'LLM', 'strict_proportions_llm')
explain('**What it means:** the same emotions stay, but the balance changes. For people this is rare; for '
        'models it is common, so models often react by re-weighting rather than switching.')
'''))
    c.append(md(r'''
## 4. Which emotions appear and disappear?

"Appears" = no votes in the neutral version, some under the persona; "disappears" = the reverse.
Several emotions can do this in one comparison, so the percentages do not add up to 100. This is
about the group's votes: it does not mean an individual person switched emotions.
'''))
    c.append(code(r'''
emotion_event_rates = emotion_events(study)
for src in ['Human', 'LLM']:
    r = emotion_event_rates[emotion_event_rates.source == src].sort_values('appears_pct', ascending=False).iloc[0]
    explain(f'**{src}:** the emotion that appears most often is {r.emotion} ({r.appears_pct:.1f}% of comparisons). '
            'This is a description, not a tested result.')
'''))
    c.append(md(r'''
## 5. Does this depend on the text group?

Share of comparisons where at least one emotion appears, and where at least one disappears, by
group. The error bars resample texts. These describe a pattern; H3 itself was tested in
Notebook 1.
'''))
    c.append(code(r'''
gains = bar(study, 'gain_any', 'new_emotions_by_category', 'At least one emotion appears: where does it happen?',
            'Comparisons with an observed appearance (%)')
losses = bar(study, 'loss_any', 'lost_emotions_by_category', 'At least one emotion disappears: where does it happen?',
             'Comparisons with an observed disappearance (%)')
explain('**Read together:** emotions appear *and* disappear most on author-relevant texts, so the mix of '
        'readings changes rather than simply growing. Human rates are higher partly because each version had '
        'different raters and only about 3 votes: one different rater is enough to add an emotion.')
'''))
    c.append(md(r'''
## 6. Do people and models show the same kind of change?

Each model comparison is matched with the human comparison for the same text and persona. Rows =
what people did, columns = what models did; each row adds up to 100%. This is descriptive: no
statistical test of agreement is run on this table.
'''))
    c.append(code(r'''
j = study.l[['text_id', 'side', 'change']].merge(study.h[['text_id', 'side', 'change']], on=['text_id', 'side'],
                                                 suffixes=('_llm', '_human'), validate='many_to_one')
cross = pd.crosstab(j.change_human, j.change_llm).reindex(index=EVENT_TYPES, columns=EVENT_TYPES, fill_value=0)
percent = cross.div(cross.sum(axis=1), axis=0).fillna(0) * 100
short = ['No change', 'Same support', 'Appear only', 'Disappear only', 'Appear + disappear']
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(percent, cmap='Blues', vmin=0, vmax=100)
for i in range(5):
    for k in range(5):
        ax.text(k, i, f'{percent.iloc[i, k]:.1f}%', ha='center', va='center',
                color='white' if percent.iloc[i, k] > 50 else 'black')
ax.set_xticks(range(5), short, rotation=25, ha='right')
ax.set_yticks(range(5), short)
ax.set_xlabel('LLM event')
ax.set_ylabel('Human event')
ax.set_title('Same text and persona: how event types correspond')
fig.colorbar(im, ax=ax, label='Within-human-row percentage')
finish(fig, 'human_model_event_correspondence')
table(cross.reset_index(), 'human_model_event_counts')
table(cross.sum(axis=1).rename('matched_comparisons').reset_index(), 'human_event_denominators')
explain('**What it shows:** whatever people did, the most common model response is "no change". Any '
        'correspondence looks weak, and it is not formally tested here.')
'''))
    c.append(md(r'''
## 7. Which emotions gain or lose share on average?

For each emotion: persona share minus neutral share, averaged per text and then over texts. Red =
gained, blue = lost. Averages can hide opposite shifts that cancel out.

**Tests (exploratory):** for each of the 66 cells (2 sources × 3 groups × 11 emotions), flip the
sign of each text's change at random 10,000 times and see how often the average is at least as far
from zero as the real one. With 66 tests, the Benjamini-Hochberg correction (`bh_fdr`) is applied.
'''))
    c.append(code(r'''
emotion_rows = []
for src, d in [('Human', study.h), ('LLM', study.l)]:
    expanded = pd.DataFrame(np.stack(d.delta), columns=E)
    expanded['text_id'] = d.text_id.to_numpy()
    expanded['category'] = d.category.to_numpy()
    t = expanded.groupby(['text_id', 'category'])[E].mean().reset_index()
    for cat in CATS:
        vals = t.loc[t.category == cat, E].to_numpy()
        rng = np.random.default_rng(808)
        null = rng.choice([-1, 1], size=(N_RANDOM, len(vals))) @ vals / len(vals)
        for k, e in enumerate(E):
            obs = vals[:, k].mean()
            p = (1 + (np.abs(null[:, k]) >= abs(obs) - 1e-12).sum()) / (N_RANDOM + 1)
            emotion_rows.append({'source': src, 'category': CAT_NAME[cat], 'emotion': e,
                                 'mean_change_pp': 100 * obs, 'p_raw': p})
emotion_tests = pd.DataFrame(emotion_rows)
emotion_tests['p_BH'] = bh_fdr(emotion_tests.p_raw)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
limit = max(abs(emotion_tests.mean_change_pp).max(), 1)
for ax, src in zip(axes, ['Human', 'LLM']):
    grid = emotion_tests[emotion_tests.source == src].pivot(index='category', columns='emotion',
                                                           values='mean_change_pp').reindex(index=CAT_NAMES, columns=E)
    im = ax.imshow(grid, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(11), E, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(3), CAT_NAMES, fontsize=9)
    ax.set_title(src)
    for i in range(3):
        for k in range(11):
            ax.text(k, i, f'{grid.iloc[i, k]:.1f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, label='Mean signed change (pp)', shrink=.7)
finish(fig, 'emotion_share_changes')
table(emotion_tests, 'exploratory_per_emotion_tests', 4)
significant = emotion_tests[emotion_tests.p_BH < .05]
explain(f'**Tests:** {len(significant)} of 66 changes are significant after correction: '
        + '; '.join(f'{r.source} {r.category.lower()} {r.emotion} {r.mean_change_pp:+.1f}'
                    for _, r in significant.iterrows())
        + '. The rest are too small to confirm. These are exploratory tests.')
'''))
    c.append(md(r'''
## 8. Do the votes spread out, and does the number of votes matter?

**Spread (entropy):** higher = votes spread over more emotions. It is not the same as how far the
votes moved.

**Equal vote counts:** people have about 3 votes per version and models up to 7, and fewer votes
can make shifts look bigger. So both sides are cut to **3 votes**: 3 random human votes per version,
and the **same 3 random models** on both sides of a model comparison, repeated 200 times. This
checks whether unequal counts could explain the gap. It cannot turn the human design (different
people per version) into the model design (same model on both versions).
'''))
    c.append(code(r'''
standardised, standardised_summary = count_sensitivity(study)
table(standardised_summary, 'count_standardisation_summary', 4)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, quantity, scale, ylabel in [(axes[0], 'tv', 100, 'Mean TV × 100'),
                                    (axes[1], 'entropy', 1, 'Mean persona − neutral entropy (nats)')]:
    for j, src in enumerate(['Human', 'LLM']):
        q = standardised_summary[standardised_summary.source == src].set_index('quantity').loc[
            [quantity + '_raw', quantity + '_standardised']]
        x = np.arange(2) + (j - .5) * .34
        ax.bar(x, q['mean'] * scale, .32, label=src, color=COLORS[src])
        ax.errorbar(x, q['mean'] * scale, yerr=[(q['mean'] - q.ci_lo) * scale, (q.ci_hi - q['mean']) * scale],
                    fmt='none', color='black', capsize=3)
    ax.set_xticks([0, 1], ['Original counts', 'Three votes each'])
    ax.set_ylabel(ylabel)
    ax.axhline(0, color='grey', lw=.8)
    ax.legend()
    ax.set_title('Distance' if quantity == 'tv' else 'Dispersion')
finish(fig, 'count_standardisation_TV_entropy')
per_text = standardised.groupby(['source', 'text_id'])[['tv_raw', 'tv_standardised']].mean().unstack('source')
gap_rows = []
for quantity in ['tv_raw', 'tv_standardised']:
    p, lo, hi = boot_mean(per_text[quantity]['Human'] - per_text[quantity]['LLM'])
    gap_rows.append({'analysis': quantity, 'mean_gap_pp': 100 * p, 'ci_lo': 100 * lo, 'ci_hi': 100 * hi})
gap = pd.DataFrame(gap_rows)
table(gap, 'count_standardised_magnitude_gap', 3)
explain(f'**Size:** the human−model gap is {gap.iloc[0].mean_gap_pp:.2f} points with the original counts and about '
        f'{gap.iloc[1].mean_gap_pp:.0f} points with 3 votes each, so unequal vote counts do not plausibly explain it. '
        'This check does not identify what causes the gap. **Spread:** human votes spread out under a persona '
        'with both counts; for models the error bar touches zero, so no increase in spread is detected.')
'''))
    c.append(md(r'''
## 9. Does every model change its answers?

Each model gave one answer per version. This counts how often a model's own two answers differ
(only where both are valid). It is separate from the ensemble test in Notebook 1, which picks up
changes several models share.
'''))
    c.append(code(r'''
ind = study.individual.groupby('model').agg(valid_pairs=('changed', 'size'), changed_pairs=('changed', 'sum'),
                                             change_rate=('changed', 'mean')).sort_values('change_rate')
ind['change_pct'] = 100 * ind.change_rate
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(ind.index, ind.change_pct, color='#d66b35')
for i, (_, r) in enumerate(ind.iterrows()):
    ax.text(r.change_pct + .3, i, f'{r.change_pct:.1f}% ({int(r.changed_pairs):,}/{int(r.valid_pairs):,})',
            va='center', fontsize=9)
ax.set_xlim(0, ind.change_pct.max() + 14)
ax.set_xlabel('Valid model pairs with different answers (%)')
ax.set_title('Individual recorded changes')
finish(fig, 'individual_model_changes')
table(ind.reset_index(), 'individual_model_change_rates', 2)
changed = study.individual.groupby(['text_id', 'side', 'config']).changed.sum()
coordination = pd.DataFrame({'discordant_models': changed.value_counts().sort_index().index,
                             'comparisons': changed.value_counts().sort_index().values})
table(coordination, 'ensemble_discordance_counts')
explain(f'**What it shows:** {study.individual.changed.sum():,} of {len(study.individual):,} model answer pairs '
        f'changed. In {(changed <= 1).mean() * 100:.1f}% of comparisons at most one model changed; those cannot '
        'count as evidence in the ensemble test, which is why these individual rates are shown too. They are '
        'descriptive: no test compared the models, so no model is shown to be more sensitive than another.')
'''))
    c.append(md(r'''
## 10. Does the prompt set-up matter?

For each text and prompt set-up, average the two persona comparisons. The Friedman test
(`prompt_test`) ranks the 6 set-ups within each text; if the set-up did not matter, all would get
similar average ranks. Left: old method; right: vector method.
'''))
    c.append(code(r'''
prompt_results = []
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric in zip(axes, ['flip', 'tv']):
    values, result = prompt_test(study, metric)
    prompt_results.append(result)
    groups = [values[c].to_numpy() * 100 for c in values]
    ax.boxplot(groups, showfliers=False)
    ax.set_xticks(range(1, 7), values.columns, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Text-level label-change rate (%)' if metric == 'flip' else 'Text-level mean TV × 100')
    ax.set_title(('Old: selected labels' if metric == 'flip' else 'New: vectors') + f' — Friedman p={result.iloc[0].p_value:.4f}')
finish(fig, 'prompt_old_and_vector')
table(pd.concat(prompt_results, ignore_index=True), 'prompt_comparison_tests')
explain('**What it means:** no statistically detectable difference between the six set-ups, so the model effect '
        'does not appear to depend on one prompt. "Not detected" does not prove the set-ups are identical.')
'''))
    c.append(md(r'''
## 11. Is it the specific persona, or just naming any author? (models only)

Three steps: neutral → "written by a person" (generic), generic → specific persona, and neutral →
persona. To keep the three comparable, only models valid in **all three** versions are used.
The three distances **do not add up**, so they are not parts of one total.
'''))
    c.append(code(r'''
control = []
for tid in study.ids:
    for side in ['a', 'b']:
        for pv, lf in study.configs:
            n = study.mvotes[(tid, 'neutral', pv, lf)]
            g = study.mvotes[(tid, 'generic_human', pv, lf)]
            p = study.mvotes[(tid, 'persona_' + side, pv, lf)]
            common = sorted(n.keys() & g.keys() & p.keys())
            assert len(common) >= 3
            for label, a, b in [('Neutral → generic', n, g), ('Generic → persona', g, p), ('Neutral → persona', n, p)]:
                pa = vec([a[m] for m in common])
                pb = vec([b[m] for m in common])
                control.append({'text_id': tid, 'side': side, 'config': pv + '/' + lf, 'contrast': label,
                                'n_models': len(common), 'flip': int(pa.argmax() != pb.argmax()),
                                'tv': abs(pa - pb).sum() / 2})
control = pd.DataFrame(control)
rows = []
for (side, contrast), q in control.groupby(['side', 'contrast']):
    for metric in ['flip', 'tv']:
        point, lo, hi = boot_mean(q.groupby('text_id')[metric].mean())
        rows.append({'side': side, 'contrast': contrast, 'metric': metric, 'mean_pct': 100 * point,
                     'ci_lo': 100 * lo, 'ci_hi': 100 * hi, 'cells': len(q), 'min_common_models': q.n_models.min()})
control_summary = pd.DataFrame(rows)
table(control_summary, 'generic_author_three_way_control', 2)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
order = ['Neutral → generic', 'Generic → persona', 'Neutral → persona']
for ax, metric in zip(axes, ['flip', 'tv']):
    for j, side in enumerate(['a', 'b']):
        q = control_summary[(control_summary.side == side) & (control_summary.metric == metric)].set_index('contrast').reindex(order)
        x = np.arange(3) + (j - .5) * .34
        ax.bar(x, q.mean_pct, .32, label='Persona ' + side.upper())
        ax.errorbar(x, q.mean_pct, yerr=[q.mean_pct - q.ci_lo, q.ci_hi - q.mean_pct], fmt='none', color='black', capsize=3)
    ax.set_xticks(range(3), order, rotation=20, ha='right')
    ax.set_ylabel('Selected-label changes (%)' if metric == 'flip' else 'Mean TV × 100')
    ax.set_title('Old method' if metric == 'flip' else 'Vector method')
    ax.legend()
finish(fig, 'generic_author_control_comparison')
explain('**What it means:** a generic author line is associated with a small shift; going from the generic author '
        'to a specific persona is associated with a shift about twice as large. So models are sensitive to *who* '
        'the author is, not only to *whether* one is named. The distances are associations, not parts of a total.')
'''))
    c.append(md(r'''
## 12. Does the old method's H3 result change when chance is allowed for?

The same sensitivity check as Notebook 1 C6, applied to the old label-change measure.
'''))
    c.append(code(r'''
old_sensitivity, _ = test_h3(study, 'flip')
table(old_sensitivity, 'old_H3_conditional_reference_sensitivity')
explain('**What it means:** the old method is also affected by chance variation. Neither the raw nor the '
        'adjusted old-method comparison shows author-relevant texts changing more than author-independent ones.')
'''))
    c.append(md(r'''
## 13. What these data are, and what this notebook adds

The data are **vote distributions under different author framings**. Human vectors summarise
different readers; model vectors summarise different systems. They are not repeated measurements of
one person's feelings, and the model ensemble is not one model's probabilities.

This notebook separates:
- a change of the winning label, and a change of the full set of tied winners;
- the same emotions with new percentages;
- emotions appearing and disappearing;
- how far and in which direction the votes move;
- spread, prompt set-ups and individual models;
- raw differences and what remains after allowing for chance.

**In one sentence:** a winning label only tells you the winner; keeping all the votes shows where
they go, which reveals changes the label hides and also shows where a result is not robust.

**What these notebooks do not do:** they do not recover who rated what, invent repeated model
answers, split the generic-author control into additive parts, or guarantee that every hypothesis
holds.
'''))
    return c


# =============================================================================================
# Notebook 00: patch text and setup in place (its analysis code is already inline)
# =============================================================================================

NB00_SETUP = r'''
# ---- Setup: libraries, folders, the rule's settings and plot style (all defined here) ----
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# The project folder is the one that contains data/.
ROOT = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
BASE = ROOT / 'data'
CORPUS = BASE / 'corpus'

# The rule's settings. The same values are in scripts/classify_ambiguity_pools.py, the script
# that originally built the sample.
THRESHOLD = 0.8                        # at least 4 of 5 readers must agree
SEED = 42                              # fixed random seed for trimming/replacing texts
OFF_LABELS = {'boredom', 'no-emotion'} # labels outside the 11 emotions
EXAMPLE_IDS = {468, 393, 417}          # texts used as prompt examples, never sampled

CATEGORY_EMOTIONS = ('anger', 'disgust', 'fear', 'guilt', 'joy', 'pride',
                     'relief', 'sadness', 'shame', 'surprise', 'trust')
CATEGORY_LABELS = ['Unambiguous', 'Author-Independent Ambiguous', 'Author-Relevant Ambiguous']
COLORS = {'Unambiguous': '#4C72B0', 'Author-Independent Ambiguous': '#DD8452',
          'Author-Relevant Ambiguous': '#C44E52', 'human': '#4C72B0', 'llm': '#DD8452'}

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 11.5, 'axes.titlesize': 13,
                     'axes.titleweight': 'bold', 'axes.labelsize': 11.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25,
                     'grid.linestyle': '-', 'legend.frameon': False, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white'})
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 140)

print(f"THRESHOLD = {THRESHOLD}   SEED = {SEED}")
print(f"Excluded labels: {sorted(OFF_LABELS)}")
print(f"Reserved prompt-example texts (never sampled): {sorted(EXAMPLE_IDS)}")
'''

NB00_MARKDOWN = {
    0: r'''
# Notebook 0: How the 300 texts were chosen

This notebook tests no hypothesis. It shows how the texts were picked and split into three groups,
because every later result depends on these choices.

**In short**
1. Start from 1,200 texts that each have the author's own emotion and 5 readers' emotions.
2. Drop texts labelled *boredom* or *no-emotion* (not among the 11 emotions).
3. **Unambiguous** = at least 4 of 5 readers agree *and* agree with the author.
4. For the rest: if two personas in a pilot test gave different emotions → **author-relevant**,
   otherwise → **author-independent**.
5. Keep 100 texts per group.

**All code is in this notebook.** Read the text cells first; the code can be skipped on a first read.
''',
    2: r'''
## 1. The corpus

crowd-enVent (Troiano et al., 2023): short descriptions of emotional experiences. For 1,200 of them
there are **two layers**: the emotion the author says they felt, and the emotion each of 5
independent readers picked (the readers never saw the author's answer). That makes it possible to
measure disagreement between author and readers, and among readers.
''',
    5: r'''
**What the two charts show.** Left: what authors meant. Right: what most readers picked. The shapes
differ, so readers and authors often disagree before any persona is added. The corpus also
contains *boredom* and *no-emotion*, which are not among the 11 emotions; those texts are removed
in Step 0 below.
''',
    6: r'''
## 2. Measuring disagreement

Two numbers per text:
- **Does the reader majority match the author?** (`author_matches_readers`). A mismatch does not mean
  the text failed; readers may see another reasonable reading, which is what this thesis studies.
- **How many of the 5 readers chose the most popular emotion?** (`majority_share`; despite the name
  it is the share of the *top* emotion). Only five values are possible:
  - 0.2: all five chose different emotions;
  - 0.4: two readers agree (possibly a tie);
  - 0.6, 0.8, 1.0: 3, 4 or 5 readers agree. **3 of 5 is already a majority.**

The rule uses **0.8** (4 of 5 readers).
''',
    8: r'''
### 2.1 Checking the stored numbers against the raw votes

The two numbers above come from a prepared file. Here they are rebuilt from the 6,000 raw reader
votes to check they are right. This covers all 1,200 candidate texts, not only the 300 sampled.
''',
    10: r'''
**Can the tie-break change anything?** For 134 texts the top was a tie, and the file picked one of
the tied emotions without recording how.
- It **cannot** change which texts count as unambiguous: 4 or 5 of 5 readers agreeing can never be
  a tie.
- It **can** change whether a text is eligible at all: in a tie between, say, anger and boredom,
  picking boredom removes the text. The next cell counts those texts and checks that none of them
  is in the final sample.
''',
    12: r'''
**Key point.** Most texts do have a majority (usually 3 of 5), but only about one text in five is
unanimous, and some have no majority at all. Some disagreement is normal, which is why the later
notebooks keep every vote instead of one winning label.
''',
    13: r'''
## 3. The grouping rule

Applied in this order:
- **Step 0:** drop texts whose author label or reader-majority label is *boredom* or *no-emotion*.
- **Step 1:** **Unambiguous** if at least 4 of 5 readers agree *and* they agree with the author.
- **Step 2:** for the rest, two personas were tried in a pilot. Different emotions →
  **Author-relevant**; the same → **Author-independent**.

**Why this order?** A text readers already agree on stays unambiguous even if some persona could
flip it. That keeps the control group clean. It does not remove one limitation: the split between
the two ambiguous groups comes entirely from the pilot persona test, so H3 later asks whether that
pilot grouping predicts new data (a *transfer* test).
''',
    15: r'''
### 3.1 What Step 0 does not remove

Step 0 looks at two labels per text (the author's and the reader majority), not at every single
reader vote. So a sampled text can still contain a stray *no-emotion* vote. The next cell counts
them; they are left out when vote shares are built.
''',
    17: r'''
### 3.2 What if the threshold were different?

0.6 = 3 of 5 readers, 0.8 = 4 of 5 (used), 1.0 = all 5. The table checks that each option still
leaves at least 100 texts per group. This shows availability; it does not prove 0.8 is the best
possible choice.
''',
    19: r'''
## 4. The final sample

The sample was **not** made by one fresh random draw. An earlier sample was kept and updated:
1. keep the existing `sampled_texts.csv`;
2. re-classify every text with the rule above;
3. if a group has more than 100 texts, drop the extra ones (fixed seed 42);
4. fill any gap with unused eligible texts, never using the 3 texts reserved as prompt examples;
5. check: 300 texts, 100 per group, no duplicates.

So the **rule** is reproducible, but the exact list of texts depends on the earlier sample.
`sampled_texts.csv` is the official record of which texts were used.
''',
    22: r'''
### 4.1 What the persona pairs look like

Each text has two personas and a reason for the pairing. The pilot emotion under each persona, and
whether the two differ (`flip`), decided Step 2.
''',
    24: r'''
**Keep in mind for H3.** Author-relevant texts have a flip rate of 100% because that is how they
were defined; author-independent texts have 0%. H3 asks whether this pilot grouping predicts
effects in the *new* human and model data. That is a fair question, but it is a transfer test, not
an independent discovery.
''',
    25: r'''
## 5. Summary

| Setting | Value |
|---|---|
| Unambiguous threshold | at least 4 of 5 readers agree **and** match the author (0.8) |
| Order of the rule | threshold first, then the pilot flip test |
| Excluded labels | boredom, no-emotion |
| Reserved prompt examples | 3 texts, never sampled |
| Random seed | 42 |
| Final sample | 300 texts, 100 per group (`sampled_texts.csv`) |

Notebooks 1 and 2 take `sampled_texts.csv` as given.
''',
}


def patch_notebook_00():
    path = NB_DIR / '00_corpus_and_sampling.ipynb'
    nb = nbformat.read(path, as_version=4)
    assert len(nb.cells) == 26, 'unexpected notebook 00 structure'
    assert nb.cells[1].cell_type == 'code'
    nb.cells[1] = code(NB00_SETUP)
    for index, text in NB00_MARKDOWN.items():
        assert nb.cells[index].cell_type == 'markdown', index
        nb.cells[index] = md(text)
    nbformat.write(nb, path)


def write(cells, name):
    nb = new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'}})
    nbformat.validate(nb)
    nbformat.write(nb, NB_DIR / name)


if __name__ == '__main__':
    patch_notebook_00()
    write(notebook_01(), '01_majority_labels_then_vectors.ipynb')
    write(notebook_02(), '02_understanding_distribution_changes.ipynb')
    print('Built notebooks 00 (patched), 01 and 02. Now run: python scripts/execute_verified_notebooks.py')
