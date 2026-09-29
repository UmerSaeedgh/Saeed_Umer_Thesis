"""
Shared plotting style + statistical-annotation helpers for the narrative notebooks. Import and
call setup_style() once per notebook. Every helper here draws directly onto a given Axes so it
composes with normal matplotlib code rather than replacing it.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CATEGORY_ORDER = ['unambiguous', 'author_independent', 'author_relevant']
CATEGORY_LABELS = ['Unambiguous', 'Author-Independent Ambiguous', 'Author-Relevant Ambiguous']
COLORS = {
    'Unambiguous': '#4C72B0',
    'Author-Independent Ambiguous': '#DD8452',
    'Author-Relevant Ambiguous': '#C44E52',
    'unambiguous': '#4C72B0',
    'author_independent': '#DD8452',
    'author_relevant': '#C44E52',
    'human': '#4C72B0',
    'llm': '#DD8452',
    'Human': '#4C72B0',
    'LLM': '#DD8452',
}


def setup_style():
    plt.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'font.size': 11.5,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '-',
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def sig_stars(p):
    if p is None or np.isnan(p):
        return 'n/a'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def sig_bracket(ax, x1, x2, y=None, p=None, h=None, text=None, fontsize=10, headroom=0.22):
    """Draws a significance bracket between two x-positions, labeled with stars (or custom text)
    and the exact p-value beneath the stars. Expands the axes' y-limit by `headroom` (fraction of
    the current data range) FIRST so the bracket and its label never collide with the axes title
    -- call this after all data is plotted but it manages its own vertical space, it does not
    borrow the title's."""
    lo, hi = ax.get_ylim()
    data_range = hi - lo
    new_hi = hi + data_range * headroom
    ax.set_ylim(lo, new_hi)
    if y is None:
        y = hi + data_range * 0.03
    if h is None:
        h = data_range * 0.025
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c='black', clip_on=False)
    label = text if text is not None else f"{sig_stars(p)}"
    ax.text((x1 + x2) / 2, y + h * 1.3, label, ha='center', va='bottom', fontsize=fontsize, fontweight='bold', clip_on=False)


def stat_box(ax, lines, loc='upper left', fontsize=9.5):
    """Puts a small boxed annotation of key stats (e.g. n, p, effect size) on the axes."""
    text = "\n".join(lines)
    locs = {
        'upper left': (0.02, 0.98, 'top', 'left'),
        'upper right': (0.98, 0.98, 'top', 'right'),
        'lower left': (0.02, 0.02, 'bottom', 'left'),
        'lower right': (0.98, 0.02, 'bottom', 'right'),
    }
    x, y, va, ha = locs.get(loc, locs['upper left'])
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize, va=va, ha=ha,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#888888', alpha=0.9))


def _n_unique_pooled(data, tol=1e-9):
    pooled = np.concatenate([np.asarray(d) for d in data if len(d) > 0]) if any(len(d) for d in data) else np.array([])
    if len(pooled) == 0:
        return 0
    pooled_sorted = np.sort(pooled)
    n_unique = 1 + np.sum(np.diff(pooled_sorted) > tol)
    return int(n_unique)


def violin_box_strip(ax, data, labels, colors=None, ylabel=None, title=None, jitter=0.06, seed=42,
                      title_pad=14, max_discrete_for_violin=8):
    """A violin (distribution shape) + slim boxplot (median/IQR) + jittered strip (raw points)
    combination -- far more informative than a bare boxplot. AUTO-DEGRADES to box+strip (no
    violin) when the pooled data has very few distinct values (<= max_discrete_for_violin): with
    small-n vote fractions (e.g. n=3 raters -> only 3 possible max-probability values), a violin's
    KDE smoothing manufactures a continuous-looking shape between a handful of discrete spikes,
    which misrepresents the data. Box+strip is always honest about discreteness; violin only adds
    real information when there ARE enough distinct values for a density estimate to mean anything."""
    rng = np.random.default_rng(seed)
    positions = np.arange(1, len(data) + 1)
    n_unique = _n_unique_pooled(data)
    use_violin = n_unique > max_discrete_for_violin

    if use_violin:
        vp = ax.violinplot(data, positions=positions, showmeans=False, showmedians=False, showextrema=False, widths=0.8)
        for i, body in enumerate(vp['bodies']):
            c = colors[i] if colors else '#4C72B0'
            body.set_facecolor(c)
            body.set_alpha(0.35)
            body.set_edgecolor(c)
        box_width = 0.15
    else:
        box_width = 0.45

    bp = ax.boxplot(data, positions=positions, widths=box_width, showfliers=False, patch_artist=True,
                     medianprops=dict(color='black', linewidth=1.8), whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
    for i, patch in enumerate(bp['boxes']):
        c = colors[i] if colors else '#4C72B0'
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for i, d in enumerate(data):
        if len(d) == 0:
            continue
        x = rng.normal(positions[i], jitter, size=len(d))
        ax.scatter(x, d, s=5, color='black', alpha=0.18, zorder=3, linewidths=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=title_pad)
    if not use_violin:
        ax.text(0.99, 0.01, f'{n_unique} distinct values -- box+points only\n(violin omitted: too few for a density estimate)',
                 transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='#666666', style='italic')
    return positions


def categorical_proportion_bars(ax, data, group_labels, colors=None, value_labels=None, ylabel='% of texts', title=None,
                                  tol=1e-6, title_pad=14):
    """For genuinely few-valued/discrete data (e.g. max-probability under n=3 raters, which can
    only take ~3 distinct values), a proportion bar chart is honest where a box/violin plot
    degenerates or misleads. `data` is a list of 1D arrays (one per group, e.g. category);
    `value_labels`, if given, maps each distinct value to a display string, else the raw value is
    shown. Returns the sorted distinct values used as bar categories."""
    all_vals = np.concatenate([np.asarray(d) for d in data if len(d) > 0])
    distinct = np.sort(np.unique(np.round(all_vals / tol) * tol))
    n_groups = len(data)
    x = np.arange(len(distinct))
    width = 0.8 / n_groups
    for gi, d in enumerate(data):
        d = np.asarray(d)
        props = []
        for v in distinct:
            props.append(np.mean(np.abs(d - v) < tol) * 100 if len(d) > 0 else 0)
        c = colors[gi] if colors else None
        ax.bar(x + (gi - (n_groups - 1) / 2) * width, props, width=width, label=group_labels[gi], color=c)
    labels = [value_labels.get(round(v, 4), f'{v:.3f}') if value_labels else f'{v:.3f}' for v in distinct]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=title_pad)
    ax.legend(fontsize=8)
    return distinct


def vector_bar_pair(ax, emotions, vec_before, vec_after, label_before='Neutral', label_after='Persona',
                     color_before='#8C8C8C', color_after='#C44E52', title=None, ylabel='Probability'):
    """Grouped bar chart comparing two 11-dim vectors side by side, for worked/case-study
    examples -- turns a printed number list into an actual, readable comparison plot."""
    x = np.arange(len(emotions))
    width = 0.38
    ax.bar(x - width / 2, [vec_before.get(e, 0) for e in emotions], width=width, label=label_before, color=color_before)
    ax.bar(x + width / 2, [vec_after.get(e, 0) for e in emotions], width=width, label=label_after, color=color_after)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
