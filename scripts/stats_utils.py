"""
Shared statistical helpers for the vector-based distribution analysis. Pure computation, no I/O.

Pseudoreplication safeguard: every bootstrap/permutation helper here resamples/permutes at the
TEXT_ID level when a text_id grouping is supplied, never at the row level, because a single text
can contribute multiple rows (2 persona sides x up to 6 prompt configs x 7 models) that are not
independent observations of the underlying framing effect.

Fixed seed: RANDOM_SEED = 42 everywhere in this analysis layer, so every reported permutation
p-value and bootstrap CI is exactly reproducible.
"""
import numpy as np
from scipy import stats

RANDOM_SEED = 42


def cliffs_delta(x, y):
    """Cliff's delta effect size for two independent samples (non-parametric, robust to
    bounded/zero-inflated data like TV distance). Range [-1, 1]; 0 = no stochastic dominance,
    +1 = every x > every y, -1 = every x < every y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return None
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (nx * ny)


def holm_correct(pvals):
    """Holm-Bonferroni step-down correction (confirmatory-family control). Returns adjusted
    p-values in the SAME order as the input, monotone-enforced (never decreasing as rank
    increases)."""
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


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction (exploratory-family control). Returns adjusted
    p-values in the same order as input."""
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


def cluster_bootstrap_ci(values, cluster_ids, statistic_fn, n_boot=10000, seed=RANDOM_SEED, ci=95):
    """Bootstrap CI that resamples whole TEXT CLUSTERS with replacement (never individual rows),
    so repeated persona/prompt/model rows from the same text never get treated as independent
    resampling units. `values` and `cluster_ids` must be same-length array-likes; statistic_fn
    takes an array of values and returns a scalar."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    cluster_ids = np.asarray(cluster_ids)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    point_estimate = statistic_fn(values)
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled_clusters])
        boot_stats[b] = statistic_fn(values[idx])
    lo_pct = (100 - ci) / 2
    hi_pct = 100 - lo_pct
    lo, hi = np.percentile(boot_stats, [lo_pct, hi_pct])
    return point_estimate, lo, hi


def permutation_test_two_groups(group_a_votes_by_text, group_b_votes_by_text, statistic_fn,
                                 n_perm=10000, seed=RANDOM_SEED):
    """Text-clustered permutation test: for each text_id, pool that text's raw votes across
    the two groups being compared, then repeatedly shuffle group membership WITHIN each text
    (preserving each text's group sizes) to build an empirical null for statistic_fn, which
    takes (group_a_votes_by_text, group_b_votes_by_text) -> scalar.

    Used for the human persona TV noise-null (F): tests whether the observed mean TV could arise
    from pure within-text label noise even with no true framing effect.
    """
    rng = np.random.default_rng(seed)
    observed = statistic_fn(group_a_votes_by_text, group_b_votes_by_text)
    text_ids = list(group_a_votes_by_text.keys())
    null_stats = np.empty(n_perm)
    for p in range(n_perm):
        perm_a, perm_b = {}, {}
        for tid in text_ids:
            a_votes = group_a_votes_by_text[tid]
            b_votes = group_b_votes_by_text[tid]
            pooled = a_votes + b_votes
            rng.shuffle(pooled)
            perm_a[tid] = pooled[:len(a_votes)]
            perm_b[tid] = pooled[len(a_votes):]
        null_stats[p] = statistic_fn(perm_a, perm_b)
    p_value = (np.sum(null_stats >= observed) + 1) / (n_perm + 1)
    return observed, null_stats, p_value


def pooled_resplit_tv_permutation(group_a_by_key, group_b_by_key, emotions, n_perm=10000,
                                   seed=RANDOM_SEED):
    """Fast pooled-and-resplit permutation test specialised to the mean-TV statistic.

    Mathematically identical to permutation_test_two_groups() with a mean-TV statistic_fn, but
    without materialising and shuffling label lists. For one cell, let c be the vector of pooled
    label counts across both groups. A random re-split that assigns n_a of those labels to group A
    is exactly a multivariate hypergeometric draw a ~ MVHG(c, n_a), and group B then holds c - a.
    So the permutation can be sampled directly from that distribution:

        TV_cell = 0.5 * sum_k | a_k/n_a - (c_k - a_k)/n_b |

    This is used to reproduce the ORIGINAL H2 test cheaply for side-by-side comparison against the
    correctly-specified paired test. It is the conservative test (see paired_swap_permutation_test
    for why); it is retained for auditability, not because it is the preferred null.
    """
    idx = {e: i for i, e in enumerate(emotions)}
    K = len(emotions)
    keys = [k for k in group_a_by_key if k in group_b_by_key]
    counts, na_list, nb_list = [], [], []
    for k in keys:
        a_labels = [x for x in group_a_by_key[k] if x in idx]
        b_labels = [x for x in group_b_by_key[k] if x in idx]
        if not a_labels or not b_labels:
            continue
        c = np.zeros(K, dtype=np.int64)
        for x in a_labels + b_labels:
            c[idx[x]] += 1
        counts.append(c)
        na_list.append(len(a_labels))
        nb_list.append(len(b_labels))
    if not counts:
        return 0.0, np.zeros(n_perm), 1.0
    C = np.vstack(counts)
    na = np.asarray(na_list, dtype=float)
    nb = np.asarray(nb_list, dtype=float)

    # observed
    obs_a = np.zeros_like(C, dtype=float)
    for i, k in enumerate(keys[:len(counts)]):
        for x in group_a_by_key[k]:
            if x in idx:
                obs_a[i, idx[x]] += 1
    observed = float((0.5 * np.abs(obs_a / na[:, None] - (C - obs_a) / nb[:, None]).sum(axis=1)).mean())

    # Vectorised re-split across ALL cells at once. Each cell's pooled labels are laid out as a
    # padded row of slot indices; one random ranking per row per permutation decides which slots
    # fall in group A, and a single einsum turns the slot assignment into per-cell label counts.
    n_cells = C.shape[0]
    T = int((na + nb).max())
    onehot = np.zeros((n_cells, T, K))
    valid = np.zeros((n_cells, T), dtype=bool)
    ci = 0
    for k in keys:
        a_labels = [x for x in group_a_by_key[k] if x in idx]
        b_labels = [x for x in group_b_by_key[k] if x in idx]
        if not a_labels or not b_labels:
            continue
        for t, x in enumerate(a_labels + b_labels):
            onehot[ci, t, idx[x]] = 1.0
            valid[ci, t] = True
        ci += 1

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_perm)
    inv_na, inv_nb = (1.0 / na)[:, None], (1.0 / nb)[:, None]
    for p in range(n_perm):
        R = rng.random((n_cells, T))
        R[~valid] = np.inf                       # padding always sorts last, never selected
        rank = np.argsort(np.argsort(R, axis=1), axis=1)
        in_a = (rank < na[:, None]).astype(float)
        a = np.einsum('ctk,ct->ck', onehot, in_a)
        null_stats[p] = (0.5 * np.abs(a * inv_na - (C - a) * inv_nb).sum(axis=1)).mean()
    p_value = (np.sum(null_stats >= observed) + 1) / (n_perm + 1)
    return observed, null_stats, p_value


def paired_swap_tv_permutation(paired_by_key, emotions, n_perm=10000, seed=RANDOM_SEED):
    """Fast paired sign-flip permutation test specialised to the mean-TV statistic.

    Mathematically identical to paired_swap_permutation_test() with a mean-TV statistic_fn, but
    fast enough to run at 10,000 iterations, using one observation about the paired swap: a rater
    that gave the SAME label in both conditions contributes the same count to both vectors, so it
    cancels in their difference and cannot affect TV no matter how it is flipped. Only DISCORDANT
    pairs matter, and for those the count difference is exactly +/-(e_x - e_y). So

        TV_cell = 0.5 * || sum_i s_i * (e_{x_i} - e_{y_i}) ||_1 / n_cell,   s_i in {+1,-1}

    over that cell's discordant pairs only. Since mean TV across cells is ~0.18 at n=7, there are
    only ~1.5 discordant pairs per cell, so the permutation operates on a few thousand rows rather
    than all 12,600 labels. Returns (observed, null_stats, p_value) like the generic version.
    """
    from scipy import sparse
    idx = {e: i for i, e in enumerate(emotions)}
    K = len(emotions)
    keys = list(paired_by_key.keys())
    rows, diff_rows, n_per_cell = [], [], []
    for ci, k in enumerate(keys):
        pairs = paired_by_key[k]
        n_per_cell.append(len(pairs))
        for x, y in pairs:
            if x == y or x not in idx or y not in idx:
                continue
            v = np.zeros(K)
            v[idx[x]] += 1.0
            v[idx[y]] -= 1.0
            diff_rows.append(v)
            rows.append(ci)
    n_cells = len(keys)
    n_arr = np.asarray(n_per_cell, dtype=float)
    if not diff_rows:
        return 0.0, np.zeros(n_perm), 1.0
    D = np.vstack(diff_rows)
    M = D.shape[0]
    P = sparse.csr_matrix((np.ones(M), (np.asarray(rows), np.arange(M))), shape=(n_cells, M))

    def mean_tv(signed):
        per_cell = P @ signed                      # (n_cells, K) summed difference vectors
        tv = 0.5 * np.abs(per_cell).sum(axis=1) / n_arr
        return float(tv.mean())

    observed = mean_tv(D)
    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_perm)
    for i in range(n_perm):
        s = rng.integers(0, 2, size=M) * 2.0 - 1.0
        null_stats[i] = mean_tv(D * s[:, None])
    p_value = (np.sum(null_stats >= observed) + 1) / (n_perm + 1)
    return observed, null_stats, p_value


def paired_swap_permutation_test(paired_by_key, statistic_fn, n_perm=10000, seed=RANDOM_SEED):
    """Paired (within-rater) permutation test, for designs where the SAME raters answer under both
    conditions -- e.g. the same 7 LLMs answering each cell under neutral and under a persona.

    `paired_by_key` maps key -> list of (label_condition_a, label_condition_b), one tuple per
    rater present in BOTH conditions for that key. Under the sharp null "the condition does not
    change this rater's answer", a rater's own two labels are exchangeable with each other, so the
    null is built by independently flipping each (rater, key) pair with probability 0.5.

    Why not pool-and-resplit (permutation_test_two_groups): pooling all of a cell's labels across
    both conditions and re-splitting permutes labels ACROSS raters, which the null does not
    license when raters are systematically different from one another (Gemini is not exchangeable
    with Gemma3). Doing so injects between-rater heterogeneity into the null, inflating it, and
    makes the test conservative -- it under-rejects. Use this function whenever rater identity is
    tracked on both sides; use permutation_test_two_groups only for genuinely between-subjects
    comparisons (e.g. the human rounds, where different people rated each condition).

    statistic_fn takes (dict_a, dict_b) of key -> list-of-labels and returns a scalar.
    """
    rng = np.random.default_rng(seed)
    keys = list(paired_by_key.keys())
    obs_a = {k: [p[0] for p in paired_by_key[k]] for k in keys}
    obs_b = {k: [p[1] for p in paired_by_key[k]] for k in keys}
    observed = statistic_fn(obs_a, obs_b)
    null_stats = np.empty(n_perm)
    for i in range(n_perm):
        perm_a, perm_b = {}, {}
        for k in keys:
            pairs = paired_by_key[k]
            flip = rng.random(len(pairs)) < 0.5
            perm_a[k] = [p[1] if f else p[0] for p, f in zip(pairs, flip)]
            perm_b[k] = [p[0] if f else p[1] for p, f in zip(pairs, flip)]
        null_stats[i] = statistic_fn(perm_a, perm_b)
    p_value = (np.sum(null_stats >= observed) + 1) / (n_perm + 1)
    return observed, null_stats, p_value


def permutation_pairing_test(human_deltas, llm_deltas, cosine_fn, n_perm=10000, seed=RANDOM_SEED):
    """Shuffled-pairing null for human-LLM directional alignment (K): repeatedly shuffles which
    LLM delta vector is paired with which human delta vector (preserving each side's marginal
    set of vectors), recomputes mean defined cosine each time, to test whether the observed
    mean cosine reflects genuine text-specific tracking rather than two arbitrary sets of
    vectors that just happen to point in broadly similar directions on average.
    `human_deltas` and `llm_deltas` are equal-length lists of 11-dim delta dict vectors, index i
    on each side must currently correspond to the same text/condition (i.e. correctly paired)."""
    rng = np.random.default_rng(seed)
    n = len(human_deltas)
    observed_vals = [cosine_fn(human_deltas[i], llm_deltas[i]) for i in range(n)]
    observed_vals = [v for v in observed_vals if v is not None]
    observed_mean = float(np.mean(observed_vals)) if observed_vals else None

    null_means = np.empty(n_perm)
    idx = np.arange(n)
    for p in range(n_perm):
        perm_idx = rng.permutation(idx)
        vals = [cosine_fn(human_deltas[i], llm_deltas[perm_idx[i]]) for i in range(n)]
        vals = [v for v in vals if v is not None]
        null_means[p] = np.mean(vals) if vals else np.nan
    valid_null = null_means[~np.isnan(null_means)]
    p_value = (np.sum(valid_null >= observed_mean) + 1) / (len(valid_null) + 1) if observed_mean is not None else None
    return observed_mean, valid_null, p_value


def kruskal_wallis(*groups):
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    stat, p = stats.kruskal(*groups)
    return stat, p


def mann_whitney(x, y):
    stat, p = stats.mannwhitneyu(x, y, alternative='two-sided')
    return stat, p


def ordered_trend_permutation_test(groups_in_order, n_perm=10000, seed=RANDOM_SEED):
    """Transparent permutation-based test for a monotonic increasing trend across an ORDERED
    set of groups (e.g. unambiguous < author_independent < author_relevant), used in place of
    Jonckheere-Terpstra for full auditability. Test statistic: Jonckheere's J = sum over all
    ordered group pairs (i<j) of the count of (x in group_i, y in group_j) pairs with x < y
    (ties count as 0.5). Larger J = stronger support for the hypothesized increasing order.
    Null: shuffle all observations across the group labels (preserving group sizes) and
    recompute J each time.
    """
    rng = np.random.default_rng(seed)
    groups_in_order = [np.asarray(g, dtype=float) for g in groups_in_order]
    sizes = [len(g) for g in groups_in_order]

    def jonckheere_j(groups):
        j = 0.0
        for i in range(len(groups)):
            for k in range(i + 1, len(groups)):
                gi, gk = groups[i], groups[k]
                diff = gk[:, None] - gi[None, :]
                j += np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
        return j

    observed_j = jonckheere_j(groups_in_order)
    pooled = np.concatenate(groups_in_order)
    null_js = np.empty(n_perm)
    for p in range(n_perm):
        shuffled = rng.permutation(pooled)
        regrouped = []
        start = 0
        for s in sizes:
            regrouped.append(shuffled[start:start + s])
            start += s
        null_js[p] = jonckheere_j(regrouped)
    p_value = (np.sum(null_js >= observed_j) + 1) / (n_perm + 1)
    return observed_j, null_js, p_value


def spearman_with_cluster_bootstrap(x, y, cluster_ids, n_boot=10000, seed=RANDOM_SEED):
    """Spearman rho between x and y, with a text-clustered bootstrap 95% CI (never row-level
    resampling, since repeated persona/prompt/model rows from the same text are not
    independent).

    AUDIT REPAIR (2026-09-22): this used to return `p` from scipy.stats.spearmanr() computed on
    ALL rows, while only the confidence interval was text-clustered. With ~12 correlated rows per
    text, that p-value treats repeated observations as independent evidence: duplicating every row
    leaves rho unchanged but shrinks the p-value, which is exactly the behaviour a clustered
    analysis is supposed to prevent. Presenting it beside a clustered CI was incoherent. The naive
    p is no longer returned. Inference now comes from the text-clustered bootstrap itself: p_boot
    is the two-sided proportion of clustered bootstrap replicates falling on the other side of
    zero, which respects the same clustering as the interval.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = stats.spearmanr(x, y).statistic

    rng = np.random.default_rng(seed)
    cluster_ids = np.asarray(cluster_ids)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    boot_rhos = np.empty(n_boot)
    for b in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled_clusters])
        boot_rhos[b] = stats.spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.percentile(boot_rhos, [2.5, 97.5])
    p_boot = 2.0 * min((boot_rhos <= 0).mean(), (boot_rhos >= 0).mean())
    p_boot = max(p_boot, 1.0 / n_boot)
    return rho, p_boot, lo, hi


def conditional_null_tv(votes_a, votes_b, emotions, n_draw=400, seed=RANDOM_SEED):
    """Expected TV between two vote multisets under the conditional sampling null: pool this
    text's votes across both conditions, reshuffle, re-split at the observed group sizes, and
    average the resulting TV.

    Why this matters (added 2026-09-22): TV between two small-sample vote vectors is NOT zero when
    there is no effect. Two 3-vote draws from the same underlying distribution disagree
    substantially by chance, and they disagree MORE when that distribution is dispersed. Because
    the ambiguous categories are, by construction, the ones with dispersed votes, a raw comparison
    of TV across categories partly measures baseline dispersion rather than any framing effect.
    Subtracting this expectation gives each text's EXCESS TV -- the part not explained by
    finite-sample noise given its own votes.
    """
    rng = np.random.default_rng(seed)
    idx = {e: i for i, e in enumerate(emotions)}
    a = [v for v in votes_a if v in idx]
    b = [v for v in votes_b if v in idx]
    if not a or not b:
        return None
    pooled = np.array([idx[v] for v in a + b])
    na, k = len(a), len(emotions)

    def _tv(ia, ib):
        va = np.bincount(ia, minlength=k).astype(float)
        vb = np.bincount(ib, minlength=k).astype(float)
        return 0.5 * np.abs(va / va.sum() - vb / vb.sum()).sum()

    vals = np.empty(n_draw)
    for d in range(n_draw):
        sh = rng.permutation(pooled)
        vals[d] = _tv(sh[:na], sh[na:])
    return float(vals.mean())


def friedman_test(*groups):
    """Friedman test for repeated-measures comparison across >=3 related conditions (e.g. the 6
    prompt configs measured on the same texts). groups must be equal-length arrays, aligned by
    the same subject (text_id) order."""
    stat, p = stats.friedmanchisquare(*groups)
    return stat, p


def wilcoxon_signed_rank(x, y):
    stat, p = stats.wilcoxon(x, y)
    return stat, p
