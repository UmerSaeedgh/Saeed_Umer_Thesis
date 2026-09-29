"""
Core math for representing emotion annotations as 11-dimensional probability distributions,
instead of collapsing each (text, condition) to a single majority label.

This module is pure computation (no file I/O) so it can be unit-tested in isolation -- see
scripts/sanity_check_emotion_vectors.py for the required sanity checks, which must pass before
this is used to recompute any real results.

Fixed emotion ordering (alphabetical -- matches the label set used throughout this project's
Google Forms and LLM prompts):
"""
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

EMOTIONS: Tuple[str, ...] = (
    'anger', 'disgust', 'fear', 'guilt', 'joy', 'pride',
    'relief', 'sadness', 'shame', 'surprise', 'trust',
)
EMOTION_INDEX = {e: i for i, e in enumerate(EMOTIONS)}
N_EMOTIONS = len(EMOTIONS)


def build_vector(votes: List[str]) -> Dict[str, float]:
    """Build an 11-dim probability distribution from a list of raw emotion votes.

    Counts occurrences of each of the 11 fixed emotions and divides by N (the number of votes).
    Returns a dict {emotion: probability} covering all 11 emotions (0.0 for any not observed),
    in EMOTIONS order. Sums to 1.0 unless votes is empty (returns None in that case -- genuinely
    missing data is not silently treated as a valid all-zero distribution).

    Any vote string not in EMOTIONS (e.g. off-list/garbage LLM output) is dropped with a note in
    the returned metadata rather than silently ignored -- see build_vector_with_meta.
    """
    vec, _ = build_vector_with_meta(votes)
    return vec


def build_vector_with_meta(votes: List[str]) -> Tuple[Optional[Dict[str, float]], dict]:
    """Same as build_vector, but also returns metadata: n (votes used), n_dropped (off-list
    votes excluded), dropped_values (what they were, for auditing)."""
    cleaned = [v.strip().lower() for v in votes if v is not None and str(v).strip() != '']
    valid = [v for v in cleaned if v in EMOTION_INDEX]
    dropped = [v for v in cleaned if v not in EMOTION_INDEX]
    n = len(valid)
    meta = {'n': n, 'n_dropped': len(dropped), 'dropped_values': dropped}
    if n == 0:
        return None, meta
    counts = Counter(valid)
    vec = {e: counts.get(e, 0) / n for e in EMOTIONS}
    return vec, meta


def vector_to_array(vec: Dict[str, float]) -> List[float]:
    """Fixed-order list representation, for when an array is needed internally."""
    return [vec[e] for e in EMOTIONS]


def total_variation(p: Dict[str, float], q: Dict[str, float]) -> float:
    """TV(P, Q) = 0.5 * sum(|P_i - Q_i|). Returns a value in [0, 1]."""
    return 0.5 * sum(abs(p[e] - q[e]) for e in EMOTIONS)


def distribution_shift_percent(p: Dict[str, float], q: Dict[str, float]) -> float:
    return total_variation(p, q) * 100.0


def distribution_overlap_percent(p: Dict[str, float], q: Dict[str, float]) -> float:
    return (1.0 - total_variation(p, q)) * 100.0


def is_in_dominant_set(vec: Dict[str, float], emotion: str, eps: float = 1e-9) -> bool:
    """True if `emotion` is a member of vec's dominant set (handles ties correctly -- an
    emotion tied for the max still counts as present, not arbitrarily excluded)."""
    return emotion in dominant_set(vec, eps)


def dominant_set(vec: Dict[str, float], eps: float = 1e-9) -> frozenset:
    """The set of emotion(s) sharing the maximum probability (ties preserved in full, not
    arbitrarily broken). eps guards against float rounding creating spurious near-ties."""
    max_p = max(vec.values())
    return frozenset(e for e, p in vec.items() if abs(p - max_p) <= eps)


@dataclass
class DominantChangeResult:
    neutral_dominant: frozenset
    persona_dominant: frozenset
    overlap: frozenset          # emotions in both dominant sets
    dominant_emotion_changed: bool   # True only if the sets are COMPLETELY disjoint
    is_tied_neutral: bool
    is_tied_persona: bool
    is_partial_change: bool     # sets differ but share >=1 emotion (not a "complete" change)


def compare_dominant(neutral_vec: Dict[str, float], persona_vec: Dict[str, float]) -> DominantChangeResult:
    """Set-overlap dominant-emotion comparison, per spec:
      - any overlap between the two dominant sets -> NOT a complete change
      - fully disjoint sets -> dominant_emotion_changed = True
    Ties (multiple emotions sharing the max) are preserved as sets, not arbitrarily resolved.
    """
    n_dom = dominant_set(neutral_vec)
    p_dom = dominant_set(persona_vec)
    overlap = n_dom & p_dom
    changed = len(overlap) == 0
    return DominantChangeResult(
        neutral_dominant=n_dom,
        persona_dominant=p_dom,
        overlap=overlap,
        dominant_emotion_changed=changed,
        is_tied_neutral=len(n_dom) > 1,
        is_tied_persona=len(p_dom) > 1,
        is_partial_change=(n_dom != p_dom) and not changed,
    )


def new_emotions(neutral_vec: Dict[str, float], persona_vec: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    """Emotions with neutral probability == 0 (not merely low) and persona probability > 0.
    Returns {emotion: persona_probability} for each newly-observed emotion.

    Naming note (per the project's own documentation requirement): this means "newly OBSERVED
    under the persona condition given the available annotations," not "impossible under neutral
    framing" -- zero here reflects absence from a finite sample, not a claim of true-zero
    probability. See module docstring / final report for this caveat.
    """
    return {e: persona_vec[e] for e in EMOTIONS if neutral_vec[e] <= eps and persona_vec[e] > eps}


def disappeared_emotions(neutral_vec: Dict[str, float], persona_vec: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    """Inverse of new_emotions: neutral probability > 0, persona probability == 0."""
    return {e: neutral_vec[e] for e in EMOTIONS if neutral_vec[e] > eps and persona_vec[e] <= eps}


def delta_vector(neutral_vec: Dict[str, float], persona_vec: Dict[str, float]) -> Dict[str, float]:
    """persona_vector - neutral_vector, per emotion. Positive = probability mass moved IN,
    negative = moved OUT. This is the signed, directional change vector."""
    return {e: persona_vec[e] - neutral_vec[e] for e in EMOTIONS}


@dataclass
class CosineResult:
    value: Optional[float]      # None when undefined
    defined: bool
    reason: str                 # explanation, populated especially when NOT defined


def safe_cosine_similarity(delta_a: Dict[str, float], delta_b: Dict[str, float], eps: float = 1e-12) -> CosineResult:
    """Cosine similarity between two DELTA vectors (never raw distributions -- callers must pass
    delta vectors, this function doesn't know or care what produced them, but naming at the call
    site should always say `delta`).

    Zero-vector safety: if either delta vector has ~zero norm (i.e. that condition produced no
    detectable shift from neutral), cosine similarity is mathematically undefined (0/0). Rather
    than returning NaN or an arbitrary fallback (e.g. 0 or 1), this is reported as
    defined=False with an explanit reason, so downstream aggregation can exclude these cases
    explicitly instead of silently corrupting a mean with NaNs or fake certainty.
    """
    a = [delta_a[e] for e in EMOTIONS]
    b = [delta_b[e] for e in EMOTIONS]
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    a_zero = norm_a <= eps
    b_zero = norm_b <= eps
    if a_zero and b_zero:
        return CosineResult(None, False, "both delta vectors are ~zero (neither side shifted from neutral)")
    if a_zero:
        return CosineResult(None, False, "human delta vector is ~zero (no shift from neutral) while LLM delta is non-zero")
    if b_zero:
        return CosineResult(None, False, "LLM delta vector is ~zero (no shift from neutral) while human delta is non-zero")
    dot = sum(x * y for x, y in zip(a, b))
    cos = dot / (norm_a * norm_b)
    # guard tiny float overshoot outside [-1, 1]
    cos = max(-1.0, min(1.0, cos))
    return CosineResult(cos, True, "")


def shannon_entropy(vec: Dict[str, float]) -> float:
    """Shannon entropy in nats-free bits-free natural units: H = -sum(p*ln(p)), 0*ln(0):=0.
    Range: [0, ln(N_EMOTIONS)]. A separate, distinct metric from TV -- this measures how
    concentrated vs. spread out a SINGLE distribution is, not how far apart two distributions
    are. 0 = fully concentrated on one emotion, ln(11) = perfectly uniform across all 11."""
    return -sum(p * math.log(p) for p in vec.values() if p > 0.0)


def normalized_entropy(vec: Dict[str, float]) -> float:
    """shannon_entropy divided by its theoretical maximum ln(N_EMOTIONS), rescaled to [0, 1]
    so entropy is comparable regardless of how many emotions happen to appear. 0 = single
    emotion, 1 = uniform across all 11."""
    max_h = math.log(N_EMOTIONS)
    return shannon_entropy(vec) / max_h


def max_probability(vec: Dict[str, float]) -> float:
    """Consensus strength: the probability mass on the single most-voted emotion (or, when
    tied, on each of the tied emotions -- this returns that shared max value, not a count).
    Range: [1/N_EMOTIONS, 1.0]. Analytically distinct from the dominant SET (which emotion(s)
    hold that mass) -- this is how much mass they hold."""
    return max(vec.values())


def support_size(vec: Dict[str, float], eps: float = 1e-12) -> int:
    """Number of emotions with non-zero probability mass (i.e. how many distinct emotions
    were actually voted for). Range: [1, N_EMOTIONS] for any vector built from >=1 vote."""
    return sum(1 for p in vec.values() if p > eps)
