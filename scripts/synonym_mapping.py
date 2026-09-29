"""
OPTIONAL, ADDITIVE synonym-mapping step for off-list LLM emotion labels -- built at the user's
explicit request as an alternative to the project's default policy (exclude off-list answers,
never force-map). This module does NOT change the default pipeline; build_emotion_distributions.py
still excludes off-list answers by default. This is used only by
scripts/run_synonym_sensitivity_analysis.py to produce a clearly-labeled SENSITIVITY ANALYSIS
comparing "exclude" vs. "map synonyms" as a methodology choice.

Two resolution stages, applied in order to any label the normalizer + fallback leaves off-list:

1. SELF-DECLARED CLOSEST MATCH: some models explicitly state their own nearest label, e.g.
   "concern (not available in list - closest match would be fear)" or
   "frustration (not available - closest is anger)". When present, the MODEL'S OWN stated
   choice is used rather than a hand-picked synonym -- this is not really "mapping a synonym",
   it's just correctly parsing an answer the model already gave in a non-standard format.

2. STATIC SYNONYM DICTIONARY: near-synonym emotion words not in the fixed 11-label set, mapped to
   the closest canonical label. Every entry has a one-line justification. This is a genuine,
   debatable judgment call -- entries marked "(debatable)" have real alternative mappings in the
   literature and should be treated as the weakest part of this analysis, not a settled fact.

Words that are NOT emotion synonyms at all (confusion, interest, curiosity, indifference, "none",
"none of the above", explicit refusals, and every API-error message) are deliberately NOT mapped
-- they remain excluded under both the default and the sensitivity-analysis pipeline, because
force-mapping a rate-limit error message or a refusal to an emotion would not be a defensible
linguistic judgment call, it would be fabricating data.
"""
import re

from emotion_vectors import EMOTIONS

# --- Stage 1: self-declared closest-match pattern ---------------------------------------------
_CLOSEST_MATCH_RE = re.compile(
    r"(?:closest\s*(?:match|is)?|best\s*match|selecting\s*closest\s*match)\s*[:\-]?\s*\**([a-z]+)\**",
    re.IGNORECASE,
)


def extract_self_declared_match(raw_lower: str):
    """If the model's own text states a closest-match label that IS one of the 11 canonical
    emotions, return it. Otherwise None."""
    m = _CLOSEST_MATCH_RE.search(raw_lower)
    if m:
        candidate = m.group(1).lower()
        if candidate in EMOTIONS:
            return candidate
    return None


# --- Stage 2: static synonym dictionary -----------------------------------------------------
# {off-list word: (canonical emotion, justification)}
SYNONYM_MAP = {
    'anxiety':          ('fear',     'anxiety is standardly treated as a fear-family/anticipatory-fear state'),
    'anxiety/fear':      ('fear',     'already contains fear explicitly'),
    'frustration':      ('anger',    'frustration is the standard blocked-goal subtype of anger'),
    'frustration/anger': ('anger',   'already contains anger explicitly'),
    'annoyance':        ('anger',    'annoyance is a low-intensity anger subtype'),
    'frustration/annoyance': ('anger', 'both terms are anger-family'),
    'disappointment':   ('sadness',  'disappointment is standardly classified as a sadness-family emotion'),
    'embarrassment':    ('shame',    'embarrassment and shame are treated as near-synonyms in most appraisal models'),
    'concern':          ('fear',     'concern is a mild/anticipatory form of fear (worry)'),
    'concern/worry':     ('fear',    'worry is a canonical fear synonym'),
    'hurt':             ('sadness',  '(debatable) emotional hurt is most often sadness-coded, though it can blend with anger'),
    'distress':         ('fear',     '(debatable) distress is closest to fear/anxiety in most taxonomies, though it can blend with sadness'),
    'regret':           ('guilt',    'regret over one’s own actions is the standard guilt-family emotion'),
    'shock':            ('surprise', 'shock is an intensified form of surprise'),
    'shock/surprise':    ('surprise', 'already contains surprise explicitly'),
    'excitement':       ('joy',      'excitement is a high-arousal positive emotion, closest to joy in this label set'),
    'love':             ('joy',      '(debatable) no "love" category exists here; joy is the closest positive-valence match'),
    'horror':           ('fear',     'horror is an intensified form of fear'),
    'envy':             ('anger',    '(debatable) envy is grouped with the hostility/anger cluster in several appraisal taxonomies'),
    'distrust':         ('disgust',  '(debatable) interpersonal distrust is often disgust-coded (moral disgust); fear is a plausible alternative'),
    'suspicion':        ('fear',     '(debatable) suspicion reflects threat-anticipation, closest to fear'),
    'fascination':      ('joy',      '(debatable) fascination is a positive-interest state, closest to joy among these 11'),
    'betrayal':         ('anger',    'betrayal is standardly anger-family (moral violation by another)'),
    'worry':            ('fear',     'worry is a canonical fear synonym'),
    'gratitude':        ('trust',    '(debatable) gratitude is linked to trust/prosocial bonding in appraisal theory; joy is a plausible alternative'),
    'outrage':          ('anger',    'outrage is an intensified form of anger'),
    'discomfort':       ('sadness',  '(debatable) generic discomfort has no clean single mapping; sadness chosen as the mildest negative-valence default'),
}

# Explicitly NOT mapped (documented here so the exclusion is a decision, not an oversight):
# confusion, interest, curiosity, indifference, none, none of the above, and any string
# containing "error" or "cannot determine"/"cannot reliably" (refusals) or "candidates"
# (a stray API-error fragment).
_NEVER_MAP_SUBSTRINGS = ('error', 'cannot determine', 'cannot reliably', "'candidates'")


def resolve_with_synonym_mapping(raw_or_normalized: str):
    """Given a label already run through the standard normalizer (and off-list fallback), try
    the two additional resolution stages above. Returns a canonical emotion string, or None if
    it should remain excluded. Never called by the default pipeline -- see module docstring."""
    if raw_or_normalized is None:
        return None
    s = str(raw_or_normalized).strip().lower()
    if s in EMOTIONS:
        return s
    if any(sub in s for sub in _NEVER_MAP_SUBSTRINGS):
        return None
    if s in ('none', 'none of the above', 'confusion', 'interest', 'curiosity', 'indifference'):
        return None

    self_declared = extract_self_declared_match(s)
    if self_declared:
        return self_declared

    if s in SYNONYM_MAP:
        return SYNONYM_MAP[s][0]

    # Compound answers like "gratitude/relief" or "discomfort/sadness (closest fit)": check each
    # "/"-separated token in order (preferring the model's FIRST-listed choice), against both the
    # canonical set and the synonym dictionary. "neutral" is skipped as a token even though it's
    # not canonical, since it means "no answer", not an emotion synonym.
    tokens = re.split(r'[\s/(]+', s)
    for tok in tokens:
        tok = tok.strip('*.,:;')
        if not tok or tok == 'neutral':
            continue
        if tok in EMOTIONS:
            return tok
        if tok in SYNONYM_MAP:
            return SYNONYM_MAP[tok][0]
    return None
