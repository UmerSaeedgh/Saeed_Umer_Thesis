"""
Builds the framed_stimulus / full_prompt rows for the 300-text balanced sample
(sampled_texts.csv), in four-condition mode (neutral / generic_human / persona_a / persona_b).

Row structure per text: 4 framing_condition x 3 prompt_variant x 2 label_format = 24 rows.
300 texts (100 per ambiguity pool) x 24 = 7,200 rows total, split into 3 files by pool to match
the schema run_classification.py expects (KEY_COLS = text_id, framing_condition, prompt_variant,
label_format).

Notes on the prompt design:
  - persona_b is a fully wired, real tested condition (see make_framed_stimulus), answering
    proposal Section 5's "does 'politician' vs 'comedian' change the perceived emotion" contrast.
  - Three few-shot/one-shot examples are drawn from the dual-annotated corpus, one per ambiguity
    type, author-free in all conditions to avoid teaching the model an "author cue -> same answer"
    shortcut.
"""

import pandas as pd
from pathlib import Path

BASE = Path.home() / 'Desktop' / 'thesis' / 'thesis final'

# -- Label list (11 labels, per ThesisProposal.pdf Section 4.1) --------------------------------
LABELS = ["anger", "disgust", "fear", "guilt", "joy", "pride",
          "relief", "sadness", "shame", "surprise", "trust"]

NUMBERED_LABEL_LIST = "\n".join(f"{i}. {lbl}" for i, lbl in enumerate(LABELS, 1))
INLINE_LABEL_LIST = ", ".join(LABELS)

NUMBERED_QUESTION = (
    "What emotion did the author feel when they wrote this? "
    "Choose exactly one label from the list below.\n\n"
    f"{NUMBERED_LABEL_LIST}\n\n"
    "Respond in this exact format:\n"
    "Label: <label>\n"
    "Reason: <one sentence>"
)

INLINE_QUESTION = (
    "What emotion did the author feel when they wrote this? "
    f"Choose exactly one label from: {INLINE_LABEL_LIST}.\n\n"
    "Respond in this exact format:\n"
    "Label: <label>\n"
    "Reason: <one sentence>"
)

QUESTION_BY_FORMAT = {"numbered": NUMBERED_QUESTION, "inline": INLINE_QUESTION}

# -- Examples: one per ambiguity type, author's own label, no Author line ----------------------
# Drawn from the dual-annotated 1,200-text corpus (thesis final/ambiguity_classification_1200.csv), outside
# both the original 300-sample and the new sampled_texts.csv, so they can never leak into the
# experimental set. Distinct labels (disgust/fear/joy) to avoid label-frequency priming.
UNAMBIGUOUS_EXAMPLE = (
    "I found some rotten onions in my cupboard that were slimy and smelly and I had to clean it up",
    "disgust",
    "The author describes finding spoiled, foul-smelling food, a classic disgust trigger.",
)
AUTHOR_INDEPENDENT_EXAMPLE = (
    "my mom fell down the stairs.",
    "fear",
    "The author describes a sudden, alarming accident involving a parent, which triggers fear for their safety.",
)
AUTHOR_RELEVANT_EXAMPLE = (
    "i recently got  a new job",
    "joy",
    "The author describes a positive life event, which typically evokes joy.",
)

ONE_SHOT_EXAMPLES = [UNAMBIGUOUS_EXAMPLE]

FEW_SHOT_EXAMPLES = [UNAMBIGUOUS_EXAMPLE, AUTHOR_INDEPENDENT_EXAMPLE, AUTHOR_RELEVANT_EXAMPLE]


def format_example(text, label, reason):
    return f"Text: {text}\nLabel: {label}\nReason: {reason}"


def make_framed_stimulus(text, framing_condition, persona_a, persona_b):
    if framing_condition == 'neutral':
        return f"Text: {text}"
    elif framing_condition == 'generic_human':
        return f"Text: {text}\nAuthor: a person"
    elif framing_condition == 'persona_a':
        return f"Text: {text}\nAuthor: {persona_a}"
    elif framing_condition == 'persona_b':
        return f"Text: {text}\nAuthor: {persona_b}"
    else:
        raise ValueError(f"Unknown framing condition: {framing_condition}")


def make_full_prompt(framed_stimulus, prompt_variant, label_format):
    question = QUESTION_BY_FORMAT[label_format]

    if prompt_variant == 'zero_shot':
        return f"{framed_stimulus}\n\n{question}"

    elif prompt_variant == 'one_shot':
        example_block = format_example(*ONE_SHOT_EXAMPLES[0])
        return (
            f"Here is an example:\n\n"
            f"{example_block}\n\n"
            f"Now do the same for this text:\n\n"
            f"{framed_stimulus}\n\n"
            f"{question}"
        )

    elif prompt_variant == 'few_shot':
        example_block = "\n\n".join(format_example(*ex) for ex in FEW_SHOT_EXAMPLES)
        return (
            f"Here are some examples:\n\n"
            f"{example_block}\n\n"
            f"Now do the same for this text:\n\n"
            f"{framed_stimulus}\n\n"
            f"{question}"
        )

    else:
        raise ValueError(f"Unknown prompt variant: {prompt_variant}")


FRAMING_CONDITIONS = ['neutral', 'generic_human', 'persona_a', 'persona_b']
PROMPT_VARIANTS = ['zero_shot', 'one_shot', 'few_shot']
LABEL_FORMATS = ['numbered', 'inline']

MODEL_COLS = []
for model in ['ChatGPT', 'Claude', 'Gemini', 'KimiK3']:
    MODEL_COLS += [f'{model}_label', f'{model}_reason']

POOL_TO_FILE = {
    'Unambiguous': BASE / 'experiment_results_unambiguous.csv',
    'Author-Independent Ambiguous': BASE / 'experiment_results_author_independent.csv',
    'Author-Relevant Ambiguous': BASE / 'experiment_results_author_relevant.csv',
}


def build_rows(sample_df):
    rows = []
    for _, r in sample_df.iterrows():
        for framing_condition in FRAMING_CONDITIONS:
            framed_stimulus = make_framed_stimulus(
                r['text'], framing_condition, r['persona_a'], r['persona_b']
            )
            for prompt_variant in PROMPT_VARIANTS:
                for label_format in LABEL_FORMATS:
                    full_prompt = make_full_prompt(framed_stimulus, prompt_variant, label_format)
                    row = {
                        'text_id': r['text_id'],
                        'text': r['text'],
                        'classification': r['classification'],
                        'author_emotion': r['author_emotion'],
                        'reader_majority': r['reader_majority'],
                        'persona_a': r['persona_a'],
                        'persona_b': r['persona_b'],
                        'framing_condition': framing_condition,
                        'framed_stimulus': framed_stimulus,
                        'prompt_variant': prompt_variant,
                        'label_format': label_format,
                        'full_prompt': full_prompt,
                    }
                    for col in MODEL_COLS:
                        row[col] = ''
                    rows.append(row)
    return pd.DataFrame(rows)


def main():
    sample = pd.read_csv(BASE / 'sampled_texts.csv')
    print(f"Loaded {len(sample)} sampled texts")

    for pool, path in POOL_TO_FILE.items():
        pool_df = sample[sample['classification'] == pool].reset_index(drop=True)
        print(f"\n{pool}: {len(pool_df)} texts -> {len(pool_df) * 24} rows")
        out_df = build_rows(pool_df)
        out_df.to_csv(path, index=False)
        print(f"  Saved {path.name} ({len(out_df)} rows)")

    print("\nDone. Showing one example per framing_condition for the first text (zero_shot, numbered):")
    sample_out = pd.read_csv(POOL_TO_FILE['Unambiguous'])
    first_tid = sample_out['text_id'].iloc[0]
    view = sample_out[(sample_out['text_id'] == first_tid) &
                       (sample_out['prompt_variant'] == 'zero_shot') &
                       (sample_out['label_format'] == 'numbered')]
    for _, row in view.iterrows():
        print(f"\n{'='*60}")
        print(f"framing_condition={row['framing_condition']}")
        print(f"{'='*60}")
        print(row['full_prompt'])


if __name__ == '__main__':
    main()
