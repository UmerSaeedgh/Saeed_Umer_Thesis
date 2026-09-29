"""Recreate the added agreement and sampled-corpus descriptive tables."""
from collections import Counter
from pathlib import Path
import hashlib
import re

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]  # thesis final/ (patched from original sibling-of-script layout)
DATA = ROOT / "data"
OUT = ROOT / "results"
EXPECTED = dict(row.split(",", 1) for row in (OUT / "input_sha256.csv").read_text().splitlines()[1:])
for filename in ("human_annotations.csv", "sampled_texts.csv", "experiment_results_all.csv"):
    path = DATA / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED[f"data/{filename}"]

humans = pd.read_csv(DATA / "human_annotations.csv")
sample = pd.read_csv(DATA / "sampled_texts.csv")
humans = humans.drop(columns="pool").merge(
    sample[["text_id", "classification"]].rename(columns={"classification": "pool"}),
    on="text_id", validate="many_to_one")
assert len(humans) == 900 and len(sample) == 300
assert humans.groupby("text_id").side.nunique().eq(3).all()
assert (humans.raters_raw.str.split("|").str.len() == humans.n_raters).all()
assert humans.text_id.nunique() == sample.text_id.nunique() == 300
categories = sorted(set("|".join(humans.raters_raw).split("|")))
assert len(categories) == 11

def fleiss_equal_three(rows):
    # Conventional Fleiss kappa: a fixed n=3 per text/condition cell.
    counts = [[Counter(v.split("|"))[emotion] for emotion in categories] for v in rows.raters_raw]
    n = len(counts)
    assert n > 0 and all(sum(row) == 3 for row in counts)
    observed = sum(sum(x * (x - 1) for x in row) / 6 for row in counts) / n
    totals = [sum(row[j] for row in counts) for j in range(11)]
    expected = sum((v / (3 * n)) ** 2 for v in totals)
    return (observed - expected) / (1 - expected)

rows = []
for (pool, side), group in humans.groupby(["pool", "side"]):
    eligible = group[group.n_raters == 3]
    rows.append({"pool": pool, "framing": side, "cells_all": len(group),
                 "cells_with_three_votes": len(eligible), "fleiss_kappa": fleiss_equal_three(eligible)})
pd.DataFrame(rows).sort_values(["pool", "framing"]).to_csv(OUT / "human_interannotator_agreement_verified.csv", index=False)

freq = sample.author_emotion.value_counts().sort_index()
assert freq.sum() == 300
pd.DataFrame({"emotion": freq.index, "author_label_count": freq.values}).to_csv(
    OUT / "sample_author_emotion_frequencies_verified.csv", index=False)
print(pd.DataFrame(rows).sort_values(["pool", "framing"]).to_string(index=False))
words = sample.text.str.split().str.len()
print(f"sample words: median={words.median():.0f}, IQR={words.quantile(.25):.0f}--{words.quantile(.75):.0f}, range={words.min()}--{words.max()}")
responses = pd.read_csv(DATA / "experiment_results_all.csv")
model_names = ["ChatGPT", "Claude", "Gemini", "Qwen3", "Gemma3", "Ministral", "Llama31"]
assert len(responses) == 7200
reason_counts = pd.DataFrame([{"model": m, "explanations_recorded":
                               int(responses[f"{m}_reason"].fillna("").astype(str).str.strip().ne("").sum())}
                              for m in model_names])
reason_counts.to_csv(OUT / "model_explanation_coverage_verified.csv", index=False)
print(reason_counts.to_string(index=False))
example = humans[humans.text_id == 2105].set_index("side").raters_raw.to_dict()
assert Counter(example["neutral"].split("|")) == Counter({"pride": 1, "joy": 2})
assert Counter(example["a"].split("|")) == Counter({"sadness": 1, "relief": 3})
assert Counter(example["b"].split("|")) == Counter(example["neutral"].split("|"))
config = responses[(responses.text_id == 2105) & (responses.prompt_variant == "zero_shot") &
                   (responses.label_format == "numbered")].set_index("framing_condition")
assert config.loc["neutral", "ChatGPT_label"] == "pride"
assert config.loc["persona_a", "ChatGPT_label"] == "relief"
assert config.loc["persona_b", "ChatGPT_label"] == "pride"

# Requested conventional benchmark, restricted to original-corpus neutral texts
# where at least three of five readers chose the stored majority label.
gold = sample.loc[sample.majority_share >= .6, ["text_id", "reader_majority"]]
assert len(gold) == 241
baseline = responses[responses.framing_condition == "neutral"].drop(
    columns="reader_majority", errors="ignore").merge(gold, on="text_id", validate="many_to_one")
assert len(baseline) == 6 * 241
def normalize_model_label(value):
    if pd.isna(value):
        return ""
    s = str(value).strip().strip("*").strip().lower()
    bare = re.fullmatch(r"(\d{1,2})\s*[.):]?", s)
    if bare:
        i = int(bare.group(1))
        return categories[i - 1] if 1 <= i <= len(categories) else s
    numbered = re.fullmatch(r"\d{1,2}\s*[.):]\s*(.+)", s)
    return numbered.group(1).strip().strip("*").strip() if numbered else s

metrics = []
for model in model_names:
    normalized = baseline[f"{model}_label_norm"]
    prediction = normalized.where(
        normalized.notna() & normalized.astype(str).str.strip().ne(""),
        baseline[f"{model}_label"].map(normalize_model_label))
    working = baseline.assign(prediction=prediction.map(normalize_model_label))
    for (variant, format_), group in working.groupby(["prompt_variant", "label_format"]):
        valid = group[group.prediction.isin(categories)]
        assert len(valid) > 0
        metrics.append({"model": model, "prompt_variant": variant, "label_format": format_,
                        "gold_texts": 241, "valid_predictions": len(valid),
                        "percentage_agreement": 100 * (valid.prediction == valid.reader_majority).mean(),
                        "macro_f1": f1_score(valid.reader_majority, valid.prediction,
                                             labels=categories, average="macro", zero_division=0)})
pd.DataFrame(metrics).sort_values(["model", "prompt_variant", "label_format"]).to_csv(
    OUT / "neutral_majority_label_benchmark_verified.csv", index=False)
print(pd.DataFrame(metrics).groupby("model")[["percentage_agreement", "macro_f1"]].mean().round(3))
