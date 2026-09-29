"""
Sanity checks for scripts/emotion_vectors.py, per the required test cases before this pipeline
touches any real project data. Run standalone: python scripts/sanity_check_emotion_vectors.py
Exits non-zero (and prints FAIL) if any check does not match the expected value.
"""
import sys
import math
from emotion_vectors import (
    EMOTIONS, N_EMOTIONS, build_vector, total_variation, distribution_shift_percent,
    distribution_overlap_percent, compare_dominant, new_emotions, disappeared_emotions,
    delta_vector, safe_cosine_similarity, is_in_dominant_set,
    shannon_entropy, normalized_entropy, max_probability, support_size,
)

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def vec(**kwargs):
    """Build a full 11-dim dict, defaulting unspecified emotions to 0."""
    v = {e: 0.0 for e in EMOTIONS}
    v.update(kwargs)
    return v


print("=== 1. No change ===")
neutral = vec(anger=0.5, sadness=0.5)
persona = vec(anger=0.5, sadness=0.5)
tv = total_variation(neutral, persona)
check("TV == 0.0", abs(tv - 0.0) < 1e-9, f"got {tv}")
check("shift% == 0", abs(distribution_shift_percent(neutral, persona) - 0.0) < 1e-6)
check("overlap% == 100", abs(distribution_overlap_percent(neutral, persona) - 100.0) < 1e-6)

print("\n=== 2. Complete replacement ===")
neutral = vec(anger=1.0)
persona = vec(sadness=1.0)
tv = total_variation(neutral, persona)
check("TV == 1.0", abs(tv - 1.0) < 1e-9, f"got {tv}")
check("shift% == 100", abs(distribution_shift_percent(neutral, persona) - 100.0) < 1e-6)

print("\n=== 3. Partial redistribution ===")
neutral = vec(anger=0.5, sadness=0.5)
persona = vec(anger=0.25, sadness=0.75)
tv = total_variation(neutral, persona)
check("TV == 0.25", abs(tv - 0.25) < 1e-9, f"got {tv}")
check("shift% == 25", abs(distribution_shift_percent(neutral, persona) - 25.0) < 1e-6)

print("\n=== 4. New emotion ===")
neutral = vec(anger=0.5, sadness=0.5, guilt=0.0)
persona = vec(anger=0.25, sadness=0.5, guilt=0.25)
tv = total_variation(neutral, persona)
check("TV == 0.25", abs(tv - 0.25) < 1e-9, f"got {tv}")
new = new_emotions(neutral, persona)
check("guilt flagged as new", 'guilt' in new and abs(new['guilt'] - 0.25) < 1e-9, f"got {new}")
check("no other new emotions", len(new) == 1, f"got {new}")

print("\n=== 5. Disappearance ===")
neutral = vec(anger=0.5, sadness=0.5)
persona = vec(anger=1.0, sadness=0.0)
tv = total_variation(neutral, persona)
check("TV == 0.5", abs(tv - 0.5) < 1e-9, f"got {tv}")
disappeared = disappeared_emotions(neutral, persona)
check("sadness flagged as disappeared", 'sadness' in disappeared and abs(disappeared['sadness'] - 0.5) < 1e-9, f"got {disappeared}")
check("no other disappeared emotions", len(disappeared) == 1, f"got {disappeared}")

print("\n=== 6. build_vector from raw votes (worked example from the task spec) ===")
votes = ['sadness', 'sadness', 'fear', 'anger', 'anger']
v = build_vector(votes)
check("sadness == 0.40", abs(v['sadness'] - 0.40) < 1e-9, f"got {v['sadness']}")
check("fear == 0.20", abs(v['fear'] - 0.20) < 1e-9, f"got {v['fear']}")
check("anger == 0.40", abs(v['anger'] - 0.40) < 1e-9, f"got {v['anger']}")
check("sums to 1", abs(sum(v.values()) - 1.0) < 1e-9)
check("remaining emotions are 0", all(v[e] == 0.0 for e in EMOTIONS if e not in ('sadness', 'fear', 'anger')))

print("\n=== 6b. 7 LLM model labels -> 11D ensemble vector (one label per model, not repeated samples) ===")
model_labels = ['anger', 'anger', 'guilt', 'sadness', 'anger', 'guilt', 'anger']  # 7 models' single votes
v_llm = build_vector(model_labels)
check("anger == 4/7", abs(v_llm['anger'] - 4/7) < 1e-9, f"got {v_llm['anger']}")
check("guilt == 2/7", abs(v_llm['guilt'] - 2/7) < 1e-9, f"got {v_llm['guilt']}")
check("sadness == 1/7", abs(v_llm['sadness'] - 1/7) < 1e-9, f"got {v_llm['sadness']}")
check("sums to 1", abs(sum(v_llm.values()) - 1.0) < 1e-9)

print("\n=== 6c. 3 human labels -> 11D vector ===")
human_labels = ['sadness', 'sadness', 'anger']
v_human = build_vector(human_labels)
check("sadness == 2/3", abs(v_human['sadness'] - 2/3) < 1e-9, f"got {v_human['sadness']}")
check("anger == 1/3", abs(v_human['anger'] - 1/3) < 1e-9, f"got {v_human['anger']}")
check("sums to 1", abs(sum(v_human.values()) - 1.0) < 1e-9)

print("\n=== 7. Tied dominant sets ===")
neutral = vec(anger=0.5, sadness=0.5)  # tie
persona = vec(guilt=1.0)
result = compare_dominant(neutral, persona)
check("neutral dominant set is {anger, sadness}", result.neutral_dominant == frozenset({'anger', 'sadness'}))
check("is_tied_neutral is True", result.is_tied_neutral is True)
check("dominant_emotion_changed True (disjoint from tie)", result.dominant_emotion_changed is True)

neutral2 = vec(anger=0.5, sadness=0.5)
persona2 = vec(anger=0.6, joy=0.4)
result2 = compare_dominant(neutral2, persona2)
check("partial overlap NOT a complete change", result2.dominant_emotion_changed is False, f"got {result2}")
check("is_partial_change True", result2.is_partial_change is True)

print("\n=== 8. Delta vector direction ===")
neutral = vec(anger=0.3, sadness=0.4, fear=0.3)
persona = vec(anger=0.6, sadness=0.2, fear=0.2)
d = delta_vector(neutral, persona)
check("anger delta == +0.30", abs(d['anger'] - 0.30) < 1e-9, f"got {d['anger']}")
check("sadness delta == -0.20", abs(d['sadness'] - (-0.20)) < 1e-9, f"got {d['sadness']}")
check("fear delta == -0.10", abs(d['fear'] - (-0.10)) < 1e-9, f"got {d['fear']}")

print("\n=== 9. Cosine similarity between delta vectors ===")
d_human = vec(anger=0.3, sadness=-0.3)
d_llm_same_dir = vec(anger=0.15, sadness=-0.15)
res = safe_cosine_similarity(d_human, d_llm_same_dir)
check("same direction -> cosine == 1.0", res.defined and abs(res.value - 1.0) < 1e-9, f"got {res}")

d_llm_opposite = vec(anger=-0.3, sadness=0.3)
res2 = safe_cosine_similarity(d_human, d_llm_opposite)
check("opposite direction -> cosine == -1.0", res2.defined and abs(res2.value - (-1.0)) < 1e-9, f"got {res2}")

d_zero = vec()
res3 = safe_cosine_similarity(d_human, d_zero)
check("zero LLM delta -> undefined, not NaN/0/1", res3.defined is False and res3.value is None, f"got {res3}")

res4 = safe_cosine_similarity(d_zero, d_zero)
check("both zero -> undefined", res4.defined is False and res4.value is None, f"got {res4}")

print("\n=== 10. Original-majority replication check (baseline validation, tie-aware) ===")
new_neutral_tied = vec(sadness=1/3, anger=1/3, fear=1/3)
check("sadness (original majority) IS in tied dominant set -> replicated",
      is_in_dominant_set(new_neutral_tied, 'sadness') is True)
new_neutral_not_tied = vec(anger=2/3, fear=1/3)
check("sadness NOT in dominant set when absent -> not replicated",
      is_in_dominant_set(new_neutral_not_tied, 'sadness') is False)

print("\n=== 11. Shannon entropy / normalized entropy ===")
v_single = vec(anger=1.0)
check("single-emotion vector -> entropy == 0", abs(shannon_entropy(v_single) - 0.0) < 1e-9,
      f"got {shannon_entropy(v_single)}")
check("single-emotion vector -> normalized entropy == 0", abs(normalized_entropy(v_single) - 0.0) < 1e-9)
v_uniform = {e: 1.0 / N_EMOTIONS for e in EMOTIONS}
check("uniform vector -> entropy == ln(11)", abs(shannon_entropy(v_uniform) - math.log(N_EMOTIONS)) < 1e-9,
      f"got {shannon_entropy(v_uniform)}")
check("uniform vector -> normalized entropy == 1", abs(normalized_entropy(v_uniform) - 1.0) < 1e-9,
      f"got {normalized_entropy(v_uniform)}")
v_two = vec(anger=0.5, sadness=0.5)
expected_h_two = -2 * (0.5 * math.log(0.5))
check("two-way tie vector -> entropy == -2*0.5*ln(0.5)", abs(shannon_entropy(v_two) - expected_h_two) < 1e-9,
      f"got {shannon_entropy(v_two)}")
check("normalized entropy in [0,1] for all test vectors",
      all(0.0 <= normalized_entropy(v) <= 1.0 for v in (v_single, v_uniform, v_two)))

print("\n=== 12. Max probability (consensus strength) ===")
check("single-emotion vector -> max_probability == 1.0", abs(max_probability(v_single) - 1.0) < 1e-9)
check("uniform vector -> max_probability == 1/11", abs(max_probability(v_uniform) - 1.0 / N_EMOTIONS) < 1e-9)
check("two-way tie vector -> max_probability == 0.5", abs(max_probability(v_two) - 0.5) < 1e-9)
v_llm_47 = build_vector(['anger', 'anger', 'guilt', 'sadness', 'anger', 'guilt', 'anger'])
check("4/7 anger vector -> max_probability == 4/7", abs(max_probability(v_llm_47) - 4 / 7) < 1e-9,
      f"got {max_probability(v_llm_47)}")

print("\n=== 13. Support size (distinct emotions with >0 mass) ===")
check("single-emotion vector -> support_size == 1", support_size(v_single) == 1)
check("uniform vector -> support_size == 11", support_size(v_uniform) == N_EMOTIONS)
check("two-way tie vector -> support_size == 2", support_size(v_two) == 2)
check("4/7 anger, 2/7 guilt, 1/7 sadness -> support_size == 3", support_size(v_llm_47) == 3,
      f"got {support_size(v_llm_47)}")
v_empty_like = vec()  # all-zero (not a real vector -- built from votes this would be None instead)
check("all-zero placeholder -> support_size == 0", support_size(v_empty_like) == 0)

print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
else:
    print("ALL SANITY CHECKS PASSED")
