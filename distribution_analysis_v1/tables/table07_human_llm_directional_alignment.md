# Table 7: Human Llm Directional Alignment

Cosine similarity of DELTA vectors (never raw distributions); NA excluded when either side had zero shift. Permutation baseline: 10,000 shuffles of which LLM delta is paired with which human delta, at the text level.

|   n_texts |   n_defined_cosine |   mean_cosine_observed |   permutation_null_mean |   permutation_null_std |   permutation_p_value |
|----------:|-------------------:|-----------------------:|------------------------:|-----------------------:|----------------------:|
|       300 |                251 |                 0.1269 |                  0.0094 |                 0.0211 |                0.0001 |