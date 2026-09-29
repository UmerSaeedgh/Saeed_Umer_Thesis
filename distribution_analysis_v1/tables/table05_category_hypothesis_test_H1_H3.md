# Table 5: Category Hypothesis Test H1 H3

Kruskal-Wallis global test, Holm-corrected planned pairwise Mann-Whitney U, Cliff's delta effect sizes, and a permutation-based ordered-trend test (H3's predicted increasing order: Unambiguous < Author-Independent < Author-Relevant).

| source                                                           | comparison                                                |   p_raw |   p_holm |   cliffs_delta |   kruskal_wallis_p |   trend_test_J |   trend_test_p |
|:-----------------------------------------------------------------|:----------------------------------------------------------|--------:|---------:|---------------:|-------------------:|---------------:|---------------:|
| HUMAN (text-level, primary/confirmatory)                         | Unambiguous vs Author-Independent Ambiguous               | 0.00769 |  0.01538 |        -0.217  |              1e-05 |        18859.5 |         0.0001 |
| HUMAN (text-level, primary/confirmatory)                         | Author-Independent Ambiguous vs Author-Relevant Ambiguous | 0.03563 |  0.03563 |        -0.1707 |              1e-05 |        18859.5 |         0.0001 |
| HUMAN (text-level, primary/confirmatory)                         | Unambiguous vs Author-Relevant Ambiguous                  | 0       |  1e-05   |        -0.3842 |              1e-05 |        18859.5 |         0.0001 |
| LLM (text-level, averaged across configs, secondary/exploratory) | Unambiguous vs Author-Independent Ambiguous               | 0.0014  |  0.0014  |        -0.2603 |              0     |        22001.5 |         0.0001 |
| LLM (text-level, averaged across configs, secondary/exploratory) | Author-Independent Ambiguous vs Author-Relevant Ambiguous | 0       |  0       |        -0.5101 |              0     |        22001.5 |         0.0001 |
| LLM (text-level, averaged across configs, secondary/exploratory) | Unambiguous vs Author-Relevant Ambiguous                  | 0       |  0       |        -0.6299 |              0     |        22001.5 |         0.0001 |