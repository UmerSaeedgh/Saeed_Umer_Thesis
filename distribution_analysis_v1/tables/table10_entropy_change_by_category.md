# Table 10: Entropy Change By Category

| source   | category                     |   n |   mean_entropy_change | direction                   |   wilcoxon_p_raw |   p_bh_fdr |
|:---------|:-----------------------------|----:|----------------------:|:----------------------------|-----------------:|-----------:|
| human    | Unambiguous                  | 100 |                0.1984 | more dispersed (entropy up) |           0      |     0.0002 |
| human    | Author-Independent Ambiguous | 100 |                0.1714 | more dispersed (entropy up) |           0.0001 |     0.0002 |
| human    | Author-Relevant Ambiguous    | 100 |                0.161  | more dispersed (entropy up) |           0.0015 |     0.003  |
| llm      | Unambiguous                  | 100 |                0.0287 | more dispersed (entropy up) |           0.1395 |     0.2093 |
| llm      | Author-Independent Ambiguous | 100 |                0.012  | more dispersed (entropy up) |           0.2173 |     0.2608 |
| llm      | Author-Relevant Ambiguous    | 100 |                0.0189 | more dispersed (entropy up) |           0.3669 |     0.3669 |