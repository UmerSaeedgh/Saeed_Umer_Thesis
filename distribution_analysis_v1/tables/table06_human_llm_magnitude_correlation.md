# Table 6: Human Llm Magnitude Correlation

Spearman rho with text-clustered bootstrap 95% CI (10,000 resamples at the text_id level, never row-level).

| category                     |   spearman_rho |   p_cluster_bootstrap |   ci95_lo |   ci95_hi |
|:-----------------------------|---------------:|----------------------:|----------:|----------:|
| ALL                          |         0.1891 |                0.0001 |    0.1227 |    0.2545 |
| Unambiguous                  |         0.1463 |                0.0272 |    0.0151 |    0.2731 |
| Author-Independent Ambiguous |         0.0633 |                0.2536 |   -0.0447 |    0.1653 |
| Author-Relevant Ambiguous    |         0.2035 |                0.0012 |    0.0865 |    0.3089 |