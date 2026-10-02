# Experiment 8 - ablation study

A  ML only                     severity-only risk, alert-driven candidates, rank-by-score
B  ML + attack graph           propagation risk, alert-driven candidates, rank-by-score
C  + risk engine               propagation risk, risk/path-driven candidates, rank-by-score
D  + classical optimisation    as C, interaction-aware simulated annealing
E  + QAOA                      as C, QAOA (p = 1)
F  full Q-ADAPT                as E + AQDO adaptation (re-optimisation, learning, re-weighting)

A-E decide once (static policy); F re-optimises every step. All are evaluated in the same
ground-truth attack environments (same seeds), by realised criticality-weighted loss.


20 seeds x 8 steps per environment; budget 0.5, max 4 actions per decision.

| environment | model | cumulative loss | final loss | critical assets lost | actions | disruption |
|---|---|---|---|---|---|---|
| demo (14 nodes) | A ML only | 5.077 ± 0.468 | 0.897 ± 0.035 | 4.65 | 1.9 | 0.01 |
| demo (14 nodes) | B ML + graph | 3.193 ± 0.437 | 0.629 ± 0.069 | 2.50 | 2.6 | 0.42 |
| demo (14 nodes) | C + risk engine | 3.000 ± 0.405 | 0.555 ± 0.061 | 1.95 | 2.5 | 0.72 |
| demo (14 nodes) | D + classical opt. | 2.851 ± 0.429 | 0.574 ± 0.087 | 2.05 | 2.1 | 0.55 |
| demo (14 nodes) | E + QAOA | 2.914 ± 0.401 | 0.583 ± 0.079 | 2.00 | 2.1 | 0.60 |
| demo (14 nodes) | F full Q-ADAPT | 1.435 ± 0.160 | 0.207 ± 0.025 | 0.00 | 10.9 | 2.00 |
| enterprise-40 | A ML only | 0.758 ± 0.334 | 0.198 ± 0.081 | 1.60 | 1.3 | 0.02 |
| enterprise-40 | B ML + graph | 0.574 ± 0.317 | 0.150 ± 0.076 | 1.25 | 2.9 | 0.28 |
| enterprise-40 | C + risk engine | 0.462 ± 0.268 | 0.119 ± 0.064 | 0.90 | 3.3 | 0.71 |
| enterprise-40 | D + classical opt. | 0.511 ± 0.290 | 0.136 ± 0.069 | 1.10 | 2.5 | 0.31 |
| enterprise-40 | E + QAOA | 0.511 ± 0.290 | 0.136 ± 0.069 | 1.10 | 2.5 | 0.35 |
| enterprise-40 | F full Q-ADAPT | 0.161 ± 0.106 | 0.029 ± 0.022 | 0.10 | 12.9 | 1.75 |

## Incremental contribution (Wilcoxon signed-rank on cumulative loss)

| environment | step | mean difference (neg. = improvement) | p-value |
|---|---|---|---|
| demo (14 nodes) | B ML + graph vs A ML only | -1.8841 | 0.0002523 |
| demo (14 nodes) | C + risk engine vs B ML + graph | -0.1929 | 0.1912 |
| demo (14 nodes) | D + classical opt. vs C + risk engine | -0.1489 | 0.6402 |
| demo (14 nodes) | E + QAOA vs D + classical opt. | +0.0626 | 0.6721 |
| demo (14 nodes) | F full Q-ADAPT vs E + QAOA | -1.4786 | 1.907e-06 |
| enterprise-40 | B ML + graph vs A ML only | -0.1838 | 0.04109 |
| enterprise-40 | C + risk engine vs B ML + graph | -0.1118 | 0.07699 |
| enterprise-40 | D + classical opt. vs C + risk engine | +0.0490 | 0.2507 |
| enterprise-40 | E + QAOA vs D + classical opt. | +0.0000 | 1 |
| enterprise-40 | F full Q-ADAPT vs E + QAOA | -0.3504 | 0.001068 |
