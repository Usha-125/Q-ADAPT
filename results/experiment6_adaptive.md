# Experiment 6 - dynamic adaptation

## A. Scripted multi-stage attack (demo network, QAOA p = 1)

| stage | event | risk before | risk after | recommended defense |
|---|---|---|---|---|
| T1 | Attack detected on Web Server | 0.775 | 0.249 | Isolate WEB-01, Block malicious IP 203.0.113.7 |
| T2 | Attacker moved to Application Server | 0.599 | 0.246 | Revoke credentials on APP-01, Increase monitoring on APP-01, Isolate DC-01 |
| T3 | Database becomes the target | 0.376 | 0.297 | Revoke credentials on DB-01, Increase monitoring on DB-01, Increase monitoring on WEB-01 |

## B. Stochastic campaigns: static vs adaptive policies

20 seeds x 8 decision steps; solver = qaoa; true effectiveness differs from nominal for: block_ip=0.25, patch_vulnerability=0.5, increase_monitoring=0.1. Loss = criticality-weighted fraction of compromised assets (lower is better).

| environment | policy | cumulative loss | final loss | critical assets lost | actions | disruption | QAOA evals |
|---|---|---|---|---|---|---|---|
| demo (14 nodes) | none | 5.077 ± 0.468 | 0.897 ± 0.035 | 4.65 | 0.0 | 0.00 | 0 |
| demo (14 nodes) | static | 2.976 ± 0.409 | 0.589 ± 0.071 | 2.05 | 2.0 | 0.54 | 80 |
| demo (14 nodes) | reoptimize | 1.501 ± 0.164 | 0.213 ± 0.022 | 0.00 | 10.8 | 1.73 | 512 |
| demo (14 nodes) | aqdo | 1.519 ± 0.167 | 0.218 ± 0.024 | 0.00 | 11.1 | 1.92 | 498 |
| enterprise-40 | none | 0.759 ± 0.334 | 0.198 ± 0.081 | 1.60 | 0.0 | 0.00 | 0 |
| enterprise-40 | static | 0.511 ± 0.290 | 0.136 ± 0.069 | 1.10 | 2.5 | 0.31 | 80 |
| enterprise-40 | reoptimize | 0.161 ± 0.106 | 0.029 ± 0.022 | 0.10 | 12.7 | 1.66 | 572 |
| enterprise-40 | aqdo | 0.161 ± 0.106 | 0.029 ± 0.022 | 0.10 | 12.3 | 1.69 | 567 |

### Paired tests (Wilcoxon signed-rank on cumulative loss)

| environment | comparison | mean difference | p-value |
|---|---|---|---|
| demo (14 nodes) | aqdo vs static | -1.4569 | 1.907e-06 |
| demo (14 nodes) | aqdo vs reoptimize | +0.0181 | 0.2157 |
| demo (14 nodes) | reoptimize vs static | -1.4750 | 1.907e-06 |
| enterprise-40 | aqdo vs static | -0.3504 | 0.001068 |
| enterprise-40 | aqdo vs reoptimize | +0.0000 | 1 |
| enterprise-40 | reoptimize vs static | -0.3504 | 0.001068 |
