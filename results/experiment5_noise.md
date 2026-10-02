# Experiment 5 - quantum noise sensitivity

10 instances with n = 10 actions. Noise levels (p1 / p2 / readout): ideal: 0/0/0, low: 0.0001/0.001/0.01, medium: 0.0005/0.005/0.03, high: 0.002/0.02/0.06. Global-depolarising approximation in the state-vector engine.

| p | noise | approx. ratio | P(optimal state) | optimality gap | same plan as ideal | Jaccard vs optimum |
|---|---|---|---|---|---|---|
| 1 | ideal | 0.965 ± 0.002 | 0.0104 ± 0.0008 | 0.0000 ± 0.0000 | 100% | 1.000 ± 0.000 |
| 1 | low | 0.948 ± 0.002 | 0.0094 ± 0.0009 | 0.0258 ± 0.0346 | 60% | 0.817 ± 0.182 |
| 1 | medium | 0.898 ± 0.002 | 0.0068 ± 0.0005 | 0.0183 ± 0.0328 | 60% | 0.817 ± 0.173 |
| 1 | high | 0.815 ± 0.002 | 0.0024 ± 0.0001 | 0.0627 ± 0.0657 | 40% | 0.615 ± 0.251 |
| 2 | ideal | 0.975 ± 0.002 | 0.0138 ± 0.0015 | 0.0051 ± 0.0116 | 100% | 0.925 ± 0.170 |
| 2 | low | 0.943 ± 0.002 | 0.0117 ± 0.0012 | 0.0013 ± 0.0029 | 80% | 0.950 ± 0.113 |
| 2 | medium | 0.861 ± 0.002 | 0.0061 ± 0.0006 | 0.0433 ± 0.0634 | 40% | 0.700 ± 0.256 |
| 2 | high | 0.792 ± 0.002 | 0.0013 ± 0.0000 | 0.0661 ± 0.0801 | 30% | 0.645 ± 0.238 |
| 3 | ideal | 0.979 ± 0.002 | 0.0153 ± 0.0022 | 0.0191 ± 0.0327 | 100% | 0.800 ± 0.193 |
| 3 | low | 0.932 ± 0.002 | 0.0118 ± 0.0017 | 0.0161 ± 0.0332 | 80% | 0.900 ± 0.161 |
| 3 | medium | 0.835 ± 0.002 | 0.0045 ± 0.0006 | 0.0272 ± 0.0394 | 70% | 0.850 ± 0.182 |
| 3 | high | 0.788 ± 0.002 | 0.0010 ± 0.0000 | 0.0847 ± 0.0694 | 0% | 0.420 ± 0.122 |

Note: final plans are decoded from samples and scored on the true objective, so a feasible good plan can survive noise even when the approximation ratio drops.
