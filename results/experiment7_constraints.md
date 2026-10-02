# Experiment 7 - resource constraints and constraint encodings

8 instances, n = 10 candidate actions. Limits are fractions of the total over all candidates (B: 25 % cost; C: 15 % time; D: 15 % disruption; E: 30/20/20 % + max 3 actions). 'QUBO min feasible' = the exact QUBO minimiser satisfies the true constraints.

| scenario | solver | qubits | feasible | QUBO min feasible | optimality gap | risk reduction | optimal risk reduction |
|---|---|---|---|---|---|---|---|
| A unlimited | qaoa_p2 (unbalanced) | 10 | 100% | 100% | 0.0084 ± 0.0138 | 0.586 ± 0.157 | 0.610 ± 0.168 |
| A unlimited | qaoa_p2 (slack) | 10 | 100% | 100% | 0.0084 ± 0.0138 | 0.586 ± 0.157 | 0.610 ± 0.168 |
| A unlimited | simulated_annealing | - | 100% | - | 0.0000 ± 0.0000 | 0.610 ± 0.168 | 0.610 ± 0.168 |
| A unlimited | milp | - | 100% | - | 0.0401 ± 0.0732 | 0.598 ± 0.186 | 0.610 ± 0.168 |
| B budget | qaoa_p2 (unbalanced) | 10 | 100% | 100% | 0.0057 ± 0.0135 | 0.491 ± 0.139 | 0.502 ± 0.137 |
| B budget | qaoa_p2 (slack) | 14 | 100% | 50% | 0.0284 ± 0.0509 | 0.477 ± 0.125 | 0.502 ± 0.137 |
| B budget | simulated_annealing | - | 100% | - | 0.0000 ± 0.0000 | 0.502 ± 0.137 | 0.502 ± 0.137 |
| B budget | milp | - | 100% | - | 0.0455 ± 0.0863 | 0.449 ± 0.111 | 0.502 ± 0.137 |
| C time | qaoa_p2 (unbalanced) | 10 | 100% | 100% | 0.0000 ± 0.0000 | 0.440 ± 0.102 | 0.440 ± 0.102 |
| C time | qaoa_p2 (slack) | 14 | 100% | 62% | 0.0000 ± 0.0000 | 0.440 ± 0.102 | 0.440 ± 0.102 |
| C time | simulated_annealing | - | 100% | - | 0.0000 ± 0.0000 | 0.440 ± 0.102 | 0.440 ± 0.102 |
| C time | milp | - | 100% | - | 0.0061 ± 0.0081 | 0.420 ± 0.115 | 0.440 ± 0.102 |
| D disruption | qaoa_p2 (unbalanced) | 10 | 100% | 100% | 0.0140 ± 0.0331 | 0.374 ± 0.092 | 0.400 ± 0.084 |
| D disruption | qaoa_p2 (slack) | 14 | 100% | 75% | 0.0135 ± 0.0320 | 0.386 ± 0.074 | 0.400 ± 0.084 |
| D disruption | simulated_annealing | - | 100% | - | 0.0000 ± 0.0000 | 0.400 ± 0.084 | 0.400 ± 0.084 |
| D disruption | milp | - | 100% | - | 0.0049 ± 0.0077 | 0.402 ± 0.085 | 0.400 ± 0.084 |
| E combined | qaoa_p2 (unbalanced) | 10 | 100% | 100% | 0.0000 ± 0.0000 | 0.379 ± 0.101 | 0.379 ± 0.101 |
| E combined | simulated_annealing | - | 100% | - | 0.0000 ± 0.0000 | 0.379 ± 0.101 | 0.379 ± 0.101 |
| E combined | milp | - | 100% | - | 0.0054 ± 0.0075 | 0.363 ± 0.087 | 0.379 ± 0.101 |
