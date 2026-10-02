# Experiment 2 - attack graph scalability

Mean over 5 random topologies per size; 2 ML threats injected per graph.

| target size | nodes | edges | density | build ms | propagation ms | batched propagation ms/portfolio | viable paths (k=5/target) | path analysis ms | total risk |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 11 | 15 | 0.1364 | 0.8 | 0.44 | 0.007 | 6.6 | 0.8 | 0.428 |
| 25 | 26 | 61 | 0.0932 | 1.4 | 0.99 | 0.026 | 10.8 | 1.2 | 0.342 |
| 50 | 51 | 183 | 0.0717 | 2.8 | 2.03 | 0.090 | 13.0 | 2.6 | 0.362 |
| 100 | 101 | 626 | 0.0620 | 7.3 | 1.25 | 0.234 | 38.6 | 14.2 | 0.668 |
| 250 | 251 | 3442 | 0.0549 | 35.2 | 1.23 | 0.754 | 129.0 | 102.2 | 0.846 |
| 500 | 501 | 13268 | 0.0530 | 126.8 | 1.93 | 4.129 | 267.0 | 582.8 | 0.895 |
