# Paper outline

**Working title:** Q-ADAPT: A Hybrid Quantum–Classical Framework for Adaptive Cyber Defense
Optimization Using Machine-Learning-Based Threat Assessment and Dynamic Attack Graphs

| § | Section | Source material in this repository |
|---|---|---|
| 1 | Introduction: detection is not decision; defense as combinatorial optimization | proposal §2, README |
| 2 | Related work: IDS, attack graphs (NIST IR 7788), cyber-risk optimization, QUBO/QAOA, quantum cybersecurity, prior art | literature review (to write) |
| 3 | Problem formulation: G = (V, E), portfolio x ∈ {0,1}^n, ground-truth objective J(x), constraints | `docs/METHODOLOGY.md` §2–4 |
| 4 | Q-ADAPT framework | `docs/ARCHITECTURE.md` |
| 5 | ML threat assessment | METHODOLOGY §1, `results/experiment1_ml.md` |
| 6 | Dynamic attack graph and risk propagation | METHODOLOGY §2, `results/experiment2_graph.md` |
| 7 | Cyber-defense QUBO: fitted surrogate, interaction terms, constraint encodings | METHODOLOGY §5 (main contribution) |
| 8 | Quantum optimization: Ising, QAOA, decoding against the true objective | METHODOLOGY §6 |
| 9 | AQDO adaptive algorithm | METHODOLOGY §7 |
| 10 | Experimental setup: instances, seeds, baselines, simulator, noise levels | `experiments/` |
| 11 | Results | `results/experiment3_optimization.md`, `experiment5_noise.md`, `experiment6_adaptive.md`, `experiment7_constraints.md` |
| 12 | Ablation | `results/experiment8_ablation.md` |
| 13 | Discussion: no speed-up claimed; surrogate fidelity; scalability via pre-screening | |
| 14 | Threats to validity: synthetic topologies, assumed probabilities and costs, global-depolarizing approximation, small instances | METHODOLOGY §8 |
| 15 | Conclusion | |
| 16 | Future work: hardware runs, decomposition for large graphs, calibration with EPSS/incident data | `docs/ROADMAP.md` |

Reproduce every table with `python -m experiments.run_all` (seeds are fixed).
