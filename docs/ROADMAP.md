# Roadmap and status against the project proposal

| Phase (proposal §59) | Status | Where |
|---|---|---|
| 1. Literature + problem formulation | ✅ formal model | `docs/METHODOLOGY.md` |
| 2. Dataset + ML | ✅ loaders (2018 / 2017 / UNSW-NB15), RF / HGB / MLP / XGBoost, host aggregation · ⏳ run on the full public datasets | `qadapt/ml_engine`, `experiments/experiment1_ml.py` |
| 3. Attack graph | ✅ topologies 10–500 nodes, CVE/NVD, propagation, paths | `qadapt/attack_graph` |
| 4. Defense model | ✅ 10 action types, effect model, policy constraints | `qadapt/defense_engine` |
| 5. QUBO | ✅ fitted surrogate, slack and unbalanced encodings, Ising | `qadapt/quantum_engine/qubo_builder.py`, `ising.py` |
| 6. QAOA | ✅ state-vector + Qiskit Aer, INTERP, CVaR, warm start, noise | `qadapt/quantum_engine/qaoa.py` |
| 7. Classical baselines | ✅ exhaustive, greedy, SA, GA, MILP, ranking, random | `qadapt/classical_baselines` |
| 8. Adaptive engine | ✅ AQDO: environment, learning, re-weighting, re-optimization | `qadapt/adaptive_engine` |
| 9. Dashboard | ✅ React + FastAPI, six screens + benchmark, human approval | `frontend/`, `qadapt/api` |
| 10. Research experiments | ✅ Experiments 1–8 with CIs and significance tests | `experiments/`, `results/` |
| 11. Paper + patent investigation | ⏳ write-up; professional prior-art search required | `paper/` |

## Next steps

1. **Real ML results.** Run Experiment 1 on the CSE-CIC-IDS2018 / CIC-IDS2017 /
   UNSW-NB15 CSVs, including the cross-dataset protocol. Store the trained detector in
   `models/` and point `QADAPT_MODEL` at it.
2. **Hardware run.** Swap the Aer simulator in `qiskit_backend.py` for an IBM Runtime
   `SamplerV2` and run a small (≤ 10 qubit) instance at p = 1. Report it as a hardware
   status check, not as an advantage.
3. **Larger instances.** Decompose by attack-path clusters instead of global
   pre-screening, and compare against MILP on 50–200 actions.
4. **Learning signal.** Experiments 6 and 8 show that re-optimization carries all of
   the measured adaptive benefit. Test longer campaigns and stronger effectiveness mis-specification
   to see whether Bayesian learning and re-weighting add a significant gain on top.
5. **Calibration.** Replace assumed edge probabilities with data such as EPSS scores and
   incident timelines.
6. **Paper.** Follow the structure in proposal §60, with the threats-to-validity section
   based on `docs/METHODOLOGY.md` §8.
