# Q-ADAPT

**Quantum-Assisted Adaptive Cyber Defense Optimization Using ML-Driven Attack Graphs**

> *A Hybrid Quantum–Classical Framework for Adaptive Cyber Defense Optimization Using
> Machine-Learning-Based Threat Assessment and Dynamic Attack Graphs*

Q-ADAPT is an adaptive, quantum-assisted **cyber-defense decision engine**: neither a
"quantum IDS" nor "QAOA for cybersecurity" in general. Its pipeline:

1. Machine learning estimates **how dangerous** current activity is.
2. A probabilistic attack graph models **how it can propagate**.
3. A QUBO captures **which combination of defensive actions** to take under limited
   budget, time and tolerance for business disruption.
4. QAOA, checked against classical baselines, searches that QUBO.
5. The loop **re-optimizes as the threat evolves**.

```
Detect → Understand → Predict → Optimize → Defend → Learn
```

![CI](https://github.com/Usha-125/Q-ADAPT/actions/workflows/ci.yml/badge.svg)

---

## Highlights

| Layer | What is implemented |
|---|---|
| **ML threat engine** | Loaders for CSE-CIC-IDS2018, CIC-IDS2017 (unified CICFlowMeter schema → cross-dataset validation) and UNSW-NB15; RF / HistGradientBoosting / MLP / XGBoost; host-level threat aggregation (probability, attack type, confidence, severity) |
| **Dynamic attack graph** | Tiered enterprise generator (10–500 nodes) + fixed demo network; CVE catalog and NVD API 2.0 parser; noisy-OR risk propagation (cycle-safe, batched); k-best attack paths |
| **Risk engine** | Three interchangeable risk models (propagation, T×V×C×P, severity-only) for the ablation study; structural criticality; downstream exposure |
| **Defense engine** | 10 action types with cost / time / disruption / effectiveness; exact batched effect model on the graph; policy constraints (conflicts, prerequisites, protected assets) |
| **Quantum engine** | Defense QUBO with a fitted quadratic surrogate of the *true* propagated risk; slack or unbalanced-penalty constraint encodings; Ising mapping; QAOA (NumPy state-vector engine verified against Qiskit to 1e-16, plus Qiskit Aer with gate-level noise), INTERP initialization, CVaR, warm starts; noise models |
| **Classical baselines** | Exhaustive (true optimum), score ranking (naive SOC practice), interaction-aware greedy, simulated annealing, genetic algorithm, MILP (HiGHS, McCormick linearization), equal-budget random sampling |
| **Adaptive engine (AQDO)** | Ground-truth attack environment, Bayesian effectiveness learning, risk-appetite objective re-weighting, QUBO regeneration, QAOA parameter warm starts, policies none / static / reoptimize / aqdo |
| **Explainability** | Why each action was selected (threat, criticality, downstream exposure, contribution) and why alternatives were not (infeasible, redundant, poor value, slow) |
| **SOC dashboard** | React + Cytoscape.js + Recharts: overview, threat detection, attack graph, risk analysis, quantum optimizer, defense plan with human approval, solver benchmark |
| **Experiments** | Eight reproducible experiments with 95% CIs and Wilcoxon tests (see [`results/`](results/)) |

## Architecture

```
            Cybersecurity data (IDS flows, CVE/NVD, topology)
                               │
                 ┌─────────────▼─────────────┐
                 │  ML threat engine         │  ml_engine/
                 │  P(attack), type, conf.   │
                 └─────────────┬─────────────┘
                 ┌─────────────▼─────────────┐
                 │  Dynamic attack graph     │  attack_graph/
                 │  noisy-OR propagation     │
                 └─────────────┬─────────────┘
                 ┌─────────────▼─────────────┐
                 │  Risk engine              │  risk_engine/
                 └─────────────┬─────────────┘
                 ┌─────────────▼─────────────┐
                 │  Defense candidates +     │  defense_engine/
                 │  effect model + policy    │
                 └─────────────┬─────────────┘
                 ┌─────────────▼─────────────┐
                 │  Defense QUBO             │  quantum_engine/qubo_builder.py
                 └──────┬──────────────┬─────┘
         ╔══════════════▼═══╗   ┌──────▼───────────────┐
         ║ QAOA (simulated) ║   │ Classical baselines  │  classical_baselines/
         ╚══════════════╤═══╝   └──────┬───────────────┘
                 ┌──────▼──────────────▼─────┐
                 │ Portfolio decoded on the  │  optimization/problem.py
                 │ ground-truth objective    │
                 └─────────────┬─────────────┘
                 ┌─────────────▼─────────────┐
                 │ Explanation + analyst     │  explainability/, api/, frontend/
                 │ approve / reject          │
                 └─────────────┬─────────────┘
                               └──► AQDO feedback loop  adaptive_engine/
```

Classical components do the ML, graphs, risk modelling, QUBO construction and decoding.
The quantum component is QAOA, run in simulation, with an optional path to cloud hardware.
Details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and the math is in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Quick start

```bash
# Python 3.10+
pip install -e ".[quantum,api,dev]"     # add ,xgboost for the XGBoost model

# one end-to-end decision on the demo network (QAOA p=2)
qadapt demo

# API + dashboard
cd frontend && npm install && npm run build && cd ..
qadapt serve            # http://127.0.0.1:8000  (API docs at /docs)
# or develop the UI with hot reload:  cd frontend && npm run dev  (proxies /api to :8000)

# tests
pytest
```

Example `qadapt demo` output (risk numbers come from the model, not hard-coded):

```
 solver          : qaoa_p2   (1.6s, feasible=True)
 risk            : 79.5% -> 34.8%   (-56.2%)
 attack paths    : 19 -> 14
 [+] Revoke credentials on APP-01
       - EXPLOIT activity on APP-01 (p=91%)
       - Asset criticality 0.75; downstream exposure HIGH
 [ ] Isolate APP-01
       - Largely redundant with revoke_credentials:APP-01
```

### CLI

| Command | Purpose |
|---|---|
| `qadapt demo [--solver qaoa\|milp\|greedy…] [--p 2] [--noise medium]` | one decision with explanations |
| `qadapt train --csv data/raw/*.csv --model random_forest` | train and save the ML detector (synthetic data without `--csv`) |
| `qadapt episode --policy aqdo --steps 8` | simulate an adaptive defense campaign |
| `qadapt serve` | FastAPI + built dashboard |
| `python -m experiments.run_all [--quick]` | full research experiment suite |

### API (selection)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/overview` | KPIs, timeline, applied defenses |
| GET/POST | `/api/threats`, `/api/threats/detect` | threat list, manual injection, ML detection on simulated traffic |
| GET | `/api/graph`, `/api/risk?model=` | Cytoscape attack graph, risk report and attack paths |
| POST | `/api/optimize` | build QUBO and solve, returning plan, metrics, QUBO and explanations |
| POST | `/api/runs/{id}/decision` | analyst approves or rejects (all or individual actions) |
| POST | `/api/compare` | all solvers on the same instance |
| POST | `/api/adaptive/next-stage` | advance the scripted multi-stage attack |

## Repository layout

```
qadapt/
  core/                 domain model (assets, vulnerabilities, threats, actions) and config
  ml_engine/            datasets, synthetic flows, preprocessing, models, ThreatDetector
  attack_graph/         topology, CVE/NVD, graph builder, propagation, paths, visualizer
  risk_engine/          risk models, criticality, threat scoring
  defense_engine/       action generator, effect model, policy constraints
  optimization/         DefenseProblem (ground-truth objective) and Solver interface
  quantum_engine/       QUBO builder, Ising, QAOA, Qiskit backend, noise models
  classical_baselines/  exhaustive / greedy / SA / GA / MILP / random
  adaptive_engine/      environment, learning, AQDO controller, staged scenario
  explainability/       action explanations
  api/                  FastAPI app, SQLite storage, session
  pipeline.py           end-to-end decision pipeline
  cli.py                command-line interface
frontend/               React SOC dashboard
experiments/            Experiments 1–8 (+ run_all)
results/                generated experiment results (JSON + Markdown)
docs/                   architecture, methodology, datasets, experiments, roadmap
tests/                  pytest suite
```

## Scientific positioning and claims

Q-ADAPT is set up to avoid the common pitfalls in quantum-security research:

* **No quantum-advantage claim.** QAOA runs in classical simulation. At the sizes that
  can be simulated, exhaustive search is faster. The experiments measure **solution
  quality** against the exact optimum, never speed.
* **Fair baselines.** Every solver is scored on the same ground-truth objective. That
  objective is the propagated residual risk, not the QUBO surrogate. Baselines include
  the true optimum, MILP, metaheuristics, naive score ranking and an equal-budget
  random-sampling control for QAOA's shots.
* **The surrogate is measured.** Each QUBO reports how well its quadratic risk surrogate
  agrees with the true risk function (R² and Spearman correlation).
* **"Adaptive" means something concrete.** The QUBO is regenerated from the new attack
  state, effectiveness estimates are updated from observed outcomes, and objective
  weights follow the organization's risk appetite. The experiments compare against a
  static policy and against plain re-optimization.
* **Data honesty.** The public IDS datasets contain no topology, asset criticality or
  defense metadata. Q-ADAPT therefore uses real datasets for the ML layer and controlled
  synthetic enterprises for the decision layer. Anything run on synthetic flows is
  labelled as such.
* **Human in the loop.** Recommendations are never executed automatically.

On patents: the generic idea of attack graph + QUBO + QAOA has prior art. The specific
adaptive mechanism (fitted risk surrogate, effectiveness learning, risk-appetite
re-weighting, warm-started re-optimization) may be worth investigating, but only after a
professional prior-art search. Nothing in this repository is a patentability opinion.

## License

MIT. See [LICENSE](LICENSE).
