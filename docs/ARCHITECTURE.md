# Architecture

## Data flow

```
ThreatDetector.assess_hosts(flows) ──► [ThreatAssessment]
                                            │ AttackGraph.apply_threats
                                            ▼
AttackGraph (networkx, mutable state) ──► CompiledGraph (numpy/scipy arrays)
                                            │
             ActionGenerator.generate() ──► [DefenseAction]
                                            │
             DefenseEvaluator(ag, actions)  │  batched residual risk R(x)
                                            ▼
             DefenseProblem(evaluator, OptimizationConfig, PolicySet)
                 ├── objective(X)    ground truth J(x), batched + cached
                 ├── feasible(x)     constraints + policy
                 └── qubo()          build_defense_qubo → QUBO
                                            │
               Solver.solve(problem) ──► SolveResult(x, objective, metrics, info)
                 ├── QAOASolver  (state-vector | Qiskit Aer)
                 └── BASELINES   (exhaustive, greedy, SA, GA, MILP, …)
                                            │
             QAdaptPipeline.decide() ──► DecisionReport (+ explanations, paths, QUBO)
                                            │
             FastAPI /api/optimize ──► dashboard ──► analyst decision
                                            │
             AQDOController (observe → learn → re-weight → regenerate → solve)
```

## Key design decisions

| Decision | Rationale |
|---|---|
| A single **ground-truth objective** (`DefenseProblem.objective`) shared by all solvers | Fair comparison. The QUBO is only a model, and its error is measured instead of hidden. |
| **Batched** propagation through a sparse incidence matrix and the log-domain effect model | Scores thousands of portfolios per second, which the exhaustive optimum, the regression surrogate and the metaheuristics all need. |
| A **custom NumPy QAOA** next to Qiskit | About 10–100× faster for parameter sweeps and noise studies. Exact agreement with Qiskit is tested. Qiskit/Aer stays available for gate-level noise and hardware. |
| QUBO **surrogate fitted to the true risk** (regression) | Gives high-fidelity quadratic models. The interaction terms `J_ij` (redundancy and synergy) come out of the risk model itself rather than being hand-specified. |
| **Unbalanced penalization** as the default encoding | No slack qubits, so a 12-action problem needs 12 qubits instead of 12 + 4 per constraint. |
| `AttackGraph.apply_defenses` **re-derives** state from base probabilities | Approved plans realize exactly the predicted risk (tested). Re-derivation also lets AQDO re-apply defenses whenever effectiveness estimates change. |
| **Human-in-the-loop** approval | Disruptive actions are never automated. Decisions are logged to SQLite for audit. |
| Qiskit/Aer imported on the **main thread** in the API | Initializing their native extensions from request worker threads caused intermittent segfaults. |

## Modules

| Package | Main types / functions |
|---|---|
| `core` | `Asset`, `Vulnerability`, `ThreatAssessment`, `DefenseAction`, `ObjectiveWeights`, `OptimizationConfig` |
| `ml_engine` | `load_cic_csvs`, `load_unsw_nb15`, `generate_flows`, `train_and_evaluate`, `ThreatDetector` |
| `attack_graph` | `Topology`, `demo_topology`, `generate_topology`, `AttackGraph`, `propagate`, `total_risk`, `top_attack_paths`, `to_cytoscape` |
| `risk_engine` | `RiskModel` (propagation / multiplicative / severity_only), `RiskReport`, `downstream_exposure` |
| `defense_engine` | `ActionGenerator`, `DefenseEvaluator`, `build_policy`, `prescreen` |
| `optimization` | `DefenseProblem`, `Solver`, `SolveResult` |
| `quantum_engine` | `build_defense_qubo`, `QUBO`, `qubo_to_ising`, `QAOASolver`, `NOISE_LEVELS` |
| `classical_baselines` | `ExhaustiveSolver`, `ScoreRankingSolver`, `GreedySolver`, `SimulatedAnnealingSolver`, `GeneticSolver`, `MILPSolver`, `RandomSamplingSolver` |
| `adaptive_engine` | `AttackEnvironment`, `EffectivenessLearner`, `AdaptiveWeights`, `AQDOController`, `run_episode`, `STAGED_DEMO` |
| `explainability` | `explain(problem, x)` |
| `api` | FastAPI `app`, `Storage` (SQLite), `SOCSession` |

## Database schema (SQLite, maps 1:1 to PostgreSQL)

```
security_events(id, ts, host, attack_type, probability, confidence, source)
optimization_runs(id, ts, solver, params JSON, risk_before, risk_after, objective, feasible, result JSON)
analyst_decisions(id, ts, run_id, action_id, decision, analyst)
```

Set `QADAPT_DB=data/qadapt.db` to persist across restarts (the default is in-memory), and
`QADAPT_MODEL=models/detector.joblib` to use a detector trained by `qadapt train`.
