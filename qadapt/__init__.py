"""Q-ADAPT: hybrid quantum-classical adaptive cyber-defense optimization.

Pipeline: Detect -> Understand -> Predict -> Optimize -> Defend -> Learn.

    ml_engine           ML threat assessment (per-host threat probability)
    attack_graph        enterprise topology, attack graph, risk propagation
    risk_engine         asset risk model, criticality, threat scoring
    defense_engine      candidate defense actions, costs, policy constraints
    quantum_engine      QUBO construction, Ising mapping, QAOA
    classical_baselines exhaustive / greedy / SA / GA / MILP baselines
    adaptive_engine     AQDO feedback loop (state update + re-optimization)
    explainability      human-readable justification of recommendations
    api                 FastAPI backend for the SOC dashboard
"""

__version__ = "0.1.0"
