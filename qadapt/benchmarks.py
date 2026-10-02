"""Reproducible benchmark instances and a quick solver comparison (used by the CLI and
the experiment suite)."""

from __future__ import annotations

import numpy as np

from qadapt.attack_graph import AttackGraph, generate_topology
from qadapt.core.config import OptimizationConfig
from qadapt.core.models import ThreatAssessment
from qadapt.defense_engine import ActionGenerator, DefenseEvaluator, prescreen
from qadapt.optimization import DefenseProblem


def random_threats(ag: AttackGraph, rng: np.random.Generator, k: int = 2) -> list[ThreatAssessment]:
    """ML-style evidence on a random internet-facing host and k-1 random internal hosts."""
    entry = list(ag.topology.entry_points)
    internal = [a for a in ag.assets if a not in entry and not a.startswith("FW")]
    hosts = [rng.choice(entry)] + list(rng.choice(internal, size=max(0, k - 1), replace=False))
    cats = ["WEB_ATTACK", "EXPLOIT", "INFILTRATION", "BOTNET", "BRUTE_FORCE"]
    return [ThreatAssessment(str(h), str(rng.choice(cats)), float(rng.uniform(0.6, 0.97)),
                             float(rng.uniform(0.75, 0.97)), int(rng.integers(5, 200)),
                             "203.0.113.7") for h in hosts]


def make_instance(n_actions: int, seed: int, n_nodes: int = 40,
                  config: OptimizationConfig | None = None) -> DefenseProblem:
    """A random defense-portfolio instance with exactly ``n_actions`` candidates."""
    rng = np.random.default_rng(seed)
    ag = AttackGraph(generate_topology(n_nodes, seed=seed))
    ag.apply_threats(random_threats(ag, rng, k=3))
    acts = ActionGenerator(ag).generate()
    ev = DefenseEvaluator(ag, acts)
    keep = prescreen(ev, n_actions)
    ev = DefenseEvaluator(ag, [acts[i] for i in keep])
    if config is None:
        total = ev.cost.sum()
        config = OptimizationConfig(budget=round(0.35 * total, 3), max_actions=max(2, n_actions // 3))
    return DefenseProblem(ev, config)


def compare_solvers(sizes: list[int], seeds: int, solvers: list[str] | None = None,
                    shots: int = 256) -> list[dict]:
    """Mean optimality gap / runtime of each solver against the exhaustive optimum."""
    from qadapt.classical_baselines import ExhaustiveSolver
    from qadapt.pipeline import make_solver
    solvers = solvers or ["qaoa", "greedy", "simulated_annealing", "genetic", "milp",
                          "score_ranking", "random_sampling"]
    rows = []
    for n in sizes:
        acc: dict[str, list] = {s: [] for s in solvers}
        for seed in range(seeds):
            prob = make_instance(n, seed)
            opt = ExhaustiveSolver().solve(prob).objective
            for s in solvers:
                kw = {"p": 2, "shots": shots, "seed": seed} if s == "qaoa" else (
                    {"n_samples": shots, "seed": seed} if s == "random_sampling" else {})
                r = make_solver(s, **kw).solve(prob)
                gap = (r.objective - opt) / abs(opt) if r.feasible and opt else np.nan
                acc[s].append((gap, r.feasible, r.runtime_s))
        for s, vals in acc.items():
            g = np.array([v[0] for v in vals], dtype=float)
            rows.append({"n": n, "solver": s, "mean_gap": float(np.nanmean(g)) if np.isfinite(g).any() else None,
                         "optimal_rate": float(np.mean(np.nan_to_num(g, nan=1.0) < 1e-9)),
                         "feasible_rate": float(np.mean([v[1] for v in vals])),
                         "runtime_s": float(np.mean([v[2] for v in vals]))})
    return rows
