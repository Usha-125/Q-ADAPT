"""Experiment 7 - resource-constrained defense (scenarios A-E) and constraint encodings."""

from __future__ import annotations

import argparse

import numpy as np

from experiments.common import fmt, make_instance, md_table, mean_ci, save
from qadapt.classical_baselines import BASELINES
from qadapt.core.config import OptimizationConfig
from qadapt.optimization import DefenseProblem
from qadapt.quantum_engine import QAOASolver


def scenarios(ev):
    c, t, d = ev.cost.sum(), ev.time.sum(), ev.disruption.sum()
    return {
        "A unlimited": OptimizationConfig(),
        "B budget": OptimizationConfig(budget=round(0.25 * c, 3)),
        "C time": OptimizationConfig(max_time=round(0.15 * t, 3)),
        "D disruption": OptimizationConfig(max_disruption=round(0.15 * d, 3)),
        "E combined": OptimizationConfig(budget=round(0.3 * c, 3), max_time=round(0.2 * t, 3),
                                         max_disruption=round(0.2 * d, 3), max_actions=3),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    seeds = 3 if a.quick else a.seeds
    records = []
    for seed in range(seeds):
        base = make_instance(a.n, seed)
        for sc_name, cfg in scenarios(base.evaluator).items():
            for enc in ("unbalanced", "slack"):
                cfg.constraint_encoding = enc
                prob = DefenseProblem(base.evaluator, cfg)
                opt = BASELINES["exhaustive"]().solve(prob)
                q = prob.qubo()
                xq, _ = q.brute_force() if q.n <= 22 else (None, None)
                runs = {"qaoa_p2": QAOASolver(p=2, shots=256, restarts=2, maxiter=150) if q.n <= 20 else None,
                        "simulated_annealing": BASELINES["simulated_annealing"](),
                        "milp": BASELINES["milp"]()}
                for name, solver in runs.items():
                    if solver is None or (name != "qaoa_p2" and enc == "slack"):
                        continue  # classical solvers do not use the encoding
                    r = solver.solve(prob)
                    records.append({
                        "seed": seed, "scenario": sc_name, "encoding": enc, "solver": name,
                        "n_qubits": q.n, "feasible": r.feasible,
                        "gap": (r.objective - opt.objective) / abs(opt.objective) if r.feasible else None,
                        "qubo_min_feasible": None if xq is None else bool(prob.feasible(xq[: q.n_decision])),
                        "risk_reduction": r.metrics.risk_reduction, "cost": r.metrics.cost,
                        "time": r.metrics.time_total, "disruption": r.metrics.disruption,
                        "opt_risk_reduction": opt.metrics.risk_reduction})
        print(f"seed {seed} done")
    rows = []
    for sc in scenarios(make_instance(a.n, 0).evaluator):
        for enc, name in (("unbalanced", "qaoa_p2"), ("slack", "qaoa_p2"),
                          ("unbalanced", "simulated_annealing"), ("unbalanced", "milp")):
            rs = [r for r in records if r["scenario"] == sc and r["encoding"] == enc and r["solver"] == name]
            if not rs:
                continue
            label = f"{name} ({enc})" if name.startswith("qaoa") else name
            rows.append([sc, label, rs[0]["n_qubits"] if name.startswith("qaoa") else "-",
                         f"{np.mean([r['feasible'] for r in rs]):.0%}",
                         f"{np.mean([bool(r['qubo_min_feasible']) for r in rs]):.0%}" if name.startswith("qaoa") else "-",
                         fmt(mean_ci([r["gap"] for r in rs])),
                         fmt(mean_ci([r["risk_reduction"] for r in rs]), 3),
                         fmt(mean_ci([r["opt_risk_reduction"] for r in rs]), 3)])
    md = ("# Experiment 7 - resource constraints and constraint encodings\n\n"
          f"{seeds} instances, n = {a.n} candidate actions. Limits are fractions of the total over all "
          "candidates (B: 25 % cost; C: 15 % time; D: 15 % disruption; E: 30/20/20 % + max 3 actions). "
          "'QUBO min feasible' = the exact QUBO minimiser satisfies the true constraints.\n\n"
          + md_table(["scenario", "solver", "qubits", "feasible", "QUBO min feasible", "optimality gap",
                      "risk reduction", "optimal risk reduction"], rows) + "\n")
    save("experiment7_constraints", {"records": records}, md)


if __name__ == "__main__":
    main()
