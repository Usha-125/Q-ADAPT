"""Experiments 3 & 4 - QAOA (p = 1..3) vs classical baselines on defense-portfolio instances.

Each instance is a random enterprise (40 nodes) with ML evidence on 3 hosts, pre-screened
to n candidate actions, with budget and cardinality constraints. The true optimum is
obtained by exhaustive enumeration of the ground-truth objective. Reported: optimality
gap (relative), feasibility rate, runtime, and for QAOA the approximation ratio, P(opt),
amplification over uniform sampling, circuit depth and function evaluations.
"""

from __future__ import annotations

import argparse

import numpy as np

from experiments.common import fmt, make_instance, md_table, mean_ci, save, wilcoxon
from qadapt.classical_baselines import BASELINES
from qadapt.quantum_engine import QAOASolver


def solvers(shots: int):
    s = {name: (lambda name=name: BASELINES[name]()) for name in
         ("score_ranking", "greedy", "simulated_annealing", "genetic", "milp")}
    s["random_sampling"] = lambda: BASELINES["random_sampling"](n_samples=shots)
    for p in (1, 2, 3):
        s[f"qaoa_p{p}"] = lambda p=p: QAOASolver(p=p, shots=shots, restarts=2, maxiter=150)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 10, 12, 14])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    sizes = [6, 8, 10] if a.quick else a.sizes
    seeds = 3 if a.quick else a.seeds
    names = list(solvers(a.shots))
    records = []
    for n in sizes:
        for seed in range(seeds):
            prob = make_instance(n, seed)
            opt = BASELINES["exhaustive"]().solve(prob)
            for name, make in solvers(a.shots).items():
                r = make().solve(prob)
                gap = (r.objective - opt.objective) / abs(opt.objective) if r.feasible else None
                rec = {"n": n, "seed": seed, "solver": name, "objective": r.objective,
                       "optimum": opt.objective, "gap": gap, "feasible": r.feasible,
                       "optimal": bool(r.feasible and abs(r.objective - opt.objective) < 1e-9),
                       "runtime_s": r.runtime_s,
                       "risk_reduction": r.metrics.risk_reduction}
                if name.startswith("qaoa"):
                    for k in ("approximation_ratio", "p_optimal_state", "optimal_state_amplification",
                              "circuit_depth", "function_evals", "n_qubits"):
                        rec[k] = r.info.get(k)
                    rec["fidelity_spearman"] = r.info["qubo_meta"].get("fidelity", {}).get("spearman")
                records.append(rec)
            print(f"n={n} seed={seed} done (opt={opt.objective:.4f})")

    md = ["# Experiments 3 & 4 - QAOA vs classical optimisation\n",
          f"Instances: {seeds} random seeds per size; 40-node enterprises; budget = 35 % of total "
          f"candidate cost; max actions = n/3. QAOA: state-vector simulation, {a.shots} shots, "
          "2 COBYLA restarts, INTERP initialisation. Gap = relative gap to the exhaustive optimum "
          "of the ground-truth objective (mean ± 95 % CI).\n"]
    rows = []
    for n in sizes:
        for name in names:
            rs = [r for r in records if r["n"] == n and r["solver"] == name]
            rows.append([n, name, fmt(mean_ci([r["gap"] for r in rs])),
                         f"{np.mean([r['optimal'] for r in rs]):.0%}",
                         f"{np.mean([r['feasible'] for r in rs]):.0%}",
                         fmt(mean_ci([r["runtime_s"] for r in rs]), 3)])
    md.append("## Solution quality\n\n" + md_table(
        ["n actions", "solver", "optimality gap", "optimal found", "feasible", "runtime s"], rows))
    qrows = []
    for n in sizes:
        for p in (1, 2, 3):
            rs = [r for r in records if r["n"] == n and r["solver"] == f"qaoa_p{p}"]
            qrows.append([n, p, rs[0]["n_qubits"], fmt(mean_ci([r["approximation_ratio"] for r in rs]), 3),
                          fmt(mean_ci([r["p_optimal_state"] for r in rs]), 4),
                          f"{np.mean([r['optimal_state_amplification'] for r in rs]):.1f}x",
                          f"{np.mean([r['circuit_depth'] for r in rs]):.0f}",
                          f"{np.mean([r['function_evals'] for r in rs]):.0f}",
                          fmt(mean_ci([r["fidelity_spearman"] for r in rs]), 3)])
    md.append("\n## QAOA circuit and landscape metrics (Experiment 4)\n\n" + md_table(
        ["n", "p", "qubits", "approx. ratio", "P(optimal QUBO state)", "amplification vs uniform",
         "scheduled depth", "function evals", "QUBO surrogate Spearman"], qrows))
    tests = []
    for name in names:
        if name == "qaoa_p2":
            continue
        a_ = [r["objective"] for r in records if r["solver"] == "qaoa_p2"]
        b_ = [r["objective"] for r in records if r["solver"] == name]
        w = wilcoxon(a_, b_)
        tests.append([f"qaoa_p2 vs {name}", f"{np.mean(np.array(a_) - np.array(b_)):+.5f}",
                      "n/a" if w["p_value"] is None else f"{w['p_value']:.4g}"])
    md.append("\n## Paired comparison (all instances, Wilcoxon signed-rank on objective)\n\n"
              + md_table(["comparison", "mean difference (neg. = QAOA better)", "p-value"], tests))
    md.append("\nNote: results are from classical simulation of QAOA; no quantum speed-up is implied. "
              "At these sizes exhaustive search is fastest; the study measures solution quality.\n")
    save("experiment3_optimization", {"records": records}, "\n".join(md))


if __name__ == "__main__":
    main()
