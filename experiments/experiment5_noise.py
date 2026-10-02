"""Experiment 5 - stability of the recommended defense under quantum noise.

State-vector engine with global-depolarising + readout noise (fast, all instances), and
an optional gate-level Qiskit Aer check on one instance (--aer).
"""

from __future__ import annotations

import argparse

import numpy as np

from experiments.common import fmt, make_instance, md_table, mean_ci, save
from qadapt.classical_baselines import ExhaustiveSolver
from qadapt.quantum_engine import NOISE_LEVELS, QAOASolver


def jaccard(a, b) -> float:
    a, b = set(np.flatnonzero(a)), set(np.flatnonzero(b))
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--aer", action="store_true", help="also run a gate-level Qiskit Aer check")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    seeds = 3 if a.quick else a.seeds
    records = []
    for seed in range(seeds):
        prob = make_instance(a.n, seed)
        opt = ExhaustiveSolver().solve(prob)
        ideal_x = {}
        for p in (1, 2, 3):
            for level in NOISE_LEVELS:
                r = QAOASolver(p=p, noise=level, shots=a.shots, restarts=2, maxiter=150).solve(prob)
                if level == "ideal":
                    ideal_x[p] = r.x
                records.append({
                    "seed": seed, "p": p, "noise": level,
                    "approximation_ratio": r.info["approximation_ratio"],
                    "p_optimal_state": r.info["p_optimal_state"],
                    "gap": (r.objective - opt.objective) / abs(opt.objective) if r.feasible else None,
                    "feasible": r.feasible,
                    "same_as_ideal": bool(np.array_equal(r.x, ideal_x[p])),
                    "jaccard_vs_ideal": jaccard(r.x, ideal_x[p]),
                    "jaccard_vs_optimum": jaccard(r.x, opt.x),
                })
        print(f"seed {seed} done")
    rows = []
    for p in (1, 2, 3):
        for level in NOISE_LEVELS:
            rs = [r for r in records if r["p"] == p and r["noise"] == level]
            rows.append([p, level, fmt(mean_ci([r["approximation_ratio"] for r in rs]), 3),
                         fmt(mean_ci([r["p_optimal_state"] for r in rs]), 4),
                         fmt(mean_ci([r["gap"] for r in rs])),
                         f"{np.mean([r['same_as_ideal'] for r in rs]):.0%}",
                         fmt(mean_ci([r["jaccard_vs_optimum"] for r in rs]), 3)])
    md = ["# Experiment 5 - quantum noise sensitivity\n",
          f"{seeds} instances with n = {a.n} actions. Noise levels (p1 / p2 / readout): "
          + ", ".join(f"{k}: {v.p1:g}/{v.p2:g}/{v.readout:g}" for k, v in NOISE_LEVELS.items())
          + ". Global-depolarising approximation in the state-vector engine.\n",
          md_table(["p", "noise", "approx. ratio", "P(optimal state)", "optimality gap",
                    "same plan as ideal", "Jaccard vs optimum"], rows)]
    aer = None
    if a.aer:
        prob = make_instance(min(a.n, 8), 0)
        aer = []
        for level in ("ideal", "medium"):
            r = QAOASolver(p=1, backend="qiskit", noise=level, shots=512, restarts=1, maxiter=40).solve(prob)
            aer.append([level, f"{r.objective:.4f}", r.feasible, r.info["circuit_depth"], f"{r.runtime_s:.1f}"])
        md.append("\n## Gate-level check (Qiskit Aer, p = 1, n = 8)\n\n"
                  + md_table(["noise", "objective", "feasible", "transpiled depth", "runtime s"], aer))
    md.append("\nNote: final plans are decoded from samples and scored on the true objective, so a "
              "feasible good plan can survive noise even when the approximation ratio drops.\n")
    save("experiment5_noise", {"records": records, "aer": aer}, "\n".join(md))


if __name__ == "__main__":
    main()
