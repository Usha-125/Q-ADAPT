"""Experiment 8 - ablation study (Models A-F).

A  ML only                     severity-only risk, alert-driven candidates, rank-by-score
B  ML + attack graph           propagation risk, alert-driven candidates, rank-by-score
C  + risk engine               propagation risk, risk/path-driven candidates, rank-by-score
D  + classical optimisation    as C, interaction-aware simulated annealing
E  + QAOA                      as C, QAOA (p = 1)
F  full Q-ADAPT                as E + AQDO adaptation (re-optimisation, learning, re-weighting)

A-E decide once (static policy); F re-optimises every step. All are evaluated in the same
ground-truth attack environments (same seeds), by realised criticality-weighted loss.
"""

from __future__ import annotations

import argparse

import numpy as np

from experiments.common import fmt, md_table, mean_ci, save, wilcoxon
from experiments.experiment6_adaptive import TRUE_EFF
from qadapt.adaptive_engine import run_episode
from qadapt.attack_graph import demo_topology, generate_topology
from qadapt.core.config import OptimizationConfig

QKW = {"p": 1, "restarts": 1, "maxiter": 80, "shots": 256}
MODELS = {
    "A ML only": dict(policy="static", risk_model="severity_only", candidate_focus="threatened",
                      solver="score_ranking"),
    "B ML + graph": dict(policy="static", risk_model="propagation", candidate_focus="threatened",
                         solver="score_ranking"),
    "C + risk engine": dict(policy="static", risk_model="propagation", candidate_focus="risk",
                            solver="score_ranking"),
    "D + classical opt.": dict(policy="static", solver="simulated_annealing"),
    "E + QAOA": dict(policy="static", solver="qaoa", solver_kw=QKW),
    "F full Q-ADAPT": dict(policy="aqdo", solver="qaoa", solver_kw=QKW),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    seeds = 4 if a.quick else a.seeds
    cfg = OptimizationConfig(budget=0.5, max_actions=4)
    envs = {"demo (14 nodes)": demo_topology, "enterprise-40": lambda: generate_topology(40, seed=3)}
    recs = []
    for env_name, topo in envs.items():
        for m, kw in MODELS.items():
            for s in range(seeds):
                r = run_episode(topo(), steps=a.steps, config=cfg, true_effectiveness=TRUE_EFF,
                                seed=s, max_qubits=10, **kw)
                r.pop("history")
                recs.append({"env": env_name, "model": m, "seed": s, **r})
            print(env_name, m, "done")
    rows, tests = [], []
    for env_name in envs:
        for m in MODELS:
            rs = [r for r in recs if r["env"] == env_name and r["model"] == m]
            rows.append([env_name, m, fmt(mean_ci([r["cumulative_loss"] for r in rs]), 3),
                         fmt(mean_ci([r["final_loss"] for r in rs]), 3),
                         f"{np.mean([r['critical_compromised'] for r in rs]):.2f}",
                         f"{np.mean([r['n_actions'] for r in rs]):.1f}",
                         f"{np.mean([r['total_disruption'] for r in rs]):.2f}"])
        names = list(MODELS)
        for prev, cur in zip(names, names[1:], strict=False):
            x = [r["cumulative_loss"] for r in recs if r["env"] == env_name and r["model"] == cur]
            y = [r["cumulative_loss"] for r in recs if r["env"] == env_name and r["model"] == prev]
            tests.append([env_name, f"{cur} vs {prev}", f"{np.mean(np.array(x) - np.array(y)):+.4f}",
                          f"{wilcoxon(x, y)['p_value']:.4g}"])
    md = ["# Experiment 8 - ablation study\n", __doc__.split("\n\n", 1)[1],
          f"\n{seeds} seeds x {a.steps} steps per environment; budget 0.5, max 4 actions per decision.\n",
          md_table(["environment", "model", "cumulative loss", "final loss", "critical assets lost",
                    "actions", "disruption"], rows),
          "\n## Incremental contribution (Wilcoxon signed-rank on cumulative loss)\n",
          md_table(["environment", "step", "mean difference (neg. = improvement)", "p-value"], tests)]
    save("experiment8_ablation", {"records": recs}, "\n".join(md) + "\n")


if __name__ == "__main__":
    main()
