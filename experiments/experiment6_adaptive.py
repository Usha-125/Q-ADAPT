"""Experiment 6 - dynamic adaptation (H5).

Part A: the scripted three-stage scenario (web -> app -> database), showing how the
risk, QUBO and recommended defense change at each stage.
Part B: stochastic attack campaigns against four policies (none / static / reoptimize /
aqdo) with mis-specified action effectiveness (the attacker rotates IPs, patches are
slow to take effect), measured by realised criticality-weighted loss.
"""

from __future__ import annotations

import argparse

import numpy as np

from experiments.common import fmt, md_table, mean_ci, save, wilcoxon
from qadapt.adaptive_engine import POLICIES, STAGED_DEMO, run_episode
from qadapt.attack_graph import AttackGraph, demo_topology, generate_topology
from qadapt.core.config import OptimizationConfig
from qadapt.core.models import ActionType
from qadapt.pipeline import QAdaptPipeline

TRUE_EFF = {ActionType.BLOCK_IP: 0.25, ActionType.PATCH_VULNERABILITY: 0.5,
            ActionType.INCREASE_MONITORING: 0.1}


def staged(solver_kw) -> tuple[list, list]:
    ag = AttackGraph(demo_topology())
    applied, rows, raw = [], [], []
    for st in STAGED_DEMO:
        for h in st.get("compromised", []):
            ag.mark_compromised(h)
        ag.apply_threats(st["threats"])
        ag.apply_defenses(applied)
        pipe = QAdaptPipeline(ag, OptimizationConfig(budget=0.4, max_actions=3), max_qubits=12)
        rep = pipe.decide("qaoa", **solver_kw)
        applied += rep.selected_actions
        rows.append([f"T{st['t']}", st["label"], f"{rep.risk_before:.3f}", f"{rep.risk_after:.3f}",
                     ", ".join(a.description for a in rep.selected_actions)])
        raw.append(rep.to_dict())
    return rows, raw


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--solver", default="qaoa")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    seeds = 4 if a.quick else a.seeds
    qkw = {"p": 1, "restarts": 1, "maxiter": 80, "shots": 256}
    rows_a, raw_a = staged(qkw)

    cfg = OptimizationConfig(budget=0.4, max_actions=3)
    episodes = []
    envs = {"demo (14 nodes)": demo_topology, "enterprise-40": lambda: generate_topology(40, seed=3)}
    for env_name, topo in envs.items():
        for policy in POLICIES:
            for s in range(seeds):
                r = run_episode(topo(), policy, a.steps, cfg, solver=a.solver,
                                solver_kw=qkw if a.solver == "qaoa" else None,
                                true_effectiveness=TRUE_EFF, seed=s, max_qubits=10)
                r.pop("history")
                episodes.append({"env": env_name, "seed": s, **r})
            print(env_name, policy, "done")
    rows_b, tests = [], []
    for env_name in envs:
        for policy in POLICIES:
            rs = [e for e in episodes if e["env"] == env_name and e["policy"] == policy]
            rows_b.append([env_name, policy, fmt(mean_ci([e["cumulative_loss"] for e in rs]), 3),
                           fmt(mean_ci([e["final_loss"] for e in rs]), 3),
                           f"{np.mean([e['critical_compromised'] for e in rs]):.2f}",
                           f"{np.mean([e['n_actions'] for e in rs]):.1f}",
                           f"{np.mean([e['total_disruption'] for e in rs]):.2f}",
                           f"{np.mean([e['function_evals'] for e in rs]):.0f}"])

        def get(p, env=env_name):
            return [e["cumulative_loss"] for e in episodes if e["env"] == env and e["policy"] == p]
        for x, y in (("aqdo", "static"), ("aqdo", "reoptimize"), ("reoptimize", "static")):
            w = wilcoxon(get(x), get(y))
            tests.append([env_name, f"{x} vs {y}", f"{np.mean(np.array(get(x)) - np.array(get(y))):+.4f}",
                          f"{w['p_value']:.4g}"])
    md = ["# Experiment 6 - dynamic adaptation\n",
          "## A. Scripted multi-stage attack (demo network, QAOA p = 1)\n",
          md_table(["stage", "event", "risk before", "risk after", "recommended defense"], rows_a),
          "\n## B. Stochastic campaigns: static vs adaptive policies\n",
          f"{seeds} seeds x {a.steps} decision steps; solver = {a.solver}; true effectiveness differs "
          f"from nominal for: {', '.join(f'{k.value}={v}' for k, v in TRUE_EFF.items())}. "
          "Loss = criticality-weighted fraction of compromised assets (lower is better).\n",
          md_table(["environment", "policy", "cumulative loss", "final loss", "critical assets lost",
                    "actions", "disruption", "QAOA evals"], rows_b),
          "\n### Paired tests (Wilcoxon signed-rank on cumulative loss)\n",
          md_table(["environment", "comparison", "mean difference", "p-value"], tests)]
    save("experiment6_adaptive", {"staged": raw_a, "episodes": episodes}, "\n".join(md) + "\n")


if __name__ == "__main__":
    main()
