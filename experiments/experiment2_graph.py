"""Experiment 2 - attack-graph generation and risk-propagation scalability."""

from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.common import md_table, random_threats, save
from qadapt.attack_graph import (
    AttackGraph,
    critical_targets,
    generate_topology,
    propagate,
    top_attack_paths,
    total_risk,
)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 25, 50, 100, 250, 500])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    sizes = [10, 25, 50, 100] if a.quick else a.sizes
    rows, raw = [], []
    for n in sizes:
        rec = {k: [] for k in ("gen_s", "nodes", "edges", "density", "paths", "path_s", "prop_s",
                               "iters_risk", "batch_ms")}
        for s in range(a.seeds):
            rng = np.random.default_rng(s)
            t0 = time.perf_counter()
            ag = AttackGraph(generate_topology(n, seed=s))
            ag.apply_threats(random_threats(ag, rng, k=2))
            cg = ag.compile()
            rec["gen_s"].append(time.perf_counter() - t0)
            st = ag.stats()
            rec["nodes"].append(st["nodes"])
            rec["edges"].append(st["edges"])
            rec["density"].append(st["density"])
            t0 = time.perf_counter()
            P = propagate(cg)
            rec["prop_s"].append(time.perf_counter() - t0)
            rec["iters_risk"].append(total_risk(cg, P))
            t0 = time.perf_counter()
            em = rng.uniform(0.3, 1.0, size=(64, cg.n_edges))
            propagate(cg, em)
            rec["batch_ms"].append(1000 * (time.perf_counter() - t0) / 64)
            t0 = time.perf_counter()
            sources = sorted(h for h, t in ag.threat.items() if t >= 0.5)
            paths = top_attack_paths(ag.g, critical_targets(ag.g), k=5, sources=sources)
            rec["path_s"].append(time.perf_counter() - t0)
            rec["paths"].append(sum(p.probability >= 0.05 for p in paths))
        m = {k: float(np.mean(v)) for k, v in rec.items()}
        raw.append({"size": n, **m})
        rows.append([n, f"{m['nodes']:.0f}", f"{m['edges']:.0f}", f"{m['density']:.4f}",
                     f"{m['gen_s'] * 1000:.1f}", f"{m['prop_s'] * 1000:.2f}", f"{m['batch_ms']:.3f}",
                     f"{m['paths']:.1f}", f"{m['path_s'] * 1000:.1f}", f"{m['iters_risk']:.3f}"])
    md = ("# Experiment 2 - attack graph scalability\n\n"
          f"Mean over {a.seeds} random topologies per size; 2 ML threats injected per graph.\n\n"
          + md_table(["target size", "nodes", "edges", "density", "build ms", "propagation ms",
                      "batched propagation ms/portfolio", "viable paths (k=5/target)",
                      "path analysis ms", "total risk"], rows) + "\n")
    save("experiment2_graph", {"results": raw}, md)


if __name__ == "__main__":
    main()
