"""Run the full experiment suite: python -m experiments.run_all [--quick]."""

from __future__ import annotations

import sys
import time

from experiments import (
    experiment1_ml,
    experiment2_graph,
    experiment3_optimization,
    experiment5_noise,
    experiment6_adaptive,
    experiment7_constraints,
    experiment8_ablation,
)

SUITE = [experiment1_ml, experiment2_graph, experiment3_optimization, experiment5_noise,
         experiment6_adaptive, experiment7_constraints, experiment8_ablation]

if __name__ == "__main__":
    args = ["--quick"] if "--quick" in sys.argv else []
    for mod in SUITE:
        t0 = time.time()
        print(f"\n===== {mod.__name__} =====")
        mod.main(args)
        print(f"[{mod.__name__}] {time.time() - t0:.1f}s")
