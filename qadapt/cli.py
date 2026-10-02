"""Command-line interface: ``qadapt {demo,serve,train,benchmark,episode}``."""

from __future__ import annotations

import argparse
import json
import sys


def _demo(args) -> None:
    from qadapt.adaptive_engine.scenarios import STAGED_DEMO
    from qadapt.attack_graph import AttackGraph, scenario_topology
    from qadapt.core.config import OptimizationConfig
    from qadapt.pipeline import QAdaptPipeline

    ag = AttackGraph(scenario_topology(args.scenario, args.seed))
    pipe = QAdaptPipeline(ag, OptimizationConfig(budget=args.budget, max_actions=args.max_actions),
                          max_qubits=args.max_qubits)
    if args.scenario == "demo":
        pipe.ingest(STAGED_DEMO[0]["threats"] + STAGED_DEMO[1]["threats"])
    kw = {"p": args.p, "noise": args.noise} if args.solver == "qaoa" else {}
    rep = pipe.decide(args.solver, **kw)
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2, default=float))
        return
    r = rep.result
    print("=" * 64)
    print(" Q-ADAPT defense recommendation")
    print("=" * 64)
    print(f" solver          : {r.solver}   ({r.runtime_s:.2f}s, feasible={r.feasible})")
    if "n_qubits" in r.info:
        print(f" qubits / depth  : {r.info['n_qubits']} / {r.info['circuit_depth']}"
              f"   approx. ratio {r.info.get('approximation_ratio', float('nan')):.3f}")
    print(f" risk            : {rep.risk_before:.1%} -> {rep.risk_after:.1%}"
          f"   (-{r.metrics.risk_reduction:.1%})")
    print(f" attack paths    : {rep.paths_before} -> {rep.paths_after}")
    print(f" cost/time/disr. : {r.metrics.cost:.2f} / {r.metrics.time_total:.2f} / "
          f"{r.metrics.disruption:.2f}")
    print("-" * 64)
    for e in rep.explanations:
        mark = "[+]" if e.selected else "[ ]"
        print(f" {mark} {e.description}")
        for reason in e.reasons:
            print(f"       - {reason}")


def _serve(args) -> None:
    import uvicorn
    uvicorn.run("qadapt.api.app:app", host=args.host, port=args.port, reload=args.reload)


def _train(args) -> None:
    from sklearn.model_selection import train_test_split

    from qadapt.ml_engine import ThreatDetector, generate_flows, load_cic_csvs
    from qadapt.ml_engine.models import train_and_evaluate
    df = load_cic_csvs(args.csv, args.sample) if args.csv else generate_flows(args.synthetic, seed=0)
    train, test = train_test_split(df, test_size=0.25, random_state=0, stratify=df["category"])
    model, rep = train_and_evaluate(args.model, train, test)
    print(json.dumps(rep.to_dict(), indent=2))
    ThreatDetector(model, args.model).save(args.out)
    print(f"saved detector to {args.out}")


def _benchmark(args) -> None:
    from experiments.experiment3_optimization import main as bench
    bench(["--sizes", *map(str, args.sizes), "--seeds", str(args.seeds)])


def _episode(args) -> None:
    from qadapt.adaptive_engine import run_episode
    from qadapt.attack_graph import scenario_topology
    from qadapt.core.config import OptimizationConfig
    res = run_episode(scenario_topology(args.scenario), args.policy, args.steps,
                      OptimizationConfig(budget=args.budget, max_actions=3), solver=args.solver,
                      seed=args.seed)
    res.pop("history") if not args.verbose else None
    print(json.dumps(res, indent=2, default=float))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="qadapt", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the end-to-end pipeline once")
    d.add_argument("--scenario", default="demo", choices=["demo", "small", "medium", "large"])
    d.add_argument("--solver", default="qaoa")
    d.add_argument("--p", type=int, default=2)
    d.add_argument("--noise", default="ideal")
    d.add_argument("--budget", type=float, default=0.5)
    d.add_argument("--max-actions", type=int, default=None)
    d.add_argument("--max-qubits", type=int, default=12)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=_demo)

    s = sub.add_parser("serve", help="start the API + dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=_serve)

    t = sub.add_parser("train", help="train and save the ML threat detector")
    t.add_argument("--csv", nargs="*", help="CIC-IDS2017 / CSE-CIC-IDS2018 CSV files")
    t.add_argument("--sample", type=int, default=50000, help="rows sampled per CSV")
    t.add_argument("--synthetic", type=int, default=8000)
    t.add_argument("--model", default="random_forest")
    t.add_argument("--out", default="models/detector.joblib")
    t.set_defaults(fn=_train)

    b = sub.add_parser("benchmark", help="QAOA vs classical baselines")
    b.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 10, 12])
    b.add_argument("--seeds", type=int, default=3)
    b.set_defaults(fn=_benchmark)

    e = sub.add_parser("episode", help="simulate an adaptive defense episode")
    e.add_argument("--scenario", default="demo")
    e.add_argument("--policy", default="aqdo", choices=["none", "static", "reoptimize", "aqdo"])
    e.add_argument("--solver", default="qaoa")
    e.add_argument("--steps", type=int, default=6)
    e.add_argument("--budget", type=float, default=0.4)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--verbose", action="store_true")
    e.set_defaults(fn=_episode)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
