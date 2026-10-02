"""Shared helpers for the Q-ADAPT experiment suite."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy import stats

from qadapt.attack_graph import AttackGraph, generate_topology
from qadapt.core.config import OptimizationConfig
from qadapt.core.models import ThreatAssessment
from qadapt.defense_engine import ActionGenerator, DefenseEvaluator, prescreen
from qadapt.optimization import DefenseProblem

RESULTS = Path(__file__).resolve().parents[1] / "results"


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


def mean_ci(values, conf: float = 0.95) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "std": None, "ci95": None, "n": 0}
    m, s = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
    h = float(stats.t.ppf((1 + conf) / 2, len(v) - 1) * s / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return {"mean": m, "std": s, "ci95": h, "n": int(len(v))}


def wilcoxon(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return {"statistic": None, "p_value": 1.0, "note": "identical"}
    r = stats.wilcoxon(a, b, zero_method="zsplit")
    return {"statistic": float(r.statistic), "p_value": float(r.pvalue)}


def fmt(stat: dict, digits: int = 4) -> str:
    if stat["mean"] is None:
        return "n/a"
    return f"{stat['mean']:.{digits}f} ± {stat['ci95']:.{digits}f}"


def save(name: str, payload: dict, markdown: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    payload = {"experiment": name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "platform": platform.platform(), "python": platform.python_version(), **payload}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_default))
    (RESULTS / f"{name}.md").write_text(markdown)
    print(markdown)
    return path


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
