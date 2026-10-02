"""In-memory SOC session: current network, threat picture and pending recommendation."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from qadapt.adaptive_engine.scenarios import STAGED_DEMO
from qadapt.attack_graph import AttackGraph, scenario_topology, total_risk
from qadapt.attack_graph.risk_propagation import propagate
from qadapt.core.models import DefenseAction, ThreatAssessment
from qadapt.optimization.problem import DefenseProblem

INITIAL_THREATS = {
    "demo": STAGED_DEMO[0]["threats"],
}


@dataclass
class Pending:
    run_id: int
    problem: DefenseProblem
    recommended: list[DefenseAction]


@dataclass
class SOCSession:
    scenario: str = "demo"
    seed: int = 0
    ag: AttackGraph = None  # type: ignore[assignment]
    applied: list[DefenseAction] = field(default_factory=list)
    pending: Pending | None = None
    stage: int = 0
    timeline: list[dict] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    detector: object | None = None

    def reset(self, scenario: str = "demo", seed: int = 0, with_initial_threats: bool = True) -> None:
        with self.lock:
            self.scenario, self.seed = scenario, seed
            self.ag = AttackGraph(scenario_topology(scenario, seed))
            self.applied, self.pending, self.stage, self.timeline = [], None, 0, []
            if with_initial_threats:
                threats = INITIAL_THREATS.get(scenario)
                if threats is None:  # generated topology: threat on the first web server
                    web = next(a for a in self.ag.assets if a.startswith("WEB"))
                    threats = [ThreatAssessment(web, "WEB_ATTACK", 0.92, 0.9, 80, "203.0.113.7")]
                self.ag.apply_threats(threats)
                self.stage = 1 if scenario == "demo" else 0
            self.snapshot("Session start")

    def risk(self) -> float:
        cg = self.ag.compile()
        return total_risk(cg, propagate(cg))

    def snapshot(self, label: str) -> None:
        self.timeline.append({"step": len(self.timeline), "label": label,
                              "risk": round(self.risk(), 5),
                              "applied": [a.id for a in self.applied],
                              "compromised": sorted(self.ag.compromised)})

    def threats(self) -> list[dict]:
        out = []
        for h, t in sorted(self.ag.threat_info.items(), key=lambda kv: -kv[1].probability):
            d = t.to_dict()
            d["asset_name"] = self.ag.assets[h].name
            d["compromised"] = h in self.ag.compromised
            out.append(d)
        return out

    def apply(self, actions: list[DefenseAction]) -> None:
        self.applied.extend(actions)
        self.ag.apply_defenses(self.applied)

    def defended_edges(self) -> dict[tuple[str, str], float]:
        return {(u, v): 1 - d["p"] / d["base_p"] for u, v, d in self.ag.g.edges(data=True)
                if d["base_p"] > 0 and d["p"] < d["base_p"] - 1e-9}


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x
