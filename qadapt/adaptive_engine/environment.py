"""Ground-truth attack environment for evaluating adaptive defense.

The defender never sees this state directly. Each tick the attacker attempts
every outgoing edge of every compromised node; defenses applied in the
environment use the *true* effectiveness of each action type, which may differ
from the nominal values the defender starts with (e.g. an attacker rotating
IPs makes ``block_ip`` far less effective). Detections of newly compromised
hosts reach the defender through the ML threat engine (or a calibrated
detection model) with misses and false positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qadapt.attack_graph.graph_builder import ATTACKER, AttackGraph
from qadapt.attack_graph.topology import Topology
from qadapt.core.models import ActionType, AssetType, DefenseAction, ThreatAssessment

ATTACK_BY_TYPE = {
    AssetType.WEB_SERVER: "WEB_ATTACK", AssetType.APP_SERVER: "EXPLOIT",
    AssetType.DATABASE: "INFILTRATION", AssetType.WORKSTATION: "BOTNET",
    AssetType.MAIL_SERVER: "EXPLOIT", AssetType.VPN_GATEWAY: "BRUTE_FORCE",
    AssetType.FILE_SERVER: "INFILTRATION", AssetType.DOMAIN_CONTROLLER: "INFILTRATION",
    AssetType.ADMIN: "INFILTRATION", AssetType.FIREWALL: "EXPLOIT",
}
CONTAINING = {ActionType.ISOLATE_HOST, ActionType.QUARANTINE_ENDPOINT}


@dataclass
class EdgeOutcome:
    """An attack attempt on a defended edge (feedback for effectiveness learning)."""

    edge: tuple[str, str]
    action_types: list[ActionType]
    blocked: bool


@dataclass
class TickResult:
    t: int
    newly_compromised: list[str]
    observations: list[ThreatAssessment]
    outcomes: list[EdgeOutcome] = field(default_factory=list)


class AttackEnvironment:
    def __init__(self, topology: Topology, true_effectiveness: dict[ActionType, float] | None = None,
                 initial: list[str] | None = None, speed: float = 0.6, detect_rate: float = 0.85,
                 false_positive_rate: float = 0.02, detector=None, seed: int = 0):
        self.ag = AttackGraph(topology, attacker_activity=1.0)
        self.true_eff = dict(true_effectiveness or {})
        self.compromised: set[str] = set(initial or [])
        self.contained: set[str] = set()
        self.applied: list[DefenseAction] = []
        self.speed = speed
        self.detect_rate = detect_rate
        self.fp_rate = false_positive_rate
        self.detector = detector
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self._edge_actions: dict[tuple[str, str], list[ActionType]] = {}

    # ---- defender interaction ------------------------------------------------------
    def eff(self, a: DefenseAction) -> float:
        return self.true_eff.get(a.type, a.effectiveness)

    def apply(self, actions: list[DefenseAction]) -> None:
        self.applied.extend(actions)
        self.ag.apply_defenses(self.applied, {a.type: self.eff(a) for a in self.applied})
        for a in actions:
            for e in a.edge_effects:
                self._edge_actions.setdefault(e, []).append(a.type)
            if a.type in CONTAINING and a.target in self.compromised:
                if self.rng.random() < self.eff(a):
                    self.contained.add(a.target)

    # ---- dynamics --------------------------------------------------------------------
    def step(self) -> TickResult:
        self.t += 1
        new: list[str] = []
        outcomes: list[EdgeOutcome] = []
        active = [ATTACKER] + sorted(self.compromised - self.contained)
        for u in active:
            for v in self.ag.g.successors(u):
                if v in self.compromised:
                    continue
                d = self.ag.g[u][v]
                r = self.rng.random()
                would = r < d["base_p"] * self.speed
                success = r < d["p"] * self.speed
                if (u, v) in self._edge_actions and would:
                    outcomes.append(EdgeOutcome((u, v), self._edge_actions[(u, v)], not success))
                if success and v not in new:
                    new.append(v)
        self.compromised.update(new)
        return TickResult(self.t, new, self._observe(new), outcomes)

    def _observe(self, new: list[str]) -> list[ThreatAssessment]:
        if self.detector is not None:
            return self._observe_with_detector(new)
        obs = []
        for v in new:
            if self.rng.random() < self.detect_rate:
                obs.append(ThreatAssessment(
                    v, ATTACK_BY_TYPE.get(self.ag.assets[v].type, "EXPLOIT"),
                    float(self.rng.uniform(0.75, 0.98)), float(self.rng.uniform(0.8, 0.97)),
                    source="203.0.113.66"))
        for v in self.ag.assets:
            if v not in self.compromised and self.rng.random() < self.fp_rate:
                obs.append(ThreatAssessment(v, "ANOMALY", float(self.rng.uniform(0.3, 0.6)),
                                            float(self.rng.uniform(0.5, 0.7))))
        return obs

    def _observe_with_detector(self, new: list[str]) -> list[ThreatAssessment]:
        from qadapt.ml_engine.synthetic import generate_flows
        assets = self.ag.assets
        attacked = {assets[v].ip: ATTACK_BY_TYPE.get(assets[v].type, "EXPLOIT").replace(
            "WEB_ATTACK", "WEB_ATTACK") for v in new if self.rng.random() < self.detect_rate}
        flows = generate_flows(
            n=400, attack_fraction=0.15 if attacked else 0.0,
            hosts=[a.ip for a in assets.values()], attacked_hosts=attacked or None,
            seed=int(self.rng.integers(1 << 31)))
        ip_map = {a.ip: a.id for a in assets.values()}
        return self.detector.assess_hosts(flows, ip_map)

    # ---- ground-truth metrics ----------------------------------------------------------
    def realised_loss(self) -> float:
        crit = {n: a.criticality for n, a in self.ag.assets.items()}
        return sum(crit[n] for n in self.compromised) / sum(crit.values())

    def critical_compromised(self, threshold: float = 0.85) -> int:
        return sum(self.ag.assets[n].criticality >= threshold for n in self.compromised)
