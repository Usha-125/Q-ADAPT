"""Dynamic risk engine.

Three interchangeable models (compared in the ablation study):

``propagation`` (default)
    Risk_i = C_i * P_i, with P_i the noisy-OR compromise probability from the
    attack graph, i.e. ML evidence *and* multi-step propagation.
``multiplicative``
    The proposal's formulation Risk_i = T_i * V_i * C_i * P_i, where T_i is the
    attacker-reachability-adjusted threat, V_i exploitability and P_i the
    propagation probability.
``severity_only``
    Alert severity times criticality, no graph (the "ML only" baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qadapt.attack_graph.graph_builder import ATTACKER, AttackGraph, CompiledGraph
from qadapt.attack_graph.risk_propagation import propagate
from qadapt.core.models import Severity

RISK_MODELS = ("propagation", "multiplicative", "severity_only")


@dataclass
class AssetRisk:
    asset_id: str
    name: str
    threat: float
    vulnerability: float
    criticality: float
    compromise_p: float
    risk: float

    @property
    def level(self) -> Severity:
        return Severity.from_score(self.risk / max(self.criticality, 1e-9) if self.criticality else 0)

    def to_dict(self) -> dict:
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}
        d["level"] = self.level.value
        return d


@dataclass
class RiskReport:
    model: str
    total_risk: float
    assets: list[AssetRisk] = field(default_factory=list)

    def top(self, k: int = 5) -> list[AssetRisk]:
        return sorted(self.assets, key=lambda a: -a.risk)[:k]

    def critical_at_risk(self, crit_threshold: float = 0.85, p_threshold: float = 0.5) -> int:
        return sum(a.criticality >= crit_threshold and a.compromise_p >= p_threshold
                   for a in self.assets)

    def to_dict(self) -> dict:
        return {"model": self.model, "total_risk": round(self.total_risk, 5),
                "critical_at_risk": self.critical_at_risk(),
                "assets": [a.to_dict() for a in sorted(self.assets, key=lambda a: -a.risk)]}


class RiskModel:
    def __init__(self, model: str = "propagation"):
        if model not in RISK_MODELS:
            raise ValueError(f"unknown risk model {model!r}; choose from {RISK_MODELS}")
        self.model = model

    def node_risk(self, cg: CompiledGraph, ag: AttackGraph,
                  edge_mult: np.ndarray | None = None,
                  node_mult: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (compromise probability, per-node risk) arrays (batched if inputs are)."""
        crit = cg.criticality.copy()
        crit[0] = 0.0
        if self.model == "severity_only":
            t = cg.threat * (np.ones(cg.n_nodes) if node_mult is None else node_mult)
            t = np.where(np.arange(cg.n_nodes) == 0, 0.0, t)
            return t, t * crit
        P = propagate(cg, edge_mult, node_mult)
        if self.model == "propagation":
            return P, P * crit
        vuln = np.array([ag.assets[n].exploitability if n in ag.assets else 1.0
                         for n in cg.nodes])
        T = np.maximum(cg.threat, cg.threat[0])  # local evidence or attacker presence
        return P, T * np.maximum(vuln, 0.05) * crit * P

    def total(self, cg: CompiledGraph, ag: AttackGraph, edge_mult=None, node_mult=None):
        _, r = self.node_risk(cg, ag, edge_mult, node_mult)
        crit = cg.criticality.copy()
        crit[0] = 0.0
        out = r.sum(axis=-1) / crit.sum()
        return float(out) if np.ndim(out) == 0 else out

    def report(self, ag: AttackGraph, cg: CompiledGraph | None = None,
               edge_mult=None, node_mult=None) -> RiskReport:
        cg = cg or ag.compile()
        P, r = self.node_risk(cg, ag, edge_mult, node_mult)
        rows = []
        for n, i in cg.index.items():
            if n == ATTACKER:
                continue
            a = ag.assets[n]
            rows.append(AssetRisk(n, a.name, float(cg.threat[i]), a.exploitability,
                                  float(cg.criticality[i]), float(P[i]), float(r[i])))
        return RiskReport(self.model, self.total(cg, ag, edge_mult, node_mult), rows)
