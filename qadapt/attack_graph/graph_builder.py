"""Probabilistic attack graph G = (V, E).

Vertices are assets plus a virtual ``ATTACKER`` source. A directed edge
(u, v) carries the probability that an attacker who controls ``u`` compromises
``v`` in one step; it is derived from the reachability link kind and the CVSS
exploitability of ``v`` (cf. NIST IR 7788 probabilistic attack graphs).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import sparse

from qadapt.attack_graph.topology import Topology
from qadapt.core.models import Asset, ThreatAssessment

ATTACKER = "ATTACKER"
P_MIN, P_MAX = 0.02, 0.98
MISCONFIG_FLOOR = 0.05  # residual exploit probability of an asset without known CVEs
DEFAULT_ATTACKER_ACTIVITY = 0.3  # prior probability that an external attacker is active


def edge_probability(kind: str, src: Asset | None, dst: Asset) -> float:
    e = max(dst.exploitability, MISCONFIG_FLOOR)
    if kind in ("network", "entry"):
        p = e
    elif kind == "credential":
        priv = any(v.requires_privileges for v in (src.vulnerabilities if src else []))
        p = 0.45 + (0.25 if priv else 0.0) + 0.2 * e
    elif kind == "lateral":
        p = 0.1 + 0.6 * e
    elif kind == "phishing":
        p = 0.35
    else:
        p = e
    return float(np.clip(p, P_MIN, P_MAX))


class AttackGraph:
    """Mutable attack-graph state (topology, edge probabilities, ML threat evidence)."""

    def __init__(self, topology: Topology,
                 attacker_activity: float = DEFAULT_ATTACKER_ACTIVITY):
        self.topology = topology
        self.attacker_activity = attacker_activity
        self.assets = topology.assets
        self.g = nx.DiGraph()
        self.g.add_node(ATTACKER, criticality=0.0, type="attacker")
        for a in self.assets.values():
            self.g.add_node(a.id, criticality=a.criticality, type=a.type.value, zone=a.zone)
        for entry in topology.entry_points:
            p = edge_probability("entry", None, self.assets[entry])
            self.g.add_edge(ATTACKER, entry, p=p, base_p=p, kind="entry")
        for u, v, kind in topology.links:
            p = edge_probability(kind, self.assets[u], self.assets[v])
            self.g.add_edge(u, v, p=p, base_p=p, kind=kind)
        self.threat: dict[str, float] = {}
        self.threat_info: dict[str, ThreatAssessment] = {}
        self.compromised: set[str] = set()

    # ---- dynamic state -------------------------------------------------------
    def apply_threats(self, threats: list[ThreatAssessment]) -> None:
        """Inject ML evidence: local compromise probability = p * confidence.

        Detected activity also raises the attacker-presence prior, since a
        detection is evidence that an adversary is currently operating.
        """
        for t in threats:
            if t.host_id in self.assets:
                score = float(t.probability * t.confidence)
                self.threat[t.host_id] = max(self.threat.get(t.host_id, 0.0), score)
                self.threat_info[t.host_id] = t
                self.attacker_activity = max(self.attacker_activity, score)

    def mark_compromised(self, host: str) -> None:
        self.compromised.add(host)
        self.threat[host] = 1.0
        self.attacker_activity = 1.0

    def harden_edges(self, factors: dict[tuple[str, str], float]) -> None:
        """Permanently reduce edge probabilities (an approved defense took effect)."""
        for (u, v), f in factors.items():
            if self.g.has_edge(u, v):
                self.g[u][v]["p"] *= 1.0 - f

    def reduce_threat(self, factors: dict[str, float]) -> None:
        for n, f in factors.items():
            if n in self.threat:
                self.threat[n] *= 1.0 - f

    def clear_threat(self, host: str) -> None:
        self.threat.pop(host, None)
        self.threat_info.pop(host, None)
        self.compromised.discard(host)

    # ---- views ---------------------------------------------------------------
    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self.g.edges())

    def compile(self) -> CompiledGraph:
        return CompiledGraph.from_attack_graph(self)

    def stats(self) -> dict:
        n, m = self.g.number_of_nodes(), self.g.number_of_edges()
        return {"nodes": n, "edges": m, "density": nx.density(self.g),
                "entry_points": len(self.topology.entry_points),
                "attacker_activity": self.attacker_activity}


@dataclass
class CompiledGraph:
    """Array form of the attack graph for fast (batched) risk propagation."""

    nodes: list[str]
    index: dict[str, int]
    edges: list[tuple[str, str]]
    edge_index: dict[tuple[str, str], int]
    src: np.ndarray
    dst: np.ndarray
    p: np.ndarray
    criticality: np.ndarray
    threat: np.ndarray
    incidence: sparse.csr_matrix  # (E x N), 1 where edge e enters node n

    @classmethod
    def from_attack_graph(cls, ag: AttackGraph) -> CompiledGraph:
        nodes = [ATTACKER] + [n for n in ag.g.nodes if n != ATTACKER]
        index = {n: i for i, n in enumerate(nodes)}
        edges = list(ag.g.edges())
        src = np.array([index[u] for u, _ in edges], dtype=int)
        dst = np.array([index[v] for _, v in edges], dtype=int)
        p = np.array([ag.g[u][v]["p"] for u, v in edges], dtype=float)
        crit = np.array([ag.g.nodes[n]["criticality"] for n in nodes], dtype=float)
        threat = np.array([ag.threat.get(n, 0.0) for n in nodes], dtype=float)
        threat[0] = ag.attacker_activity
        inc = sparse.csr_matrix((np.ones(len(edges)), (np.arange(len(edges)), dst)),
                                shape=(len(edges), len(nodes)))
        return cls(nodes, index, edges, {e: i for i, e in enumerate(edges)},
                   src, dst, p, crit, threat, inc)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)
