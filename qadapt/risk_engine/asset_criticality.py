"""Asset criticality: business value blended with attack-graph structural importance."""

from __future__ import annotations

import networkx as nx

from qadapt.attack_graph.graph_builder import ATTACKER, AttackGraph


def structural_importance(ag: AttackGraph) -> dict[str, float]:
    """Betweenness centrality on -log(p) weights: how often a node is a pivot."""
    import math
    h = nx.DiGraph()
    for u, v, d in ag.g.edges(data=True):
        h.add_edge(u, v, w=-math.log(max(d["p"], 1e-9)))
    bc = nx.betweenness_centrality(h, weight="w", normalized=True)
    m = max(bc.values(), default=0.0) or 1.0
    return {n: bc.get(n, 0.0) / m for n in ag.assets}


def downstream_exposure(ag: AttackGraph, node: str) -> float:
    """Total criticality reachable from ``node`` (excluding itself), normalised."""
    reach = nx.descendants(ag.g, node) - {ATTACKER}
    total = sum(a.criticality for a in ag.assets.values()) or 1.0
    return sum(ag.assets[n].criticality for n in reach) / total


def blended_criticality(ag: AttackGraph, weight_business: float = 0.8) -> dict[str, float]:
    imp = structural_importance(ag)
    return {n: weight_business * a.criticality + (1 - weight_business) * imp[n]
            for n, a in ag.assets.items()}


def apply_criticality(ag: AttackGraph, crit: dict[str, float]) -> None:
    for n, c in crit.items():
        ag.g.nodes[n]["criticality"] = float(c)
