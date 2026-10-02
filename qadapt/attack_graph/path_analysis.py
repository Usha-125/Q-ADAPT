"""Attack-path analysis: most likely paths, k-best paths, blocked-path metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import islice

import networkx as nx

from qadapt.attack_graph.graph_builder import ATTACKER


@dataclass
class AttackPath:
    nodes: list[str]
    probability: float

    def to_dict(self) -> dict:
        return {"nodes": self.nodes, "probability": round(self.probability, 5)}


def _weighted(g: nx.DiGraph, edge_mult: dict[tuple[str, str], float] | None = None) -> nx.DiGraph:
    h = nx.DiGraph()
    for u, v, d in g.edges(data=True):
        p = d["p"] * (edge_mult or {}).get((u, v), 1.0)
        if p > 1e-9:
            h.add_edge(u, v, p=p, w=-math.log(p))
    return h


def path_probability(g: nx.DiGraph, nodes: list[str]) -> float:
    prob = 1.0
    for u, v in zip(nodes, nodes[1:], strict=False):
        prob *= g[u][v]["p"]
    return prob


def top_attack_paths(g: nx.DiGraph, targets: list[str], k: int = 5,
                     source: str = ATTACKER, sources: list[str] | None = None,
                     edge_mult: dict[tuple[str, str], float] | None = None) -> list[AttackPath]:
    """k most probable simple paths (Yen's algorithm on -log p) to each target.

    ``sources`` adds extra starting points (e.g. hosts the ML engine already
    flags as compromised), modelled as a super-source.
    """
    h = _weighted(g, edge_mult)
    starts = [source] + [s for s in (sources or []) if s != source]
    if len(starts) > 1:
        for s in starts:
            if s in h:
                h.add_edge("__SRC__", s, p=1.0, w=0.0)
        source = "__SRC__"
    out: list[AttackPath] = []
    for t in targets:
        if t not in h or source not in h or not nx.has_path(h, source, t):
            continue
        for path in islice(nx.shortest_simple_paths(h, source, t, weight="w"), k):
            path = [n for n in path if n != "__SRC__"]
            out.append(AttackPath(path, path_probability(h, path)))
    return sorted(out, key=lambda p: -p.probability)


def critical_targets(g: nx.DiGraph, threshold: float = 0.85) -> list[str]:
    return [n for n, d in g.nodes(data=True) if n != ATTACKER and d.get("criticality", 0) >= threshold]


def count_viable_paths(g: nx.DiGraph, targets: list[str], min_prob: float = 0.05, k: int = 10,
                       sources: list[str] | None = None,
                       edge_mult: dict[tuple[str, str], float] | None = None) -> int:
    """Number of (k-best per target) attack paths with probability >= ``min_prob``."""
    return sum(p.probability >= min_prob
               for p in top_attack_paths(g, targets, k, sources=sources, edge_mult=edge_mult))
