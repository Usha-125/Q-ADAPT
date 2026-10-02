"""Serialisation of the attack graph for the dashboard (Cytoscape.js elements)."""

from __future__ import annotations

import numpy as np

from qadapt.attack_graph.graph_builder import ATTACKER, AttackGraph, CompiledGraph


def to_cytoscape(ag: AttackGraph, cg: CompiledGraph, P: np.ndarray,
                 highlight_path: list[str] | None = None,
                 defended_edges: dict[tuple[str, str], float] | None = None) -> dict:
    on_path = set(zip(highlight_path or [], (highlight_path or [])[1:], strict=False))
    nodes = []
    for n in cg.nodes:
        i = cg.index[n]
        a = ag.assets.get(n)
        nodes.append({"data": {
            "id": n,
            "label": a.name if a else "Attacker",
            "type": a.type.value if a else "attacker",
            "zone": a.zone if a else "external",
            "criticality": round(float(cg.criticality[i]), 3),
            "compromise_p": round(float(P[i]), 4),
            "risk": round(float(P[i] * cg.criticality[i]), 4),
            "threat": round(float(cg.threat[i]), 4),
            "compromised": n in ag.compromised,
            "cves": [v.cve_id for v in a.vulnerabilities] if a else [],
        }})
    edges = []
    for (u, v), i in cg.edge_index.items():
        d = ag.g[u][v]
        edges.append({"data": {
            "id": f"{u}->{v}", "source": u, "target": v, "kind": d["kind"],
            "p": round(float(cg.p[i]), 4),
            "on_path": (u, v) in on_path,
            "defended": round(float((defended_edges or {}).get((u, v), 0.0)), 3),
        }})
    return {"nodes": nodes, "edges": edges, "attacker": ATTACKER}
