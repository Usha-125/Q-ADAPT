"""Probabilistic risk propagation over the attack graph.

Compromise probability of node v (noisy-OR over local ML evidence and every
incoming attack step)::

    P(v) = 1 - (1 - t_v * m_v) * prod_{(u,v) in E} (1 - P(u) * p_uv * m_uv)

with P(ATTACKER) = attacker activity prior, ML evidence t_v, and defense multipliers m in [0, 1].
The fixed point is reached by monotone iteration from P = 0, which also handles
cycles (lateral movement). All operations are batched so that many candidate
defense portfolios can be scored in one call.
"""

from __future__ import annotations

import numpy as np

from qadapt.attack_graph.graph_builder import CompiledGraph

_EPS = 1e-12


def propagate(cg: CompiledGraph, edge_mult: np.ndarray | None = None,
              node_mult: np.ndarray | None = None, tol: float = 1e-9,
              max_iter: int | None = None) -> np.ndarray:
    """Return compromise probabilities.

    ``edge_mult``: shape (E,) or (B, E); ``node_mult``: shape (N,) or (B, N).
    Output shape is (N,) or (B, N) accordingly.
    """
    single = (edge_mult is None or np.ndim(edge_mult) == 1) and (
        node_mult is None or np.ndim(node_mult) == 1)
    em = np.atleast_2d(np.ones(cg.n_edges) if edge_mult is None else edge_mult)
    nm = np.atleast_2d(np.ones(cg.n_nodes) if node_mult is None else node_mult)
    batch = max(em.shape[0], nm.shape[0])
    em = np.broadcast_to(em, (batch, cg.n_edges))
    nm = np.broadcast_to(nm, (batch, cg.n_nodes))

    p_eff = cg.p[None, :] * em  # (B, E)
    log_local = np.log1p(-np.clip(cg.threat[None, :] * nm, 0.0, 1.0 - _EPS))  # (B, N)
    activity = cg.threat[0]
    P = np.zeros((batch, cg.n_nodes))
    P[:, 0] = activity
    max_iter = max_iter or (cg.n_nodes + 5)
    for _ in range(max_iter):
        step = np.clip(P[:, cg.src] * p_eff, 0.0, 1.0 - _EPS)
        log_in = np.asarray((np.log1p(-step)) @ cg.incidence)  # (B, N)
        P_new = 1.0 - np.exp(log_local + log_in)
        P_new[:, 0] = activity
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    return P[0] if single else P


def asset_risk(cg: CompiledGraph, P: np.ndarray) -> np.ndarray:
    """Expected loss per node: criticality x compromise probability."""
    return P * cg.criticality


def total_risk(cg: CompiledGraph, P: np.ndarray) -> np.ndarray | float:
    """Criticality-weighted network risk normalised to [0, 1] (attacker node excluded)."""
    crit = cg.criticality.copy()
    crit[0] = 0.0
    r = (P * crit).sum(axis=-1) / crit.sum()
    return float(r) if np.ndim(r) == 0 else r
