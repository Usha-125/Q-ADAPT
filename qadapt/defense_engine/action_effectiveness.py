"""Defense effect model: maps a binary portfolio x to residual network risk.

For a portfolio x in {0,1}^n the multiplier on attack edge e is

    m_e(x) = prod_i (1 - eff_i * effect_{i,e}) ** x_i = exp(x @ L_e)

with L_e[i, e] = log(1 - eff_i * effect_{i,e}). This makes evaluating a whole
batch of portfolios a single matrix product followed by batched propagation,
and is the *ground-truth* objective every solver is scored against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qadapt.attack_graph.graph_builder import AttackGraph, CompiledGraph
from qadapt.core.models import DefenseAction
from qadapt.risk_engine.risk_model import RiskModel

_FLOOR = 1e-6


@dataclass
class PortfolioMetrics:
    residual_risk: float
    risk_reduction: float
    cost: float
    time_total: float
    time_to_effect: float
    disruption: float
    n_actions: int

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class DefenseEvaluator:
    def __init__(self, ag: AttackGraph, actions: list[DefenseAction],
                 risk_model: RiskModel | None = None, cg: CompiledGraph | None = None):
        self.ag = ag
        self.actions = actions
        self.risk_model = risk_model or RiskModel()
        self.cg = cg or ag.compile()
        n, E, N = len(actions), self.cg.n_edges, self.cg.n_nodes
        self.L_edge = np.zeros((n, E))
        self.L_node = np.zeros((n, N))
        for i, a in enumerate(actions):
            for e, f in a.edge_effects.items():
                j = self.cg.edge_index.get(e)
                if j is not None:
                    self.L_edge[i, j] = np.log(max(1.0 - a.effectiveness * f, _FLOOR))
            for v, f in a.node_effects.items():
                j = self.cg.index.get(v)
                if j is not None:
                    self.L_node[i, j] = np.log(max(1.0 - a.effectiveness * f, _FLOOR))
        self.cost = np.array([a.cost for a in actions])
        self.time = np.array([a.time for a in actions])
        self.disruption = np.array([a.disruption for a in actions])
        self.base_risk = float(self.risk_model.total(self.cg, ag))
        self._cache: dict[bytes, float] = {}

    @property
    def n(self) -> int:
        return len(self.actions)

    def multipliers(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(X).astype(float)
        return np.exp(X @ self.L_edge), np.exp(X @ self.L_node)

    def residual_risk(self, X: np.ndarray) -> np.ndarray | float:
        """Residual total risk for one portfolio (n,) or a batch (B, n)."""
        X = np.asarray(X)
        single = X.ndim == 1
        X2 = np.atleast_2d(X).astype(np.int8)
        out = np.empty(len(X2))
        todo = []
        for b, row in enumerate(X2):
            key = row.tobytes()
            if key in self._cache:
                out[b] = self._cache[key]
            else:
                todo.append(b)
        for start in range(0, len(todo), 512):
            idx = todo[start:start + 512]
            em, nm = self.multipliers(X2[idx])
            r = np.atleast_1d(self.risk_model.total(self.cg, self.ag, em, nm))
            for b, val in zip(idx, r, strict=True):
                out[b] = val
                self._cache[X2[b].tobytes()] = float(val)
        return float(out[0]) if single else out

    def metrics(self, x: np.ndarray) -> PortfolioMetrics:
        x = np.asarray(x).astype(float)
        r = float(self.residual_risk(x))
        sel = x > 0.5
        return PortfolioMetrics(
            residual_risk=r,
            risk_reduction=(self.base_risk - r) / self.base_risk if self.base_risk > 0 else 0.0,
            cost=float(self.cost @ x),
            time_total=float(self.time @ x),
            time_to_effect=float(self.time[sel].max()) if sel.any() else 0.0,
            disruption=float(self.disruption @ x),
            n_actions=int(sel.sum()),
        )

    def edge_reduction(self, x: np.ndarray) -> dict[tuple[str, str], float]:
        em, _ = self.multipliers(np.asarray(x))
        return {e: float(1 - em[0, j]) for e, j in self.cg.edge_index.items() if em[0, j] < 0.999}

    def marginal_reductions(self) -> np.ndarray:
        """Risk reduction of each action applied alone."""
        return self.base_risk - self.residual_risk(np.eye(self.n, dtype=int))
