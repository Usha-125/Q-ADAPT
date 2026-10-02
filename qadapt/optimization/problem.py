"""The defense-portfolio optimisation problem.

Ground-truth objective (to be minimised) for x in {0,1}^n::

    J(x) = alpha * R(x)/R(0) + beta * C(x) + gamma * T(x) + delta * D(x)

subject to budget / time / disruption / cardinality limits and the policy set.
R(x) is the residual network risk obtained by propagating the attack graph with
the portfolio's defenses applied. Every solver - quantum or classical - is
scored against this same function.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from qadapt.core.config import OptimizationConfig
from qadapt.core.models import DefenseAction
from qadapt.defense_engine.action_effectiveness import DefenseEvaluator, PortfolioMetrics
from qadapt.defense_engine.policy_constraints import PolicySet, build_policy


class DefenseProblem:
    def __init__(self, evaluator: DefenseEvaluator, config: OptimizationConfig | None = None,
                 policy: PolicySet | None = None):
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        self.policy = policy or build_policy(evaluator.actions, self.config.protected_assets)
        self._qubo = None

    # ---- basic properties -----------------------------------------------------
    @property
    def n(self) -> int:
        return self.evaluator.n

    @property
    def actions(self) -> list[DefenseAction]:
        return self.evaluator.actions

    def constraints(self) -> list[tuple[str, np.ndarray, float]]:
        """Inequality constraints as (name, weights, limit): weights @ x <= limit."""
        ev, c = self.evaluator, self.config
        out = []
        if c.budget is not None:
            out.append(("budget", ev.cost, c.budget))
        if c.max_time is not None:
            out.append(("time", ev.time, c.max_time))
        if c.max_disruption is not None:
            out.append(("disruption", ev.disruption, c.max_disruption))
        if c.max_actions is not None:
            out.append(("cardinality", np.ones(self.n), float(c.max_actions)))
        return out

    # ---- ground truth ----------------------------------------------------------
    def objective(self, X: np.ndarray, cache: bool = True) -> np.ndarray | float:
        """True objective J(x) for one portfolio or a batch."""
        X = np.asarray(X)
        w = self.config.weights
        ev = self.evaluator
        r = ev.residual_risk(X, cache=cache)
        rel = np.asarray(r) / ev.base_risk if ev.base_risk > 0 else np.asarray(r) * 0
        Xf = np.atleast_2d(X).astype(float)
        out = (w.alpha * np.atleast_1d(rel) + w.beta * Xf @ ev.cost + w.gamma * Xf @ ev.time
               + w.delta * Xf @ ev.disruption)
        return float(out[0]) if X.ndim == 1 else out

    def violations(self, x: np.ndarray) -> list[str]:
        x = np.asarray(x).round().astype(int)
        v = [f"{name}: {w @ x:.3f} > {lim:.3f}" for name, w, lim in self.constraints()
             if w @ x > lim + 1e-9]
        return v + self.policy.violations(x)

    def feasible(self, x: np.ndarray) -> bool:
        return not self.violations(x)

    def feasible_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised feasibility for a batch of portfolios."""
        X = np.atleast_2d(np.asarray(X)).round().astype(int)
        ok = np.ones(len(X), dtype=bool)
        for _, w, lim in self.constraints():
            ok &= X @ w <= lim + 1e-9
        pol = self.policy
        for i, j in pol.conflicts:
            ok &= ~((X[:, i] == 1) & (X[:, j] == 1))
        for i, j in pol.prerequisites:
            ok &= ~((X[:, i] == 1) & (X[:, j] == 0))
        for i in pol.forbidden:
            ok &= X[:, i] == 0
        for i in pol.mandatory:
            ok &= X[:, i] == 1
        return ok

    def violation_amount(self, X: np.ndarray) -> np.ndarray:
        """Continuous violation measure (0 when feasible) for penalty-based heuristics."""
        X = np.atleast_2d(np.asarray(X)).round().astype(int)
        v = np.zeros(len(X))
        for _, w, lim in self.constraints():
            v += np.maximum(0.0, X @ w - lim) / max(lim, 1e-9)
        pol = self.policy
        for i, j in pol.conflicts:
            v += (X[:, i] & X[:, j])
        for i, j in pol.prerequisites:
            v += X[:, i] * (1 - X[:, j])
        for i in pol.forbidden:
            v += X[:, i]
        for i in pol.mandatory:
            v += 1 - X[:, i]
        return v

    def metrics(self, x: np.ndarray) -> PortfolioMetrics:
        return self.evaluator.metrics(x)

    # ---- QUBO ------------------------------------------------------------------
    def qubo(self, **kw):
        """Lazily build (and cache) the cybersecurity QUBO for this problem."""
        from qadapt.quantum_engine.qubo_builder import build_defense_qubo
        if self._qubo is None or kw:
            self._qubo = build_defense_qubo(self, **kw)
        return self._qubo

    def best_of(self, candidates: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Pick the best candidate: feasible first, then lowest true objective."""
        cands = np.unique(np.atleast_2d(candidates).round().astype(int), axis=0)
        objs = np.atleast_1d(self.objective(cands))
        feas = self.feasible_batch(cands)
        key = np.where(feas, objs, objs + 1e6)
        i = int(np.argmin(key))
        return cands[i], float(objs[i]), bool(feas[i])


@dataclass
class SolveResult:
    solver: str
    x: np.ndarray
    objective: float
    feasible: bool
    runtime_s: float
    metrics: PortfolioMetrics | None = None
    selected: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "solver": self.solver,
            "x": [int(v) for v in self.x],
            "objective": round(self.objective, 6),
            "feasible": self.feasible,
            "runtime_s": round(self.runtime_s, 4),
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "selected": self.selected,
            "info": _jsonable(self.info),
        }


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


class Solver(ABC):
    name = "solver"

    def solve(self, problem: DefenseProblem) -> SolveResult:
        t0 = time.perf_counter()
        if problem.n == 0:  # nothing to decide: the empty portfolio is the only option
            x, info = np.zeros(0, dtype=int), {"note": "no candidate actions"}
        else:
            x, info = self._solve(problem)
        runtime = time.perf_counter() - t0
        x = np.asarray(x).round().astype(int)
        return SolveResult(
            solver=self.name,
            x=x,
            objective=float(problem.objective(x)),
            feasible=problem.feasible(x),
            runtime_s=runtime,
            metrics=problem.metrics(x),
            selected=[problem.actions[i].id for i in np.flatnonzero(x)],
            info=info,
        )

    @abstractmethod
    def _solve(self, problem: DefenseProblem) -> tuple[np.ndarray, dict]:
        ...
