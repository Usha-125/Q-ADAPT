"""Classical baselines for defense-portfolio selection.

All heuristics optimise the *true* objective with an exterior penalty on
constraint violations (stronger baselines than optimising the QUBO surrogate),
except :class:`MILPSolver`, which solves the QUBO surrogate exactly with hard
linear constraints (McCormick linearisation of the quadratic terms).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from qadapt.optimization.problem import DefenseProblem, Solver


def _penalised(problem: DefenseProblem, X: np.ndarray, penalty: float = 10.0) -> np.ndarray:
    X = np.atleast_2d(X)
    return np.atleast_1d(problem.objective(X)) + penalty * problem.violation_amount(X)


class ExhaustiveSolver(Solver):
    """Exact optimum of the true objective by enumeration (n <= 22)."""

    name = "exhaustive"

    def __init__(self, max_n: int = 22, chunk: int = 4096):
        self.max_n, self.chunk = max_n, chunk

    def _solve(self, problem):
        n = problem.n
        if n > self.max_n:
            raise ValueError(f"exhaustive search limited to n <= {self.max_n} (got {n})")
        best_x, best_v, n_feasible = np.zeros(n, int), np.inf, 0
        for start in range(0, 1 << n, self.chunk):
            ks = np.arange(start, min(start + self.chunk, 1 << n))
            X = ((ks[:, None] >> np.arange(n)) & 1).astype(np.int8)
            feas = problem.feasible_batch(X)
            n_feasible += int(feas.sum())
            if not feas.any():
                continue
            Xf = X[feas]
            v = np.atleast_1d(problem.objective(Xf, cache=False))
            i = int(np.argmin(v))
            if v[i] < best_v:
                best_v, best_x = float(v[i]), Xf[i].astype(int)
        return best_x, {"evaluated": 1 << n, "feasible_portfolios": n_feasible}


class ScoreRankingSolver(Solver):
    """Naive SOC practice: rank actions by individual score, take while feasible.

    Ignores interactions between actions (redundancy / synergy).
    """

    name = "score_ranking"

    def _solve(self, problem):
        ev, w = problem.evaluator, problem.config.weights
        gain = ev.marginal_reductions() / max(ev.base_risk, 1e-12)
        score = w.alpha * gain - w.beta * ev.cost - w.gamma * ev.time - w.delta * ev.disruption
        x = np.zeros(problem.n, dtype=int)
        for i in np.argsort(-score):
            if score[i] <= 0:
                break
            x[i] = 1
            if not problem.feasible(x):
                x[i] = 0
        return x, {}


class GreedySolver(Solver):
    """Interaction-aware forward greedy: add the action with the best marginal true gain."""

    name = "greedy"

    def _solve(self, problem):
        n = problem.n
        x = np.zeros(n, dtype=int)
        cur = problem.objective(x)
        evals = 1
        while True:
            cands = np.flatnonzero(x == 0)
            if not len(cands):
                break
            X = np.repeat(x[None, :], len(cands), axis=0)
            X[np.arange(len(cands)), cands] = 1
            feas = problem.feasible_batch(X)
            if not feas.any():
                break
            objs = np.atleast_1d(problem.objective(X[feas]))
            evals += len(objs)
            j = int(np.argmin(objs))
            if objs[j] >= cur - 1e-12:
                break
            x, cur = X[feas][j], objs[j]
        return x, {"evaluations": evals}


class SimulatedAnnealingSolver(Solver):
    name = "simulated_annealing"

    def __init__(self, n_steps: int = 4000, t0: float = 0.2, t1: float = 1e-3,
                 restarts: int = 4, seed: int = 0):
        self.n_steps, self.t0, self.t1, self.restarts, self.seed = n_steps, t0, t1, restarts, seed

    def _solve(self, problem):
        rng = np.random.default_rng(self.seed)
        n = problem.n
        best_x, best_v = None, np.inf
        for _ in range(self.restarts):
            x = np.zeros(n, dtype=int)
            v = float(_penalised(problem, x)[0])
            for k in range(self.n_steps):
                t = self.t0 * (self.t1 / self.t0) ** (k / max(self.n_steps - 1, 1))
                y = x.copy()
                flips = rng.integers(n, size=1 if rng.random() < 0.8 else 2)
                y[flips] ^= 1
                vy = float(_penalised(problem, y)[0])
                if vy < v or rng.random() < np.exp(-(vy - v) / t):
                    x, v = y, vy
                    if v < best_v and problem.feasible(x):
                        best_x, best_v = x.copy(), v
        if best_x is None:
            best_x = np.zeros(n, dtype=int)
        return best_x, {"steps": self.n_steps * self.restarts}


class GeneticSolver(Solver):
    name = "genetic"

    def __init__(self, pop: int = 60, generations: int = 80, p_mut: float | None = None,
                 elite: int = 4, seed: int = 0):
        self.pop, self.generations, self.p_mut, self.elite, self.seed = pop, generations, p_mut, elite, seed

    def _solve(self, problem):
        rng = np.random.default_rng(self.seed)
        n = problem.n
        p_mut = self.p_mut or 1.0 / n
        P = (rng.random((self.pop, n)) < 0.2).astype(int)
        P[0] = 0
        fit = _penalised(problem, P)
        for _ in range(self.generations):
            order = np.argsort(fit)
            P, fit = P[order], fit[order]
            children = [P[i] for i in range(self.elite)]
            while len(children) < self.pop:
                a, b = (min(rng.integers(self.pop, size=3), key=lambda i: fit[i]) for _ in range(2))
                mask = rng.random(n) < 0.5
                child = np.where(mask, P[a], P[b])
                child ^= (rng.random(n) < p_mut).astype(int)
                children.append(child)
            P = np.array(children)
            fit = _penalised(problem, P)
        x, _, _ = problem.best_of(P)
        return x, {"evaluations": self.pop * (self.generations + 1)}


class RandomSamplingSolver(Solver):
    """Control: uniform random portfolios with the same sample budget as QAOA shots."""

    name = "random_sampling"

    def __init__(self, n_samples: int = 2048, seed: int = 0):
        self.n_samples, self.seed = n_samples, seed

    def _solve(self, problem):
        rng = np.random.default_rng(self.seed)
        X = rng.integers(0, 2, size=(self.n_samples, problem.n))
        x, _, feas = problem.best_of(X)
        return x, {"samples": self.n_samples, "found_feasible": feas}


class MILPSolver(Solver):
    """Exact optimum of the QUBO surrogate with hard constraints (HiGHS via SciPy).

    Quadratic terms J_ij x_i x_j are linearised with y_ij (McCormick):
    y <= x_i, y <= x_j, y >= x_i + x_j - 1.
    """

    name = "milp"

    def __init__(self, time_limit: float = 60.0, surrogate: str = "regression"):
        self.time_limit, self.surrogate = time_limit, surrogate

    def _solve(self, problem):
        from qadapt.quantum_engine.qubo_builder import _expansion, _regression
        n, ev, w = problem.n, problem.evaluator, problem.config.weights
        h, J = _expansion(problem) if self.surrogate == "expansion" else _regression(problem, None, 1e-3, 0)
        iu, ju = np.nonzero(np.abs(np.triu(J, 1)) > 1e-9)
        m = len(iu)
        c = np.concatenate([w.alpha * h + w.beta * ev.cost + w.gamma * ev.time
                            + w.delta * ev.disruption, w.alpha * J[iu, ju]])
        A, lb, ub = [], [], []

        def row(coeffs: dict[int, float], lo: float, hi: float):
            r = np.zeros(n + m)
            for k, v in coeffs.items():
                r[k] += v
            A.append(r)
            lb.append(lo)
            ub.append(hi)

        for k, (i, j) in enumerate(zip(iu, ju, strict=True)):
            row({n + k: 1, i: -1}, -np.inf, 0)
            row({n + k: 1, j: -1}, -np.inf, 0)
            row({n + k: 1, i: -1, j: -1}, -1, np.inf)
        for _, wts, lim in problem.constraints():
            row({i: wts[i] for i in range(n)}, -np.inf, lim)
        pol = problem.policy
        for i, j in pol.conflicts:
            row({i: 1, j: 1}, -np.inf, 1)
        for i, j in pol.prerequisites:
            row({i: 1, j: -1}, -np.inf, 0)
        lo = np.zeros(n + m)
        hi = np.ones(n + m)
        for i in pol.forbidden:
            hi[i] = 0
        for i in pol.mandatory:
            lo[i] = 1
        cons = [LinearConstraint(np.array(A), lb, ub)] if A else []
        res = milp(c, constraints=cons, integrality=np.ones(n + m), bounds=Bounds(lo, hi),
                   options={"time_limit": self.time_limit})
        if res.x is None:
            return np.zeros(n, dtype=int), {"status": res.message}
        return np.round(res.x[:n]).astype(int), {"status": res.message, "n_aux": m,
                                                 "surrogate_objective": float(res.fun + w.alpha)}


BASELINES = {
    "exhaustive": ExhaustiveSolver,
    "score_ranking": ScoreRankingSolver,
    "greedy": GreedySolver,
    "simulated_annealing": SimulatedAnnealingSolver,
    "genetic": GeneticSolver,
    "random_sampling": RandomSamplingSolver,
    "milp": MILPSolver,
}
