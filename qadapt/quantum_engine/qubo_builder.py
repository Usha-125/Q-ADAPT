"""Construction of the cybersecurity defense QUBO.

Residual risk R(x) is a non-linear (noisy-OR propagated) function of the
portfolio. The QUBO needs a quadratic pseudo-Boolean surrogate::

    R(x) / R(0) ~= 1 + sum_i h_i x_i + sum_{i<j} J_ij x_i x_j

Two surrogates are provided:

``expansion``  second-order Moebius expansion around the empty portfolio
               (h_i = r_i - r_0, J_ij = r_ij - r_i - r_j + r_0): exact for
               singletons and pairs, captures redundancy (J > 0) and synergy
               (J < 0) between actions explicitly.
``regression`` ridge least-squares fit of the same quadratic form on portfolios
               sampled across the operating cardinality range; more faithful
               for larger portfolios where effects saturate.

Operational terms (cost, time, disruption) are linear. Constraints are added as
penalties: policy rules exactly; inequality limits either via binary slack
variables (exact, extra qubits) or by unbalanced penalisation (no extra qubits,
Montanez-Barrera et al., 2022).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

SURROGATES = ("expansion", "regression")


@dataclass
class QUBO:
    """f(x) = offset + linear @ x + x @ quad @ x   (quad strictly upper-triangular)."""

    linear: np.ndarray
    quad: np.ndarray
    offset: float = 0.0
    var_names: list[str] = field(default_factory=list)
    n_decision: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.linear)

    @property
    def n_slack(self) -> int:
        return self.n - self.n_decision

    def energy(self, X: np.ndarray) -> np.ndarray | float:
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        X2 = np.atleast_2d(X)
        e = self.offset + X2 @ self.linear + np.einsum("bi,ij,bj->b", X2, self.quad, X2)
        return float(e[0]) if single else e

    def matrix(self) -> np.ndarray:
        """Upper-triangular Q with the linear terms on the diagonal (x^T Q x form)."""
        return np.triu(self.quad, 1) + np.diag(self.linear)

    def add_linear(self, i: int, v: float) -> None:
        self.linear[i] += v

    def add_quad(self, i: int, j: int, v: float) -> None:
        if i == j:
            self.linear[i] += v  # x_i^2 = x_i
        else:
            a, b = (i, j) if i < j else (j, i)
            self.quad[a, b] += v

    def add_square(self, coeffs: dict[int, float], const: float, weight: float) -> None:
        """Add weight * (const + sum_i c_i x_i)^2."""
        items = list(coeffs.items())
        self.offset += weight * const * const
        for a, (i, ci) in enumerate(items):
            self.linear[i] += weight * (ci * ci + 2 * const * ci)
            for j, cj in items[a + 1:]:
                self.add_quad(i, j, 2 * weight * ci * cj)

    def brute_force(self, chunk: int = 1 << 16) -> tuple[np.ndarray, float]:
        """Exact minimum by enumeration (n <= ~24)."""
        if self.n > 26:
            raise ValueError("brute force limited to n <= 26")
        best_e, best_k = np.inf, 0
        for start in range(0, 1 << self.n, chunk):
            ks = np.arange(start, min(start + chunk, 1 << self.n))
            X = ((ks[:, None] >> np.arange(self.n)) & 1).astype(float)
            e = self.energy(X)
            i = int(np.argmin(e))
            if e[i] < best_e:
                best_e, best_k = float(e[i]), int(ks[i])
        return (best_k >> np.arange(self.n)) & 1, best_e

    def to_dict(self) -> dict:
        return {"n": self.n, "n_decision": self.n_decision, "offset": self.offset,
                "var_names": self.var_names, "linear": self.linear.tolist(),
                "quad": self.quad.tolist(), "meta": self.meta}


# ---------------------------------------------------------------------------
# surrogate construction
# ---------------------------------------------------------------------------
def _expansion(problem) -> tuple[np.ndarray, np.ndarray]:
    ev, n = problem.evaluator, problem.n
    r0 = ev.base_risk
    eye = np.eye(n, dtype=int)
    r1 = np.atleast_1d(ev.residual_risk(eye))
    iu, ju = np.triu_indices(n, 1)
    pairs = eye[iu] + eye[ju]
    r2 = np.atleast_1d(ev.residual_risk(pairs)) if len(pairs) else np.array([])
    h = (r1 - r0) / r0
    J = np.zeros((n, n))
    J[iu, ju] = (r2 - r1[iu] - r1[ju] + r0) / r0
    return h, J


def _sample_portfolios(n: int, m: int, kmax: int, rng: np.random.Generator) -> np.ndarray:
    ks = rng.integers(0, kmax + 1, size=m)
    X = np.zeros((m, n), dtype=int)
    for b, k in enumerate(ks):
        X[b, rng.choice(n, size=k, replace=False)] = 1
    return X


def _regression(problem, n_samples: int | None, ridge: float, seed: int):
    ev, n = problem.evaluator, problem.n
    r0 = ev.base_risk
    rng = np.random.default_rng(seed)
    kmax = problem.config.max_actions or max(2, int(np.ceil(n / 2)))
    kmax = min(n, max(2, kmax + 1))
    iu, ju = np.triu_indices(n, 1)
    n_feat = n + len(iu)
    m = n_samples or max(400, 3 * n_feat)
    eye = np.eye(n, dtype=int)
    X = np.vstack([eye, eye[iu] + eye[ju] if len(iu) else np.zeros((0, n), int),
                   _sample_portfolios(n, m, kmax, rng)])
    y = np.asarray(ev.residual_risk(X)) / r0 - 1.0
    F = np.hstack([X, X[:, iu] * X[:, ju]]).astype(float)
    A = F.T @ F + ridge * np.eye(n_feat)
    coef = np.linalg.solve(A, F.T @ y)
    h = coef[:n]
    J = np.zeros((n, n))
    J[iu, ju] = coef[n:]
    return h, J


def surrogate_fidelity(problem, h: np.ndarray, J: np.ndarray, n_samples: int = 300,
                       seed: int = 123) -> dict:
    """Agreement between quadratic surrogate and true relative risk on held-out portfolios."""
    ev, n = problem.evaluator, problem.n
    kmax = min(n, (problem.config.max_actions or max(2, n // 2)) + 1)
    X = _sample_portfolios(n, n_samples, kmax, np.random.default_rng(seed))
    true = np.asarray(ev.residual_risk(X)) / ev.base_risk
    pred = 1.0 + X @ h + np.einsum("bi,ij,bj->b", X, J, X)
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum()) or 1.0
    rho = spearmanr(true, pred).statistic if np.ptp(true) > 0 else 1.0
    return {"r2": 1 - ss_res / ss_tot, "spearman": float(rho),
            "mae": float(np.abs(true - pred).mean())}


# ---------------------------------------------------------------------------
# full QUBO
# ---------------------------------------------------------------------------
def build_defense_qubo(problem, surrogate: str = "regression", encoding: str | None = None,
                       penalty: float | None = None, n_samples: int | None = None,
                       ridge: float = 1e-3, seed: int = 0, unbalanced: tuple[float, float] = (1.0, 1.0),
                       compute_fidelity: bool = True) -> QUBO:
    if surrogate not in SURROGATES:
        raise ValueError(f"surrogate must be one of {SURROGATES}")
    cfg, ev, n = problem.config, problem.evaluator, problem.n
    w = cfg.weights
    encoding = encoding or cfg.constraint_encoding
    if encoding not in ("slack", "unbalanced"):
        raise ValueError("encoding must be 'slack' or 'unbalanced'")

    h, J = _expansion(problem) if surrogate == "expansion" else _regression(problem, n_samples, ridge, seed)

    # objective part --------------------------------------------------------------
    lin = w.alpha * h + w.beta * ev.cost + w.gamma * ev.time + w.delta * ev.disruption
    quad = w.alpha * J
    obj_offset = w.alpha

    # automatic penalty: no single bit flip into infeasibility can pay off
    if penalty is None:
        penalty = cfg.penalty
    if penalty is None:
        swing = np.abs(lin) + np.abs(quad).sum(0) + np.abs(quad).sum(1)
        penalty = float(2.0 * swing.max() + 1e-3)

    constraints = problem.constraints()
    slack_names: list[str] = []
    slack_specs = []
    if encoding == "slack":
        for name, wts, lim in constraints:
            if wts.sum() <= lim:  # never binding
                continue
            k = cfg.slack_resolution
            step = lim / (2 ** k - 1)
            slack_specs.append((name, wts, lim, len(slack_names), k, step))
            slack_names += [f"slack_{name}_{b}" for b in range(k)]

    N = n + len(slack_names)
    q = QUBO(np.zeros(N), np.zeros((N, N)), obj_offset,
             [a.id for a in problem.actions] + slack_names, n)
    q.linear[:n] += lin
    q.quad[:n, :n] += np.triu(quad, 1)

    # policy penalties -------------------------------------------------------------
    pol = problem.policy
    for i, j in pol.conflicts:
        q.add_quad(i, j, penalty)
    for i, j in pol.prerequisites:  # x_i (1 - x_j)
        q.add_linear(i, penalty)
        q.add_quad(i, j, -penalty)
    for i in pol.forbidden:
        q.add_linear(i, penalty)
    for i in pol.mandatory:
        q.offset += penalty
        q.add_linear(i, -penalty)

    # inequality constraints ---------------------------------------------------------
    for name, wts, lim in constraints:
        if wts.sum() <= lim:
            continue
        scale = 1.0 / max(lim, 1e-9)  # normalise constraint to O(1)
        if encoding == "slack":
            spec = next(s for s in slack_specs if s[0] == name)
            _, _, _, start, k, step = spec
            coeffs = {i: wts[i] * scale for i in range(n) if wts[i] != 0}
            for b in range(k):
                coeffs[n + start + b] = (2 ** b) * step * scale
            q.add_square(coeffs, -lim * scale, penalty)
        else:
            l1, l2 = unbalanced
            # h'(x) = 1 - (w @ x)/lim ; penalty * (-l1 h' + l2 h'^2)
            coeffs = {i: -wts[i] * scale for i in range(n) if wts[i] != 0}
            q.offset += penalty * (-l1)
            for i, c in coeffs.items():
                q.add_linear(i, penalty * (-l1) * c)
            q.add_square(coeffs, 1.0, penalty * l2)

    q.meta = {"surrogate": surrogate, "encoding": encoding, "penalty": penalty,
              "n_constraints": len(constraints), "n_slack": len(slack_names),
              "weights": w.as_dict()}
    if compute_fidelity and n >= 2:
        q.meta["fidelity"] = surrogate_fidelity(problem, h, J)
    q.meta["h"] = h.tolist()
    return q
