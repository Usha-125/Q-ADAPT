"""QAOA for the defense QUBO.

    |psi(gamma, beta)> = prod_{l=1..p} e^{-i beta_l sum_i X_i} e^{-i gamma_l H_C} |+>^n

Backends
--------
``statevector``  NumPy state-vector simulator operating directly on the QUBO
                 diagonal (fast, n <= ~22). Noise is modelled with a global
                 depolarising channel (fidelity from gate counts) plus readout
                 bit flips.
``qiskit``       Qiskit ``QAOAAnsatz`` executed on Qiskit Aer (shots, optional
                 gate-level noise model) - see :mod:`qiskit_backend`.

Classical outer loop: COBYLA with multiple restarts, INTERP initialisation for
p > 1 (Zhou et al., PRX 2020), optional CVaR-alpha objective (Barkoutsos et
al., Quantum 2020), and warm starts from previously optimised parameters (used
by the adaptive engine when the threat state changes only slightly).

Hybrid decoding: the final distribution is sampled, candidates are scored by
the ground-truth objective, and the best *feasible* portfolio is returned;
an optional 1-flip local search polishes it (reported separately).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from qadapt.optimization.problem import DefenseProblem, Solver
from qadapt.quantum_engine.ising import diagonal_energies, qubo_to_ising
from qadapt.quantum_engine.noise_models import circuit_fidelity, get_noise
from qadapt.quantum_engine.qubo_builder import QUBO


# ---------------------------------------------------------------------------
# state-vector engine
# ---------------------------------------------------------------------------
def _apply_mixer(psi: np.ndarray, n: int, beta: float) -> np.ndarray:
    c, s = np.cos(beta), -1j * np.sin(beta)
    for i in range(n):
        v = psi.reshape(-1, 2, 1 << i)
        a, b = v[:, 0, :].copy(), v[:, 1, :]
        v[:, 0, :] = c * a + s * b
        v[:, 1, :] = s * a + c * b
    return psi


def qaoa_state(diag: np.ndarray, n: int, gammas, betas) -> np.ndarray:
    psi = np.full(1 << n, (1 << n) ** -0.5, dtype=complex)
    for g, b in zip(gammas, betas, strict=True):
        psi *= np.exp(-1j * g * diag)
        psi = _apply_mixer(psi, n, b)
    return psi


def gate_counts(q: QUBO, p: int) -> dict:
    """Analytic gate counts of the QAOA circuit (CX-RZ-CX decomposition of each ZZ)."""
    ising = qubo_to_ising(q)
    n_z, n_zz = ising.n_terms()
    n_1q = q.n + p * (n_z + q.n)  # H layer + (RZ fields + RX mixer) per layer
    n_2q = p * 2 * n_zz  # each ZZ term -> CX RZ CX
    depth_ub = 1 + p * (3 * n_zz + 2)  # serial upper bound
    return {"n_1q": n_1q, "n_2q": n_2q, "n_zz_terms": n_zz, "depth_upper_bound": depth_ub}


def transpiled_depth(q: QUBO, p: int) -> dict | None:
    """Depth / CX count after transpiling QAOAAnsatz to {cx, rz, sx, x} (needs Qiskit)."""
    try:
        from qiskit import transpile
        from qiskit.circuit.library import QAOAAnsatz
    except ImportError:  # pragma: no cover
        return None
    op = qubo_to_ising(q).to_sparse_pauli_op()
    circ = transpile(QAOAAnsatz(op, reps=p), basis_gates=["cx", "rz", "sx", "x"],
                     optimization_level=1, seed_transpiler=0)
    ops = circ.count_ops()
    return {"depth": int(circ.depth()), "cx": int(ops.get("cx", 0)),
            "total_gates": int(sum(ops.values()))}


class StatevectorQAOA:
    """Expectation/sampling engine for one QUBO."""

    def __init__(self, q: QUBO, p: int, noise=None, cvar_alpha: float = 1.0, seed: int = 0):
        if q.n > 22:
            raise ValueError(f"{q.n} qubits exceeds the state-vector limit (22); "
                             "pre-screen actions or use the 'unbalanced' encoding")
        self.q, self.p, self.n = q, p, q.n
        self.energies = diagonal_energies(q)
        self.e_min, self.e_max = float(self.energies.min()), float(self.energies.max())
        # rescale so that typical gamma in [0, pi] is meaningful
        self.scale = max(self.e_max - self.e_min, 1e-9)
        self.diag = (self.energies - self.e_min) / self.scale
        self.noise = get_noise(noise)
        gc = gate_counts(q, p)
        self.fidelity = circuit_fidelity(self.noise, gc["n_1q"], gc["n_2q"])
        self.cvar_alpha = cvar_alpha
        self.rng = np.random.default_rng(seed)
        self.n_evals = 0

    def probabilities(self, theta: np.ndarray) -> np.ndarray:
        psi = qaoa_state(self.diag, self.n, theta[: self.p], theta[self.p:])
        probs = np.abs(psi) ** 2
        if self.fidelity < 1.0:
            probs = self.fidelity * probs + (1 - self.fidelity) / probs.size
        return probs

    def expectation(self, theta: np.ndarray) -> float:
        self.n_evals += 1
        probs = self.probabilities(theta)
        if self.cvar_alpha >= 1.0:
            return float(probs @ self.diag)
        order = np.argsort(self.diag)
        cum = np.cumsum(probs[order])
        k = int(np.searchsorted(cum, self.cvar_alpha)) + 1
        w = probs[order][:k].copy()
        w[-1] -= max(cum[k - 1] - self.cvar_alpha, 0.0)
        return float(w @ self.diag[order][:k] / self.cvar_alpha)

    def sample(self, theta: np.ndarray, shots: int) -> np.ndarray:
        probs = self.probabilities(theta)
        ks = self.rng.choice(probs.size, size=shots, p=probs / probs.sum())
        X = ((ks[:, None] >> np.arange(self.n)) & 1).astype(np.int8)
        if self.noise.readout > 0:
            X ^= (self.rng.random(X.shape) < self.noise.readout).astype(np.int8)
        return X

    def top_states(self, theta: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        probs = self.probabilities(theta)
        idx = np.argsort(-probs)[:k]
        return ((idx[:, None] >> np.arange(self.n)) & 1).astype(np.int8), probs[idx]


def _interp(theta: np.ndarray, p: int) -> np.ndarray:
    """INTERP: extend optimal depth-p parameters to depth p+1."""
    g, b = theta[:p], theta[p:]

    def ext(v):
        v = np.concatenate([[0.0], v, [0.0]])
        return np.array([(i / p) * v[i] + ((p - i) / p) * v[i + 1] for i in range(p + 1)])
    return np.concatenate([ext(g), ext(b)])


def linear_ramp(p: int, dt: float = 0.75) -> np.ndarray:
    k = (np.arange(p) + 0.5) / p
    return np.concatenate([k * dt * np.pi / 2, (1 - k) * dt * np.pi / 4])


def local_search(problem: DefenseProblem, x: np.ndarray, max_rounds: int = 50) -> np.ndarray:
    """Best-improvement 1-flip descent on the true objective (feasible moves only)."""
    x = x.copy()
    cur = problem.objective(x) if problem.feasible(x) else np.inf
    for _ in range(max_rounds):
        neigh = np.repeat(x[None, :], problem.n, axis=0)
        neigh[np.arange(problem.n), np.arange(problem.n)] ^= 1
        objs = np.atleast_1d(problem.objective(neigh))
        feas = np.array([problem.feasible(v) for v in neigh])
        objs = np.where(feas, objs, np.inf)
        j = int(np.argmin(objs))
        if objs[j] < cur - 1e-12:
            x, cur = neigh[j], objs[j]
        else:
            break
    return x


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
class QAOASolver(Solver):
    name = "qaoa"

    def __init__(self, p: int = 1, backend: str = "statevector", noise=None,
                 shots: int | None = 2048, restarts: int = 3, maxiter: int = 200,
                 cvar_alpha: float = 1.0, init: str = "interp", seed: int = 0,
                 warm_start: np.ndarray | None = None, polish: bool = False,
                 top_k: int = 64, qubo_kwargs: dict | None = None):
        self.p, self.backend, self.noise, self.shots = p, backend, noise, shots
        self.restarts, self.maxiter, self.cvar_alpha = restarts, maxiter, cvar_alpha
        self.init, self.seed, self.warm_start = init, seed, warm_start
        self.polish, self.top_k = polish, top_k
        self.qubo_kwargs = qubo_kwargs or {}
        self.name = f"qaoa_p{p}" + ("" if get_noise(noise).is_ideal else f"_{get_noise(noise).name}")
        self.last_params: np.ndarray | None = None

    # -- parameter optimisation ------------------------------------------------------
    def _optimise(self, engine_for_p, rng) -> tuple[np.ndarray, list[float], int]:
        history: list[float] = []
        n_evals = 0
        if self.warm_start is not None and len(self.warm_start) == 2 * self.p:
            starts_p = [np.asarray(self.warm_start, float)]
            ladder = [self.p]
        elif self.init == "interp" and self.p > 1:
            ladder = list(range(1, self.p + 1))
            starts_p = None
        else:
            ladder = [self.p]
            starts_p = None

        theta = None
        for depth in ladder:
            eng = engine_for_p(depth)
            if starts_p is not None:
                starts = starts_p
            elif theta is not None:
                starts = [_interp(theta, depth - 1)]
            else:
                starts = [linear_ramp(depth)] + [
                    np.concatenate([rng.uniform(0, np.pi, depth), rng.uniform(0, np.pi / 2, depth)])
                    for _ in range(max(0, self.restarts - 1))]
            best = None
            for s in starts:
                trace: list[float] = []

                def f(t, eng=eng, trace=trace):
                    v = eng.expectation(t)
                    trace.append(v)
                    return v
                res = minimize(f, s, method="COBYLA",
                               options={"maxiter": self.maxiter, "rhobeg": 0.3})
                if best is None or res.fun < best[1]:
                    best = (res.x, res.fun, trace)
            theta = best[0]
            history = best[2]
            n_evals += eng.n_evals
        return theta, history, n_evals

    def _solve(self, problem: DefenseProblem):
        rng = np.random.default_rng(self.seed)
        t_q = time.perf_counter()
        q = problem.qubo(**self.qubo_kwargs) if self.qubo_kwargs else problem.qubo()
        t_build = time.perf_counter() - t_q
        n_dec = q.n_decision

        if self.backend == "statevector":
            engines: dict[int, StatevectorQAOA] = {}

            def engine_for_p(d):
                if d not in engines:
                    engines[d] = StatevectorQAOA(q, d, self.noise, self.cvar_alpha, self.seed)
                return engines[d]
            t0 = time.perf_counter()
            theta, history, n_evals = self._optimise(engine_for_p, rng)
            t_opt = time.perf_counter() - t0
            eng = engine_for_p(self.p)
            if self.shots:
                samples = eng.sample(theta, self.shots)
            else:
                samples, _ = eng.top_states(theta, self.top_k)
            probs = eng.probabilities(theta)
            exp_e = float(probs @ eng.energies)
            e_min, e_max = eng.e_min, eng.e_max
            ks_opt = np.flatnonzero(np.isclose(eng.energies, e_min))
            p_opt = float(probs[ks_opt].sum())
            amplification = p_opt / (len(ks_opt) / probs.size)
            td = transpiled_depth(q, self.p) if q.n <= 22 else None
            depth = td["depth"] if td else gate_counts(q, self.p)["depth_upper_bound"]
        elif self.backend in ("qiskit", "aer"):
            from qadapt.quantum_engine.qiskit_backend import QiskitQAOARunner
            runner = QiskitQAOARunner(q, self.p, self.noise, self.shots or 2048, self.seed,
                                      self.cvar_alpha)
            t0 = time.perf_counter()
            theta, history, n_evals = self._optimise(lambda d: runner.for_depth(d), rng)
            t_opt = time.perf_counter() - t0
            samples = runner.for_depth(self.p).sample(theta)
            exp_e = float(np.mean(q.energy(samples)))
            e_min = e_max = None
            p_opt = amplification = None
            td = None
            depth = runner.for_depth(self.p).depth
        else:
            raise ValueError(f"unknown backend {self.backend!r}")

        self.last_params = theta
        decision = samples[:, :n_dec]
        x, _, feas = problem.best_of(decision)
        info = {
            "backend": self.backend,
            "p": self.p,
            "noise": get_noise(self.noise).name,
            "n_qubits": q.n,
            "n_slack_qubits": q.n_slack,
            "circuit_depth": depth,
            "gate_counts": gate_counts(q, self.p),
            "transpiled": td,
            "params": theta.tolist(),
            "function_evals": n_evals,
            "convergence": [float(v) for v in history],
            "qubo_build_s": t_build,
            "optimise_s": t_opt,
            "expected_energy": exp_e,
            "unique_samples": int(len(np.unique(decision, axis=0))),
            "qubo_meta": {k: v for k, v in q.meta.items() if k != "h"},
        }
        if e_min is not None:
            info["approximation_ratio"] = (e_max - exp_e) / max(e_max - e_min, 1e-12)
            info["p_optimal_state"] = p_opt
            info["optimal_state_amplification"] = amplification
            info["qubo_optimum"] = e_min
        if self.polish:
            x_pol = local_search(problem, x)
            info["polished_gain"] = float(problem.objective(x) - problem.objective(x_pol))
            x = x_pol
        info["raw_feasible"] = feas
        return x, info
