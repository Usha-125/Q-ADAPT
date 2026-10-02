"""Qiskit / Qiskit Aer execution of the QAOA circuit.

Builds ``QAOAAnsatz`` from the Ising ``SparsePauliOp``, transpiles once per
depth for the Aer simulator (optionally with a gate-level noise model) and
evaluates the (CVaR) expectation from measured shots. The same code path can
target IBM Quantum hardware by swapping the backend for a runtime sampler.
"""

from __future__ import annotations

import numpy as np

from qadapt.quantum_engine.ising import qubo_to_ising
from qadapt.quantum_engine.noise_models import aer_noise_model, get_noise
from qadapt.quantum_engine.qubo_builder import QUBO


def qiskit_available() -> bool:
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
        return True
    except ImportError:
        return False


class _DepthRunner:
    def __init__(self, parent: QiskitQAOARunner, p: int):
        from qiskit import transpile
        from qiskit.circuit.library import QAOAAnsatz
        self.parent = parent
        self.p = p
        ansatz = QAOAAnsatz(parent.op, reps=p)
        ansatz.measure_all()
        self.circuit = transpile(ansatz, parent.sim, optimization_level=1, seed_transpiler=parent.seed)
        self.params = list(ansatz.parameters)  # sorted: beta[0..p-1], gamma[0..p-1]
        self.depth = int(self.circuit.depth())
        self.n_evals = 0

    def _bind(self, theta: np.ndarray):
        g, b = theta[: self.p], theta[self.p:]
        values = {}
        for prm in self.params:
            name, idx = prm.name, prm.index if hasattr(prm, "index") else 0
            values[prm] = float(b[idx] if name.startswith("β") or "beta" in name else g[idx])
        return self.circuit.assign_parameters(values)

    def sample(self, theta: np.ndarray) -> np.ndarray:
        par = self.parent
        counts = par.sim.run(self._bind(theta), shots=par.shots,
                             seed_simulator=par.seed).result().get_counts()
        rows = []
        for bitstr, c in counts.items():
            bits = np.array([int(ch) for ch in bitstr.replace(" ", "")[::-1]], dtype=np.int8)
            rows.extend([bits] * c)
        return np.array(rows, dtype=np.int8)

    def expectation(self, theta: np.ndarray) -> float:
        self.n_evals += 1
        X = self.sample(theta)
        e = (np.asarray(self.parent.q.energy(X)) - self.parent.e_ref) / self.parent.scale
        a = self.parent.cvar_alpha
        if a >= 1.0:
            return float(e.mean())
        e.sort()
        return float(e[: max(1, int(np.ceil(a * len(e))))].mean())


class QiskitQAOARunner:
    def __init__(self, q: QUBO, p: int, noise=None, shots: int = 2048, seed: int = 0,
                 cvar_alpha: float = 1.0):
        if not qiskit_available():
            raise ImportError("Qiskit backend requires: pip install 'qadapt[quantum]'")
        from qiskit_aer import AerSimulator
        self.q, self.shots, self.seed, self.cvar_alpha = q, shots, seed, cvar_alpha
        ising = qubo_to_ising(q)
        # same normalisation as the state-vector engine: spread of |h|+|J| as energy scale
        self.scale = max(float(np.abs(ising.h).sum() + np.abs(ising.J).sum()), 1e-9)
        self.e_ref = ising.offset - self.scale
        self.op = ising.to_sparse_pauli_op(scale=1.0 / self.scale)
        nl = get_noise(noise)
        self.sim = AerSimulator(noise_model=None if nl.is_ideal else aer_noise_model(nl),
                                seed_simulator=seed)
        self._runners: dict[int, _DepthRunner] = {}

    def for_depth(self, p: int) -> _DepthRunner:
        if p not in self._runners:
            self._runners[p] = _DepthRunner(self, p)
        return self._runners[p]
