"""QUBO <-> Ising mapping and Pauli-operator export.

With x_i = (1 - z_i) / 2 (bit 1 <-> Z eigenvalue -1, Qiskit convention)::

    f(x) = offset' + sum_i hz_i z_i + sum_{i<j} Jz_ij z_i z_j
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qadapt.quantum_engine.qubo_builder import QUBO


@dataclass
class Ising:
    h: np.ndarray  # local fields
    J: np.ndarray  # couplings, strictly upper-triangular
    offset: float

    @property
    def n(self) -> int:
        return len(self.h)

    def energy_z(self, Z: np.ndarray) -> np.ndarray | float:
        Z = np.asarray(Z, dtype=float)
        single = Z.ndim == 1
        Z2 = np.atleast_2d(Z)
        e = self.offset + Z2 @ self.h + np.einsum("bi,ij,bj->b", Z2, self.J, Z2)
        return float(e[0]) if single else e

    def n_terms(self) -> tuple[int, int]:
        return int(np.count_nonzero(self.h)), int(np.count_nonzero(np.triu(self.J, 1)))

    def to_sparse_pauli_op(self, scale: float = 1.0):
        """Qiskit ``SparsePauliOp`` (qubit i <-> variable i, little-endian labels)."""
        from qiskit.quantum_info import SparsePauliOp
        n = self.n
        terms = []
        for i in range(n):
            if abs(self.h[i]) > 1e-12:
                terms.append(("Z", [i], self.h[i] * scale))
        iu, ju = np.nonzero(np.triu(self.J, 1))
        for i, j in zip(iu, ju, strict=True):
            if abs(self.J[i, j]) > 1e-12:
                terms.append(("ZZ", [int(i), int(j)], self.J[i, j] * scale))
        if not terms:
            terms.append(("I", [0], 0.0))
        return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def qubo_to_ising(q: QUBO) -> Ising:
    Jq = np.triu(q.quad, 1)
    Js = Jq + Jq.T
    h = -q.linear / 2.0 - Js.sum(axis=1) / 4.0
    offset = q.offset + q.linear.sum() / 2.0 + Jq.sum() / 4.0
    return Ising(h, Jq / 4.0, float(offset))


def bits_to_spins(X: np.ndarray) -> np.ndarray:
    return 1 - 2 * np.asarray(X)


def all_bitstrings(n: int) -> np.ndarray:
    ks = np.arange(1 << n)
    return ((ks[:, None] >> np.arange(n)) & 1).astype(np.int8)


def diagonal_energies(q: QUBO, chunk: int = 1 << 16) -> np.ndarray:
    """QUBO energy of every computational basis state (index k, bit i = x_i)."""
    if q.n > 24:
        raise ValueError("state-vector diagonal limited to n <= 24")
    out = np.empty(1 << q.n)
    for start in range(0, 1 << q.n, chunk):
        ks = np.arange(start, min(start + chunk, 1 << q.n))
        X = ((ks[:, None] >> np.arange(q.n)) & 1).astype(float)
        out[start:start + len(ks)] = q.energy(X)
    return out
