"""Quantum noise levels for the noise-robustness experiments.

Each level defines single-qubit / two-qubit depolarising error rates and a
symmetric readout error. They are used (a) to build Qiskit Aer noise models and
(b) by the fast state-vector engine through a global-depolarising + readout
approximation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseLevel:
    name: str
    p1: float  # 1-qubit depolarising probability per gate
    p2: float  # 2-qubit depolarising probability per gate
    readout: float  # bit-flip probability at measurement

    @property
    def is_ideal(self) -> bool:
        return self.p1 == 0 and self.p2 == 0 and self.readout == 0


NOISE_LEVELS: dict[str, NoiseLevel] = {
    "ideal": NoiseLevel("ideal", 0.0, 0.0, 0.0),
    "low": NoiseLevel("low", 1e-4, 1e-3, 0.01),
    "medium": NoiseLevel("medium", 5e-4, 5e-3, 0.03),
    "high": NoiseLevel("high", 2e-3, 2e-2, 0.06),
}


def get_noise(level: str | NoiseLevel | None) -> NoiseLevel:
    if level is None:
        return NOISE_LEVELS["ideal"]
    if isinstance(level, NoiseLevel):
        return level
    if level not in NOISE_LEVELS:
        raise KeyError(f"unknown noise level {level!r}; choose from {list(NOISE_LEVELS)}")
    return NOISE_LEVELS[level]


def circuit_fidelity(level: NoiseLevel, n_1q: int, n_2q: int) -> float:
    """Probability that no gate error occurs (global depolarising approximation)."""
    return (1 - level.p1) ** n_1q * (1 - level.p2) ** n_2q


def aer_noise_model(level: str | NoiseLevel):  # pragma: no cover - requires qiskit-aer
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    lv = get_noise(level)
    nm = NoiseModel()
    if lv.is_ideal:
        return nm
    nm.add_all_qubit_quantum_error(depolarizing_error(lv.p1, 1), ["rx", "rz", "h", "sx", "x", "u"])
    nm.add_all_qubit_quantum_error(depolarizing_error(lv.p2, 2), ["cx", "rzz", "cz"])
    r = lv.readout
    nm.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    return nm
