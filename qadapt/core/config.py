"""Optimization configuration (objective weights and operational constraints)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObjectiveWeights:
    """Weights of Q(x) = alpha*Risk(x) + beta*C(x) + gamma*T(x) + delta*D(x) + P(x)."""

    alpha: float = 1.0  # residual risk (security)
    beta: float = 0.15  # defense cost
    gamma: float = 0.10  # response delay
    delta: float = 0.20  # business disruption

    def as_dict(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "delta": self.delta}


@dataclass
class OptimizationConfig:
    """Operational constraints for a defense portfolio.

    ``None`` disables a constraint. Budgets are expressed in the same normalised
    units as the action attributes (sum over the selected actions).
    """

    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    budget: float | None = None
    max_time: float | None = None
    max_disruption: float | None = None
    max_actions: int | None = None
    # How inequality constraints are embedded into the QUBO:
    #   "slack"      exact, adds ceil(log2) slack qubits per constraint
    #   "unbalanced" no extra qubits (Montanez-Barrera et al., 2022 penalisation)
    constraint_encoding: str = "unbalanced"
    penalty: float | None = None  # None = derived automatically from objective scale
    slack_resolution: int = 4  # bits of discretisation for slack encoding
    protected_assets: tuple[str, ...] = ()  # assets that must not be disrupted/isolated
