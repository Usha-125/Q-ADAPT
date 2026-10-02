"""Online learning of defense effectiveness and risk-adaptive objective weights."""

from __future__ import annotations

from dataclasses import dataclass, field

from qadapt.core.config import ObjectiveWeights
from qadapt.core.models import ActionType
from qadapt.defense_engine.action_generator import TEMPLATES


@dataclass
class EffectivenessLearner:
    """Beta-Bernoulli posterior over each action type's effectiveness.

    Prior Beta(s * e0, s * (1 - e0)) centred on the nominal effectiveness e0
    with pseudo-count strength ``s``. Each observed attack attempt on a defended
    edge that *would* have succeeded without the defense is a Bernoulli trial
    (blocked = success of the defense).
    """

    strength: float = 6.0
    alpha: dict[ActionType, float] = field(default_factory=dict)
    beta: dict[ActionType, float] = field(default_factory=dict)

    def __post_init__(self):
        for t, tpl in TEMPLATES.items():
            self.alpha.setdefault(t, self.strength * tpl.effectiveness)
            self.beta.setdefault(t, self.strength * (1 - tpl.effectiveness))

    def update(self, action_type: ActionType, blocked: bool, weight: float = 1.0) -> None:
        if blocked:
            self.alpha[action_type] += weight
        else:
            self.beta[action_type] += weight

    def estimate(self, action_type: ActionType) -> float:
        a, b = self.alpha[action_type], self.beta[action_type]
        return a / (a + b)

    def estimates(self) -> dict[ActionType, float]:
        return {t: self.estimate(t) for t in self.alpha}


@dataclass
class AdaptiveWeights:
    """Risk-adaptive re-weighting of the multi-objective QUBO.

    alpha_t = alpha_0 * (1 + kappa * max(0, R_t - R_ref) / R_ref)

    When the threat escalates beyond the reference level the security term
    gains weight relative to cost, delay and disruption, so the optimiser
    accepts more disruptive containment; when risk falls back the operational
    terms regain influence.
    """

    base: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    kappa: float = 1.5
    max_alpha_factor: float = 3.0
    reference_risk: float | None = None

    def weights(self, risk: float) -> ObjectiveWeights:
        if self.reference_risk is None:
            self.reference_risk = max(risk, 1e-6)
        rel = max(0.0, risk - self.reference_risk) / self.reference_risk
        factor = min(1.0 + self.kappa * rel, self.max_alpha_factor)
        b = self.base
        return ObjectiveWeights(b.alpha * factor, b.beta, b.gamma, b.delta)
