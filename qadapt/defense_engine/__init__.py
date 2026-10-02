"""Defense engine: candidate actions, effect model and policy constraints."""

from qadapt.defense_engine.action_effectiveness import DefenseEvaluator, PortfolioMetrics
from qadapt.defense_engine.action_generator import TEMPLATES, ActionGenerator
from qadapt.defense_engine.policy_constraints import PolicySet, build_policy


def prescreen(evaluator: DefenseEvaluator, k: int) -> list[int]:
    """Indices of the ``k`` actions with the best risk-reduction per unit of burden."""
    import numpy as np
    gain = evaluator.marginal_reductions()
    burden = 0.5 + evaluator.cost + evaluator.time + evaluator.disruption
    order = np.argsort(-(gain / burden))
    return sorted(int(i) for i in order[:k] if gain[i] > 1e-6)


__all__ = [
    "TEMPLATES",
    "ActionGenerator",
    "DefenseEvaluator",
    "PolicySet",
    "PortfolioMetrics",
    "build_policy",
    "prescreen",
]
