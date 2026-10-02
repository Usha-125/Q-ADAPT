"""Explanations for recommended (and rejected) defense actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qadapt.optimization.problem import DefenseProblem
from qadapt.risk_engine.asset_criticality import downstream_exposure


def _level(v: float, lo: float = 0.1, hi: float = 0.3) -> str:
    return "LOW" if v < lo else "MEDIUM" if v < hi else "HIGH"


@dataclass
class ActionExplanation:
    action_id: str
    description: str
    selected: bool
    reasons: list[str] = field(default_factory=list)
    factors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"action_id": self.action_id, "description": self.description,
                "selected": self.selected, "reasons": self.reasons, "factors": self.factors}


def explain(problem: DefenseProblem, x: np.ndarray, n_rejected: int = 4) -> list[ActionExplanation]:
    """Explain each selected action and the strongest non-selected alternatives.

    * contribution = J(x without a) - J(x): how much worse the plan gets without it
    * standalone   = risk reduction of the action applied alone
    * for rejected actions: J(x with a) - J(x), constraint violations, redundancy
    """
    x = np.asarray(x).astype(int)
    ev, ag = problem.evaluator, problem.evaluator.ag
    base_obj = problem.objective(x)
    standalone = ev.marginal_reductions() / max(ev.base_risk, 1e-12)
    out: list[ActionExplanation] = []

    def asset_factors(a):
        t = ag.threat_info.get(a.target)
        fac = {"standalone_risk_reduction": round(float(standalone[i]), 4),
               "cost": round(a.cost, 3), "time": round(a.time, 3),
               "disruption": round(a.disruption, 3)}
        if a.target in ag.assets:
            asset = ag.assets[a.target]
            fac.update({"target_criticality": round(asset.criticality, 3),
                        "downstream_exposure": round(downstream_exposure(ag, a.target), 3)})
        if t:
            fac.update({"attack_type": t.attack_type, "attack_probability": round(t.probability, 3)})
        return fac

    for i in np.flatnonzero(x):
        a = problem.actions[i]
        y = x.copy()
        y[i] = 0
        contrib = float(problem.objective(y) - base_obj)
        fac = asset_factors(a)
        fac["portfolio_contribution"] = round(contrib, 4)
        reasons = []
        if "attack_probability" in fac:
            reasons.append(f"{fac['attack_type']} activity on {a.target} "
                           f"(p={fac['attack_probability']:.0%})")
        if "target_criticality" in fac:
            reasons.append(f"Asset criticality {fac['target_criticality']:.2f}; downstream exposure "
                           f"{_level(fac['downstream_exposure'], 0.15, 0.4)}")
        reasons.append(f"Expected risk reduction alone: {standalone[i]:.1%}")
        reasons.append(f"Removing it would worsen the plan objective by {contrib:+.4f}")
        reasons.append(f"Cost {_level(a.cost)}, response time {_level(a.time, 0.1, 0.4)}, "
                       f"business disruption {_level(a.disruption, 0.15, 0.4)}")
        out.append(ActionExplanation(a.id, a.description, True, reasons, fac))

    rejected = [i for i in np.argsort(-standalone) if not x[i]][:n_rejected]
    q = problem._qubo
    for i in rejected:
        a = problem.actions[i]
        y = x.copy()
        y[i] = 1
        delta = float(problem.objective(y) - base_obj)
        fac = asset_factors(a)
        fac["objective_if_added"] = round(delta, 4)
        reasons = []
        viol = problem.violations(y)
        if viol:
            names = []
            for v in viol:
                if v.startswith("conflict"):
                    j = [int(k) for k in v[9:-1].split(",") if int(k) != i]
                    names.append(f"conflicts with {problem.actions[j[0]].id}" if j else v)
                else:
                    names.append(f"would violate {v.split(':')[0]} limit")
            reasons.append("Infeasible: " + "; ".join(names))
        if q is not None and i < q.n_decision and "h" in q.meta:
            # redundancy: positive interaction cancels a large share of the standalone gain
            alpha = max(problem.config.weights.alpha, 1e-12)
            gain_i = abs(q.meta["h"][i])
            red = [problem.actions[j].id for j in np.flatnonzero(x)
                   if j < q.n_decision and (min(i, j), max(i, j)) not in problem.policy.conflicts
                   and q.quad[min(i, j), max(i, j)] / alpha > 0.4 * gain_i]
            if red:
                reasons.append("Largely redundant with " + ", ".join(red))
        if delta > 0:
            reasons.append(f"Adding it would worsen the objective by {delta:+.4f} "
                           f"(extra risk reduction does not justify cost/time/disruption)")
        if a.time >= 0.4:
            reasons.append("Long time-to-effect; faster actions give more immediate protection")
        if not reasons:
            reasons.append("Marginal benefit too small")
        out.append(ActionExplanation(a.id, a.description, False, reasons, fac))
    return out
