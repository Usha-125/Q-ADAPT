"""AQDO - Adaptive Quantum Defense Optimization controller.

Loop:  observe -> update attack graph -> re-weight objective -> regenerate
       candidates & QUBO -> solve (QAOA, warm-started) -> analyst approval ->
       apply -> learn effectiveness from outcomes -> repeat on state change.

Policies for the adaptation experiment (H5):
  ``none``      no defense
  ``static``    optimise once at t=0, never again
  ``reoptimize`` re-optimise on state change, nominal effectiveness, fixed weights
  ``aqdo``      re-optimise + Bayesian effectiveness learning + adaptive weights
                + QAOA parameter warm starts
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from qadapt.adaptive_engine.environment import AttackEnvironment
from qadapt.adaptive_engine.learning import AdaptiveWeights, EffectivenessLearner
from qadapt.attack_graph.graph_builder import AttackGraph
from qadapt.attack_graph.topology import Topology
from qadapt.core.config import OptimizationConfig
from qadapt.core.models import DefenseAction, ThreatAssessment
from qadapt.pipeline import DecisionReport, QAdaptPipeline, make_solver

POLICIES = ("none", "static", "reoptimize", "aqdo")


@dataclass
class StepRecord:
    t: int
    belief_risk: float
    true_loss: float
    compromised: int
    detected: int
    reoptimized: bool
    applied: list[str] = field(default_factory=list)
    alpha: float = 1.0
    function_evals: int = 0
    effectiveness: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class AQDOController:
    def __init__(self, topology: Topology, config: OptimizationConfig | None = None,
                 policy: str = "aqdo", solver: str = "qaoa", solver_kw: dict | None = None,
                 max_qubits: int = 12, reopt_threshold: float = 0.02,
                 approve: Callable[[DecisionReport], list[DefenseAction]] | None = None,
                 risk_model: str = "propagation", candidate_focus: str = "risk"):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.policy = policy
        self.config = config or OptimizationConfig()
        self.ag = AttackGraph(topology)
        self.learner = EffectivenessLearner()
        self.weights = AdaptiveWeights(base=copy.deepcopy(self.config.weights))
        self.solver_name = solver
        self.solver_kw = dict(solver_kw or {})
        self.max_qubits = max_qubits
        self.reopt_threshold = reopt_threshold
        self.approve = approve or (lambda report: report.selected_actions)
        self.risk_model = risk_model
        self.candidate_focus = candidate_focus
        self.applied: list[DefenseAction] = []
        self.history: list[StepRecord] = []
        self.reports: list[DecisionReport] = []
        self._last_risk: float | None = None
        self._qaoa_params: np.ndarray | None = None
        self._optimised_once = False

    # ------------------------------------------------------------------------------
    def observe(self, observations: list[ThreatAssessment], outcomes=()) -> None:
        self.ag.apply_threats(observations)
        for o in observations:
            if o.probability * o.confidence >= 0.7:
                self.ag.mark_compromised(o.host_id)
        if self.policy == "aqdo":
            for out in outcomes:
                for t in set(out.action_types):
                    self.learner.update(t, out.blocked)
            self.ag.apply_defenses(self.applied, self.learner.estimates())
        else:
            self.ag.apply_defenses(self.applied)

    def _should_reoptimize(self, risk: float, new_detections: bool) -> bool:
        if self.policy == "none":
            return False
        if self.policy == "static":
            return not self._optimised_once
        if self._last_risk is None or new_detections:
            return True
        return abs(risk - self._last_risk) > self.reopt_threshold

    def decide(self, new_detections: bool = True) -> tuple[DecisionReport | None, StepRecord]:
        eff = self.learner.estimates() if self.policy == "aqdo" else {}
        cfg = copy.deepcopy(self.config)
        pipe = QAdaptPipeline(self.ag, cfg, risk_model=self.risk_model, max_qubits=self.max_qubits,
                              effectiveness=eff, candidate_focus=self.candidate_focus)
        risk = pipe.risk_model.total(self.ag.compile(), self.ag)
        if self.policy == "aqdo":
            cfg.weights = self.weights.weights(risk)
        report, evals = None, 0
        if self._should_reoptimize(risk, new_detections):
            problem = pipe.build_problem(exclude={a.id for a in self.applied})
            if problem.n:
                kw = dict(self.solver_kw)
                if self.solver_name == "qaoa" and self.policy == "aqdo" and self._qaoa_params is not None:
                    kw["warm_start"] = self._qaoa_params
                solver = make_solver(self.solver_name, **kw)
                report = pipe.decide(solver, problem)
                if self.solver_name == "qaoa":
                    self._qaoa_params = getattr(solver, "last_params", None)
                    evals = report.result.info.get("function_evals", 0)
                self.reports.append(report)
            self._optimised_once = True
            self._last_risk = risk
        rec = StepRecord(t=len(self.history), belief_risk=risk, true_loss=0.0, compromised=0,
                         detected=len(self.ag.compromised), reoptimized=report is not None,
                         alpha=cfg.weights.alpha, function_evals=evals,
                         effectiveness={t.value: round(v, 3) for t, v in eff.items()})
        return report, rec

    def commit(self, actions: list[DefenseAction]) -> None:
        self.applied.extend(actions)
        eff = self.learner.estimates() if self.policy == "aqdo" else None
        self.ag.apply_defenses(self.applied, eff)


def run_episode(topology: Topology, policy: str = "aqdo", steps: int = 8,
                config: OptimizationConfig | None = None, solver: str = "qaoa",
                solver_kw: dict | None = None, initial: list[str] | None = None,
                true_effectiveness: dict | None = None, seed: int = 0, detector=None,
                max_qubits: int = 12, env_kw: dict | None = None,
                risk_model: str = "propagation", candidate_focus: str = "risk") -> dict:
    """Simulate an attack campaign against one defense policy; return the trajectory."""
    env = AttackEnvironment(topology, true_effectiveness, initial=initial, seed=seed,
                            detector=detector, **(env_kw or {}))
    ctl = AQDOController(topology, config, policy, solver, solver_kw, max_qubits=max_qubits,
                         risk_model=risk_model, candidate_focus=candidate_focus)
    first = env.step()
    ctl.observe(first.observations)
    new_det = bool(first.observations)
    cumulative = 0.0
    for _ in range(steps):
        report, rec = ctl.decide(new_detections=new_det)
        chosen = ctl.approve(report) if report else []
        ctl.commit(chosen)
        env.apply(chosen)
        tick = env.step()
        ctl.observe(tick.observations, tick.outcomes)
        new_det = bool(tick.observations)
        rec.applied = [a.id for a in chosen]
        rec.true_loss = env.realised_loss()
        rec.compromised = len(env.compromised)
        cumulative += rec.true_loss
        ctl.history.append(rec)
    return {
        "policy": policy,
        "final_loss": env.realised_loss(),
        "cumulative_loss": cumulative,
        "critical_compromised": env.critical_compromised(),
        "compromised": sorted(env.compromised),
        "n_actions": len(ctl.applied),
        "total_cost": float(sum(a.cost for a in ctl.applied)),
        "total_disruption": float(sum(a.disruption for a in ctl.applied)),
        "function_evals": int(sum(r.function_evals for r in ctl.history)),
        "history": [r.to_dict() for r in ctl.history],
        "learned_effectiveness": {t.value: round(v, 3) for t, v in ctl.learner.estimates().items()},
    }
