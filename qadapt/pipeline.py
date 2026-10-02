"""End-to-end Q-ADAPT decision pipeline.

    threats -> attack graph -> risk -> candidate actions -> QUBO -> solver
            -> defense portfolio -> explanation -> (analyst approval)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qadapt.attack_graph.graph_builder import AttackGraph
from qadapt.attack_graph.path_analysis import critical_targets, top_attack_paths
from qadapt.core.config import OptimizationConfig
from qadapt.core.models import ActionType, DefenseAction, ThreatAssessment
from qadapt.defense_engine import ActionGenerator, DefenseEvaluator, build_policy, prescreen
from qadapt.explainability import explain
from qadapt.optimization.problem import DefenseProblem, Solver, SolveResult
from qadapt.risk_engine.risk_model import RiskModel

SOLVERS = ("qaoa", "exhaustive", "greedy", "score_ranking", "simulated_annealing",
           "genetic", "milp", "random_sampling")


def make_solver(name: str = "qaoa", **kw) -> Solver:
    if name == "qaoa":
        from qadapt.quantum_engine import QAOASolver
        return QAOASolver(**kw)
    from qadapt.classical_baselines import BASELINES
    if name not in BASELINES:
        raise KeyError(f"unknown solver {name!r}; choose from {SOLVERS}")
    return BASELINES[name](**kw)


@dataclass
class DecisionReport:
    risk_before: float
    risk_after: float
    paths_before: int
    paths_after: int
    result: SolveResult
    actions: list[DefenseAction]
    explanations: list = field(default_factory=list)
    qubo_info: dict = field(default_factory=dict)
    top_paths: list = field(default_factory=list)
    n_candidates: int = 0

    @property
    def selected_actions(self) -> list[DefenseAction]:
        return [self.actions[i] for i in np.flatnonzero(self.result.x)]

    def to_dict(self) -> dict:
        m = self.result.metrics
        return {
            "risk_before": round(self.risk_before, 5),
            "risk_after": round(self.risk_after, 5),
            "risk_reduction": round(m.risk_reduction if m else 0.0, 5),
            "paths_before": self.paths_before,
            "paths_after": self.paths_after,
            "n_candidates": self.n_candidates,
            "candidates": [a.to_dict() for a in self.actions],
            "selected": [a.to_dict() for a in self.selected_actions],
            "result": self.result.to_dict(),
            "explanations": [e.to_dict() for e in self.explanations],
            "qubo": self.qubo_info,
            "top_paths": [p.to_dict() for p in self.top_paths],
        }


class QAdaptPipeline:
    def __init__(self, ag: AttackGraph, config: OptimizationConfig | None = None,
                 risk_model: str = "propagation", max_qubits: int = 14,
                 effectiveness: dict[ActionType, float] | None = None,
                 path_threshold: float = 0.05, candidate_focus: str = "risk"):
        self.ag = ag
        self.config = config or OptimizationConfig()
        self.risk_model = RiskModel(risk_model)
        self.max_qubits = max_qubits
        self.effectiveness = effectiveness or {}
        self.path_threshold = path_threshold
        if candidate_focus not in ("risk", "threatened"):
            raise ValueError("candidate_focus must be 'risk' or 'threatened'")
        self.candidate_focus = candidate_focus

    def ingest(self, threats: list[ThreatAssessment]) -> None:
        self.ag.apply_threats(threats)

    def build_problem(self, exclude: set[str] | None = None) -> DefenseProblem:
        gen = ActionGenerator(self.ag, self.effectiveness)
        if self.candidate_focus == "threatened":  # alert-driven: only hosts with ML evidence
            actions = gen.generate(focus=sorted(self.ag.threat), include_segmentation=False)
        else:
            actions = gen.generate()
        actions = [a for a in actions if a.id not in (exclude or set())]
        cg = self.ag.compile()
        ev = DefenseEvaluator(self.ag, actions, self.risk_model, cg)
        if len(actions) > self.max_qubits:
            keep = prescreen(ev, self.max_qubits)
            actions = [actions[i] for i in keep]
            ev = DefenseEvaluator(self.ag, actions, self.risk_model, cg)
        policy = build_policy(actions, self.config.protected_assets)
        return DefenseProblem(ev, self.config, policy)

    def _paths(self, edge_mult=None) -> tuple[int, list]:
        targets = critical_targets(self.ag.g)
        sources = sorted(self.ag.compromised | {h for h, t in self.ag.threat.items() if t >= 0.5})
        paths = top_attack_paths(self.ag.g, targets, k=5, sources=sources, edge_mult=edge_mult)
        return sum(p.probability >= self.path_threshold for p in paths), paths

    def decide(self, solver: Solver | str = "qaoa", problem: DefenseProblem | None = None,
               **solver_kw) -> DecisionReport:
        problem = problem or self.build_problem()
        solver = make_solver(solver, **solver_kw) if isinstance(solver, str) else solver
        result = solver.solve(problem)
        ev = problem.evaluator
        n_before, paths = self._paths()
        edge_red = ev.edge_reduction(result.x) if result.x.any() else {}
        n_after, _ = self._paths({e: 1 - r for e, r in edge_red.items()})
        q = problem._qubo
        qinfo = {}
        if q is not None:
            qinfo = {"n_variables": q.n, "n_decision": q.n_decision, "n_slack": q.n_slack,
                     **{k: v for k, v in q.meta.items() if k != "h"},
                     "matrix": np.round(q.matrix(), 5).tolist(), "var_names": q.var_names}
        return DecisionReport(
            risk_before=ev.base_risk,
            risk_after=result.metrics.residual_risk,
            paths_before=n_before,
            paths_after=n_after,
            result=result,
            actions=problem.actions,
            explanations=explain(problem, result.x),
            qubo_info=qinfo,
            top_paths=paths[:8],
            n_candidates=problem.n,
        )
