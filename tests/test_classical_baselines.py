import pytest

from qadapt.attack_graph import AttackGraph, demo_topology
from qadapt.classical_baselines import BASELINES, ExhaustiveSolver
from qadapt.core.config import OptimizationConfig
from qadapt.core.models import ThreatAssessment
from qadapt.defense_engine import ActionGenerator, DefenseEvaluator, prescreen
from qadapt.optimization import DefenseProblem


@pytest.fixture(scope="module")
def problem():
    ag = AttackGraph(demo_topology())
    ag.apply_threats([ThreatAssessment("WEB-01", "WEB_ATTACK", 0.93, 0.9, source="203.0.113.7"),
                      ThreatAssessment("APP-01", "EXPLOIT", 0.7, 0.85)])
    acts = ActionGenerator(ag).generate()
    ev = DefenseEvaluator(ag, acts)
    ev = DefenseEvaluator(ag, [acts[i] for i in prescreen(ev, 12)])
    return DefenseProblem(ev, OptimizationConfig(budget=0.35, max_disruption=0.5, max_actions=4))


@pytest.fixture(scope="module")
def optimum(problem):
    return ExhaustiveSolver().solve(problem)


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baseline_feasible_and_bounded_by_optimum(problem, optimum, name):
    res = BASELINES[name]().solve(problem)
    assert res.feasible, res.selected
    assert res.objective >= optimum.objective - 1e-9


def test_strong_baselines_near_optimal(problem, optimum):
    for name in ("simulated_annealing", "genetic", "milp"):
        res = BASELINES[name]().solve(problem)
        assert res.objective <= optimum.objective * 1.1, name


def test_exhaustive_reports_feasible_count(optimum):
    assert optimum.info["feasible_portfolios"] > 0
