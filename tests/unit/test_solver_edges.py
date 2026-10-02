import numpy as np
import pytest

from qadapt.classical_baselines import BASELINES, ExhaustiveSolver, MILPSolver
from qadapt.core.config import OptimizationConfig
from qadapt.defense_engine import DefenseEvaluator
from qadapt.optimization import DefenseProblem, SolveResult
from qadapt.pipeline import SOLVERS, make_solver

FAST_KW = {"qaoa": {"p": 1, "restarts": 1, "maxiter": 30, "shots": 128},
           "simulated_annealing": {"n_steps": 400, "restarts": 1},
           "genetic": {"pop": 20, "generations": 10}}


def solver(name):
    return make_solver(name, **FAST_KW.get(name, {}))


@pytest.mark.parametrize("name", SOLVERS)
def test_zero_action_problem(demo_graph, name):
    prob = DefenseProblem(DefenseEvaluator(demo_graph, []), OptimizationConfig(budget=0.2))
    r = solver(name).solve(prob)
    assert r.feasible and r.x.shape == (0,) and r.selected == []
    assert r.metrics.risk_reduction == 0.0


@pytest.mark.parametrize("name", SOLVERS)
def test_single_action_problem(small_evaluator, name):
    prob = DefenseProblem(DefenseEvaluator(small_evaluator.ag, small_evaluator.actions[:1]), OptimizationConfig())
    r = solver(name).solve(prob)
    assert r.feasible and r.x.shape == (1,)


@pytest.mark.parametrize("name", SOLVERS)
def test_zero_budget_selects_nothing_costly(make_problem, name):
    r = solver(name).solve(make_problem(budget=0))
    assert r.feasible and r.metrics.cost == 0


@pytest.mark.parametrize("name", SOLVERS)
def test_every_solver_respects_policy(make_problem, name):
    prob = make_problem(budget=0.4)
    prob.policy.forbidden.add(0)
    prob.policy.conflicts.add((1, 2))
    prob.policy.mandatory.add(3)
    r = solver(name).solve(prob)
    if name in ("random_sampling", "score_ranking", "greedy"):
        # heuristics without constraint repair may fail a mandatory rule; they must say so
        assert r.feasible == (not prob.violations(r.x))
    else:
        assert r.feasible, prob.violations(r.x)


def test_infeasible_problem_is_reported(make_problem):
    prob = make_problem()
    prob.policy.mandatory.update({0, 1})
    prob.policy.conflicts.add((0, 1))  # mandatory + conflicting: no feasible portfolio exists
    r = ExhaustiveSolver().solve(prob)
    assert not r.feasible and r.info["feasible_portfolios"] == 0
    assert MILPSolver().solve(prob).info["status"]  # MILP reports infeasibility instead of crashing


def test_exhaustive_size_limit(make_problem):
    with pytest.raises(ValueError):
        ExhaustiveSolver(max_n=4).solve(make_problem())


def test_exhaustive_is_optimal_vs_all(make_problem):
    prob = make_problem(budget=0.35, max_actions=3)
    opt = ExhaustiveSolver().solve(prob).objective
    for name in BASELINES:
        r = solver(name).solve(prob)
        assert not r.feasible or r.objective >= opt - 1e-9


@pytest.mark.parametrize("surrogate", ["expansion", "regression"])
def test_milp_surrogates(make_problem, surrogate):
    r = MILPSolver(surrogate=surrogate).solve(make_problem(budget=0.3))
    assert r.feasible and "surrogate_objective" in r.info


def test_unknown_solver():
    with pytest.raises(KeyError):
        make_solver("annealer-9000")


def test_solve_result_json_roundtrip(make_problem):
    import json
    r = solver("qaoa").solve(make_problem())
    d = r.to_dict()
    json.dumps(d)
    assert isinstance(r, SolveResult) and d["x"] == [int(v) for v in r.x]


def test_seeded_heuristics_reproducible(make_problem):
    prob = make_problem(budget=0.3)
    for name in ("simulated_annealing", "genetic", "random_sampling"):
        a, b = solver(name).solve(prob), solver(name).solve(prob)
        assert np.array_equal(a.x, b.x), name
