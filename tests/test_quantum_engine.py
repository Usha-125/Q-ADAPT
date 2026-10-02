import numpy as np
import pytest

from qadapt.attack_graph import AttackGraph, demo_topology
from qadapt.core.config import OptimizationConfig
from qadapt.core.models import ThreatAssessment
from qadapt.defense_engine import ActionGenerator, DefenseEvaluator, prescreen
from qadapt.optimization import DefenseProblem
from qadapt.quantum_engine import (
    QAOASolver,
    diagonal_energies,
    qiskit_available,
    qubo_to_ising,
)
from qadapt.quantum_engine.ising import all_bitstrings
from qadapt.quantum_engine.qaoa import qaoa_state


@pytest.fixture(scope="module")
def evaluator():
    ag = AttackGraph(demo_topology())
    ag.apply_threats([ThreatAssessment("WEB-01", "WEB_ATTACK", 0.93, 0.9, source="203.0.113.7"),
                      ThreatAssessment("APP-01", "EXPLOIT", 0.7, 0.85)])
    acts = ActionGenerator(ag).generate()
    ev = DefenseEvaluator(ag, acts)
    return DefenseEvaluator(ag, [acts[i] for i in prescreen(ev, 8)])


def _problem(ev, **cfg):
    return DefenseProblem(ev, OptimizationConfig(**cfg))


@pytest.mark.parametrize("encoding", ["unbalanced", "slack"])
@pytest.mark.parametrize("surrogate", ["expansion", "regression"])
def test_qubo_ising_equivalence(evaluator, encoding, surrogate):
    q = _problem(evaluator, budget=0.3, constraint_encoding=encoding).qubo(surrogate=surrogate)
    X = all_bitstrings(q.n)[:512].astype(float)
    np.testing.assert_allclose(q.energy(X), qubo_to_ising(q).energy_z(1 - 2 * X), atol=1e-9)


def test_expansion_exact_on_singletons(evaluator):
    prob = _problem(evaluator)
    q = prob.qubo(surrogate="expansion")
    h = np.array(q.meta["h"])
    true = np.asarray(evaluator.residual_risk(np.eye(prob.n, dtype=int))) / evaluator.base_risk
    np.testing.assert_allclose(1 + h, true, atol=1e-9)


def test_regression_surrogate_is_faithful(evaluator):
    q = _problem(evaluator).qubo(surrogate="regression")
    assert q.meta["fidelity"]["spearman"] > 0.9


def test_policy_penalty_excludes_conflicts(evaluator):
    prob = _problem(evaluator)
    q = prob.qubo()
    x, _ = q.brute_force()
    assert not prob.policy.violations(x[: q.n_decision])


def test_slack_encoding_qubo_minimum_is_feasible(evaluator):
    prob = _problem(evaluator, budget=0.25, constraint_encoding="slack")
    q = prob.qubo()
    x, _ = q.brute_force()
    assert prob.feasible(x[: q.n_decision])


def test_statevector_matches_qiskit(evaluator):
    if not qiskit_available():
        pytest.skip("qiskit not installed")
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import Statevector
    q = _problem(evaluator, budget=0.3).qubo()
    E = diagonal_energies(q)
    scale = E.max() - E.min()
    op = qubo_to_ising(q).to_sparse_pauli_op(scale=1 / scale)
    g, b = [0.4, 0.9], [0.7, 0.3]
    ans = QAOAAnsatz(op, reps=2)
    vals = {p: (b if p.name.startswith("β") else g)[p.index] for p in ans.parameters}
    ref = Statevector(ans.assign_parameters(vals)).probabilities()
    mine = np.abs(qaoa_state(E / scale, q.n, g, b)) ** 2
    np.testing.assert_allclose(mine, ref, atol=1e-10)


def test_qaoa_finds_feasible_good_solution(evaluator):
    prob = _problem(evaluator, budget=0.3)
    res = QAOASolver(p=2, restarts=2, maxiter=100).solve(prob)
    _, opt, _ = prob.best_of(all_bitstrings(prob.n))
    assert res.feasible
    assert res.objective <= opt + 0.05
    assert 0.5 < res.info["approximation_ratio"] <= 1.0
    assert res.info["optimal_state_amplification"] > 1.0


def test_noise_degrades_expectation(evaluator):
    prob = _problem(evaluator, budget=0.3)
    ideal = QAOASolver(p=2, restarts=1, maxiter=80).solve(prob)
    noisy = QAOASolver(p=2, restarts=1, maxiter=80, noise="high").solve(prob)
    assert noisy.info["approximation_ratio"] < ideal.info["approximation_ratio"]


def test_warm_start_uses_given_params(evaluator):
    prob = _problem(evaluator, budget=0.3)
    s1 = QAOASolver(p=1, restarts=1, maxiter=60)
    s1.solve(prob)
    s2 = QAOASolver(p=1, maxiter=30, warm_start=s1.last_params)
    r2 = s2.solve(prob)
    assert r2.feasible and r2.info["function_evals"] <= 30


def test_qiskit_backend_runs(evaluator):
    if not qiskit_available():
        pytest.skip("qiskit not installed")
    prob = _problem(evaluator, budget=0.3)
    res = QAOASolver(p=1, backend="qiskit", shots=256, restarts=1, maxiter=15).solve(prob)
    assert res.feasible and res.info["circuit_depth"] > 0
