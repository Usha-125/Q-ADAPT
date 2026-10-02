"""Property-based tests: invariants that must hold for *any* input."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qadapt.attack_graph import AttackGraph, generate_topology, propagate, total_risk
from qadapt.core.config import OptimizationConfig
from qadapt.core.models import ThreatAssessment
from qadapt.defense_engine import ActionGenerator, DefenseEvaluator, prescreen
from qadapt.optimization import DefenseProblem
from qadapt.quantum_engine import QUBO, qubo_to_ising
from qadapt.quantum_engine.ising import all_bitstrings
from qadapt.quantum_engine.qaoa import qaoa_state

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
floats = st.floats(-5, 5, allow_nan=False, allow_infinity=False)


@st.composite
def qubos(draw, max_n=6):
    n = draw(st.integers(1, max_n))
    lin = draw(arrays(float, n, elements=floats))
    quad = np.triu(draw(arrays(float, (n, n), elements=floats)), 1)
    return QUBO(lin, quad, draw(floats))


@SETTINGS
@given(qubos())
def test_ising_mapping_preserves_every_energy(q):
    X = all_bitstrings(q.n).astype(float)
    np.testing.assert_allclose(q.energy(X), qubo_to_ising(q).energy_z(1 - 2 * X), atol=1e-8)


@SETTINGS
@given(qubos(), st.lists(st.floats(0, 3), min_size=2, max_size=6))
def test_qaoa_state_is_normalised(q, params):
    p = len(params) // 2
    psi = qaoa_state(np.random.default_rng(0).random(1 << q.n), q.n, params[:p], params[p:2 * p])
    assert abs(np.vdot(psi, psi).real - 1) < 1e-9


@SETTINGS
@given(qubos(max_n=5))
def test_brute_force_finds_global_minimum(q):
    x, e = q.brute_force()
    assert e == min(q.energy(all_bitstrings(q.n).astype(float)))
    assert q.energy(x.astype(float)) == e


# ---- attack graph / risk ----------------------------------------------------------------------
@st.composite
def graphs(draw):
    seed = draw(st.integers(0, 50))
    ag = AttackGraph(generate_topology(draw(st.integers(8, 30)), seed=seed),
                     attacker_activity=draw(st.floats(0, 1)))
    hosts = list(ag.assets)
    k = draw(st.integers(0, 3))
    for h in draw(st.lists(st.sampled_from(hosts), min_size=k, max_size=k)):
        ag.apply_threats([ThreatAssessment(h, "EXPLOIT", draw(st.floats(0, 1)), draw(st.floats(0, 1)))])
    return ag


@SETTINGS
@given(graphs(), st.integers(0, 2 ** 31 - 1))
def test_propagation_bounded_and_monotone(ag, seed):
    cg = ag.compile()
    rng = np.random.default_rng(seed)
    m1 = rng.uniform(0, 1, cg.n_edges)
    m2 = m1 * rng.uniform(0, 1, cg.n_edges)  # weaker everywhere
    P1, P2 = propagate(cg, m1), propagate(cg, m2)
    assert np.all((P1 >= -1e-12) & (P1 <= 1 + 1e-12))
    assert np.all(P2 <= P1 + 1e-9)
    assert 0 <= total_risk(cg, P1) <= 1


@SETTINGS
@given(graphs())
def test_risk_increases_with_evidence(ag):
    cg = ag.compile()
    before = total_risk(cg, propagate(cg))
    ag.mark_compromised(next(iter(ag.assets)))
    cg2 = ag.compile()
    assert total_risk(cg2, propagate(cg2)) >= before - 1e-12


@st.composite
def problems(draw):
    ag = draw(graphs())
    ag.attacker_activity = max(ag.attacker_activity, 0.2)
    acts = ActionGenerator(ag).generate()
    if not acts:
        acts = ActionGenerator(ag).generate(focus=list(ag.assets)[:3])
    ev = DefenseEvaluator(ag, acts)
    keep = prescreen(ev, draw(st.integers(1, 7))) or [0]
    ev = DefenseEvaluator(ag, [acts[i] for i in keep])
    cfg = OptimizationConfig(budget=draw(st.none() | st.floats(0, 1)),
                             max_actions=draw(st.none() | st.integers(0, 4)),
                             constraint_encoding=draw(st.sampled_from(["unbalanced", "slack"])))
    return DefenseProblem(ev, cfg)


@SETTINGS
@given(problems(), st.integers(0, 2 ** 31 - 1))
def test_adding_an_action_never_increases_residual_risk(prob, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, prob.n)
    for i in range(prob.n):
        y = x.copy()
        y[i] = 1
        assert prob.evaluator.residual_risk(y) <= prob.evaluator.residual_risk(x) + 1e-12


@SETTINGS
@given(problems())
def test_qubo_matches_objective_shape_and_finite(prob):
    q = prob.qubo()
    assert q.n_decision == prob.n and q.n >= prob.n
    assert np.all(np.isfinite(q.linear)) and np.all(np.isfinite(q.quad))
    assert np.allclose(np.tril(q.quad), 0)  # strictly upper-triangular


@SETTINGS
@given(problems())
def test_exhaustive_optimum_is_feasible_and_minimal(prob):
    from qadapt.classical_baselines import ExhaustiveSolver, GreedySolver
    opt = ExhaustiveSolver().solve(prob)
    assert opt.feasible  # the empty portfolio is always feasible here
    g = GreedySolver().solve(prob)
    assert g.objective >= opt.objective - 1e-9


@SETTINGS
@given(problems())
def test_feasibility_batch_consistent(prob):
    X = all_bitstrings(prob.n)
    assert list(prob.feasible_batch(X)) == [prob.feasible(x) for x in X]
