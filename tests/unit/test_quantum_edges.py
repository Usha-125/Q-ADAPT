import numpy as np
import pytest

from qadapt.quantum_engine import (
    NOISE_LEVELS,
    QUBO,
    QAOASolver,
    StatevectorQAOA,
    build_defense_qubo,
    get_noise,
    local_search,
    qubo_to_ising,
)
from qadapt.quantum_engine.ising import all_bitstrings, bits_to_spins, diagonal_energies
from qadapt.quantum_engine.noise_models import circuit_fidelity
from qadapt.quantum_engine.qaoa import _interp, gate_counts, linear_ramp, transpiled_depth

FAST = {"restarts": 1, "maxiter": 40, "shots": 256}


def _brute(q):
    X = all_bitstrings(q.n)
    e = q.energy(X)
    return X[int(np.argmin(e))]


# ---- QUBO container -------------------------------------------------------------------
def test_add_square_matches_expansion():
    q = QUBO(np.zeros(3), np.zeros((3, 3)))
    q.add_square({0: 1.0, 1: -2.0, 2: 0.5}, const=-0.7, weight=2.0)
    X = all_bitstrings(3).astype(float)
    expected = 2.0 * (X @ np.array([1.0, -2.0, 0.5]) - 0.7) ** 2
    np.testing.assert_allclose(q.energy(X), expected, atol=1e-12)


def test_add_quad_diagonal_goes_linear_and_order_insensitive():
    q = QUBO(np.zeros(2), np.zeros((2, 2)))
    q.add_quad(1, 0, 3.0)
    q.add_quad(1, 1, 2.0)
    assert q.quad[0, 1] == 3.0 and q.quad[1, 0] == 0.0 and q.linear[1] == 2.0


def test_matrix_form_matches_energy():
    rng = np.random.default_rng(0)
    q = QUBO(rng.normal(size=4), np.triu(rng.normal(size=(4, 4)), 1), 0.3)
    Q = q.matrix()
    for x in all_bitstrings(4).astype(float):
        assert x @ Q @ x + q.offset == pytest.approx(q.energy(x))


def test_brute_force_limit():
    with pytest.raises(ValueError):
        QUBO(np.zeros(27), np.zeros((27, 27))).brute_force()


def test_diagonal_energy_limit_and_order():
    q = QUBO(np.array([1.0, 2.0, 4.0]), np.zeros((3, 3)))
    np.testing.assert_allclose(diagonal_energies(q), np.arange(8))  # index k <-> bits of k
    with pytest.raises(ValueError):
        diagonal_energies(QUBO(np.zeros(25), np.zeros((25, 25))))


def test_spin_mapping():
    np.testing.assert_array_equal(bits_to_spins(np.array([0, 1])), [1, -1])


# ---- defense QUBO -----------------------------------------------------------------------
def test_invalid_surrogate_and_encoding(make_problem):
    prob = make_problem()
    with pytest.raises(ValueError):
        build_defense_qubo(prob, surrogate="cubic")
    with pytest.raises(ValueError):
        build_defense_qubo(prob, encoding="magic")


@pytest.mark.parametrize("encoding", ["unbalanced", "slack"])
@pytest.mark.parametrize("field", ["budget", "max_time", "max_disruption", "max_actions"])
def test_zero_limits_are_exact_and_well_scaled(make_problem, encoding, field):
    prob = make_problem(**{field: 0}, constraint_encoding=encoding)
    q = prob.qubo()
    assert np.abs(q.quad).max() < 10 and np.abs(q.linear).max() < 10  # no 1/lim blow-up
    assert q.n_slack == 0
    x = _brute(q)[: q.n_decision]
    assert prob.feasible(x)


def test_slack_adds_qubits_only_for_binding_constraints(make_problem):
    prob = make_problem(budget=1e6, max_actions=2, constraint_encoding="slack", slack_resolution=3)
    q = prob.qubo()
    assert q.n_slack == 3 and q.n == prob.n + 3
    assert all(name.startswith("slack_cardinality") for name in q.var_names[prob.n:])


def test_unbalanced_adds_no_qubits(make_problem):
    q = make_problem(budget=0.2, max_time=0.2, max_disruption=0.2, max_actions=2).qubo()
    assert q.n_slack == 0 and q.meta["n_constraints"] == 4


@pytest.mark.parametrize("kind", ["prerequisite", "mandatory", "forbidden", "conflict"])
def test_policy_rules_hold_at_qubo_minimum(make_problem, kind):
    prob = make_problem()
    # make the rule bind: force the otherwise-preferred choice to violate it
    if kind == "prerequisite":
        prob.policy.prerequisites.add((0, prob.n - 1))
    elif kind == "mandatory":
        prob.policy.mandatory.add(prob.n - 1)
    elif kind == "forbidden":
        prob.policy.forbidden.add(0)
    else:
        prob.policy.conflicts.add((0, 1))
    q = build_defense_qubo(prob)
    x = _brute(q)[: q.n_decision]
    assert not prob.policy.violations(x), prob.policy.violations(x)


def test_explicit_penalty_is_used(make_problem):
    assert make_problem(penalty=42.0).qubo().meta["penalty"] == 42.0


def test_qubo_cache_and_rebuild(make_problem):
    prob = make_problem()
    q1 = prob.qubo()
    assert prob.qubo() is q1
    assert prob.qubo(surrogate="expansion") is not q1


def test_single_action_problem(small_evaluator):
    from qadapt.core.config import OptimizationConfig
    from qadapt.defense_engine import DefenseEvaluator
    from qadapt.optimization import DefenseProblem
    ev = DefenseEvaluator(small_evaluator.ag, small_evaluator.actions[:1])
    prob = DefenseProblem(ev, OptimizationConfig())
    q = prob.qubo()
    assert q.n == 1 and "fidelity" not in q.meta
    r = QAOASolver(p=1, **FAST).solve(prob)
    assert r.feasible and r.x.shape == (1,)


# ---- noise ---------------------------------------------------------------------------------
def test_noise_levels():
    assert get_noise(None).is_ideal and get_noise("high") is NOISE_LEVELS["high"]
    assert get_noise(NOISE_LEVELS["low"]) is NOISE_LEVELS["low"]
    with pytest.raises(KeyError):
        get_noise("extreme")
    assert circuit_fidelity(NOISE_LEVELS["ideal"], 100, 100) == 1.0
    assert 0 < circuit_fidelity(NOISE_LEVELS["high"], 100, 100) < circuit_fidelity(NOISE_LEVELS["low"], 100, 100)


# ---- QAOA engine -----------------------------------------------------------------------------
def test_statevector_qubit_limit():
    with pytest.raises(ValueError, match="state-vector limit"):
        StatevectorQAOA(QUBO(np.zeros(23), np.zeros((23, 23))), 1)


def test_probabilities_normalised_and_cvar_bounds(make_problem):
    q = make_problem(budget=0.3).qubo()
    theta = linear_ramp(2)
    eng = StatevectorQAOA(q, 2)
    probs = eng.probabilities(theta)
    assert probs.sum() == pytest.approx(1.0) and np.all(probs >= 0)
    full = eng.expectation(theta)
    eng_cvar = StatevectorQAOA(q, 2, cvar_alpha=0.1)
    assert eng_cvar.expectation(theta) <= full + 1e-12  # CVaR averages only the best tail
    assert 0.0 <= full <= 1.0


def test_noisy_engine_mixes_towards_uniform(make_problem):
    q = make_problem().qubo()
    theta = linear_ramp(1)
    ideal = StatevectorQAOA(q, 1).probabilities(theta)
    noisy = StatevectorQAOA(q, 1, noise="high").probabilities(theta)
    uniform = 1 / ideal.size
    assert np.abs(noisy - uniform).sum() < np.abs(ideal - uniform).sum()
    assert noisy.sum() == pytest.approx(1.0)


def test_sampling_and_top_states(make_problem):
    q = make_problem().qubo()
    eng = StatevectorQAOA(q, 1, noise="medium", seed=1)
    X = eng.sample(linear_ramp(1), 50)
    assert X.shape == (50, q.n) and set(np.unique(X)) <= {0, 1}
    top, p = eng.top_states(linear_ramp(1), 5)
    assert top.shape == (5, q.n) and np.all(np.diff(p) <= 1e-15)


def test_interp_extends_parameters():
    theta = np.array([0.2, 0.6, 0.5, 0.1])  # p = 2
    ext = _interp(theta, 2)
    assert ext.shape == (6,)
    np.testing.assert_allclose(ext[:3], [0.2, 0.4, 0.6])  # gamma: endpoints kept, midpoint averaged


def test_gate_counts_and_depth(make_problem):
    q = make_problem().qubo()
    g1, g2 = gate_counts(q, 1), gate_counts(q, 2)
    assert g2["n_2q"] == 2 * g1["n_2q"] and g1["depth_scheduled"] <= g1["depth_upper_bound"]
    td = transpiled_depth(q, 1)
    if td is not None:
        assert td["depth"] > 0 and td["cx"] >= 0


def test_qaoa_no_shots_uses_top_states(make_problem):
    r = QAOASolver(p=1, restarts=1, maxiter=30, shots=None, top_k=16).solve(make_problem(budget=0.3))
    assert r.feasible and r.info["unique_samples"] <= 16


def test_qaoa_cvar_and_polish(make_problem):
    prob = make_problem(budget=0.3)
    r = QAOASolver(p=1, cvar_alpha=0.2, polish=True, **FAST).solve(prob)
    assert r.feasible and r.info["polished_gain"] >= 0
    assert local_search(prob, r.x).tolist() == r.x.tolist()  # polish result is a local optimum


def test_qaoa_random_init_and_bad_warm_start(make_problem):
    prob = make_problem()
    r = QAOASolver(p=2, init="random", restarts=2, maxiter=30, warm_start=np.zeros(3)).solve(prob)
    assert r.feasible and len(r.info["params"]) == 4


def test_qaoa_unknown_backend(make_problem):
    with pytest.raises(ValueError):
        QAOASolver(backend="photonic").solve(make_problem())


def test_qaoa_is_reproducible(make_problem):
    prob = make_problem(budget=0.3)
    a = QAOASolver(p=1, seed=3, **FAST).solve(prob)
    b = QAOASolver(p=1, seed=3, **FAST).solve(prob)
    assert a.x.tolist() == b.x.tolist() and a.info["params"] == b.info["params"]


def test_qaoa_solver_name_reflects_noise():
    assert QAOASolver(p=2).name == "qaoa_p2"
    assert QAOASolver(p=1, noise="high").name == "qaoa_p1_high"


def test_ising_offset_on_empty_portfolio(make_problem):
    q = make_problem(budget=0.3).qubo()
    isg = qubo_to_ising(q)
    assert isg.energy_z(np.ones(q.n)) == pytest.approx(q.energy(np.zeros(q.n)))
