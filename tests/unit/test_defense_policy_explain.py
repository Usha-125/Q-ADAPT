import numpy as np
import pytest

from qadapt.core.config import OptimizationConfig
from qadapt.core.models import ActionType, AssetType
from qadapt.defense_engine import (
    TEMPLATES,
    ActionGenerator,
    DefenseEvaluator,
    build_policy,
    prescreen,
)
from qadapt.defense_engine.policy_constraints import PolicySet
from qadapt.explainability import explain
from qadapt.optimization import DefenseProblem


def test_every_action_type_has_a_template():
    assert set(TEMPLATES) == set(ActionType)


def test_generated_actions_are_unique_and_bounded(demo_graph):
    acts = ActionGenerator(demo_graph).generate()
    assert len({a.id for a in acts}) == len(acts)
    for a in acts:
        assert 0 <= a.cost <= 1 and 0 <= a.time <= 1 and 0 <= a.disruption <= 1
        assert 0 < a.effectiveness <= 1 and (a.edge_effects or a.node_effects)
        assert all(e in demo_graph.g.edges for e in a.edge_effects)


def test_type_specific_actions(demo_graph):
    acts = {a.id: a for a in ActionGenerator(demo_graph).generate(focus=["PC-21", "DB-01", "FW-01"])}
    assert "quarantine_endpoint:PC-21" in acts and "disable_account:PC-21" in acts
    assert "protect_database:DB-01" in acts
    assert "patch_vulnerability:FW-01" not in acts  # firewall has no known CVEs
    assert not any(a.startswith("quarantine_endpoint:DB") for a in acts)


def test_generator_options(demo_graph):
    gen = ActionGenerator(demo_graph)
    acts = gen.generate(focus=["WEB-01"], include_segmentation=False, include_block_ip=False)
    assert all(a.type not in (ActionType.SEGMENT_NETWORK, ActionType.BLOCK_IP) for a in acts)
    assert gen.generate(focus=[], include_segmentation=False, include_block_ip=False) == []


def test_effectiveness_override(demo_graph):
    acts = ActionGenerator(demo_graph, {ActionType.ISOLATE_HOST: 0.1}).generate(focus=["WEB-01"])
    assert next(a for a in acts if a.type == ActionType.ISOLATE_HOST).effectiveness == 0.1


def test_disruption_scales_with_criticality(demo_graph):
    acts = {a.id: a for a in ActionGenerator(demo_graph).generate(focus=["PC-21", "DB-01"])}
    assert acts["isolate_host:DB-01"].disruption > acts["isolate_host:PC-21"].disruption
    assert demo_graph.assets["PC-21"].type == AssetType.WORKSTATION


def test_evaluator_with_no_actions(demo_graph):
    ev = DefenseEvaluator(demo_graph, [])
    assert ev.n == 0 and ev.residual_risk(np.zeros(0)) == pytest.approx(ev.base_risk)


def test_evaluator_metrics_and_cache(small_evaluator):
    x = np.zeros(small_evaluator.n, dtype=int)
    x[[0, 1]] = 1
    m = small_evaluator.metrics(x)
    assert m.n_actions == 2 and m.cost == pytest.approx(small_evaluator.cost[[0, 1]].sum())
    assert m.time_to_effect == pytest.approx(small_evaluator.time[[0, 1]].max())
    assert small_evaluator.residual_risk(x) == small_evaluator.residual_risk(x, cache=False)


def test_prescreen_k_larger_than_actions(small_evaluator):
    assert len(prescreen(small_evaluator, 100)) <= small_evaluator.n


def test_policy_violations_each_kind():
    pol = PolicySet(conflicts={(0, 1)}, prerequisites={(2, 3)}, forbidden={4}, mandatory={5})
    assert pol.violations([0, 0, 0, 0, 0, 1]) == []
    assert set(pol.violations([1, 1, 1, 0, 1, 0])) == {"conflict(0,1)", "requires(2,3)", "forbidden(4)",
                                                       "mandatory(5)"}
    assert pol.to_dict()["mandatory"] == [5]


def test_build_policy_extra_rules_by_id(small_evaluator):
    acts = small_evaluator.actions
    pol = build_policy(acts, extra_conflicts=[(acts[1].id, acts[0].id), ("nope", acts[0].id)],
                       extra_prerequisites=[(acts[2].id, acts[3].id)], mandatory=[acts[4].id, "nope"])
    assert (0, 1) in pol.conflicts and (2, 3) in pol.prerequisites and pol.mandatory == {4}


def test_problem_feasible_batch_matches_scalar(make_problem):
    prob = make_problem(budget=0.3, max_actions=2)
    prob.policy.prerequisites.add((0, 1))
    X = np.random.default_rng(0).integers(0, 2, (64, prob.n))
    assert list(prob.feasible_batch(X)) == [prob.feasible(x) for x in X]
    assert np.all((prob.violation_amount(X) > 0) == ~prob.feasible_batch(X))


def test_best_of_prefers_feasible(make_problem):
    prob = make_problem(max_actions=1)
    x, _, feas = prob.best_of(np.ones((1, prob.n), dtype=int))
    assert not feas  # only an infeasible candidate available
    x, _, feas = prob.best_of(np.vstack([np.ones(prob.n, dtype=int), np.zeros(prob.n, dtype=int)]))
    assert feas and x.sum() == 0


def test_objective_with_zero_base_risk(demo_graph):
    demo_graph.attacker_activity = 0.0
    demo_graph.base_threat.clear()
    demo_graph._refresh_threat()
    acts = ActionGenerator(demo_graph).generate(focus=["WEB-01"])
    prob = DefenseProblem(DefenseEvaluator(demo_graph, acts), OptimizationConfig())
    assert prob.evaluator.base_risk == 0.0
    assert np.isfinite(prob.objective(np.ones(prob.n, dtype=int)))


def test_explain_selected_and_rejected(make_problem):
    prob = make_problem(budget=0.3)
    prob.qubo()
    from qadapt.classical_baselines import ExhaustiveSolver
    x = ExhaustiveSolver().solve(prob).x
    ex = explain(prob, x, n_rejected=3)
    sel = [e for e in ex if e.selected]
    rej = [e for e in ex if not e.selected]
    assert len(sel) == x.sum() and 0 < len(rej) <= 3
    assert all(e.reasons for e in ex)
    assert all("portfolio_contribution" in e.factors for e in sel)
    assert all("objective_if_added" in e.factors for e in rej)


def test_explain_empty_plan_and_conflict_reason(make_problem):
    prob = make_problem()
    i, j = 0, 1
    prob.policy.conflicts.add((i, j))
    x = np.zeros(prob.n, dtype=int)
    x[j] = 1
    ex = explain(prob, x, n_rejected=prob.n)
    rej = next(e for e in ex if e.action_id == prob.actions[i].id)
    assert any("conflicts with" in r for r in rej.reasons)
    assert explain(prob, np.zeros(prob.n, dtype=int))  # no crash on empty plan
