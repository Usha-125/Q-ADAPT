import numpy as np
import pytest

from qadapt.attack_graph import AttackGraph, demo_topology
from qadapt.core.models import ActionType, ThreatAssessment
from qadapt.defense_engine import ActionGenerator, DefenseEvaluator, build_policy, prescreen


@pytest.fixture
def setup():
    ag = AttackGraph(demo_topology())
    ag.apply_threats([ThreatAssessment("WEB-01", "WEB_ATTACK", 0.93, 0.9, source="203.0.113.7"),
                      ThreatAssessment("APP-01", "EXPLOIT", 0.7, 0.85)])
    actions = ActionGenerator(ag).generate()
    return ag, actions, DefenseEvaluator(ag, actions)


def test_generator_covers_threatened_hosts(setup):
    _, actions, _ = setup
    ids = {a.id for a in actions}
    assert "isolate_host:APP-01" in ids
    assert any(a.type == ActionType.BLOCK_IP for a in actions)
    assert all(0 <= a.disruption <= 1 for a in actions)


def test_empty_portfolio_equals_base_risk(setup):
    _, actions, ev = setup
    assert ev.residual_risk(np.zeros(len(actions))) == pytest.approx(ev.base_risk)


def test_every_action_reduces_risk_or_is_neutral(setup):
    _, _, ev = setup
    assert np.all(ev.marginal_reductions() >= -1e-12)


def test_more_actions_never_increase_risk(setup):
    _, actions, ev = setup
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 2, len(actions))
        y = x.copy()
        y[rng.integers(len(actions))] = 1
        assert ev.residual_risk(y) <= ev.residual_risk(x) + 1e-12


def test_batch_matches_single(setup):
    _, actions, ev = setup
    X = np.random.default_rng(1).integers(0, 2, (8, len(actions)))
    batched = ev.residual_risk(X)
    for b in range(8):
        assert batched[b] == pytest.approx(ev.residual_risk(X[b]))


def test_policy_conflicts_and_protected(setup):
    _, actions, _ = setup
    pol = build_policy(actions, protected_assets=("APP-01",))
    iso = next(i for i, a in enumerate(actions) if a.id == "isolate_host:APP-01")
    assert iso in pol.forbidden
    x = np.zeros(len(actions))
    x[iso] = 1
    assert pol.violations(x)


def test_prescreen_returns_k(setup):
    _, _, ev = setup
    idx = prescreen(ev, 8)
    assert len(idx) == 8 and len(set(idx)) == 8
