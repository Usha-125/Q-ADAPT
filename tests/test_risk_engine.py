import pytest

from qadapt.attack_graph import AttackGraph, demo_topology
from qadapt.core.models import ThreatAssessment
from qadapt.risk_engine import (
    RISK_MODELS,
    RiskModel,
    downstream_exposure,
    structural_importance,
    threat_score,
)


@pytest.fixture
def ag():
    g = AttackGraph(demo_topology())
    g.apply_threats([ThreatAssessment("APP-01", "EXPLOIT", 0.92, 0.9)])
    return g


@pytest.mark.parametrize("model", RISK_MODELS)
def test_risk_models_bounded(ag, model):
    rep = RiskModel(model).report(ag)
    assert 0.0 <= rep.total_risk <= 1.0
    assert all(0.0 <= a.risk <= 1.0 for a in rep.assets)


def test_propagation_model_ranks_attacked_path_high(ag):
    top = [a.asset_id for a in RiskModel().report(ag).top(6)]
    assert "APP-01" in top


def test_severity_only_ignores_graph(ag):
    rep = RiskModel("severity_only").report(ag)
    nonzero = [a.asset_id for a in rep.assets if a.risk > 0]
    assert nonzero == ["APP-01"]


def test_downstream_exposure_ordering(ag):
    assert downstream_exposure(ag, "WEB-01") > downstream_exposure(ag, "ADMIN")


def test_structural_importance_normalised(ag):
    imp = structural_importance(ag)
    assert max(imp.values()) == pytest.approx(1.0)


def test_threat_score_category_weighting():
    a = ThreatAssessment("X", "INFILTRATION", 0.9, 0.9)
    b = ThreatAssessment("X", "RECON", 0.9, 0.9)
    assert threat_score(a) > threat_score(b)
