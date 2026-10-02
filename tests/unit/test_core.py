import pytest

from qadapt.core.config import ObjectiveWeights, OptimizationConfig
from qadapt.core.models import (
    ActionType,
    Asset,
    AssetType,
    DefenseAction,
    Severity,
    ThreatAssessment,
    Vulnerability,
)


@pytest.mark.parametrize("score,level", [
    (0.0, Severity.LOW), (0.299, Severity.LOW), (0.3, Severity.MEDIUM), (0.599, Severity.MEDIUM),
    (0.6, Severity.HIGH), (0.849, Severity.HIGH), (0.85, Severity.CRITICAL), (1.0, Severity.CRITICAL),
])
def test_severity_boundaries(score, level):
    assert Severity.from_score(score) is level


def test_asset_exploitability_is_noisy_or():
    a = Asset("X", "x", AssetType.WEB_SERVER, 0.5, vulnerabilities=[
        Vulnerability("A", 9.0, 0.5), Vulnerability("B", 7.0, 0.5)])
    assert a.exploitability == pytest.approx(0.75)
    assert a.max_cvss == 9.0


def test_asset_without_vulnerabilities():
    a = Asset("X", "x", AssetType.WORKSTATION, 0.2)
    assert a.exploitability == 0.0 and a.max_cvss == 0.0


def test_vulnerability_severity_normalised():
    assert Vulnerability("A", 7.5, 0.3).severity == pytest.approx(0.75)


def test_threat_assessment_severity_and_dict():
    t = ThreatAssessment("H", "DDOS", 0.95, 0.9, 3, "1.2.3.4")
    assert t.severity is Severity.CRITICAL
    d = t.to_dict()
    assert d["severity"] == "CRITICAL" and d["host_id"] == "H" and d["n_events"] == 3


def test_defense_action_to_dict_rounds():
    a = DefenseAction("i", ActionType.BLOCK_IP, "t", "d", 0.123456, 0.1, 0.2, 0.9)
    assert a.to_dict()["cost"] == 0.1235 and a.to_dict()["type"] == "block_ip"


def test_weights_as_dict():
    assert ObjectiveWeights().as_dict().keys() == {"alpha", "beta", "gamma", "delta"}


@pytest.mark.parametrize("field", ["budget", "max_time", "max_disruption", "max_actions"])
def test_config_rejects_negative_limits(field):
    with pytest.raises(ValueError, match=field):
        OptimizationConfig(**{field: -1})


def test_config_rejects_bad_encoding_and_penalty():
    with pytest.raises(ValueError):
        OptimizationConfig(constraint_encoding="magic")
    with pytest.raises(ValueError):
        OptimizationConfig(penalty=0)
    with pytest.raises(ValueError):
        OptimizationConfig(slack_resolution=0)


def test_config_accepts_zero_limits():
    cfg = OptimizationConfig(budget=0, max_actions=0)
    assert cfg.budget == 0 and cfg.max_actions == 0
