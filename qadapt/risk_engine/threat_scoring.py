"""Threat scoring: ML output -> calibrated per-host threat score."""

from __future__ import annotations

from qadapt.core.models import Severity, ThreatAssessment

# Relative impact of an attack category on host integrity (used to weight ML
# evidence: a confirmed infiltration matters more than a port scan).
CATEGORY_IMPACT: dict[str, float] = {
    "INFILTRATION": 1.0,
    "EXPLOIT": 0.95,
    "BOTNET": 0.9,
    "WEB_ATTACK": 0.85,
    "BRUTE_FORCE": 0.75,
    "DDOS": 0.6,
    "DOS": 0.55,
    "RECON": 0.35,
    "ANOMALY": 0.5,
}


def threat_score(t: ThreatAssessment) -> float:
    return float(t.probability * t.confidence * CATEGORY_IMPACT.get(t.attack_type, 0.6))


def weight_threats(threats: list[ThreatAssessment]) -> list[ThreatAssessment]:
    """Return threats whose probability is scaled by category impact."""
    return [ThreatAssessment(t.host_id, t.attack_type,
                             t.probability * CATEGORY_IMPACT.get(t.attack_type, 0.6),
                             t.confidence, t.n_events, t.source) for t in threats]


def severity(score: float) -> Severity:
    return Severity.from_score(score)
