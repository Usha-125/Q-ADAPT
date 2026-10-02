"""Scripted multi-stage attack scenario from the proposal (Experiment 6)."""

from __future__ import annotations

from qadapt.core.models import ThreatAssessment

# T1: web server attacked; T2: application server compromised; T3: database targeted
STAGED_DEMO = [
    {"t": 1, "label": "Attack detected on Web Server",
     "threats": [ThreatAssessment("WEB-01", "WEB_ATTACK", 0.94, 0.91, 120, "203.0.113.7")]},
    {"t": 2, "label": "Attacker moved to Application Server",
     "threats": [ThreatAssessment("APP-01", "EXPLOIT", 0.91, 0.9, 64, "10.0.2.11")],
     "compromised": ["WEB-01"]},
    {"t": 3, "label": "Database becomes the target",
     "threats": [ThreatAssessment("DB-01", "INFILTRATION", 0.88, 0.86, 41, "10.0.3.14")],
     "compromised": ["APP-01"]},
]
