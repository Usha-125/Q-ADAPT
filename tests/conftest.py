"""Shared fixtures for unit, integration and property tests."""

from __future__ import annotations

import warnings

import pytest

from qadapt.attack_graph import AttackGraph, demo_topology
from qadapt.core.config import OptimizationConfig
from qadapt.core.models import ThreatAssessment
from qadapt.defense_engine import ActionGenerator, DefenseEvaluator, prescreen
from qadapt.optimization import DefenseProblem

warnings.filterwarnings("ignore", category=DeprecationWarning)

FAST_QAOA = {"p": 1, "restarts": 1, "maxiter": 40, "shots": 256}

WEB_THREAT = ThreatAssessment("WEB-01", "WEB_ATTACK", 0.93, 0.9, 50, "203.0.113.7")
APP_THREAT = ThreatAssessment("APP-01", "EXPLOIT", 0.7, 0.85, 20)


@pytest.fixture
def demo_graph() -> AttackGraph:
    ag = AttackGraph(demo_topology())
    ag.apply_threats([WEB_THREAT, APP_THREAT])
    return ag


@pytest.fixture
def small_evaluator(demo_graph) -> DefenseEvaluator:
    acts = ActionGenerator(demo_graph).generate()
    ev = DefenseEvaluator(demo_graph, acts)
    return DefenseEvaluator(demo_graph, [acts[i] for i in prescreen(ev, 8)])


@pytest.fixture
def make_problem(small_evaluator):
    def _make(**cfg) -> DefenseProblem:
        return DefenseProblem(small_evaluator, OptimizationConfig(**cfg))
    return _make
