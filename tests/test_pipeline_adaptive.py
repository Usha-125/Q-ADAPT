import numpy as np
import pytest

from qadapt.adaptive_engine import (
    STAGED_DEMO,
    AdaptiveWeights,
    AQDOController,
    AttackEnvironment,
    EffectivenessLearner,
    run_episode,
)
from qadapt.attack_graph import AttackGraph, demo_topology
from qadapt.core.config import OptimizationConfig
from qadapt.core.models import ActionType
from qadapt.pipeline import QAdaptPipeline

FAST_QAOA = dict(p=1, restarts=1, maxiter=40, shots=256)


def test_pipeline_decision_report():
    ag = AttackGraph(demo_topology())
    pipe = QAdaptPipeline(ag, OptimizationConfig(budget=0.4), max_qubits=10)
    pipe.ingest(STAGED_DEMO[0]["threats"] + STAGED_DEMO[1]["threats"])
    rep = pipe.decide("qaoa", **FAST_QAOA)
    d = rep.to_dict()
    assert rep.result.feasible
    assert rep.risk_after <= rep.risk_before
    assert d["selected"] and d["explanations"]
    assert any(not e["selected"] for e in d["explanations"])
    assert d["qubo"]["n_decision"] == rep.n_candidates <= 10
    assert rep.paths_after <= rep.paths_before


@pytest.mark.parametrize("solver", ["greedy", "milp", "exhaustive"])
def test_pipeline_classical_solvers(solver):
    ag = AttackGraph(demo_topology())
    pipe = QAdaptPipeline(ag, OptimizationConfig(budget=0.3), max_qubits=10)
    pipe.ingest(STAGED_DEMO[0]["threats"])
    assert pipe.decide(solver).result.feasible


def test_learner_moves_towards_observations():
    lr = EffectivenessLearner()
    before = lr.estimate(ActionType.BLOCK_IP)
    for _ in range(20):
        lr.update(ActionType.BLOCK_IP, blocked=False)
    assert lr.estimate(ActionType.BLOCK_IP) < before - 0.3


def test_adaptive_weights_escalate_with_risk():
    aw = AdaptiveWeights()
    w0 = aw.weights(0.3)
    w1 = aw.weights(0.6)
    assert w1.alpha > w0.alpha and w1.delta == w0.delta


def test_environment_defenses_slow_attacker():
    topo = demo_topology()
    env_open = AttackEnvironment(topo, seed=1, speed=1.0)
    env_def = AttackEnvironment(topo, seed=1, speed=1.0)
    ctl = AQDOController(topo)
    from qadapt.defense_engine import ActionGenerator
    acts = ActionGenerator(ctl.ag).generate(focus=[])
    block = [a for a in acts if a.type == ActionType.BLOCK_IP]
    env_def.apply(block)
    env_def.true_eff[ActionType.BLOCK_IP] = 1.0
    env_def.apply([])
    for _ in range(3):
        env_open.step()
        env_def.step()
    assert len(env_def.compromised) <= len(env_open.compromised)


def test_adaptive_policy_beats_no_defense():
    cfg = OptimizationConfig(budget=0.4, max_actions=3)
    none = [run_episode(demo_topology(), "none", 4, cfg, seed=s)["cumulative_loss"] for s in range(3)]
    aqdo = [run_episode(demo_topology(), "aqdo", 4, cfg, solver="greedy", seed=s)["cumulative_loss"]
            for s in range(3)]
    assert np.mean(aqdo) < np.mean(none)


def test_episode_with_qaoa_and_warm_start():
    res = run_episode(demo_topology(), "aqdo", 3, OptimizationConfig(budget=0.4, max_actions=3),
                      solver_kw=FAST_QAOA, max_qubits=8, seed=0)
    assert len(res["history"]) == 3
    assert res["n_actions"] >= 1
