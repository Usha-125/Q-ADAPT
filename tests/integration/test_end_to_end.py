"""End-to-end integration: ML -> graph -> risk -> QUBO -> solver -> defense -> feedback."""

import json

import numpy as np
import pytest

from qadapt.adaptive_engine import POLICIES, STAGED_DEMO, AQDOController, run_episode
from qadapt.attack_graph import AttackGraph, demo_topology, generate_topology, propagate, total_risk
from qadapt.core.config import OptimizationConfig
from qadapt.ml_engine import ThreatDetector, generate_flows
from qadapt.pipeline import SOLVERS, QAdaptPipeline

FAST_QAOA = {"p": 1, "restarts": 1, "maxiter": 40, "shots": 256}


@pytest.fixture(scope="module")
def detector():
    return ThreatDetector.train(generate_flows(3000, seed=0), "hist_gradient_boosting")


def test_ml_to_defense_full_chain(detector):
    """Live traffic is classified, mapped onto assets, and a defense is chosen and applied."""
    topo = demo_topology()
    ag = AttackGraph(topo)
    flows = generate_flows(800, attack_fraction=0.2, hosts=[a.ip for a in topo.assets.values()],
                           attacked_hosts={topo.assets["APP-01"].ip: "EXPLOIT"}, seed=9)
    threats = detector.assess_hosts(flows, topo.ip_map())
    assert threats[0].host_id == "APP-01"
    pipe = QAdaptPipeline(ag, OptimizationConfig(budget=0.4), max_qubits=10)
    pipe.ingest(threats)
    rep = pipe.decide("qaoa", **FAST_QAOA)
    assert rep.result.feasible and rep.selected_actions
    ag.apply_defenses(rep.selected_actions)
    cg = ag.compile()
    assert total_risk(cg, propagate(cg)) == pytest.approx(rep.risk_after, abs=1e-9)
    assert rep.risk_after < rep.risk_before
    json.dumps(rep.to_dict(), default=float)


@pytest.mark.parametrize("solver", SOLVERS)
def test_every_solver_in_the_pipeline(solver):
    ag = AttackGraph(demo_topology())
    pipe = QAdaptPipeline(ag, OptimizationConfig(budget=0.4, max_actions=3), max_qubits=10)
    pipe.ingest(STAGED_DEMO[0]["threats"])
    kw = FAST_QAOA if solver == "qaoa" else {}
    rep = pipe.decide(solver, **kw)
    assert rep.result.feasible and rep.risk_after <= rep.risk_before + 1e-12
    assert rep.paths_after <= rep.paths_before


@pytest.mark.parametrize("risk_model", ["propagation", "multiplicative", "severity_only"])
@pytest.mark.parametrize("focus", ["risk", "threatened"])
def test_pipeline_configurations(risk_model, focus):
    ag = AttackGraph(demo_topology())
    pipe = QAdaptPipeline(ag, OptimizationConfig(budget=0.4), risk_model=risk_model, max_qubits=8,
                          candidate_focus=focus)
    pipe.ingest(STAGED_DEMO[1]["threats"])
    rep = pipe.decide("greedy")
    assert rep.result.feasible and rep.n_candidates <= 8


def test_pipeline_rejects_bad_focus():
    with pytest.raises(ValueError):
        QAdaptPipeline(AttackGraph(demo_topology()), candidate_focus="vibes")


def test_protected_assets_are_never_isolated():
    ag = AttackGraph(demo_topology())
    pipe = QAdaptPipeline(ag, OptimizationConfig(budget=1.0, protected_assets=("APP-01", "WEB-01"),
                                                 weights=__import__("qadapt").core.ObjectiveWeights(alpha=5, delta=0)),
                          max_qubits=12)
    pipe.ingest(STAGED_DEMO[0]["threats"] + STAGED_DEMO[1]["threats"])
    for solver in ("exhaustive", "milp", "simulated_annealing"):
        rep = pipe.decide(solver)
        assert not any(a.target in ("APP-01", "WEB-01") and a.type.value in ("isolate_host", "block_port")
                       for a in rep.selected_actions), solver


def test_large_network_pipeline_stays_tractable():
    ag = AttackGraph(generate_topology(300, seed=4))
    web = next(a for a in ag.assets if a.startswith("WEB"))
    pipe = QAdaptPipeline(ag, OptimizationConfig(budget=0.4), max_qubits=12)
    pipe.ingest([STAGED_DEMO[0]["threats"][0].__class__(web, "WEB_ATTACK", 0.9, 0.9, 1, "x")])
    rep = pipe.decide("qaoa", **FAST_QAOA)
    assert rep.n_candidates <= 12 and rep.result.feasible


def test_staged_scenario_adapts_defense():
    ag = AttackGraph(demo_topology())
    applied, chosen = [], []
    for st in STAGED_DEMO:
        for h in st.get("compromised", []):
            ag.mark_compromised(h)
        ag.apply_threats(st["threats"])
        ag.apply_defenses(applied)
        rep = QAdaptPipeline(ag, OptimizationConfig(budget=0.4, max_actions=3), max_qubits=10).decide("exhaustive")
        applied += rep.selected_actions
        chosen.append({a.id for a in rep.selected_actions})
    assert chosen[0] != chosen[1]  # the plan changes as the attack moves
    assert not chosen[0] & chosen[1]  # applied actions are not re-recommended


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_runs(policy):
    res = run_episode(demo_topology(), policy, 3, OptimizationConfig(budget=0.4, max_actions=2),
                      solver="greedy", seed=1, max_qubits=8)
    assert len(res["history"]) == 3 and 0 <= res["final_loss"] <= 1
    if policy == "none":
        assert res["n_actions"] == 0
    if policy == "static":
        assert sum(h["reoptimized"] for h in res["history"]) <= 1


def test_episode_reproducible():
    a = run_episode(demo_topology(), "aqdo", 3, solver="greedy", seed=5, max_qubits=8)
    b = run_episode(demo_topology(), "aqdo", 3, solver="greedy", seed=5, max_qubits=8)
    assert a["compromised"] == b["compromised"] and a["cumulative_loss"] == b["cumulative_loss"]


def test_adaptive_policy_with_real_ml_detector(detector):
    res = run_episode(generate_topology(25, seed=2), "aqdo", 3, OptimizationConfig(budget=0.4, max_actions=2),
                      solver="greedy", seed=0, detector=detector, max_qubits=8)
    assert len(res["history"]) == 3


def test_learning_moves_effectiveness_with_evidence():
    from qadapt.adaptive_engine import AttackEnvironment
    from qadapt.core.models import ActionType
    topo = demo_topology()
    ctl = AQDOController(topo, policy="aqdo", solver="greedy")
    env = AttackEnvironment(topo, {ActionType.BLOCK_IP: 0.0}, seed=0, speed=1.0)
    block = [a for a in __import__("qadapt").defense_engine.ActionGenerator(ctl.ag).generate()
             if a.type == ActionType.BLOCK_IP]
    ctl.commit(block)
    env.apply(block)
    before = ctl.learner.estimate(ActionType.BLOCK_IP)
    for _ in range(4):
        tick = env.step()
        ctl.observe(tick.observations, tick.outcomes)
    assert ctl.learner.estimate(ActionType.BLOCK_IP) < before


def test_controller_rejects_unknown_policy():
    with pytest.raises(ValueError):
        AQDOController(demo_topology(), policy="yolo")


def test_human_rejection_applies_nothing():
    ctl = AQDOController(demo_topology(), OptimizationConfig(budget=0.4), solver="greedy",
                         approve=lambda report: [])
    ctl.observe(STAGED_DEMO[0]["threats"])
    report, _ = ctl.decide()
    ctl.commit(ctl.approve(report))
    assert report.selected_actions and ctl.applied == []


def test_benchmark_instances_are_deterministic():
    from qadapt.benchmarks import compare_solvers, make_instance
    a, b = make_instance(8, 3), make_instance(8, 3)
    assert [x.id for x in a.actions] == [x.id for x in b.actions]
    np.testing.assert_allclose(a.evaluator.cost, b.evaluator.cost)
    rows = compare_solvers([6], 1, ["greedy", "milp"])
    assert {r["solver"] for r in rows} == {"greedy", "milp"} and all(r["feasible_rate"] == 1 for r in rows)
