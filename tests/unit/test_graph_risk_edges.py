import numpy as np
import pytest

from qadapt.attack_graph import (
    ATTACKER,
    AttackGraph,
    count_viable_paths,
    demo_topology,
    generate_topology,
    propagate,
    scenario_topology,
    top_attack_paths,
    total_risk,
)
from qadapt.attack_graph.graph_builder import P_MAX, P_MIN, edge_probability
from qadapt.attack_graph.topology import SCENARIO_SIZES, Topology
from qadapt.attack_graph.visualizer import to_cytoscape
from qadapt.core.models import Asset, AssetType, ThreatAssessment, Vulnerability
from qadapt.risk_engine import RiskModel, blended_criticality, weight_threats
from qadapt.risk_engine.asset_criticality import apply_criticality
from qadapt.risk_engine.threat_scoring import severity


def _chain(n=3, p=0.5, activity=1.0, cycle=False):
    ids = [f"N{i}" for i in range(n)]
    assets = {k: Asset(k, k, AssetType.APP_SERVER, 1.0) for k in ids}
    links = [(a, b, "x") for a, b in zip(ids, ids[1:], strict=False)]
    if cycle:
        links.append((ids[-1], ids[0], "x"))
    ag = AttackGraph(Topology("t", assets, links, [ids[0]]), attacker_activity=activity)
    for u, v in ag.g.edges:
        ag.g[u][v]["p"] = ag.g[u][v]["base_p"] = p
    return ag, ids


@pytest.mark.parametrize("kind", ["entry", "network", "credential", "lateral", "phishing", "other"])
def test_edge_probability_is_clipped(kind):
    hot = Asset("v", "v", AssetType.WEB_SERVER, 1, vulnerabilities=[Vulnerability("x", 10, 1.0, True)])
    cold = Asset("v", "v", AssetType.WEB_SERVER, 1)
    for src in (hot, None):
        for dst in (hot, cold):
            assert P_MIN <= edge_probability(kind, src, dst) <= P_MAX


def test_propagation_handles_cycles_and_converges():
    ag, ids = _chain(4, p=0.9, cycle=True)
    P = propagate(ag.compile())
    assert np.all((P >= 0) & (P <= 1))
    cg = ag.compile()
    # with a cycle every node gets extra paths, so P(N1) exceeds the acyclic value 0.9*0.9
    assert P[cg.index["N1"]] > 0.81 - 1e-9


def test_zero_attacker_activity_and_no_evidence_means_zero_risk():
    ag, _ = _chain(3, activity=0.0)
    cg = ag.compile()
    assert total_risk(cg, propagate(cg)) == pytest.approx(0.0)


def test_local_evidence_without_attacker_still_propagates():
    ag, ids = _chain(3, p=0.5, activity=0.0)
    ag.base_threat["N0"] = 1.0
    ag._refresh_threat()
    cg = ag.compile()
    P = propagate(cg)
    assert P[cg.index["N0"]] == pytest.approx(1.0) and P[cg.index["N2"]] == pytest.approx(0.25)


def test_propagation_monotone_in_edge_multipliers(demo_graph):
    cg = demo_graph.compile()
    rng = np.random.default_rng(0)
    em = rng.uniform(0, 1, cg.n_edges)
    assert np.all(propagate(cg, em * 0.5) <= propagate(cg, em) + 1e-12)


def test_node_and_edge_multiplier_batches_broadcast(demo_graph):
    cg = demo_graph.compile()
    nm = np.ones((3, cg.n_nodes))
    nm[1, cg.index["WEB-01"]] = 0.0
    out = propagate(cg, None, nm)
    assert out.shape == (3, cg.n_nodes)
    assert out[1, cg.index["WEB-01"]] <= out[0, cg.index["WEB-01"]]


def test_apply_threats_ignores_unknown_hosts_and_keeps_max(demo_graph):
    demo_graph.apply_threats([ThreatAssessment("NOPE", "X", 1, 1),
                              ThreatAssessment("WEB-01", "X", 0.1, 0.1)])
    assert "NOPE" not in demo_graph.threat
    assert demo_graph.threat["WEB-01"] == pytest.approx(0.93 * 0.9)


def test_clear_threat_and_compromise(demo_graph):
    demo_graph.mark_compromised("DB-01")
    assert demo_graph.threat["DB-01"] == 1.0 and demo_graph.attacker_activity == 1.0
    demo_graph.clear_threat("DB-01")
    assert "DB-01" not in demo_graph.threat and "DB-01" not in demo_graph.compromised


def test_apply_defenses_is_idempotent_and_resettable(demo_graph):
    from qadapt.defense_engine import ActionGenerator
    acts = ActionGenerator(demo_graph).generate()[:5]
    demo_graph.apply_defenses(acts)
    once = {(u, v): d["p"] for u, v, d in demo_graph.g.edges(data=True)}
    demo_graph.apply_defenses(acts)
    assert once == {(u, v): d["p"] for u, v, d in demo_graph.g.edges(data=True)}
    demo_graph.apply_defenses([])
    assert all(d["p"] == d["base_p"] for *_, d in demo_graph.g.edges(data=True))
    assert demo_graph.node_mult == {}


def test_effectiveness_override_changes_hardening(demo_graph):
    from qadapt.core.models import ActionType
    from qadapt.defense_engine import ActionGenerator
    block = [a for a in ActionGenerator(demo_graph).generate() if a.type == ActionType.BLOCK_IP]
    demo_graph.apply_defenses(block, {ActionType.BLOCK_IP: 0.0})
    assert all(d["p"] == d["base_p"] for *_, d in demo_graph.g.edges(data=True))


def test_paths_with_unreachable_target_and_sources(demo_graph):
    assert top_attack_paths(demo_graph.g, ["DOES-NOT-EXIST"]) == []
    paths = top_attack_paths(demo_graph.g, ["ADMIN"], k=2, sources=["APP-01", "NOT-A-NODE"])
    assert paths and all(p.nodes[-1] == "ADMIN" for p in paths)


def test_cutting_all_edges_leaves_no_viable_paths(demo_graph):
    mult = {e: 0.0 for e in demo_graph.edges}
    assert count_viable_paths(demo_graph.g, ["ADMIN"], edge_mult=mult) == 0


def test_cytoscape_export_is_complete(demo_graph):
    cg = demo_graph.compile()
    out = to_cytoscape(demo_graph, cg, propagate(cg), ["ATTACKER", "WEB-01"], {("ATTACKER", "WEB-01"): 0.5})
    assert len(out["nodes"]) == cg.n_nodes and len(out["edges"]) == cg.n_edges
    edge = next(e for e in out["edges"] if e["data"]["id"] == "ATTACKER->WEB-01")
    assert edge["data"]["on_path"] and edge["data"]["defended"] == 0.5


def test_generate_topology_rejects_tiny_and_names_scenarios():
    with pytest.raises(ValueError):
        generate_topology(5)
    with pytest.raises(KeyError):
        scenario_topology("huge")
    for size, n in SCENARIO_SIZES.items():
        topo = scenario_topology(size, seed=1)
        assert abs(len(topo.assets) - n) <= 2 and topo.entry_points


def test_generate_topology_minimum_size_is_connected_from_attacker():
    import networkx as nx
    ag = AttackGraph(generate_topology(8, seed=0))
    assert set(ag.topology.entry_points) <= set(nx.descendants(ag.g, ATTACKER))


def test_generated_topology_is_deterministic():
    a, b = generate_topology(40, seed=7), generate_topology(40, seed=7)
    assert a.links == b.links and list(a.assets) == list(b.assets)


def test_ip_map_unique():
    topo = demo_topology()
    assert len(topo.ip_map()) == len(topo.assets)


# ---- risk engine -------------------------------------------------------------------
def test_unknown_risk_model():
    with pytest.raises(ValueError):
        RiskModel("bogus")


def test_risk_report_sorted_and_serialisable(demo_graph):
    rep = RiskModel().report(demo_graph)
    d = rep.to_dict()
    risks = [a["risk"] for a in d["assets"]]
    assert risks == sorted(risks, reverse=True) and ATTACKER not in [a["asset_id"] for a in d["assets"]]
    assert rep.top(3) == sorted(rep.assets, key=lambda a: -a.risk)[:3]


def test_batched_total_risk_matches_single(demo_graph):
    cg = demo_graph.compile()
    em = np.random.default_rng(0).uniform(0, 1, (4, cg.n_edges))
    for model in ("propagation", "multiplicative", "severity_only"):
        rm = RiskModel(model)
        batched = rm.total(cg, demo_graph, em)
        for b in range(4):
            assert batched[b] == pytest.approx(rm.total(cg, demo_graph, em[b]))


def test_blended_criticality_bounds_and_apply(demo_graph):
    crit = blended_criticality(demo_graph, 0.5)
    assert all(0 <= c <= 1 for c in crit.values())
    apply_criticality(demo_graph, crit)
    assert demo_graph.g.nodes["WEB-01"]["criticality"] == pytest.approx(crit["WEB-01"])


def test_weight_threats_scales_by_category():
    out = weight_threats([ThreatAssessment("h", "RECON", 1.0, 1.0), ThreatAssessment("h", "UNKNOWN", 1.0, 1.0)])
    assert out[0].probability == pytest.approx(0.35) and out[1].probability == pytest.approx(0.6)
    assert severity(0.9).value == "CRITICAL"
