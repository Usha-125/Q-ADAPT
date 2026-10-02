import numpy as np
import pytest

from qadapt.attack_graph import (
    ATTACKER,
    AttackGraph,
    critical_targets,
    demo_topology,
    generate_topology,
    propagate,
    top_attack_paths,
    total_risk,
)
from qadapt.attack_graph.vulnerabilities import parse_nvd_record
from qadapt.core.models import ThreatAssessment


@pytest.fixture
def ag():
    return AttackGraph(demo_topology())


def test_demo_graph_structure(ag):
    assert ag.g.has_edge(ATTACKER, "WEB-01")
    assert ag.g.has_edge("APP-01", "DB-01")
    for _, _, d in ag.g.edges(data=True):
        assert 0.0 < d["p"] < 1.0


def test_propagation_matches_closed_form_on_chain():
    from qadapt.attack_graph.topology import Topology
    from qadapt.core.models import Asset, AssetType
    assets = {k: Asset(k, k, AssetType.APP_SERVER, 1.0) for k in "ABC"}
    ag = AttackGraph(Topology("chain", assets, [("A", "B", "x"), ("B", "C", "x")], ["A"]))
    for u, v in ag.g.edges:
        ag.g[u][v]["p"] = 0.5
    P = propagate(ag.compile())
    cg = ag.compile()
    assert P[cg.index["A"]] == pytest.approx(0.5)
    assert P[cg.index["B"]] == pytest.approx(0.25)
    assert P[cg.index["C"]] == pytest.approx(0.125)


def test_threat_evidence_increases_downstream_risk(ag):
    base = total_risk(ag.compile(), propagate(ag.compile()))
    ag.apply_threats([ThreatAssessment("APP-01", "EXPLOIT", 0.95, 0.95)])
    cg = ag.compile()
    P = propagate(cg)
    assert total_risk(cg, P) > base
    assert P[cg.index["DB-01"]] > 0.4


def test_batched_propagation_consistent(ag):
    cg = ag.compile()
    rng = np.random.default_rng(0)
    em = rng.uniform(0, 1, size=(4, cg.n_edges))
    batched = propagate(cg, em)
    for b in range(4):
        np.testing.assert_allclose(batched[b], propagate(cg, em[b]), atol=1e-9)


def test_cutting_entry_edges_removes_risk(ag):
    cg = ag.compile()
    em = np.ones(cg.n_edges)
    for (u, _), i in cg.edge_index.items():
        if u == ATTACKER:
            em[i] = 0.0
    assert total_risk(cg, propagate(cg, em)) == pytest.approx(0.0, abs=1e-12)


def test_attack_paths_reach_admin(ag):
    paths = top_attack_paths(ag.g, ["ADMIN"], k=3)
    assert paths and paths[0].nodes[0] == ATTACKER and paths[0].nodes[-1] == "ADMIN"
    assert paths[0].probability >= paths[-1].probability
    assert "ADMIN" in critical_targets(ag.g)


@pytest.mark.parametrize("n", [15, 75, 300])
def test_generated_topologies_scale(n):
    topo = generate_topology(n, seed=1)
    ag = AttackGraph(topo)
    assert abs(len(topo.assets) - n) <= 2
    P = propagate(ag.compile())
    assert np.all((P >= 0) & (P <= 1))


def test_parse_nvd_record():
    rec = {"cve": {"id": "CVE-2099-0001", "descriptions": [{"lang": "en", "value": "x"}],
                   "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 9.8,
                                                               "privilegesRequired": "NONE"},
                                                  "exploitabilityScore": 3.9}]}}}
    v = parse_nvd_record(rec)
    assert v.cvss_base == 9.8 and v.exploitability == pytest.approx(1.0)
    assert not v.requires_privileges
