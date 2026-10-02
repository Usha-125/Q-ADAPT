import pytest
from fastapi.testclient import TestClient

from qadapt.api.app import app


@pytest.fixture
def client():
    c = TestClient(app)
    c.post("/api/session/reset", json={"scenario": "demo"})
    return c


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and "qaoa" in r["solvers"]


def test_overview_and_graph(client):
    ov = client.get("/api/overview").json()
    assert ov["threats_detected"] >= 1 and 0 < ov["current_risk"] <= 1
    g = client.get("/api/graph").json()
    assert any(n["data"]["id"] == "ATTACKER" for n in g["nodes"])
    assert g["edges"]


def test_risk_models(client):
    for m in ("propagation", "multiplicative", "severity_only"):
        assert client.get(f"/api/risk?model={m}").status_code == 200
    assert client.get("/api/risk?model=bogus").status_code == 400


def test_optimize_approve_reduces_risk(client):
    before = client.get("/api/overview").json()["current_risk"]
    rep = client.post("/api/optimize", json={"solver": "qaoa", "p": 1, "shots": 256,
                                             "max_qubits": 8, "budget": 0.4}).json()
    assert rep["result"]["feasible"] and rep["selected"]
    ov = client.post(f"/api/runs/{rep['run_id']}/decision", json={"decision": "approve"}).json()
    assert ov["current_risk"] < before
    # the realised risk after approval must equal the optimiser's prediction
    assert ov["current_risk"] == pytest.approx(rep["risk_after"], abs=1e-4)
    assert ov["applied_actions"]
    assert client.get(f"/api/runs/{rep['run_id']}").json()["decisions"]


def test_reject_does_not_apply(client):
    rep = client.post("/api/optimize", json={"solver": "greedy", "max_qubits": 8}).json()
    ov = client.post(f"/api/runs/{rep['run_id']}/decision", json={"decision": "reject"}).json()
    assert ov["applied_actions"] == []


def test_compare(client):
    r = client.post("/api/compare", json={"solvers": ["greedy", "milp", "exhaustive"],
                                          "max_qubits": 8}).json()
    assert len(r["results"]) == 3
    assert min(x["gap_to_best"] for x in r["results"]) == 0


def test_staged_attack_and_ml_detection(client):
    r1 = client.post("/api/adaptive/next-stage").json()
    assert r1["stage"] == 2
    det = client.post("/api/threats/detect", json={"attacked": {"FS-01": "INFILTRATION"}}).json()
    assert any(d["host_id"] == "FS-01" for d in det["detections"])


def test_inject_unknown_asset_404(client):
    assert client.post("/api/threats", json={"host_id": "NOPE"}).status_code == 404
