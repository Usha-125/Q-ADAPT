"""API integration: validation, error handling and multi-step analyst workflows."""

import pytest
from fastapi.testclient import TestClient

from qadapt.api import app as app_module
from qadapt.api.app import app
from qadapt.api.storage import Storage

FAST = {"solver": "qaoa", "p": 1, "shots": 128, "max_qubits": 8}


@pytest.fixture
def client():
    c = TestClient(app)
    assert c.post("/api/session/reset", json={"scenario": "demo"}).status_code == 200
    return c


@pytest.mark.parametrize("body", [
    {"budget": -0.1}, {"max_time": -1}, {"max_disruption": -1}, {"max_actions": -1},
    {"p": 0}, {"p": 9}, {"shots": 10}, {"max_qubits": 1}, {"max_qubits": 50},
    {"noise": "extreme"}, {"backend": "ibm"}, {"encoding": "magic"}, {"surrogate": "cubic"},
    {"weights": {"alpha": -1}},
])
def test_optimize_rejects_invalid_input(client, body):
    assert client.post("/api/optimize", json=body).status_code == 422


def test_optimize_unknown_solver(client):
    assert client.post("/api/optimize", json={"solver": "nope"}).status_code == 400


def test_too_many_qubits_is_a_clear_400(client):
    r = client.post("/api/optimize", json={"solver": "qaoa", "max_qubits": 20, "encoding": "slack",
                                           "budget": 0.3, "max_time": 0.3, "max_disruption": 0.3,
                                           "max_actions": 3})
    assert r.status_code == 400 and "qubits" in r.json()["detail"]


def test_compare_reports_per_solver_errors(client):
    r = client.post("/api/compare", json={"solvers": ["qaoa", "greedy"], "max_qubits": 20, "encoding": "slack",
                                          "budget": 0.3, "max_time": 0.3, "max_disruption": 0.3,
                                          "max_actions": 3}).json()
    assert [x["solver"] for x in r["results"]] == ["greedy"] and "qaoa" in r["errors"]


def test_compare_rejects_unknown_solver(client):
    assert client.post("/api/compare", json={"solvers": ["greedy", "warp"]}).status_code == 400


def test_compare_gap_is_zero_for_best(client):
    r = client.post("/api/compare", json={"solvers": ["exhaustive", "greedy", "score_ranking"],
                                          "max_qubits": 8}).json()
    ex = next(x for x in r["results"] if x["solver"] == "exhaustive")
    assert ex["gap_to_best"] == 0 and all(x["gap_to_best"] >= 0 for x in r["results"] if x["feasible"])


def test_empty_action_list_is_rejected_not_approve_all(client):
    rep = client.post("/api/optimize", json=FAST).json()
    r = client.post(f"/api/runs/{rep['run_id']}/decision", json={"decision": "approve", "action_ids": []})
    assert r.status_code == 400
    assert client.get("/api/overview").json()["applied_actions"] == []


def test_partial_approval_keeps_rest_pending(client):
    rep = client.post("/api/optimize", json={"solver": "exhaustive", "max_qubits": 10, "budget": 1.0,
                                             "weights": {"alpha": 3, "beta": 0, "gamma": 0, "delta": 0}}).json()
    ids = [a["id"] for a in rep["selected"]]
    assert len(ids) >= 2
    ov = client.post(f"/api/runs/{rep['run_id']}/decision",
                     json={"decision": "approve", "action_ids": ids[:1]}).json()
    assert ov["pending_run"] == rep["run_id"] and len(ov["applied_actions"]) == 1
    ov = client.post(f"/api/runs/{rep['run_id']}/decision", json={"decision": "reject"}).json()
    assert ov["pending_run"] is None and len(ov["applied_actions"]) == 1
    decisions = client.get(f"/api/runs/{rep['run_id']}").json()["decisions"]
    assert [d["decision"] for d in decisions] == ["approve"] + ["reject"] * (len(ids) - 1)


def test_decision_on_stale_or_unknown_run(client):
    rep = client.post("/api/optimize", json=FAST).json()
    client.post("/api/optimize", json=FAST)  # newer run replaces the pending one
    assert client.post(f"/api/runs/{rep['run_id']}/decision", json={}).status_code == 409
    assert client.get("/api/runs/999999").status_code == 404


def test_decision_with_unknown_action(client):
    rep = client.post("/api/optimize", json=FAST).json()
    r = client.post(f"/api/runs/{rep['run_id']}/decision", json={"action_ids": ["made_up:X"]})
    assert r.status_code == 404


def test_approved_actions_are_not_recommended_again(client):
    rep = client.post("/api/optimize", json=FAST).json()
    client.post(f"/api/runs/{rep['run_id']}/decision", json={"decision": "approve"})
    rep2 = client.post("/api/optimize", json=FAST).json()
    assert not {a["id"] for a in rep["selected"]} & {a["id"] for a in rep2["candidates"]}


def test_staged_attack_runs_to_completion(client):
    stages = []
    for _ in range(2):  # demo starts at stage 1 of 3
        r = client.post("/api/adaptive/next-stage")
        assert r.status_code == 200
        stages.append(r.json()["stage"])
    assert stages == [2, 3]
    assert client.post("/api/adaptive/next-stage").status_code == 409
    ov = client.get("/api/overview").json()
    assert ov["threats_detected"] == 3 and len(ov["timeline"]) == 3


def test_next_stage_only_for_demo(client):
    client.post("/api/session/reset", json={"scenario": "small"})
    assert client.post("/api/adaptive/next-stage").status_code == 400


@pytest.mark.parametrize("scenario", ["small", "medium"])
def test_generated_scenarios_end_to_end(client, scenario):
    ov = client.post("/api/session/reset", json={"scenario": scenario, "seed": 2}).json()
    assert ov["threats_detected"] == 1 and ov["stages"] == []
    rep = client.post("/api/optimize", json={**FAST, "max_qubits": 10}).json()
    assert rep["result"]["feasible"] and rep["risk_after"] <= rep["risk_before"]


def test_reset_without_threats(client):
    ov = client.post("/api/session/reset", json={"scenario": "demo", "with_initial_threats": False}).json()
    assert ov["threats_detected"] == 0 and ov["current_risk"] > 0  # attacker-activity prior


def test_unknown_scenario_and_asset(client):
    assert client.post("/api/session/reset", json={"scenario": "galaxy"}).status_code == 422
    assert client.post("/api/threats/detect", json={"attacked": {"NOPE": "DOS"}}).status_code == 404


def test_inject_threat_and_audit(client):
    client.post("/api/threats", json={"host_id": "HR-DB", "attack_type": "INFILTRATION",
                                      "probability": 0.8, "confidence": 0.9})
    assert any(t["host_id"] == "HR-DB" for t in client.get("/api/threats").json())
    events = client.get("/api/audit").json()["events"]
    assert events[0]["host"] == "HR-DB"
    assert client.post("/api/threats", json={"host_id": "HR-DB", "probability": 1.5}).status_code == 422


def test_detection_with_no_attack_traffic(client):
    r = client.post("/api/threats/detect", json={"attacked": {}, "n_flows": 200}).json()
    assert r["flows"] == 200


def test_runs_listing(client):
    client.post("/api/optimize", json=FAST)
    runs = client.get("/api/runs").json()
    assert runs and {"id", "solver", "risk_before", "risk_after"} <= runs[0].keys()


def test_graph_highlights_defended_edges_after_approval(client):
    rep = client.post("/api/optimize", json=FAST).json()
    client.post(f"/api/runs/{rep['run_id']}/decision", json={"decision": "approve"})
    edges = client.get("/api/graph").json()["edges"]
    assert any(e["data"]["defended"] > 0 for e in edges)


def test_all_responses_are_strict_json(client):
    import json
    rep = client.post("/api/optimize", json=FAST)
    json.loads(rep.text, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))


def test_storage_persists_to_file(tmp_path, monkeypatch):
    path = tmp_path / "db" / "q.db"
    store = Storage(path)
    store.log_event("H", "DOS", 0.9, 0.8, "1.2.3.4")
    rid = store.save_run("greedy", {"a": 1}, {"risk_before": 0.5, "risk_after": 0.2,
                                              "result": {"objective": 0.1, "feasible": True}})
    store.log_decision(rid, "x", "approve")
    again = Storage(path)  # reopen: data survived
    assert again.get_run(rid)["params"] == {"a": 1}
    assert again.decisions(rid)[0]["action_id"] == "x" and again.events()[0]["host"] == "H"
    assert again.get_run(rid + 1) is None


def test_health_reports_capabilities(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and isinstance(h["qiskit"], bool) and app_module.QISKIT == h["qiskit"]
