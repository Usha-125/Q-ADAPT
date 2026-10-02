import json

import pytest

from qadapt.cli import main


def test_demo_text(capsys):
    main(["demo", "--solver", "greedy"])
    out = capsys.readouterr().out
    assert "Q-ADAPT defense recommendation" in out and "[+]" in out and "risk" in out


def test_demo_json_qaoa(capsys):
    main(["demo", "--p", "1", "--max-qubits", "8", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["result"]["feasible"] and d["risk_after"] <= d["risk_before"]


@pytest.mark.parametrize("scenario", ["small", "medium"])
def test_demo_generated_scenarios(capsys, scenario):
    main(["demo", "--scenario", scenario, "--solver", "milp", "--json"])
    assert json.loads(capsys.readouterr().out)["result"]["feasible"]


def test_episode(capsys):
    main(["episode", "--policy", "static", "--steps", "2", "--solver", "greedy"])
    d = json.loads(capsys.readouterr().out)
    assert d["policy"] == "static" and "history" not in d


def test_train_synthetic(tmp_path, capsys):
    out = tmp_path / "det.joblib"
    main(["train", "--synthetic", "800", "--model", "random_forest", "--out", str(out)])
    assert out.exists() and "accuracy" in capsys.readouterr().out
    from qadapt.ml_engine import ThreatDetector
    assert ThreatDetector.load(out).model_name == "random_forest"


def test_train_from_csv(tmp_path, capsys):
    from qadapt.ml_engine import generate_flows
    df = generate_flows(400, seed=0)
    df = df.rename(columns={"flow_duration": "Flow Duration", "tot_fwd_pkts": "Tot Fwd Pkts"})
    df["Label"] = df.pop("category").map(lambda c: "Benign" if c == "BENIGN" else "DDOS attack-HOIC")
    df.to_csv(tmp_path / "x.csv", index=False)
    main(["train", "--csv", str(tmp_path / "x.csv"), "--out", str(tmp_path / "m.joblib")])
    assert (tmp_path / "m.joblib").exists()


def test_benchmark_json(capsys):
    main(["benchmark", "--sizes", "6", "--seeds", "1", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert any(r["solver"] == "qaoa" for r in rows)


def test_bad_arguments():
    with pytest.raises(SystemExit):
        main(["demo", "--scenario", "galaxy"])
    with pytest.raises(SystemExit):
        main([])
