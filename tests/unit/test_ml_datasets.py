import numpy as np
import pandas as pd
import pytest

from qadapt.ml_engine import (
    ATTACK_CATEGORIES,
    BENIGN,
    CANONICAL_FEATURES,
    MODEL_FACTORIES,
    ThreatDetector,
    canonicalize_cic,
    fit_model,
    generate_flows,
    load_cic_csvs,
    load_unsw_nb15,
    map_label,
)
from qadapt.ml_engine.datasets import find_dataset_files
from qadapt.ml_engine.models import build_model, evaluate
from qadapt.ml_engine.preprocessing import FlowPreprocessor, split_xy


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_label_is_benign(raw):
    assert map_label(raw) == BENIGN


def test_unknown_label_is_treated_as_attack():
    assert map_label("totally-new-attack") == "EXPLOIT"


def test_every_mapped_label_is_in_taxonomy():
    from qadapt.ml_engine.datasets import _LABEL_MAP
    assert set(_LABEL_MAP.values()) <= set(ATTACK_CATEGORIES)


def test_canonicalize_requires_label():
    with pytest.raises(ValueError, match="Label"):
        canonicalize_cic(pd.DataFrame({"Flow Duration": [1]}))


def test_canonicalize_missing_features_filled_and_ips_kept():
    df = pd.DataFrame({"Flow Duration": ["5"], "Label": ["Bot"], "Src IP": ["1.1.1.1"],
                       "Dst IP": ["2.2.2.2"]})
    out = canonicalize_cic(df)
    assert list(out.columns[: len(CANONICAL_FEATURES)]) == list(CANONICAL_FEATURES)
    assert out.loc[0, "flow_duration"] == 5 and out.loc[0, "tot_fwd_pkts"] == 0.0
    assert out.loc[0, "category"] == "BOTNET" and out.loc[0, "dst_ip"] == "2.2.2.2"


def _write_cic(path, label_col="Label", rows=20, repeated_header=False):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Dst Port": rng.integers(1, 1000, rows), "Flow Duration": rng.random(rows),
                       "Flow Byts/s": rng.random(rows), label_col: ["Benign", "DDOS attack-HOIC"] * (rows // 2)})
    df.loc[0, "Flow Byts/s"] = np.inf
    text = df.to_csv(index=False)
    if repeated_header:  # CSE-CIC-IDS2018 files repeat the header mid-file
        lines = text.splitlines()
        text = "\n".join(lines[:5] + [lines[0]] + lines[5:]) + "\n"
    path.write_text(text)
    return path


def test_load_cic_csvs_handles_repeated_header_and_sampling(tmp_path):
    f1 = _write_cic(tmp_path / "a.csv", repeated_header=True)
    f2 = _write_cic(tmp_path / "b.csv", label_col=" Label")
    df = load_cic_csvs([f1, f2], sample_per_file=10)
    assert len(df) == 20
    assert set(df["category"]) <= {BENIGN, "DDOS"}
    assert find_dataset_files(tmp_path) == [f1, f2]


def test_load_cic_csvs_requires_files():
    with pytest.raises(FileNotFoundError):
        load_cic_csvs([])


def test_load_unsw(tmp_path):
    pd.DataFrame({"id": [1, 2, 3], "dur": [0.1, 0.2, 0.3], "sbytes": [10, 20, 30],
                  "attack_cat": ["Normal", "Exploits", None], "label": [0, 1, 0]}).to_csv(
        tmp_path / "u.csv", index=False)
    df = load_unsw_nb15(tmp_path / "u.csv")
    assert list(df["category"]) == [BENIGN, "EXPLOIT", BENIGN]
    assert "id" not in df and "label" not in df


def test_load_unsw_requires_attack_cat(tmp_path):
    pd.DataFrame({"dur": [1]}).to_csv(tmp_path / "u.csv", index=False)
    with pytest.raises(ValueError):
        load_unsw_nb15(tmp_path / "u.csv")


def test_preprocessor_handles_inf_nan_and_unseen_columns():
    df = generate_flows(200, seed=0)
    X, _ = split_xy(df)
    prep = FlowPreprocessor().fit(X)
    bad = X.copy()
    bad["flow_duration"] = bad["flow_duration"].astype(float)
    bad.loc[bad.index[0], "flow_duration"] = np.inf
    bad.loc[bad.index[1], "flow_byts_s"] = np.nan
    bad.loc[bad.index[2], "flow_pkts_s"] = -np.inf
    bad = bad.drop(columns=["idle_mean"])  # column missing at inference time
    out = prep.transform(bad)
    assert np.isfinite(out).all() and out.shape[1] == len(prep.columns_)


def test_preprocessor_drops_constant_columns():
    df = generate_flows(100, seed=0)
    X, _ = split_xy(df)
    X["constant"] = 7.0
    assert "constant" not in FlowPreprocessor().fit(X).columns_


def test_synthetic_generator_shapes_and_pins_attacks():
    df = generate_flows(300, attack_fraction=0.3, hosts=["h1", "h2"], attacked_hosts={"h9": "DOS"}, seed=1)
    assert len(df) == 300
    attacks = df[df["category"] != BENIGN]
    assert set(attacks["category"]) == {"DOS"} and set(attacks["dst_ip"]) == {"h9"}
    assert (df[list(CANONICAL_FEATURES[1:])] >= 0).all().all()


def test_synthetic_generator_is_deterministic():
    pd.testing.assert_frame_equal(generate_flows(50, seed=3), generate_flows(50, seed=3))


def test_unknown_model_name():
    with pytest.raises(KeyError):
        build_model("svm-quantum")


@pytest.mark.parametrize("name", [m for m in MODEL_FACTORIES if m != "xgboost"])
def test_every_model_trains_and_reports(name):
    train, test = generate_flows(600, seed=1), generate_flows(200, seed=2)
    model = fit_model(name, train)
    rep = evaluate(name, model, test)
    assert 0 <= rep.accuracy <= 1 and 0 <= rep.false_positive_rate <= 1
    assert rep.classes == sorted(rep.classes)
    proba = model.predict_proba(split_xy(test)[0])
    np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-6)


def test_xgboost_optional():
    pytest.importorskip("xgboost")
    rep = evaluate("xgboost", fit_model("xgboost", generate_flows(400, seed=1)), generate_flows(100, seed=2))
    assert rep.accuracy > 0.3


def test_auc_is_nan_when_test_has_one_class():
    train = generate_flows(500, seed=1)
    test = generate_flows(100, attack_fraction=0.0, seed=2)
    rep = evaluate("rf", fit_model("random_forest", train), test)
    assert np.isnan(rep.roc_auc_binary)


def test_detector_requires_dst_ip():
    det = ThreatDetector.train(generate_flows(400, seed=0))
    with pytest.raises(ValueError, match="dst_ip"):
        det.assess_hosts(generate_flows(50, seed=1).drop(columns=["dst_ip"]))


def test_detector_returns_nothing_for_benign_traffic():
    det = ThreatDetector.train(generate_flows(1500, seed=0), flag_threshold=0.95)
    assert det.assess_hosts(generate_flows(100, attack_fraction=0.0, seed=5)) == []


def test_detector_scores_unlabelled_flows():
    det = ThreatDetector.train(generate_flows(500, seed=0))
    flows = generate_flows(50, seed=1).drop(columns=["category"])
    scored = det.score_flows(flows)
    assert scored["p_malicious"].between(0, 1).all() and len(scored) == 50


def test_detector_probability_bounded_and_evidence_grows():
    det = ThreatDetector.train(generate_flows(1500, seed=0))
    few = generate_flows(60, attack_fraction=0.05, hosts=["a"], attacked_hosts={"v": "DDOS"}, seed=1)
    many = generate_flows(60, attack_fraction=0.5, hosts=["a"], attacked_hosts={"v": "DDOS"}, seed=1)
    p_few = next(t for t in det.assess_hosts(few) if t.host_id == "v").probability
    p_many = next(t for t in det.assess_hosts(many) if t.host_id == "v").probability
    assert 0 < p_few < p_many <= 1


def test_detector_save_load_roundtrip(tmp_path):
    det = ThreatDetector.train(generate_flows(400, seed=0))
    path = tmp_path / "m" / "det.joblib"
    det.save(path)
    loaded = ThreatDetector.load(path)
    flows = generate_flows(80, seed=4)
    pd.testing.assert_frame_equal(det.score_flows(flows), loaded.score_flows(flows))
