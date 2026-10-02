import pandas as pd
import pytest

from qadapt.ml_engine import (
    BENIGN,
    ThreatDetector,
    canonicalize_cic,
    generate_flows,
    map_label,
    train_and_evaluate,
)


@pytest.mark.parametrize("raw,expected", [
    ("Benign", BENIGN),
    ("BENIGN", BENIGN),
    ("DDOS attack-HOIC", "DDOS"),
    ("DoS attacks-Hulk", "DOS"),
    ("SSH-Bruteforce", "BRUTE_FORCE"),
    ("Web Attack \x96 XSS", "WEB_ATTACK"),
    ("Infilteration", "INFILTRATION"),
    ("PortScan", "RECON"),
    ("Exploits", "EXPLOIT"),
])
def test_label_mapping(raw, expected):
    assert map_label(raw) == expected


def test_canonicalize_handles_2017_and_2018_spellings():
    df17 = pd.DataFrame({" Destination Port": [80], " Flow Duration": [10],
                         " Total Fwd Packets": [3], " Label": ["DDoS"]})
    df18 = pd.DataFrame({"Dst Port": [80], "Flow Duration": [10],
                         "Tot Fwd Pkts": [3], "Label": ["Benign"]})
    a, b = canonicalize_cic(df17), canonicalize_cic(df18)
    assert list(a.columns) == list(b.columns)
    assert a.loc[0, "tot_fwd_pkts"] == b.loc[0, "tot_fwd_pkts"] == 3
    assert a.loc[0, "category"] == "DDOS" and b.loc[0, "category"] == BENIGN


def test_train_and_evaluate_random_forest():
    train = generate_flows(2000, seed=1)
    test = generate_flows(800, seed=2)
    _, report = train_and_evaluate("random_forest", train, test)
    assert report.accuracy > 0.6
    assert 0.5 < report.roc_auc_binary <= 1.0


def test_detector_attributes_threat_to_attacked_host():
    train = generate_flows(2000, seed=3)
    det = ThreatDetector.train(train, "random_forest")
    live = generate_flows(600, attack_fraction=0.2,
                          hosts=["10.0.0.5", "10.0.0.6", "10.0.0.7"],
                          attacked_hosts={"10.0.0.5": "DDOS"}, seed=4)
    threats = det.assess_hosts(live, {"10.0.0.5": "WEB-01"})
    assert threats and threats[0].host_id == "WEB-01"
    assert threats[0].attack_type == "DDOS"
    assert 0.0 < threats[0].probability <= 1.0
