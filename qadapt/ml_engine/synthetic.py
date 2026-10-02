"""Synthetic CICFlowMeter-style flow generator.

Used for unit tests, CI and offline demos when the public datasets are not
available. Each attack category has a class-conditional log-normal profile over
the canonical features with deliberate overlap, so classifiers do not reach a
trivial 100 % accuracy. Results produced on synthetic data must NOT be reported
as dataset results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qadapt.ml_engine.datasets import ATTACK_CATEGORIES, BENIGN, CANONICAL_FEATURES

# (log-mean multiplier, typical dst ports) per category; base profile is BENIGN
_PROFILES: dict[str, dict] = {
    BENIGN: {"shift": {}, "ports": [80, 443, 53, 445, 3389]},
    "DOS": {"shift": {"flow_pkts_s": 2.0, "tot_fwd_pkts": 1.2, "flow_iat_mean": -1.8,
                      "syn_flag_cnt": 1.5, "flow_duration": 0.8}, "ports": [80, 443]},
    "DDOS": {"shift": {"flow_pkts_s": 2.6, "flow_byts_s": 2.0, "flow_iat_mean": -2.2,
                       "tot_fwd_pkts": 1.6, "init_fwd_win_byts": -1.0}, "ports": [80]},
    "BRUTE_FORCE": {"shift": {"tot_fwd_pkts": 0.6, "psh_flag_cnt": 1.4, "flow_duration": -0.8,
                              "fwd_pkt_len_mean": -0.6}, "ports": [21, 22]},
    "WEB_ATTACK": {"shift": {"fwd_pkt_len_mean": 1.1, "totlen_fwd_pkts": 1.3,
                             "psh_flag_cnt": 0.9, "ack_flag_cnt": 0.6}, "ports": [80, 443]},
    "BOTNET": {"shift": {"idle_mean": 1.8, "active_mean": -0.8, "flow_iat_std": 1.2,
                         "pkt_size_avg": -0.7}, "ports": [8080, 6667]},
    "INFILTRATION": {"shift": {"bwd_pkt_len_mean": 1.0, "totlen_bwd_pkts": 1.5,
                               "flow_duration": 1.2}, "ports": [445, 135, 139]},
    "RECON": {"shift": {"tot_fwd_pkts": -1.4, "tot_bwd_pkts": -1.4, "rst_flag_cnt": 2.0,
                        "flow_duration": -2.0}, "ports": list(range(1, 1024, 37))},
    "EXPLOIT": {"shift": {"fwd_pkt_len_mean": 1.6, "bwd_pkt_len_mean": 1.4,
                          "init_bwd_win_byts": 1.0}, "ports": [443, 8443, 3306]},
}

_BASE_LOGMEAN = {f: 3.0 + (i % 5) * 0.7 for i, f in enumerate(CANONICAL_FEATURES)}


def generate_flows(n: int = 5000, attack_fraction: float = 0.35,
                   categories: tuple[str, ...] = ATTACK_CATEGORIES,
                   hosts: list[str] | None = None, attacked_hosts: dict[str, str] | None = None,
                   noise: float = 0.9, seed: int = 0) -> pd.DataFrame:
    """Generate ``n`` labelled flows.

    ``attacked_hosts`` (ip -> category) pins attack flows of a category to a
    destination host, which is how a scenario injects attacks on specific assets.
    """
    rng = np.random.default_rng(seed)
    attack_cats = [c for c in categories if c != BENIGN]
    n_attack = int(n * attack_fraction)
    labels = [BENIGN] * (n - n_attack)
    if attacked_hosts:
        cats = list(attacked_hosts.values())
        labels += [cats[i % len(cats)] for i in range(n_attack)]
    else:
        labels += list(rng.choice(attack_cats, size=n_attack))
    labels = np.array(labels)
    rng.shuffle(labels)

    hosts = hosts or [f"10.0.0.{i}" for i in range(10, 30)]
    target_of = {c: [ip for ip, cc in (attacked_hosts or {}).items() if cc == c] for c in attack_cats}

    rows = np.empty((n, len(CANONICAL_FEATURES)))
    dst_ip, dst_port = [], []
    for k, lab in enumerate(labels):
        prof = _PROFILES[lab]
        for j, f in enumerate(CANONICAL_FEATURES):
            mu = _BASE_LOGMEAN[f] + prof["shift"].get(f, 0.0)
            rows[k, j] = np.expm1(max(0.0, rng.normal(mu, noise)))
        dst_port.append(int(rng.choice(prof["ports"])))
        pool = target_of.get(lab) or hosts
        dst_ip.append(str(rng.choice(pool)))

    df = pd.DataFrame(rows, columns=list(CANONICAL_FEATURES))
    df["dst_port"] = dst_port
    df["src_ip"] = [f"203.0.113.{rng.integers(1, 255)}" if lab != BENIGN
                    else f"10.0.1.{rng.integers(1, 255)}" for lab in labels]
    df["dst_ip"] = dst_ip
    df["category"] = labels
    return df
