"""Loaders for public IDS datasets and a unified label/feature schema.

Supported:
  * CSE-CIC-IDS2018 (primary)   - CICFlowMeter-V3 CSVs, column ``Label``
  * CIC-IDS2017 (cross-dataset) - CICFlowMeter CSVs, column `` Label``
  * UNSW-NB15 (third validation) - column ``attack_cat`` / ``label``

The two CIC datasets use slightly different column spellings for the same
CICFlowMeter features, so both are mapped onto :data:`CANONICAL_FEATURES`.
That enables train-on-2018 / test-on-2017 external validation.

The raw datasets are not redistributed. Download them into ``data/raw/``
(see ``docs/DATASETS.md``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

BENIGN = "BENIGN"
ATTACK_CATEGORIES = (
    BENIGN,
    "DOS",
    "DDOS",
    "BRUTE_FORCE",
    "WEB_ATTACK",
    "BOTNET",
    "INFILTRATION",
    "RECON",
    "EXPLOIT",
)

# Canonical CICFlowMeter features shared by CIC-IDS2017 and CSE-CIC-IDS2018.
# key = canonical name, value = known aliases (normalised: lower-case, alnum only)
_FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "dst_port": ("dstport", "destinationport"),
    "flow_duration": ("flowduration",),
    "tot_fwd_pkts": ("totfwdpkts", "totalfwdpackets"),
    "tot_bwd_pkts": ("totbwdpkts", "totalbackwardpackets"),
    "totlen_fwd_pkts": ("totlenfwdpkts", "totallengthoffwdpackets"),
    "totlen_bwd_pkts": ("totlenbwdpkts", "totallengthofbwdpackets"),
    "fwd_pkt_len_mean": ("fwdpktlenmean", "fwdpacketlengthmean"),
    "bwd_pkt_len_mean": ("bwdpktlenmean", "bwdpacketlengthmean"),
    "flow_byts_s": ("flowbytss", "flowbytess"),
    "flow_pkts_s": ("flowpktss", "flowpacketss"),
    "flow_iat_mean": ("flowiatmean",),
    "flow_iat_std": ("flowiatstd",),
    "fwd_iat_mean": ("fwdiatmean",),
    "bwd_iat_mean": ("bwdiatmean",),
    "syn_flag_cnt": ("synflagcnt", "synflagcount"),
    "rst_flag_cnt": ("rstflagcnt", "rstflagcount"),
    "psh_flag_cnt": ("pshflagcnt", "pshflagcount"),
    "ack_flag_cnt": ("ackflagcnt", "ackflagcount"),
    "pkt_size_avg": ("pktsizeavg", "averagepacketsize"),
    "init_fwd_win_byts": ("initfwdwinbyts", "initwinbytesforward"),
    "init_bwd_win_byts": ("initbwdwinbyts", "initwinbytesbackward"),
    "active_mean": ("activemean",),
    "idle_mean": ("idlemean",),
}
CANONICAL_FEATURES: tuple[str, ...] = tuple(_FEATURE_ALIASES)

# Raw label -> unified category. Keys are normalised (lower-case, alnum only).
_LABEL_MAP: dict[str, str] = {
    # common
    "benign": BENIGN,
    "normal": BENIGN,
    # CSE-CIC-IDS2018
    "bot": "BOTNET",
    "bruteforceweb": "WEB_ATTACK",
    "bruteforcexss": "WEB_ATTACK",
    "sqlinjection": "WEB_ATTACK",
    "ddosattackhoic": "DDOS",
    "ddosattackloicudp": "DDOS",
    "ddosattacksloichttp": "DDOS",
    "dosattacksgoldeneye": "DOS",
    "dosattackshulk": "DOS",
    "dosattacksslowhttptest": "DOS",
    "dosattacksslowloris": "DOS",
    "ftpbruteforce": "BRUTE_FORCE",
    "sshbruteforce": "BRUTE_FORCE",
    "infilteration": "INFILTRATION",
    "infiltration": "INFILTRATION",
    # CIC-IDS2017
    "doshulk": "DOS",
    "dosgoldeneye": "DOS",
    "dosslowloris": "DOS",
    "dosslowhttptest": "DOS",
    "ddos": "DDOS",
    "portscan": "RECON",
    "ftppatator": "BRUTE_FORCE",
    "sshpatator": "BRUTE_FORCE",
    "webattackbruteforce": "WEB_ATTACK",
    "webattackxss": "WEB_ATTACK",
    "webattacksqlinjection": "WEB_ATTACK",
    "heartbleed": "EXPLOIT",
    # UNSW-NB15 attack_cat
    "generic": "EXPLOIT",
    "exploits": "EXPLOIT",
    "fuzzers": "RECON",
    "dos": "DOS",
    "reconnaissance": "RECON",
    "analysis": "RECON",
    "backdoor": "INFILTRATION",
    "backdoors": "INFILTRATION",
    "shellcode": "EXPLOIT",
    "worms": "BOTNET",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def map_label(raw: str) -> str:
    """Map a raw dataset label onto the unified attack taxonomy."""
    key = _norm(raw)
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    # tolerate encoding artefacts such as "Web Attack \x96 XSS"
    for k, v in _LABEL_MAP.items():
        if len(k) > 4 and k in key:
            return v
    return "EXPLOIT" if key else BENIGN


def canonicalize_cic(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CICFlowMeter columns to canonical names and add a ``category`` column."""
    norm_cols = {_norm(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for canon, aliases in _FEATURE_ALIASES.items():
        src = next((norm_cols[a] for a in aliases if a in norm_cols), None)
        out[canon] = pd.to_numeric(df[src], errors="coerce") if src is not None else 0.0
    label_col = norm_cols.get("label")
    if label_col is None:
        raise ValueError("CIC dataset is missing a 'Label' column")
    out["category"] = df[label_col].astype(str).map(map_label)
    for meta, aliases in {"src_ip": ("srcip", "sourceip"), "dst_ip": ("dstip", "destinationip")}.items():
        src = next((norm_cols[a] for a in aliases if a in norm_cols), None)
        if src is not None:
            out[meta] = df[src].astype(str)
    return out


def load_cic_csvs(paths: Iterable[str | Path], sample_per_file: int | None = None,
                  seed: int = 0) -> pd.DataFrame:
    """Load and canonicalise one or more CIC-IDS2017 / CSE-CIC-IDS2018 CSV files."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        df = df[df.iloc[:, 0].astype(str) != df.columns[0]]  # repeated header rows (2018)
        if sample_per_file and len(df) > sample_per_file:
            df = df.sample(sample_per_file, random_state=seed)
        frames.append(canonicalize_cic(df))
    if not frames:
        raise FileNotFoundError("no CSV files given")
    return pd.concat(frames, ignore_index=True)


def load_unsw_nb15(path: str | Path) -> pd.DataFrame:
    """Load a UNSW-NB15 training/testing CSV; numeric features + ``category``."""
    df = pd.read_csv(path, low_memory=False)
    cat_col = "attack_cat" if "attack_cat" in df.columns else None
    if cat_col is None:
        raise ValueError("UNSW-NB15 file must contain 'attack_cat'")
    cats = df[cat_col].fillna("Normal").astype(str).map(map_label)
    num = df.select_dtypes(include=[np.number]).drop(columns=["id", "label"], errors="ignore")
    num["category"] = cats.values
    return num


def find_dataset_files(root: str | Path, pattern: str = "*.csv") -> list[Path]:
    return sorted(Path(root).glob(pattern))
