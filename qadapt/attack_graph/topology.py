"""Synthetic enterprise network topologies.

Public IDS datasets contain no topology, asset criticality or defense metadata,
so Q-ADAPT evaluates decisions on controlled enterprise environments built here
(small / medium / large, plus the fixed demonstration network).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qadapt.attack_graph.vulnerabilities import APPLICABLE, CATALOG
from qadapt.core.models import Asset, AssetType

ZONE_OF = {
    AssetType.FIREWALL: "perimeter",
    AssetType.WEB_SERVER: "dmz",
    AssetType.MAIL_SERVER: "dmz",
    AssetType.VPN_GATEWAY: "dmz",
    AssetType.APP_SERVER: "app",
    AssetType.DATABASE: "data",
    AssetType.FILE_SERVER: "data",
    AssetType.WORKSTATION: "corp",
    AssetType.DOMAIN_CONTROLLER: "mgmt",
    AssetType.ADMIN: "mgmt",
}

BASE_CRITICALITY = {
    AssetType.FIREWALL: 0.6,
    AssetType.WEB_SERVER: 0.55,
    AssetType.MAIL_SERVER: 0.5,
    AssetType.VPN_GATEWAY: 0.6,
    AssetType.APP_SERVER: 0.7,
    AssetType.DATABASE: 0.9,
    AssetType.FILE_SERVER: 0.65,
    AssetType.WORKSTATION: 0.25,
    AssetType.DOMAIN_CONTROLLER: 0.95,
    AssetType.ADMIN: 1.0,
}

USERS = {
    AssetType.FIREWALL: 500, AssetType.WEB_SERVER: 300, AssetType.MAIL_SERVER: 400,
    AssetType.VPN_GATEWAY: 150, AssetType.APP_SERVER: 200, AssetType.DATABASE: 120,
    AssetType.FILE_SERVER: 150, AssetType.WORKSTATION: 1, AssetType.DOMAIN_CONTROLLER: 500,
    AssetType.ADMIN: 3,
}


@dataclass
class Topology:
    """Assets plus directed reachability links (u can initiate traffic to v)."""

    name: str
    assets: dict[str, Asset]
    links: list[tuple[str, str, str]] = field(default_factory=list)  # (u, v, kind)
    entry_points: list[str] = field(default_factory=list)  # reachable from the Internet

    def ip_map(self) -> dict[str, str]:
        return {a.ip: a.id for a in self.assets.values() if a.ip}


def _asset(aid: str, name: str, t: AssetType, rng: np.random.Generator, idx: int,
           crit: float | None = None, n_vulns: int | None = None) -> Asset:
    pool = APPLICABLE.get(t, ())
    k = n_vulns if n_vulns is not None else int(rng.integers(0, min(2, len(pool)) + 1)) if pool else 0
    vulns = [CATALOG[c] for c in rng.choice(pool, size=min(k, len(pool)), replace=False)] if k else []
    base = BASE_CRITICALITY[t] if crit is None else crit
    criticality = float(np.clip(base + rng.normal(0, 0.03), 0.05, 1.0)) if crit is None else crit
    zone = ZONE_OF[t]
    octet = {"perimeter": 1, "dmz": 2, "app": 3, "data": 4, "corp": 5, "mgmt": 6}[zone]
    return Asset(aid, name, t, criticality, zone, f"10.0.{octet}.{10 + idx}", vulns, USERS[t])


def demo_topology(seed: int = 7) -> Topology:
    """The fixed demonstration enterprise from the project proposal.

    Internet -> Firewall -> {Web, VPN}; Web -> App -> DB tier -> Admin;
    VPN -> employee PCs -> File server / DC.
    """
    rng = np.random.default_rng(seed)
    spec = [
        ("FW-01", "Perimeter Firewall", AssetType.FIREWALL, 0.6, 0),
        ("WEB-01", "Public Web Server", AssetType.WEB_SERVER, 0.6, 2),
        ("MAIL-01", "Mail Server", AssetType.MAIL_SERVER, 0.5, 1),
        ("VPN-01", "VPN Gateway", AssetType.VPN_GATEWAY, 0.6, 1),
        ("APP-01", "Application Server", AssetType.APP_SERVER, 0.75, 2),
        ("DB-01", "Customer Database", AssetType.DATABASE, 0.9, 1),
        ("FIN-DB", "Finance Database", AssetType.DATABASE, 0.95, 1),
        ("HR-DB", "HR Database", AssetType.DATABASE, 0.85, 1),
        ("FS-01", "File Server", AssetType.FILE_SERVER, 0.65, 1),
        ("PC-21", "Employee PC 21", AssetType.WORKSTATION, 0.25, 2),
        ("PC-22", "Employee PC 22", AssetType.WORKSTATION, 0.25, 1),
        ("PC-23", "Employee PC 23", AssetType.WORKSTATION, 0.25, 1),
        ("DC-01", "Domain Controller", AssetType.DOMAIN_CONTROLLER, 0.95, 1),
        ("ADMIN", "Admin Account / Jump Host", AssetType.ADMIN, 1.0, 1),
    ]
    assets = {aid: _asset(aid, n, t, rng, i, c, k) for i, (aid, n, t, c, k) in enumerate(spec)}
    links = [
        ("FW-01", "WEB-01", "network"), ("FW-01", "MAIL-01", "network"),
        ("FW-01", "VPN-01", "network"),
        ("WEB-01", "APP-01", "network"), ("MAIL-01", "PC-21", "phishing"),
        ("MAIL-01", "PC-22", "phishing"),
        ("VPN-01", "PC-21", "network"), ("VPN-01", "PC-22", "network"),
        ("VPN-01", "PC-23", "network"),
        ("APP-01", "DB-01", "credential"), ("APP-01", "FIN-DB", "credential"),
        ("APP-01", "HR-DB", "credential"),
        ("PC-21", "FS-01", "network"), ("PC-22", "FS-01", "network"),
        ("PC-23", "FS-01", "network"), ("PC-21", "PC-22", "lateral"),
        ("PC-22", "PC-23", "lateral"),
        ("PC-23", "DC-01", "credential"), ("FS-01", "DC-01", "credential"),
        ("DB-01", "ADMIN", "credential"), ("FIN-DB", "ADMIN", "credential"),
        ("DC-01", "ADMIN", "credential"), ("HR-DB", "DC-01", "credential"),
    ]
    return Topology("demo", assets, links, entry_points=["WEB-01", "MAIL-01", "VPN-01"])


def generate_topology(n_nodes: int = 50, seed: int = 0, name: str | None = None,
                      lateral_density: float = 0.08) -> Topology:
    """Random tiered enterprise with roughly ``n_nodes`` assets.

    Tier proportions (approx.): 8 % DMZ, 12 % app, 10 % data, 66 % workstations,
    4 % management, plus one firewall.
    """
    if n_nodes < 8:
        raise ValueError("n_nodes must be >= 8")
    rng = np.random.default_rng(seed)
    n = n_nodes - 1
    counts = {
        AssetType.WEB_SERVER: max(1, round(0.05 * n)),
        AssetType.MAIL_SERVER: 1,
        AssetType.VPN_GATEWAY: max(1, round(0.02 * n)),
        AssetType.APP_SERVER: max(1, round(0.12 * n)),
        AssetType.DATABASE: max(1, round(0.07 * n)),
        AssetType.FILE_SERVER: max(1, round(0.03 * n)),
        AssetType.DOMAIN_CONTROLLER: max(1, round(0.02 * n)),
        AssetType.ADMIN: max(1, round(0.02 * n)),
    }
    counts[AssetType.WORKSTATION] = max(1, n - sum(counts.values()))
    prefix = {AssetType.WEB_SERVER: "WEB", AssetType.MAIL_SERVER: "MAIL",
              AssetType.VPN_GATEWAY: "VPN", AssetType.APP_SERVER: "APP",
              AssetType.DATABASE: "DB", AssetType.FILE_SERVER: "FS",
              AssetType.DOMAIN_CONTROLLER: "DC", AssetType.ADMIN: "ADMIN",
              AssetType.WORKSTATION: "PC"}
    assets: dict[str, Asset] = {"FW-01": _asset("FW-01", "Firewall", AssetType.FIREWALL, rng, 0, n_vulns=0)}
    by_type: dict[AssetType, list[str]] = {}
    idx = 1
    for t, c in counts.items():
        for k in range(c):
            aid = f"{prefix[t]}-{k + 1:02d}"
            assets[aid] = _asset(aid, f"{t.value.replace('_', ' ').title()} {k + 1}", t, rng, idx)
            by_type.setdefault(t, []).append(aid)
            idx += 1

    links: list[tuple[str, str, str]] = []

    def connect(src_types, dst_types, prob, kind):
        for st in src_types:
            for u in by_type.get(st, []):
                targets = [v for dt in dst_types for v in by_type.get(dt, []) if v != u]
                if not targets:
                    continue
                chosen = [v for v in targets if rng.random() < prob]
                if not chosen:  # keep the tier connected
                    chosen = [targets[int(rng.integers(len(targets)))]]
                links.extend((u, v, kind) for v in chosen)

    dmz = [AssetType.WEB_SERVER, AssetType.MAIL_SERVER, AssetType.VPN_GATEWAY]
    links += [("FW-01", v, "network") for t in dmz for v in by_type[t]]
    connect([AssetType.WEB_SERVER], [AssetType.APP_SERVER], 0.5, "network")
    connect([AssetType.MAIL_SERVER], [AssetType.WORKSTATION], 0.15, "phishing")
    connect([AssetType.VPN_GATEWAY], [AssetType.WORKSTATION], 0.2, "network")
    connect([AssetType.APP_SERVER], [AssetType.DATABASE], 0.35, "credential")
    connect([AssetType.WORKSTATION], [AssetType.FILE_SERVER], 0.3, "network")
    connect([AssetType.WORKSTATION], [AssetType.WORKSTATION], lateral_density, "lateral")
    connect([AssetType.WORKSTATION, AssetType.FILE_SERVER], [AssetType.DOMAIN_CONTROLLER],
            0.04, "credential")
    connect([AssetType.DATABASE, AssetType.DOMAIN_CONTROLLER], [AssetType.ADMIN], 0.5, "credential")
    entry = [v for t in dmz for v in by_type[t]]
    return Topology(name or f"enterprise-{n_nodes}", assets, sorted(set(links)), entry)


SCENARIO_SIZES = {"small": 15, "medium": 75, "large": 300}


def scenario_topology(size: str = "demo", seed: int = 0) -> Topology:
    if size == "demo":
        return demo_topology()
    if size not in SCENARIO_SIZES:
        raise KeyError(f"unknown scenario size '{size}'")
    return generate_topology(SCENARIO_SIZES[size], seed=seed, name=size)
