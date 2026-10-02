"""Domain model shared by every Q-ADAPT layer.

All scores are normalised to [0, 1] unless stated otherwise so that the
objective terms of the defense QUBO are commensurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AssetType(str, Enum):
    INTERNET = "internet"
    FIREWALL = "firewall"
    WEB_SERVER = "web_server"
    APP_SERVER = "app_server"
    DATABASE = "database"
    VPN_GATEWAY = "vpn_gateway"
    WORKSTATION = "workstation"
    DOMAIN_CONTROLLER = "domain_controller"
    ADMIN = "admin"
    MAIL_SERVER = "mail_server"
    FILE_SERVER = "file_server"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_score(cls, score: float) -> Severity:
        if score >= 0.85:
            return cls.CRITICAL
        if score >= 0.6:
            return cls.HIGH
        if score >= 0.3:
            return cls.MEDIUM
        return cls.LOW


class ActionType(str, Enum):
    ISOLATE_HOST = "isolate_host"
    BLOCK_IP = "block_ip"
    REVOKE_CREDENTIALS = "revoke_credentials"
    DISABLE_ACCOUNT = "disable_account"
    PATCH_VULNERABILITY = "patch_vulnerability"
    SEGMENT_NETWORK = "segment_network"
    INCREASE_MONITORING = "increase_monitoring"
    PROTECT_DATABASE = "protect_database"
    BLOCK_PORT = "block_port"
    QUARANTINE_ENDPOINT = "quarantine_endpoint"


@dataclass(frozen=True)
class Vulnerability:
    """A vulnerability instance on an asset (CVE + CVSS-derived scores)."""

    cve_id: str
    cvss_base: float  # 0..10
    exploitability: float  # 0..1, likelihood an attacker can exploit it
    requires_privileges: bool = False
    description: str = ""

    @property
    def severity(self) -> float:
        return self.cvss_base / 10.0


@dataclass
class Asset:
    """A node of the enterprise network."""

    id: str
    name: str
    type: AssetType
    criticality: float  # business/security impact if compromised, 0..1
    zone: str = "internal"
    ip: str = ""
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    users: int = 1  # people affected by disruption of this asset

    @property
    def exploitability(self) -> float:
        """Probability that at least one vulnerability is exploitable (noisy-OR)."""
        p_safe = 1.0
        for v in self.vulnerabilities:
            p_safe *= 1.0 - v.exploitability
        return 1.0 - p_safe

    @property
    def max_cvss(self) -> float:
        return max((v.cvss_base for v in self.vulnerabilities), default=0.0)


@dataclass
class ThreatAssessment:
    """Output of the ML threat engine for a single host."""

    host_id: str
    attack_type: str
    probability: float
    confidence: float
    n_events: int = 1
    source: str = ""

    @property
    def severity(self) -> Severity:
        return Severity.from_score(self.probability * self.confidence)

    def to_dict(self) -> dict:
        return {
            "host_id": self.host_id,
            "attack_type": self.attack_type,
            "probability": round(self.probability, 4),
            "confidence": round(self.confidence, 4),
            "severity": self.severity.value,
            "n_events": self.n_events,
            "source": self.source,
        }


@dataclass
class DefenseAction:
    """A candidate defensive action and its operational attributes.

    ``edge_effects`` maps attack-graph edges ``(u, v)`` to the fraction of the
    exploit probability that the action removes (0 = no effect, 1 = edge cut).
    ``node_effects`` maps a node to the fractional reduction of its local
    (ML-evidenced) compromise probability.
    """

    id: str
    type: ActionType
    target: str
    description: str
    cost: float  # normalised monetary/analyst cost, 0..1
    time: float  # normalised time-to-effect, 0..1
    disruption: float  # normalised business disruption, 0..1
    effectiveness: float  # nominal effectiveness, 0..1
    edge_effects: dict[tuple[str, str], float] = field(default_factory=dict)
    node_effects: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "target": self.target,
            "description": self.description,
            "cost": round(self.cost, 4),
            "time": round(self.time, 4),
            "disruption": round(self.disruption, 4),
            "effectiveness": round(self.effectiveness, 4),
        }
