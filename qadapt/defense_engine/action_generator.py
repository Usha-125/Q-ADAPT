"""Candidate defense action generation.

Actions are generated around the current threat picture (hosts with ML
evidence, high-risk assets, entry points, attack-path pivots) and then
pre-screened by their individual marginal risk reduction so the optimisation
problem fits the qubit budget of the quantum solver.
"""

from __future__ import annotations

from dataclasses import dataclass

from qadapt.attack_graph.graph_builder import ATTACKER, AttackGraph
from qadapt.core.models import ActionType, AssetType, DefenseAction

SERVER_TYPES = {AssetType.WEB_SERVER, AssetType.APP_SERVER, AssetType.DATABASE,
                AssetType.MAIL_SERVER, AssetType.FILE_SERVER, AssetType.VPN_GATEWAY,
                AssetType.DOMAIN_CONTROLLER}


@dataclass
class ActionTemplate:
    cost: float
    time: float
    base_disruption: float
    effectiveness: float
    label: str


# Nominal attributes (normalised 0..1). Disruption is scaled by the target's
# business criticality and user count at generation time.
TEMPLATES: dict[ActionType, ActionTemplate] = {
    ActionType.ISOLATE_HOST: ActionTemplate(0.15, 0.10, 0.55, 0.95, "Isolate {t}"),
    ActionType.BLOCK_IP: ActionTemplate(0.05, 0.02, 0.02, 0.70, "Block malicious IP {t}"),
    ActionType.REVOKE_CREDENTIALS: ActionTemplate(0.10, 0.08, 0.20, 0.90, "Revoke credentials on {t}"),
    ActionType.DISABLE_ACCOUNT: ActionTemplate(0.05, 0.03, 0.10, 0.70, "Disable accounts on {t}"),
    ActionType.PATCH_VULNERABILITY: ActionTemplate(0.25, 0.60, 0.15, 0.85, "Patch {t}"),
    ActionType.SEGMENT_NETWORK: ActionTemplate(0.35, 0.40, 0.35, 0.85, "Segment {t}"),
    ActionType.INCREASE_MONITORING: ActionTemplate(0.05, 0.05, 0.00, 0.25, "Increase monitoring on {t}"),
    ActionType.PROTECT_DATABASE: ActionTemplate(0.20, 0.20, 0.10, 0.80, "Protect database {t}"),
    ActionType.BLOCK_PORT: ActionTemplate(0.05, 0.05, 0.20, 0.60, "Block exposed ports on {t}"),
    ActionType.QUARANTINE_ENDPOINT: ActionTemplate(0.10, 0.15, 0.10, 0.90, "Quarantine endpoint {t}"),
}


class ActionGenerator:
    def __init__(self, ag: AttackGraph, effectiveness: dict[ActionType, float] | None = None):
        self.ag = ag
        self.eff = {t: tpl.effectiveness for t, tpl in TEMPLATES.items()}
        self.eff.update(effectiveness or {})

    def _disruption(self, kind: ActionType, target: str) -> float:
        base = TEMPLATES[kind].base_disruption
        a = self.ag.assets.get(target)
        if a is None:
            return base
        user_factor = min(1.0, a.users / 300.0)
        return float(min(1.0, base * (0.4 + 0.6 * max(a.criticality, user_factor))))

    def _make(self, kind: ActionType, target: str, edge_effects, node_effects=None,
              label_target: str | None = None) -> DefenseAction | None:
        if not edge_effects and not node_effects:
            return None
        tpl = TEMPLATES[kind]
        return DefenseAction(
            id=f"{kind.value}:{target}",
            type=kind,
            target=target,
            description=tpl.label.format(t=label_target or target),
            cost=tpl.cost,
            time=tpl.time,
            disruption=self._disruption(kind, target),
            effectiveness=self.eff[kind],
            edge_effects=dict(edge_effects),
            node_effects=dict(node_effects or {}),
        )

    # ---- per-type builders ------------------------------------------------------
    def actions_for(self, v: str) -> list[DefenseAction]:
        g, a = self.ag.g, self.ag.assets[v]
        ins = list(g.in_edges(v, data=True))
        outs = list(g.out_edges(v, data=True))
        acts = [
            self._make(ActionType.ISOLATE_HOST, v,
                       {(x, y): 1.0 for x, y, _ in ins + outs}),
            self._make(ActionType.REVOKE_CREDENTIALS, v,
                       {(x, y): (1.0 if d["kind"] == "credential" else 0.5)
                        for x, y, d in outs if d["kind"] in ("credential", "lateral")}),
            self._make(ActionType.INCREASE_MONITORING, v,
                       {(x, y): 0.6 for x, y, _ in ins}, {v: 1.0}),
        ]
        if a.vulnerabilities:
            acts.append(self._make(ActionType.PATCH_VULNERABILITY, v,
                                   {(x, y): 1.0 for x, y, d in ins
                                    if d["kind"] in ("network", "entry", "lateral")},
                                   label_target=f"{v} ({', '.join(x.cve_id for x in a.vulnerabilities)})"))
        if a.type == AssetType.DATABASE:
            acts.append(self._make(ActionType.PROTECT_DATABASE, v,
                                   {(x, y): 1.0 for x, y, d in ins if d["kind"] == "credential"}))
        if a.type == AssetType.WORKSTATION:
            acts.append(self._make(ActionType.QUARANTINE_ENDPOINT, v,
                                   {(x, y): 1.0 for x, y, _ in outs}, {v: 1.0}))
            acts.append(self._make(ActionType.DISABLE_ACCOUNT, v,
                                   {(x, y): 1.0 for x, y, d in outs
                                    if d["kind"] in ("credential", "lateral")}))
        if a.type in SERVER_TYPES and any(d["kind"] in ("entry", "network") for _, _, d in ins):
            acts.append(self._make(ActionType.BLOCK_PORT, v,
                                   {(x, y): 1.0 for x, y, d in ins
                                    if d["kind"] in ("entry", "network")}))
        return [x for x in acts if x is not None]

    def block_ip(self, source: str) -> DefenseAction | None:
        effects = {(ATTACKER, v): 1.0 for v in self.ag.g.successors(ATTACKER)}
        return self._make(ActionType.BLOCK_IP, ATTACKER, effects, label_target=source or "attacker")

    def segment_zones(self, zone_a: str, zone_b: str) -> DefenseAction | None:
        assets = self.ag.assets
        effects = {(u, v): 1.0 for u, v in self.ag.g.edges
                   if u in assets and v in assets
                   and assets[u].zone == zone_a and assets[v].zone == zone_b}
        return self._make(ActionType.SEGMENT_NETWORK, f"{zone_a}->{zone_b}", effects,
                          label_target=f"{zone_a} -> {zone_b} boundary")

    # ---- candidate set ------------------------------------------------------------
    def generate(self, focus: list[str] | None = None, include_segmentation: bool = True,
                 include_block_ip: bool = True) -> list[DefenseAction]:
        """All candidate actions around ``focus`` hosts (default: threatened + critical)."""
        if focus is None:
            focus = sorted(set(self.ag.threat) | {n for n, a in self.ag.assets.items()
                                                  if a.criticality >= 0.85})
        actions: list[DefenseAction] = []
        for v in focus:
            if v in self.ag.assets:
                actions.extend(self.actions_for(v))
        if include_block_ip:
            src = next((t.source for t in self.ag.threat_info.values() if t.source), "")
            b = self.block_ip(src)
            if b:
                actions.append(b)
        if include_segmentation:
            zones = {(self.ag.assets[u].zone, self.ag.assets[v].zone)
                     for u, v in self.ag.g.edges if u in self.ag.assets and v in self.ag.assets}
            for za, zb in sorted(zones):
                if za != zb:
                    s = self.segment_zones(za, zb)
                    if s:
                        actions.append(s)
        seen, unique = set(), []
        for a in actions:
            if a.id not in seen:
                seen.add(a.id)
                unique.append(a)
        return unique
