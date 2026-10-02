"""Security-policy constraints on defense portfolios.

* conflicts      pairs that must not be selected together  (x_i * x_j = 0)
* prerequisites  i requires j                              (x_i <= x_j)
* forbidden      actions disallowed by policy             (x_i = 0)
* mandatory      actions required by policy               (x_i = 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qadapt.core.models import ActionType, DefenseAction

# Pairs of action types on the same target that are mutually exclusive.
EXCLUSIVE_SAME_TARGET = {
    frozenset({ActionType.ISOLATE_HOST, ActionType.QUARANTINE_ENDPOINT}),
    frozenset({ActionType.ISOLATE_HOST, ActionType.BLOCK_PORT}),
    frozenset({ActionType.ISOLATE_HOST, ActionType.PATCH_VULNERABILITY}),
}
DISRUPTIVE = {ActionType.ISOLATE_HOST, ActionType.QUARANTINE_ENDPOINT, ActionType.BLOCK_PORT}


@dataclass
class PolicySet:
    conflicts: set[tuple[int, int]] = field(default_factory=set)
    prerequisites: set[tuple[int, int]] = field(default_factory=set)  # (i, j): i requires j
    forbidden: set[int] = field(default_factory=set)
    mandatory: set[int] = field(default_factory=set)

    def violations(self, x: np.ndarray) -> list[str]:
        x = np.asarray(x).round().astype(int)
        v = [f"conflict({i},{j})" for i, j in self.conflicts if x[i] and x[j]]
        v += [f"requires({i},{j})" for i, j in self.prerequisites if x[i] and not x[j]]
        v += [f"forbidden({i})" for i in self.forbidden if x[i]]
        v += [f"mandatory({i})" for i in self.mandatory if not x[i]]
        return v

    def to_dict(self) -> dict:
        return {"conflicts": sorted(self.conflicts), "prerequisites": sorted(self.prerequisites),
                "forbidden": sorted(self.forbidden), "mandatory": sorted(self.mandatory)}


def build_policy(actions: list[DefenseAction], protected_assets: tuple[str, ...] = (),
                 extra_conflicts: list[tuple[str, str]] | None = None,
                 extra_prerequisites: list[tuple[str, str]] | None = None,
                 mandatory: list[str] | None = None) -> PolicySet:
    """Derive the default policy and add operator-specified rules (by action id)."""
    pos = {a.id: i for i, a in enumerate(actions)}
    pol = PolicySet()
    for i, a in enumerate(actions):
        for j in range(i + 1, len(actions)):
            b = actions[j]
            if a.target == b.target and frozenset({a.type, b.type}) in EXCLUSIVE_SAME_TARGET:
                pol.conflicts.add((i, j))
        if a.target in protected_assets and a.type in DISRUPTIVE:
            pol.forbidden.add(i)
    for u, v in extra_conflicts or []:
        if u in pos and v in pos:
            pol.conflicts.add(tuple(sorted((pos[u], pos[v]))))
    for u, v in extra_prerequisites or []:
        if u in pos and v in pos:
            pol.prerequisites.add((pos[u], pos[v]))
    for m in mandatory or []:
        if m in pos:
            pol.mandatory.add(pos[m])
    return pol
