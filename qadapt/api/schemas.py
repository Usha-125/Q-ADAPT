"""Request/response models for the Q-ADAPT API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    scenario: Literal["demo", "small", "medium", "large"] = "demo"
    seed: int = 0
    with_initial_threats: bool = True


class ThreatIn(BaseModel):
    host_id: str
    attack_type: str = "EXPLOIT"
    probability: float = Field(0.9, ge=0, le=1)
    confidence: float = Field(0.9, ge=0, le=1)
    source: str = ""


class DetectRequest(BaseModel):
    attacked: dict[str, str] = Field(default_factory=dict, description="asset id -> attack category")
    n_flows: int = Field(600, ge=50, le=20000)
    seed: int = 0


class WeightsIn(BaseModel):
    alpha: float = 1.0
    beta: float = 0.15
    gamma: float = 0.10
    delta: float = 0.20


class OptimizeRequest(BaseModel):
    solver: str = "qaoa"
    p: int = Field(2, ge=1, le=5)
    noise: Literal["ideal", "low", "medium", "high"] = "ideal"
    backend: Literal["statevector", "qiskit"] = "statevector"
    shots: int = Field(1024, ge=64, le=100000)
    max_qubits: int = Field(12, ge=2, le=20)
    budget: float | None = 0.5
    max_time: float | None = None
    max_disruption: float | None = None
    max_actions: int | None = None
    encoding: Literal["unbalanced", "slack"] = "unbalanced"
    surrogate: Literal["regression", "expansion"] = "regression"
    weights: WeightsIn = Field(default_factory=WeightsIn)
    protected_assets: list[str] = Field(default_factory=list)
    seed: int = 0


class CompareRequest(OptimizeRequest):
    solvers: list[str] = Field(default_factory=lambda: [
        "qaoa", "exhaustive", "milp", "greedy", "simulated_annealing", "genetic",
        "score_ranking", "random_sampling"])


class DecisionIn(BaseModel):
    action_ids: list[str] | None = None  # None = all recommended actions
    decision: Literal["approve", "reject"] = "approve"
    analyst: str = "analyst"
