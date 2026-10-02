# Q-ADAPT methodology

This document states the mathematical model that the code implements. Each section
names the module that implements it.

## 1. Threat assessment (`ml_engine/`)

Flows use the canonical CICFlowMeter features shared by CIC-IDS2017 and CSE-CIC-IDS2018
(23 features, see `datasets.CANONICAL_FEATURES`). Raw labels are mapped onto a unified
taxonomy: BENIGN, DOS, DDOS, BRUTE_FORCE, WEB_ATTACK, BOTNET, INFILTRATION, RECON,
EXPLOIT. The preprocessing is: clean → signed `log1p` → drop constant columns →
standardize.

For a flow *f* the classifier returns class probabilities. Maliciousness is
`m_f = 1 − P(BENIGN | f)`. For a host *h* with flows *F_h*:

```
p_h = mean(top-k m_f) · (1 − exp(−n_flagged / τ))        (k = 5, τ = 3, flag: m_f ≥ 0.5)
```

The attack type is the most frequent predicted class among flagged flows. Confidence is
the mean maximum class probability over those flows. The evidence term stops a single
suspicious flow from saturating the score.

## 2. Attack graph (`attack_graph/`)

`G = (V, E)` where `V` is the set of assets plus a virtual attacker node `a`. An edge
`(u, v)` means "an attacker controlling *u* can attempt to compromise *v*". Its
one-step success probability `p_uv` depends on the link kind and on the CVSS-derived
exploitability of *v*:

| kind | p_uv |
|---|---|
| entry / network | `e_v` |
| credential | `0.45 + 0.25·[u has privilege-requiring vuln] + 0.2·e_v` |
| lateral | `0.1 + 0.6·e_v` |
| phishing | `0.35` |

Here `e_v = 1 − Π_k (1 − expl_k)` is a noisy-OR over the asset's vulnerabilities, with
a floor of 0.05 for misconfiguration. Probabilities are clipped to [0.02, 0.98].

**Risk propagation.** Let `t_v` be the local ML evidence (`p_h · confidence`) and `A`
the attacker-activity prior (0.3, raised by detections). The compromise probabilities
are the least fixed point of

```
P(a) = A
P(v) = 1 − (1 − t_v·m_v) · Π_{(u,v)∈E} (1 − P(u) · p_uv · m_uv)
```

Here `m ∈ [0,1]` are the defense multipliers (Section 4). Iterating from `P = 0` is
monotone and converges on cyclic graphs. Each step is a sparse matrix product, so a
whole batch of portfolios is propagated at once.

**Network risk.** With criticality `C_v ∈ [0,1]`:

```
R = Σ_v C_v · P(v) / Σ_v C_v
```

The alternatives used in the ablation (`risk_engine/`) are:

* `multiplicative`: `R_v = T_v · V_v · C_v · P_v`, the proposal's formula
* `severity_only`: `R_v = t_v · C_v`, with no graph

## 3. Candidate defenses (`defense_engine/`)

Each action *i* has a cost `c_i`, a time `τ_i`, a business disruption `d_i` (scaled by
the target's criticality and user count), a nominal effectiveness `η_i`, and effect
weights `φ_ie ∈ [0,1]` on edges and `ψ_iv` on nodes. Examples:

* *isolate host* cuts every in- and out-edge
* *revoke credentials* cuts outgoing credential edges and halves lateral edges
* *patch* cuts incoming network edges
* *quarantine* cleans the node and cuts its out-edges

Candidates are generated around hosts with ML evidence, high-criticality assets,
entry points and zone boundaries. They are then **pre-screened** to the qubit budget by
the ratio of standalone risk reduction to `(0.5 + c + τ + d)`.

## 4. Portfolio effect model and ground-truth objective (`optimization/problem.py`)

For a portfolio `x ∈ {0,1}^n`:

```
m_e(x) = Π_i (1 − η_i φ_ie)^{x_i} = exp(x · L_e),   L_ie = log(1 − η_i φ_ie)
```

and likewise for node multipliers. The residual risk `R(x)` is obtained by propagating
with these multipliers. The **ground-truth objective** is

```
J(x) = α · R(x)/R(0) + β · cᵀx + γ · τᵀx + δ · dᵀx
```

It is subject to `cᵀx ≤ B`, `τᵀx ≤ T_max`, `dᵀx ≤ D_max`, `1ᵀx ≤ K`, plus the policy
constraints: conflicts `x_i x_j = 0`, prerequisites `x_i ≤ x_j`, forbidden `x_i = 0`
(e.g. isolating a protected asset) and mandatory `x_i = 1`. **Every solver is scored on
J**, not on its own internal model.

## 5. Defense QUBO (`quantum_engine/qubo_builder.py`)

`R(x)` is not quadratic, so the QUBO uses a quadratic pseudo-Boolean surrogate:

```
R(x)/R(0) ≈ 1 + Σ_i h_i x_i + Σ_{i<j} J_ij x_i x_j
```

* **expansion**: a second-order Möbius expansion at the empty portfolio,
  `h_i = r_i − r_0` and `J_ij = r_ij − r_i − r_j + r_0`. It is exact on singletons and
  pairs. `J_ij > 0` means the actions are **redundant** and `J_ij < 0` means they have
  **synergy**. This is how action interactions enter the optimization.
* **regression** (default): a ridge least-squares fit of the same form on portfolios
  sampled across the operating cardinality range, plus all singletons and pairs. It is
  more faithful for larger portfolios, where effects saturate.

Surrogate fidelity (R², Spearman, MAE against `R(x)` on held-out portfolios) is stored
in `qubo.meta["fidelity"]`. On the demo instance the regression surrogate reaches
Spearman ≈ 0.99 and R² ≈ 0.98, against ≈ 0.89 and 0.66 for the expansion.

**QUBO:**

```
Q(x) = α(1 + hᵀx + xᵀJx) + βcᵀx + γτᵀx + δdᵀx + P · [policy terms] + [inequality terms]
```

* conflicts: `P x_i x_j`
* prerequisites: `P x_i (1 − x_j)`
* forbidden: `P x_i`
* mandatory: `P (1 − x_i)`
* inequality `wᵀx ≤ L` uses one of two encodings:
  * `slack`: `P (wᵀx/L + Σ_b 2^b Δ s_b/L − 1)²`, with `⌈log₂⌉` slack qubits per
    constraint. It is exact but adds qubits.
  * `unbalanced` (default; Montañez-Barrera et al., 2022): with `ĥ = 1 − wᵀx/L`, the
    term is `P (−λ₁ ĥ + λ₂ ĥ²)`. It needs no extra qubits, rewards slack only boundedly
    and penalizes violations quadratically.
* By default `P = 2 · max_i(|lin_i| + Σ_j |Q_ij|)`, the largest single-flip swing, so no
  single bit flip into infeasibility can pay off.

## 6. Ising mapping and QAOA (`quantum_engine/ising.py`, `qaoa.py`)

Substituting `x_i = (1 − z_i)/2`, where bit 1 corresponds to Z eigenvalue −1 (the
Qiskit convention), gives `H_C = c + Σ h_i Z_i + Σ J_ij Z_i Z_j`.

```
|ψ(γ, β)⟩ = Π_{l=1..p} e^{−iβ_l Σ X_i} e^{−iγ_l H_C} |+⟩^{⊗n}
```

* **State-vector engine.** Applies the cost layer as a diagonal phase and the mixer as
  per-qubit RX via reshapes. The energy is rescaled to [0, 1]. Its probabilities match
  Qiskit's `QAOAAnsatz` + `Statevector` to ~1e-16 (`tests/test_quantum_engine.py`).
* **Qiskit/Aer backend.** Transpiles `QAOAAnsatz` once per depth, estimates
  expectations from shots, and optionally adds a gate-level depolarizing + readout
  noise model.
* **Outer loop.** COBYLA with restarts. For `p > 1` it uses INTERP initialization
  (Zhou et al., 2020), growing the depth one layer at a time. CVaR-α is optional
  (Barkoutsos et al., 2020). Warm starts reuse previous parameters.
* **Noise in the state-vector engine.** A global depolarizing channel with fidelity
  `F = (1−p₁)^{n_1q} (1−p₂)^{n_2q}` (gate counts with each ZZ decomposed as CX-RZ-CX),
  plus readout bit flips when sampling.
* **Decoding.** Bitstrings are sampled. Candidates are deduplicated, scored on `J`, and
  the best **feasible** portfolio is returned. A 1-flip local search is available but is
  reported separately (`polish=True`).
* **Metrics.** Approximation ratio `(E_max − ⟨E⟩)/(E_max − E_min)` over the QUBO,
  `P(optimal QUBO state)`, amplification over uniform sampling, circuit depth
  (edge-coloring schedule, or transpiled), number of qubits and function evaluations.

## 7. AQDO: adaptive quantum defense optimization (`adaptive_engine/`)

At each decision step *t*:

1. **Observe.** The ML engine's detections update `t_v`. High-confidence detections mark
   hosts as compromised.
2. **Learn.** Each attack attempt on a defended edge that *would* have succeeded without
   the defense is a Bernoulli trial for the defending action type. The posterior is
   `Beta(s·η⁰ + blocked, s·(1−η⁰) + not_blocked)`. The belief graph is then re-derived
   from base probabilities with the posterior-mean effectiveness.
3. **Re-weight.** `α_t = α₀ · min(1 + κ · max(0, R_t − R*)/R*, α_max)`, where R* is the
   risk appetite. Security dominates while risk exceeds the appetite.
4. **Regenerate** the candidates (excluding applied actions) and the QUBO from the new
   state.
5. **Solve** with QAOA, warm-started from the previous optimal `(γ, β)`.
6. **Approve.** The analyst approves or rejects; only approved actions are applied.
7. **Trigger.** Re-optimization happens on new detections or when `|ΔR| > θ`.

The evaluation environment (`AttackEnvironment`) keeps a hidden ground truth:

* stochastic attacker progression along edges
* **true** action effectiveness that can differ from nominal
* imperfect detection with false positives (or detection by the trained ML detector on
  synthetic flows)

Policies `none`, `static`, `reoptimize` and `aqdo` are compared by realized
criticality-weighted loss.

## 8. Known limitations

* Edge probabilities and action attributes are modelling assumptions. They are
  calibrated to CVSS, not to incident data.
* Simulating QAOA limits problems to about 20 qubits. Larger instances rely on
  pre-screening, which is a heuristic decomposition.
* The global-depolarizing approximation is coarse. Gate-level Aer runs are provided as a
  check.
* The surrogate is quadratic, so higher-order interactions are only fitted, not
  represented. The decoding step corrects for this by scoring against the true objective.
