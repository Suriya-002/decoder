# PROJECT HANDOFF — Calibration-Free Estimation of CZ-Pair Atom-Loss Correlation

**Purpose of this document.** This is a complete cold-start briefing. Paste it into a fresh session
and the assistant should be able to continue the project without any other context. It contains every
result, every essential number, every design decision, every mistake made and how it was caught, and
every open item. Written 17 July 2026.

**Repository:** `D:\Decoder` (Windows, PowerShell primary). Python venv at `.venv`. Dependencies:
`stim>=1.16`, `pymatching>=2.4`, `numpy>=2.0`, plus `scipy` for the superseded substep estimator.
All shell instructions must be given as copy-pasteable PowerShell.

---

## 1. ORIGIN, AND THE THREE PIVOTS

The project began as a PhD-application research direction: build a neural decoder (graph neural
network) for **correlated atom loss from Rydberg gates in neutral-atom surface codes**, reproduce
Wang et al. (arXiv:2604.14269), add a correlation parameter to the noise model, and beat their
reported loss-identification recall of 0.654. Estimated at four months with GPU training.

That plan was destroyed and replaced three times, each time by a measurement:

**Pivot 1 — the mechanism in the original brief was physically wrong.** The brief claimed that
"flicker on atom A is direct Bayesian evidence about its gate partner B," implying data–data CZ
pairing. In a surface-code syndrome-extraction circuit **there are no data–data two-qubit gates**;
every CZ is data↔ancilla. Verified directly: 80 two-qubit gates on d=5, 80 unique Tanner edges,
zero violations. The correct mechanism is that the co-lost partner is an **ancilla**, whose loss is
transient (refreshed by its own `MR` each round) rather than persistent.

**Pivot 2 — the central thesis was falsified.** Knowing the correlation parameter η does **not**
meaningfully improve loss *identification*. Measured three ways (Section 4). The four-month neural
decoder project was therefore pointless, and this was established in roughly one day with zero GPU
hours.

**Pivot 3 — what replaced it.** η itself is **estimable in closed form from the raw measurement
record**, with no loss-resolving readout, no teleportation-based loss-detection unit, and no
dedicated calibration experiment. Nobody in the literature does this. Perrin *assumes* η; Liu
*assumes* η; Wang does not model it at all.

---

## 2. PHYSICS SETUP AND NOTATION

The system is a rotated surface code memory experiment (later generalized to other codes), generated
by `stim.Circuit.generated("surface_code:rotated_memory_z", ...)`. The layout is **parsed out of
Stim's own generated circuit** rather than hand-rolled, so the geometry and the hook-error-avoiding
CZ schedule are Stim's known-good ones.

Standard configuration used throughout unless stated: **d=5, T=12 rounds, p=0.002** (uniform
circuit-level depolarizing: `after_clifford_depolarization`, `before_round_data_depolarization`,
`before_measure_flip_probability`, `after_reset_flip_probability`), **p_g = 0.00054** (Evered's
measured 0.054% atom loss per atom per CZ gate).

On d=5 rotated: 25 data qubits, 24 ancillas (12 X-type, 12 Z-type), 80 CZ gates per round arranged
in **4 substeps** of 20 gates each. Each qubit participates in at most one gate per substep. Detector
words are 13 bits for the memory-basis ancilla type (includes the final destructive readout) and 11
bits for the other.

**η is defined as** P(partner atom lost | this atom lost) within a single CZ gate.

---

## 3. THE LOSS MODEL — THREE ARMS WITH MATCHED MARGINALS

This is the most important design decision in the entire project and it must not be altered casually.

| Arm | Model | Source |
|---|---|---|
| **A** | round-level i.i.d. loss | Wang et al. 2604.14269 — each physical qubit has probability P_loss of being removed **in each round** |
| **B** | gate-level, η = 0 (independent within CZ) | Perrin/Jandura/Pupillo 2412.07841 |
| **C** | gate-level, η > 0 (CZ-pair correlated) | Evered 2604.25987 physics |

Per CZ gate on atoms (a, b):

```
P(both lost)   = η · p_g
P(only a lost) = P(only b lost) = (1 − η) · p_g
⇒ marginal P(a lost at this gate) = p_g, INDEPENDENT OF η
```

**Sweeping η therefore changes only the correlation, never the total loss budget.** Without matched
marginals, an η sweep confounds correlation with loss rate and every result becomes uninterpretable.
Verified: measured per-atom loss rate 0.00995 / 0.00998 / 0.01004 / 0.00992 at η = 0.00 / 0.25 / 0.50
/ 1.00 against a target of 0.01; co-lost atom fraction 0.0000 / 0.5051 / 1.0000 at η = 0.0 / 0.5 / 1.0.

**Arm A is NOT the η=0 limit of arms B/C.** It is a different loss *mechanism* — round-level rather
than gate-level. It exists only as an external reproduction anchor for Wang. Comparing a gate-level
η sweep against Wang changes two things at once and the separation becomes unattributable.

Note that strict independence corresponds to η = p_g (giving P(both) = p_g²), not η = 0. At
p_g ≈ 1e-3 the difference is negligible but the harness is exact about it.

---

## 4. RESULT 1 — THE CEILING. THE RECALL THESIS IS DEAD.

`scripts/run_ceiling.py`. Bayes-optimal AUC for detecting a data-qubit loss, computed as a
likelihood-ratio test over the exact empirical distribution of detector patterns on the victim's
Tanner-partner ancillas. No architecture can beat these numbers on the same window.

| window | arm B (η=0) | arm C (η=1) | Δ |
|---|---|---|---|
| final-round loss, **without** final readout (memory_Z) | 0.8426 | 0.8098 | **−0.033** |
| final-round loss, **with** final readout (memory_Z) | 0.8534 | 0.8874 | **+0.034** |
| final-round loss, with final readout (memory_X) | 0.9194 | 0.9120 | −0.008 |
| early loss + 2 rounds of flicker (memory_Z) | 0.9871 | 0.9872 | **+0.000** |
| early loss + 2 rounds of flicker (memory_X) | 0.9870 | 0.9884 | +0.001 |

**Reading:** the effect is ±0.03 AUC, the *sign flips* depending on whether the final destructive
readout is in the window, and it goes to **exactly zero** once post-loss flicker has time to
accumulate. A pair-aware decoder will not produce a headline recall improvement. This is what killed
the original project.

**What survives from the ceiling:** AUC(arm C vs arm B) = **0.72–0.75**, stable across basis and
window (0.7243 / 0.7465 / 0.7179 / 0.6111 / 0.5645 across configurations, with the strongest values
where flicker is available). That is the identifiability of η itself, and it is the one number that
is large, stable, and survived every subsequent correction.

**Critical implementation note:** Stim places the final destructive data-readout detectors at
**t = T**, not t = T−1. An early version of the window filter used `r < T` and silently excluded
them, which is what produced the sign flip when corrected. The `--no-final` flag reproduces the
buggy-window behaviour for comparison.

---

## 5. RESULT 2 — THE MECHANISM, AND THE ANCILLA-TYPE BIMODALITY

`scripts/run_eta_probe.py`. Oracle-conditioned: the probe is handed the ground-truth loss list and
asked what the partner ancilla's detector did in the loss round.

At η=0 (partner always alive): **0.4996 ± 0.0154**. This is not fitted — theory pins it at exactly
0.5, because a data qubit that is traced out leaves truncated X- and Z-stabilizers that anticommute
and collapse to ±1 with equal probability. The harness reproducing 0.4996 is a free known-answer
check and it lands.

At η=1 (partner always co-lost): **0.2878 ± 0.0139**, n = 4062. At η=0.5: co-lost 0.3065 ± 0.0207,
alive 0.4702 ± 0.0220.

**Then the pooled number turned out to be hiding the actual finding.** Splitting by ancilla type:

**memory_Z:**

| ancilla type | partner ALIVE | partner CO-LOST | signal |
|---|---|---|---|
| **Z-type** (deterministic sign) | 0.4718 ± 0.0173 | **0.1052 ± 0.0107** | **+0.3666** |
| X-type (randomly projected sign) | 0.4995 ± 0.0185 | 0.5090 ± 0.0184 | **−0.0095** |

**memory_X** (mirrored exactly):

| ancilla type | partner ALIVE | partner CO-LOST | signal |
|---|---|---|---|
| Z-type | 0.4921 ± 0.0183 | 0.4931 ± 0.0181 | −0.0010 |
| **X-type** | 0.4841 ± 0.0173 | **0.1097 ± 0.0109** | **+0.3743** |

**Half the array carries the entire signal; the other half carries none.** The pooled 0.289 was an
average over a bimodal population, and pooling had been discarding the whole structure.

**Why:** in memory_Z the Z-stabilizers are deterministic, so the ancilla reads 0 in steady state and
a dead ancilla's forced 0 gives XOR(0,0) = 0 → suppressed to 0.11. The X-stabilizers were projected
to a *random but fixed* sign b in round 0, so a dead ancilla gives XOR(0, b) = b → a coin flip. **The
detector's XOR destroys the signal on the randomly-projected half.**

---

## 6. RESULT 3 — THE RAW MEASUREMENT RECORD AND THE CLOSED FORM

`scripts/run_raw_anchors.py`, `scripts/run_eta_raw.py`. Going below the detector abstraction to
Stim's raw measurement record (`compile_sampler()` rather than `compile_detector_sampler()`):

| | P(m=0) ALIVE | P(m=0) CO-LOST | signal |
|---|---|---|---|
| memory_Z, Z-anc | 0.5221 | **1.0000** | +0.478 |
| memory_Z, X-anc | 0.4897 | **1.0000** | +0.510 |
| memory_X, Z-anc | 0.4960 | **1.0000** | +0.504 |
| memory_X, X-anc | 0.5186 | **1.0000** | +0.481 |

**In 12,473 co-lost events the raw measurement was never once 1.** Signal is +0.48 to +0.51
*everywhere*, versus the detector record's +0.37 on half the array and *exactly zero* on the other.

### The estimator

```
P(m=0 | ALIVE partner on a truncated stabilizer) = 0.5        ← THEORY (steady state only)
P(m=0 | CO-LOST partner)                         = 1 − ε      ← THEORY + measurable hardware ε

⇒   η̂ = (P(m=0) − 0.5) / (0.5 − ε)
```

ε is the **false-bright rate** of a lost atom (dark counts, stray scatter). It is not a fudge factor:
it is measured on hardware by reading out an empty trap.

**High-statistics validation of the 0.5 anchor** (forced injection, no loss sampling, 9.6M samples):
**0.49998 ± 0.00032**, −0.1σ. Every stratum within ±1σ: degree-4 bulk 0.50006 ± 0.00047; degree-3
0.49987 ± 0.00047; degree-2 corner 0.50017 ± 0.00100; Z-anc 0.49989; X-anc 0.50007; substeps 0.50009
/ 0.50001 / 0.49969 / 0.50014.

**End-to-end recovery** at physical p_g = 0.00054, Rule 1 applied, ~14k events per point:

| η_true | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| **η̂** | −0.005 | 0.249 | 0.507 | 0.759 | 1.000 |

Independent runs gave −0.010 / 0.245 / 0.488 / 0.763 / 1.000 and −0.006 / 0.249 / 0.507 / 0.759 /
1.000. All points inside their 95% CIs.

---

## 7. RESULT 4 — THE GAUGE ARGUMENT (the intellectual content)

**Detectors are gauge-invariant. That is exactly why the field uses them, and exactly why they fail
for atom loss.**

A stabilizer's absolute value is fixed by a random projection in round 0 and carries no information
about Pauli errors — only *changes* do. The detector (XOR of consecutive rounds) quotients that gauge
away. Correct, and optimal, for Pauli noise.

**But a lost atom forces m = 0 regardless of the gauge. Loss breaks the gauge.** So the XOR that
defines a detector is precisely the operation that destroys the loss signature — and it destroys it
completely on whichever half of the array carries a randomly-projected stabilizer sign.

Reduced to code-agnostic form: a dead ancilla's detector fires with probability P(m(r−1) = 1) =
P(the stabilizer sign is randomly projected). Deterministic-sign stabilizer → detector blind to the
dead ancilla; projected stabilizer → detector fires 0.5, carrying no information. The raw record
reads 0 in both cases.

---

## 8. RESULT 5 — LOCALIZER SWEEP: NO NEURAL DECODER IS NEEDED

`scripts/run_localizer_sweep.py`. The estimator needs a front end to identify loss events. This
sweeps front-end quality, substep-blind (a real localizer gives (qubit, round) but not the CZ
substep, so all ≤4 partners must be averaged, diluting the oracle swing ~4×), with false positives
and timing errors both simulated. False positives bias η̂ **up** (a phantom loss looks like
suppression); timing errors bias η̂ **down** (a mistimed loss looks like a live partner). Both are in.

σ_η at 15k shots, with 1/√N projections:

| localizer (recall / precision) | slope *b* | σ_η @15k | @100k | @1M |
|---|---|---|---|---|
| oracle (1.00 / 1.00) | −0.0497 | 0.059 | 0.023 | **0.007** |
| **Wang STGNN (0.654 / 0.845)** | −0.0392 | 0.083 | 0.032 | **0.010** |
| crude (0.40 / 0.60) | −0.0280 | 0.115 | 0.045 | **0.014** |
| very crude (0.25 / 0.40) | −0.0163 | 0.187 | 0.072 | **0.023** |

**A localizer that is wrong three times out of four costs only a factor of 2–3.** A QEC memory run
routinely collects ≥10⁶ shots. η is measurable to ±0.02 with a crude flicker threshold. **No STGNN,
no PyTorch Geometric, no GPU.**

Note `obs(0) = 0.3385`, not 0.50, because partners whose CZ fired *before* the loss substep were
never truncated that round and sit at background (~0.03–0.04). That mix ratio across the four
partners encodes the substep — recovering it would restore the full −0.21 oracle swing, roughly 4×
tighter η. This is an unexploited upside.

---

## 9. RESULT 6 — RULE 1: ROUND 0 IS NOT 0.5

Seven independent replicates of P(m=0 | ALIVE) all landed **above** 0.5 (0.5159, 0.5067, 0.5003,
0.5022, 0.5091, 0.5086, 0.5130). The pooled deviation was only +1.7σ and was initially dismissed as
a fluctuation. **That was wrong — the sign consistency was the signal, not the magnitude.**
Seven-for-seven is p ≈ 0.008 under the null.

Sweeping the loss round (1.6M samples per cell):

| loss round | Z-type partner | X-type partner |
|---|---|---|
| **0** | **0.6684 ± 0.0007** | 0.4998 ± 0.0008 |
| 1 | 0.4999 | 0.4997 |
| 2–11 | **0.5000** (all) | **0.5000** (all) |

In round 0 the data is still |0…0⟩ — a Z **product state**. A truncated Z-stabilizer on a product
state is **still deterministic**, so it reads 0 with high probability, not 0.5. The claim "a truncated
stabilizer anticommutes and flickers at 0.5" is a **steady-state** claim only.

Arithmetic closes: (1/12 of losses in round 0) × (1/2 have Z-type partners) × (0.668 − 0.500) × 2 =
**+0.007 predicted**; observed pooled bias **+0.008**.

**RULE 1: drop round-0 losses.** η̂ bias at η=0 falls from 0.023 to 0.009.

---

## 10. RESULT 7 — RULE 2: THE PERSISTENCE ARTIFACT

Three hypotheses for the residual bias were proposed and all three were **falsified by measurement**
before the real mechanism was found:

- *"Partner ancilla dead via a different gate"* (rate ≈ 3·p_g). Real, but explains only ~35%.
  At p_g=0.010: no exclusion 0.0720, excluding partner-death 0.0467, also excluding
  other-neighbour multiloss 0.0451.
- *"Other-neighbour truncation."* Adds almost nothing beyond the above.
- *"Duplicate late-round entries of already-lost qubits."* Onset-only restriction gave 0.0686 vs
  0.0702 — no effect.

Conditioning directly on ground truth instead of guessing, at p_g = 0.01, bucketing ALIVE-classified
events by how many of the stabilizer's data neighbours were lost that round:

| # neighbours lost | n | P(m=0) | deviation |
|---|---|---|---|
| **1 (clean)** | 39,296 | **0.5242 ± 0.0049** | **+9.6σ** |
| 2 | 3,441 | 0.5376 ± 0.0167 | +4.4σ |
| 3 | 139 | 0.7122 ± 0.0753 | +5.5σ |

Even the clean k=1 bucket is badly biased, so it was never multi-loss contamination.

**The real mechanism: the 0.5 anchor holds ONLY in the round the data qubit FIRST goes missing on
that stabilizer.** After the truncated outcome is recorded, the surviving qubits are projected and
the stabilizer reads deterministically thereafter. Data loss is persistent, so a naive per-round scan
counts the same stabilizer in many later rounds where it is no longer 0.5. Same determinism physics
as round 0, arriving via persistence instead of initialization.

**RULE 2** (`src/decoder/events.py::clean_events`): take only the **first truncated round** of each
stabilizer; require the ancilla itself alive that round; require exactly one newly-lost data qubit
on it.

η_true = 0, so η̂ must be 0:

| p_g | NAIVE η̂ | CLEAN η̂ |
|---|---|---|
| **0.00054** (Evered's rate) | −0.0118 ± 0.0164 | 0.0085 ± 0.0169 |
| 0.002 | 0.0083 ± 0.0125 | 0.0107 ± 0.0136 |
| 0.005 | 0.0353 ± 0.0112 *(biased)* | −0.0047 ± 0.0136 |
| 0.010 | 0.0590 ± 0.0098 *(biased)* | 0.0036 ± 0.0143 |
| 0.020 | 0.1570 ± 0.0092 *(biased)* | 0.0027 ± 0.0181 |

The naive bias scales with p_g; the measured δ (partner dead via another gate) tracks 3·p_g exactly
(0.00139 vs 0.0016; 0.00506 vs 0.0060; 0.01285 vs 0.0150; 0.02653 vs 0.0300).

### CRUCIAL SCOPE LIMIT ON `clean_events`

It keeps only clean ALIVE onsets — **the partner is alive by construction** — so it deliberately
drops co-loss events, which *are* the η signal. It is an **anchor / η=0 verification tool, NOT an
η>0 estimator.** An attempt to use it as the estimator returned ≈0 for every η and was caught by the
known-answer test within two runs. **The production estimator uses the naive per-round collector with
Rule 1 applied**, which is unbiased at the physical loss rate; Rule 2's bias only appears above
roughly 10× hardware p_g.

---

## 11. RESULT 8 — REALISM: NEUTRAL-ATOM PHYSICS

`scripts/run_realistic_noise.py`. Added the two things a neutral-atom referee checks:

- **Survivor biased-Z** (Perrin 2412.07841 §8): when exactly one atom of a CZ pair is lost, the
  surviving partner suffers a maximally biased Z error (`Z_ERROR` p = 0.5).
- **Evered residual Pauli budget** (2604.25987): the CZ error decomposes as **0.061% loss, 0.035%
  leakage, 0.024% Pauli-Z, 0.003% Pauli-X(Y)**. The Pauli-Z/XY parts applied to every surviving CZ.
- **False-bright ε** on dead-atom readout.

Both anchors are **immune**:

| model | P(m=0 \| alive) | P(m=0 \| co-lost) |
|---|---|---|
| baseline | 0.5006 | 1.0000 |
| + survivor biased-Z | 0.5000 | 1.0000 |
| + Evered Pauli budget | 0.4999 | 1.0000 |
| + ε = 0.01 | 0.4995 | 0.9900 (= 1 − ε) |

**Why:** the survivor-Z acts on the atom that *stays*; it cannot touch the *lost* atom's ancilla
reading, which is the only observable the estimator uses.

ε robustness sweep, end-to-end:

| ε | η=0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| 0.000 | 0.012 | 0.250 | 0.517 | 0.750 | 1.000 |
| 0.005 | 0.020 | 0.264 | 0.507 | 0.761 | 0.998 |
| 0.020 | 0.005 | 0.272 | 0.504 | 0.740 | 0.996 |
| **0.050** | 0.015 | 0.247 | 0.490 | 0.735 | 0.990 |

At η=0.5, high statistics: refined physics **0.4979 ± 0.0125** vs clean model 0.4931 ± 0.0124 —
statistically identical.

**Implementation caveat:** the refined physics uses `fast.build_fast`'s experimental
`survivor_z` / `pz` / `pxy` arguments, which are **NOT byte-verified against the reference builder**
(the reference does not implement them). Validated only by the forced-injection anchor test and the
high-statistics run. With all three at 0 (default) the fast builder *is* byte-identical to the
reference and that is checked.

---

## 12. RESULT 9 — GENERALITY OF THE GAUGE LAW

`scripts/run_generality.py`. Forced-dead ancilla at round 6, memory_Z, noiseless:

| code | d | anc type | stabilizer | raw P(m=0) | detector fires |
|---|---|---|---|---|---|
| surface rotated | 3, 5, 7 | Z | deterministic | 1.0000 | **0.0000** |
| surface rotated | 3, 5, 7 | X | projected | 1.0000 | ~0.50 |
| surface **unrotated** | 5 | Z / X | det / proj | 1.0000 | 0.0000 / ~0.50 |
| **color code** (memory_xyz) | 5 | Z | **projected** | 1.0000 | **0.50** |

**The color code is the sharpest confirmation.** Its Z-type ancilla measures a *projected* stabilizer
in this basis — verified by steady-state measurement: surface-code Z-anc ⟨m⟩ = **0.000**
(deterministic), color-code Z-anc ⟨m⟩ = **0.498** (projected). So its detector fires 0.50 and the
*opposite* type is blind.

**The law is not "X-type is blind." It is "the projected-stabilizer type is blind, whichever it is."**
The label flips with the code and basis; the law does not. Raw P(m=0) = 1.0 on every code.

---

## 13. RESULT 10 — ESTIMATOR GENERALITY BEYOND THE ROTATED SURFACE CODE

`scripts/run_estimator_codes.py`. The harness originally hardcoded 4 CX layers per round
(`divmod(cx_count, 4)`). Removed: `genlayout.parse_any` detects the CX-layer period per code and
`fast.set_n_sub` applies it.

| code | data / anc | n_sub | anchor alive | anchor co-lost | η recovery |
|---|---|---|---|---|---|
| rotated surface (baseline) | 25 / 24 | 4 | 0.500 | 1.000 | in CI |
| unrotated surface | 41 / 40 | 4 | 0.5005 | 1.0000 | in CI at 0 / .25 / .5 / .75 / 1 |
| **color code** | 19 / 9 | **6** | 0.5009 | 1.0000 | in CI at 0 / .5 / 1 |

Unrotated η̂: −0.006 / 0.276 / 0.488 / 0.747 / 1.000 at 12k shots. The 0.276 was a 2.6σ fluctuation;
at 30k shots it reads **0.2456 ± 0.0110** against η=0 giving −0.0076 ± 0.0113. Color η̂: 0.013 /
0.508 / 1.000.

**`set_n_sub` is a global-state footgun.** It must be restored to 4 after use or downstream checks
silently corrupt. The byte-identical-builder check runs after the generality checks and serves as
the guardrail.

---

## 14. RESULT 11 — HEAD-TO-HEAD vs THE FIELD'S ESTIMATORS

`scripts/run_pij_baseline.py`. Three representations of the same co-loss discrimination:

| representation | Z-type gap | X-type gap |
|---|---|---|
| **RAW** m(r) | 0.50 | **0.49** |
| 1-step detector m(r) ⊕ m(r−1) [BKY / Spitz / Google p_ij] | 0.38 | **0.003** |
| 2-step m(r) ⊕ m(r−2) [Bultink leakage syndrome] | 0.39 | **0.008** |

**Both XOR-based constructions are gauge-blind on the projected type.** Bultink's two-step leakage
syndrome references round r−2, still a projected round, so the forced 0 is XORed against a random
value exactly as the one-step is. Only the raw record works on both types.

The BKY p_ij estimator (`src/decoder/pij.py`, their Eq. 36:
`p_ij = 1/2 − 1/2·√((⟨z_i⟩⟨z_j⟩)/⟨z_i z_j⟩)` with `z = (−1)^detector`) was implemented faithfully
and validated against planted correlations: planted 0.05 → recovered 0.0504; 0.12 → 0.1197;
0.03 → 0.0300; 0.00 → −0.0005.

**HONEST SCOPE — do not overclaim.** A raw detector-**pair** p_ij does **not** cleanly "track η." A
co-lost data qubit's persistent flicker dominates any detector pair equally at every η, swamping the
co-loss-specific term; measured per-pair deltas were −0.008 and +0.012, i.e. noise. This confound is
*why* a bespoke raw-record estimator is needed, and it is why the **representation gap** — not a p_ij
number — is the result. Positioning against BKY is "the leading general method has a nameable blind
spot for this error class, and here is the physics," **not** "our estimator beats theirs."

---

## 15. LOSS-INJECTION IMPLEMENTATION (must be preserved exactly)

Loss is **not** a Pauli channel. Implemented in `src/decoder/circuit.py` (reference) and
`src/decoder/fast.py` (38× faster, byte-identical):

- At the loss moment the atom is traced out: **`DEPOLARIZE1(0.75) q`**. The uniform Pauli twirl *is*
  the completely depolarizing channel, i.e. exactly "trace out q and replace with maximally mixed."
- Every subsequent operation touching q is **erased** until reload. A lost atom carries out no
  entangling gate on its surviving partner.
- The atom's readout returns **0**, because atom-array readout detects |1⟩ population, so "no atom"
  is indistinguishable from |0⟩. Forced with an `R` (or `RX` in the X basis) before the `M`.
- Ancillas are reloaded by their own `MR` each round → **ancilla loss is transient**.
- Data qubits are never reloaded → **data loss is persistent** (flicker).
- **Measurements are NEVER deleted**, so DETECTOR `rec[]` offsets stay valid.

**Basis handling is a landmine.** `memory_z` uses `R`/`M`/`MR`; `memory_x` uses `RX`/`MX`/`MRX`. An
early version had `MX` in the one-qubit filter list, which *deleted* the final measurement of a lost
data qubit and shifted every downstream `rec[]` offset, corrupting every memory_X result. The
mapping `_MEAS = {"M":"R", "MR":"R", "MX":"RX", "MRX":"RX", "MY":"RY", "MRY":"RY"}` and
`_MEAS_RELOADS = {"MR","MRX","MRY"}` must cover every measurement op.

**Performance:** reference builder 14 shots/sec (67.8 ms per circuit build, dominated by
`targets_copy()` called ~500× per shot). Fast builder pre-renders the base circuit once into plain
tuples and splices strings per shot: **559 shots/sec, 38× faster**. The `DEPOLARIZE1` targets must be
**sorted** or the two builders diverge.

---

## 16. THE CHECK SUITE — 30 KNOWN-ANSWER CHECKS

`scripts/run_checks.py`. **Must all pass before any result is trusted.** Reproduced cross-platform
(Python 3.12/Linux ↔ Python 3.11/Windows, stim 1.16.0).

1. Noiseless + lossless ⇒ exactly 0 detector clicks.
2. No-loss rebuild is byte-identical to Stim's flattened base circuit (DEM 3717 errors both sides).
3. Every two-qubit gate is data↔ancilla — 80 gates, 80 unique Tanner edges, 0 violations.
4–6. One lost data qubit flickers *exactly* its Tanner-partner ancillas (~0.5), zero leakage
   elsewhere. Bulk q=27 → partners {26,28,37,39}; corner q=1 → {2,13}; q=9 → {19,21}; max
   non-partner rate 0.0000.
7–10. Marginal per-atom loss rate independent of η.
11–13. Co-lost atom fraction == η exactly.
14–15. Loss never changes the measurement or detector count, in *either* basis (313 measurements,
   288 detectors). **This is the check that catches the MX class of bug.**
16–17. Flicker check in both bases.
18–19. Fast builder byte-identical to reference, 200/200 random loss configs at random η, both bases.
20–23. The two theory anchors via **forced injection at a bulk round** (r=6), at ε=0 and ε=0.02.
   320k samples, 4σ tolerance: alive 0.50059 ± 0.00173, co-lost 1.00000 / 0.98005.
24–25. `clean_events` removes the naive Rule-2 bias — stated as a **ratio** (clean ≥3× closer to 0
   than naive) rather than a knife-edge tolerance. Measured 10× and 173× closer.
26–28. p_ij recovers planted p12 (BKY Eq. 36).
29–30. Generality (unrotated): dead ancilla raw = 1.0, detector ~0.0 on deterministic type and ~0.5
   on projected type. Plus estimator generality: n_sub detected == 4 (unrotated) and 6 (color),
   anchors 0.5 / 1.0 on both.

**Two checks were originally flaky and were fixed.** The anchor check sampled loss configs (which
include anomalous round-0 losses, putting a ~+0.007 bias right at a 3σ tolerance) — replaced with
deterministic forced injection. The Rule-2 check used a fixed tolerance near the noise floor —
replaced with a ratio assertion. **A check that passes 80% of the time is worse than no check.**
Principle: *don't sample when you can force the known answer deterministically.*

---

## 17. CODEBASE MAP

**`src/decoder/`**

| file | role |
|---|---|
| `layout.py` | rotated surface code layout parsed from Stim's generated circuit; `make_base`, `parse_layout`, `detector_map` |
| `circuit.py` | **reference** loss injector `build_with_loss`. Audited, slow, correct. Everything is checked against it. |
| `fast.py` | `prerender` + `build_fast`, 38× faster string splicer. Byte-identical to reference (checked). Experimental `survivor_z`/`pz`/`pxy` refined-physics args. `set_n_sub` global. |
| `loss_models.py` | the three arms; `sample_round_iid` (Wang), `sample_gate` (B and C, matched marginals), `loss_truth_masks` |
| `raw.py` | `measurement_index`, `eta_from_counts`; the two Rules documented in the docstring |
| `events.py` | `clean_events` — Rule 1 + Rule 2. **Anchor-only, not an η>0 estimator.** |
| `pij.py` | BKY / Spitz p_ij Eq. 36 baseline |
| `genlayout.py` | `parse_any` — code-general layout with CX-period detection |
| `codeagnostic.py` | `parse_generic`, `raw_measurement_index`, `detector_index`, `force_ancilla_dead` |
| `substep.py` | **SUPERSEDED** detector-record estimator; kept as honest research log |

**`scripts/`** — `run_checks.py` (run first, always), `run_ceiling.py`, `run_eta_probe.py`,
`run_eta_raw.py` (production estimator; `--pg-scan` for the Rule-2 proof), `run_raw_anchors.py`,
`run_localizer_sweep.py`, `run_pij_baseline.py`, `run_generality.py`, `run_realistic_noise.py`,
`run_estimator_codes.py`, `run_substep_eta.py` (superseded).

**`FINDINGS.md`** — the running research log, §1–14, which doubles as the paper skeleton.

---

## 18. PRIOR ART AND POSITIONING (prior-art pass completed 14 July 2026)

**The broad claim "use the raw measurement record, not the detector record" is NOT novel.** It is
established for **leakage** detection, and the closest prior art states the gauge asymmetry almost
verbatim. Do not claim the general idea.

| reference | what it is | relation |
|---|---|---|
| **Bultink et al. (Delft) leakage HMM**, arXiv:1905.12731 / Science Advances | detects leakage from the RAW parity record; a leaked ancilla reads a persistent fixed outcome because measurement cannot discern \|2⟩ from \|1⟩. Defines a **two-step** syndrome s_D[m] = M_A[m]·M_A[m−2] explicitly because the consecutive detector destroys the signal. | **Found "the detector XOR is the wrong operation" for leakage, in 2019.** Must cite. Measured: their two-step construction does NOT recover the atom-loss signal on the projected type (gap 0.008). |
| **Transmon leakage HMM on Surface-17**, npj QI (2020) | extends HMM-on-raw-record to the surface code with analog ancilla readout | same lineage |
| **Blume-Kohout & Young**, arXiv:2504.14643 | DEM estimation from syndrome data; p_ij / detector-covariance (Eq. 36). Explicitly transforms raw syndrome into detector histories by XOR of consecutive pairs. | the head-to-head baseline |
| **Wang / Nie / Dai / Ni / Zhang / Zhai / Chen**, arXiv:2604.14269 (v1 15 Apr, v2 25 May 2026) | STGNN loss decoder, Tsinghua IAS + iFLYTEK + Intelligent Quantum Inception. **Recall 0.654, precision 0.845 at 10 rounds**; modified AlphaQubit 0.652 / 0.856. Hyperparameters: d=5, D=256, N_layer=6, α=0.1, Conv1D kernel 3. **>85% miss rate for final-round losses**; only ~3% of unrecognized losses originate from the earliest round. Round-level i.i.d. loss model. **No code released.** | the original target; now the loss-localizer reference point |
| **Perrin et al.**, arXiv:2603.24237 | "Correlated Atom Loss as a Resource" — order-of-magnitude logical error reduction, loss threshold 3.2% → 4%. Requires teleportation-based loss-detection units. | *assumes* η |
| **Perrin / Jandura / Pupillo**, arXiv:2412.07841 | gate-level loss model; **§8 survivor biased-Z** | the realism source |
| **Liu**, arXiv:2603.04156 | Lemma 10: correlated atom loss of an X-type ancilla with a data qubit | *assumes* η |
| **Baranes**, arXiv:2502.20558 → PRX 16 011002 | neutral-atom loss QEC | context |
| **Evered**, arXiv:2604.25987 | CZ error budget: **0.061% loss, 0.035% leakage, 0.024% Pauli-Z, 0.003% Pauli-X(Y)**; 0.054% loss per atom per CZ | the physical loss rate |
| swap-LRC Rydberg-decay decoder, *PRA* (April 2026) | | check before submitting |

### THE ONE-LINE POSITIONING

> The advantage of the raw measurement record over the detector record is established for leakage
> detection [Bultink 1905.12731; npj QI 2020]. We show it specializes to correlated atom loss and
> yields a calibration-free, closed-form estimator of the gate-pair loss-correlation η that neither
> detector-covariance methods [BKY 2504.14643] nor the two-step leakage construction [Bultink] can
> produce.

**This is a METHODS NOTE, not a field-shifting representation claim.** That is the right size. Lead
the abstract with **η and calibration-free**, not with "detectors are wrong" — the latter is the
mechanism, not the headline.

---

## 19. ERRORS MADE AND CAUGHT (methodology — this matters as much as the results)

Every one was caught by a check or a measurement, never by reading the code:

1. **`MX` deleting measurements** — corrupted every memory_X result. Caught by adding the
   measurement/detector-count invariant.
2. **Fast builder divergence** — unsorted set iteration for `DEPOLARIZE1` targets. Caught by pinning
   byte-for-byte against the reference over 200 random loss configs.
3. **Degenerate one-round substep estimator** — η traded freely against both nuisance parameters and
   the fit pegged at 1.0 for every planted η. Fixed by a 3-round window (r−1 / r / r+1); the module
   was later superseded entirely.
4. **"Correlated loss helps recall"** — the entire original thesis. Falsified by the ceiling.
5. **"The 0.516 is a fluctuation"** — it was round-0 determinism. The sign consistency across seven
   replicates was the signal; the magnitude was not.
6. **Substep-dependence hypothesis for r_co** — falsified (0.2961 / 0.2918 / 0.2817 / 0.2875 across
   substeps, flat). The real structure was ancilla-type bimodality.
7. **"p_ij tracks η"** — confounded by persistent flicker; retracted.
8. **`clean_events` as the estimator** — stripped the η signal; caught by the known-answer test in
   two runs and reverted.

**Operating principles that emerged:** build the check that must give a known answer and discard the
result when it doesn't; report a confidence interval, not a mean; do the prior-art check *before*
building; don't sample when you can force a deterministic known answer; a flaky check is worse than
no check; state a conclusion only after measuring it.

---

## 20. DECISIONS LOCKED (do not relitigate)

- **Authorship:** Suriya Narayanan Rajavel **first author**; **Prof. Sara Behdad senior /
  corresponding**. Guo, Heidari, and Feng are **not** available in any role (co-author, letter
  writer, or arXiv endorser) — the relationship with all three is mediated entirely through a friend
  who presented the work under his own name; they do not know Suriya.
- **Framing:** soft-sensor / quantum-metrology, i.e. inferring a hardware parameter from
  instrumentation already present. **Not** the gauge-physics headline, which has no senior backer.
  Do **not** adopt a bio-inspired or environmental framing — the idea came from gauge structure, and
  claiming otherwise is a provenance falsehood.
- **Venue:** **IEEE TQE** is the fit (scope explicitly includes metrology; publishes QEC decoders and
  characterization; engineering framing is native; a systems senior author is unremarkable there;
  IF ≈ 4.6–5.2, Q1). **QST** is more prestigious (IF 4.9, SJR 1.911 vs 1.045, h-index 62 vs 28) but
  states it is highly selective and selects only a small proportion — estimated 20–30% as the paper
  currently stands. APC $3690 / £2640 / €3010. IOP transfer mechanism limits downside.
- **arXiv:** try **quant-ph** first (check UF institutional auto-endorsement via `su.rajavel@ufl.edu`);
  **eess.SY** or **cs.ET** are legitimate fallback primary categories that sidestep the quant-ph
  endorsement gate.
- **Timeline reality:** publication will not land before the December 2026 PhD deadlines and does not
  need to. The **arXiv preprint** is what carries the applications and faculty outreach.

---

## 21. WHAT IS NOT ESTABLISHED

- Only **memory_Z / memory_X**, and only the CSS codes Stim generates (surface rotated/unrotated,
  repetition, color). Not tested: logical circuits with transversal gates, LDPC codes, non-CSS codes.
- Circuit-level depolarizing noise plus the §11 refined loss physics. **No full hardware noise model,
  no crosstalk, no atom-transport/shuttling loss between rounds, no hardware data at all.**
- **The estimator measures η but never demonstrates that knowing η improves anything measurable.**
  No decoder is shown to get better once it has the estimate. This is the single largest gap and the
  main reason the paper sits below QST's stated bar.
- The refined-physics code path is not byte-verified against the reference builder.
- The substep-recovery upside (≈4× tighter η, §8) is identified but unexploited.

---

## 22. IMMEDIATE NEXT STEPS

1. **Close the impact gap** — feed η̂ back into a decoder and show a measurable improvement. Within
   reach on the existing harness. This is what would move the paper from TQE-grade to QST-plausible.
2. **Publication figures** — the ceiling, the three-way representation gap, the localizer sweep, the
   generality sweep, the realism study. Currently terminal tables.
3. **The writeup** — weeks of prose; `FINDINGS.md` §1–14 is the skeleton, in order. Methods is the
   most mechanical section and will not change with framing, so it drafts first.
4. **Re-run the prior-art scan the week of submission.** The field moved four times in 2026 (Perrin,
   Liu, Wang, Blume-Kohout–Young) and a swap-LRC Rydberg-decay decoder landed in *PRA* in April.

---

## 23. HOW TO RESTART COLD

```powershell
Set-Location D:\Decoder
.\.venv\Scripts\Activate.ps1
python scripts\run_checks.py        # 30 checks; must print ALL CHECKS PASSED
```

If any check fails, **stop** — the harness behaves differently in that environment and nothing
downstream is trustworthy until it is resolved. Then read `FINDINGS.md` for the running log, and this
document for the full picture.

## 24. POSITIONING CORRECTION - 19 Jul 2026 (SUPERSEDES SECTION 18)

Section 18's one-line positioning is RETRACTED. Wang et al. (2604.14269) already decode atom loss
from the raw stabilizer record without LDUs, feeding BOTH the binary outcome AND the XOR detector
as separate features. "The raw record beats the detector record for atom loss" is NOT a novel
claim. Do not make it.

NOVEL: the closed-form calibration-free estimator of the gate-pair loss correlation eta, requiring
no loss-detection units. Perrin et al. (2603.24237) define this exact parameter as p_c - "the
conditional probability of losing the second atom given that the first atom has already been lost
during the same CZ gate" - and SWEEP it 0 to 1 as a free input, because nobody can measure it.
Their threshold result (3.2% -> 4%) depends on its value; Pauli Envelope's (5.15% -> 7.82%) too.
That gap is the motivation. See FINDINGS.md section 11b for the exact wording to use.

## 25. IMPACT EXPERIMENT DONE - 19 Jul 2026 (SUPERSEDES SECTION 21 bullet 3 AND SECTION 22 item 1)

Section 21 states the estimator "never demonstrates that knowing eta improves anything measurable"
and Section 22 lists closing that gap as next step 1. BOTH ARE OUT OF DATE. The experiment was
built and run - see FINDINGS.md section 14 and scripts/run_impact.py.

RESULT: no benefit. d=5, T=12, p=0.002, p_g=0.008, eta=1, 5500 shots.
  arm 0 IGNORANT        LER 0.34636
  arm A data-loss aware LER 0.21945
  arm B eta known       LER 0.23455
  arm O oracle ancilla  LER 0.23455
CONTROL ignorant - A = +0.12691 +- 0.01667 (~15 sigma) proves the machinery works, so the A-vs-B
null is real. A - B = -0.01509 +- 0.01565, consistently negative. Perfect eta knowledge (arm O)
buys nothing.

WHY: discounting the co-lost ancilla treats its reading as erased, but a dead ancilla reads m=0
DETERMINISTICALLY - the same physics the estimator relies on. Erasing it loses more than it gains.

SCOPE: tests ONE strategy (detector discounting, no heralds). Perrin 2603.24237 uses their
correlation parameter inside an LDU-heralded loss graph and reports up to an order-of-magnitude
logical error reduction. NOT a contradiction - different mechanism - but the paper MUST address
the tension explicitly, naming Perrin's number.

CONSEQUENCE: venue is settled as IEEE TQE. QST needed a positive impact result; there isn't one via
this route. Do not re-run this experiment. Perrin-style superstabilizer integration remains genuine
future work.
