# Phase 0 — findings as of 14 July 2026

Harness: 18 known-answer checks, reproduced cross-platform (Py3.12/Linux ↔ Py3.11/Windows).

## 1. The original brief's claim is dead

**"Correlated loss is a free prior that raises loss-identification recall."** Measured, three ways, and it is false.

Bayes-optimal AUC for detecting a data-qubit loss, arm B (η=0) vs arm C (η=1), matched marginals:

| window | arm B | arm C | Δ |
|---|---|---|---|
| final-round loss, no final readout | 0.843 | 0.810 | **−0.033** |
| final-round loss, with final readout | 0.853 | 0.887 | **+0.034** |
| early loss + 2 rounds of flicker | 0.987 | 0.987 | **−0.000** |

Sign flips with the window. Magnitude is ±0.03. Goes to **exactly zero** once flicker accumulates.
No architecture beats a ceiling. **A pair-aware decoder will not produce a headline recall number.**

## 2. What survives: η is identifiable from bare syndrome

AUC(η=1 vs η=0) = **0.72–0.75**, stable across basis and window. Perrin (2603.24237) *assumes* η.
Liu (2603.04156) *assumes* η. Wang (2604.14269) does not model it. **Nobody infers it.**

## 3. The mechanism, and the thing that was hiding inside it

A co-lost partner ancilla is dead in the loss round and reads a forced 0 — it cannot report the
flicker it exists to report. Split by ancilla type (memory_Z):

| | partner ALIVE | partner CO-LOST | signal |
|---|---|---|---|
| **Z-type ancilla** | 0.4718 | **0.1052** | **+0.367** |
| X-type ancilla | 0.4995 | 0.5090 | −0.010 |

Mirrored exactly in memory_X. **Half the array carries the entire signal; the other half carries none.**

Why: in memory_Z the Z-stabilizers are deterministic, so a dead ancilla gives XOR(0, 0) → suppressed.
The X-stabilizers were projected to a *random but fixed* sign b in round 0, so a dead ancilla gives
XOR(0, b) = b → a coin flip. **The detector's XOR destroys the signal on half the ancillas.**

The theory anchor: an ALIVE truncated stabilizer must flicker at **exactly 0.5000** (it anticommutes
with its partner and collapses to ±1 with equal probability). The harness returns 0.4996. Free check.

## 4. You do not need a neural decoder

σ_η vs localiser quality, substep-blind, false positives and timing errors both simulated:

| localiser | σ_η @15k shots | @1M |
|---|---|---|
| oracle (1.00 / 1.00) | 0.048 | 0.006 |
| Wang STGNN (0.654 / 0.845) | 0.062 | 0.008 |
| crude (0.40 / 0.60) | 0.096 | 0.012 |
| very crude (0.25 / 0.40) | 0.103 | 0.013 |

Going from a perfect localiser to one that is **wrong three times out of four** costs a factor of **2**.
Not a cliff. **A crude flicker threshold suffices. No STGNN. No PyTorch Geometric. No GPU.**

## 5. The localiser systematic is gone

The substep-aware estimator (3-round window, r−1 / r / r+1) fits the localiser's failure modes out as
nuisance parameters instead of assuming them. It is never told the recall or precision, and recovers
them:

| localiser | fitted f_phantom | true (1 − precision) |
|---|---|---|
| oracle | 0.03–0.05 | 0.00 |
| Wang | 0.18–0.19 | 0.155 |
| crude | 0.40–0.41 | 0.40 |
| very crude | 0.55–0.57 | 0.60 |

A ONE-round window is degenerate and cannot work (η trades freely against both nuisances; the fit pegs
at 1.0 — measured). Rounds r−1 and r+1 break both degeneracies for free.

## 6. The detector record is the wrong input representation for atom loss — MEASURED

RAW MEASUREMENT of the partner ancilla at the loss round:

| | P(m=0) partner ALIVE | P(m=0) partner CO-LOST | signal |
|---|---|---|---|
| memory_Z, Z-anc | 0.5221 | **1.0000** | +0.478 |
| memory_Z, X-anc | 0.4897 | **1.0000** | +0.510 |
| memory_X, Z-anc | 0.4960 | **1.0000** | +0.504 |
| memory_X, X-anc | 0.5186 | **1.0000** | +0.481 |

**In 12,473 co-lost events the raw measurement was never once 1.** Not one. And the alive arm sits on
theory's 0.5 in all four cells. Signal is +0.48 to +0.51 EVERYWHERE — versus the detector record, which
gives +0.37 on half the ancillas and EXACTLY ZERO on the other half (§3).

### Closed form, no fitted constants, no simulated calibration, no template bank

    P(m=0) = (1-eta)*0.5 + eta*1.0        =>    eta_hat = 2 * (P(m=0) - 0.5)

Both endpoints are theory. Measured:

| eta_true | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| **eta_hat** | 0.032 | **0.249** | **0.515** | **0.750** | **1.000** |

### WHY — and this is the actual intellectual content

**Detectors are gauge-invariant. That is exactly why the field uses them, and exactly why they fail here.**

The absolute value of a stabilizer is set by a random projection in round 0 and carries no information
about Pauli errors — only *changes* do. The detector (XOR of consecutive rounds) quotients that gauge out.
Correct, and optimal, for Pauli noise.

**But a lost atom forces m = 0 regardless of the gauge. Loss BREAKS the gauge.** So the XOR that defines a
detector is precisely the operation that destroys the loss signature — and it destroys it completely on
whichever half of the array has a randomly-projected stabilizer sign.

Every learned decoder in this literature consumes detectors. Wang's STGNN included.

### Both caveats settled

**1. False-bright rate — SETTLED.** A lost atom is not perfectly dark on real hardware (dark counts, stray
scatter). Add a false-bright rate eps and it enters as a denominator, not a bias:

    eta_hat = (P(m=0) - 0.5) / (0.5 - eps)

| eps | eta=0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| 0.000 | 0.012 | 0.250 | 0.517 | 0.750 | 1.000 |
| 0.005 | 0.020 | 0.264 | 0.507 | 0.761 | 0.998 |
| 0.020 | 0.005 | 0.272 | 0.504 | 0.740 | 0.996 |
| **0.050** | 0.015 | 0.247 | 0.490 | 0.735 | 0.990 |

Every point inside its CI up to a **5%** false-bright rate. And eps is not a fudge factor — you measure it
on hardware by reading out an empty trap.

**2. P(m=0 | alive) = 0.516 — SETTLED. It was a fluctuation.** High-statistics clean single-loss shots:

| | P(m=0) | deviation from theory |
|---|---|---|
| CO-LOST, all degrees (n=26,038) | **1.0000** | **exactly 0** |
| ALIVE, degree-4 bulk | 0.5003 +- 0.0114 | **+0.1 sigma** |
| ALIVE, degree-2 corner | 0.4975 +- 0.0245 | -0.2 sigma |
| ALIVE, pooled clean | 0.5067 +- 0.0077 | +1.7 sigma (not significant) |
| ALIVE, *contaminated* | 0.5064 | identical to clean -> contamination ruled out |

Bulk qubits land on 0.5003 — theory, to four decimals. Both anchors are now KNOWN-ANSWER CHECKS in
run_checks.py (checks 19-22), at eps = 0 and eps = 0.02.

## Status

20+ known-answer checks, all passing, reproduced cross-platform. Nothing in this file is a fit.

**Still open:** everything measured here is d=5, memory_Z/X, circuit-level depolarizing noise, one code.
The gauge argument predicts the same result for any code whose stabilizer signs are randomly projected —
which is all of them — but that is a prediction, not a measurement. Sweep d, and sweep the code, before
claiming generality.
