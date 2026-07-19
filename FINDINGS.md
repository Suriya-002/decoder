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

## 7. The 0.5 is a STEADY-STATE statement — round 0 is not 0.5

Seven independent replicates of P(m=0 | ALIVE) all landed ABOVE 0.5 (0.5159, 0.5067, 0.5003, 0.5022,
0.5091, 0.5086, 0.5130). I called that a fluctuation on a +1.7 sigma pooled result. **That was wrong —
the sign consistency was the signal, not the magnitude.** Seven-for-seven is p ~ 0.008 under the null.

Forced single-loss injection, 9.6M samples, no loss sampling at all:

    P(m=0 | ALIVE) = 0.49998 +- 0.00032   (-0.1 sigma)

Exactly 0.5. Every stratum — degree, ancilla type, substep — within +-1 sigma. **The physics is clean.**
The bias was in what the SAMPLED path includes and the forced test excluded: **round 0.**

| loss round | Z-type partner | X-type partner |
|---|---|---|
| **0** | **0.6684 +- 0.0007** | 0.4998 |
| 1 .. 11 | **0.5000** (all) | **0.5000** (all) |

In round 0 the data is still |0...0> — a Z PRODUCT STATE. A truncated Z-stabilizer on a product state
is STILL deterministic, so it reads 0 with high probability, not 0.5. The claim "a truncated stabilizer
anticommutes and flickers at 0.5" holds only once the state is generic. It is a **steady-state** claim.

Arithmetic closes: (1/12 of losses in round 0) x (1/2 have Z-type partners) x (0.668 - 0.500) x 2
= **+0.007 predicted**. Observed pooled bias: **+0.008**.

**Fix: drop round-0 losses.** eta_hat bias at eta=0 falls from 0.023 to 0.009.

## Status

22+ known-answer checks, all passing, reproduced cross-platform. Nothing in this file is a fit.

## 8. RULE 2 — the residual is a persistence artifact, now UNDERSTOOD and REMOVED

I first guessed the residual was "partner ancilla dead via a different gate" (rate ~3*p_g). Measured:
that channel is real but explains only ~35% of the bias. Excluding it left ~0.045 at p_g=0.01. Two
more guesses (other-neighbour truncation; duplicate late-round entries) also failed. So I stopped
guessing and conditioned on ground truth:

    ALIVE partner, eta=0, p_g=0.01, bucketed by # neighbour losses on the stabilizer:
        even the clean k=1 bucket read P(m=0) = 0.5242 +- 0.0049   (+9.6 sigma)

**The real mechanism: the 0.5 anchor holds ONLY in the round the data qubit FIRST goes missing on
that stabilizer.** After the truncated outcome is recorded, the surviving qubits are projected and
the stabilizer reads DETERMINISTICALLY thereafter. Data loss is persistent, so a naive per-round scan
counts the same stabilizer in many later rounds where it is no longer 0.5. Same determinism physics as
round 0 (section 7), from persistence instead of initialization.

`events.clean_events` (Rule 1 + "first-truncation round, fresh stabilizer, partner alive, one newly
lost data qubit") removes it completely. eta_true = 0, so eta_hat must be 0:

| p_g | NAIVE eta_hat | CLEAN eta_hat |
|---|---|---|
| 0.00054 — Evered's rate | -0.012 +- 0.016 | 0.009 +- 0.017 |
| 0.005 | 0.035 +- 0.011 (biased) | -0.005 +- 0.014 |
| 0.010 | 0.059 +- 0.010 (biased) | 0.004 +- 0.014 |
| 0.020 | 0.157 +- 0.009 (biased) | 0.003 +- 0.018 |

Now a known-answer check (run_checks.py #23-24).

**CRUCIAL SCOPE LIMIT ON clean_events.** It keeps only clean ALIVE onsets — the partner is alive by
construction — so it DELIBERATELY drops co-loss events, which ARE the eta signal. It is therefore an
ANCHOR / eta=0 verification tool, NOT an eta>0 estimator. The production estimator (run_eta_raw.py,
naive collector) keeps co-loss events and is unbiased at the physical loss rate; the Rule-2 bias only
appears at >~10x hardware p_g. A one-line attempt to use clean_events AS the estimator was made and
CAUGHT by the known-answer test (it returned ~0 for every eta, because it strips the signal). Fixed.

## 9. HEAD-TO-HEAD vs the field's detector-statistics estimators (BKY 2504.14643)

Blume-Kohout & Young estimate DEM event rates from syndrome data via the p_ij / detector-covariance
method (their Eq. 36, re-deriving Spitz and Google Quantum AI). It is the strongest general syndrome-statistics
estimator in the literature and it handles readout errors. **It is built entirely on the detector
record** -- the XOR of consecutive syndrome measurements. That is exactly the operation section 3
shows destroys the loss signature on the randomly-projected ancilla type.

Same co-loss discrimination, both representations, split by ancilla type (p_ij implementation
validated against a planted correlation, run_checks.py #25-27):

| representation | anc type | gap (co-lost minus alive) |
|---|---|---|
| RAW measurement m_a | Z-type | **+0.54** |
| RAW measurement m_a | X-type | **+0.50** |
| DETECTOR (BKY XOR) d_a | Z-type | -0.36 |
| DETECTOR (BKY XOR) d_a | X-type | **-0.02  (blind)** |

The detector representation is gauge-blind on the X-type half; the raw record keeps the signal on
both. This is a statement about the INPUT REPRESENTATION every learned and statistical loss
estimator in this literature -- BKY, Spitz, Google Quantum AI, Wang's STGNN -- is built on.

**HONEST SCOPE (do not overclaim).** A raw detector-PAIR p_ij does NOT cleanly "track eta": a
co-lost data qubit's persistent flicker dominates any detector pair equally at every eta, swamping
the co-loss-specific term (measured; the per-pair delta is noise). That confound is precisely why a
bespoke raw-record estimator is needed, and it is why the REPRESENTATION GAP -- not a p_ij number --
is the result. Positioning against BKY is "the leading general method has a nameable blind spot for
this error class, here is the physics," not "our estimator beats theirs."

## 10. GENERALITY: the gauge law is not surface-code-specific — MEASURED

Reduced to its code-agnostic form, the gauge law is: a dead ancilla reads raw m=0, but its
detector is m(r) XOR m(r-1), so the detector fires with P(m(r-1)=1) = P(the stabilizer sign is
randomly projected). Deterministic-sign stabilizer -> detector blind to the dead ancilla;
projected stabilizer -> detector fires 0.5, no info. Raw reads 0 either way. This depends only on
stabilizer projection, not the code.

Forced-dead ancilla at round 6, memory_Z, noiseless (run_generality.py; unrotated case is
run_checks.py #28-29):

| code | d | anc type | stabilizer | raw P(m=0) | detector fires |
|---|---|---|---|---|---|
| surface rotated | 3, 5, 7 | Z | deterministic | 1.0000 | 0.0000 |
| surface rotated | 3, 5, 7 | X | projected | 1.0000 | ~0.50 |
| surface unrotated | 5 | Z / X | det / proj | 1.0000 | 0.0000 / ~0.50 |
| color code | 5 | Z | **projected** | 1.0000 | **0.50** |

The color code is the sharpest confirmation: its Z-type ancilla measures a PROJECTED stabilizer in
this basis (steady-state <m>=0.498, vs 0.000 for the surface code), so its detector fires 0.50 --
the OPPOSITE type is blind. The blind type's label flips with the code+basis; the law does not. Raw
P(m=0)=1.0 on every code. The detector representation's blindness to atom loss, and the raw record's
immunity, are properties of CSS-code memory in general.

## 11. PRIOR ART and HONEST POSITIONING (prior-art pass, 14 Jul 2026)

The broad claim "use the raw measurement record, not the detector record" is NOT novel. It is
established for LEAKAGE detection, and the closest prior art states the gauge asymmetry almost
verbatim. Do not claim the general idea. Claim the specific specialization.

WHAT EXISTS (must be cited, and built around):
  * Bultink et al. (Delft) leakage HMM -- Science Advances / arXiv:1905.12731. Detects leakage from the RAW parity
    record. A leaked ancilla reads a persistent fixed outcome because measurement cannot discern
    |2> from |1> -- structurally identical to a lost atom reading m=0. They define a data-qubit
    syndrome as the TWO-STEP product s_D[m] = M_A[m] * M_A[m-2], explicitly because the ordinary
    consecutive detector destroys the signal. They found "the detector XOR is the wrong operation"
    for leakage, in 2019.
  * Transmon leakage-detection-via-HMM on Surface-17 -- npj Quantum Information (2020). Extends the
    HMM-on-raw-record idea to the surface code, using the analog ancilla readout plus the
    defect-probability increase.
  * Blume-Kohout & Young 2504.14643 -- the detector-covariance p_ij estimator (Section 9 head-to-head).

MEASURED: is the Bultink two-step construction enough to recover the ATOM-LOSS signal? NO. Three
representations, co-loss discrimination gap on the partner ancilla (run_pij_baseline.py):

| representation | Z-type (deterministic) | X-type (projected) |
|---|---|---|
| RAW m(r) | 0.50 | **0.49** |
| 1-step detector m(r)^m(r-1) [BKY/Spitz] | 0.38 | **0.003** |
| 2-step m(r)^m(r-2) [Bultink leakage] | 0.39 | **0.008** |

Both XOR-based constructions -- the ordinary detector AND Bultink's two-step leakage syndrome -- are
gauge-blind on the projected type. The two-step references round r-2, still a projected round, so
the forced-0 is XORed against a random value exactly as the one-step is. Only the raw record works
on both types.

WHAT SURVIVES AS THE CONTRIBUTION (narrow, defensible, publishable):
  Specializing the known raw-record advantage to CORRELATED ATOM LOSS in neutral-atom codes, and
  deriving a CALIBRATION-FREE CLOSED-FORM estimator of the CZ-pair loss-correlation eta -- a
  quantity nobody estimates, from the m=0 coincidence observable that no published loss decoder
  uses, shown gauge-general across codes (Section 10), and shown to sit outside what both the
  detector-covariance method [BKY] and the two-step leakage construction [Bultink] can produce.

ONE-LINE POSITIONING FOR THE PAPER:
  "The advantage of the raw measurement record over the detector record is established for leakage
   detection [Bultink 1905.12731; npj QI 2020]. We show it specializes to correlated atom loss and
   yields a calibration-free, closed-form estimator of the gate-pair loss-correlation eta that
   neither detector-covariance methods [BKY 2504.14643] nor the two-step leakage construction
   [Google] can produce."

This is a METHODS NOTE, not a field-shifting representation claim. That is the right size.

## 12. What is STILL NOT established

- Only memory_Z / memory_X, and only the CSS codes Stim generates (surface rotated/unrotated,
  repetition, color). Not tested: logical circuits with transversal gates, LDPC codes, non-CSS codes.
- The full eta ESTIMATOR (not just the gauge law) has been validated only on the rotated surface
  code. The generality sweep confirms the underlying REPRESENTATION claim on other codes, not the
  end-to-end estimator.
- Everything is circuit-level depolarizing noise. No hardware noise model, no leakage beyond the
  gate-cancellation loss model, no crosstalk.
- Re-run the prior-art search before writing: Perrin, Liu, Wang, and Blume-Kohout-Young all landed
  within the last few months and the area is moving fast.

## 14. IMPACT EXPERIMENT - does knowing eta improve DECODING? MEASURED: no, via this strategy

DESIGN. A lossy circuit has non-deterministic detectors AND a non-deterministic logical observable
(a lost data qubit in the logical support destroys the operator), so Stim cannot build a DEM from
it - proper loss decoding needs superstabilizer merging and logical-operator deformation, which is
what Perrin arXiv:2603.24237 IS. Sidestepped: SAMPLE from the true lossy circuit, DECODE with the
clean DEM whose edge weights are modified per shot to encode the decoder's beliefs. Freeing an edge
= probability 0.5 = weight 0. Verified lossless: rebuilding the DEM unmodified reproduces the
baseline LER exactly (0.00198 both ways).

Four arms (d=5, T=12, p=0.002, p_g=0.008, eta=1, 5500 shots):

| arm | knows | LER |
|---|---|---|
| 0 IGNORANT | nothing | 0.34636 |
| A | data-qubit losses; assumes ancillas alive | 0.21945 |
| B | + discounts co-lost ancilla detectors (eta known) | 0.23455 |
| O ORACLE | + every truly-dead ancilla | 0.23455 |

CONTROL - the machinery works: ignorant - A = +0.12691 +- 0.01667, ~15 sigma. Loss information
massively improves decoding, so the A-vs-B null is a REAL null, not a broken pipeline.

RESULT: A - B = -0.01509 +- 0.01565, consistently negative across runs. Knowing which ancillas were
co-lost does not help and slightly hurts. Not an estimation-quality problem: the ORACLE arm with
perfect ancilla knowledge is identical to arm B.

CONSISTENCY CHECK: at eta=1, arm B == arm O EXACTLY (1290/5500 reference run; independently
reproduced as 1251/5500 on a second machine), because every partner ancilla really is co-lost.

WHY IT FAILS, AND WHY THAT IS THE INTERESTING PART. Freeing the co-lost ancilla's measurement edge
treats its reading as ERASED. But a dead ancilla reads m = 0 DETERMINISTICALLY - that is the entire
basis of this project's estimator. Its detector d = 0 XOR m(r-1) still carries information about
m(r-1). Erasing it throws away more than it gains. The dead ancilla's reading is informative, so
discarding it is exactly the wrong move. The negative result and the paper's central claim are the
same physics.

STATISTICS CAVEAT - DO NOT POOL RUNS. run_impact.py hardcodes np.random.default_rng(7), so separate
runs of the shipped script share identical loss realizations; only Stim's Pauli/measurement sampling
differs. Pooling understates the variance and would manufacture false significance. Honest
statement: A - B is consistently negative across runs (-0.014, -0.0151, -0.0135) but NOT
individually significant. Resolving whether the slight harm is real needs independent seeds; judged
not worth the compute, since "no benefit" and "slight harm" support the same conclusion and the
~15-sigma control already establishes the null is real.

SCOPE - DO NOT OVERCLAIM. This tests ONE strategy for using eta (discount the corrupted detector).
Perrin uses eta differently, inside superstabilizer / loss-detection construction. Evidence that eta
has no value for THIS decoding strategy; NOT evidence that eta is useless for decoding in general.

## 11b. POSITIONING - CORRECTED 19 Jul 2026 (supersedes the one-line positioning above)

Wang et al. (2604.14269) already decode atom loss from the raw stabilizer record without LDUs,
and explicitly feed BOTH the binary measurement outcome AND the XOR detector as separate
features. The mechanism claim "the raw record beats the detector record for atom loss" is
therefore NOT novel. Do not make it.

WHAT IS NOVEL: the closed-form calibration-free estimator of the gate-pair loss correlation eta,
and the fact that it needs no loss-detection units. Perrin et al. (2603.24237) define exactly
this parameter - p_c, "the conditional probability of losing the second atom given that the first
atom has already been lost during the same CZ gate" - and SWEEP it as a free input (0 <= p_c <= 1),
because no one can measure it. Their loss threshold claim (3.2% -> 4%) depends on its value, as
does Pauli Envelope's (5.15% -> 7.82%). That is the motivation.

USE THIS LINE:
  Raw stabilizer-record information has been used for leakage detection [Bultink 2020; Varbanov
  2020] and, for atom loss, by a learned decoder consuming both raw outcomes and detectors [Wang
  2026]. We quantify the representation gap that motivates this choice - the detector XOR destroys
  the loss signature on gauge-projected stabilizers - and, unlike prior work, use it to derive a
  calibration-free closed-form estimator of the gate-pair loss correlation eta. Existing
  correlated-loss decoders [Perrin 2026] sweep this parameter as a free input and require
  loss-detection units; our estimator needs neither.
