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

## 6. OPEN — and it is the highest-value next build

η̂ is monotone and localiser-independent but **affinely compressed**: 0.11 → 0.71 instead of 0 → 1.
Calibratable, but it means the likelihood model is still misspecified.

**The likely cause, and the fix, are the same thing: stop using the detector record.**

A lost ancilla reads **m = 0 deterministically** — that is the physics (atom-array readout detects |1⟩
population; no atom ⇒ no population). A live ancilla on a truncated stabilizer reads a **random** bit.

    P(m = 0 | ancilla dead)            = 1.0
    P(m = 0 | ancilla alive, truncated) = 0.5

That is a factor of two, on the **whole array**, in **either basis**, with **no XOR to destroy it**.

The detector record (XOR of consecutive rounds) throws this away on half the ancillas by construction.
**Every learned decoder in this literature — Wang's STGNN included — consumes detectors.** If the raw
measurement record carries loss information that the detector record structurally cannot, that is a real
architectural claim and it is cheap to test with this harness.

PREDICTION, NOT YET MEASURED. Test it before believing it.
