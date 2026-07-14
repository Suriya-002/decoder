# Phase 0 — Information ceiling for a gate-pair-aware loss decoder

Kill-or-greenlight harness. No GNN, no training, no GPU. Stim + numpy only.

## What this answers

Does CZ-pair-correlated atom loss carry information that an independence-assuming
decoder (Wang et al., arXiv:2604.14269) cannot use?

## The three arms (matched marginals — this is the whole point)

| Arm | Model | Source |
|---|---|---|
| **A** | round-level i.i.d. loss | Wang et al. 2604.14269 — *"each physical qubit has a probability P_loss of being removed in each round"* |
| **B** | gate-level, η = 0 (independent within CZ) | Perrin/Jandura/Pupillo 2412.07841 |
| **C** | gate-level, η > 0 (CZ-pair correlated) | Evered 2604.25987 physics |

η = P(partner atom lost | this atom lost). Per CZ gate: P(both) = η·p_g,
P(only one) = (1−η)·p_g each ⇒ **marginal per-atom loss = p_g regardless of η.**
Sweeping η changes *only* the correlation, never the loss budget. Without this,
an η sweep confounds correlation with loss rate and every result is uninterpretable.

**Arm A is NOT the η=0 limit of arms B/C.** It is a different loss *mechanism*
(round-level, not gate-level). It exists only as the external reproduction anchor.

## Loss physics (not a Pauli channel)

- Loss moment: `DEPOLARIZE1(0.75) q` — the uniform Pauli twirl *is* the completely
  depolarizing channel, i.e. exactly "trace out q, replace with maximally mixed".
- All subsequent operations on the lost atom are erased. No entangling gate is
  carried out on the surviving partner.
- Readout of a lost atom returns **0** (atom-array readout detects |1⟩ population).
  Forced with an `R` before the `M`. Measurements are never deleted, so DETECTOR
  record offsets stay valid.
- Ancilla loss is **transient** (its own `MR` reloads it). Data loss is **persistent**.

## Run

```powershell
python scripts\run_checks.py      # 12 known-answer checks — must all pass first
python scripts\run_ceiling.py     # the Phase 0 measurement
```

`run_ceiling.py --window N --future M` controls how much of the syndrome record
the Bayes-optimal test is allowed to see. `--future 0` = loss round only (the
final-round regime). `--future 2` = two rounds of flicker accumulation.

## Known-answer checks

1. noiseless + lossless ⇒ exactly 0 detector clicks
2. no-loss rebuild is byte-identical to Stim's flattened base circuit
3. every two-qubit gate is data↔ancilla (**there are no data–data CZ gates in a surface code**)
4. one lost data qubit ⇒ exactly its Tanner-partner ancillas flicker at ~0.5, zero leakage elsewhere
5. marginal per-atom loss rate is independent of η
6. co-lost atom fraction == η exactly
