"""eta from the RAW MEASUREMENT record. Two theory constants, one measurable hardware number.

    P(m = 0 | partner ancilla ALIVE on a truncated stabilizer) = 0.5        <- THEORY
    P(m = 0 | partner ancilla CO-LOST)                         = 1 - eps    <- THEORY + hardware

    =>   eta_hat = (P(m=0) - 0.5) / (0.5 - eps)

eps is the false-bright rate of a LOST atom (dark counts, stray scatter). It is not a fudge
factor: you measure it on hardware by reading out an empty trap. Verified robust to eps <= 5%.

WHY THE DETECTOR RECORD CANNOT DO THIS
  Detectors are gauge-invariant. A stabilizer's ABSOLUTE value is fixed by a random projection in
  round 0 and says nothing about Pauli errors -- only CHANGES do. The detector (XOR of consecutive
  rounds) quotients that gauge away. Correct, and optimal, for Pauli noise.

  But a lost atom forces m = 0 REGARDLESS of the gauge. Loss BREAKS the gauge. So the XOR that
  defines a detector is precisely the operation that destroys the loss signature -- and it destroys
  it COMPLETELY on whichever half of the array carries a randomly-projected stabilizer sign.

  Measured, memory_Z:   detector record -> Z-anc +0.367,  X-anc  -0.010  (half the array is blind)
                        raw record      -> Z-anc +0.478,  X-anc  +0.510  (all of it works)
"""
import numpy as np


def measurement_index(base):
    """(ancilla, round) -> index into the raw measurement record."""
    idx, k, rnd = {}, 0, 0
    for i in base.flattened():
        if i.name in ("M", "MR", "MX", "MRX"):
            tg = [t.value for t in i.targets_copy()]
            if i.name in ("MR", "MRX"):                 # an ancilla round
                for q in tg:
                    idx[(q, rnd)] = k
                    k += 1
                rnd += 1
            else:                                       # final destructive data readout
                k += len(tg)
    return idx


def eta_from_counts(n_zero, n_total, eps=0.0):
    """Returns (eta_hat, half_width_95). Both endpoints are theory; eps is measured on hardware."""
    P = n_zero / n_total
    se = np.sqrt(max(P * (1 - P), 1e-12) / n_total)
    denom = 0.5 - eps
    return float((P - 0.5) / denom), float(1.96 * se / denom)
