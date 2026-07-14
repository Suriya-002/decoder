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


# ---------------------------------------------------------------------------------------
# TWO RULES YOU MUST FOLLOW WHEN BUILDING THE EVENT LIST. Both were found the hard way.
#
# RULE 1 -- DROP ROUND-0 LOSSES.
#   "A truncated stabilizer flickers at 0.5" is a STEADY-STATE statement. In round 0 the data
#   is still a product state in the memory basis, so a truncated SAME-BASIS stabilizer is still
#   deterministic and reads 0 with high probability. Measured, memory_Z:
#
#       loss round |  Z-type partner  |  X-type partner
#             0    |     0.6684       |     0.4998        <- round 0 is anomalous
#          1..11   |     0.5000       |     0.5000        <- exactly theory, 1.6M samples each
#
#   Including round 0 biases eta_hat by +(1/12)*(1/2)*(0.668-0.500)*2 = +0.007. Measured: +0.008.
#
# RULE 2 -- EXCLUDE EVENTS WHERE THE PARTNER ANCILLA WAS LOST ANYWHERE IN THAT ROUND,
#   not merely at the loss substep. An ancilla lost via its gate with some OTHER data qubit is
#   DEAD and reads 0, but a naive check ("was it co-lost at THIS substep?") calls it ALIVE.
#   This is the leading suspect for the residual ~+0.01 in eta_hat that survives Rule 1.
#   NOT YET CONFIRMED. Confirm it before trusting eta_hat to better than +-0.02.
# ---------------------------------------------------------------------------------------


def eta_from_counts(n_zero, n_total, eps=0.0):
    """Returns (eta_hat, half_width_95). Both endpoints are theory; eps is measured on hardware."""
    P = n_zero / n_total
    se = np.sqrt(max(P * (1 - P), 1e-12) / n_total)
    denom = 0.5 - eps
    return float((P - 0.5) / denom), float(1.96 * se / denom)
