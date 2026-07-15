"""The p_ij / detector-covariance estimator of Spitz et al., Google Quantum AI, and
Blume-Kohout & Young (arXiv:2504.14643).

For a pair of detectors (i, j), the rate of a DEM event that flips BOTH is

    p_ij = 1/2 - 1/2 * sqrt( (<z_i><z_j>) / <z_i z_j> ),     z_k = (-1)^{detector_k}   (BKY Eq. 36)

This is THE standard way the field estimates correlated events from syndrome data. It is
built ENTIRELY on the detector record -- the XOR of consecutive syndrome measurements. BKY
state this explicitly: raw syndrome histories are transformed into detector histories by
XOR-ing every consecutive pair of syndrome bits, and every downstream quantity is a function
of detector polarizations.

That XOR is exactly the operation this project shows destroys the atom-loss signature on the
ancilla type whose stabilizer sign is randomly projected. So any p_ij-family estimator inherits
a gauge blind spot for loss correlation. This module implements p_ij faithfully so the blindness
can be demonstrated with the field's actual method, not a strawman.
"""
import numpy as np


def polarization(bits):
    """<z> = <(-1)^bit> = 1 - 2 E[bit], per column."""
    return 1.0 - 2.0 * np.asarray(bits, float).mean(axis=0)


def pij_pair(det_i, det_j):
    """BKY Eq. 36 for a single detector pair. det_i, det_j are 1-D bool arrays over shots."""
    zi = 1.0 - 2.0 * det_i.mean()
    zj = 1.0 - 2.0 * det_j.mean()
    zij = 1.0 - 2.0 * (det_i ^ det_j).mean()          # <z_i z_j> = <(-1)^(x_i XOR x_j)>
    if zij == 0:
        return float("nan")
    ratio = (zi * zj) / zij
    if ratio < 0:
        return float("nan")                            # sqrt of negative -> undefined (noise)
    return 0.5 - 0.5 * np.sqrt(ratio)
