"""Substep-aware eta estimator over a 3-ROUND window. Localiser quality is fitted out.

A ONE-ROUND window is DEGENERATE and cannot work. Both failure modes of the localiser mimic
correlation in round r alone:
    phantom claim  -> all bits pushed to background (q_bg ~ 0.04)
    co-loss        -> ONE bit pushed to r_co (~0.29) instead of 0.50
  With the substep marginalised, the co-loss effect smears across bit positions and is
  indistinguishable from a phantom. And a true onset at substep 0 is almost identical to a
  mistimed claim. eta then trades freely against BOTH nuisances and the fit pegs at 1.0.
  (Measured: it does exactly that. See git history.)

ROUNDS r-1 AND r+1 BREAK BOTH DEGENERACIES, FOR FREE:

    round r-1   true onset at r -> no loss yet     -> ALL partners at q_bg
                mistimed (loss earlier)            -> ALL partners already flickering at 0.50
                phantom                            -> ALL partners at q_bg
                => SEPARATES mistimed from true.

    round r     the substep step-function. Partners whose CZ already fired sit at q_bg;
                partners whose CZ never fired are truncated and sit at 0.50; the partner AT
                the loss substep sits at 0.50 if it survived, r_co if it was CO-LOST.
                => THIS is the eta signal, and s* is the edge of the step.

    round r+1   true onset -> data qubit is gone, EVERY partner truncated -> ALL at 0.50
                phantom    -> ALL at q_bg
                => SEPARATES phantom from true, without touching eta.

  eta now moves exactly ONE bit in the whole 3k-bit window. It is cleanly identified.

FIXED CONSTANTS (not fitted)
    0.50   theory. A truncated stabilizer anticommutes with its partner and collapses to +-1
           with equal probability. Not a fit. The harness reproduces it to 4 decimals.
    q_bg   measured from loss-free data.
    r_co   the one simulated input. It is NOT jointly identifiable with eta -- only the product
           eta*(0.5 - r_co) enters the s* bit -- so it must be fixed, never fitted.

FITTED
    eta, f_phantom, f_mistimed.  The localiser's recall and precision are never told to the
    estimator; they enter only through f_phantom / f_mistimed and are marginalised out.
"""
import numpy as np


def _logp(bits, p):
    return np.where(bits, np.log(p), np.log1p(-p))


def event_loglik(B, q_bg, r_co, eta, f_p, f_l):
    """B: (N, 3, k) bits at rounds (r-1, r, r+1), partners ORDERED BY CZ SUBSTEP."""
    N, _, k = B.shape
    Lp = _logp(B, q_bg).sum(axis=(1, 2))                       # phantom: everything background
    Ll = np.full(N, 3 * k * np.log(0.5))                       # mistimed: everything truncated
    p_star = (1 - eta) * 0.5 + eta * r_co
    Lt = np.empty((N, k))
    for s in range(k):                                         # true onset at substep s
        pr = np.where(np.arange(k) < s, q_bg, 0.5)             # CZ already fired -> q_bg
        pr[s] = p_star                                         # the co-loss slot
        Lt[:, s] = (_logp(B[:, 0, :], q_bg).sum(1)             # r-1 : no loss yet
                    + _logp(B[:, 1, :], pr).sum(1)             # r   : the step function
                    + B[:, 2, :].shape[1] * np.log(0.5))       # r+1 : all truncated
        Lt[:, s] += np.where(B[:, 2, :], 0.0, 0.0).sum(1)      # (Bern(0.5) -> bit-independent)
    m = np.maximum(np.maximum(Lp, Ll), Lt.max(1))
    mix = (f_p * np.exp(Lp - m) + f_l * np.exp(Ll - m)
           + (1 - f_p - f_l) * np.exp(Lt - m[:, None]).mean(1))
    return np.log(np.clip(mix, 1e-300, None)) + m


def nll(theta, groups, q_bg, r_co):
    eta, f_p, f_l = theta
    if not (0 <= eta <= 1 and 0 <= f_p <= 1 and 0 <= f_l <= 1 and f_p + f_l <= 0.98):
        return 1e12
    return -sum(float(event_loglik(B, q_bg, r_co, eta, f_p, f_l).sum()) for B in groups.values())


def fit(groups, q_bg, r_co):
    from scipy.optimize import minimize
    best, bv = None, np.inf
    for e in np.linspace(0, 1, 6):
        for fp in (0.05, 0.2, 0.4):
            for fl in (0.05, 0.2):
                v = nll((e, fp, fl), groups, q_bg, r_co)
                if v < bv:
                    bv, best = v, (e, fp, fl)
    r = minimize(nll, best, args=(groups, q_bg, r_co), method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-3, maxiter=3000))
    return tuple(np.clip(r.x, 0, 1))
