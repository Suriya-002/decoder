"""The three arms.

ARM A  round_iid      Wang et al. (arXiv:2604.14269) exactly.
                      "each physical qubit has a probability P_loss of being removed
                       in each round".  Round-level, i.i.d., NOT gate-level.
                      This arm exists ONLY as the external reproduction anchor.

ARM B  gate(eta=p_g)  Perrin/Jandura/Pupillo (arXiv:2412.07841).
                      Each CZ gate: each atom lost with prob p_g, INDEPENDENTLY.
                      This is the CORRECT baseline for a gate-level claim.

ARM C  gate(eta>p_g)  CZ-pair-correlated loss.
                      eta = P(partner lost | this atom lost).

CRITICAL DESIGN POINT -- MATCHED MARGINALS.
Per CZ gate on atoms (a, b):
    P(both lost)      = eta * p_g
    P(only a lost)    = P(only b lost) = (1 - eta) * p_g
  =>  marginal P(a lost at this gate) = p_g, INDEPENDENT OF eta.
So sweeping eta changes ONLY the correlation, never the total loss budget.
If you do not do this, an eta sweep confounds correlation with loss rate and
any separation you see is uninterpretable.

Independence corresponds to eta = p_g (giving P(both) = p_g^2), not eta = 0.
At p_g ~ 1e-3 the difference is negligible, but the harness is exact about it.
"""
import numpy as np


def sample_round_iid(layout, rounds, p_loss, rng):
    """ARM A -- Wang. Qubit-level, per-round, i.i.d. Loss lands at substep 0."""
    ev, truth = {}, []
    for r in range(rounds):
        for q in layout.data + layout.anc:
            if rng.random() < p_loss:
                ev.setdefault((r, 0), set()).add(q)
                truth.append((q, r, 0))
    return ev, truth


def sample_gate(layout, rounds, p_g, eta, rng):
    """ARM B (eta=p_g) and ARM C (eta>p_g). Gate-level, CZ-pair correlated, matched marginals."""
    p_both = eta * p_g
    p_one = (1.0 - eta) * p_g
    ev, truth = {}, []
    for r in range(rounds):
        for s in range(4):
            for (dq, a) in layout.pairs[s]:
                u = rng.random()
                if u < p_both:
                    hit = (dq, a)
                elif u < p_both + p_one:
                    hit = (dq,)
                elif u < p_both + 2 * p_one:
                    hit = (a,)
                else:
                    continue
                for q in hit:
                    ev.setdefault((r, s), set()).add(q)
                    truth.append((q, r, s))
    return ev, truth


def loss_truth_masks(layout, rounds, truth):
    """(data_lost[round, data_qubit], anc_lost[round, ancilla]) ground-truth boolean masks.

    Data loss is PERSISTENT: once lost at round r, the qubit is lost for all r' >= r.
    Ancilla loss is TRANSIENT: it holds for round r only.
    """
    di = {q: i for i, q in enumerate(layout.data)}
    ai = {q: i for i, q in enumerate(layout.anc)}
    D = np.zeros((rounds, len(layout.data)), bool)
    A = np.zeros((rounds, len(layout.anc)), bool)
    for (q, r, _s) in truth:
        if q in di:
            D[r:, di[q]] = True
        else:
            A[r, ai[q]] = True
    return D, A
