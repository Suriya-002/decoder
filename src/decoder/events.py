"""Turn a ground-truth loss list into the CLEAN event list the raw-record estimator needs.

This is the piece that was missing, and getting it wrong biases eta_hat high in a way that
SCALES WITH p_g. Two rules, both found the hard way, both now known-answer-checked.

RULE 1 -- DROP ROUND-0 LOSSES.
  In round 0 the data is a product state in the memory basis, so a truncated same-basis stabilizer
  is STILL deterministic and its ALIVE partner reads 0.668, not 0.5. (run_raw_anchors.py)

RULE 2 -- ONE EVENT PER (STABILIZER, LOSS), AT THE FIRST TRUNCATED ROUND ONLY.
  "An alive partner on a truncated stabilizer reads 0.5" holds ONLY in the round the data qubit
  FIRST goes missing on that stabilizer. After the truncated outcome is recorded, the surviving
  qubits are projected and the stabilizer reads DETERMINISTICALLY thereafter. Data loss is
  persistent, so a naive per-round scan counts the same stabilizer many times, in rounds where it
  is no longer 0.5. Measured directly (eta_true = 0, so eta_hat must be 0):

      alive partner, by round since first truncation:   first round -> 0.500   later -> biased high
      naive (all events)  eta_hat @ p_g=0.02  = 0.157
      clean (this rule)   eta_hat @ p_g=0.02  = 0.003

  The rule: for each stabilizer, take only the FIRST round in which any of its data qubits is lost;
  require the ancilla itself alive that round; require exactly one of its data qubits newly lost.

DEPLOYMENT NOTE. This uses the ground-truth loss list, so it is a CALIBRATION / ANALYSIS tool, not
a drop-in deployable front end. A real localiser gives (qubit, round) claims; to apply Rule 2 in
deployment the localiser must resolve loss ONSET (first missing round), which a flicker-based
localiser does. At the physical loss rate (p_g = 0.00054) the naive bias is already below the noise
floor (eta_hat = 0.012 +- 0.016), so the rule matters only when p_g is pushed to 10-20x hardware.
"""


def clean_events(layout, rounds, truth, anbr=None):
    """Yield (data_qubit, round, substep, partner_ancilla) for CLEAN onset events.

    layout   : Layout
    rounds   : number of QEC rounds
    truth    : list of (qubit, round, substep) from loss_models.sample_gate
    anbr     : optional {ancilla: set(data neighbours)} cache
    """
    dset = set(layout.data)
    if anbr is None:
        anbr = {a: set() for a in layout.anc}
        for s in range(4):
            for (dq, a) in layout.pairs[s]:
                anbr[a].add(dq)

    lost_dr, anc_lost_r = {}, {}
    for (x, r, s) in truth:
        (lost_dr if x in dset else anc_lost_r).setdefault(r, set()).add(x)

    # first truncated round per stabilizer (ancilla)
    first_trunc = {}
    for a in layout.anc:
        rs = [r for r in range(rounds) if anbr[a] & lost_dr.get(r, set())]
        if rs:
            first_trunc[a] = min(rs)

    for (q, r, sub) in truth:
        if q not in dset or r < 1:                       # RULE 1
            continue
        a = layout.partner.get((q, sub))
        if a is None:
            continue
        if first_trunc.get(a) != r:                      # RULE 2: first truncated round only
            continue
        if a in anc_lost_r.get(r, set()):                # partner must be ALIVE this round
            continue
        if (anbr[a] & lost_dr.get(r, set())) != {q}:     # exactly one newly-lost data qubit
            continue
        yield (q, r, sub, a)
