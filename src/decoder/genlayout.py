"""Code-general layout + loss model, for validating the eta estimator beyond the rotated surface code.

layout.py is specific to the rotated surface code. This module parses the CX schedule of ANY
generated CSS-code memory circuit into the same (substep -> [(data, ancilla)]) structure, provided
the schedule is a clean sequence of CX layers with each qubit in at most one gate per layer and each
gate acting between one data and one ancilla. Measured true for rotated + unrotated surface codes.

It reuses the SAME loss physics as loss_models.sample_gate (matched marginals) and the SAME builder
(circuit.build_with_loss / fast.build_fast), which key loss events by (round, substep) and count CX
layers with divmod(cx, n_sub). n_sub is detected here rather than hardcoded to 4.
"""
from dataclasses import dataclass, field
import stim
from .codeagnostic import parse_generic


@dataclass
class GenLayout:
    d: int
    coords: dict
    data: list
    anc: list
    anc_x: set
    anc_z: set
    pairs: list                       # n_sub layers, each [(data, ancilla)]
    n_sub: int
    partner: dict = field(default_factory=dict)   # (qubit, substep) -> partner


def parse_any(base: stim.Circuit, d: int) -> GenLayout:
    info = parse_generic(base)
    anc, data = set(info["anc"]), set(info["data"])

    # collect CX layers until the schedule repeats (one round's worth)
    raw_layers = []
    for inst in base.flattened():
        if inst.name == "CX":
            ts = [t.value for t in inst.targets_copy()]
            raw_layers.append(tuple((ts[i], ts[i + 1]) for i in range(0, len(ts), 2)))

    # detect period: the layer sequence repeats every n_sub layers
    n_sub = None
    for cand in range(1, len(raw_layers) + 1):
        if len(raw_layers) % cand == 0 and all(
            raw_layers[i] == raw_layers[i % cand] for i in range(len(raw_layers))
        ):
            n_sub = cand
            break
    if n_sub is None:
        n_sub = len(raw_layers)

    pairs, partner = [], {}
    for s in range(n_sub):
        P = []
        for (c, t) in raw_layers[s]:
            a, dq = (c, t) if c in anc else (t, c)
            assert (c in anc) ^ (t in anc), "gate is not data<->ancilla"
            P.append((dq, a))
            partner[(dq, s)] = a
            partner[(a, s)] = dq
        pairs.append(P)

    return GenLayout(d, info["coords"], sorted(data), sorted(anc),
                     info["anc_x"], info["anc_z"], pairs, n_sub, partner)
