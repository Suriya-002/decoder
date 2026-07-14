"""Rotated surface code layout, parsed from Stim's own generated circuit.

We never hand-roll the layout. We parse it out of stim.Circuit.generated(), so the
code geometry and the CZ schedule are Stim's (known-good, hook-error-avoiding) ones.
"""
from dataclasses import dataclass, field
import stim


def make_base(d: int, rounds: int, p: float, basis: str = "z") -> stim.Circuit:
    """Baseline rotated surface code memory circuit, uniform circuit-level depolarizing noise."""
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=d, rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


@dataclass
class Layout:
    d: int
    coords: dict            # qubit -> (x, y)
    data: list              # data qubit ids
    anc: list               # ancilla qubit ids
    anc_x: set              # X-type ancillas (the ones that get H)
    cx_layers: list         # 4 layers, each a list of (control, target)
    pairs: list             # 4 layers, each a list of (data, ancilla)
    partner: dict = field(default_factory=dict)   # (qubit, substep) -> its CZ partner

    @property
    def anc_z(self):
        return set(self.anc) - self.anc_x


def parse_layout(base: stim.Circuit, d: int) -> Layout:
    coords = {int(k): tuple(v) for k, v in base.get_final_qubit_coordinates().items()}
    anc_x, cx_layers, seen_h = set(), [], False
    for inst in base.flattened():
        if inst.name == "H" and not seen_h:
            anc_x = {t.value for t in inst.targets_copy()}
            seen_h = True
        if inst.name == "CX" and len(cx_layers) < 4:
            ts = [t.value for t in inst.targets_copy()]
            cx_layers.append([(ts[i], ts[i + 1]) for i in range(0, len(ts), 2)])

    anc = sorted({q for L in cx_layers for pr in L for q in pr if q in anc_x}
                 | {t for L in cx_layers for (c, t) in L if c not in anc_x})
    data = sorted(set(coords) - set(anc))

    pairs, partner = [], {}
    for s, L in enumerate(cx_layers):
        P = []
        for (c, t) in L:
            a, dq = (c, t) if c in anc else (t, c)
            P.append((dq, a))
            partner[(dq, s)] = a
            partner[(a, s)] = dq
        pairs.append(P)
    return Layout(d, coords, data, anc, anc_x, cx_layers, pairs, partner)


def detector_map(circ: stim.Circuit, layout: Layout) -> dict:
    """detector index -> (ancilla qubit, round)."""
    xy2anc = {tuple(layout.coords[a]): a for a in layout.anc}
    return {i: (xy2anc[(cd[0], cd[1])], int(cd[2]))
            for i, cd in circ.get_detector_coordinates().items()
            if (cd[0], cd[1]) in xy2anc}
