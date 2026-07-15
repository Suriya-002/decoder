"""Code-agnostic parsing + loss injection, for the generality sweep.

The main harness (layout.py) is specific to the rotated surface code. The gauge argument,
however, is about individual stabilizers and should hold for ANY CSS code. This module extracts
just what the gauge probe needs -- data/ancilla split, ancilla X/Z type, and per-round raw and
detector indices -- from any stim.Circuit.generated(...) circuit, and injects a forced ancilla
loss without the surface-code-specific partner machinery.
"""
import stim


def parse_generic(base):
    """Return dict with data, anc, anc_x (set), anc_z (set), and coords, for any generated code."""
    coords = {int(k): tuple(v) for k, v in base.get_final_qubit_coordinates().items()}
    # ancillas are reset+measured every round (MR / MRX); data are measured once at the end (M/MX)
    anc, seen_h = set(), False
    anc_x = set()
    for inst in base.flattened():
        if inst.name in ("MR", "MRX"):
            anc.update(t.value for t in inst.targets_copy())
        if inst.name == "H" and not seen_h:
            anc_x = {t.value for t in inst.targets_copy()}
            seen_h = True
    # X-type ancillas are those that get the basis-change H (for CX-based extraction).
    anc_x &= anc
    anc_z = anc - anc_x
    data = sorted(set(coords) - anc)
    return dict(coords=coords, data=data, anc=sorted(anc), anc_x=anc_x, anc_z=anc_z)


def raw_measurement_index(base):
    """(ancilla, round) -> raw measurement-record index, for any code."""
    idx, k, rnd = {}, 0, 0
    for i in base.flattened():
        if i.name in ("M", "MR", "MX", "MRX"):
            tg = [t.value for t in i.targets_copy()]
            if i.name in ("MR", "MRX"):
                for q in tg:
                    idx[(q, rnd)] = k
                    k += 1
                rnd += 1
            else:
                k += len(tg)
    return idx


def detector_index(base, info):
    """detector index -> (ancilla, round), matching detectors to ancilla coordinates."""
    xy2a = {}
    for a in info["anc"]:
        c = info["coords"].get(a)
        if c is None or len(c) < 2:
            continue
        xy2a.setdefault((c[0], c[1]), a)
    out = {}
    for i, cd in base.get_detector_coordinates().items():
        if len(cd) < 3:
            continue
        key = (cd[0], cd[1])
        if key in xy2a:
            out[i] = (xy2a[key], int(cd[2]))
    return out


def force_ancilla_dead(base, ancilla, rnd):
    """Rebuild the circuit with `ancilla` traced out and forced to read 0 at round `rnd` only.

    This is the minimal gauge probe: a dead ancilla reads m=0; its detector is m(r) XOR m(r-1),
    so whether the detector fires depends on the PREVIOUS round's value -- deterministic for a
    same-basis stabilizer, randomly projected otherwise.
    """
    out = stim.Circuit()
    mr_round = 0
    for inst in base.flattened():
        n = inst.name
        if n in ("MR", "MRX"):
            tg = [t.value for t in inst.targets_copy()]
            if mr_round == rnd and ancilla in tg:
                # force this ancilla's outcome to 0: reset just before measuring
                out.append("R" if n == "MR" else "RX", [ancilla])
            out.append(inst)
            mr_round += 1
        else:
            out.append(inst)
    return out
