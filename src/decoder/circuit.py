"""Inject atom loss into a Stim surface-code circuit.

Loss is NOT a Pauli channel. We model it the way Perrin/Liu/Baranes do:

  * At the loss moment, the atom is traced out:  DEPOLARIZE1(0.75) q
    (the uniform Pauli twirl IS the completely depolarizing channel, i.e. exactly
     "trace out q and replace with maximally mixed" -- this decouples q correctly).
  * Every subsequent operation touching q is ERASED until the atom is reloaded.
    A lost qubit carries out no entangling gate on its surviving partner.
  * The atom's readout returns 0, because atom-array readout detects |1> population,
    so "no atom" is indistinguishable from |0>. We force this with an R before the M.
  * Ancillas are reloaded by their own MR each round  -> ancilla loss is TRANSIENT.
  * Data qubits are never reloaded            -> data loss is PERSISTENT (flicker).

Measurements are NEVER deleted, so DETECTOR record offsets stay valid.
"""
import stim

# EVERY measurement op must be handled here, never in _ONE_Q. Deleting a measurement
# shifts all downstream DETECTOR rec[] offsets and silently corrupts the whole circuit.
# memory_x uses RX/MX, memory_z uses R/M -- both must be covered.
_MEAS = {"M": "R", "MR": "R", "MX": "RX", "MRX": "RX", "MY": "RY", "MRY": "RY"}
_MEAS_RELOADS = {"MR", "MRX", "MRY"}          # these reset the ancilla -> loss is transient
_ONE_Q = ("H", "X_ERROR", "Z_ERROR", "DEPOLARIZE1", "R", "RX", "RY", "S", "S_DAG")


def build_with_loss(base: stim.Circuit, layout, loss_events: dict) -> stim.Circuit:
    """loss_events: {(round, substep): {qubit, ...}} -- atoms lost at the START of that CZ layer."""
    anc_set = set(layout.anc)
    out = stim.Circuit()
    lost = set()
    cx_count = 0

    for inst in base.flattened():
        n, args, tg = inst.name, inst.gate_args_copy(), inst.targets_copy()

        if n == "CX":
            r, s = divmod(cx_count, 4)
            newly = loss_events.get((r, s), set()) - lost
            if newly:
                out.append("DEPOLARIZE1", sorted(newly), 0.75)   # <- trace the atom out
                lost |= newly
            cx_count += 1
            v = [t.value for t in tg]
            keep = [q for j in range(0, len(v), 2) for q in (v[j], v[j + 1])
                    if v[j] not in lost and v[j + 1] not in lost]
            if keep:
                out.append("CX", keep, args)

        elif n == "DEPOLARIZE2":
            v = [t.value for t in tg]
            keep = [q for j in range(0, len(v), 2) for q in (v[j], v[j + 1])
                    if v[j] not in lost and v[j + 1] not in lost]
            if keep:
                out.append("DEPOLARIZE2", keep, args)

        elif n in _MEAS:
            v = [t.value for t in tg]
            dead = [q for q in v if q in lost]
            if dead:
                out.append(_MEAS[n], dead)     # no atom -> readout reports 0, in ANY basis
            out.append(n, v, args)             # NEVER delete a measurement
            if n in _MEAS_RELOADS:
                lost -= anc_set                # ancillas are reloaded every round

        elif n in _ONE_Q:
            keep = [t.value for t in tg if t.value not in lost]
            if keep:
                out.append(n, keep, args)

        else:
            out.append(inst)                   # DETECTOR / OBSERVABLE / SHIFT_COORDS / TICK
    return out
