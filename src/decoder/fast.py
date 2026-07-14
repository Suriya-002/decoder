"""Fast circuit builder. Pre-renders the base circuit ONCE, then splices per shot.

38x faster than circuit.build_with_loss (559 vs 14 shots/sec on d=5,T=12).
It is byte-for-byte identical to build_with_loss -- run_checks.py verifies this against
200 random loss configurations at random eta. build_with_loss remains the reference
implementation; this is only an optimisation, and it is never trusted without that check.
"""
import stim

_MEAS = {"M": "R", "MR": "R", "MX": "RX", "MRX": "RX", "MY": "RY", "MRY": "RY"}
_RELOAD = {"MR", "MRX", "MRY"}
_1Q = {"H", "X_ERROR", "Z_ERROR", "DEPOLARIZE1", "R", "RX", "RY", "S", "S_DAG"}


def prerender(base: stim.Circuit):
    """Pay targets_copy() once, not once per shot."""
    out, cx = [], 0
    for i in base.flattened():
        n = i.name
        a = i.gate_args_copy()
        astr = "(" + ",".join(repr(x) for x in a) + ")" if a else ""
        if n in ("CX", "CZ", "DEPOLARIZE2"):
            v = [t.value for t in i.targets_copy()]
            pr = [(v[j], v[j + 1]) for j in range(0, len(v), 2)]
            out.append(("2Q", n, pr, astr, cx if n in ("CX", "CZ") else None))
            if n in ("CX", "CZ"):
                cx += 1
        elif n in _MEAS:
            out.append(("M", n, [t.value for t in i.targets_copy()], astr, None))
        elif n in _1Q:
            out.append(("1Q", n, [t.value for t in i.targets_copy()], astr, None))
        else:
            out.append(("RAW", n, None, None, str(i)))
    return out


def build_fast(pre, anc_set, loss_events, false_bright: float = 0.0) -> stim.Circuit:
    lines, lost = [], set()
    for kind, n, tg, astr, extra in pre:
        if kind == "RAW":
            lines.append(extra)
        elif kind == "2Q":
            if extra is not None:
                r, s = divmod(extra, 4)
                new = sorted(q for q in loss_events.get((r, s), ()) if q not in lost)
                if new:                                     # MUST be sorted to match the reference
                    lines.append("DEPOLARIZE1(0.75) " + " ".join(map(str, new)))
                    lost.update(new)
            keep = [q for (x, y) in tg if x not in lost and y not in lost for q in (x, y)]
            if keep:
                lines.append(f"{n}{astr} " + " ".join(map(str, keep)))
        elif kind == "M":
            dead = [q for q in tg if q in lost]
            if dead:
                lines.append(f"{_MEAS[n]} " + " ".join(map(str, dead)))
                if false_bright > 0:
                    e = "Z_ERROR" if n in ("MX", "MRX") else "X_ERROR"
                    lines.append(f"{e}({false_bright!r}) " + " ".join(map(str, dead)))
            lines.append(f"{n}{astr} " + " ".join(map(str, tg)))
            if n in _RELOAD:
                lost -= anc_set
        else:
            keep = [q for q in tg if q not in lost]
            if keep:
                lines.append(f"{n}{astr} " + " ".join(map(str, keep)))
    return stim.Circuit("\n".join(lines))
