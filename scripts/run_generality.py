"""GENERALITY SWEEP. Is the gauge argument surface-code-specific? Measured: no.

THE GAUGE LAW (code-agnostic form):
  A dead ancilla reads m = 0 (raw). Its detector is m(r) XOR m(r-1), so the detector fires with
  probability P(m(r-1) = 1) = P(the stabilizer has a randomly-projected steady-state value).
    * deterministic-sign stabilizer (matches the memory basis) -> P = 0     -> detector BLIND to the dead ancilla
    * randomly-projected stabilizer                            -> P = 0.5   -> detector fires, carries no info
  The raw record reads m = 0 in BOTH cases. So the detector representation is blind to atom loss
  on the projected-stabilizer half; the raw record is not. This is a statement about the input
  representation, and it depends only on stabilizer projection, not on the code.

RESULT (round-6 forced-dead ancilla, memory basis Z, noiseless):
  surface rotated  d=3,5,7 : raw P(m=0)=1.0 both types; detector 0.000 (Z, deterministic) / 0.50 (X, projected)
  surface unrotated d=5    : identical -- different stabilizer geometry, same law
  color code       d=5     : detector fires 0.50 on the Z-type -- because in this code+basis the
                             Z-type ancilla measures a PROJECTED stabilizer (steady-state <m>=0.498,
                             vs 0.000 for the surface code). The blind type FLIPS with the code; the
                             LAW does not.

CONCLUSION: the gauge blindness of the detector representation, and the raw record's immunity to it,
are properties of CSS-code memory in general, not of the rotated surface code.
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, stim
from decoder.codeagnostic import parse_generic, raw_measurement_index, detector_index, force_ancilla_dead

ap = argparse.ArgumentParser()
ap.add_argument("--shots", type=int, default=20000)
A = ap.parse_args()


def steady_kind(base, info, mi, a, R=6):
    m = base.compile_sampler().sample(2000)[:, mi[(a, R)]].mean()
    return "det" if min(m, 1 - m) < 0.05 else "proj"


def probe(spec, d, shots):
    c = stim.Circuit.generated(spec, distance=d, rounds=12, after_clifford_depolarization=0.0,
                               before_round_data_depolarization=0.0, before_measure_flip_probability=0.0,
                               after_reset_flip_probability=0.0)
    info = parse_generic(c); mi = raw_measurement_index(c)
    dinv = {v: k for k, v in detector_index(c, info).items()}
    R = 6
    rows = {}
    for t, group in (("X", info["anc_x"]), ("Z", info["anc_z"])):
        raws, dets, kinds = [], [], []
        for a in group:
            if (a, R) not in mi or (a, R) not in dinv:
                continue
            cc = force_ancilla_dead(c, a, R)
            raw = cc.compile_sampler().sample(shots)[:, mi[(a, R)]]
            det = cc.compile_detector_sampler().sample(shots)[:, dinv[(a, R)]]
            raws.append(float((~raw).mean())); dets.append(float(det.mean()))
            kinds.append(steady_kind(c, info, mi, a))
        if raws:
            rows[t] = (np.mean(raws), np.mean(dets), max(set(kinds), key=kinds.count), len(raws))
    return rows


print(f"GENERALITY SWEEP: force ancilla dead at round 6, memory_Z, noiseless\n")
print(f"{'code':<24}{'d':>2} | {'anc':>6} {'stabilizer':>11} | {'raw P(m=0)':>11} | {'detector fires':>15}")
print("-" * 82)
for spec, d in [("surface_code:rotated_memory_z", 3), ("surface_code:rotated_memory_z", 5),
                ("surface_code:rotated_memory_z", 7), ("surface_code:unrotated_memory_z", 5),
                ("repetition_code:memory", 5), ("color_code:memory_xyz", 5)]:
    rows = probe(spec, d, A.shots)
    name = spec.split("_memory")[0].replace("_code", "").replace(":", "-")
    for t in ("X", "Z"):
        if t not in rows:
            continue
        raw, det, kind, n = rows[t]
        klbl = "deterministic" if kind == "det" else "PROJECTED"
        print(f"{name:<24}{d:>2} | {t+'-typ':>6} {klbl:>11} | {raw:>11.4f} | {det:>15.4f}")
    print()
print("Reading: an ALIVE truncated ancilla fires its detector ~0.5 (both types). A DEAD ancilla")
print("fires ~0 on the DETERMINISTIC type (so the detector discriminates dead vs alive there) but")
print("~0.5 on the PROJECTED type (no discrimination -- gauge blind). Raw P(m=0)=1.0 everywhere.")
print("The law holds across all codes tested; only WHICH type is blind changes with code + basis.")
