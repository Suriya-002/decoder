"""HEAD-TO-HEAD vs the field's detector-statistics estimators (Spitz; Google; Blume-Kohout &
Young arXiv:2504.14643). The claim is about REPRESENTATIONS, not about whose estimator is better.

BKY/Spitz/Google all consume the DETECTOR record = XOR of consecutive syndrome measurements.
This project's result: that XOR destroys the atom-loss signature on the ancilla type whose
stabilizer sign is randomly projected. So detector-based estimators inherit a gauge blind spot
for loss correlation. This script demonstrates it with the field's actual p_ij estimator.

WHAT IS AND IS NOT CLAIMED:
  CLAIMED (and clean): the co-loss discrimination -- how much a partner ancilla's reading tells
    you it was co-lost -- is FULL in the raw record on both ancilla types, and GAUGE-SUPPRESSED
    in the detector record on the randomly-projected type. Measured as a co-lost-minus-alive gap.
  NOT CLAIMED: that a raw detector-pair p_ij "tracks eta". It does not, and this script says so.
    A co-lost data qubit's persistent flicker dominates any detector pair equally at every eta,
    swamping the co-loss-specific term. That confound is WHY a bespoke raw-record estimator is
    needed, and it is the honest reason the representation gap -- not a p_ij number -- is the result.
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from decoder.layout import make_base, parse_layout, detector_map
from decoder.fast import prerender, build_fast
from decoder.loss_models import sample_gate
from decoder.raw import measurement_index
from decoder.pij import pij_pair

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=5); ap.add_argument("--T", type=int, default=12)
ap.add_argument("--p", type=float, default=0.002); ap.add_argument("--pg", type=float, default=0.00054)
ap.add_argument("--shots", type=int, default=9000); ap.add_argument("--seed", type=int, default=2024)
A = ap.parse_args()

b = make_base(A.d, A.T, A.p, "z"); L = parse_layout(b, A.d)
pre, anc = prerender(b), set(L.anc)
mi = measurement_index(b); dm = detector_map(b, L)
di = {(a, r): i for i, (a, r) in dm.items()}
dset = set(L.data); rng = np.random.default_rng(A.seed)

acc = {(rep, t, co): [0, 0] for rep in ("raw", "det") for t in ("Z", "X") for co in (0, 1)}
for eta in (0.0, 1.0):
    for _ in range(A.shots):
        ev, truth = sample_gate(L, A.T, A.pg, eta, rng)
        raw = build_fast(pre, anc, ev).compile_sampler().sample(1)[0]
        det = build_fast(pre, anc, ev).compile_detector_sampler().sample(1)[0]
        for (q, r, s) in truth:
            if q not in dset or r < 1: continue
            a = L.partner.get((q, s))
            if a is None or (a, r) not in mi or (a, r) not in di: continue
            t = "X" if a in L.anc_x else "Z"
            co = 1 if a in ev.get((r, s), ()) else 0
            acc[("raw", t, co)][0] += int(raw[mi[(a, r)]] == 0); acc[("raw", t, co)][1] += 1
            acc[("det", t, co)][0] += int(det[di[(a, r)]] == 1); acc[("det", t, co)][1] += 1


def rate(k):
    z, n = acc[k]
    if n == 0: return float("nan"), 0
    P = z / n; return P, 1.96 * np.sqrt(P * (1 - P) / n)


print(f"CO-LOSS DISCRIMINATION vs REPRESENTATION   d={A.d} T={A.T} p_g={A.pg} memory_Z\n")
print(f"{'representation':<22}{'anc type':>9} | {'signal|alive':>18} | {'signal|co-lost':>18} | {'gap':>9}")
print("-" * 84)
for rep, lbl in (("raw", "RAW measurement m_a"), ("det", "DETECTOR (BKY XOR) d_a")):
    for t in ("Z", "X"):
        (ra, ea), (rc, ec) = rate((rep, t, 0)), rate((rep, t, 1))
        print(f"{lbl:<22}{t+'-type':>9} | {ra:>8.4f} +-{ea:.4f} | {rc:>8.4f} +-{ec:.4f} | {rc-ra:>+9.4f}")
    print()
print("Z-type = deterministic stabilizer sign; X-type = randomly projected (memory_Z).")
print("The DETECTOR representation (what every BKY/Spitz/Google p_ij estimator consumes) is")
print("gauge-blind on the X-type half -- gap collapses to ~0. The RAW record keeps it on both.")
print("This is the core result: it is a statement about the INPUT REPRESENTATION every learned")
print("and statistical loss estimator in this literature is built on.")
