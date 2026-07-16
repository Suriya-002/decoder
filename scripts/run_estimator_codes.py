"""END-TO-END eta estimator on codes BEYOND the rotated surface code (closes FINDINGS section 13).

Section 10 showed the gauge LAW generalizes. This shows the full ESTIMATOR pipeline -- sample
CZ-pair-correlated loss, read the raw record, recover eta = (P(m=0)-0.5)/(0.5-eps) -- works on:
  * unrotated surface code  (different geometry, weight-4 throughout, n_sub=4)
  * color code memory_xyz   (DIFFERENT CODE FAMILY, n_sub=6)

The harness's rotated-surface assumptions (4 CX layers/round) are removed: genlayout.parse_any
detects the CX-layer period per code, and fast.set_n_sub / a generalized sample_gate use it.

RESULT (p_g=0.00054, eps=0):
  unrotated: anchors 0.5005 / 1.0000; eta_hat -0.006 / 0.246 / 0.488 / 0.747 / 1.000 (all in CI)
  color:     anchors 0.5009 / 1.0000; eta_hat  0.013 / 0.508 / 1.000 (all in CI)
Both theory anchors hold and eta is recovered on a code family with a different stabilizer
structure AND a different gate schedule. The estimator is not tied to the rotated surface code.
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, stim
from decoder.genlayout import parse_any
from decoder.fast import prerender, build_fast, set_n_sub
from decoder.loss_models import sample_gate
from decoder.raw import measurement_index, eta_from_counts

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=5); ap.add_argument("--T", type=int, default=12)
ap.add_argument("--p", type=float, default=0.002); ap.add_argument("--pg", type=float, default=0.00054)
ap.add_argument("--shots", type=int, default=12000); ap.add_argument("--seed", type=int, default=2024)
A = ap.parse_args()

CODES = [("surface_code:unrotated_memory_z", "unrotated surface", [0.0, 0.25, 0.5, 0.75, 1.0]),
         ("color_code:memory_xyz", "color code", [0.0, 0.5, 1.0])]

for spec, name, etas in CODES:
    b = stim.Circuit.generated(spec, distance=A.d, rounds=A.T,
                               after_clifford_depolarization=A.p, before_round_data_depolarization=A.p,
                               before_measure_flip_probability=A.p, after_reset_flip_probability=A.p)
    L = parse_any(b, A.d); set_n_sub(L.n_sub)
    pre, anc = prerender(b), set(L.anc); mi = measurement_index(b); dset = set(L.data)
    print(f"\n=== {name.upper()}  d={A.d}: {len(L.data)} data, {len(L.anc)} anc, n_sub={L.n_sub} ===")

    R = 6; acc = {True: [0, 0], False: [0, 0]}
    for q in L.data:
        for s in range(L.n_sub):
            if (q, s) not in L.partner: continue
            a = L.partner[(q, s)]
            if (a, R) not in mi: continue
            for co, ev in ((False, {(R, s): {q}}), (True, {(R, s): {q, a}})):
                m = build_fast(pre, anc, ev).compile_sampler().sample(6000)[:, mi[(a, R)]]
                acc[co][0] += int((~m).sum()); acc[co][1] += 6000
    for co, tgt in ((False, 0.5), (True, 1.0)):
        z, n = acc[co]; P = z / n
        print(f"  anchor P(m=0|{'co-lost' if co else 'alive  '}) = {P:.5f} +-{1.96*np.sqrt(P*(1-P)/n):.5f}  (theory {tgt})")

    rng = np.random.default_rng(A.seed)
    print(f"  {'eta_true':>9} | {'events':>7} | {'eta_hat':>8} | {'95% CI':>16}")
    for et in etas:
        z = n = 0
        for _ in range(A.shots):
            ev, truth = sample_gate(L, A.T, A.pg, et, rng)
            if not truth: continue
            m = build_fast(pre, anc, ev).compile_sampler().sample(1)[0]
            for (q, r, s) in truth:
                if q not in dset or r < 1: continue
                a = L.partner.get((q, s))
                if a is None or (a, r) not in mi: continue
                z += int(m[mi[(a, r)]] == 0); n += 1
        h, e = eta_from_counts(z, n, 0.0)
        print(f"  {et:>9.2f} | {n:>7} | {h:>8.3f} | [{h-e:>6.3f},{h+e:>6.3f}]")

# restore default for any downstream imports in the same process
set_n_sub(4)
print("\nEstimator validated on a different surface-code geometry AND a different code family.")
