"""Substep-aware eta estimator vs the substep-blind one. KNOWN-ANSWER TEST.

The localiser's recall/precision are NOT told to the estimator -- they are fitted out as
nuisance parameters (f_phantom, f_mistimed). If eta is recovered across localiser qualities
WITHOUT being told the quality, the systematic is gone.
"""
import sys, pathlib, argparse, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from decoder.layout import make_base, parse_layout, detector_map
from decoder.fast import prerender, build_fast
from decoder.loss_models import sample_gate
from decoder.substep import fit

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=5); ap.add_argument("--T", type=int, default=12)
ap.add_argument("--p", type=float, default=0.002); ap.add_argument("--pg", type=float, default=0.00054)
ap.add_argument("--basis", default="z"); ap.add_argument("--shots", type=int, default=8000)
ap.add_argument("--rco", type=float, default=0.29)
ap.add_argument("--boot", type=int, default=60); ap.add_argument("--seed", type=int, default=21)
A = ap.parse_args()

b = make_base(A.d, A.T, A.p, A.basis); L = parse_layout(b, A.d)
pre, anc = prerender(b), set(L.anc)
key = {(a, r): i for i, (a, r) in detector_map(b, L).items()}
dset = set(L.data)
# partners ORDERED BY CZ SUBSTEP -- this ordering is the entire point
ORD = {q: [L.partner[(q, s)] for s in range(4) if (q, s) in L.partner] for q in L.data}
rng = np.random.default_rng(A.seed)

# q_bg: background detector rate, measured from LOSS-FREE data. No fitting.
bg = build_fast(pre, anc, {}).compile_detector_sampler().sample(20000)
q_bg = float(bg.mean())
print(f"SUBSTEP-AWARE ETA   d={A.d} T={A.T} p={A.p} p_g={A.pg}  {A.shots} shots/eta")
print(f"q_bg measured from loss-free data = {q_bg:.4f}   |   r_co = {A.rco} (simulated constant)")
print(f"theory anchor: an ALIVE truncated stabilizer flickers at EXACTLY 0.5000\n")

QUAL = [("oracle", 1.00, 1.00), ("Wang STGNN", 0.654, 0.845),
        ("crude", 0.40, 0.60), ("very crude", 0.25, 0.40)]
ETAS = [0.0, 0.25, 0.5, 0.75, 1.0]

sim = {}
t0 = time.time()
for e in ETAS:
    bits = np.empty((A.shots, b.num_detectors), bool); tru = []
    for i in range(A.shots):
        ev, truth = sample_gate(L, A.T, A.pg, e, rng)
        bits[i] = build_fast(pre, anc, ev).compile_detector_sampler().sample(1)[0]
        tru.append([(q, r) for (q, r, s) in truth if q in dset])
    sim[e] = (bits, tru)
print(f"simulated ({time.time()-t0:.0f}s)\n")


def hists(eta, recall, precision, rg):
    """3-ROUND window (r-1, r, r+1) x partners ordered by CZ substep."""
    bits, tru = sim[eta]
    G = {}
    for i, ev in enumerate(tru):
        cl = [(q, r) for (q, r) in ev if rg.random() < recall]
        nfp = int(rg.poisson(len(cl) * (1 - precision) / precision)) if precision < 1 and cl else 0
        cl += [(int(rg.choice(L.data)), int(rg.integers(2, A.T - 2))) for _ in range(nfp)]
        for (q, r) in cl:
            ps = ORD[q]
            if any((a, rr) not in key for a in ps for rr in (r - 1, r, r + 1)):
                continue
            k = len(ps)
            w = np.array([[bits[i, key[(a, rr)]] for a in ps] for rr in (r - 1, r, r + 1)], bool)
            G.setdefault(k, []).append(w)
    return {k: np.array(v) for k, v in G.items()}


print(f"{'localiser':<13}" + "".join(f"{'eta='+str(e):>22}" for e in ETAS))
print(f"{'':13}" + "".join(f"{'eta_hat [95% CI]':>22}" for _ in ETAS))
print("-" * 103)
for name, rc, pr in QUAL:
    row = f"{name:<13}"
    for e in ETAS:
        H = hists(e, rc, pr, rng)
        hat = fit(H, q_bg, A.rco)[0]
        bs = []
        for _ in range(A.boot):
            Hb = {k: B[rng.integers(0, len(B), len(B))] for k, B in H.items()}
            bs.append(fit(Hb, q_bg, A.rco)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        mark = "" if lo - 1e-9 <= e <= hi + 1e-9 else "*"
        row += f"{hat:>8.3f} [{lo:.2f},{hi:.2f}]{mark:<2}"
    print(row)
print("\n* = planted eta OUTSIDE its own 95% CI (a failure).")
print("The estimator is NEVER told the localiser's recall or precision -- both are fitted out.")
