"""HOW GOOD DOES THE LOSS LOCALISER HAVE TO BE?

The mechanism probe (run_eta_probe.py) is oracle-fed: it knows the lost qubit, the round,
AND the CZ substep. A real localiser knows none of that perfectly, and knows the substep
NOT AT ALL -- Wang's STGNN has no substep feature. So:

  * NO SUBSTEP -> you cannot tell which of the data qubit's <=4 partner ancillas was the
    co-lost one. You must average over all of them. Only 1 of 4 is suppressed, so the
    oracle swing (0.50 -> 0.29, i.e. -0.21) dilutes by ~4x to about -0.05.

  * IMPERFECT RECALL -> fewer events, larger error bars. Costs precision on eta_hat, not bias.

  * FALSE POSITIVES -> a phantom loss has no truncated stabilizer, so its "partners" sit at
    the background detector rate (~0.04), far BELOW 0.50. That mimics suppression and biases
    eta_hat UP.

  * TIMING ERRORS -> a loss claimed a round late has partners already flickering at 0.50 in
    steady state, with nothing co-lost. That mimics no-correlation and biases eta_hat DOWN.

  The two error modes push opposite ways. Both are simulated. A simulation-built calibration
  curve at known (p_g, recall, precision) absorbs both -- which is exactly how any detector
  response function is handled.

THE NUMBER THAT DECIDES THE PROJECT:
    slope  b       = obs(eta=1) - obs(eta=0)      how much signal a localiser of this quality leaves
    sigma_eta      = sigma_obs / |b|              the resolution on eta, per N shots
If sigma_eta is small at Wang-level quality, cite Wang as a front end and ship an estimator.
If it is small at CRUDE quality, no neural decoder is needed at all.
"""
import sys, pathlib, argparse, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from decoder.layout import make_base, parse_layout, detector_map
from decoder.fast import prerender, build_fast
from decoder.loss_models import sample_gate

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=5)
ap.add_argument("--T", type=int, default=12)
ap.add_argument("--p", type=float, default=0.002)
ap.add_argument("--pg", type=float, default=0.00054)
ap.add_argument("--basis", default="z")
ap.add_argument("--shots", type=int, default=6000)
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--cache", default="results/sweep_cache.npz")
A = ap.parse_args()

b = make_base(A.d, A.T, A.p, A.basis); L = parse_layout(b, A.d)
pre, anc = prerender(b), set(L.anc)
key = {(a, r): i for i, (a, r) in detector_map(b, L).items()}
dset = set(L.data)
PARTNERS = {q: sorted({L.partner[(q, s)] for s in range(4) if (q, s) in L.partner}) for q in L.data}
rng = np.random.default_rng(A.seed)
ETAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# ---- simulate ONCE per eta; every localiser quality is then applied to the SAME shots ----
cache = pathlib.Path(A.cache)
if cache.exists():
    z = np.load(cache, allow_pickle=True); DATA = z["data"].item()
    print(f"loaded cached shots from {cache}\n")
else:
    DATA, t0 = {}, time.time()
    for e in ETAS:
        bits = np.empty((A.shots, b.num_detectors), bool); tru = []
        for i in range(A.shots):
            ev, truth = sample_gate(L, A.T, A.pg, e, rng)
            bits[i] = build_fast(pre, anc, ev).compile_detector_sampler().sample(1)[0]
            tru.append([(q, r) for (q, r, s) in truth if q in dset])   # (qubit, ONSET round) only
        DATA[e] = (bits, tru)
        print(f"  simulated eta={e:.2f}  ({time.time()-t0:.0f}s)")
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, data=np.array(DATA, dtype=object))
    print(f"\ncached -> {cache}\n")


def observed(eta, recall, precision, rng):
    """Run the substep-blind estimator through a localiser of the given quality."""
    bits, tru = DATA[eta]
    tot = hit = 0
    for i, events in enumerate(tru):
        claims = [(q, r) for (q, r) in events if rng.random() < recall]     # recall loss
        ntp = len(claims)
        nfp = int(rng.poisson(ntp * (1 - precision) / precision)) if precision < 1 and ntp else 0
        for _ in range(nfp):                                               # phantom claims
            claims.append((int(rng.choice(L.data)), int(rng.integers(1, A.T - 1))))
        for (q, r) in claims:
            for a in PARTNERS[q]:                                          # SUBSTEP-BLIND: all partners
                if (a, r) in key:
                    tot += 1; hit += int(bits[i, key[(a, r)]])
    return (hit / tot if tot else np.nan), tot


QUAL = [("oracle (substep-blind)", 1.00, 1.00),
        ("Wang STGNN",             0.654, 0.845),
        ("crude",                  0.40, 0.60),
        ("very crude",             0.25, 0.40)]

print(f"LOCALISER SWEEP   d={A.d} T={A.T} p={A.p} p_g={A.pg}  {A.shots} shots/eta   [SUBSTEP-BLIND]\n")
print(f"{'localiser':<24} {'obs(0)':>8} {'obs(1)':>8} {'slope b':>9} {'sig_obs':>9} {'sig_eta':>9} | recovery eta=0.25/0.50/0.75")
print("-" * 118)
for name, rc, pr in QUAL:
    o = {}; n = {}
    for e in ETAS:
        o[e], n[e] = observed(e, rc, pr, rng)
    bslope = o[1.0] - o[0.0]
    sig_obs = np.sqrt(o[0.5] * (1 - o[0.5]) / n[0.5])
    sig_eta = abs(sig_obs / bslope) if bslope else np.inf
    rec = [np.clip((o[e] - o[0.0]) / bslope, -0.5, 1.5) if bslope else np.nan for e in (0.25, 0.5, 0.75)]
    print(f"{name:<24} {o[0.0]:>8.4f} {o[1.0]:>8.4f} {bslope:>+9.4f} {sig_obs:>9.4f} {sig_eta:>9.3f} | "
          f"{rec[0]:>5.2f}  {rec[1]:>5.2f}  {rec[2]:>5.2f}")

print(f"\nsig_eta = resolution on eta from {A.shots} shots. Scales as 1/sqrt(shots).")
print("Recovery columns are held-out: calibrated ONLY on the eta=0 and eta=1 endpoints.")
