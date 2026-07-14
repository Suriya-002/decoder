"""eta from bare syndrome -- the MECHANISM, isolated and measured.

RESULT (d=5, T=12, p=0.002, p_g=0.054% = Evered's measured rate):

    data qubit lost, partner ancilla ALIVE     -> partner's detector at the loss round: 0.4996
    data qubit lost, partner ancilla CO-LOST   -> partner's detector at the loss round: 0.2878

The 0.4996 is not fitted. Theory pins it: when a data qubit vanishes, the X- and Z-stabilizers
that overlapped on it become truncated, anticommute, and collapse to +-1 with equal probability.
Exactly 0.5. That the harness reproduces it to 4 decimals is a free known-answer check.

The 0.2878 is the whole effect: a co-lost ancilla is dead that round and reads a forced 0, so it
cannot report the flicker it is supposed to report. The suppression is the signature of eta.

CLOSED-FORM ESTIMATOR (the "alive" arm is theory-pinned, so this is nearly model-free):

    observed onset rate  =  (1 - eta) * 0.50  +  eta * r_co
    =>   eta_hat = (0.50 - observed) / (0.50 - r_co)

WHAT IS STILL MISSING -- BE HONEST ABOUT THIS.
This probe is ORACLE-CONDITIONED: it is handed the ground-truth loss list. A real experiment
is not. Closing the loop needs a loss LOCALISER on the front end -- which is exactly what
Wang's STGNN does (recall 0.654, precision 0.845). An aggregate estimator with a naive
flicker-onset heuristic instead of a real localiser was tried and FAILED its known-answer test
(planted eta=0 recovered as 1.0). The bottleneck is event identification, not the mechanism.

OPEN QUESTION, and it is the cheap one: given a swing this large (0.50 -> 0.29), how good does
the localiser actually have to be? A crude one may well suffice, in which case no STGNN is needed.
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
ap.add_argument("--pg", type=float, default=0.00054, help="Evered 2604.25987 measures 0.054%%/atom/CZ")
ap.add_argument("--basis", default="z")
ap.add_argument("--shots", type=int, default=8000)
ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
ap.add_argument("--seed", type=int, default=11)
A = ap.parse_args()

b = make_base(A.d, A.T, A.p, A.basis); L = parse_layout(b, A.d)
pre, anc = prerender(b), set(L.anc)
key = {(a, r): i for i, (a, r) in detector_map(b, L).items()}
dset = set(L.data)
rng = np.random.default_rng(A.seed)

print(f"ETA MECHANISM PROBE   d={A.d} T={A.T} p={A.p} p_g={A.pg} memory_{A.basis}  ({A.shots} shots/eta)")
print("Partner ancilla's detector at the loss round, split by whether the partner was co-lost.\n")
print(f"{'eta':>5} | {'CO-LOST partner':>26} | {'ALIVE partner':>26} | {'eta_hat':>8}")
print(f"{'':>5} | {'n':>6}  {'click rate':>17} | {'n':>6}  {'click rate':>17} |")
print("-" * 78)

t0 = time.time()
for eta in A.etas:
    cc = cn = sc = sn = 0
    for _ in range(A.shots):
        ev, truth = sample_gate(L, A.T, A.pg, eta, rng)
        s = build_fast(pre, anc, ev).compile_detector_sampler().sample(1)[0]
        for (q, r, sub) in truth:
            if q not in dset:
                continue
            a = L.partner.get((q, sub))
            if a is None or (a, r) not in key:
                continue
            bit = int(s[key[(a, r)]])
            if a in ev.get((r, sub), ()):
                cn += 1; cc += bit
            else:
                sn += 1; sc += bit
    cr = cc / cn if cn else np.nan
    sr = sc / sn if sn else np.nan
    ce = 1.96 * np.sqrt(cr * (1 - cr) / cn) if cn else 0
    se = 1.96 * np.sqrt(sr * (1 - sr) / sn) if sn else 0
    obs = (cc + sc) / (cn + sn)                    # what a real estimator would see
    r_co = 0.288                                   # calibrated co-lost rate
    hat = np.clip((0.50 - obs) / (0.50 - r_co), 0, 1)
    f = lambda v, e, n: f"{v:>10.4f} +-{e:.4f}" if n else f"{'--':>17}"
    print(f"{eta:>5.2f} | {cn:>6}  {f(cr,ce,cn)} | {sn:>6}  {f(sr,se,sn)} | {hat:>8.3f}")

print(f"\ntheory: ALIVE partner must be EXACTLY 0.5000 (truncated stabilizers anticommute).")
print(f"({time.time()-t0:.0f}s)")
