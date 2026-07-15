"""THEORY ANCHORS of the raw-record estimator. Forced injection -- no loss sampling at all.

One circuit build gives N shots, so this reaches ~10^7 samples in minutes and resolves 0.0005.

    P(m=0 | partner ALIVE on a truncated stabilizer) = 0.5000   <- THEORY (steady state only!)
    P(m=0 | partner CO-LOST)                         = 1 - eps  <- THEORY + measurable hardware eps

THE ROUND-0 CAVEAT, WHICH IS REAL PHYSICS AND COST ME HALF A DAY:
  "A truncated stabilizer anticommutes with its partner and flickers at 0.5" is a STEADY-STATE
  statement. In round 0 the data is still |0...0> -- a Z PRODUCT STATE -- so a truncated
  Z-stabilizer is STILL deterministic and reads 0 with high probability. Measured: 0.6684.
  By round 1 the state is generic and it is 0.5000 to four decimals, forever after.

  => RULE 1: DROP ROUND-0 LOSSES FROM THE ESTIMATOR.
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from decoder.layout import make_base, parse_layout
from decoder.fast import prerender, build_fast
from decoder.raw import measurement_index

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=5)
ap.add_argument("--T", type=int, default=12)
ap.add_argument("--p", type=float, default=0.002)
ap.add_argument("--eps", type=float, default=0.0, help="false-bright rate of a LOST atom")
ap.add_argument("--basis", default="z")
ap.add_argument("--shots", type=int, default=40000, help="shots PER injected loss location")
A = ap.parse_args()

b = make_base(A.d, A.T, A.p, A.basis)
L = parse_layout(b, A.d)
pre, anc = prerender(b), set(L.anc)
mi = measurement_index(b)
DEG = {q: len([s for s in range(4) if (q, s) in L.partner]) for q in L.data}

alive, colost = {}, {}
for q in L.data:
    for s in range(4):
        if (q, s) not in L.partner:
            continue
        a = L.partner[(q, s)]
        t = "X" if a in L.anc_x else "Z"
        for r in range(A.T):
            if (a, r) not in mi:
                continue
            # SOLO loss -> partner stays ALIVE.  PAIR loss -> partner is CO-LOST.
            for tag, ev, acc in (("alive", {(r, s): {q}}, alive),
                                 ("colost", {(r, s): {q, a}}, colost)):
                m = build_fast(pre, anc, ev, false_bright=A.eps).compile_sampler().sample(A.shots)
                z = int((~m[:, mi[(a, r)]]).sum())
                for k in (("ALL",), ("round", r), ("anc", t), ("deg", DEG[q]), ("sub", s)):
                    acc.setdefault(k, [0, 0])
                    acc[k][0] += z
                    acc[k][1] += A.shots


def fmt(acc, k, target):
    if k not in acc:
        return f"{'--':>26}"
    z, n = acc[k]
    P = z / n
    se = np.sqrt(max(P * (1 - P), 1e-12) / n)
    dev = (P - target) / se if se > 0 else 0.0
    return f"{P:.5f} +-{1.96*se:.5f} {dev:>+6.1f}s"


tgt_a, tgt_c = 0.5, 1.0 - A.eps
print(f"RAW-RECORD THEORY ANCHORS   d={A.d} T={A.T} p={A.p} eps={A.eps} memory_{A.basis}")
print(f"THEORY:  ALIVE -> {tgt_a:.4f}   CO-LOST -> {tgt_c:.4f}\n")
print(f"{'':22} {'partner ALIVE':>26}   {'partner CO-LOST':>26}")
print("-" * 80)
print(f"{'ALL (rounds >= 1)':<22} "
      f"{fmt({k: [sum(alive[('round', r)][0] for r in range(1, A.T)), sum(alive[('round', r)][1] for r in range(1, A.T))] for k in [('ALL',)]}, ('ALL',), tgt_a)}   "
      f"{fmt({k: [sum(colost[('round', r)][0] for r in range(1, A.T)), sum(colost[('round', r)][1] for r in range(1, A.T))] for k in [('ALL',)]}, ('ALL',), tgt_c)}")
print()
print("BY LOSS ROUND  <- round 0 is the anomaly. Everything else is exactly 0.5.")
for r in range(A.T):
    mark = "   <-- NOT 0.5 (product state)" if r == 0 else ""
    print(f"  round {r:<15} {fmt(alive, ('round', r), tgt_a)}   {fmt(colost, ('round', r), tgt_c)}{mark}")
print()
for lbl, keys in (("BY ANCILLA TYPE", [("anc", t) for t in ("Z", "X")]),
                  ("BY QUBIT DEGREE", [("deg", g) for g in (4, 3, 2)]),
                  ("BY CZ SUBSTEP",   [("sub", s) for s in range(4)])):
    print(lbl + "   (pooled over ALL rounds, so round 0 drags ALIVE up -- expected)")
    for k in keys:
        print(f"  {str(k[1]):<20} {fmt(alive, k, tgt_a)}   {fmt(colost, k, tgt_c)}")
    print()
