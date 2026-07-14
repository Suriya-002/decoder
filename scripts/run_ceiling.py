"""PHASE 0 -- the information ceiling. No GNN. No training. No GPU.

QUESTION: does CZ-pair correlation ADD information about where/when a data qubit
          was lost, relative to an independence-assuming model?

METHOD:   Bayes-optimal AUC. Take a data qubit v, force it lost at round RL, and
          compare the detector pattern on v's Tanner-partner ancillas against the
          no-loss control. Do this under ARM B (eta=0, independent) and ARM C
          (eta=1, fully CZ-pair correlated), with MATCHED marginals.

          AUC(loss vs no-loss)  = how detectable the loss is.
          AUC(C vs B)           = whether a decoder can even TELL the arms apart
                                  (i.e. whether eta is inferable from syndrome alone).

          These are ceilings on a LIKELIHOOD-RATIO TEST over the exact empirical
          distribution. No architecture can beat them on this window.
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from decoder.layout import make_base, parse_layout, detector_map
from decoder.circuit import build_with_loss

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=5)
ap.add_argument("--T", type=int, default=12)
ap.add_argument("--p", type=float, default=0.002)
ap.add_argument("--shots", type=int, default=400_000)
ap.add_argument("--victim", type=int, default=27)
ap.add_argument("--window", type=int, default=2,
                help="rounds of syndrome the test may see, ending at the loss round. "
                     "Use a big number to give the decoder the FULL post-loss flicker record.")
ap.add_argument("--future", type=int, default=0,
                help="rounds AFTER the loss round the test may also see (flicker accumulation).")
ap.add_argument("--no-final", action="store_true",
                help="EXCLUDE the final destructive-readout detectors (they live at t=T). "
                     "They are IN by default: a lost data qubit reads 0 at final readout, so they "
                     "carry direct loss information -- and for a final-round loss they are the only "
                     "evidence that exists after the loss round.")
A = ap.parse_args()


def patterns(circ, layout, ancs, rounds_keep, N):
    dm = detector_map(circ, layout)
    cols = sorted([i for i, (a, r) in dm.items() if a in ancs and r in rounds_keep],
                  key=lambda i: (dm[i][1], dm[i][0]))
    s = circ.compile_detector_sampler().sample(N)[:, cols].astype(np.int64)
    return s @ (1 << np.arange(len(cols), dtype=np.int64)), len(cols)


def hist(pat, K):
    h = np.bincount(pat, minlength=1 << K).astype(float)
    return (h + 0.5) / (h.sum() + 0.5 * (1 << K))        # Krichevsky-Trofimov


def auc(p1, p0):
    o = np.argsort(-(p1 / p0))
    tpr = np.concatenate([[0], np.cumsum(p1[o])])
    fpr = np.concatenate([[0], np.cumsum(p0[o])])
    return float(np.trapezoid(tpr, fpr))


print(f"PHASE 0 CEILING   d={A.d} p={A.p} T={A.T} shots={A.shots} victim={A.victim}")
print(f"window = {A.window} round(s) up to and including the loss round, +{A.future} round(s) after"
      f"   | final destructive readout: {'EXCLUDED' if A.no_final else 'INCLUDED'}\n")

for basis in ("z", "x"):
    b = make_base(A.d, A.T, A.p, basis)
    L = parse_layout(b, A.d)
    ancs = {L.partner[(A.victim, s)] for s in range(4) if (A.victim, s) in L.partner}

    for RL, tag in ((2, "EARLY (round 2)"), (A.T - 1, f"LATE  (round {A.T-1}, final)")):
        hi = RL + 1 + A.future
        if not A.no_final:
            hi = max(hi, A.T + 1)          # t = T is the final destructive readout
        W = [r for r in range(RL - A.window + 1, hi) if 0 <= r <= A.T]
        if A.no_final:
            W = [r for r in W if r < A.T]
        h0, K = patterns(build_with_loss(b, L, {}), L, ancs, W, A.shots)
        if K > 20:
            print(f"  window too wide ({K} detectors); reduce --window/--future"); continue
        H0 = hist(h0, K)

        B = np.zeros(1 << K); C = np.zeros(1 << K)
        for s in range(4):
            if (A.victim, s) not in L.partner:
                continue
            a = L.partner[(A.victim, s)]
            pb, _ = patterns(build_with_loss(b, L, {(RL, s): {A.victim}}),    L, ancs, W, A.shots // 4)
            pc, _ = patterns(build_with_loss(b, L, {(RL, s): {A.victim, a}}), L, ancs, W, A.shots // 4)
            B += hist(pb, K) / 4
            C += hist(pc, K) / 4

        aB, aC, aCB = auc(B, H0), auc(C, H0), auc(C, B)
        arrow = "HELPS" if aC > aB + 0.005 else ("HURTS" if aC < aB - 0.005 else "no change")
        print(f"  memory_{basis.upper()} {tag:<22} K={K:2d} det | "
              f"AUC(detect loss): arm B (eta=0) {aB:.4f}   arm C (eta=1) {aC:.4f}   "
              f"delta {aC-aB:+.4f}  [{arrow}]   |  AUC(C vs B) = {aCB:.4f}")
    print()
