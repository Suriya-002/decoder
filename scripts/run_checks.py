"""Known-answer checks. Every one of these MUST pass before a single line of decoder is written.

If any check fails, the harness is wrong and every number downstream is garbage.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, stim
from decoder.layout import make_base, parse_layout, detector_map
from decoder.circuit import build_with_loss
from decoder.loss_models import sample_gate
from decoder.fast import prerender, build_fast
from decoder.raw import measurement_index

FAIL = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok: FAIL.append(name)

print("KNOWN-ANSWER CHECKS  (rotated surface code, Stim-generated base)\n")

# 1 -- noiseless & lossless => literally zero detector clicks
b = make_base(5, 10, 0.0); L = parse_layout(b, 5)
n = build_with_loss(b, L, {}).compile_detector_sampler().sample(2000).sum()
check("noiseless+lossless -> 0 detector clicks", n == 0, f"clicks={n}")

# 2 -- with no loss, the rebuilt circuit must be BYTE-IDENTICAL to the flattened base
b = make_base(5, 10, 0.001); L = parse_layout(b, 5)
c = build_with_loss(b, L, {})
same = str(b.flattened()) == str(c)
check("no-loss rebuild == flattened base", same,
      f"DEM errors base={b.flattened().detector_error_model().num_errors} rebuilt={c.detector_error_model().num_errors}")

# 3 -- every 2-qubit gate is data<->ancilla. NO data-data CZ exists in a surface code.
bad = [(dq, a) for P in L.pairs for (dq, a) in P if dq not in L.data or a not in L.anc]
edges = [e for P in L.pairs for e in P]
check("all CZ pairs are data<->ancilla", not bad, f"{len(edges)} gates, {len(set(edges))} unique Tanner edges, {len(bad)} violations")

# 4 -- one lost data qubit => EXACTLY its Tanner-partner ancillas flicker at ~0.5, nothing else
b = make_base(5, 12, 0.0); L = parse_layout(b, 5)
for v in (27, L.data[0], L.data[4]):
    nb = {L.partner[(v, s)] for s in range(4) if (v, s) in L.partner}
    c = build_with_loss(b, L, {(0, 0): {v}})
    dm = detector_map(c, L); smp = c.compile_detector_sampler().sample(4000)
    rate = {}
    for i, (a, r) in dm.items():
        if 5 <= r <= 10: rate.setdefault(a, []).append(smp[:, i].mean())
    rate = {a: float(np.mean(x)) for a, x in rate.items()}
    flick = {a for a, x in rate.items() if x > 0.05}
    leak = max((x for a, x in rate.items() if a not in nb), default=0.0)
    check(f"data loss q={v} flickers exactly its partners", flick == nb,
          f"partners={sorted(nb)} flicker={sorted(flick)} max_nonpartner_rate={leak:.4f}")

# 5 -- matched marginals: per-atom loss rate must NOT depend on eta
rng = np.random.default_rng(0); p_g, R = 0.01, 40
for eta in (0.0, 0.25, 0.5, 1.0):
    tot = 0
    for _ in range(200):
        _, t = sample_gate(L, R, p_g, eta, rng); tot += len(t)
    per_gate = tot / (200 * R * len(edges) * 2)
    check(f"marginal per-atom loss @ eta={eta:.2f} == p_g", abs(per_gate - p_g) < 0.0008,
          f"measured={per_gate:.5f} target={p_g}")

# 6 -- eta=1 => every loss is a CZ PAIR; eta~0 => the two atoms of one gate are NEVER both lost.
#      (a (round,substep) slot holds losses from MANY gates, so we must test Tanner-pair containment,
#       not slot size. Within a layer each qubit is in exactly one gate, so a contained pair is
#       unambiguously a co-loss of that gate.)
for eta in (0.0, 0.5, 1.0):
    ev, truth = sample_gate(L, 400, 0.02, eta, rng)
    pairs_hit = atoms = 0
    for (r, s), lost in ev.items():
        for (dq, a) in L.pairs[s]:
            if dq in lost and a in lost: pairs_hit += 1
        atoms += len(lost)
    frac = 2 * pairs_hit / atoms if atoms else 0.0     # fraction of lost atoms that are in a co-lost pair
    ok = abs(frac - eta) < 0.03
    check(f"eta={eta:.1f} -> co-lost fraction == eta", ok,
          f"measured co-lost atom fraction={frac:.4f}  target={eta}  ({pairs_hit} pairs / {atoms} atoms)")

# 7 -- HARD INVARIANT: loss must NEVER change the measurement or detector count, in ANY basis.
#      Deleting a measurement shifts every downstream DETECTOR rec[] offset and silently
#      corrupts the circuit. THIS is the check that catches MX/M handling bugs.
for basis in ("z", "x"):
    b = make_base(5, 12, 0.002, basis); L = parse_layout(b, 5)
    nm, nd = b.num_measurements, b.num_detectors
    ok = True
    for v in (27, L.data[0]):
        for s_ in range(4):
            if (v, s_) not in L.partner: continue
            a = L.partner[(v, s_)]
            for cfg in ({(6, s_): {v}}, {(6, s_): {v, a}}, {(11, s_): {v, a}}, {(0, s_): {a}}):
                c = build_with_loss(b, L, cfg)
                if c.num_measurements != nm or c.num_detectors != nd: ok = False
    check(f"memory_{basis} : loss never changes measurement/detector count", ok,
          f"base has {nm} measurements, {nd} detectors")

# 8 -- flicker check in BOTH bases (the original only ever ran memory_z)
for basis in ("z", "x"):
    b = make_base(5, 12, 0.0, basis); L = parse_layout(b, 5)
    nb = {L.partner[(27, s_)] for s_ in range(4)}
    c = build_with_loss(b, L, {(0, 0): {27}})
    dm = detector_map(c, L); smp = c.compile_detector_sampler().sample(4000)
    rate = {}
    for i, (a, r) in dm.items():
        if 5 <= r <= 10: rate.setdefault(a, []).append(smp[:, i].mean())
    fl = {a for a, x in rate.items() if np.mean(x) > 0.05}
    check(f"memory_{basis} : data loss flickers exactly its partners", fl == nb,
          f"partners={sorted(nb)} flicker={sorted(fl)}")

# 9 -- the FAST builder must be byte-identical to the audited reference builder.
#      An optimisation is never trusted without this. 200 random loss configs, random eta.
for basis in ("z", "x"):
    b = make_base(5, 12, 0.002, basis); L = parse_layout(b, 5)
    pre = prerender(b); anc = set(L.anc); bad = 0
    for _ in range(200):
        ev, _ = sample_gate(L, 12, 0.002, float(rng.random()), rng)
        if str(build_fast(pre, anc, ev).flattened()) != str(build_with_loss(b, L, ev).flattened()):
            bad += 1
    check(f"memory_{basis} : fast builder == reference builder", bad == 0,
          f"{200-bad}/200 random loss configs byte-identical")

# 10 -- THE TWO THEORY ANCHORS of the raw-record estimator. These are not fits.
#   ALIVE partner on a truncated stabilizer -> m is a coin flip                  -> P(m=0) = 0.5
#   CO-LOST partner -> no atom, no |1> population, m = 0 (up to false-bright eps) -> P(m=0) = 1-eps
for eps in (0.0, 0.02):
    b = make_base(5, 12, 0.002, "z"); L = parse_layout(b, 5)
    pre, anc = prerender(b), set(L.anc)
    mi = measurement_index(b); dst = set(L.data)
    acc = {True: [0, 0], False: [0, 0]}
    for _ in range(30000):
        for eta in (0.0, 1.0):
            ev, tr = sample_gate(L, 12, 0.00054, eta, rng)
            dl = [(q, r, s_) for (q, r, s_) in tr if q in dst]
            if len(dl) != 1: continue
            q, r, s_ = dl[0]; a = L.partner.get((q, s_))
            if a is None or (a, r) not in mi: continue
            if any(x not in {q, a} for (x, _, _) in tr): continue     # clean single-loss shots only
            m = build_fast(pre, anc, ev, false_bright=eps).compile_sampler().sample(1)[0]
            co = a in ev.get((r, s_), ())
            acc[co][0] += int(m[mi[(a, r)]] == 0); acc[co][1] += 1
    for co, target in ((False, 0.5), (True, 1.0 - eps)):
        z, n = acc[co]
        P = z / n; se = np.sqrt(max(P * (1 - P), 1e-12) / n)
        ok = abs(P - target) < max(3 * se, 0.004)
        check(f"eps={eps:.2f}: P(m=0 | {'CO-LOST' if co else 'ALIVE  '}) == {target:.2f} (theory)", ok,
              f"measured={P:.4f} +-{1.96*se:.4f}  n={n}")

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
sys.exit(1 if FAIL else 0)
