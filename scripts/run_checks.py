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
from decoder.events import clean_events
from decoder.pij import pij_pair
from decoder.codeagnostic import parse_generic, raw_measurement_index, detector_index, force_ancilla_dead

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

# 10 -- THE TWO THEORY ANCHORS, via FORCED INJECTION at a BULK round (Rule 1: never round 0).
#   Sampling the loss configs made this FLAKY: round-0 losses are anomalous (P(m=0|alive)=0.668,
#   not 0.5 -- FINDINGS section 7), so a sampled ALIVE class carries a ~+0.007 bias sitting right
#   at the tolerance. Forced injection at a fixed round >= 1 gives the theory value to 4 decimals
#   and is seed- and platform-stable.
#     ALIVE partner on a truncated stabilizer -> coin flip      -> P(m=0) = 0.5    (any eps; the
#                                                                    alive atom is not affected by eps)
#     CO-LOST partner -> no atom, m=0 up to false-bright eps    -> P(m=0) = 1 - eps
for eps in (0.0, 0.02):
    b = make_base(5, 12, 0.002, "z"); L = parse_layout(b, 5)
    pre, anc = prerender(b), set(L.anc); mi = measurement_index(b)
    R = 6                                                   # a bulk round, far from both boundaries
    acc = {True: [0, 0], False: [0, 0]}
    for q in L.data:
        for s_ in range(4):
            if (q, s_) not in L.partner: continue
            a = L.partner[(q, s_)]
            if (a, R) not in mi: continue
            for co, ev in ((False, {(R, s_): {q}}), (True, {(R, s_): {q, a}})):
                m = build_fast(pre, anc, ev, false_bright=eps).compile_sampler().sample(4000)
                acc[co][0] += int((~m[:, mi[(a, R)]]).sum()); acc[co][1] += 4000
    for co, target in ((False, 0.5), (True, 1.0 - eps)):
        z, n = acc[co]
        P = z / n; se = np.sqrt(max(P * (1 - P), 1e-12) / n)
        ok = abs(P - target) < max(4 * se, 0.003)
        check(f"eps={eps:.2f}: P(m=0 | {'CO-LOST' if co else 'ALIVE  '}) == {target:.2f} (theory, forced inj r={R})",
              ok, f"measured={P:.5f} +-{1.96*se:.5f}  n={n}")

# 11 -- RULE 2: the clean-event list removes the multi-loss bias that scales with p_g.
#      At eta=0 the estimator must return 0. The NAIVE per-round scan is biased high at elevated
#      p_g; clean_events (first-truncation-round-only) must stay unbiased. Stated as a RATIO
#      (clean at least 3x closer to 0 than naive) so it is not a knife-edge tolerance.
b = make_base(5, 12, 0.002, "z"); L = parse_layout(b, 5)
pre, anc = prerender(b), set(L.anc); mi = measurement_index(b); dst = set(L.data)
ANBR = {a: set() for a in L.anc}
for s_ in range(4):
    for (dq, a) in L.pairs[s_]: ANBR[a].add(dq)
for pg, sh in ((0.010, 4000), (0.020, 3000)):
    zc = nc = zn = nn = 0
    for _ in range(sh):
        ev, tr = sample_gate(L, 12, pg, 0.0, rng)          # eta = 0 -> estimator MUST return 0
        if not tr: continue
        m = build_fast(pre, anc, ev).compile_sampler().sample(1)[0]
        for (q, r, sub, a) in clean_events(L, 12, tr, ANBR):
            if (a, r) in mi: zc += int(m[mi[(a, r)]] == 0); nc += 1
        for (q, r, sub) in tr:
            if q not in dst or r < 1: continue
            a = L.partner.get((q, sub))
            if a is None or (a, r) not in mi: continue
            zn += int(m[mi[(a, r)]] == 0); nn += 1
    hc = 2 * (zc / nc - 0.5); hn = 2 * (zn / nn - 0.5)
    sc = 2 * np.sqrt((zc / nc) * (1 - zc / nc) / nc)
    ok = (abs(hc) < 3 * sc + 0.01) and (abs(hc) < abs(hn) / 3)
    check(f"p_g={pg:.3f}: clean_events removes the naive bias", ok,
          f"clean eta_hat={hc:+.4f} +-{1.96*sc:.4f}   naive eta_hat={hn:+.4f}   ({abs(hn)/max(abs(hc),1e-6):.0f}x closer)")

# 12 -- the BKY/Spitz p_ij estimator (Eq. 36) must recover a PLANTED correlated-flip rate.
#      Validates the baseline implementation before it is used in the head-to-head.
for p1, p2, p12 in ((0.10, 0.15, 0.05), (0.20, 0.05, 0.12), (0.02, 0.02, 0.03)):
    N = 300000
    d1 = rng.random(N) < p1; d2 = rng.random(N) < p2; both = rng.random(N) < p12
    d1 = d1 ^ both; d2 = d2 ^ both
    hat = pij_pair(d1, d2); se = 0.002
    check(f"p_ij recovers planted p12={p12:.2f} (BKY Eq. 36)", abs(hat - p12) < 0.004,
          f"recovered={hat:.4f}")

# 13 -- GENERALITY: the gauge law is not surface-code-specific. On a DIFFERENT code (unrotated
#      surface), a dead ancilla still reads raw m=0, and its detector still fires ~0 on the
#      deterministic-stabilizer type and ~0.5 on the projected type.
for spec in ("surface_code:unrotated_memory_z",):
    cc0 = stim.Circuit.generated(spec, distance=5, rounds=12)
    info = parse_generic(cc0); mi = raw_measurement_index(cc0)
    dinv = {v: k for k, v in detector_index(cc0, info).items()}
    R = 6
    for t, group, want_det in (("Z", info["anc_z"], 0.0), ("X", info["anc_x"], 0.5)):
        raws, dets = [], []
        for a in group:
            if (a, R) not in mi or (a, R) not in dinv: continue
            c2 = force_ancilla_dead(cc0, a, R)
            raws.append(float((~c2.compile_sampler().sample(3000)[:, mi[(a, R)]]).mean()))
            dets.append(float(c2.compile_detector_sampler().sample(3000)[:, dinv[(a, R)]].mean()))
        raw_ok = abs(np.mean(raws) - 1.0) < 1e-9
        det_ok = abs(np.mean(dets) - want_det) < 0.03
        check(f"generality (unrotated): {t}-type dead ancilla raw=1.0, detector~{want_det}", raw_ok and det_ok,
              f"raw P(m=0)={np.mean(raws):.4f}  detector fires={np.mean(dets):.4f}")

# 30-31 -- REALISM: under refined neutral-atom physics (survivor biased-Z + Evered Pauli budget +
#      false-bright eps), the estimator anchors must hold. Survivor-Z acts on the atom that STAYS,
#      so it cannot touch the lost atom's ancilla reading. Forced injection, eps=0.01.
rb = make_base(5, 12, 0.002, "z"); rL = parse_layout(rb, 5)
rpre, ranc = prerender(rb), set(rL.anc); rmi = measurement_index(rb)
rpartner = {}
for rs in range(4):
    for (rdq, ra) in rL.pairs[rs]:
        rpartner[(rdq, rs)] = ra; rpartner[(ra, rs)] = rdq
rR, racc = 6, {True: [0, 0], False: [0, 0]}
for rq in rL.data:
    for rs in range(4):
        if (rq, rs) not in rL.partner: continue
        ra = rL.partner[(rq, rs)]
        if (ra, rR) not in rmi: continue
        for rco, rev in ((False, {(rR, rs): {rq}}), (True, {(rR, rs): {rq, ra}})):
            rm = build_fast(rpre, ranc, rev, false_bright=0.01, partner=rpartner,
                            survivor_z=0.5, pz=0.00024, pxy=0.00003).compile_sampler().sample(4000)[:, rmi[(ra, rR)]]
            racc[rco][0] += int((~rm).sum()); racc[rco][1] += 4000
for rco, rtgt in ((False, 0.5), (True, 0.99)):
    rz, rn = racc[rco]; rP = rz / rn
    check(f"realism: P(m=0|{'co-lost' if rco else 'alive'}) == {rtgt} (survivor-Z+Pauli+eps)",
          abs(rP - rtgt) < 0.012, f"measured={rP:.4f}")

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
sys.exit(1 if FAIL else 0)
