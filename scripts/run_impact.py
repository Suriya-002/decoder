"""IMPACT EXPERIMENT -- does knowing eta improve DECODING, not just characterization?

This is the experiment that decides whether the eta estimator has downstream value or is a pure
characterization tool. Answer, measured: NO BENEFIT via this strategy, and the reason is
physically interesting.

DESIGN -- the key move that makes this tractable.
  A lossy circuit has NON-DETERMINISTIC detectors AND a non-deterministic logical observable (a
  lost data qubit in the logical support destroys the operator), so Stim cannot build a DEM from
  it, and proper loss decoding needs superstabilizer merging plus logical-operator deformation
  (that machinery IS Perrin arXiv:2603.24237). We sidestep it: SAMPLE from the true lossy circuit,
  but DECODE with the CLEAN DEM whose edge weights are modified per shot to encode what the
  decoder believes about losses. The clean DEM stays deterministic and decomposable. Freeing an
  edge = setting its probability to 0.5, i.e. weight 0.

  Two maps are built once:
    * measurement-error edges, located by detector signature -- a measurement error on ancilla a
      at round r flips exactly D(a,r) and D(a,r+1). 252 of 264 (anc,round) slots located.
    * data-qubit error mechanisms, via Circuit.explain_detector_error_model_errors().
  KNOWN-ANSWER CHECK: rebuilding the DEM with no modifications reproduces the baseline LER
  exactly (0.00198 both ways), confirming the rewrite pipeline is lossless.

FOUR ARMS.
  0  IGNORANT  no loss information at all
  A  eta assumed 0 : knows the data-qubit losses, assumes ancillas alive
  B  eta known     : also discounts the detectors of ancillas believed co-lost
  O  ORACLE        : knows every truly-dead ancilla (upper bound on what any eta estimate buys)
  A and B differ ONLY in ancilla co-loss knowledge, which is exactly what eta tells you. Data
  loss handling is identical in both, so it is common-mode and cancels.

RESULT (d=5, T=12, p=0.002, p_g=0.008, eta=1, 5500 shots):
    arm 0 IGNORANT          LER = 0.34636
    arm A data-loss aware   LER = 0.21945
    arm B eta known         LER = 0.23455
    arm O oracle ancilla    LER = 0.23455

  CONTROL: ignorant - A = +0.12691 +- 0.01667 -> the machinery WORKS (~15 sigma). Loss
  information massively improves decoding, so a null between A and B is a REAL null, not a
  broken pipeline.

  A - B = -0.01509 +- 0.01565, consistently NEGATIVE across runs (-0.014 at 2000 shots, -0.0151
  at 5500). Knowing which ancillas were co-lost does not help and slightly hurts.

  CONSISTENCY CHECK: at eta=1, arm B == arm O EXACTLY (1290/5500 both), because every partner
  ancilla really is co-lost so the decoder's belief equals the truth.

WHY IT FAILS, AND WHY THAT IS INTERESTING.
  Freeing the co-lost ancilla's measurement edge treats its reading as ERASED. But a dead ancilla
  reads m = 0 DETERMINISTICALLY -- that is the entire basis of this project's estimator. Its
  detector d = 0 XOR m(r-1) still carries information about m(r-1). Discarding it throws away
  more than it gains. The dead ancilla's reading is informative, so erasing it is exactly the
  wrong move.

SCOPE -- DO NOT OVERCLAIM. This tests ONE strategy for using eta (discount the corrupted
detector). Perrin uses eta differently, inside superstabilizer / loss-detection construction.
This is evidence that eta has no value for THIS decoding strategy; it is NOT evidence that eta
is useless for decoding in general.

USAGE:  python scripts\run_impact.py [p_g] [shots] [eta]
"""
import sys, time; sys.path.insert(0,"src")
import numpy as np, stim, pymatching
from decoder.layout import make_base, parse_layout, detector_map
from decoder.fast import prerender, build_fast
from decoder.loss_models import sample_gate

d,T,p = 5,12,0.002
PG = float(sys.argv[1]) if len(sys.argv)>1 else 0.004
SHOTS = int(sys.argv[2]) if len(sys.argv)>2 else 3000
ETA = float(sys.argv[3]) if len(sys.argv)>3 else 1.0

b=make_base(d,T,p,"z"); L=parse_layout(b,d); pre,anc=prerender(b),set(L.anc)
dem=b.detector_error_model(decompose_errors=True)
dmap=detector_map(b,L); det_of={v:k for k,v in dmap.items()}
errs=[[i.args_copy()[0], i.targets_copy()] for i in dem.flattened() if i.type=="error"]

# measurement-error edge index by (ancilla, round)
sig2={}
for i,(pr,tg) in enumerate(errs):
    dts=tuple(sorted(t.val for t in tg if t.is_relative_detector_id()))
    if len(dts)==2 and not any(t.is_logical_observable_id() for t in tg):
        sig2.setdefault(dts,[]).append(i)
meas_edge={}
for a in L.anc:
    for r in range(T-1):
        i1,i2=det_of.get((a,r)),det_of.get((a,r+1))
        if i1 is not None and i2 is not None:
            k=tuple(sorted((i1,i2)))
            if k in sig2: meas_edge[(a,r)]=sig2[k]

# data-qubit error mechanisms
sig2e={}
for i,(pr,tg) in enumerate(errs):
    dts=tuple(sorted(t.val for t in tg if t.is_relative_detector_id()))
    lgs=tuple(sorted(t.val for t in tg if t.is_logical_observable_id()))
    sig2e.setdefault((dts,lgs),[]).append(i)
qerr={q:set() for q in L.data}
for e in b.explain_detector_error_model_errors(dem_filter=None, reduce_to_one_representative_error=True):
    dts=tuple(sorted(t.dem_target.val for t in e.dem_error_terms if t.dem_target.is_relative_detector_id()))
    lgs=tuple(sorted(t.dem_target.val for t in e.dem_error_terms if t.dem_target.is_logical_observable_id()))
    idxs=sig2e.get((dts,lgs))
    if not idxs: continue
    qs=set()
    for loc in e.circuit_error_locations:
        for gt in loc.flipped_pauli_product: qs.add(gt.gate_target.qubit_value)
    for q in qs & set(L.data): qerr[q].update(idxs)

_cache={}
def matching_for(free):
    key=frozenset(free)
    if key in _cache: return _cache[key]
    out=stim.DetectorErrorModel()
    for i,(pr,tg) in enumerate(errs):
        out.append("error", 0.5 if i in key else pr, tg)
    m=pymatching.Matching.from_detector_error_model(out)
    if len(_cache)<400: _cache[key]=m
    return m

rng=np.random.default_rng(7)
res={"0":[0,0],"A":[0,0],"B":[0,0],"O":[0,0]}
t0=time.time(); dset=set(L.data)
for _ in range(SHOTS):
    ev,truth=sample_gate(L,T,PG,ETA,rng)
    c=build_fast(pre,anc,ev)
    det,obs=c.compile_detector_sampler().sample(1, separate_observables=True)
    dlost={q for (q,r,s) in truth if q in dset}
    # ARM A: data losses only (ancillas assumed alive)
    freeA=set()
    for q in dlost: freeA|=qerr[q]
    # ARM B: also discount detectors of ancillas believed co-lost (eta tells you this)
    freeB=set(freeA)
    for (q,r,s) in truth:
        if q not in dset: continue
        a=L.partner.get((q,s))
        if a is None: continue
        if rng.random() < ETA:                     # decoder's eta-informed belief
            for rr in (r-1,r):
                if (a,rr) in meas_edge: freeB|=set(meas_edge[(a,rr)])
    # ORACLE: knows data losses AND every truly-dead ancilla
    freeO=set(freeA)
    for (q,r,s) in truth:
        if q in dset: continue
        for rr in (r-1,r):
            if (q,rr) in meas_edge: freeO|=set(meas_edge[(q,rr)])
    for tag,fr in (("0",set()),("A",freeA),("B",freeB),("O",freeO)):
        pred=matching_for(fr).decode(det[0])
        res[tag][0]+=int(pred[0]!=obs[0,0]); res[tag][1]+=1

print(f"IMPACT EXPERIMENT  d={d} T={T} p={p} p_g={PG} eta_true={ETA}  {SHOTS} shots  ({time.time()-t0:.0f}s)")
print(f"  Arm 0 (IGNORANT: no loss info at all)      : LER = {res['0'][0]/res['0'][1]:.5f}  ({res['0'][0]}/{res['0'][1]})")
print(f"  Arm A (eta assumed 0, ancillas assumed alive): LER = {res['A'][0]/res['A'][1]:.5f}  ({res['A'][0]}/{res['A'][1]})")
print(f"  Arm B (eta known, co-lost ancillas discounted): LER = {res['B'][0]/res['B'][1]:.5f}  ({res['B'][0]}/{res['B'][1]})")
print(f"  Arm O (ORACLE: knows every dead ancilla)   : LER = {res['O'][0]/res['O'][1]:.5f}  ({res['O'][0]}/{res['O'][1]})")
p0=res['0'][0]/res['0'][1]
pa,pb=res['A'][0]/res['A'][1], res['B'][0]/res['B'][1]
n0=res['0'][1]; se0=np.sqrt(p0*(1-p0)/n0+pa*(1-pa)/n0)
print(f"\n  CONTROL  ignorant - A = {p0-pa:+.5f} +- {1.96*se0:.5f}  ({'machinery WORKS' if p0-pa>1.96*se0 else 'NO RESPONSE -> machinery suspect'})")
n=res['A'][1]
se=np.sqrt(pa*(1-pa)/n+pb*(1-pb)/n)
print(f"  difference A-B = {pa-pb:+.5f} +- {1.96*se:.5f}   ({'B BETTER' if pa-pb>1.96*se else 'not significant'})")
