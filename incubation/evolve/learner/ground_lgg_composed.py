#!/usr/bin/env python3
"""Branch-B anti-unification CASH-OUT (closes ready-to-scale gate #1).

The lgg.py precondition proved the MECHANISM (schemas re-instantiate per-task 1.00) but the efficiency read 5-6x
on the small grammar (blind finds recolor early). The design said the 10x+ cashes out in the TOO-BIG-TO-ENUMERATE
COMPOSED regime. Here we test that honestly: mine a schema LIBRARY (distinct skeletons that solved >=1 ARC-train
task) and race a RECOGNIZER-RANKED library solver vs BLIND enumeration in the composed space, under a FIXED
induce-budget, in TWO regimes:
  * SHALLOW = 8 single pre-ops (the ground_arc_v2 regime, ~enumerable)
  * DEEP    = 64 two-step pre-op combos (genuinely too big for the budget)
Measure coverage (solved within budget) + induce-calls-among-solved. The library should (a) match coverage with
fewer induces everywhere, and (b) HOLD coverage in DEEP where blind COLLAPSES (budget burned on the pre-op
explosion). That is the 'makes the bigger search tractable' claim = gate #1.
Run: /data/llm/.venv/bin/python ground_lgg_composed.py"""
import sys, time, json
from collections import defaultdict
import numpy as np
import torch
import grammar as G
import grammar_comp as GC
import ground_arc as GA
from train_v2 import V2, task_VO, FEATS, EFFECTS

EVOLVE = "/data/Windows-files/Documents/airfoil/incubation/evolve"
sys.path.insert(0, EVOLVE)
import harness

EFF_IX = {e: i for i, e in enumerate(EFFECTS)}
FEAT_IX = {f: i for i, f in enumerate(FEATS)}
BUDGET = 200


def apply_pre(spec, g):
    if isinstance(spec, str):
        return GC._pre(spec, g)
    g2 = GC._pre(spec[0], g)
    return None if (g2 is None or g2.size == 0) else GC._pre(spec[1], g2)


def induce_comp(spec, eff, dec, feat, train):
    tr2 = []
    for gi, go in train:
        g2 = apply_pre(spec, gi)
        if g2 is None or g2.size == 0:
            return None
        tr2.append((g2, go))
    rel = G.induce(eff, dec, feat, tr2)
    if rel is None:
        return None
    for gi, go in train:
        g2 = apply_pre(spec, gi); out = G.apply_relation(rel, g2) if g2 is not None else None
        if out is None or out.shape != go.shape or not np.array_equal(out, go):
            return None
    return rel


def gen_comp(spec, rel, test):
    for gi, go in test:
        g2 = apply_pre(spec, gi)
        out = G.apply_relation(rel, g2) if (g2 is not None and g2.size) else None
        if out is None or out.shape != go.shape or not np.array_equal(out, go):
            return False
    return True


def mine_library():
    """distinct (effect, decomp, feature) skeletons that solved >=1 ARC-1-train task + colormap."""
    sk = set(); cm = False
    for tid, train, test in harness.load_split("arc1-train"):
        for eff, feat, rel in GA.winning_relations(train, test):
            if eff == "colormap":
                cm = True
            else:
                sk.add((eff, rel["decomp"], feat))
    return sorted(sk), cm


def rank_lib(net, tt, lib):
    """recognizer ranks library skeletons by (effect-logit + feature-logit) on the transformed demos."""
    V, O, m, g = task_VO(tt)
    with torch.no_grad():
        le, lf = net(torch.from_numpy(V[None]), torch.from_numpy(O[None]),
                     torch.from_numpy(m[None]), torch.from_numpy(g[None]))
    le, lf = le[0], lf[0]
    return sorted(lib, key=lambda s: -(le[EFF_IX[s[0]]].item() + lf[FEAT_IX[s[2]]].item()))


def lib_solve(net, train, test, pres, lib, cm, budget):
    n = 0
    for spec in pres:
        tt = [(apply_pre(spec, gi), go) for gi, go in train]
        if any(t[0] is None or t[0].size == 0 for t in tt):
            continue
        if cm:
            n += 1
            if n > budget: return False, n
            rel = G.induce("colormap", None, None, tt)
            if rel is not None and gen_comp(spec, rel, test): return True, n
        for (eff, dec, feat) in rank_lib(net, tt, lib):
            n += 1
            if n > budget: return False, n
            rel = induce_comp(spec, eff, dec, feat, train)
            if rel is not None and gen_comp(spec, rel, test): return True, n
    return False, n


def blind_solve(train, test, pres, cm, budget):
    n = 0
    for spec in pres:
        if cm:
            n += 1
            if n > budget: return False, n
            tt = [(apply_pre(spec, gi), go) for gi, go in train]
            if not any(t[0] is None or t[0].size == 0 for t in tt):
                rel = G.induce("colormap", None, None, tt)
                if rel is not None and gen_comp(spec, rel, test): return True, n
        for eff in ("recolor", "select"):
            for dec in G.DECOMPS:
                for feat in FEATS:
                    n += 1
                    if n > budget: return False, n
                    rel = induce_comp(spec, eff, dec, feat, train)
                    if rel is not None and gen_comp(spec, rel, test): return True, n
    return False, n


def run(net, tasks, pres, lib, cm, label):
    lib_s = set(); lib_n = []; bl_s = set(); bl_n = []
    for tid, train, test in tasks:
        s, n = lib_solve(net, train, test, pres, lib, cm, BUDGET); lib_n.append(n)
        if s: lib_s.add(tid)
        s, n = blind_solve(train, test, pres, cm, BUDGET); bl_n.append(n)
        if s: bl_s.add(tid)
    li = [lib_n[i] for i, t in enumerate(tasks) if t[0] in lib_s]
    bi = [bl_n[i] for i, t in enumerate(tasks) if t[0] in bl_s]
    print(f"\n{label} (budget {BUDGET} induce-calls/task):")
    print(f"  LIBRARY ({len(lib)} schemas, recognizer-ranked): solved {len(lib_s):2d}  | median induce-to-solve {int(np.median(li)) if li else 0}")
    print(f"  BLIND  (full grammar enum):                       solved {len(bl_s):2d}  | median induce-to-solve {int(np.median(bi)) if bi else 0}")
    return lib_s, bl_s


def main():
    net = V2(); net.load_state_dict(torch.load("learner_v2.pt")); net.eval()
    lib, cm = mine_library()
    t0 = time.time()
    print(f"mined library: {len(lib)} distinct skeletons + colormap={cm}  [{time.time()-t0:.0f}s]  (full grammar = {len(FEATS)*len(G.DECOMPS)*2+1})")
    tasks = harness.load_split("arc1-eval")
    SHALLOW = list(GC.PRE_OPS)
    DEEP = [(p1, p2) for p1 in GC.PRE_OPS for p2 in GC.PRE_OPS]
    run(net, tasks, SHALLOW, lib, cm, f"SHALLOW  ({len(SHALLOW)} single pre-ops)")
    run(net, tasks, DEEP, lib, cm, f"DEEP  ({len(DEEP)} two-step pre-op combos -- too big to enumerate)")
    print(f"\n[{time.time()-t0:.0f}s] READ: library>=blind coverage with fewer induces = efficiency cashes; library "
          "HOLDS in DEEP where blind COLLAPSES = library makes the bigger composed search tractable = gate #1.")


if __name__ == "__main__":
    main()
