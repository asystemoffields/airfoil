#!/usr/bin/env python3
"""Vine — real-ARC TEST of the analogical COMPOSE layer (Alex's step 3). Adds a CELL:compose path AFTER the pure
cell loop: for tasks the pure invariance/periodic earner MISSES, try COMPOSE(cell-effect, colormap) -- apply a cell
completion then a global recolor, verify. The hypothesis: the beyond-gen6 tasks the pure loop missed are composed
near-misses (periodic/symmetry + a twist), which COMPOSE recovers. Measures beyond_gen6 delta from compose.
Run: /data/llm/.venv/bin/python ground_vine_adapt.py"""
import sys, time
from collections import Counter
import numpy as np
import torch
import rel_dsl as D
import substrate_eye as SE
from open_loop import ranked
from ground_v2_relational import REL_PREDS
from effect_faculty import earn_effect
from ground_arc import recognizer_solve
from ground_arc_v2 import GEN2, GEN6
from train_v2 import V2
import cell_evolve as CE
import schema_adapt as SA

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
import harness
V2NET = V2(); V2NET.load_state_dict(torch.load("learner_v2.pt")); V2NET.eval()


def cell_compose(train, test):
    """THINGY-mod COMPOSE on real tasks: a bounded cell-effect (isometry / eye-period) THEN a colormap, verified."""
    g0 = train[0][0]
    maps = list(CE.iso_maps().items())
    for N in range(10):
        for (axis, P) in CE.detect_periods(g0, N):
            for s in (P, -P):
                mfn = (lambda r, c, H, W, s=s: (r, c+s)) if axis == "W" else (lambda r, c, H, W, s=s: (r+s, c))
                maps.append((f"p{axis}{s}", mfn))
    for name, mapfn in maps:
        for N in range(10):
            inters = [CE._apply_invariance(gi, name, mapfn, N) for gi, _ in train]
            tab = SA._cmap(inters, [np.asarray(go, int) for _, go in train])
            if tab is None or all(k == v for k, v in tab.items()):       # skip identity (= pure invariance, already tried)
                continue
            if all((lambda x: x is not None and np.array_equal(SA._apply_cmap(x, tab), np.asarray(go, int)))(
                    CE._apply_invariance(gi, name, mapfn, N)) for gi, go in test):
                return f"CELL:compose({name})"
    return None


def vine_solve(train, test, topk=10):
    try:
        s, _ = recognizer_solve(V2NET, train, test, 2, 3)
        if s: return "grammar"
    except Exception:
        pass
    r = ranked(train)
    for key in r[:topk]:
        if isinstance(key, (D.Quantify, SE.SubQuantify)):
            prog = D.induce_recolor(key, train)
            if prog is not None and D.verify(prog, train, test): return "earned-sense"
    for outer in [k for k in r[:topk] if isinstance(k, D.Quantify)]:
        for inner in REL_PREDS:
            prog = D.induce_recolor(D.Composed(outer.ch, outer.value, inner, outer.mode), train)
            if prog is not None and D.verify(prog, train, test): return "composition"
    try:
        if earn_effect(train, test) is not None: return "gesture"
    except Exception:
        pass
    try:
        e = CE.earn_cell_effect(train, test)
        if e is not None: return f"CELL:{e.split('(')[0]}"
    except Exception:
        pass
    try:
        c = cell_compose(train, test)                                    # NEW: analogical COMPOSE path
        if c is not None: return c
    except Exception:
        pass
    return None


def main():
    tasks = harness.load_split("arc1-eval")
    t0 = time.time(); solved = {}
    for i, (tid, train, test) in enumerate(tasks):
        try:
            p = vine_solve(train, test)
        except Exception:
            p = None
        if p: solved[tid] = p
        if (i + 1) % 100 == 0:
            nc = sum(1 for v in solved.values() if "compose" in v)
            print(f"  ...{i+1}/400  solved {len(solved)} (compose {nc})  [{time.time()-t0:.0f}s]", flush=True)
    S = set(solved)
    comp = {t: p for t, p in solved.items() if "compose" in p}
    print(f"\nVINE + ANALOGICAL COMPOSE on ARC-1 eval(400) [{time.time()-t0:.0f}s]:")
    print(f"  solved: {len(S)}  by path: {dict(Counter(solved.values()))}")
    print(f"  COMPOSE solves: {len(comp)} -> {comp}")
    print(f"  beyond gen2_base: {len(S - GEN2)}   beyond gen6_base: {len(S - GEN6)}  {sorted(S - GEN6)}")
    print(f"  COMPOSE solves beyond gen6_base: {sorted(set(comp) - GEN6)}")
    print("READ: COMPOSE solves > 0 (esp. beyond_gen6) = the analogical layer recovers composed near-miss tasks the "
          "pure cell loop misses -> 'bend a known schema to fit' adds real-ARC reach. Zero = these misses need other "
          "structural mods (add-step, different base) or richer perception.")


if __name__ == "__main__":
    main()
