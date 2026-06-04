#!/usr/bin/env python3
"""Branch-B scale-prep BOX-PREP 2 — the GEOMETRY-COVERAGE FALSIFIER (gates the GPU spend).

Recolor/select preserve shape, so RESHAPING ARC tasks (input.shape != output.shape) are 0/0 grammar-solvable
today. Question: does composing a SHAPE-CHANGING pre-op (tile/scale/crop/downscale/trim + geometry) BEFORE the
grammar relation make any reshaping task NEWLY expressible = a beyond_gen6 on the box? If ~0, the geometry head's
COVERAGE contribution is FALSIFIED before any Kaggle spend. Also: does the synthetic-trained V3-GEO rank the
solving pre-op top-3 on these REAL reshaping tasks (structural transfer)? Run: /data/llm/.venv/bin/python ground_v3_geo.py"""
import sys, time, json
import numpy as np
import torch
import grammar as G
import grammar_comp as GC
from train_v3_geo import V3GEO, task_stats, BANK, BANK_IX, _pre
from ground_arc_v2 import gen_comp, GEN6

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
import harness

FEATS = G.FEATURE_NAMES


def is_reshaping(train, test):
    return any(np.asarray(gi).shape != np.asarray(go).shape for gi, go in list(train) + list(test))


def solve_reshape(train, test):
    """blind enumerate shape-changing-pre x grammar; pre must produce the output shape (recolor case). -> (solved, pre)."""
    gi0 = np.asarray(train[0][0], int); go0 = np.asarray(train[0][1], int)
    for pre in BANK:
        gp = _pre(pre, gi0)
        if gp is None or gp.size == 0 or gp.shape != go0.shape:   # pre must reshape input -> output shape
            continue
        if pre == "identity":
            continue                                              # identity = no reshape, not the geometry contribution
        rel = G.induce("colormap", None, None, [(_pre(pre, np.asarray(g, int)), go) for g, go in train])
        if rel is not None and gen_comp((pre, rel), test):
            return True, pre
        for eff in ("recolor", "select"):
            for dec in G.DECOMPS:
                for feat in FEATS:
                    comp = GC.induce_composed(pre, eff, dec, feat, train)
                    if comp is not None and gen_comp(comp, test):
                        return True, pre
    return False, None


def main():
    net = V3GEO(); net.load_state_dict(torch.load("learner_v3_geo.pt")); net.eval()
    t0 = time.time()
    n_reshape = 0; solved = []; v3_top3 = 0
    for split in ("arc1-train", "arc1-eval", "arc2-train"):
        for tid, train, test in harness.load_split(split):
            if not is_reshaping(train, test):
                continue
            n_reshape += 1
            try:
                ok, pre = solve_reshape(train, test)
            except Exception:
                continue
            if ok:
                solved.append((split, tid, pre))
                # V3-GEO structural transfer: does it rank the solving pre-op top-3 on this REAL task?
                with torch.no_grad():
                    S = torch.from_numpy(task_stats(train)[None])
                    top3 = net(S)[0].topk(3).indices.tolist()
                v3_top3 += int(BANK_IX[pre] in top3)
    print(f"GEOMETRY-COVERAGE FALSIFIER (ARC-1 train+eval + ARC-2 train) [{time.time()-t0:.0f}s]")
    print(f"  reshaping tasks (input.shape != output.shape): {n_reshape}")
    print(f"  NEWLY EXPRESSIBLE via shape-changing pre-op + grammar: {len(solved)}")
    print(f"     all beyond_gen6 by construction (gen6 had 0 reshaping solves): {[t for _s,t,_p in solved]}")
    print(f"     solving pre-ops: {[(t,p) for _s,t,p in solved]}")
    print(f"  V3-GEO ranks the solving pre-op top-3 on these REAL reshaping tasks: {v3_top3}/{len(solved)}")
    print("READ: NEWLY-EXPRESSIBLE > 0 = shape-changing composition lifts coverage on reshaping tasks (the geometry "
          "head earns its place; new beyond_gen6) AND V3-GEO transfers structurally to real reshaping tasks. = 0 = "
          "geometry-head coverage contribution FALSIFIED before Kaggle.")


if __name__ == "__main__":
    main()
