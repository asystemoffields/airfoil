#!/usr/bin/env python3
"""Branch-B push #2 — SHAPE-EFFECTS (counting/construction), the biggest family the hand-authored families miss.

Tests the EXPRESSIVENESS question deterministically (global count-features are few -> enumerate+verify, no learner
needed yet): does 'output = f(count)' reach any ARC-1 eval task BEYOND gen6_base (the full families)? Effects:
  * solid block (k_h, k_w) of a color, where k = a global count feature
  * bar (k x 1 / 1 x k)
  * tile the input k times (k x 1 / 1 x k / k x k)
The construction color is induced (read from the demo output + checked consistent). If this adds beyond_gen6, the
new expressiveness is real and worth wiring into the learner's grammar; if 0, the families already cover it.
Run: /data/llm/.venv/bin/python ground_shape.py"""
import sys, time, json
import numpy as np
import grammar as G

EVOLVE = "/data/Windows-files/Documents/airfoil/incubation/evolve"
sys.path.insert(0, EVOLVE)
import harness

GEN2 = set(json.load(open(EVOLVE + "/runs/gen2_base_solved.json"))["arc1_eval"]["solved"])
BEYOND14 = set("0a2355a6 21f83797 281123b4 3194b014 358ba94e 37d3e8b2 6a11f6da 84db8fc4 "
               "9a4bb226 aa18de87 cd3c21df d282b262 e0fb7511 e872b94a".split())
GEN6 = GEN2 | BEYOND14


def counts(g):
    g = np.asarray(g, int)
    o4 = G.objects(g, 4, False); o8 = G.objects(g, 8, False)
    cols = [int(c) for c in np.unique(g) if c != 0]
    return {"n_obj4": len(o4), "n_obj8": len(o8), "n_colors": len(cols),
            "max_size": max((o["size"] for o in o4), default=0),
            "min_size": min((o["size"] for o in o4), default=0),
            "n_cells": int((g != 0).sum())}

CFEATS = ["n_obj4", "n_obj8", "n_colors", "max_size", "min_size", "n_cells"]


def builders(k, g):
    """candidate output grids for a count k (color filled later by caller for solids; tiles use input color)."""
    g = np.asarray(g, int)
    out = {}
    if 1 <= k <= 30:
        out["square"] = (k, k); out["barv"] = (k, 1); out["barh"] = (1, k)
        # tiles use the input itself
        out["tile_v"] = np.tile(g, (k, 1)) if k * g.shape[0] <= 30 else None
        out["tile_h"] = np.tile(g, (1, k)) if k * g.shape[1] <= 30 else None
        out["tile_sq"] = np.tile(g, (k, k)) if k * g.shape[0] <= 30 and k * g.shape[1] <= 30 else None
    return out


def histograms(g):
    """content-conditioned: per-color count -> bars. Returns {name: grid} candidate outputs."""
    g = np.asarray(g, int)
    cols = sorted(int(c) for c in np.unique(g) if c != 0)
    if not cols:
        return {}
    objs = G.objects(g, 4, True)
    outs = {}
    for kind in ("cells", "objs"):
        cnt = {c: (int((g == c).sum()) if kind == "cells" else sum(o["color"] == c for o in objs)) for c in cols}
        mx = max(cnt.values())
        if not (1 <= mx <= 30):
            continue
        for order in ("val", "asc", "desc"):
            sc = cols if order == "val" else sorted(cols, key=lambda c: cnt[c], reverse=(order == "desc"))
            h = np.zeros((len(sc), mx), int)
            for i, c in enumerate(sc):
                h[i, :cnt[c]] = c
            outs[f"hist_h:{kind}:{order}"] = h
            outs[f"hist_v:{kind}:{order}"] = h.T
    return outs


def _match(o, t):
    return o is not None and not isinstance(o, tuple) and getattr(o, "shape", None) == t.shape and np.array_equal(o, t)


def solve_shape(train, test):
    # content-conditioned: histograms (per-color counts -> bars)
    names = set(histograms(train[0][0]))
    for nm in names:
        if all(_match(histograms(gi).get(nm), go) for gi, go in train) and \
           all(_match(histograms(gi).get(nm), go) for gi, go in test):
            return True, nm
    """enumerate (count-feature, builder, color); verify across train; return True if a verified rule generalizes."""
    # solids: shape from count, single color (induced from the first demo's output)
    for feat in CFEATS:
        for shp in ("square", "barv", "barh"):
            def build_solid(gi, c):
                k = counts(gi)[feat]; b = builders(k, gi).get(shp)
                if not isinstance(b, tuple): return None
                return np.full(b, c, int)
            # induce color from first train output (must be a single color)
            go0 = train[0][1]
            cset = set(np.unique(go0).tolist())
            for c in cset:
                ok = all((lambda o, t: o is not None and o.shape == t.shape and np.array_equal(o, t))
                         (build_solid(gi, c), go) for gi, go in train)
                if ok and all((lambda o, t: o is not None and o.shape == t.shape and np.array_equal(o, t))
                              (build_solid(gi, c), go) for gi, go in test):
                    return True, f"solid:{shp}:{feat}:{c}"
        # tiles: shape = input tiled k times (color = input)
        for shp in ("tile_v", "tile_h", "tile_sq"):
            def build_tile(gi):
                k = counts(gi)[feat]; return builders(k, gi).get(shp)
            ok = all((lambda o, t: o is not None and not isinstance(o, tuple) and o.shape == t.shape and np.array_equal(o, t))
                     (build_tile(gi), go) for gi, go in train)
            if ok and all((lambda o, t: o is not None and not isinstance(o, tuple) and o.shape == t.shape and np.array_equal(o, t))
                          (build_tile(gi), go) for gi, go in test):
                return True, f"tile:{shp}:{feat}"
    return False, None


def main():
    tasks = harness.load_split("arc1-eval")
    t0 = time.time(); solved = {}
    for tid, train, test in tasks:
        s, how = solve_shape(train, test)
        if s: solved[tid] = how
    print(f"SHAPE-EFFECTS (count-construction) on ARC-1 eval(400) [{time.time()-t0:.0f}s]")
    print(f"  solved: {len(solved)}")
    print(f"  beyond gen2_base (retrieval): {len(set(solved)-GEN2)}  {sorted(set(solved)-GEN2)}")
    print(f"  beyond gen6_base (families):  {len(set(solved)-GEN6)}  {sorted(set(solved)-GEN6)}")
    for tid in sorted(set(solved) - GEN6):
        print(f"     NEW {tid}: {solved[tid]}")
    print("READ: beyond_gen6 > 0 = shape-effects reach the counting/construction family the families miss -> "
          "wire this effect class into the learner's grammar. =0 = the families already cover it; need genuinely "
          "novel relations (anti-unification / scale).")


if __name__ == "__main__":
    main()
