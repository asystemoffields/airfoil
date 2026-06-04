#!/usr/bin/env python3
"""Heuristic auto-categorization of the 318 missed arc1-train tasks into transformation FAMILIES,
calibrated to mechanisms observed by direct inspection. Read-only (analysis, not solve-time)."""
import sys, os, json
import numpy as np
from collections import deque, Counter

HERE = "/data/Windows-files/Documents/airfoil/incubation/evolve"
TRAIN_DIR = "/data/arc/data/training"
res = json.load(open(os.path.join(HERE, "runs", "gen2_base_solved.json")))
missed = res["arc1_train"]["missed"]


def load(tid):
    d = json.load(open(os.path.join(TRAIN_DIR, tid + ".json")))
    return [(np.array(p["input"]), np.array(p["output"])) for p in d["train"]]


def bg(g):
    v, c = np.unique(g, return_counts=True); return int(v[c.argmax()])


def comps(g, b, diag=True):
    h, w = g.shape; seen = np.zeros((h, w), bool)
    nb = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] if diag else [(-1,0),(1,0),(0,-1),(0,1)]
    out = []
    for i in range(h):
        for j in range(w):
            if g[i,j] != b and not seen[i,j]:
                cells=[]; q=deque([(i,j)]); seen[i,j]=True
                while q:
                    a,bb=q.popleft(); cells.append((a,bb))
                    for di,dj in nb:
                        x,y=a+di,bb+dj
                        if 0<=x<h and 0<=y<w and g[x,y]!=b and not seen[x,y]:
                            seen[x,y]=True; q.append((x,y))
                out.append(cells)
    return out


def has_gridlines(g):
    h, w = g.shape
    full_rows = sum(1 for r in range(h) if len(set(g[r].tolist())) == 1 and g[r,0] != bg(g))
    full_cols = sum(1 for c in range(w) if len(set(g[:,c].tolist())) == 1 and g[0,c] != bg(g))
    return full_rows >= 1 or full_cols >= 1


def categorize(tid):
    tr = load(tid)
    same = all(i.shape == o.shape for i, o in tr)
    osz = [o.size for i, o in tr]; isz = [i.size for i, o in tr]
    out_shapes = [o.shape for i, o in tr]
    in_shapes = [i.shape for i, o in tr]
    shrink = all(o.size < i.size for i, o in tr)
    grow = all(o.size > i.size for i, o in tr)
    # integer scale of input?
    int_scale = all(i.shape[0] and i.shape[1] and o.shape[0] % i.shape[0]==0 and o.shape[1] % i.shape[1]==0
                    and (o.shape[0]//i.shape[0])*(o.shape[1]//i.shape[1])>1 for i,o in tr)

    inc=set(); outc=set()
    for i,o in tr:
        inc|=set(np.unique(i).tolist()); outc|=set(np.unique(o).tolist())
    new_colors = outc - inc

    # object counts on input
    ncomps = [len(comps(i, bg(i))) for i, o in tr]
    avg_nc = np.mean(ncomps)

    # ---- shrink family branches ----
    if shrink:
        if all(o.shape == (1,1) for i, o in tr) or all(o.size <= 3 for i, o in tr):
            return "counting/construction"  # reduce to count/winner/selection token
        # selection of a sub-object / panel / dedup-of-objects
        return "counting/construction"

    # ---- grow family branches ----
    if grow:
        if int_scale:
            # tiling/fractal-like construction conditioned on content
            return "counting/construction"
        return "counting/construction"

    # ---- same-shape branches ----
    if same:
        # symmetry / periodic completion with NO single occluder color (fill bg from mirrors/periodicity
        # in a way the spine's occluder-based repair won't catch) -> symmetry/occlusion beyond simple repair
        # heuristic: output fills many bg(0) cells using surrounding structure, palette unchanged
        pix = np.mean([(i!=o).mean() for i,o in tr])
        # gridline / panel-cell paint tasks
        if any(has_gridlines(i) for i,o in tr):
            return "relational-recolor"
        # rays / lines: output adds straight runs from seed cells (connect/draw beyond same-color collinear)
        # heuristic: new pixels form long thin runs, few objects, colors preserved-ish
        if not new_colors and avg_nc <= 6 and pix < 0.25:
            return "line/ray-draw-connect"
        if new_colors:
            # marker-driven copy / recolor by relation / paint
            if avg_nc >= 3:
                return "relational-recolor"
            return "object-to-marker-copy"
        # object movement (gravity / shift / align) — footprint of colored cells moves
        if avg_nc >= 2 and pix < 0.4:
            return "object-movement-by-rule"
        # multi-object interaction (denoise, overlay, merge)
        if avg_nc >= 6:
            return "multi-object-interaction"
        return "symmetry/occlusion-beyond-repair"
    return "other"


cat = {}
for tid in missed:
    try:
        cat[tid] = categorize(tid)
    except Exception as e:
        cat[tid] = "other"

from collections import defaultdict
fam = defaultdict(list)
for tid, f in cat.items():
    fam[f].append(tid)

for f in sorted(fam, key=lambda k: -len(fam[k])):
    print(f"{f:40s} {len(fam[f]):3d}  e.g. {' '.join(sorted(fam[f])[:6])}")
json.dump({f: sorted(v) for f, v in fam.items()}, open(os.path.join(HERE,"runs","_miss_families_auto.json"),"w"), indent=1)
print("total", sum(len(v) for v in fam.values()))
