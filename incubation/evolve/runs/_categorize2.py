#!/usr/bin/env python3
"""Refined family categorization (v2) for the 318 missed arc1-train tasks, with mechanism-aware
heuristics calibrated against direct inspection. Read-only analysis."""
import os, json
import numpy as np
from collections import deque, defaultdict

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


def gridline_count(g):
    h, w = g.shape; b = bg(g)
    fr = [r for r in range(h) if len(set(g[r].tolist()))==1 and g[r,0]!=b]
    fc = [c for c in range(w) if len(set(g[:,c].tolist()))==1 and g[0,c]!=b]
    return len(fr), len(fc)


def categorize(tid):
    tr = load(tid)
    same = all(i.shape == o.shape for i, o in tr)
    shrink = all(o.size < i.size for i, o in tr)
    grow = all(o.size > i.size for i, o in tr)
    int_scale = all(i.shape[0] and i.shape[1] and o.shape[0]%i.shape[0]==0 and o.shape[1]%i.shape[1]==0
                    and (o.shape[0]//i.shape[0])*(o.shape[1]//i.shape[1])>1 for i,o in tr)
    inc=set(); outc=set()
    for i,o in tr:
        inc|=set(np.unique(i).tolist()); outc|=set(np.unique(o).tolist())
    new_colors = outc - inc
    ncomps = [len(comps(i, bg(i))) for i, o in tr]
    avg_nc = float(np.mean(ncomps))
    gl = [gridline_count(i) for i,o in tr]
    has_grid = any(fr>=1 or fc>=1 for fr,fc in gl)
    many_grid = any(fr>=2 or fc>=2 for fr,fc in gl)

    # ===== GROW =====
    if grow:
        # construction: tile/fractal/scale conditioned on content, or assemble multi-panel output
        return "counting/construction"   # all grow misses = construction beyond spine tiling/fractal

    # ===== SHRINK =====
    if shrink:
        out11 = all(o.shape==(1,1) for i,o in tr)
        outtiny = all(o.size<=4 for i,o in tr)
        if out11 or outtiny:
            return "counting/construction"   # reduce to a symbol/count/winner
        # panel selection: input partitioned by gridlines, output == one cell/panel
        if many_grid:
            return "multi-object-interaction"   # choose/compare panels
        # else: crop/select a sub-object or summarize block -> selection
        return "counting/construction"

    # ===== SAME SHAPE =====
    if same:
        pix = float(np.mean([(i!=o).mean() for i,o in tr]))
        # panel-grid recolor/paint inside cells
        if many_grid:
            return "relational-recolor"
        # line / ray drawing & connecting: thin straight runs added, palette ~preserved, few seeds
        if not new_colors and avg_nc <= 8 and pix < 0.30:
            return "line/ray-draw-connect"
        if not new_colors and avg_nc >= 9:
            # many small objects, no new color, footprint changes -> movement OR denoise
            if pix < 0.35:
                return "object-movement-by-rule"
            return "multi-object-interaction"
        if new_colors:
            # marker-conditioned copy/stamp vs relational recolor by property
            if avg_nc >= 4:
                return "relational-recolor"
            return "object-to-marker-copy"
        # palette preserved, moderate objects: movement/gravity/align
        if avg_nc >= 2 and pix < 0.45:
            return "object-movement-by-rule"
        return "symmetry/occlusion-beyond-repair"
    return "other"


cat = {}
for tid in missed:
    try: cat[tid] = categorize(tid)
    except Exception: cat[tid] = "other"

fam = defaultdict(list)
for tid, f in cat.items(): fam[f].append(tid)
for f in sorted(fam, key=lambda k: -len(fam[k])):
    print(f"{f:38s} {len(fam[f]):3d}  {' '.join(sorted(fam[f])[:8])}")
json.dump({f: sorted(v) for f, v in fam.items()}, open(os.path.join(HERE,"runs","_miss_families_v2.json"),"w"), indent=1)
print("total", sum(len(v) for v in fam.values()))
