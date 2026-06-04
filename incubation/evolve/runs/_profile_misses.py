#!/usr/bin/env python3
"""Profile gen2_base MISSED arc1-train tasks: load train pairs, compute structural signatures to
help categorize transformation FAMILIES. Read-only analysis (NOT at solve time)."""
import sys, os, json
import numpy as np
from collections import deque, Counter

HERE = "/data/Windows-files/Documents/airfoil/incubation/evolve"
TRAIN_DIR = "/data/arc/data/training"

res = json.load(open(os.path.join(HERE, "runs", "gen2_base_solved.json")))
missed = res["arc1_train"]["missed"]


def load(tid):
    d = json.load(open(os.path.join(TRAIN_DIR, tid + ".json")))
    tr = [(np.array(p["input"]), np.array(p["output"])) for p in d["train"]]
    te = [(np.array(p["input"]), np.array(p["output"])) for p in d["test"]]
    return tr, te


def bg(g):
    v, c = np.unique(g, return_counts=True)
    return int(v[c.argmax()])


def comps(g, b, diag=True):
    h, w = g.shape
    seen = np.zeros((h, w), bool)
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


def sig(tid):
    tr, te = load(tid)
    s = {"id": tid, "n_train": len(tr)}
    same_shape = all(i.shape == o.shape for i, o in tr)
    s["same_shape"] = same_shape
    in_shapes = [i.shape for i, o in tr]; out_shapes = [o.shape for i, o in tr]
    s["in_shapes"] = in_shapes; s["out_shapes"] = out_shapes
    # size relationship
    if same_shape:
        s["size_rel"] = "same"
    elif all(o.size < i.size for i,o in tr):
        s["size_rel"] = "shrink"
    elif all(o.size > i.size for i,o in tr):
        s["size_rel"] = "grow"
    else:
        s["size_rel"] = "mixed"
    # palette
    inc = set(); outc = set()
    for i,o in tr:
        inc |= set(np.unique(i).tolist()); outc |= set(np.unique(o).tolist())
    s["in_colors"] = sorted(inc); s["out_colors"] = sorted(outc)
    s["new_colors"] = sorted(outc - inc)
    s["lost_colors"] = sorted(inc - outc)
    # object counts (8-conn over modal bg)
    ncomp = []
    for i,o in tr:
        b = bg(i)
        ncomp.append(len(comps(i, b)))
    s["in_ncomp"] = ncomp
    # output constant?
    s["out_constant_shape"] = len(set(out_shapes)) == 1
    s["out_is_1x1"] = all(o.shape == (1,1) for i,o in tr)
    s["out_small"] = all(o.size <= 9 for i,o in tr)
    # pixel diff fraction when same shape
    if same_shape:
        s["pix_changed"] = round(float(np.mean([(i!=o).mean() for i,o in tr])), 3)
    return s


prof = {}
for tid in missed:
    try:
        prof[tid] = sig(tid)
    except Exception as e:
        prof[tid] = {"id": tid, "err": str(e)}

json.dump(prof, open(os.path.join(HERE, "runs", "_miss_profiles.json"), "w"))
print("n_missed:", len(missed))

# quick aggregate buckets to guide sampling
b_same = [t for t,p in prof.items() if p.get("same_shape")]
b_shrink = [t for t,p in prof.items() if p.get("size_rel")=="shrink"]
b_grow = [t for t,p in prof.items() if p.get("size_rel")=="grow"]
b_mixed = [t for t,p in prof.items() if p.get("size_rel")=="mixed"]
b_small_out = [t for t,p in prof.items() if p.get("out_small")]
b_newcol = [t for t,p in prof.items() if p.get("new_colors")]
print("same_shape:", len(b_same))
print("shrink:", len(b_shrink))
print("grow:", len(b_grow))
print("mixed:", len(b_mixed))
print("small_out(<=9):", len(b_small_out))
print("introduces_new_colors:", len(b_newcol))
