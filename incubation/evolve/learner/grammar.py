#!/usr/bin/env python3
"""Branch-B learner — the RELATION GRAMMAR + self-gen CURRICULUM (the trainable data engine).

A RELATION is a typed cause->effect rule: (decomposition, feature, effect[, params]). The grammar is the
explicit space the campaign's hand-authored families covered, turned into something a model can (a) generate
OVER and (b) generalize WITHIN. Two directions:
  * apply_relation(relation, grid) -> grid        (forward: used to RENDER curriculum tasks)
  * induce(rtype, train) -> relation|None         (inverse: fit params from a task's demos + exact-verify)

The curriculum samples a relation type + params, renders random demos, and labels the task with its rtype.
The model will learn demos -> rtype; inference = top-K rtypes -> induce params -> verify (verifier = precision).

v0 scope (extensible): decompositions {objects-4/8-conn, by-color or agnostic}; features {size, size-rank,
color, holes, height, width, is-unique-size/color/shape}; effects {recolor-by-feature-table, select-extreme-
crop, whole-grid colormap}. Pure numpy. Run: /data/llm/.venv/bin/python grammar.py  (prints a smoke test)."""
import os, sys
from collections import deque
import numpy as np

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl  # shared grid helpers


# ======================================================================================
# DECOMPOSITION — grid -> list of object property dicts
# ======================================================================================
def _components(g, conn, by_color, bg):
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if conn == 8:
        nbrs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    comps = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != bg and not seen[i, j]:
                col = g[i, j]; cells = []; q = deque([(i, j)]); seen[i, j] = True
                while q:
                    a, b = q.popleft(); cells.append((a, b))
                    for di, dj in nbrs:
                        x, y = a + di, b + dj
                        if 0 <= x < h and 0 <= y < w and not seen[x, y] and g[x, y] != bg and \
                           (not by_color or g[x, y] == col):
                            seen[x, y] = True; q.append((x, y))
                comps.append(cells)
    return comps


def _holes(g, cells, bg):
    """count enclosed bg regions inside the object's bbox not touching the bbox border."""
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    r0, c0, r1, c1 = min(rs), min(cs), max(rs) + 1, max(cs) + 1
    occ = np.zeros((r1 - r0, c1 - c0), bool)
    for r, c in cells:
        occ[r - r0, c - c0] = True
    hh, ww = occ.shape
    seen = np.zeros((hh, ww), bool); border = deque()
    for i in range(hh):
        for j in (0, ww - 1):
            if not occ[i, j] and not seen[i, j]:
                seen[i, j] = True; border.append((i, j))
    for j in range(ww):
        for i in (0, hh - 1):
            if not occ[i, j] and not seen[i, j]:
                seen[i, j] = True; border.append((i, j))
    while border:
        a, b = border.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = a + di, b + dj
            if 0 <= x < hh and 0 <= y < ww and not occ[x, y] and not seen[x, y]:
                seen[x, y] = True; border.append((x, y))
    # remaining unseen bg cells = enclosed; count connected hole regions
    holes = 0
    for i in range(hh):
        for j in range(ww):
            if not occ[i, j] and not seen[i, j]:
                holes += 1; q = deque([(i, j)]); seen[i, j] = True
                while q:
                    a, b = q.popleft()
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + di, b + dj
                        if 0 <= x < hh and 0 <= y < ww and not occ[x, y] and not seen[x, y]:
                            seen[x, y] = True; q.append((x, y))
    return holes


def objects(g, conn=4, by_color=True, bg=0):
    out = []
    for cells in _components(g, conn, by_color, bg):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        cols = set(int(g[r, c]) for r, c in cells)
        out.append({
            "cells": cells, "size": len(cells),
            "color": (cols.pop() if len(cols) == 1 else -1),
            "h": max(rs) - min(rs) + 1, "w": max(cs) - min(cs) + 1,
            "r0": min(rs), "c0": min(cs), "holes": _holes(g, cells, bg),
            "shape": frozenset((r - min(rs), c - min(cs)) for r, c in cells),
        })
    return out


# ======================================================================================
# FEATURES — object -> hashable key (the candidate CAUSE). Some need the full object list (ranks/uniqueness).
# ======================================================================================
def _rank(objs, o, key, desc):
    vals = sorted(set(x[key] for x in objs), reverse=desc)
    return vals.index(o[key])

FEATURES = {
    "size":        lambda o, objs: o["size"],
    "color":       lambda o, objs: o["color"],
    "holes":       lambda o, objs: o["holes"],
    "height":      lambda o, objs: o["h"],
    "width":       lambda o, objs: o["w"],
    "rank_size":   lambda o, objs: _rank(objs, o, "size", False),
    "rank_size_d": lambda o, objs: _rank(objs, o, "size", True),
    "uniq_size":   lambda o, objs: sum(x["size"] == o["size"] for x in objs) == 1,
    "uniq_color":  lambda o, objs: sum(x["color"] == o["color"] for x in objs) == 1,
    "uniq_shape":  lambda o, objs: sum(x["shape"] == o["shape"] for x in objs) == 1,
}
FEATURE_NAMES = list(FEATURES)


# ======================================================================================
# RELATION TYPES — (effect, decomposition, feature). A concrete RELATION adds params (a table / a mode).
# ======================================================================================
DECOMPS = [(4, True), (4, False), (8, True), (8, False)]   # (conn, by_color)
EFFECTS = ["recolor", "select"]                            # + whole-grid "colormap" handled specially
SELECT_MODES = ["argmax", "argmin", "unique_true"]         # for boolean features unique_true; else argmax/min


def rtype_id(effect, decomp, feature):
    return f"{effect}|{decomp[0]}{'c' if decomp[1] else 'a'}|{feature}"


def all_rtypes():
    """enumerate the discrete relation-TYPE space (labels the model predicts)."""
    ids = ["colormap"]
    for d in DECOMPS:
        for f in FEATURE_NAMES:
            ids.append(rtype_id("recolor", d, f))
            ids.append(rtype_id("select", d, f))
    return ids


# ---------- forward: apply a concrete relation to a grid ----------
def apply_relation(rel, g, bg=0):
    g = np.asarray(g, int)
    if rel["effect"] == "colormap":
        out = g.copy()
        for a, b in rel["table"].items():
            out[g == a] = b
        return out
    conn, by_color = rel["decomp"]
    objs = objects(g, conn, by_color, bg)
    if not objs:
        return None
    ff = FEATURES[rel["feature"]]
    if rel["effect"] == "recolor":
        out = g.copy()
        for o in objs:
            k = ff(o, objs)
            if k not in rel["table"]:
                return None
            for r, c in o["cells"]:
                out[r, c] = rel["table"][k]
        return out
    if rel["effect"] == "select":
        keys = [ff(o, objs) for o in objs]
        idx = _select_index(keys, rel["mode"])
        if idx is None:
            return None
        cells = objs[idx]["cells"]
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        return g[min(rs):max(rs) + 1, min(cs):max(cs) + 1].copy()
    return None


def _select_index(keys, mode):
    if mode == "unique_true":
        idxs = [i for i, k in enumerate(keys) if k is True]
        return idxs[0] if len(idxs) == 1 else None
    numeric = [k for k in keys if isinstance(k, (int, float)) and not isinstance(k, bool)]
    if len(numeric) != len(keys):
        return None
    if mode == "argmax":
        m = max(keys); idxs = [i for i, k in enumerate(keys) if k == m]
    else:
        m = min(keys); idxs = [i for i, k in enumerate(keys) if k == m]
    return idxs[0] if len(idxs) == 1 else None


# ---------- inverse: induce a relation of a given TYPE from a task's demos ----------
def induce(effect, decomp, feature, train, bg=0):
    """Fit params (table / mode) consistent across ALL train pairs; exact-verify; return relation or None."""
    if effect == "colormap":
        table = {}
        for gi, go in train:
            if gi.shape != go.shape:
                return None
            for a, b in zip(gi.flatten().tolist(), go.flatten().tolist()):
                if a in table and table[a] != b:
                    return None
                table[a] = b
        if all(k == v for k, v in table.items()):
            return None
        rel = {"effect": "colormap", "table": table}
        return rel if _verify(rel, train) else None

    conn, by_color = decomp
    ff = FEATURES[feature]
    if effect == "recolor":
        table = {}
        for gi, go in train:
            if gi.shape != go.shape:
                return None
            objs = objects(gi, conn, by_color, bg)
            if not objs:
                return None
            for o in objs:
                ocols = set(int(go[r, c]) for r, c in o["cells"])
                if len(ocols) != 1:
                    return None
                k = ff(o, objs)
                v = ocols.pop()
                if k in table and table[k] != v:
                    return None
                table[k] = v
            bgmask = (gi == bg)
            if not np.array_equal(go[bgmask], gi[bgmask]):
                return None
        rel = {"effect": "recolor", "decomp": decomp, "feature": feature, "table": table}
        return rel if _verify(rel, train) else None

    if effect == "select":
        for mode in SELECT_MODES:
            rel = {"effect": "select", "decomp": decomp, "feature": feature, "mode": mode}
            if _verify(rel, train):
                return rel
        return None
    return None


def _verify(rel, train):
    for gi, go in train:
        out = apply_relation(rel, gi)
        if out is None or out.shape != go.shape or not np.array_equal(out, go):
            return False
    return True


# ======================================================================================
# CURRICULUM — sample a relation, render random demos, label with the relation TYPE.
# ======================================================================================
def _rng(seed):
    return np.random.RandomState(seed)


def _rand_grid_with_objects(rng, n_obj, palette, hgrid, wgrid):
    """place n_obj small random monochrome blobs on a bg grid (so object features vary)."""
    g = np.zeros((hgrid, wgrid), int)
    placed = 0; tries = 0
    while placed < n_obj and tries < 40:
        tries += 1
        oh, ow = rng.randint(1, 4), rng.randint(1, 4)
        r, c = rng.randint(0, hgrid - oh + 1), rng.randint(0, wgrid - ow + 1)
        if (g[r:r + oh, c:c + ow] != 0).any():
            continue
        col = palette[rng.randint(0, len(palette))]
        # sometimes a hollow rectangle (gives holes), else solid
        if oh >= 3 and ow >= 3 and rng.rand() < 0.4:
            g[r:r + oh, c:c + ow] = col
            g[r + 1:r + oh - 1, c + 1:c + ow - 1] = 0
        else:
            g[r:r + oh, c:c + ow] = col
        placed += 1
    return g


def sample_task(rng, rtype=None, n_demos=4):
    """Return (demos, rtype_label) or None if degenerate. demos = list of (input,output)."""
    types = all_rtypes()
    if rtype is None:
        rtype = types[rng.randint(0, len(types))]
    palette = [c for c in range(1, 10)]
    rng.shuffle(palette)
    if rtype == "colormap":
        # random bijection on a subset of colors
        src = list(range(1, 10)); rng.shuffle(src)
        k = rng.randint(2, 5); a = src[:k]; b = src[k:2 * k] if 2 * k <= 9 else src[:k][::-1]
        table = {0: 0}
        for x, y in zip(a, b):
            table[x] = y
        rel = {"effect": "colormap", "table": table}
        rtype_lab = "colormap"
    else:
        effect, dpart, feature = rtype.split("|")
        conn = int(dpart[0]); by_color = dpart[1] == "c"; decomp = (conn, by_color)
        if effect == "recolor":
            target = [c for c in range(1, 10) if c not in palette[:4]]
            rng.shuffle(target)
            rel = {"effect": "recolor", "decomp": decomp, "feature": feature,
                   "table": {}, "_lazy": (feature, target)}
        else:
            rel = {"effect": "select", "decomp": decomp, "feature": feature,
                   "mode": SELECT_MODES[rng.randint(0, len(SELECT_MODES))]}
        rtype_lab = rtype

    demos = []
    for _ in range(n_demos):
        for _try in range(8):
            g = _rand_grid_with_objects(rng, rng.randint(2, 5), palette[:5], rng.randint(6, 12), rng.randint(6, 12))
            r = _materialize(rel, g, rng)
            if r is None:
                continue
            out = apply_relation(r, g)
            if out is None or out.size == 0 or np.array_equal(out, g):
                continue
            demos.append((g, out)); rel = r
            break
        else:
            return None
    # final consistency: the induced rtype must reproduce all demos (the label is honest)
    if not _verify(rel, demos):
        return None
    return demos, rtype_lab


def _materialize(rel, g, rng):
    """fill a 'recolor' relation's table lazily for the feature-values present in g (so it's well-defined)."""
    if rel["effect"] != "recolor" or "_lazy" not in rel:
        return rel
    feature, target = rel["_lazy"]
    objs = objects(g, rel["decomp"][0], rel["decomp"][1])
    if not objs:
        return None
    ff = FEATURES[feature]
    table = dict(rel["table"])
    for o in objs:
        k = ff(o, objs)
        if k not in table:
            table[k] = target[len(table) % len(target)]
    out = {"effect": "recolor", "decomp": rel["decomp"], "feature": feature, "table": table}
    return out


# ======================================================================================
# SMOKE TEST
# ======================================================================================
if __name__ == "__main__":
    rng = _rng(0)
    types = all_rtypes()
    print(f"grammar: {len(types)} relation types. examples: {types[:4]} ... colormap, select|4c|uniq_size, ...")
    ok = 0; tried = 0; per = {}
    for i in range(400):
        t = types[rng.randint(0, len(types))]
        out = sample_task(_rng(1000 + i), rtype=t)
        tried += 1
        if out is None:
            continue
        demos, lab = out
        # round-trip: can induce() recover a verifying relation of this type from the demos?
        eff = "colormap" if lab == "colormap" else lab.split("|")[0]
        if lab == "colormap":
            rel = induce("colormap", None, None, demos)
        else:
            _e, dp, ft = lab.split("|"); dec = (int(dp[0]), dp[1] == "c")
            rel = induce(_e, dec, ft, demos)
        if rel is not None:
            ok += 1; per[lab] = per.get(lab, 0) + 1
    print(f"curriculum round-trip: {ok}/{tried} sampled tasks are induce-recoverable (label honest).")
    print(f"distinct rtypes generated+recovered: {len(per)} / {len(types)}")
    # show one task
    d, lab = sample_task(_rng(7), rtype="recolor|4c|holes")
    print(f"\nexample task rtype={lab}, demo[0] in->out shapes {d[0][0].shape}->{d[0][1].shape}")
    print("input:\n", d[0][0]); print("output:\n", d[0][1])
