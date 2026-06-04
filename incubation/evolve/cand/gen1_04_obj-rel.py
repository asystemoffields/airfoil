#!/usr/bin/env python3
"""Gen-1 mutation #4 — OBJECT-RELATIONAL concept store + linker.

Thesis served: creativity = a CONCEPT STORE filled by EXPERIENCE, plus a LINKER that
recombines concepts in NOVEL ways, filtered by exact verify. The proven ARC-AGI-1
bottleneck is BREADTH (the rule isn't expressible in the 32-primitive base DSL), so
this candidate EXPANDS the vocabulary with rich object/relation concepts and lets a
small linker COMPOSE them (incl. functional repurposing) under exact train-verify.

Two experience channels:
  * BAKED / self-generated: a couple of concepts are *parametric learners* — they fit a
    mapping (global color permutation, rank->color, per-object recolor-by-shape, value
    map) from the CURRENT task's train pairs at solve time. (No ARC files read.)
  * IN-SESSION library (module-level state): every program TEMPLATE that exactly solved
    a prior task in this run is cached and re-tried first on later tasks — concepts proven
    by experience are cheap to reuse and recombine.

Object concepts added (the linker's new alphabet):
  segmentation (4/8-conn, by-color or color-agnostic), select-by-rank (size/area/color),
  keep/delete-by-property, recolor-objects-by-size-rank (learned map), learned GLOBAL
  color map, learned per-cell VALUE map (same-shape tasks), count objects/colors ->
  property grid, majority/unique object & panel (odd-one-out / dedup), fill object
  interior / bbox, dilate (halo), object gravity-snap, symmetric occlusion repair.

Interface: META + solve(train, test_inputs, budget) -> per-test list of up to 2 grids,
best-first, MDL-preferred. Pure python + numpy. Imported with /data/llm/.venv/bin/python.
"""
import sys
from collections import deque, Counter
import numpy as np

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl  # reuse base 32-primitive DSL + verifier + loader

META = {"name": "obj_relational_v1",
        "desc": "object/relation concept store + linker over base DSL, learned maps, in-session library"}

# ----------------------------------------------------------------------------
# IN-SESSION EXPERIENCE LIBRARY (module-level, persists across solve() calls).
# Stores program TEMPLATES (lists of concept-op specs) that VERIFIED-exactly on a
# prior task. We re-try these first so reused concept-links cost almost nothing.
# ----------------------------------------------------------------------------
_LIBRARY = []          # list of templates (each a tuple of step-specs)
_LIB_SEEN = set()      # dedup


def _remember(template):
    key = repr(template)
    if key not in _LIB_SEEN:
        _LIB_SEEN.add(key)
        _LIBRARY.append(template)


# ============================================================================
# OBJECT MACHINERY
# ============================================================================
def _bg(g):
    v, ct = np.unique(g, return_counts=True)
    return int(v[ct.argmax()])


def _components(g, conn=4, by_color=True, bg=0):
    """Connected components of non-bg cells. Returns list of dicts with cells/color/bbox."""
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    if conn == 8:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    comps = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != bg and not seen[i, j]:
                c0 = g[i, j]
                cells = []
                q = deque([(i, j)])
                seen[i, j] = True
                while q:
                    a, b = q.popleft()
                    cells.append((a, b))
                    for di, dj in nbrs:
                        x, y = a + di, b + dj
                        if 0 <= x < h and 0 <= y < w and not seen[x, y] and g[x, y] != bg:
                            if (not by_color) or g[x, y] == c0:
                                seen[x, y] = True
                                q.append((x, y))
                rs = [r for r, _ in cells]
                cs = [c for _, c in cells]
                cols = Counter(g[r, c] for r, c in cells)
                comps.append({
                    "cells": cells,
                    "color": int(cols.most_common(1)[0][0]),
                    "ncolors": len(cols),
                    "size": len(cells),
                    "bbox": (min(rs), min(cs), max(rs) + 1, max(cs) + 1),
                    "r": min(rs), "c": min(cs),
                })
    return comps


def _paint(shape, comps, bg=0):
    out = np.full(shape, bg, int)
    return out


def _comp_to_grid(g, comp, bg=0):
    out = np.full(g.shape, bg, int)
    for r, c in comp["cells"]:
        out[r, c] = g[r, c]
    return out


def _crop(g, comp, bg=0):
    r0, c0, r1, c1 = comp["bbox"]
    sub = np.full((r1 - r0, c1 - c0), bg, int)
    for r, c in comp["cells"]:
        sub[r - r0, c - c0] = g[r, c]
    return sub


def _norm_shape(comp):
    """Binary footprint of a component normalized to its bbox (for shape-matching)."""
    r0, c0, r1, c1 = comp["bbox"]
    m = np.zeros((r1 - r0, c1 - c0), int)
    for r, c in comp["cells"]:
        m[r - r0, c - c0] = 1
    return m


# ============================================================================
# CONCEPT OPS  (grid -> grid; the new alphabet for the linker)
# Each is small + composable; many are FUNCTIONAL REPURPOSINGS of segmentation.
# ============================================================================
def keep_largest(g, conn=4, by_color=True):
    comps = _components(g, conn, by_color, _bg(g))
    if not comps:
        return g
    best = max(comps, key=lambda c: c["size"])
    return _comp_to_grid(g, best, _bg(g))


def keep_smallest_obj(g, conn=4, by_color=True):
    comps = _components(g, conn, by_color, _bg(g))
    if not comps:
        return g
    best = min(comps, key=lambda c: c["size"])
    return _comp_to_grid(g, best, _bg(g))


def crop_largest(g, conn=4, by_color=True):
    comps = _components(g, conn, by_color, _bg(g))
    if not comps:
        return g
    best = max(comps, key=lambda c: c["size"])
    return _crop(g, best, _bg(g))


def crop_smallest(g, conn=4, by_color=True):
    comps = _components(g, conn, by_color, _bg(g))
    if not comps:
        return g
    best = min(comps, key=lambda c: c["size"])
    return _crop(g, best, _bg(g))


def crop_unique_color(g, conn=4):
    """Crop the object whose color is unique among objects (odd-one-out by color)."""
    bg = _bg(g)
    comps = _components(g, conn, False, bg)
    if not comps:
        return g
    cc = Counter(c["color"] for c in comps)
    uni = [c for c in comps if cc[c["color"]] == 1]
    if len(uni) == 1:
        return _crop(g, uni[0], bg)
    return g


def crop_majority_shape(g, conn=4):
    """Among objects, crop one whose shape is the MAJORITY (dedup -> keep the common)."""
    bg = _bg(g)
    comps = _components(g, conn, False, bg)
    if len(comps) < 2:
        return g
    sigs = [(_norm_shape(c).shape, tuple(_norm_shape(c).ravel())) for c in comps]
    cnt = Counter(sigs)
    maj = cnt.most_common(1)[0][0]
    for c, s in zip(comps, sigs):
        if s == maj:
            return _crop(g, c, bg)
    return g


def crop_unique_shape(g, conn=4):
    """Crop the object whose shape appears exactly once (the odd-one-out)."""
    bg = _bg(g)
    comps = _components(g, conn, False, bg)
    if len(comps) < 2:
        return g
    sigs = [(_norm_shape(c).shape, tuple(_norm_shape(c).ravel())) for c in comps]
    cnt = Counter(sigs)
    uni = [c for c, s in zip(comps, sigs) if cnt[s] == 1]
    if len(uni) == 1:
        return _crop(g, uni[0], bg)
    return g


def fill_object_interior(g, color, conn=4):
    """Fill the bg-holes strictly inside each object's bbox with `color` (rectangle fill)."""
    bg = _bg(g)
    comps = _components(g, conn, True, bg)
    out = g.copy()
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        interior = out[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
        interior[interior == bg] = color
    return out


def dilate_objects(g, conn=4):
    """Add a 1-cell halo around each object using its own color (4-neighborhood)."""
    bg = _bg(g)
    h, w = g.shape
    out = g.copy()
    nz = np.argwhere(g != bg)
    for r, c in nz:
        col = g[r, c]
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            x, y = r + di, c + dj
            if 0 <= x < h and 0 <= y < w and g[x, y] == bg:
                out[x, y] = col
    return out


def recolor_by_size_rank(g, mapping, conn=4, by_color=False):
    """mapping: rank(1=biggest..)->color. Recolor each object by its size rank."""
    bg = _bg(g)
    comps = _components(g, conn, by_color, bg)
    if not comps:
        return g
    order = sorted(comps, key=lambda c: -c["size"])
    out = g.copy()
    for rank, comp in enumerate(order, 1):
        if rank in mapping:
            for r, c in comp["cells"]:
                out[r, c] = mapping[rank]
    return out


def apply_color_map(g, cmap):
    """Global value->value remap (learned color permutation / mapping)."""
    out = g.copy()
    for a, b in cmap.items():
        out[g == a] = b
    return out


def apply_value_map(g, vmap):
    """Per-cell-value lookup map across the WHOLE grid (covers global recolors)."""
    return apply_color_map(g, vmap)


def most_frequent_color_fill(g):
    """Fill an entire grid (same shape) with the most-frequent NON-bg color."""
    vals = g[g != _bg(g)]
    if vals.size == 0:
        return g
    col = Counter(vals.tolist()).most_common(1)[0][0]
    return np.full(g.shape, col, int)


def count_objects_grid(g, conn=4, by_color=False):
    """Property grid: a 1x1 grid whose value = number of objects (count concept)."""
    comps = _components(g, conn, by_color, _bg(g))
    return np.array([[len(comps)]], int)


def sym_complete(g):
    """Occlusion repair: fill bg cells using whichever of {LR,UD,rot180} mirrors is
    consistent everywhere it overlaps (symmetric-object completion)."""
    bg = _bg(g)
    out = g.copy()
    for m in (g[:, ::-1], g[::-1, :], g[::-1, ::-1]):
        if m.shape != g.shape:
            continue
        # consistency: where both known, they must agree
        both = (out != bg) & (m != bg)
        if np.any(both) and not np.all(out[both] == m[both]):
            continue
        fillable = (out == bg) & (m != bg)
        out[fillable] = m[fillable]
    return out


def gravity_objects_down(g):
    """Move every object down until it rests on another object or the floor (whole-object)."""
    bg = _bg(g)
    comps = _components(g, 4, True, bg)
    h, w = g.shape
    out = np.full(g.shape, bg, int)
    # process bottom-most objects first
    comps.sort(key=lambda c: -c["bbox"][2])
    occ = np.zeros((h, w), bool)
    for comp in comps:
        cells = comp["cells"]
        drop = 0
        while True:
            ok = True
            for r, c in cells:
                nr = r + drop + 1
                if nr >= h or occ[nr, c]:
                    ok = False
                    break
            if not ok:
                break
            drop += 1
        for r, c in cells:
            out[r + drop, c] = g[r, c]
            occ[r + drop, c] = True
    return out


# ============================================================================
# PARAMETRIC LEARNERS  (fit a concept's params from the CURRENT task's train).
# These are the in-task "experience" channel: induce a mapping, then verify.
# ============================================================================
def learn_global_color_map(train):
    """If every train pair is same-shape and a consistent value->value map explains all
    cells, return that map (a learned color permutation). Else None."""
    cmap = {}
    for gi, go in train:
        if gi.shape != go.shape:
            return None
        for a, b in zip(gi.ravel(), go.ravel()):
            a, b = int(a), int(b)
            if a in cmap and cmap[a] != b:
                return None
            cmap[a] = b
    # must actually change something (avoid identity collisions w/ base DSL)
    if all(a == b for a, b in cmap.items()):
        return None
    return cmap


def learn_size_rank_map(train, conn=4, by_color=False):
    """If recolor-by-size-rank with a consistent rank->color map explains all train, return it."""
    mapping = {}
    for gi, go in train:
        if gi.shape != go.shape:
            return None
        bg = _bg(gi)
        comps = _components(gi, conn, by_color, bg)
        if not comps:
            return None
        order = sorted(comps, key=lambda c: -c["size"])
        for rank, comp in enumerate(order, 1):
            outcols = Counter(go[r, c] for r, c in comp["cells"])
            if len(outcols) != 1:
                return None
            col = int(outcols.most_common(1)[0][0])
            if rank in mapping and mapping[rank] != col:
                return None
            mapping[rank] = col
    if not mapping:
        return None
    return mapping


def learn_shape_recolor_map(train, conn=4):
    """Recolor each object by a color decided by its normalized SHAPE (signature->color).
    Captures 'recolor objects by which template they match'. Returns dict sig->color."""
    smap = {}
    for gi, go in train:
        if gi.shape != go.shape:
            return None
        bg = _bg(gi)
        comps = _components(gi, conn, False, bg)
        if not comps:
            return None
        for comp in comps:
            sig = (_norm_shape(comp).shape, tuple(_norm_shape(comp).ravel()))
            outcols = Counter(go[r, c] for r, c in comp["cells"])
            if len(outcols) != 1:
                return None
            col = int(outcols.most_common(1)[0][0])
            if sig in smap and smap[sig] != col:
                return None
            smap[sig] = col
    return smap or None


def apply_shape_recolor(g, smap, conn=4):
    bg = _bg(g)
    comps = _components(g, conn, False, bg)
    out = g.copy()
    for comp in comps:
        sig = (_norm_shape(comp).shape, tuple(_norm_shape(comp).ravel()))
        if sig in smap:
            for r, c in comp["cells"]:
                out[r, c] = smap[sig]
    return out


# ============================================================================
# THE CONCEPT LIBRARY (linker alphabet). Each entry: spec -> callable(g).
# Cost = MDL weight (smaller tried first). Some are "learners": built per task.
# ============================================================================
def _build_concept_ops(train, palette):
    """Return a list of (cost, spec, fn) concept ops. spec is a hashable template piece.
    Learners that can't fit the current train are omitted (cheap pruning)."""
    ops = []
    colors = [c for c in palette if c != 0]

    def add(cost, spec, fn):
        ops.append((cost, spec, fn))

    # ---- learned global maps (highest value, low cost: fit from this task) ----
    cmap = learn_global_color_map(train)
    if cmap is not None:
        add(1, ("colormap",), lambda g, m=cmap: apply_color_map(g, m))

    for conn in (4, 8):
        for byc in (False, True):
            rm = learn_size_rank_map(train, conn, byc)
            if rm is not None:
                add(2, ("sizerank", conn, byc), lambda g, m=rm, cn=conn, b=byc: recolor_by_size_rank(g, m, cn, b))
    sm = learn_shape_recolor_map(train, 8)
    if sm is not None:
        add(2, ("shaperecolor",), lambda g, m=sm: apply_shape_recolor(g, m, 8))

    # ---- parameter-free object concepts ----
    for conn in (4, 8):
        for byc in (True, False):
            add(2, ("keep_largest", conn, byc), lambda g, cn=conn, b=byc: keep_largest(g, cn, b))
            add(2, ("keep_smallest", conn, byc), lambda g, cn=conn, b=byc: keep_smallest_obj(g, cn, b))
            add(2, ("crop_largest", conn, byc), lambda g, cn=conn, b=byc: crop_largest(g, cn, b))
            add(2, ("crop_smallest", conn, byc), lambda g, cn=conn, b=byc: crop_smallest(g, cn, b))
        add(2, ("crop_unique_color", conn), lambda g, cn=conn: crop_unique_color(g, cn))
        add(2, ("crop_majority_shape", conn), lambda g, cn=conn: crop_majority_shape(g, cn))
        add(2, ("crop_unique_shape", conn), lambda g, cn=conn: crop_unique_shape(g, cn))
        add(3, ("count_objects", conn), lambda g, cn=conn: count_objects_grid(g, cn, False))

    add(2, ("most_freq_fill",), most_frequent_color_fill)
    add(2, ("sym_complete",), sym_complete)
    add(3, ("dilate",), lambda g: dilate_objects(g))
    add(3, ("gravity_objs",), gravity_objects_down)
    for c in colors:
        add(3, ("fill_interior", c), lambda g, col=c: fill_object_interior(g, col))

    return ops


# ============================================================================
# BASE DSL single-op concepts (reused, so the linker can compose new+old ideas)
# ============================================================================
def _base_ops(palette):
    colors = [c for c in palette if c != 0]
    ops = []
    for name, (_fn, nc) in dsl.OPS.items():
        if name == "identity":
            continue
        if nc == 0:
            ops.append((1, ("base", name, ()), (name, ())))
        elif nc == 1:
            for c in colors:
                ops.append((2, ("base", name, (c,)), (name, (c,))))
        elif nc == 2:
            for a in colors:
                for b in colors:
                    if a != b:
                        ops.append((3, ("base", name, (a, b)), (name, (a, b))))
    return ops


def _apply_base(g, spec):
    name, args = spec
    try:
        return dsl.OPS[name][0](g, *args)
    except Exception:
        return None


# ============================================================================
# LINKER  — compose concept ops (+ base ops), verify exactly on train, MDL order.
# ============================================================================
def _verify(steps, train):
    """steps: list of callables g->g (or None on failure). True iff every train pair matches."""
    for gi, go in train:
        g = gi
        for fn in steps:
            try:
                g = fn(g)
            except Exception:
                return False
            if g is None:
                return False
        if g is None or g.shape != go.shape or not np.array_equal(g, go):
            return False
    return True


def _spec_to_fn(spec, concept_ops, base_ops):
    """Resolve a template-step spec back into a callable (for library reuse)."""
    for cost, s, fn in concept_ops:
        if s == spec:
            return fn
    for cost, s, prog in base_ops:
        if s == spec:
            return lambda g, p=prog: _apply_base(g, p)
    return None


def _search(train, budget):
    """Return list of (template, [callables]) train-consistent programs, MDL-sorted.
    template = tuple of step-specs (hashable; cached in the in-session library)."""
    palette = dsl.palette(train)
    concept_ops = _build_concept_ops(train, palette)
    base_ops = _base_ops(palette)
    found = []
    nexec = 0
    seen_out = {}  # cache outputs by template prefix to compose length-2 cheaply

    # ---- channel 0: IN-SESSION LIBRARY reuse (try learned templates first) ----
    for tmpl in list(_LIBRARY):
        if nexec >= budget:
            break
        fns = []
        ok = True
        for spec in tmpl:
            fn = _spec_to_fn(spec, concept_ops, base_ops)
            if fn is None:
                ok = False
                break
            fns.append(fn)
        if not ok:
            continue
        nexec += len(train)
        if _verify(fns, train):
            found.append((tmpl, fns))

    # ---- channel 1: length-1 (all concept ops, then base ops) ----
    pool = concept_ops + [(cost, s, lambda g, p=prog: _apply_base(g, p)) for cost, s, prog in base_ops]
    pool.sort(key=lambda x: x[0])

    for cost, spec, fn in pool:
        if nexec >= budget:
            break
        nexec += len(train)
        if _verify([fn], train):
            found.append(((spec,), [fn]))

    if found:
        # dedup by template
        seen = set()
        uniq = []
        for tmpl, fns in found:
            if tmpl not in seen:
                seen.add(tmpl)
                uniq.append((tmpl, fns))
        return uniq, nexec, concept_ops, base_ops

    # ---- channel 2: length-2 compositions (concept->concept / concept->base / base->base)
    # Keep it bounded: precompute first-step outputs on train; chain a small second pool.
    first_pool = pool[:60]      # cheapest first steps
    second_pool = pool[:60]
    # cache first-step outputs
    cache = []
    for cost1, spec1, fn1 in first_pool:
        if nexec >= budget:
            break
        outs = []
        ok = True
        for gi, _ in train:
            try:
                o = fn1(gi)
            except Exception:
                ok = False
                break
            if o is None:
                ok = False
                break
            outs.append(o)
        nexec += len(train)
        if ok:
            cache.append((cost1, spec1, fn1, outs))

    targets = [go for _, go in train]
    for cost1, spec1, fn1, outs in cache:
        if nexec >= budget:
            break
        for cost2, spec2, fn2 in second_pool:
            if nexec >= budget:
                break
            nexec += len(train)
            good = True
            for o, t in zip(outs, targets):
                try:
                    o2 = fn2(o)
                except Exception:
                    good = False
                    break
                if o2 is None or o2.shape != t.shape or not np.array_equal(o2, t):
                    good = False
                    break
            if good:
                found.append(((spec1, spec2), [fn1, fn2]))
                if len(found) >= 4:
                    break
        if len(found) >= 4:
            break

    seen = set()
    uniq = []
    for tmpl, fns in found:
        if tmpl not in seen:
            seen.add(tmpl)
            uniq.append((tmpl, fns))
    return uniq, nexec, concept_ops, base_ops


def _tmpl_cost(tmpl):
    # MDL: prefer shorter & those using cheap (low-arity / learned) steps
    return (len(tmpl), sum(len(repr(s)) for s in tmpl))


# ============================================================================
# PUBLIC API
# ============================================================================
def solve(train, test_inputs, budget):
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    try:
        progs, nexec, _, _ = _search(train, budget)
    except Exception:
        progs = []
    progs = sorted(progs, key=lambda p: _tmpl_cost(p[0]))

    # remember verified templates as in-session experience (concept reuse channel)
    for tmpl, _fns in progs[:2]:
        _remember(tmpl)

    chosen = progs[:2]
    attempts = []
    for gi in test_inputs:
        gi = np.asarray(gi, int)
        cand = []
        for tmpl, fns in chosen:
            g = gi
            ok = True
            for fn in fns:
                try:
                    g = fn(g)
                except Exception:
                    ok = False
                    break
                if g is None:
                    ok = False
                    break
            if ok and g is not None:
                cand.append(g)
        # Fallback so we never return empty: identity (cheap, sometimes shape-correct)
        if not cand:
            cand.append(gi)
        attempts.append(cand[:2])
    return attempts
