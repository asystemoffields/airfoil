#!/usr/bin/env python3
"""GEN-3 INVENTOR #3 — CAUSAL-DECOMPOSITION (unrestricted grasp, made generative).

THESIS (the two halves of creativity this solver operationalizes)
-----------------------------------------------------------------
(1) UNRESTRICTED GRASP OF CAUSE-AND-EFFECT. We do not guess a whole-grid transform. We DECOMPOSE each
    train output into LOCAL parts — per CELL, per OBJECT (connected component), per REGION — and for
    each part we INDUCE, from the cross-example invariance, what INPUT feature(s) determine it and how.
    Cross-example invariance is the causal license: a per-part rule that holds on EVERY train pair is a
    causal dependency, not a coincidence. The held-out test is the intervention that verifies it.

(2) REAL-TIME INVENTION (not retrieval). The global mechanism is SYNTHESIZED bottom-up by COMPOSING the
    induced local causal rules — a sentence assembled this task, not a whole-mechanism template pulled
    from a menu. A single colormap, a per-object recolor-by-size, a per-object halo-spawn, a
    region-conditional transform: each is built by composing primitive relations (the dsl ALPHABET) under
    an induced dependency. The experience-prior (mechanism_curriculum) ranks which composition SHAPE to
    try first, making the invention FAST; it never supplies a finished answer.

ABLATION (what makes a solve COUNT as creative). `solve_ablated` is the SAME knowledge restricted to
    single-whole-template RETRIEVAL: try each whole-grid dsl op (len<=1, plus a couple fixed 2-step
    whole-grid templates) and keep one that matches all train pairs. NO per-part decomposition, NO
    composition of induced local rules. INVENTED = solved - ablated_solved counts exactly the tasks whose
    solve requires composing induced local causal parts — unreachable by whole-template retrieval.

EXPERIENCE LIBRARY / TRANSFER. Verified-correct induced mechanisms are abstracted to a SCHEMA (the
    decomposition kind + the parameter family, colors generalized away) and stored module-level this run.
    On a later task, a stored schema is re-instantiated against the new task's palette/objects and tried
    FIRST. reset_library() clears it so transfer_invention can isolate cross-task reuse.

INTEGRITY. solve()/solve_ablated() learn ONLY from (a) the current task's train pairs, (b) module-level
    schemas from PRIOR verified solves this run, (c) the import-time self-gen prior in
    mechanism_curriculum. They receive test INPUTS only; held-out outputs are withheld by the gate. No
    ARC files read at solve time, no network, no LLM. Pure python + numpy. Respect budget.

Run/scored with /data/llm/.venv/bin/python from .../incubation/evolve via invention_gate.
"""
import os
import sys
import time
from collections import deque, Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVOLVE = os.path.dirname(HERE)
ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
for p in (EVOLVE, ARC):
    if p not in sys.path:
        sys.path.insert(0, p)

import dsl  # the ALPHABET (knowledge)

# Experience-prior over composition SHAPES (self-generated at import; no ARC reads). Optional accelerant.
try:
    import mechanism_curriculum as MC
    _HAVE_PRIOR = True
except Exception:
    MC = None
    _HAVE_PRIOR = False

META = {"name": "gen3_03_causal-decomposition",
        "desc": "induce per-object/per-region/per-cell cause->effect rules from train invariance, then "
                "COMPOSE them into a task-specific global mechanism (no whole-grid template); "
                "experience-prior ranks composition shapes; ablation = single-whole-template retrieval"}


# ===========================================================================
# GRID / OBJECT UTILITIES (relations over the parts we decompose into)
# ===========================================================================
def _bg(g):
    v, ct = np.unique(g, return_counts=True)
    return int(v[ct.argmax()])


def _components(g, bg=0, diag=False):
    """Connected nonzero(!=bg) components -> list of dicts with cells/bbox/color/size."""
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    nbr = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        nbr += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    comps = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != bg and not seen[i, j]:
                cells = []
                q = deque([(i, j)])
                seen[i, j] = True
                while q:
                    a, b = q.popleft()
                    cells.append((a, b))
                    for di, dj in nbr:
                        x, y = a + di, b + dj
                        if 0 <= x < h and 0 <= y < w and g[x, y] != bg and not seen[x, y]:
                            seen[x, y] = True
                            q.append((x, y))
                rs = [r for r, _ in cells]
                cs = [c for _, c in cells]
                cols = Counter(g[r, c] for r, c in cells)
                comps.append({
                    "cells": cells, "size": len(cells),
                    "bbox": (min(rs), min(cs), max(rs) + 1, max(cs) + 1),
                    "color": cols.most_common(1)[0][0], "ncolors": len(cols),
                    "colors": cols,
                })
    return comps


def _palette_nonbg(g, bg=0):
    return sorted(c for c in np.unique(g).tolist() if c != bg)


def _norm_shape(comp, g):
    """Binary shape mask of a component, cropped to its bbox (for shape-equality across colors)."""
    r0, c0, r1, c1 = comp["bbox"]
    m = np.zeros((r1 - r0, c1 - c0), int)
    for (a, b) in comp["cells"]:
        m[a - r0, b - c0] = 1
    return m


# ===========================================================================
# THE INVENTOR — a set of CAUSAL-DECOMPOSITION inducers. Each tries to induce a global mechanism as a
# COMPOSITION of LOCAL cause->effect rules, verifying the rule holds on EVERY train pair (the invariance
# that licenses causal induction). Each inducer returns a callable transform(grid)->grid, or None.
#
# These are NOT whole-grid templates: every one is parameterized by a per-part dependency learned from
# the decomposition. A whole-template retrieval cannot produce them (that's the ablation boundary).
# ===========================================================================

def _verify(fn, train):
    for gi, go in train:
        try:
            out = fn(gi)
        except Exception:
            return False
        if out is None or out.shape != go.shape or not np.array_equal(out, go):
            return False
    return True


# ---- (A) PER-CELL colormap: induce a fixed input-color -> output-color map that holds cell-wise on all
#          pairs (same shape). This is the simplest causal decomposition: each cell's output is a function
#          of its own input color, abstracted across position. COMPOSITION = apply the induced map.
def induce_cellmap(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None, None
    mapping = {}
    for gi, go in train:
        for a, b in zip(gi.flatten().tolist(), go.flatten().tolist()):
            if a in mapping and mapping[a] != b:
                return None, None
            mapping[a] = b
    if all(k == v for k, v in mapping.items()):
        return None, None  # identity is not a mechanism

    def fn(g, mp=dict(mapping)):
        out = g.copy()
        flat = out.reshape(-1)
        for i, v in enumerate(flat):
            if v in mp:
                flat[i] = mp[v]
        return out
    schema = ("cellmap", {})  # colors abstracted away in schema (re-induced per task)
    return fn, schema


# ---- (B) PER-OBJECT recolor by an induced ORDERING / property. Decompose into objects; for each object
#          the output color is a function of an object PROPERTY (size, size-rank, original color, #cells).
#          Induce the property->color table from train, require it consistent across ALL pairs.
def _object_recolor_inducer(train, key_fn, diag=False):
    """key_fn(comp, allcomps) -> hashable key. Induce key->out_color; require shape/positions preserved
    (only colors change), and the map consistent across every pair."""
    table = {}
    for gi, go in train:
        if gi.shape != go.shape:
            return None
        comps = _components(gi, diag=diag)
        if not comps:
            return None
        for comp in comps:
            # output color of this object = the color the SAME cells take in go (must be uniform)
            ocols = set(int(go[r, c]) for (r, c) in comp["cells"])
            if len(ocols) != 1:
                return None
            ocol = ocols.pop()
            k = key_fn(comp, comps)
            if k in table and table[k] != ocol:
                return None
            table[k] = ocol
        # background cells must be unchanged (objects-only recolor)
        bgmask = (gi == 0)
        if not np.array_equal(go[bgmask], gi[bgmask]):
            return None
    return table


def induce_object_recolor(train, diag=False):
    """Try several object KEYS (the induced 'cause'): size-rank, size, color, #cells parity. The KEY is the
    abstracted feature the output color causally depends on."""
    keys = {
        "by_size_rank": lambda c, cs: sorted(set(x["size"] for x in cs)).index(c["size"]),
        "by_size": lambda c, cs: c["size"],
        "by_color": lambda c, cs: c["color"],
        "by_size_rank_desc": lambda c, cs: sorted(set(x["size"] for x in cs), reverse=True).index(c["size"]),
        "by_ncolors": lambda c, cs: c["ncolors"],
    }
    for kname, kfn in keys.items():
        table = _object_recolor_inducer(train, kfn, diag=diag)
        if table is None:
            continue

        def fn(g, kfn=kfn, table=dict(table), diag=diag):
            comps = _components(g, diag=diag)
            out = g.copy()
            for comp in comps:
                k = kfn(comp, comps)
                if k not in table:
                    return None  # unseen key -> abstain (causal rule undefined here)
                col = table[k]
                for (r, c) in comp["cells"]:
                    out[r, c] = col
            return out
        if _verify(fn, train):
            return fn, ("object_recolor", {"key": kname, "diag": diag})
    return None, None


# ---- (C) PER-OBJECT geometric / cleanup map: each object is independently transformed in place (e.g.
#          mirror within its bbox, fill its bbox, outline). Induce ONE op that, applied per-object, matches.
def _per_object_apply(g, op_fn, diag=False):
    comps = _components(g, diag=diag)
    out = g.copy()
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        sub = g[r0:r1, c0:c1]
        try:
            sub2 = op_fn(sub)
        except Exception:
            return None
        if sub2.shape != sub.shape:
            return None
        # paste back only where the object lived OR where op writes (keep it local to bbox)
        out[r0:r1, c0:c1] = sub2
    return out


def induce_per_object_geom(train, diag=False):
    cand_ops = [
        ("reflect_h", lambda s: s[:, ::-1]),
        ("reflect_v", lambda s: s[::-1, :]),
        ("rot180", lambda s: np.rot90(s, 2)),
        ("transpose", lambda s: s.T if s.shape[0] == s.shape[1] else s),
        ("sym_lr", dsl.sym_lr), ("sym_ud", dsl.sym_ud),
    ]
    if any(gi.shape != go.shape for gi, go in train):
        return None, None
    for oname, op in cand_ops:
        def fn(g, op=op, diag=diag):
            return _per_object_apply(g, op, diag=diag)
        if _verify(fn, train):
            return fn, ("per_object_geom", {"op": oname, "diag": diag})
    return None, None


# ---- (D) PER-OBJECT SPAWN / HALO: each object causes a colored neighborhood (e.g. a diagonal/orthogonal
#          halo) whose color depends on the object's color. Induce: for each object color, what cells
#          around it get painted what color. Decompose into per-object effect templates relative to cells.
def induce_object_neighbor_paint(train, diag=True):
    """Induce a PER-SOURCE-COLOR local STAMP: for every nonzero source cell of color k, the output paints a
    learned set of (dr,dc) offsets around it with learned colors — the stamp is a FUNCTION of the source
    color (so e.g. color-2 spawns a diagonal-4 halo while color-1 spawns an orthogonal-7 cross). This is a
    causal decomposition: each object's color CAUSES its own local effect template, induced from the train
    invariance. Composition = overlay all per-cell stamps. Source cells stay fixed; bg-only cells change."""
    if any(gi.shape != go.shape for gi, go in train):
        return None, None
    R = 2  # consider offsets within Chebyshev radius 2 (covers halos/crosses)
    cand_offsets = [(dr, dc) for dr in range(-R, R + 1) for dc in range(-R, R + 1)
                    if not (dr == 0 and dc == 0)]
    # stamp[src_color] = dict{(dr,dc): paint_color}; induced consistently across all pairs.
    stamp = {}
    ok = True
    for gi, go in train:
        h, w = gi.shape
        # source cells must be unchanged (we only ADD around objects)
        if not np.array_equal(go[gi != 0], gi[gi != 0]):
            ok = False
            break
        # which cells were newly painted (bg -> color)
        painted = (gi == 0) & (go != 0)
        # attribute each painted cell to the UNIQUE source cell offset that explains it.
        for i in range(h):
            for j in range(w):
                if not painted[i, j]:
                    continue
                pc = int(go[i, j])
                # find candidate source cells whose stamp could include this target
                explains = []
                for dr, dc in cand_offsets:
                    a, b = i - dr, j - dc  # source at (a,b) with offset (dr,dc) hits (i,j)
                    if 0 <= a < h and 0 <= b < w and gi[a, b] != 0:
                        explains.append((int(gi[a, b]), (dr, dc)))
                if not explains:
                    ok = False
                    break
                # require a consistent single (src_color, offset) explanation; if multiple sources, they
                # must all be the same color and the offset chosen is the one nearest (smallest |.|).
                explains.sort(key=lambda e: abs(e[1][0]) + abs(e[1][1]))
                scol, off = explains[0]
                stamp.setdefault(scol, {})
                if off in stamp[scol] and stamp[scol][off] != pc:
                    ok = False
                    break
                stamp[scol][off] = pc
            if not ok:
                break
        if not ok:
            break
    if not ok or not stamp:
        return None, None

    def fn(g, stamp={k: dict(v) for k, v in stamp.items()}):
        h, w = g.shape
        out = g.copy()
        for i in range(h):
            for j in range(w):
                sc = int(g[i, j])
                if sc == 0 or sc not in stamp:
                    continue
                for (dr, dc), pc in stamp[sc].items():
                    a, b = i + dr, j + dc
                    if 0 <= a < h and 0 <= b < w and g[a, b] == 0:
                        out[a, b] = pc
        return out
    if _verify(fn, train):
        return fn, ("object_neighbor_paint", {"stamp_colors": sorted(stamp.keys())})
    return None, None


# ---- (E) REGION-CONDITIONAL / REGION-RESTRICT composition: the cause acts only inside a sub-region (a
#          half, or the content bbox), the rest invariant. Induce the region + the local op.
def _region_box(g, region):
    h, w = g.shape
    if region == "top":
        return 0, 0, max(1, h // 2), w
    if region == "bot":
        return h // 2, 0, h, w
    if region == "left":
        return 0, 0, h, max(1, w // 2)
    if region == "right":
        return 0, w // 2, h, w
    nz = np.argwhere(g != 0)
    if nz.size == 0:
        return 0, 0, h, w
    (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def induce_region_restrict(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None, None
    ops = [("reflect_h", lambda s: s[:, ::-1]), ("reflect_v", lambda s: s[::-1, :]),
           ("rot180", lambda s: np.rot90(s, 2)), ("sym_lr", dsl.sym_lr), ("sym_ud", dsl.sym_ud)]
    for region in ("top", "bot", "left", "right", "bbox"):
        for oname, op in ops:
            def fn(g, region=region, op=op):
                r0, c0, r1, c1 = _region_box(g, region)
                sub = g[r0:r1, c0:c1]
                try:
                    sub2 = op(sub)
                except Exception:
                    return None
                if sub2.shape != sub.shape:
                    return None
                out = g.copy()
                out[r0:r1, c0:c1] = sub2
                return out
            if _verify(fn, train):
                return fn, ("region_restrict", {"region": region, "op": oname})
    return None, None


# ---- (F) MIRROR/SYMMETRY-COMPLETION: induce that the output COMPLETES the input under a symmetry —
#          composition of (a copy of) a reflected view overlaid (handles occlusion-repair-style tasks).
def induce_symmetry_complete(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None, None
    views = [("h", lambda g: g[:, ::-1]), ("v", lambda g: g[::-1, :]),
             ("180", lambda g: np.rot90(g, 2))]
    # try overlaying one or two symmetric views (nonzero of view fills zeros)
    for n1, v1 in views:
        def fn1(g, v1=v1):
            out = g.copy()
            m = v1(g)
            fill = (out == 0) & (m != 0)
            out[fill] = m[fill]
            return out
        if _verify(fn1, train):
            return fn1, ("symmetry_complete", {"views": [n1]})
    for n1, v1 in views:
        for n2, v2 in views:
            if n2 <= n1:
                continue
            def fn2(g, v1=v1, v2=v2):
                out = g.copy()
                for m in (v1(g), v2(g), v2(v1(g))):
                    fill = (out == 0) & (m != 0)
                    out[fill] = m[fill]
                return out
            if _verify(fn2, train):
                return fn2, ("symmetry_complete", {"views": [n1, n2]})
    return None, None


# ---- (G) PER-OBJECT MOVE/GRAVITY composition with a global op: induce a single global object-op that
#          rearranges (gravity in 4 dirs, then optional recolor). Lightweight, composes select+move.
def induce_gravity_then_map(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None, None
    gravs = [("down", dsl.gravity_down), ("up", dsl.gravity_up),
             ("left", dsl.gravity_left), ("right", dsl.gravity_right)]
    for gn, gf in gravs:
        def fn(g, gf=gf):
            return gf(g)
        if _verify(fn, train):
            return fn, ("gravity", {"dir": gn})
    return None, None


# ---- (H) FILL-ENCLOSED-REGION (fill holes) with an induced color (per-cell causal: a 0-cell enclosed by
#          an object becomes color X). Composition: select-enclosed -> paint.
def induce_fill_holes(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None, None
    # induce fill color from train (the color that newly appears in enclosed regions)
    fill_color = None
    for gi, go in train:
        diff = (gi != go)
        if not diff.any():
            return None, None
        newcols = set(int(go[i, j]) for i, j in zip(*np.where(diff)))
        if len(newcols) != 1:
            return None, None
        c = newcols.pop()
        if fill_color is None:
            fill_color = c
        elif fill_color != c:
            return None, None
    if fill_color is None:
        return None, None

    def fn(g, c=fill_color):
        return dsl.fill_holes(g, c)
    if _verify(fn, train):
        return fn, ("fill_holes", {"color": fill_color})
    return None, None


# ---- (I) TILING / MIRROR-TILING with shape change: output is the input arranged in a kxk block layout
#          where each block is a (possibly mirrored) view of the input — a classic compositional fractal/
#          tiling sentence. Induce the block-layout (which view goes in each block) from train.
_VIEW_FNS = {
    "I": lambda g: g, "H": lambda g: g[:, ::-1], "V": lambda g: g[::-1, :],
    "R": lambda g: np.rot90(g, 2), "T": lambda g: g.T,
}


def induce_periodic_complete(train):
    """Induce a PERIODIC mechanism: the grid is generated by a (pr,pc)-periodic tile; the output FILLS the
    zero cells using the period inferred from the nonzero cells (and may also repair mismatches). Causal
    decomposition: each cell's value is determined by its (row mod pr, col mod pc) class, induced from the
    invariant non-bg cells. Handles symmetry/periodicity repair where zeros are occlusions."""
    if any(gi.shape != go.shape for gi, go in train):
        return None, None
    # only meaningful when zeros get filled (output has fewer/zero background where input had holes)
    fills = any(((gi == 0) & (go != 0)).any() for gi, go in train)
    if not fills:
        return None, None

    def infer_period(g):
        h, w = g.shape
        best = None
        for pr in range(1, h + 1):
            for pc in range(1, w + 1):
                if pr == h and pc == w:
                    continue
                # check: within each residue class, all NONZERO cells agree
                table = {}
                ok = True
                for i in range(h):
                    for j in range(w):
                        if g[i, j] == 0:
                            continue
                        key = (i % pr, j % pc)
                        if key in table and table[key] != g[i, j]:
                            ok = False
                            break
                        table[key] = g[i, j]
                    if not ok:
                        break
                if ok and len(table) == pr * pc:  # period fully determined
                    best = (pr, pc, table)
                    return best
        return None

    def fn(g):
        info = infer_period(g)
        if info is None:
            return None
        pr, pc, table = info
        h, w = g.shape
        out = g.copy()
        for i in range(h):
            for j in range(w):
                if out[i, j] == 0:
                    v = table.get((i % pr, j % pc))
                    if v is not None:
                        out[i, j] = v
        return out
    if _verify(fn, train):
        return fn, ("periodic_complete", {})
    return None, None


def induce_denoise_keep_majority(train):
    """Induce a DENOISE mechanism: remove minority 'noise' cells, keeping the majority/structure. Two
    variants: (a) recolor every nonzero cell to the single dominant nonzero color; (b) drop cells of the
    least-frequent nonzero color to bg. Causal decomposition over color classes (the effect on a cell is a
    function of its color's global frequency)."""
    if any(gi.shape != go.shape for gi, go in train):
        return None, None

    # variant (b): remove the single noise color (present in input, gone/reduced in output)
    def fn_remove(g):
        cnt = Counter(int(v) for v in g.flatten() if v != 0)
        if len(cnt) < 2:
            return None
        noise = min(cnt, key=lambda c: cnt[c])
        out = g.copy()
        out[g == noise] = 0
        return out
    if _verify(fn_remove, train):
        return fn_remove, ("denoise_remove_minority", {})

    # variant (a): unify all nonzero to the dominant color
    def fn_unify(g):
        cnt = Counter(int(v) for v in g.flatten() if v != 0)
        if not cnt:
            return None
        dom = max(cnt, key=lambda c: cnt[c])
        out = g.copy()
        out[g != 0] = dom
        return out
    if _verify(fn_unify, train):
        return fn_unify, ("denoise_unify_majority", {})
    return None, None


def induce_block_tiling(train):
    # determine block grid (kr, kc) from output/input shape ratio, consistent across pairs
    ratios = set()
    for gi, go in train:
        if go.shape[0] % gi.shape[0] or go.shape[1] % gi.shape[1]:
            return None, None
        ratios.add((go.shape[0] // gi.shape[0], go.shape[1] // gi.shape[1]))
    if len(ratios) != 1:
        return None, None
    kr, kc = ratios.pop()
    if (kr, kc) == (1, 1) or kr > 3 or kc > 3:
        return None, None
    # for each block position induce which view of input it equals (must be consistent across pairs)
    layout = {}
    for br in range(kr):
        for bc in range(kc):
            chosen = None
            for vname, vf in _VIEW_FNS.items():
                good = True
                for gi, go in train:
                    h, w = gi.shape
                    blk = go[br * h:(br + 1) * h, bc * w:(bc + 1) * w]
                    vv = vf(gi)
                    if vv.shape != blk.shape or not np.array_equal(vv, blk):
                        good = False
                        break
                if good:
                    chosen = vname
                    break
            if chosen is None:
                return None, None
            layout[(br, bc)] = chosen

    def fn(g, kr=kr, kc=kc, layout=dict(layout)):
        h, w = g.shape
        out = np.zeros((kr * h, kc * w), int)
        for br in range(kr):
            for bc in range(kc):
                out[br * h:(br + 1) * h, bc * w:(bc + 1) * w] = _VIEW_FNS[layout[(br, bc)]](g)
        return out
    if _verify(fn, train):
        return fn, ("block_tiling", {"kr": kr, "kc": kc, "layout": {str(k): v for k, v in layout.items()}})
    return None, None


# ---- (J) GENERIC COMPOSED SEQUENCE (guided by the experience-prior): a short composition of dsl ops, but
#          ONLY accepted when it is a genuine multi-step sentence (len>=2) — i.e. invention, not a single
#          retrievable op. The prior proposes which relation KINDS to draw from, shrinking the search.
def _instantiate_ops(pal, kinds=None):
    colors = [c for c in pal if c != 0]
    insts = []
    allow = None
    if kinds and _HAVE_PRIOR:
        allow = set()
        for k in kinds:
            for op, _nc in MC.RELATIONS.get(k, []):
                allow.add(op)
    for name, (_fn, nc) in dsl.OPS.items():
        if allow is not None and name not in allow and name != "identity":
            continue
        if nc == 0:
            insts.append((name, ()))
        elif nc == 1:
            for c in colors:
                insts.append((name, (c,)))
        elif nc == 2:
            for a in colors:
                for b in colors:
                    if a != b:
                        insts.append((name, (a, b)))
    return insts


def induce_composed_sequence(train, kinds, budget_exec):
    """Greedy depth-2/3 composition over a prior-narrowed op set. Returns a 2+ step program or None."""
    pal = dsl.palette(train)
    insts = _instantiate_ops(pal, kinds=kinds)
    if len(insts) > 40:
        insts = insts[:40]
    ins = [gi for gi, _ in train]
    tgt = [go for _, go in train]

    def apply_all(grids, inst):
        res = []
        for g in grids:
            try:
                res.append(dsl.OPS[inst[0]][0](g, *inst[1]))
            except Exception:
                return None
        return res

    def exact(grids):
        return all(o is not None and o.shape == t.shape and np.array_equal(o, t)
                   for o, t in zip(grids, tgt))

    nexec = 0
    # frontier of (program, outs); only programs of length>=2 are accepted as a creative mechanism
    frontier = [([], ins)]
    for depth in range(2):  # build up to length 2 here; len-3 via one more expansion
        nxt = []
        for prog, outs in frontier:
            for inst in insts:
                outs2 = apply_all(outs, inst)
                nexec += 1
                if outs2 is None:
                    if nexec > budget_exec:
                        return None, None
                    continue
                prog2 = prog + [inst]
                if len(prog2) >= 2 and exact(outs2):
                    fn = lambda g, prog2=list(prog2): dsl.apply_prog(g, prog2)
                    return fn, ("composed_sequence", {"prog": [(n, list(a)) for n, a in prog2]})
                if len(prog2) < 2:
                    nxt.append((prog2, outs2))
                if nexec > budget_exec:
                    return None, None
        # keep frontier small
        frontier = nxt[:24]
    # one more layer for length-3 from best frontier
    for prog, outs in frontier[:8]:
        for inst in insts:
            outs2 = apply_all(outs, inst)
            nexec += 1
            if outs2 is None:
                if nexec > budget_exec:
                    return None, None
                continue
            prog2 = prog + [inst]
            if exact(outs2):
                fn = lambda g, prog2=list(prog2): dsl.apply_prog(g, prog2)
                return fn, ("composed_sequence", {"prog": [(n, list(a)) for n, a in prog2]})
            if nexec > budget_exec:
                return None, None
    return None, None


# ===========================================================================
# EXPERIENCE LIBRARY — abstracted schemas of verified mechanisms, reused across tasks this run.
# A schema is (kind, params). On a new task we re-instantiate it (colors re-induced from the new task),
# so reuse is genuine ABSTRACTION transfer, not memorized output.
# ===========================================================================
_LIBRARY = []  # list of (kind, params) schemas, most-recent first


def reset_library():
    global _LIBRARY
    _LIBRARY = []


def _schema_to_inducer(schema):
    """Map a stored schema back to the inducer family that produces it. Re-induction re-binds colors etc.
    from the NEW task, so a stored 'object_recolor by size_rank' is retried first on the new task."""
    kind = schema[0]
    table = {
        "cellmap": lambda tr: induce_cellmap(tr),
        "object_recolor": lambda tr: induce_object_recolor(tr, diag=schema[1].get("diag", False)),
        "per_object_geom": lambda tr: induce_per_object_geom(tr, diag=schema[1].get("diag", False)),
        "object_neighbor_paint": lambda tr: induce_object_neighbor_paint(tr),
        "region_restrict": lambda tr: induce_region_restrict(tr),
        "symmetry_complete": lambda tr: induce_symmetry_complete(tr),
        "periodic_complete": lambda tr: induce_periodic_complete(tr),
        "gravity": lambda tr: induce_gravity_then_map(tr),
        "fill_holes": lambda tr: induce_fill_holes(tr),
        "denoise_remove_minority": lambda tr: induce_denoise_keep_majority(tr),
        "denoise_unify_majority": lambda tr: induce_denoise_keep_majority(tr),
        "block_tiling": lambda tr: induce_block_tiling(tr),
    }
    return table.get(kind)


# ===========================================================================
# THE INDUCER PIPELINE (the INVENTION path). Order is set by the experience-prior's proposed composition
# SHAPES (fast), with the full battery as fallback. Each inducer COMPOSES local causal rules.
# ===========================================================================
_ALL_INDUCERS = [
    ("cellmap", lambda tr: induce_cellmap(tr)),
    ("object_recolor", lambda tr: induce_object_recolor(tr, diag=False)),
    ("object_recolor_diag", lambda tr: induce_object_recolor(tr, diag=True)),
    ("object_neighbor_paint", lambda tr: induce_object_neighbor_paint(tr)),
    ("per_object_geom", lambda tr: induce_per_object_geom(tr, diag=False)),
    ("region_restrict", lambda tr: induce_region_restrict(tr)),
    ("symmetry_complete", lambda tr: induce_symmetry_complete(tr)),
    ("periodic_complete", lambda tr: induce_periodic_complete(tr)),
    ("fill_holes", lambda tr: induce_fill_holes(tr)),
    ("denoise", lambda tr: induce_denoise_keep_majority(tr)),
    ("gravity", lambda tr: induce_gravity_then_map(tr)),
    ("block_tiling", lambda tr: induce_block_tiling(tr)),
]

# Map a prior combinator -> which inducer families it favors (so the prior REORDERS the battery).
_COMB_TO_INDUCERS = {
    "per_object_map": ["object_recolor", "object_recolor_diag", "per_object_geom"],
    "region_restrict": ["region_restrict", "symmetry_complete", "periodic_complete"],
    "repurpose_overlay": ["symmetry_complete", "periodic_complete", "block_tiling"],
    "feature_conditional": ["region_restrict", "gravity", "denoise"],
    "sequence": ["block_tiling", "fill_holes", "denoise", "gravity", "cellmap"],
}


def _prior_order(train):
    """Use mechanism_curriculum.propose_compositions to RANK inducer families. Returns a name list."""
    order = []
    if _HAVE_PRIOR:
        try:
            props = MC.propose_compositions(train, k=5)
            for p in props:
                for nm in _COMB_TO_INDUCERS.get(p["combinator"], []):
                    if nm not in order:
                        order.append(nm)
            # kind hints: colormap/paint -> cellmap / neighbor_paint first
            top_kinds = props[0]["kinds"] if props else []
            if "colormap" in top_kinds and "cellmap" not in order:
                order.insert(0, "cellmap")
            if "paint" in top_kinds and "object_neighbor_paint" not in order:
                order.insert(0, "object_neighbor_paint")
        except Exception:
            pass
    # append anything not yet ordered (full battery fallback)
    for nm, _ in _ALL_INDUCERS:
        if nm not in order:
            order.append(nm)
    return order


_INDUCER_BY_NAME = dict(_ALL_INDUCERS)


def _invent(train, budget, use_prior=True, use_library=True):
    """THE INVENTION PATH: induce + compose local causal rules into a global mechanism. Returns
    (fn, schema) or (None, None). Tries (1) library schemas (transfer), (2) prior-ranked inducers,
    (3) a prior-narrowed composed sequence as a last creative resort."""
    t0 = time.time()
    deadline = t0 + max(0.5, budget / 2500.0)  # budget is an exec count; convert to a soft time cap

    # (1) experience transfer: retry abstracted schemas from prior verified solves, re-induced here.
    if use_library:
        for schema in list(_LIBRARY):
            ind = _schema_to_inducer(schema)
            if ind is None:
                continue
            try:
                fn, sch = ind(train)
            except Exception:
                fn, sch = None, None
            if fn is not None and _verify(fn, train):
                return fn, sch

    # (2) prior-ranked inducer battery (the core invention path).
    order = _prior_order(train) if use_prior else [nm for nm, _ in _ALL_INDUCERS]
    for nm in order:
        ind = _INDUCER_BY_NAME.get(nm)
        if ind is None:
            continue
        try:
            fn, sch = ind(train)
        except Exception:
            fn, sch = None, None
        if fn is not None:
            return fn, sch
        if time.time() > deadline:
            break

    # (3) prior-narrowed composed sequence (>=2 steps == invention, not single-op retrieval).
    kinds = None
    if _HAVE_PRIOR:
        try:
            props = MC.propose_compositions(train, k=3)
            kinds = []
            for p in props[:2]:
                kinds += p["kinds"]
            kinds = list(dict.fromkeys(kinds))
        except Exception:
            kinds = None
    fn, sch = induce_composed_sequence(train, kinds, budget_exec=min(budget, 2500))
    if fn is not None:
        return fn, sch
    return None, None


# ===========================================================================
# ABLATION PATH — single-whole-template RETRIEVAL only. NO decomposition, NO composition of local rules.
# Try each whole-grid dsl op (len<=1) + a SMALL fixed menu of whole-grid 2-step templates; keep one that
# matches all train pairs. This is exactly what gen-1/2 template-induction could do.
# ===========================================================================
_ABLATION_TEMPLATES = [
    # fixed whole-grid "templates" (a menu, not composed-on-the-fly per part)
    [("tile_2x2", ())], [("scale2", ())], [("tile_h2", ())], [("tile_v2", ())],
    [("rot90", ())], [("rot180", ())], [("rot270", ())], [("reflect_h", ())], [("reflect_v", ())],
    [("transpose", ())], [("sym_lr", ())], [("sym_ud", ())],
    [("gravity_down", ())], [("gravity_up", ())], [("gravity_left", ())], [("gravity_right", ())],
    [("crop_content", ())], [("largest_object", ())], [("keep_smallest", ())], [("trim_border", ())],
    [("downscale2", ())],
]


def _retrieve_template(train):
    """Single-whole-template retrieval: scan length-1 dsl ops (color-instantiated) + the fixed menu; return
    the first whole-grid program matching ALL train pairs. NO per-part decomposition."""
    pal = dsl.palette(train)
    colors = [c for c in pal if c != 0]
    progs = list(_ABLATION_TEMPLATES)
    # color-parametric length-1 ops (still a single whole-grid template, just instantiated)
    for name, (_fn, nc) in dsl.OPS.items():
        if nc == 1:
            for c in colors:
                progs.append([(name, (c,))])
        elif nc == 2:
            for a in colors:
                for b in colors:
                    if a != b:
                        progs.append([(name, (a, b))])
    for prog in progs:
        if dsl.solves(prog, train):
            # reject identity-equivalent
            if all(np.array_equal(dsl.apply_prog(gi, prog), gi) for gi, _ in train):
                continue
            return lambda g, prog=list(prog): dsl.apply_prog(g, prog)
    return None


# ===========================================================================
# PUBLIC API
# ===========================================================================
def _attempts_from_fn(fn, fn2, test_inputs):
    """Build up to 2 attempts per test input from up to 2 mechanism callables."""
    attempts = []
    for gi in test_inputs:
        cand = []
        for f in (fn, fn2):
            if f is None:
                continue
            try:
                o = f(gi)
            except Exception:
                o = None
            if o is not None and getattr(o, "ndim", 0) == 2 and o.size > 0:
                cand.append(o)
        attempts.append(cand[:2])
    return attempts


def solve(train, test_inputs, budget):
    """FULL inventor: induce per-part causal rules, COMPOSE into a task mechanism, verify, apply to test.
    Records a verified schema into the experience library for cross-task reuse this run."""
    fn, schema = _invent(train, budget, use_prior=True, use_library=True)
    if fn is None:
        # fall back to retrieval so we never under-report coverage (but these won't count as INVENTED)
        fn = _retrieve_template(train)
        schema = None
    if fn is None:
        return [[] for _ in test_inputs]

    # second attempt: a composed-sequence alternative (if the primary was a structural inducer) OR the
    # retrieval template, giving the ARC 2-attempt budget without inflating invention.
    fn2 = None
    if schema is not None:
        fn2 = _retrieve_template(train)  # harmless backup; primary (fn) is the invented one

    # store verified abstracted schema for transfer (only genuine invented mechanisms)
    if schema is not None and schema[0] not in (None,):
        sig = (schema[0], tuple(sorted((k, str(v)) for k, v in schema[1].items())
                                if schema[0] != "block_tiling" else ()))
        if not any(s[0] == schema[0] and s[1] == schema[1] for s in _LIBRARY):
            _LIBRARY.insert(0, schema)
            del _LIBRARY[8:]  # keep small

    return _attempts_from_fn(fn, fn2, test_inputs)


def solve_ablated(train, test_inputs, budget):
    """INVENTION DISABLED: single-whole-template retrieval ONLY. No decomposition, no composition of
    induced local rules, no experience library. This is the boundary INVENTED must beat."""
    fn = _retrieve_template(train)
    if fn is None:
        return [[] for _ in test_inputs]
    return _attempts_from_fn(fn, None, test_inputs)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    # quick sanity: a per-cell colormap task the FULL solver should invent but ablation should not.
    gi1 = np.array([[1, 2], [3, 1]]); go1 = np.array([[4, 5], [6, 4]])
    gi2 = np.array([[2, 3], [1, 2]]); go2 = np.array([[5, 6], [4, 5]])
    train = [(gi1, go1), (gi2, go2)]
    reset_library()
    att = solve(train, [np.array([[3, 1], [2, 3]])], 3000)
    abl = solve_ablated(train, [np.array([[3, 1], [2, 3]])], 3000)
    print("cellmap full attempt:", att[0][0].tolist() if att[0] else None)
    print("cellmap ablated:", abl[0][0].tolist() if abl[0] else "(none/abstain)")
    print("library after solve:", _LIBRARY)
