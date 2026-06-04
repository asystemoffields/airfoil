#!/usr/bin/env python3
"""GEN-4 INVENTOR #3 — NON-DIRECTED COVERAGE.

THE LEVER. Greedy grid-distance search (the gen-0..gen-2 backbone) fixates on compositions that make
IMMEDIATE progress toward the target grid. It therefore never tries SETUP moves — a first step that
makes NO grid-distance progress (often makes the grid look *worse*) but OPENS a structural dimension a
later step needs: mark-then-move, draw-scaffold-then-fill, copy-template-to-marker, gravity-then-recolor.
The toy result that motivates this generation: a coverage-driven explorer finds the 1-in-N "repurposing"
op that greedy search never reaches, because it ranks candidates by the DIVERSITY of structural effects
they open, not by distance shrunk.

WHAT THIS MODULE IS.
  * A modest RELATIONAL ALPHABET of grid->grid primitives that gen2_base's parametric menu cannot express:
    8-direction ray drawing from seeds, connect-same-color in all 8 directions (incl. diagonal),
    mark-the-distinguished-cell's neighborhood, per-object gravity / snap-to-wall, stamp-a-template-to-
    each-marker, relational recolor by rank/position, scaffold (bbox / cross) primitives.
  * A COVERAGE-DRIVEN COMPOSITIONAL EXPLORER: BFS over short programs (len<=3) where the frontier is kept
    not by grid-distance but by EFFECT-COVERAGE — we keep a program if it produces a structurally NOVEL
    intermediate (a new "effect signature": changed-cell pattern / new colors / new component structure),
    even when it is FARTHER from the target. This surfaces multi-step chains a greedy beam prunes.
  * An EXACT VERIFIER: only programs reproducing EVERY train output exactly survive; the test attempt is
    then the verified program applied to the test input. The proposer is broad; the verifier is merciless.

STANDARDIZED GATE (non-negotiable).
    solve_ablated(train, test_inputs, budget)  ==  gen2_base.solve   (imported verbatim)
    solve(train, test_inputs, budget)          ==  gen2_base.solve as attempt-1 backstop,
                                                    + THIS invention as attempt-2.
  Then invention_gate's INVENTED = solves beyond gen2_base = the real creativity number. We develop/tune
  ONLY against gen2_base's TRAIN misses; arc1-eval is held-out (reported, never tuned to).

INTEGRITY. solve()/solve_ablated() learn ONLY from (a) the current task's train pairs, (b) module-level
state from prior solve() calls this run, (c) self-generated synthetic data at import. NEVER read ARC task
files or test OUTPUTS at solve time, no network, no LLM. Respect budget. Pure python + numpy. Build <~90s.
Run/imported with /data/llm/.venv/bin/python from .../incubation/evolve."""
import os
import sys
import time
from collections import deque, Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
for p in (HERE, ARC):
    if p not in sys.path:
        sys.path.insert(0, p)

import dsl  # the shared alphabet / loader

# Import the STANDARDIZED ablation = the best retrieval solver. solve_ablated IS gen2_base.solve, and it
# is also our attempt-1 backstop, so we never regress below retrieval.
import importlib.util as _ilu
_g2spec = _ilu.spec_from_file_location("gen2_base_for_g4_03", os.path.join(HERE, "gen2_base.py"))
gen2_base = _ilu.module_from_spec(_g2spec)
_g2spec.loader.exec_module(gen2_base)

META = {"name": "gen4_03_nondirected-coverage",
        "desc": "gen2_base retrieval backstop (attempt-1) + non-directed coverage-driven compositional "
                "invention over a relational alphabet (8-dir rays/connect, mark-then-move, scaffold-then-"
                "fill, stamp-to-marker, gravity, relational recolor), exact-verified (attempt-2)."}


# ===========================================================================
# small helpers
# ===========================================================================
def _eq(a, b):
    return a is not None and getattr(a, "shape", None) == getattr(b, "shape", None) and np.array_equal(a, b)


def _bg(g):
    v, ct = np.unique(g, return_counts=True)
    return int(v[ct.argmax()])


def _bg_train(train):
    c = Counter()
    for gi, _ in train:
        v, ct = np.unique(gi, return_counts=True)
        for vi, ci in zip(v, ct):
            c[int(vi)] += int(ci)
    return c.most_common(1)[0][0] if c else 0


def _components(g, bg=0, diag=False, by_color=True):
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    if diag:
        nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
                    for di, dj in nb:
                        x, y = a + di, b + dj
                        if 0 <= x < h and 0 <= y < w and g[x, y] != bg and not seen[x, y]:
                            if (not by_color) or g[x, y] == c0:
                                seen[x, y] = True
                                q.append((x, y))
                comps.append(cells)
    return comps


# ===========================================================================
# THE RELATIONAL ALPHABET
# ---------------------------------------------------------------------------
# Each entry is a parametric FAMILY: given the task (bg, palette), it yields a small set of concrete
# grid->grid closures with a human-readable tag. These are the primitives the coverage explorer composes.
# Crucially several of them make NO greedy progress on their own (a bare scaffold/mark step) — that is the
# point: the explorer keeps them for the dimensions they open.
# ===========================================================================
_DIRS8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRD = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _ray_draw(g, bg, dirs, fixed_color=None):
    """From every non-bg seed cell, shoot a ray in each `dir` until the grid edge (or until it hits
    another non-bg cell), painting bg cells along the way with the seed's color (or fixed_color)."""
    h, w = g.shape
    out = g.copy()
    seeds = np.argwhere(g != bg)
    for r, c in seeds:
        col = fixed_color if fixed_color is not None else g[r, c]
        for dr, dc in dirs:
            i, j = r + dr, c + dc
            while 0 <= i < h and 0 <= j < w:
                if g[i, j] != bg:
                    break
                out[i, j] = col
                i += dr
                j += dc
    return out


def _connect_same(g, bg, dirs):
    """Connect every pair of same-color seeds that are aligned along one of `dirs` and have only bg
    between them; paint the gap with that color. Covers row/col AND diagonal connects."""
    h, w = g.shape
    out = g.copy()
    for col in np.unique(g):
        if col == bg:
            continue
        pts = [tuple(p) for p in np.argwhere(g == col)]
        ps = set(pts)
        for (r, c) in pts:
            for dr, dc in dirs:
                if dr < 0 or (dr == 0 and dc < 0):
                    continue  # each axis once (the partner search handles the other direction)
                i, j = r + dr, c + dc
                path = []
                hit = False
                while 0 <= i < h and 0 <= j < w:
                    if (i, j) in ps:
                        hit = True
                        break
                    if g[i, j] != bg:
                        break
                    path.append((i, j))
                    i += dr
                    j += dc
                if hit and path:
                    for (a, b) in path:
                        out[a, b] = col
    return out


def _mark_distinguished_box(g, bg, fill_color):
    """Find the cell/object whose color is UNIQUE (appears once) — the 'distinguished' seed — and return
    a small grid that is the 3x3 box around it recolored to fill_color (a classic mark-then-crop)."""
    cnt = Counter(int(x) for x in g.flatten() if x != bg)
    uniq = [c for c, k in cnt.items() if k == 1]
    if len(uniq) != 1:
        return None
    pos = np.argwhere(g == uniq[0])
    if len(pos) != 1:
        return None
    r, c = pos[0]
    h, w = g.shape
    r0, r1 = max(0, r - 1), min(h, r + 2)
    c0, c1 = max(0, c - 1), min(w, c + 2)
    sub = g[r0:r1, c0:c1].copy()
    out = np.full_like(sub, fill_color)
    out[1 if r > 0 else 0, 1 if c > 0 else 0] = uniq[0] if uniq[0] != fill_color else fill_color
    return out


def _distinguished_cell(g, bg):
    """The cell whose COLOR is unique (appears exactly once). Returns (r,c,color) or None."""
    cnt = Counter(int(x) for x in g.flatten() if x != bg)
    uniq = [c for c, k in cnt.items() if k == 1]
    if len(uniq) != 1:
        return None
    pos = np.argwhere(g == uniq[0])
    if len(pos) != 1:
        return None
    r, c = pos[0]
    return int(r), int(c), uniq[0]


def _markbox_inplace(g, bg, surround):
    """In-place: clear the grid to bg, then around the DISTINGUISHED cell draw a 3x3 box whose ring is
    `surround` and whose center keeps the distinguished color. A clean mark-then-isolate setup."""
    d = _distinguished_cell(g, bg)
    if d is None:
        return None
    r, c, col = d
    h, w = g.shape
    out = np.full_like(g, bg)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            i, j = r + dr, c + dc
            if 0 <= i < h and 0 <= j < w:
                out[i, j] = col if (dr == 0 and dc == 0) else surround
    return out


def _keep_distinguished_object(g, bg, diag=True):
    """Keep only the object that is UNIQUE by some property (unique color, or unique size); clear the
    rest to bg. A reduce-to-winner construction move."""
    comps = _components(g, bg=bg, diag=diag, by_color=False)
    if len(comps) < 2:
        return None
    # unique by size
    sizes = [len(c) for c in comps]
    sc = Counter(sizes)
    uniq_sz = [s for s, k in sc.items() if k == 1]
    target = None
    if len(uniq_sz) == 1:
        target = next(c for c in comps if len(c) == uniq_sz[0])
    else:
        # unique by color
        colors = [int(g[c[0][0], c[0][1]]) for c in comps]
        cc = Counter(colors)
        uniq_col = [col for col, k in cc.items() if k == 1]
        if len(uniq_col) == 1:
            target = next(c for c in comps if int(g[c[0][0], c[0][1]]) == uniq_col[0])
    if target is None:
        return None
    out = np.full_like(g, bg)
    for a, b in target:
        out[a, b] = g[a, b]
    return out


def _object_gravity(g, bg, direction):
    """Move each connected OBJECT as a rigid body until it hits a wall or another object (object-level
    gravity, distinct from per-column cell gravity)."""
    dr, dc = direction
    h, w = g.shape
    comps = _components(g, bg=bg, diag=True, by_color=False)
    # order objects so the leading edge moves first
    def key(comp):
        rs = [a for a, _ in comp]; cs = [b for _, b in comp]
        return (-max(rs) if dr > 0 else min(rs)) if dr != 0 else (-max(cs) if dc > 0 else min(cs))
    comps.sort(key=key)
    out = np.full_like(g, bg)
    occ = np.zeros((h, w), bool)
    for comp in comps:
        cells = comp
        cols = {(a, b): g[a, b] for a, b in cells}
        shift = 0
        while True:
            ok = True
            for (a, b) in cells:
                na, nb = a + dr * (shift + 1), b + dc * (shift + 1)
                if not (0 <= na < h and 0 <= nb < w) or occ[na, nb]:
                    ok = False
                    break
            if not ok:
                break
            shift += 1
        for (a, b) in cells:
            na, nb = a + dr * shift, b + dc * shift
            out[na, nb] = cols[(a, b)]
            occ[na, nb] = True
    return out


def _stamp_to_markers(g, bg):
    """If the grid contains ONE multi-cell template object and several single-cell markers of another
    color, stamp the template (anchored at its own seed-marker cell if present, else center) onto each
    marker. A canonical object-to-marker copy. Returns None if structure not present."""
    comps = _components(g, bg=bg, diag=True, by_color=False)
    if len(comps) < 2:
        return None
    sizes = [len(c) for c in comps]
    big = [c for c in comps if len(c) == max(sizes)]
    if len(big) != 1 or max(sizes) < 2:
        return None
    template = big[0]
    markers = [c for c in comps if len(c) == 1]
    if not markers:
        return None
    rs = [a for a, _ in template]; cs = [b for _, b in template]
    r0, c0 = min(rs), min(cs)
    rel = [(a - r0, b - c0, g[a, b]) for a, b in template]
    th = max(rs) - r0
    tw = max(cs) - c0
    # anchor = the cell in the template matching a marker color, else template's own min corner
    marker_color = g[markers[0][0]]
    anchor = None
    for (da, db, cc) in rel:
        if cc == marker_color:
            anchor = (da, db)
            break
    if anchor is None:
        anchor = (th // 2, tw // 2)
    h, w = g.shape
    out = g.copy()
    for mk in markers:
        mr, mc = mk[0]
        for (da, db, cc) in rel:
            na, nb = mr - anchor[0] + da, mc - anchor[1] + db
            if 0 <= na < h and 0 <= nb < w:
                out[na, nb] = cc
    return out


def _fill_enclosed(g, bg, color):
    """Flood from the border over bg; bg cells NOT reached are 'enclosed' -> paint color (scaffold-fill)."""
    h, w = g.shape
    reach = np.zeros((h, w), bool)
    q = deque()
    for i in range(h):
        for j in (0, w - 1):
            if g[i, j] == bg and not reach[i, j]:
                reach[i, j] = True; q.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if g[i, j] == bg and not reach[i, j]:
                reach[i, j] = True; q.append((i, j))
    while q:
        i, j = q.popleft()
        for dr, dc in _DIRS4:
            a, b = i + dr, j + dc
            if 0 <= a < h and 0 <= b < w and g[a, b] == bg and not reach[a, b]:
                reach[a, b] = True; q.append((a, b))
    out = g.copy()
    out[(g == bg) & ~reach] = color
    return out


def _bbox_scaffold(g, bg, color, mode):
    """Per-object scaffold: draw the bounding-box outline ('frame') or fill ('solid') of each object in
    `color`. A setup move (frame) that a later fill can use."""
    comps = _components(g, bg=bg, diag=True, by_color=False)
    out = g.copy()
    for comp in comps:
        rs = [a for a, _ in comp]; cs = [b for _, b in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if mode == "frame":
            for c in range(c0, c1 + 1):
                if out[r0, c] == bg: out[r0, c] = color
                if out[r1, c] == bg: out[r1, c] = color
            for r in range(r0, r1 + 1):
                if out[r, c0] == bg: out[r, c0] = color
                if out[r, c1] == bg: out[r, c1] = color
        else:  # solid
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if out[r, c] == bg:
                        out[r, c] = color
    return out


def _recolor_by_rank(g, bg, mapping, key):
    """Recolor each object solid by a learned key->color map; key in {'size','rank_desc','rank_asc'}."""
    comps = _components(g, bg=bg, diag=True, by_color=False)
    if not comps:
        return None
    sizes = sorted({len(c) for c in comps})
    out = g.copy()
    for comp in comps:
        if key == "size":
            k = len(comp)
        elif key == "rank_asc":
            k = sizes.index(len(comp))
        else:
            k = len(sizes) - 1 - sizes.index(len(comp))
        if k not in mapping:
            return None
        for a, b in comp:
            out[a, b] = mapping[k]
    return out


def build_alphabet(train):
    """Construct the concrete relational primitive closures for THIS task (uses only train inputs/palette,
    never outputs). Returns a list of (tag, fn). Designed so several primitives make no greedy progress on
    their own (bare scaffolds / marks) — the coverage explorer is what makes those pay."""
    bg = _bg_train(train)
    pal = sorted(set().union(*[set(np.unique(gi).tolist()) for gi, _ in train]) |
                 set().union(*[set(np.unique(go).tolist()) for _, go in train]))
    fg_colors = [c for c in pal if c != bg]
    alpha = []

    # --- ray drawing (8 / 4 / diagonal), seed-colored and fixed-color ---
    for dn, dirs in (("8", _DIRS8), ("4", _DIRS4), ("diag", _DIRD)):
        alpha.append(("ray_%s" % dn, lambda g, _d=dirs, _bg=bg: _ray_draw(g, _bg, _d)))
        for fc in fg_colors:
            alpha.append(("ray_%s_c%d" % (dn, fc),
                          lambda g, _d=dirs, _bg=bg, _fc=fc: _ray_draw(g, _bg, _d, _fc)))

    # --- connect same-color seeds (row/col + diagonal + all 8) ---
    for dn, dirs in (("4", _DIRS4), ("diag", _DIRD), ("8", _DIRS8)):
        alpha.append(("connect_%s" % dn, lambda g, _d=dirs, _bg=bg: _connect_same(g, _bg, _d)))

    # --- mark distinguished cell -> 3x3 box (mark-then-crop; SHAPE-CHANGING setup) ---
    for fc in fg_colors:
        alpha.append(("markbox_c%d" % fc, lambda g, _bg=bg, _fc=fc: _mark_distinguished_box(g, _bg, _fc)))

    # --- in-place mark distinguished cell with a surround ring (clear rest) ---
    for fc in fg_colors:
        alpha.append(("markbox_ip_c%d" % fc, lambda g, _bg=bg, _fc=fc: _markbox_inplace(g, _bg, _fc)))

    # --- keep only the distinguished object (reduce-to-winner) ---
    alpha.append(("keep_distinguished", lambda g, _bg=bg: _keep_distinguished_object(g, _bg)))

    # --- object-level gravity / snap to each wall ---
    for nm, d in (("down", (1, 0)), ("up", (-1, 0)), ("left", (0, -1)), ("right", (0, 1))):
        alpha.append(("objgrav_%s" % nm, lambda g, _d=d, _bg=bg: _object_gravity(g, _bg, _d)))

    # --- stamp template to markers (object-to-marker copy) ---
    alpha.append(("stamp_markers", lambda g, _bg=bg: _stamp_to_markers(g, _bg)))

    # --- scaffold-then-fill primitives ---
    for fc in fg_colors:
        alpha.append(("fill_enclosed_c%d" % fc, lambda g, _bg=bg, _fc=fc: _fill_enclosed(g, _bg, _fc)))
        alpha.append(("frame_c%d" % fc, lambda g, _bg=bg, _fc=fc: _bbox_scaffold(g, _bg, _fc, "frame")))
        alpha.append(("solid_c%d" % fc, lambda g, _bg=bg, _fc=fc: _bbox_scaffold(g, _bg, _fc, "solid")))

    # --- relational recolor by object rank (learned mapping from train; added below in fit) ---
    # (added by fit_recolor_rank, which needs outputs; kept out of the bare alphabet)

    # --- a few cheap geometric repurposings that can serve as the FIRST coverage-opening move ---
    geo = [("crop_content", dsl.crop_content), ("reflect_h", dsl.reflect_h), ("reflect_v", dsl.reflect_v),
           ("rot90", dsl.rot90), ("rot180", dsl.rot180), ("transpose", dsl.transpose),
           ("gravity_down", dsl.gravity_down), ("gravity_up", dsl.gravity_up),
           ("gravity_left", dsl.gravity_left), ("gravity_right", dsl.gravity_right)]
    for nm, fn in geo:
        alpha.append((nm, fn))

    # color-perm closures: any single-recolor a->b among palette (cheap relational recolors)
    for a in fg_colors:
        for b in pal:
            if a != b:
                alpha.append(("recolor_%d_%d" % (a, b),
                              lambda g, _a=a, _b=b: np.where(g == _a, _b, g)))

    return alpha, bg, pal, fg_colors


# ===========================================================================
# fitted relational-recolor by object rank (needs train outputs; learned, then becomes a primitive)
# ===========================================================================
def fit_recolor_rank(train, bg):
    if not all(i.shape == o.shape for i, o in train):
        return []
    out = []
    for key in ("size", "rank_desc", "rank_asc"):
        mapping = {}
        ok = True
        for i, o in train:
            if np.any((i != bg) != (o != bg)):
                ok = False; break
            comps = _components(i, bg=bg, diag=True, by_color=False)
            if not comps:
                ok = False; break
            sizes = sorted({len(c) for c in comps})
            for comp in comps:
                ocs = {int(o[a, b]) for a, b in comp}
                if len(ocs) != 1:
                    ok = False; break
                oc = ocs.pop()
                if key == "size":
                    k = len(comp)
                elif key == "rank_asc":
                    k = sizes.index(len(comp))
                else:
                    k = len(sizes) - 1 - sizes.index(len(comp))
                if k in mapping and mapping[k] != oc:
                    ok = False; break
                mapping[k] = oc
            if not ok:
                break
        if ok and mapping:
            out.append(("recolor_rank_%s" % key,
                        lambda g, _m=dict(mapping), _k=key, _bg=bg: _recolor_by_rank(g, _bg, _m, _k)))
    return out


# ===========================================================================
# EFFECT SIGNATURE — the coverage axis. Two intermediate grids have the SAME effect-signature if they
# are structurally indistinguishable for the purpose of opening new compositions. We keep one program per
# novel signature so the frontier stays diverse instead of clustering near the (greedy) target.
# ===========================================================================
def _effect_sig(g0, g):
    """A compact structural signature of the EFFECT of transforming g0 -> g (relative to the start grid),
    not the absolute grid. Captures: shape, palette, #changed cells bucketed, #components bucketed, and a
    coarse changed-mask hash. Coverage = number of DISTINCT signatures explored."""
    if g is None:
        return ("none",)
    sh = g.shape
    pal = tuple(sorted(np.unique(g).tolist()))
    if g0 is not None and g0.shape == g.shape:
        ch = int((g0 != g).sum())
        chb = 0 if ch == 0 else (1 if ch <= 3 else (2 if ch <= 12 else 3))
    else:
        chb = 9  # shape-changing step
    try:
        ncomp = len(_components(g, bg=_bg(g), diag=True, by_color=False))
    except Exception:
        ncomp = -1
    ncb = 0 if ncomp == 0 else (1 if ncomp <= 2 else (2 if ncomp <= 6 else 3))
    return (sh, pal, chb, ncb)


# ===========================================================================
# THE NON-DIRECTED COVERAGE EXPLORER
# ---------------------------------------------------------------------------
# BFS over programs of length <= max_len. The frontier is selected by EFFECT-COVERAGE: at each depth we
# keep at most one representative program per novel effect-signature (deduplicated across the run so a
# setup move that "looks worse" is retained for the dimension it opens). Every program is exact-checked
# against the train outputs; the FIRST verified program is returned. Greedy distance is used ONLY as a
# soft tiebreak among same-signature candidates (so we still prefer cleaner intermediates), never to prune
# a whole signature.
# ===========================================================================
def _apply(fn, grids):
    out = []
    for g in grids:
        try:
            r = fn(g)
        except Exception:
            return None
        if r is None or getattr(r, "ndim", None) != 2 or r.size == 0 or r.size > 2500:
            return None
        out.append(np.asarray(r, int))
    return out


def _verify_on_train(prog_outs, train):
    return all(_eq(o, go) for o, (_, go) in zip(prog_outs, train))


def _gdist(outs, tgts):
    s = 0.0
    for a, b in zip(outs, tgts):
        if a.shape != b.shape:
            s += 1.0 + abs(a.size - b.size) / max(a.size, b.size, 1)
        else:
            s += float((a != b).mean())
    return s / max(len(outs), 1)


def coverage_explore(train, alphabet, budget, max_len=3, per_depth=160, time_cap=6.0):
    """Return a list of verified programs (each = list of (tag, fn)) found by non-directed coverage BFS.
    budget bounds the number of primitive applications (exec count); time_cap bounds wall-clock seconds."""
    ins = [gi for gi, _ in train]
    tgts = [go for _, go in train]
    t_start = time.time()

    # frontier entries: (prog, outs)  ; programs grouped by effect-signature for coverage dedup
    start = (tuple(), ins)
    frontier = [start]
    seen_sigs = set()
    seen_sigs.add(tuple(_effect_sig(None, g) for g in ins))
    verified = []
    nexec = 0

    for depth in range(max_len):
        # generate children of the current frontier, bucketed by effect-signature for coverage
        buckets = {}  # sig-tuple -> (gdist, prog, outs)
        for prog, outs in frontier:
            if time.time() - t_start > time_cap:
                return verified
            for tag, fn in alphabet:
                if nexec >= budget:
                    break
                outs2 = _apply(fn, outs)
                nexec += 1
                if outs2 is None:
                    continue
                # exact verify FIRST (the merciless verifier) — any verified program is banked immediately
                if _verify_on_train(outs2, train):
                    verified.append(prog + ((tag, fn),))
                    if len(verified) >= 6:
                        return verified
                    continue
                # coverage key: per-train-pair effect signature relative to that pair's START input
                sig = tuple(_effect_sig(i0, o2) for i0, o2 in zip(ins, outs2))
                if sig in seen_sigs:
                    # already covered this structural effect at some depth -> still allow if a fresher
                    # (lower-distance) representative, but never expand the count (coverage, not greed)
                    pass
                d = _gdist(outs2, tgts)
                cur = buckets.get(sig)
                if cur is None or d < cur[0]:
                    buckets[sig] = (d, prog + ((tag, fn),), outs2)
            if nexec >= budget:
                break

        # promote: keep NOVEL signatures first (coverage), then fill remaining slots by best-distance among
        # already-seen signatures (so we don't dead-end). This is the non-directed selection: a structurally
        # new effect is kept even if it is FARTHER from the target than a familiar one.
        novel = [(d, prog, outs, sig) for sig, (d, prog, outs) in buckets.items() if sig not in seen_sigs]
        repeat = [(d, prog, outs, sig) for sig, (d, prog, outs) in buckets.items() if sig in seen_sigs]
        novel.sort(key=lambda x: x[0])
        repeat.sort(key=lambda x: x[0])
        chosen = novel[:per_depth]
        if len(chosen) < per_depth:
            chosen += repeat[:(per_depth - len(chosen))]
        new_frontier = []
        for d, prog, outs, sig in chosen:
            seen_sigs.add(sig)
            new_frontier.append((prog, outs))
        frontier = new_frontier
        if not frontier or nexec >= budget:
            break

    return verified


# ===========================================================================
# in-session experience: tags of relational primitives that have verified before go FIRST next time
# (an experience prior on the coverage alphabet; verified-correct only; no grids stored).
# ===========================================================================
_TAG_HITS = Counter()
_TASK_N = [0]


def reset_library():
    """Clear cross-task experience (the coverage-prior) so the gate can measure a cold run."""
    _TAG_HITS.clear()
    _TASK_N[0] = 0
    if hasattr(gen2_base, "reset_library"):
        try:
            gen2_base.reset_library()
        except Exception:
            pass
    # also reset gen2_base's library state if exposed via _LIB
    try:
        lib = getattr(gen2_base, "_LIB", None)
        if lib is not None:
            lib.concept_hits.clear(); lib.closures.clear(); lib.closure_tags.clear()
            lib.macro_src.clear(); lib.macros.clear(); lib.op_hits.clear()
            lib.solved_progs.clear(); lib.audit.clear()
    except Exception:
        pass


def _order_alphabet(alpha):
    """Experience prior: primitives whose tag-base verified before go first (coverage stays the selector;
    this only reorders WHICH novel signatures are discovered first under a tight budget)."""
    def base(tag):
        return tag.split("_c")[0].split("_")[0]
    return sorted(alpha, key=lambda tf: -_TAG_HITS.get(base(tf[0]), 0))


# ===========================================================================
# THE INVENTION (attempt-2): build alphabet, fit learned relational recolors, run coverage explorer,
# return the verified program's output on each test input. Returns None if nothing verifies.
# ===========================================================================
def _invent(train, test_inputs, budget):
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    if not train:
        return None
    alpha, bg, pal, fg = build_alphabet(train)
    alpha = alpha + fit_recolor_rank(train, bg)
    alpha = _order_alphabet(alpha)

    # budget for coverage exec: a fraction of the per-task budget (the rest is gen2_base's own search,
    # which already ran as attempt-1). Cap exec to keep within time. Scale the multi-step frontier DOWN
    # for large grids (per-primitive cost is O(cells); depth-1 relational solves stay cheap regardless,
    # and the rare multi-step chains tolerate a smaller frontier).
    cells = max((gi.size for gi, _ in train), default=64)
    expl_budget = max(600, min(int(budget) * 2, 7000))
    if cells <= 100:
        per_depth, tcap = 160, 3.0
    elif cells <= 400:
        per_depth, tcap = 90, 3.0
    else:
        per_depth, tcap = 40, 2.5
    progs = coverage_explore(train, alpha, expl_budget, max_len=3,
                             per_depth=per_depth, time_cap=tcap)
    if not progs:
        return None

    # shortest verified program first (MDL razor among verified); record experience
    progs.sort(key=len)
    best = progs[0]
    for tag, _ in best:
        _TAG_HITS[tag.split("_c")[0].split("_")[0]] += 1

    # apply each verified program to the test inputs; collect up to 2 distinct candidate outputs/test
    test_inputs = [np.asarray(t, int) for t in test_inputs]
    attempts = []
    for gi in test_inputs:
        cand = []
        for prog in progs[:4]:
            g = gi
            ok = True
            for tag, fn in prog:
                try:
                    g = fn(g)
                except Exception:
                    ok = False; break
                if g is None or getattr(g, "ndim", None) != 2 or g.size == 0:
                    ok = False; break
                g = np.asarray(g, int)
            if ok and not any(_eq(g, c) for c in cand):
                cand.append(g)
            if len(cand) >= 2:
                break
        attempts.append(cand)
    if all(len(a) == 0 for a in attempts):
        return None
    return attempts


# ===========================================================================
# PUBLIC ENTRYPOINTS — the standardized gate.
# ===========================================================================
def solve_ablated(train, test_inputs, budget):
    """ABLATION = the best retrieval solver, verbatim. Invention disabled."""
    return gen2_base.solve(train, test_inputs, budget)


def solve(train, test_inputs, budget):
    """FULL = gen2_base retrieval (BOTH attempts, BACKSTOP, priority) + THIS coverage-invention filling
    any remaining slot.

    ZERO-REGRESSION MERGE. Per test input the candidate order is
        [ gen2_base.attempt0, gen2_base.attempt1, invention.attempt0, invention.attempt1 ]
    deduped, capped at the ARC limit of 2. gen2_base keeps BOTH of its (<=2) attempts, so the full solver
    can NEVER score below gen2_base (no retrieval solve is ever crowded out). On the tasks gen2_base MISSES
    it produces 0-1 attempts (its menu/seed-search find nothing), leaving slot(s) FREE — and there the
    coverage invention fills in. A task gen2_base misses that the invention solves is therefore a certified
    solve-beyond-retrieval (INVENTED) under the standardized gate, with no risk to gen2_base's coverage."""
    _TASK_N[0] += 1
    # retrieval backstop (gen2_base also runs its own concept menu + seed search)
    try:
        base_att = gen2_base.solve(train, test_inputs, budget)
    except Exception:
        base_att = []
    base_att = base_att or []

    # the coverage invention (only consumed for slots gen2_base left free)
    try:
        inv_att = _invent(train, test_inputs, budget)
    except Exception:
        inv_att = None

    n = len(test_inputs)
    out = []
    for k in range(n):
        cand = []
        # 1) gen2_base FIRST and in FULL (both attempts) — guarantees no regression below retrieval
        if k < len(base_att) and base_att[k]:
            for o in base_att[k][:2]:
                if o is not None:
                    oo = np.asarray(o, int)
                    if oo.ndim == 2 and oo.size > 0 and not any(_eq(oo, c) for c in cand):
                        cand.append(oo)
                if len(cand) >= 2:
                    break
        # 2) invention fills any remaining slot(s) (the creativity payload on gen2_base's misses)
        if len(cand) < 2 and inv_att is not None and k < len(inv_att):
            for o in inv_att[k][:2]:
                if o is not None:
                    oo = np.asarray(o, int)
                    if oo.ndim == 2 and oo.size > 0 and not any(_eq(oo, c) for c in cand):
                        cand.append(oo)
                if len(cand) >= 2:
                    break
        out.append(cand[:2])
    return out


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    import harness
    t0 = time.time()
    tasks = harness.load_split("arc1-train", n=10)
    sv = 0
    for tid, tr, te in tasks:
        ti = [gi for gi, _ in te]
        att = solve(tr, ti, 2000)
        ok = all(any(_eq(c, go) for c in (att[k] or [])[:2]) for k, (_, go) in enumerate(te))
        sv += int(ok)
    print("smoke solved", sv, "/", len(tasks), "in %.1fs" % (time.time() - t0))
