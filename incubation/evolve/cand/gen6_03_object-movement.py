#!/usr/bin/env python3
"""GEN-6 RELATION-INDUCER #3 — OBJECT-MOVEMENT / INTERACTION  (+ panel selection/combination).

THE FAMILY (gen2_base TRAIN misses we attack as RELATION-induction, not verb-composition):
  object-movement-by-rule (19) + multi-object-interaction (15). gen-4 found a few of these on TRAIN by
  composition but they did NOT generalize. Per the gen-5 DIAGNOSTIC the lever is per-task FITTED
  cause->effect RELATIONS, exact-verified, invariant across all train pairs. Here the effect is MOTION /
  INTERACTION / panel-selection rather than recolor:

  A) MOVE-rule induction (object granularity, shape-preserving grids):
     - translate_const          : every object shifts by ONE fitted vector (dr,dc).
     - gravity_blocked          : every object falls in a fitted cardinal direction until it hits the wall
                                   or another object (sokoban gravity / stacking).
     - move_to_edge             : every object snaps to a fitted grid edge (kept from gen-5, re-verified).
     - move_objs_to_markers     : multi-cell objects translate so they OVERLAY matching single-cell markers
                                   (match by color, else by relative geometry); the classic
                                   object-moves-onto-its-target family (e6721834, 7df24a62-ish).
     - stamp_template_at_markers: a TEMPLATE object (the unique multi-cell shape, possibly taken from a
                                   gridline-separated panel) is copied, anchored, at every marker cell
                                   (363442ee-style copy/interaction).

  B) PANEL selection / combination (gridline-separated OR equal-size stacked panels):
     - select_panel             : split into panels, SELECT the one chosen by an induced relation
                                   (the unique panel / the most-common panel / argmax-min of a panel
                                   feature); output = that panel verbatim (662c240a, a87f7484, 1190e5a7).
     - combine_panels           : overlay panels under a fitted boolean rule (AND/OR/XOR/DIFF on the
                                   non-bg mask) and paint with a fitted output color (995c5fa3, 7b7f7511).

  Each rule is a FUNCTION fitted to be CONSTANT across ALL train pairs and EXACT-verified on the train
  outputs; the held-out test is the intervention that certifies it.

STANDARDIZED GATE (non-negotiable):
  solve_ablated(train,test_inputs,budget) == EXACTLY gen2_base.solve  (imported verbatim, strong retrieval).
  solve() = gen2_base attempt-1 backstop  +  THIS movement/interaction induction as attempt-2.
  INVENTED = solves beyond gen2_base.

INTEGRITY. solve() learns ONLY from (a) the current task's train pairs, (b) module-level state from PRIOR
solve() calls this run (verified-correct only), (c) self-generated synthetic data built at import. NEVER reads
ARC task files / test OUTPUTS, no network, no LLM. Respects budget. Pure python + numpy. build < ~90s.
Run with /data/llm/.venv/bin/python from .../incubation/evolve.
"""
import os
import sys
import time
from collections import deque, Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVOLVE = os.path.dirname(HERE)
for p in (HERE, EVOLVE):
    if p not in sys.path:
        sys.path.insert(0, p)

# The standardized strong-retrieval ablation, imported verbatim.
import gen2_base as BASE

META = {"name": "gen6_03_object-movement",
        "desc": "systematic per-task MOVE/INTERACTION relation-induction (translate/gravity/move-to-marker/"
                "stamp-at-markers) + panel selection/combination, exact-verified across all train pairs; "
                "gen2_base strong-retrieval backstop"}


# ===========================================================================
# tiny grid helpers
# ===========================================================================
def _eq(a, b):
    return a is not None and getattr(a, "shape", None) == getattr(b, "shape", None) and np.array_equal(a, b)


def _bg_train(train):
    c = Counter()
    for gi, _ in train:
        v, ct = np.unique(gi, return_counts=True)
        for vi, ci in zip(v, ct):
            c[int(vi)] += int(ci)
    return c.most_common(1)[0][0] if c else 0


def _verify(fn, train):
    for gi, go in train:
        try:
            o = fn(gi)
        except Exception:
            return False
        if not _eq(o, go):
            return False
    return True


def _components(g, bg=0, diag=False, by_color=False):
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


def _bbox(cells):
    rs = [a for a, _ in cells]
    cs = [b for _, b in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _footprint(g, cells):
    """Canonical (top-left normalized) (dr,dc,color) frozenset of an object."""
    r0, _, c0, _ = _bbox(cells)
    return frozenset((a - r0, b - c0, int(g[a, b])) for a, b in cells)


def _shape_only(cells):
    r0, _, c0, _ = _bbox(cells)
    return frozenset((a - r0, b - c0) for a, b in cells)


# ===========================================================================
# segmentations to try (fitted by which one lets a relation verify)
# ===========================================================================
def _segmentations():
    for diag in (True, False):
        for by_color in (False, True):
            yield (diag, by_color)


# ===========================================================================
# A) MOVE-RULE INDUCERS  (shape-preserving)
# ===========================================================================
def _place_objs(shape, objs_with_colors, bg):
    """objs_with_colors: list of (cells, dr, dc) with original colors looked up from a source grid by the
    inducer. Here we accept (list of (r,c,color)) pre-translated cells."""
    out = np.full(shape, bg, int)
    for cells in objs_with_colors:
        for (a, b, col) in cells:
            if 0 <= a < shape[0] and 0 <= b < shape[1]:
                out[a, b] = col
    return out


def induce_translate_const(train, bg, seg):
    """RELATION: every object (and hence the whole foreground) shifts by ONE fitted (dr,dc). Detect the
    vector from pair 0 by cross-correlating the non-bg masks; verify on all pairs by a strict roll-compare."""
    diag, by_color = seg
    if any(gi.shape != go.shape for gi, go in train):
        return None
    gi0, go0 = train[0]
    h, w = gi0.shape
    mi = (gi0 != bg)
    mo = (go0 != bg)
    if mi.sum() == 0 or mi.sum() != mo.sum():
        return None
    # candidate shifts: try every (dr,dc) that maps a chosen input fg cell to some output fg cell — but
    # cheaper: brute force small shift range bounded by grid size, ranked by |dr|+|dc|.
    cand = []
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            cand.append((abs(dr) + abs(dc), dr, dc))
    cand.sort()
    for _m, dr, dc in cand:
        rolled = np.full_like(gi0, bg)
        ok_shift = True
        for (a, b) in zip(*np.where(mi)):
            x, y = a + dr, b + dc
            if not (0 <= x < h and 0 <= y < w):
                ok_shift = False
                break
            rolled[x, y] = gi0[a, b]
        if not ok_shift:
            continue
        if _eq(rolled, go0):
            def fn(g, dr=dr, dc=dc, bg=bg):
                h2, w2 = g.shape
                m = (g != bg)
                out = np.full_like(g, bg)
                for (a, b) in zip(*np.where(m)):
                    x, y = a + dr, b + dc
                    if not (0 <= x < h2 and 0 <= y < w2):
                        return None
                    out[x, y] = g[a, b]
                return out
            if _verify(fn, train):
                return fn
    return None


def induce_gravity_blocked(train, bg, seg):
    """RELATION: every object slides in a fitted cardinal direction until it hits the wall or another
    object (objects retain shape+color; collisions stack). Try all 4 directions; verify exactly."""
    diag, by_color = seg
    if any(gi.shape != go.shape for gi, go in train):
        return None

    def apply(g, direction, diag, by_color, bg):
        h, w = g.shape
        comps = _components(g, bg=bg, diag=diag, by_color=by_color)
        if not comps:
            return None
        objs = []
        for cells in comps:
            objs.append([(a, b, int(g[a, b])) for a, b in cells])
        # process near-wall first so stacking resolves
        if direction == "down":
            objs.sort(key=lambda o: -max(a for a, _, _ in o))
            step = (1, 0)
        elif direction == "up":
            objs.sort(key=lambda o: min(a for a, _, _ in o))
            step = (-1, 0)
        elif direction == "right":
            objs.sort(key=lambda o: -max(b for _, b, _ in o))
            step = (0, 1)
        else:
            objs.sort(key=lambda o: min(b for _, b, _ in o))
            step = (0, -1)
        occ = np.zeros((h, w), bool)
        out = np.full((h, w), bg, int)
        for o in objs:
            dr = dc = 0
            while True:
                ok = True
                for (a, b, _c) in o:
                    x, y = a + dr + step[0], b + dc + step[1]
                    if not (0 <= x < h and 0 <= y < w) or occ[x, y]:
                        ok = False
                        break
                if not ok:
                    break
                dr += step[0]
                dc += step[1]
            for (a, b, c) in o:
                x, y = a + dr, b + dc
                occ[x, y] = True
                out[x, y] = c
        return out

    for direction in ("down", "up", "left", "right"):
        def fn(g, direction=direction, diag=diag, by_color=by_color, bg=bg):
            return apply(g, direction, diag, by_color, bg)
        if _verify(fn, train):
            return fn
    return None


def induce_move_to_edge(train, bg, seg):
    """RELATION: every object snaps to a fitted grid edge (per-object gravity, no inter-object blocking)."""
    diag, by_color = seg
    if any(gi.shape != go.shape for gi, go in train):
        return None

    def apply_edge(g, edge, diag, by_color, bg):
        h, w = g.shape
        comps = _components(g, bg=bg, diag=diag, by_color=by_color)
        if not comps:
            return None
        out = np.full_like(g, bg)
        order = sorted(comps, key=lambda cells: _bbox(cells)[{"top": 0, "bottom": 1, "left": 2, "right": 3}[edge]]
                       * ({"top": 1, "bottom": -1, "left": 1, "right": -1}[edge]))
        for cells in order:
            r0, r1, c0, c1 = _bbox(cells)
            if edge == "top":
                dr, dc = -r0, 0
            elif edge == "bottom":
                dr, dc = (h - 1 - r1), 0
            elif edge == "left":
                dr, dc = 0, -c0
            else:
                dr, dc = 0, (w - 1 - c1)
            for a, b in cells:
                out[a + dr, b + dc] = g[a, b]
        return out

    for edge in ("top", "bottom", "left", "right"):
        def fn(g, edge=edge, diag=diag, by_color=by_color, bg=bg):
            return apply_edge(g, edge, diag, by_color, bg)
        if _verify(fn, train):
            return fn
    return None


def induce_move_objs_to_markers(train, bg, seg):
    """RELATION: multi-cell objects translate so each OVERLAYS a matching single-cell marker. A marker is a
    size-1 component; an object is a size>=2 component. Match an object to its marker by COLOR (object's
    cell-color set contains the marker color, or they share a color), then translate the object so a chosen
    reference cell lands on the marker (markers consumed; objects removed from their old location).
    Output: markers replaced by the translated objects, everything else bg. Shape-preserving."""
    diag, by_color = seg
    if any(gi.shape != go.shape for gi, go in train):
        return None

    def apply(g, diag, by_color, bg):
        h, w = g.shape
        comps = _components(g, bg=bg, diag=diag, by_color=by_color)
        if len(comps) < 2:
            return None
        objs = [c for c in comps if len(c) >= 2]
        markers = [c for c in comps if len(c) == 1]
        if not objs or not markers:
            return None
        out = g.copy()
        # clear all originals first (objects + markers -> bg)
        for c in comps:
            for a, b in c:
                out[a, b] = bg
        used_marker = [False] * len(markers)
        for o in objs:
            # the object's "anchor" is the cell whose color matches a marker if such a cell exists,
            # else the bbox top-left.
            r0, r1, c0, c1 = _bbox(o)
            ocolors = Counter(int(g[a, b]) for a, b in o)
            # find a marker matching this object
            best = None
            for mi, m in enumerate(markers):
                if used_marker[mi]:
                    continue
                mr, mc = m[0]
                mcol = int(g[mr, mc])
                if mcol in ocolors:
                    # anchor = the object cell of color mcol nearest its bbox center
                    anchor_cells = [(a, b) for a, b in o if int(g[a, b]) == mcol]
                    ar, ac = anchor_cells[0]
                    best = (mi, mr - ar, mc - ac)
                    break
            if best is None:
                return None
            mi, dr, dc = best
            used_marker[mi] = True
            for a, b in o:
                x, y = a + dr, b + dc
                if not (0 <= x < h and 0 <= y < w):
                    return None
                out[x, y] = g[a, b]
        return out

    def fn(g, diag=diag, by_color=by_color, bg=bg):
        return apply(g, diag, by_color, bg)
    if _verify(fn, train):
        return fn
    return None


def induce_stamp_template_at_markers(train, bg, seg):
    """RELATION: ONE template object (the unique multi-cell shape with >=2 cells; the others are size-1
    markers OR a small set) is copied, centered on each marker cell, recolored to the marker's color or
    kept. Shape-preserving. Handles both a gridline-separated template panel and an in-grid template."""
    diag, by_color = seg
    if any(gi.shape != go.shape for gi, go in train):
        return None

    def build(g, bg):
        comps = _components(g, bg=bg, diag=True, by_color=False)
        if len(comps) < 2:
            return None
        comps_sorted = sorted(comps, key=lambda c: -len(c))
        tmpl = comps_sorted[0]
        if len(tmpl) < 2:
            return None
        markers = [c for c in comps if c is not tmpl and len(c) == 1]
        if not markers:
            return None
        return tmpl, markers

    def stamp(g, bg, recolor, anchor_mode):
        info = build(g, bg)
        if info is None:
            return None
        tmpl, markers = info
        r0, r1, c0, c1 = _bbox(tmpl)
        rel = [(a - r0, b - c0, int(g[a, b])) for a, b in tmpl]
        th, tw = r1 - r0 + 1, c1 - c0 + 1
        if anchor_mode == "center":
            ah, aw = th // 2, tw // 2
        else:
            ah, aw = 0, 0
        out = g.copy()
        h, w = g.shape
        for m in markers:
            mr, mc = m[0]
            mcol = int(g[mr, mc])
            for (dr, dc, col) in rel:
                rr, cc = mr - ah + dr, mc - aw + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = mcol if recolor else col
        return out

    for recolor in (False, True):
        for anchor_mode in ("topleft", "center"):
            def fn(g, recolor=recolor, anchor_mode=anchor_mode, bg=bg):
                return stamp(g, bg, recolor, anchor_mode)
            if _verify(fn, train):
                return fn
    return None


def induce_stamp_panel_template(train, bg, seg):
    """RELATION: the grid is split by a single separator (row/col line of one constant color) into a
    TEMPLATE panel (contains a multi-cell block) and a FIELD panel (contains markers). Copy the template
    block onto the field at each marker, anchored top-left of the template block. (363442ee family.)"""
    if any(gi.shape != go.shape for gi, go in train):
        return None

    def panels(g):
        # find a full separator line (single color spanning a whole row or column)
        h, w = g.shape
        sep_rows = [r for r in range(h) if len(set(g[r, :].tolist())) == 1 and g[r, 0] != bg]
        sep_cols = [c for c in range(w) if len(set(g[:, c].tolist())) == 1 and g[0, c] != bg]
        return sep_rows, sep_cols

    def _stamp(g, tmpl_g, field_g, roff, coff, anchor, clear_markers):
        """Stamp the largest template-panel block at each field marker. roff/coff map field-local coords to
        full-grid. anchor in {'topleft','center'}. Returns rendered grid or None."""
        h, w = g.shape
        comps = _components(tmpl_g, bg=bg, diag=True, by_color=False)
        if not comps:
            return None
        tmpl = max(comps, key=len)
        if len(tmpl) < 2:
            return None
        tr0, tr1, tc0, tc1 = _bbox(tmpl)
        rel = [(a - tr0, b - tc0, int(tmpl_g[a, b])) for a, b in tmpl]
        th, tw = tr1 - tr0 + 1, tc1 - tc0 + 1
        ah, aw = (th // 2, tw // 2) if anchor == "center" else (0, 0)
        fcomps = _components(field_g, bg=bg, diag=True, by_color=False)
        markers = [c for c in fcomps if len(c) == 1]
        if not markers:
            return None
        out = g.copy()
        if clear_markers:
            for m in markers:
                mr, mc = m[0]
                out[mr + roff, mc + coff] = bg
        for m in markers:
            mr, mc = m[0]
            for (dr, dc, col) in rel:
                rr, cc = mr + roff - ah + dr, mc + coff - aw + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = col
        return out

    def renders(g, bg):
        h, w = g.shape
        sep_rows, sep_cols = panels(g)
        layouts = []
        if len(sep_cols) == 1 and not sep_rows:
            sc = sep_cols[0]
            layouts.append((g[:, :sc], g[:, sc + 1:], 0, sc + 1))    # tmpl=left, field=right
            layouts.append((g[:, sc + 1:], g[:, :sc], 0, 0))          # tmpl=right, field=left
        if len(sep_rows) == 1 and not sep_cols:
            sr = sep_rows[0]
            layouts.append((g[:sr, :], g[sr + 1:, :], sr + 1, 0))     # tmpl=top, field=bottom
            layouts.append((g[sr + 1:, :], g[:sr, :], 0, 0))          # tmpl=bottom, field=top
        for (tmpl_g, field_g, roff, coff) in layouts:
            for anchor in ("topleft", "center"):
                for clear in (False, True):
                    r = _stamp(g, tmpl_g, field_g, roff, coff, anchor, clear)
                    if r is not None:
                        yield r

    gi0, go0 = train[0]
    renders0 = list(renders(gi0, bg))
    for idx in range(len(renders0)):
        def fn(g, idx=idx, bg=bg):
            rs = list(renders(g, bg))
            return rs[idx] if idx < len(rs) else None
        if _verify(fn, train):
            return fn
    return None


def induce_attract_to_cluster(train, bg, seg):
    """RELATION: for each color, there is ONE cluster (multi-cell) and several stray single cells; each
    stray moves to become orthogonally adjacent to the cluster, forming a tight plus/blob. Approximated as:
    keep the cluster, drop the strays, and the strays reappear as the cluster's 4-neighborhood of that
    color. Verified exactly. (ae3edfdc family — attract scattered same-color cells to the magnet.)"""
    diag, by_color = seg
    if any(gi.shape != go.shape for gi, go in train):
        return None

    def apply(g, bg):
        h, w = g.shape
        out = np.full_like(g, bg)
        colors = [c for c in np.unique(g) if c != bg]
        for col in colors:
            mask = (g == col)
            comps = _components((mask).astype(int) * int(col), bg=0, diag=True, by_color=False)
            if not comps:
                continue
            # the magnet = the largest component of this color
            magnet = max(comps, key=len)
            if len(magnet) < 2:
                return None
            strays = [c for c in comps if c is not magnet]
            cr = sum(a for a, _ in magnet) / len(magnet)
            cc = sum(b for _, b in magnet) / len(magnet)
            # keep the magnet
            for a, b in magnet:
                out[a, b] = col
            # place each stray on the nearest empty 4-neighbor of the magnet toward it
            occ = set((a, b) for a, b in magnet)
            for s in strays:
                sa, sb = s[0]
                # candidate ring around magnet center
                best = None
                bestd = None
                for a, b in magnet:
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + di, b + dj
                        if 0 <= x < h and 0 <= y < w and (x, y) not in occ:
                            d = (x - sa) ** 2 + (y - sb) ** 2
                            if bestd is None or d < bestd:
                                bestd = d
                                best = (x, y)
                if best is None:
                    return None
                occ.add(best)
                out[best] = col
        return out

    def fn(g, bg=bg):
        return apply(g, bg)
    if _verify(fn, train):
        return fn
    return None


# ===========================================================================
# B) PANEL SELECTION / COMBINATION  (shape-changing allowed)
# ===========================================================================
def _split_into_panels(g, bg):
    """Split a grid into equal-size panels. Two modes:
      (1) gridline separators: full rows/cols of a single non-bg color used as dividers.
      (2) equal tiling: if the grid divides evenly into k>=2 equal blocks along one axis with no separators.
    Returns list of (panel_array, (r0,c0)) and the layout key, or [] if no clean split."""
    h, w = g.shape
    results = []
    # mode 1: separator lines (a constant full row/col, color != typical content). Find the separator color.
    def const_rows(g):
        return [r for r in range(h) if len(set(g[r, :].tolist())) == 1]
    def const_cols(g):
        return [c for c in range(w) if len(set(g[:, c].tolist())) == 1]
    crows = const_rows(g)
    ccols = const_cols(g)
    # separator color = the color that appears as full lines in BOTH or the dominant full-line color
    sep_color = None
    line_colors = Counter()
    for r in crows:
        line_colors[int(g[r, 0])] += 1
    for c in ccols:
        line_colors[int(g[0, c])] += 1
    if line_colors:
        sep_color = line_colors.most_common(1)[0][0]
    sep_rows = sorted([r for r in crows if int(g[r, 0]) == sep_color])
    sep_cols = sorted([c for c in ccols if int(g[0, c]) == sep_color])
    if sep_rows or sep_cols:
        # build the grid of panels between separators
        rseg = _segments(h, set(sep_rows))
        cseg = _segments(w, set(sep_cols))
        panels = []
        ok = True
        for (ra, rb) in rseg:
            for (ca, cb) in cseg:
                if rb < ra or cb < ca:
                    ok = False
                    break
                panels.append((g[ra:rb + 1, ca:cb + 1], (ra, ca)))
            if not ok:
                break
        if ok and len(panels) >= 2:
            # require all panels equal-shape for selection/combination
            shp = panels[0][0].shape
            if all(p.shape == shp for p, _ in panels) and shp[0] > 0 and shp[1] > 0:
                results.append(("sep", sep_color, panels, (len(rseg), len(cseg))))
    # mode 2: equal stacking with no separators (divide axis into k equal blocks)
    for axis in (0, 1):
        n = h if axis == 0 else w
        for k in range(2, 7):
            if n % k != 0:
                continue
            blk = n // k
            if blk < 1:
                continue
            panels = []
            for i in range(k):
                if axis == 0:
                    panels.append((g[i * blk:(i + 1) * blk, :], (i * blk, 0)))
                else:
                    panels.append((g[:, i * blk:(i + 1) * blk], (0, i * blk)))
            results.append(("tile%d_%d" % (axis, k), None, panels, (k if axis == 0 else 1, k if axis == 1 else 1)))
    return results


def _segments(n, seps):
    """Contiguous index ranges of [0,n) excluding separator indices; returns [(a,b),...] inclusive."""
    segs = []
    a = None
    for i in range(n):
        if i in seps:
            if a is not None:
                segs.append((a, i - 1))
                a = None
        else:
            if a is None:
                a = i
    if a is not None:
        segs.append((a, n - 1))
    return segs


def _panel_key(p):
    return p.tobytes() + bytes(p.shape)


def _panel_bg(p):
    v, ct = np.unique(p, return_counts=True)
    return int(v[ct.argmax()])


def _shape_key(p, b):
    """Color-agnostic non-bg footprint key of a panel (relative to its own bg)."""
    return ((p != b).astype(np.uint8).tobytes(), p.shape)


def _recolor_canon_key(p):
    """Pattern key invariant to a color RELABELING: first-occurrence order canonicalization."""
    flat = p.flatten().tolist()
    seen = {}
    out = []
    for v in flat:
        if v not in seen:
            seen[v] = len(seen)
        out.append(seen[v])
    return (bytes(out), p.shape)


def induce_select_panel(train, bg):
    """RELATION: split each input into equal panels; SELECT exactly one by an induced relation; output =
    that panel verbatim. Selection vocabulary (each fitted to be CONSTANT across train + exact-verified):
      uniqueness criteria — the lone panel that is UNIQUE under an equivalence (exact / non-bg shape /
        color-relabel-canonical), or conversely the MAJORITY panel under that equivalence;
      extremeness criteria — argmax/argmin of a panel feature (#non-bg cells [own bg], #distinct colors,
        #connected components).  Verified across all train pairs."""
    gi0, go0 = train[0]
    splits0 = _split_into_panels(gi0, bg)

    def count_nonbg(p):
        return int((p != _panel_bg(p)).sum())

    def ndistinct(p):
        return len(set(p.flatten().tolist()))

    def ncomp(p):
        return len(_components(p, bg=_panel_bg(p), diag=True, by_color=False))

    feat_funcs = {
        "max_nonbg": count_nonbg, "min_nonbg": count_nonbg,
        "max_ndist": ndistinct, "min_ndist": ndistinct,
        "max_ncomp": ncomp, "min_ncomp": ncomp,
    }
    equivs = {
        "exact": lambda p: _panel_key(p),
        "shape": lambda p: _shape_key(p, _panel_bg(p)),
        "relabel": lambda p: _recolor_canon_key(p),
    }

    def pick(panels, crit):
        arrs = [p for p, _ in panels]
        if crit[0] in ("uniq", "maj"):
            keyf = equivs[crit[1]]
            keys = [keyf(p) for p in arrs]
            ct = Counter(keys)
            if crit[0] == "uniq":
                lone = [p for p, k in zip(arrs, keys) if ct[k] == 1]
                return lone[0] if len(lone) == 1 else None
            # majority: the strict-most-common equivalence class, return the FIRST member
            key, n = ct.most_common(1)[0]
            if n >= 2 and list(ct.values()).count(n) == 1:
                for p, k in zip(arrs, keys):
                    if k == key:
                        return p
            return None
        # extremeness
        f = feat_funcs[crit[0]]
        vals = [f(p) for p in arrs]
        target = max(vals) if "max" in crit[0] else min(vals)
        if vals.count(target) != 1:
            return None
        return arrs[vals.index(target)]

    crits = [("uniq", e) for e in equivs] + [("maj", e) for e in equivs] + \
            [(f,) for f in feat_funcs]

    # The layout may differ across pairs (a 9x3 stack in one, a 3x12 row in another) yet the panel SHAPE
    # and the selection RELATION are constant. So we fit (panel_shape, crit): at apply time, pick whichever
    # split yields panels of that shape. panel_shape is anchored to the output shape (= one panel).
    pshape = go0.shape
    cand_specs = []
    for (kind, sep_color, panels, layout) in splits0:
        if panels[0][0].shape != pshape:
            continue
        for crit in crits:
            sel = pick(panels, crit)
            if sel is not None and _eq(sel, go0):
                cand_specs.append(crit)

    def make_fn(crit, pshape):
        def fn(g, crit=crit, pshape=pshape, bg=bg):
            for (k2, sc2, panels2, lay2) in _split_into_panels(g, bg):
                if panels2[0][0].shape != pshape:
                    continue
                sel = pick(panels2, crit)
                if sel is not None:
                    return sel
            return None
        return fn

    seen = set()
    for crit in cand_specs:
        if crit in seen:
            continue
        seen.add(crit)
        fn = make_fn(crit, pshape)
        if _verify(fn, train):
            return fn
    return None


def induce_combine_panels(train, bg):
    """RELATION: split into equal panels; OVERLAY them under a fitted boolean rule on the non-bg mask
    (AND / OR / XOR / DIFF) and paint the result with a fitted output color. Output shape = one panel.
    Handles the panel-overlay family (995c5fa3, 7b7f7511, cf98881b)."""
    gi0, go0 = train[0]
    splits0 = _split_into_panels(gi0, bg)

    def masks(panels):
        return [(p != bg) for p, _ in panels]

    def combine(panels, rule):
        ms = masks(panels)
        if not ms:
            return None
        acc = ms[0].copy()
        if rule == "and":
            for m in ms[1:]:
                acc &= m
        elif rule == "or":
            for m in ms[1:]:
                acc |= m
        elif rule == "xor":
            for m in ms[1:]:
                acc ^= m
        elif rule == "diff":  # in first but not any other
            other = np.zeros_like(acc)
            for m in ms[1:]:
                other |= m
            acc = acc & ~other
        elif rule == "nor":
            allm = np.zeros_like(acc)
            for m in ms:
                allm |= m
            acc = ~allm
        return acc

    # output color: read from pair0 where mask True
    def fit_color(panels, rule, go):
        acc = combine(panels, rule)
        if acc is None or acc.shape != go.shape:
            return None
        on = go[acc]
        off = go[~acc]
        if on.size == 0:
            return None
        on_cols = set(on.tolist())
        off_cols = set(off.tolist())
        if len(on_cols) != 1:
            return None
        oc = on_cols.pop()
        if len(off_cols) > 1:
            return None
        ob = off_cols.pop() if off_cols else bg
        if oc == ob:
            return None
        return oc, ob

    cand_specs = []
    for (kind, sc, panels, lay) in splits0:
        for rule in ("and", "or", "xor", "diff", "nor"):
            fc = fit_color(panels, rule, go0)
            if fc is not None:
                cand_specs.append((kind, rule, fc[0], fc[1]))

    def make_fn(kind, rule, oc, ob):
        def fn(g, kind=kind, rule=rule, oc=oc, ob=ob, bg=bg):
            splits = _split_into_panels(g, bg)
            for (k2, sc2, panels2, lay2) in splits:
                if k2 != kind:
                    continue
                acc = combine(panels2, rule)
                if acc is None:
                    return None
                out = np.full(acc.shape, ob, int)
                out[acc] = oc
                return out
            return None
        return fn

    for kind, rule, oc, ob in cand_specs:
        fn = make_fn(kind, rule, oc, ob)
        if _verify(fn, train):
            return fn
    return None


# ===========================================================================
# the induction menus
# ===========================================================================
MOVE_INDUCERS = [
    ("translate_const", induce_translate_const),
    ("gravity_blocked", induce_gravity_blocked),
    ("move_to_edge", induce_move_to_edge),
    ("move_objs_to_markers", induce_move_objs_to_markers),
    ("stamp_template_at_markers", induce_stamp_template_at_markers),
    ("attract_to_cluster", induce_attract_to_cluster),
]
PANEL_INDUCERS = [
    ("select_panel", induce_select_panel),
    ("combine_panels", induce_combine_panels),
]
# stamp_panel_template does its own paneling (separator); doesn't need a segmentation loop
SPECIAL_INDUCERS = [
    ("stamp_panel_template", induce_stamp_panel_template),
]


# ===========================================================================
# INVENTION ENTRYPOINT
# ===========================================================================
def _invent(train, test_inputs, budget):
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    bg = _bg_train(train)

    fitted = []
    seen_sig = set()
    t0 = time.time()
    time_budget = 12.0  # seconds, per task; well under harness budget but generous for these inducers

    def add(iname, fn):
        try:
            sig = (iname,) + tuple(fn(gi).tobytes() if fn(gi) is not None else b"" for gi, _ in train)
        except Exception:
            sig = None
        if sig is not None and sig in seen_sig:
            return
        if sig is not None:
            seen_sig.add(sig)
        fitted.append((iname, fn))
        _remember(iname)

    # MOVE inducers across segmentations
    for seg in _segmentations():
        if time.time() - t0 > time_budget:
            break
        for iname, inducer in MOVE_INDUCERS:
            try:
                fn = inducer(train, bg, seg)
            except Exception:
                fn = None
            if fn is not None:
                add(iname, fn)
        if len(fitted) >= 4:
            break

    # PANEL inducers (segmentation-agnostic; split is fitted internally)
    for iname, inducer in PANEL_INDUCERS:
        if time.time() - t0 > time_budget:
            break
        try:
            fn = inducer(train, bg)
        except Exception:
            fn = None
        if fn is not None:
            add(iname, fn)

    # SPECIAL (panel-template stamp)
    for iname, inducer in SPECIAL_INDUCERS:
        if time.time() - t0 > time_budget:
            break
        try:
            fn = inducer(train, bg, (True, False))
        except Exception:
            fn = None
        if fn is not None:
            add(iname, fn)

    attempts = []
    for gi in test_inputs:
        gi = np.asarray(gi, int)
        cand = []
        for _iname, fn in fitted:
            try:
                o = fn(gi)
            except Exception:
                o = None
            if o is not None and getattr(o, "ndim", None) == 2 and o.size > 0:
                o = np.asarray(o, int)
                if not any(_eq(o, c) for c in cand):
                    cand.append(o)
            if len(cand) >= 2:
                break
        attempts.append(cand[:2])
    return attempts


# ===========================================================================
# IN-RUN EXPERIENCE (verified-only): remember which inducers fired so later tasks try them first.
# ===========================================================================
_MEM = []
_MEM_SEEN = set()


def _remember(iname):
    if iname not in _MEM_SEEN:
        _MEM_SEEN.add(iname)
        _MEM.append(iname)


def reset_library():
    _MEM.clear()
    _MEM_SEEN.clear()
    if hasattr(BASE, "reset_library"):
        try:
            BASE.reset_library()
        except Exception:
            pass
    else:
        try:
            BASE._LIB.__init__()
        except Exception:
            pass


# ===========================================================================
# STANDARDIZED GATE WIRING
# ===========================================================================
def solve_ablated(train, test_inputs, budget):
    """EXACTLY gen2_base.solve — the standardized strong-retrieval ablation. INVENTED = solved - this."""
    return BASE.solve(train, test_inputs, budget)


def solve(train, test_inputs, budget):
    """gen2_base as attempt-1 backstop; movement/interaction invention fills the remaining attempt slot."""
    try:
        base_attempts = BASE.solve(train, test_inputs, budget)
    except Exception:
        base_attempts = []
    if not isinstance(base_attempts, list):
        base_attempts = []
    norm = []
    for k in range(len(test_inputs)):
        a = base_attempts[k] if k < len(base_attempts) else []
        if a is None:
            a = []
        norm.append([np.asarray(x, int) for x in a if x is not None][:2])

    try:
        inv = _invent(train, test_inputs, max(800, budget))
    except Exception:
        inv = [[] for _ in test_inputs]

    merged = []
    for k in range(len(test_inputs)):
        b = list(norm[k])
        iv = [o for o in (inv[k] if k < len(inv) else []) if o is not None]
        cand = []
        if b:
            cand.append(b[0])                      # attempt 1 = base backstop
        for o in iv:                               # attempt 2 = first invention not already present
            if not any(_eq(o, c) for c in cand):
                cand.append(o)
                break
        if len(cand) < 2:                          # backfill unused slot
            for o in (b[1:] + iv):
                if not any(_eq(o, c) for c in cand):
                    cand.append(o)
                if len(cand) >= 2:
                    break
        merged.append(cand[:2])
    return merged


# ===========================================================================
# import-time: tiny self-test exercising every inducer once (build < ~1s)
# ===========================================================================
def _selftest():
    g = np.zeros((10, 10), int)
    g[2, 2] = 3
    g[5, 5] = 3
    g[7, 1] = 4
    train = [(g, g)]
    for _n, ind in MOVE_INDUCERS:
        try:
            ind(train, 0, (True, False))
        except Exception:
            pass
    for _n, ind in PANEL_INDUCERS:
        try:
            ind(train, 0)
        except Exception:
            pass
    for _n, ind in SPECIAL_INDUCERS:
        try:
            ind(train, 0, (True, False))
        except Exception:
            pass
    # exercise panel split + invent path
    try:
        _split_into_panels(g, 0)
        _invent(train, [g], 2000)
    except Exception:
        pass


_selftest()
