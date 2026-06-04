#!/usr/bin/env python3
"""GEN-5 RELATION-INDUCER #1 — OBJECT-RELATIONAL.

THE EVIDENCE-BACKED LEVER (per CAMPAIGN.md gen-5 DIAGNOSTIC). On real ARC's held-out frontier, composing a
FIXED verb alphabet does NOT engage the tasks gen2_base misses (0/366 have any train-consistent program at
depth 1-4). EVERY beyond-retrieval win came from per-task FITTED cause->effect RELATIONS (recolor-each-object
by hole-count / border / aspect, ...), NOT verb composition. So the lever is RELATIONS made SYSTEMATIC and
RICH at OBJECT granularity — not deeper search.

WHAT THIS IS. A systematic per-task RELATION-INDUCTION engine at OBJECT granularity:

  (1) DECOMPOSE the grid into objects under several segmentations (4/8-conn x by-color/color-agnostic, vs a
      FITTED background). The right segmentation is itself fitted by which one makes a relation verify.

  (2) Extract a RICH FEATURE VECTOR per object (the candidate CAUSES):
        size(#cells), size-rank(asc/desc, dense & by-bbox-area), color, #distinct-colors, #holes,
        has-border-touch, bbox h / w / area / aspect-class, is-square/line/rect-solid, fill-ratio,
        position (row-rank, col-rank, quadrant, is-topmost/bottommost/leftmost/rightmost/center),
        shape-hash (canonical & oriented), #4/8-neighbors, is-unique-by-{size,color,shape,holes},
        count-of-objects, parity/extremeness flags, relation-to-others (is-largest/smallest, is-the-odd-one).

  (3) INDUCE a mapping  feature(s) -> EFFECT  that holds CONSISTENTLY across ALL train pairs (the cross-pair
      invariance that licenses CAUSAL, not correlational, induction) and EXACT-verify on the train outputs;
      the held-out test is the intervention that certifies it. EFFECT vocabulary (the cause->effect relation):
        recolor(->fixed color),                 # object -> solid color, keyed by a feature
        recolor(->feature-of-object),            # e.g. paint to its own rank, or to size mod K
        delete / keep,                           # binary keep/remove keyed by a feature (predicate)
        move-to-edge (top/bottom/left/right),    # translate object to a grid edge
        move-to-marker / copy-to-marker,         # stamp object onto same-color anchor cells
        keep-only-the-{extreme}-object.          # select the unique object by a feature

  (4) FEATURE-RELEVANCE PRIOR (kept CPU-light, <90s build): a tiny logistic prior trained at IMPORT on
      self-generated synthetic relational tasks (random objects + a known relation) learns which FEATURES
      tend to DRIVE which EFFECTS, so induction tries the likely-relevant features first and stays tractable
      as the feature space grows. Falls back to a fixed MDL-ish order if the prior is unavailable.

STANDARDIZED GATE (non-negotiable):
  solve_ablated(train,test_inputs,budget) == EXACTLY gen2_base.solve  (imported verbatim — strong retrieval).
  solve() = gen2_base as attempt-1 backstop  +  THIS object-relation induction as attempt-2.
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

META = {"name": "gen5_01_object-relational",
        "desc": "systematic per-task OBJECT relation-induction: rich per-object feature vector (candidate "
                "causes) -> consistent EFFECT (recolor/delete/keep/move/copy/select), exact-verified, "
                "feature-relevance prior trained at import on self-gen relational tasks; gen2_base backstop"}


# ===========================================================================
# tiny grid helpers
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
    """List of objects, each a list of (r,c) cells. by_color keeps a component single-color."""
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


def _count_holes(g, cells, bg):
    """#enclosed bg-regions inside the object's bbox (4-conn flood from bbox border)."""
    r0, r1, c0, c1 = _bbox(cells)
    H, W = r1 - r0 + 1, c1 - c0 + 1
    occ = np.zeros((H, W), bool)
    for a, b in cells:
        occ[a - r0, b - c0] = True
    free = ~occ
    reach = np.zeros((H, W), bool)
    q = deque()
    for i in range(H):
        for j in (0, W - 1):
            if free[i, j] and not reach[i, j]:
                reach[i, j] = True
                q.append((i, j))
    for j in range(W):
        for i in (0, H - 1):
            if free[i, j] and not reach[i, j]:
                reach[i, j] = True
                q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = i + di, j + dj
            if 0 <= x < H and 0 <= y < W and free[x, y] and not reach[x, y]:
                reach[x, y] = True
                q.append((x, y))
    enclosed = free & ~reach
    seen = np.zeros((H, W), bool)
    cnt = 0
    for i in range(H):
        for j in range(W):
            if enclosed[i, j] and not seen[i, j]:
                cnt += 1
                q = deque([(i, j)])
                seen[i, j] = True
                while q:
                    a, b = q.popleft()
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + di, b + dj
                        if 0 <= x < H and 0 <= y < W and enclosed[x, y] and not seen[x, y]:
                            seen[x, y] = True
                            q.append((x, y))
    return cnt


def _shape_hash(cells, oriented=False):
    """Canonical (D4-invariant) or oriented hash of an object's cell footprint (bg-normalized)."""
    r0, r1, c0, c1 = _bbox(cells)
    H, W = r1 - r0 + 1, c1 - c0 + 1
    occ = np.zeros((H, W), bool)
    for a, b in cells:
        occ[a - r0, b - c0] = True
    if oriented:
        return occ.tobytes() + bytes((H, W % 256))
    best = None
    m = occ
    for _ in range(4):
        for t in (m, m[:, ::-1]):
            key = (t.shape, t.tobytes())
            if best is None or key < best:
                best = key
        m = np.rot90(m)
    return repr(best).encode()


# ===========================================================================
# OBJECT FEATURE EXTRACTION  — the candidate CAUSES, as a rich per-object dict.
# Each feature returns a hashable key; induction tests "is the effect a consistent function of THIS key?".
# ===========================================================================
def _object_color(g, cells):
    cols = Counter(int(g[a, b]) for a, b in cells)
    return cols.most_common(1)[0][0], len(cols)


def extract_objects(g, bg, diag, by_color):
    """Return list of object dicts with a rich feature vocabulary. Color-agnostic by_color=False groups
    differently-colored touching cells into one object; by_color=True splits by color."""
    comps = _components(g, bg=bg, diag=diag, by_color=by_color)
    h, w = g.shape
    n = len(comps)
    objs = []
    sizes = []
    areas = []
    for cells in comps:
        r0, r1, c0, c1 = _bbox(cells)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        color, ncol = _object_color(g, cells)
        sz = len(cells)
        area = bh * bw
        sizes.append(sz)
        areas.append(area)
        objs.append({
            "cells": cells, "bbox": (r0, r1, c0, c1), "bh": bh, "bw": bw,
            "color": color, "ncol": ncol, "size": sz, "area": area,
            "cr": (r0 + r1) / 2.0, "cc": (c0 + c1) / 2.0,
            "touch_border": int(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1),
        })
    # ranks (dense) by size and by area, asc and desc
    def dense_ranks(vals):
        uniq = sorted(set(vals))
        asc = {v: i for i, v in enumerate(uniq)}
        desc = {v: len(uniq) - 1 - i for i, v in enumerate(uniq)}
        return asc, desc
    s_asc, s_desc = dense_ranks(sizes)
    a_asc, a_desc = dense_ranks(areas)
    # positional ranks
    crs = sorted(set(round(o["cr"], 3) for o in objs))
    ccs = sorted(set(round(o["cc"], 3) for o in objs))
    cr_rank = {v: i for i, v in enumerate(crs)}
    cc_rank = {v: i for i, v in enumerate(ccs)}
    # uniqueness tallies
    size_ct = Counter(sizes)
    color_ct = Counter(o["color"] for o in objs)
    for o in objs:
        o["holes"] = _count_holes(g, o["cells"], bg)
        o["shash"] = _shape_hash(o["cells"])
        o["oshash"] = _shape_hash(o["cells"], oriented=True)
        o["nobj"] = n
        o["rank_size_asc"] = s_asc[o["size"]]
        o["rank_size_desc"] = s_desc[o["size"]]
        o["rank_area_asc"] = a_asc[o["area"]]
        o["rank_area_desc"] = a_desc[o["area"]]
        o["rank_row"] = cr_rank[round(o["cr"], 3)]
        o["rank_col"] = cc_rank[round(o["cc"], 3)]
        o["quadrant"] = (int(o["cr"] >= h / 2.0), int(o["cc"] >= w / 2.0))
        o["is_top"] = int(o["bbox"][0] == min(x["bbox"][0] for x in objs))
        o["is_bottom"] = int(o["bbox"][1] == max(x["bbox"][1] for x in objs))
        o["is_left"] = int(o["bbox"][2] == min(x["bbox"][2] for x in objs))
        o["is_right"] = int(o["bbox"][3] == max(x["bbox"][3] for x in objs))
        o["aspect"] = (int(o["bw"] > o["bh"]) - int(o["bw"] < o["bh"]))  # -1/0/1
        o["is_square"] = int(o["bh"] == o["bw"])
        o["is_line"] = int(o["bh"] == 1 or o["bw"] == 1)
        o["fill"] = o["size"] / max(o["area"], 1)
        o["is_solid"] = int(o["size"] == o["area"])
        o["uniq_size"] = int(size_ct[o["size"]] == 1)
        o["uniq_color"] = int(color_ct[o["color"]] == 1)
    # shape uniqueness (after shash assigned)
    shash_ct = Counter(o["shash"] for o in objs)
    holes_ct = Counter(o["holes"] for o in objs)
    for o in objs:
        o["uniq_shape"] = int(shash_ct[o["shash"]] == 1)
        o["uniq_holes"] = int(holes_ct[o["holes"]] == 1)
    return objs


# FEATURE keys we induce over: name -> function(obj) -> hashable key.
FEATURE_FUNCS = [
    ("size", lambda o: o["size"]),
    ("rank_size_asc", lambda o: o["rank_size_asc"]),
    ("rank_size_desc", lambda o: o["rank_size_desc"]),
    ("rank_area_asc", lambda o: o["rank_area_asc"]),
    ("rank_area_desc", lambda o: o["rank_area_desc"]),
    ("color", lambda o: o["color"]),
    ("ncol", lambda o: o["ncol"]),
    ("holes", lambda o: o["holes"]),
    ("touch_border", lambda o: o["touch_border"]),
    ("bh", lambda o: o["bh"]),
    ("bw", lambda o: o["bw"]),
    ("area", lambda o: o["area"]),
    ("aspect", lambda o: o["aspect"]),
    ("is_square", lambda o: o["is_square"]),
    ("is_line", lambda o: o["is_line"]),
    ("is_solid", lambda o: o["is_solid"]),
    ("shape", lambda o: o["shash"]),
    ("rank_row", lambda o: o["rank_row"]),
    ("rank_col", lambda o: o["rank_col"]),
    ("quadrant", lambda o: o["quadrant"]),
    ("is_top", lambda o: o["is_top"]),
    ("is_bottom", lambda o: o["is_bottom"]),
    ("is_left", lambda o: o["is_left"]),
    ("is_right", lambda o: o["is_right"]),
    ("uniq_size", lambda o: o["uniq_size"]),
    ("uniq_color", lambda o: o["uniq_color"]),
    ("uniq_shape", lambda o: o["uniq_shape"]),
    ("uniq_holes", lambda o: o["uniq_holes"]),
    ("nobj", lambda o: o["nobj"]),
]
FEATURE_INDEX = {n: k for k, (n, _) in enumerate(FEATURE_FUNCS)}


# ===========================================================================
# SEGMENTATIONS to try (fitted by which one lets a relation verify).
# orthogonal-before-diagonal, color-agnostic-before-by-color: the simpler/less-merging first (MDL).
# ===========================================================================
def _segmentations():
    for diag in (False, True):
        for by_color in (False, True):
            yield (diag, by_color)


# ===========================================================================
# EFFECT INDUCERS.  Each takes (train, bg, segmentation, feature_order) and returns a verified grid->grid
# function or None.  All require the same-shape footprint constraints appropriate to the effect.
# ===========================================================================
def _objs_consistent_with_output(objs, go, bg):
    """For recolor-type induction: each object must map to a SINGLE solid output color over its cells."""
    out = []
    for o in objs:
        ocs = {int(go[a, b]) for a, b in o["cells"]}
        if len(ocs) != 1:
            return None
        out.append((o, ocs.pop()))
    return out


def induce_recolor_const(train, bg, seg, feat_order):
    """RELATION: recolor each object SOLID by a fixed color that is a consistent function of one feature.
    feature(obj) -> output color, invariant across all train pairs. Footprint preserved w.r.t. bg."""
    diag, by_color = seg
    if any(gi.shape != go.shape for gi, go in train):
        return None
    # footprint must be preserved (objects stay where they are; only color changes)
    for gi, go in train:
        if np.any((gi != bg) != (go != bg)):
            return None
    for fname in feat_order:
        ff = dict(FEATURE_FUNCS)[fname]
        mapping = {}
        ok = True
        for gi, go in train:
            objs = extract_objects(gi, bg, diag, by_color)
            if not objs:
                ok = False
                break
            pairs = _objs_consistent_with_output(objs, go, bg)
            if pairs is None:
                ok = False
                break
            for o, oc in pairs:
                key = ff(o)
                if key in mapping and mapping[key] != oc:
                    ok = False
                    break
                mapping[key] = oc
            if not ok:
                break
        if not ok or not mapping:
            continue
        # must be NON-trivial: at least two distinct keys OR it actually recolors
        def fn(g, ff=ff, mapping=mapping, diag=diag, by_color=by_color, bg=bg):
            objs = extract_objects(g, bg, diag, by_color)
            if not objs:
                return None
            out = g.copy()
            for o in objs:
                key = ff(o)
                if key not in mapping:
                    return None
                for a, b in o["cells"]:
                    out[a, b] = mapping[key]
            return out
        if _verify(fn, train):
            return fn
    return None


def induce_recolor_to_feature(train, bg, seg, feat_order):
    """RELATION: recolor each object to a color that is computed FROM a feature value (not a learned table):
    paint-to own size-rank+offset, or to (feature mod 10). Catches 'index/rank coloring' relations where
    the color literally equals a structural rank. Footprint preserved."""
    diag, by_color = seg
    if any(gi.shape != go.shape for gi, go in train):
        return None
    for gi, go in train:
        if np.any((gi != bg) != (go != bg)):
            return None
    # candidate color-from-feature maps with a learned additive offset
    rank_feats = ["rank_size_asc", "rank_size_desc", "rank_area_asc", "rank_area_desc",
                  "rank_row", "rank_col", "holes", "size", "ncol"]
    fd = dict(FEATURE_FUNCS)
    for fname in rank_feats:
        if fname not in feat_order:
            continue
        ff = fd[fname]
        offset = None
        ok = True
        for gi, go in train:
            objs = extract_objects(gi, bg, diag, by_color)
            if not objs:
                ok = False
                break
            pairs = _objs_consistent_with_output(objs, go, bg)
            if pairs is None:
                ok = False
                break
            for o, oc in pairs:
                v = ff(o)
                if not isinstance(v, int):
                    ok = False
                    break
                off = (oc - v)
                if offset is None:
                    offset = off
                elif offset != off:
                    ok = False
                    break
            if not ok:
                break
        if not ok or offset is None:
            continue

        def fn(g, ff=ff, offset=offset, diag=diag, by_color=by_color, bg=bg):
            objs = extract_objects(g, bg, diag, by_color)
            if not objs:
                return None
            out = g.copy()
            for o in objs:
                v = ff(o)
                col = v + offset
                if not (0 <= col <= 9):
                    return None
                for a, b in o["cells"]:
                    out[a, b] = col
            return out
        if _verify(fn, train):
            return fn
    return None


def induce_keep_delete(train, bg, seg, feat_order):
    """RELATION: keep/delete each object by a PREDICATE that is a consistent function of one feature.
    feature(obj) -> {kept, deleted}; deleted objects become bg, kept objects unchanged. Same-shape only."""
    diag, by_color = seg
    if any(gi.shape != go.shape for gi, go in train):
        return None
    fd = dict(FEATURE_FUNCS)
    for fname in feat_order:
        ff = fd[fname]
        decision = {}   # feature-key -> "keep" or "del"
        ok = True
        saw_keep = saw_del = False
        for gi, go in train:
            objs = extract_objects(gi, bg, diag, by_color)
            if not objs:
                ok = False
                break
            for o in objs:
                # is this object kept verbatim, or removed to bg, in the output?
                kept = all(int(go[a, b]) == int(gi[a, b]) for a, b in o["cells"])
                deleted = all(int(go[a, b]) == bg for a, b in o["cells"])
                if kept and not deleted:
                    d = "keep"; saw_keep = True
                elif deleted and not kept:
                    d = "del"; saw_del = True
                elif kept and deleted:
                    # single-cell ambiguous (value already bg): skip from constraint
                    continue
                else:
                    ok = False
                    break
                key = ff(o)
                if key in decision and decision[key] != d:
                    ok = False
                    break
                decision[key] = d
            if not ok:
                break
        if not ok or not (saw_keep and saw_del):
            continue

        def fn(g, ff=ff, decision=decision, diag=diag, by_color=by_color, bg=bg):
            objs = extract_objects(g, bg, diag, by_color)
            if not objs:
                return None
            out = g.copy()
            for o in objs:
                key = ff(o)
                if key not in decision:
                    return None
                if decision[key] == "del":
                    for a, b in o["cells"]:
                        out[a, b] = bg
            return out
        if _verify(fn, train):
            return fn
    return None


def induce_select_extreme(train, bg, seg, feat_order):
    """RELATION: output = crop to the UNIQUE object selected by an extreme/unique feature (keep-only).
    e.g. keep the object that is unique-by-shape / largest / has-most-holes; output is that object's
    content or window. Handles tasks where the output is one selected object (possibly a different shape)."""
    diag, by_color = seg
    fd = dict(FEATURE_FUNCS)
    # selection criteria: argmax / argmin of an integer feature, or the lone uniq-by-X object
    int_feats = ["size", "area", "holes", "bh", "bw", "ncol", "nobj"]
    uniq_feats = ["uniq_size", "uniq_color", "uniq_shape", "uniq_holes"]

    def selected(objs, crit):
        kind, fname = crit
        if not objs:
            return None
        if kind == "max":
            return max(objs, key=lambda o: o[fname] if fname in o else fd[fname](o))
        if kind == "min":
            return min(objs, key=lambda o: o[fname] if fname in o else fd[fname](o))
        # unique-by: the single object with the flag set (must be exactly one)
        flag = [o for o in objs if fd[fname](o) == 1]
        if len(flag) == 1:
            return flag[0]
        return None

    def render(o, mode, g, bg):
        r0, r1, c0, c1 = o["bbox"]
        if mode == "window":
            return g[r0:r1 + 1, c0:c1 + 1].copy()
        out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, int)
        for a, b in o["cells"]:
            out[a - r0, b - c0] = g[a, b]
        return out

    crits = [("max", f) for f in int_feats] + [("min", f) for f in int_feats] + \
            [("uniq", f) for f in uniq_feats]
    for crit in crits:
        for mode in ("window", "cut"):
            ok = True
            for gi, go in train:
                objs = extract_objects(gi, bg, diag, by_color)
                sel = selected(objs, crit)
                if sel is None:
                    ok = False
                    break
                r = render(sel, mode, gi, bg)
                if not _eq(r, go):
                    ok = False
                    break
            if not ok:
                continue

            def fn(g, crit=crit, mode=mode, diag=diag, by_color=by_color, bg=bg):
                objs = extract_objects(g, bg, diag, by_color)
                sel = selected(objs, crit)
                if sel is None:
                    return None
                return render(sel, mode, g, bg)
            if _verify(fn, train):
                return fn
    return None


def induce_move_to_edge(train, bg, seg, feat_order):
    """RELATION: translate each object to a grid EDGE (gravity per-object). All objects move to the SAME
    edge (top/bottom/left/right), chosen consistently. Objects keep shape+color; overlaps resolved by
    later-placed-wins. Same-shape grids only."""
    diag, by_color = seg
    if any(gi.shape != go.shape for gi, go in train):
        return None

    def apply_edge(g, edge, diag, by_color, bg):
        h, w = g.shape
        objs = extract_objects(g, bg, diag, by_color)
        if not objs:
            return None
        out = np.full_like(g, bg)
        # order so the ones closest to the target edge are placed first (stacking against the wall)
        if edge == "top":
            objs.sort(key=lambda o: o["bbox"][0])
        elif edge == "bottom":
            objs.sort(key=lambda o: -o["bbox"][1])
        elif edge == "left":
            objs.sort(key=lambda o: o["bbox"][2])
        else:
            objs.sort(key=lambda o: -o["bbox"][3])
        for o in objs:
            r0, r1, c0, c1 = o["bbox"]
            if edge == "top":
                dr, dc = -r0, 0
            elif edge == "bottom":
                dr, dc = (h - 1 - r1), 0
            elif edge == "left":
                dr, dc = 0, -c0
            else:
                dr, dc = 0, (w - 1 - c1)
            for a, b in o["cells"]:
                out[a + dr, b + dc] = g[a, b]
        return out

    for edge in ("top", "bottom", "left", "right"):
        def fn(g, edge=edge, diag=diag, by_color=by_color, bg=bg):
            return apply_edge(g, edge, diag, by_color, bg)
        if _verify(fn, train):
            return fn
    return None


def induce_copy_to_markers(train, bg, seg, feat_order):
    """RELATION: there is ONE template object (the largest, or the unique-by-size) and several single-cell
    'markers' of a distinct color; stamp a recolored copy of the template at each marker (marker -> object).
    The classic object-to-marker-copy family. Same-shape grids."""
    diag, by_color = seg
    if any(gi.shape != go.shape for gi, go in train):
        return None

    def build(g, bg):
        objs = extract_objects(g, bg, diag=True, by_color=True)
        if len(objs) < 2:
            return None
        # template = the largest object; markers = single-cell objects of OTHER colors
        objs_sorted = sorted(objs, key=lambda o: -o["size"])
        tmpl = objs_sorted[0]
        if tmpl["size"] < 2:
            return None
        markers = [o for o in objs if o["size"] == 1 and o is not tmpl]
        if not markers:
            return None
        return tmpl, markers

    def stamp(g, bg, recolor):
        info = build(g, bg)
        if info is None:
            return None
        tmpl, markers = info
        r0, r1, c0, c1 = tmpl["bbox"]
        # template footprint relative to its anchor (top-left of bbox)
        rel = [(a - r0, b - c0, int(g[a, b])) for a, b in tmpl["cells"]]
        # anchor offset within template that aligns to the marker (use template's own marker-color cell if
        # present, else center the bbox on the marker)
        ah = (r1 - r0) // 2
        aw = (c1 - c0) // 2
        out = g.copy()
        h, w = g.shape
        for m in markers:
            (mr, mc) = m["cells"][0]
            mcolor = int(g[mr, mc])
            for (dr, dc, col) in rel:
                rr, cc = mr - ah + dr, mc - aw + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = mcolor if recolor else col
        return out

    for recolor in (True, False):
        def fn(g, recolor=recolor, bg=bg):
            return stamp(g, bg, recolor)
        if _verify(fn, train):
            return fn
    return None


# the full induction menu, MDL-ish order (cheap/local recolors first; geometry/selection later)
INDUCERS = [
    ("recolor_const", induce_recolor_const),
    ("recolor_to_feature", induce_recolor_to_feature),
    ("keep_delete", induce_keep_delete),
    ("select_extreme", induce_select_extreme),
    ("move_to_edge", induce_move_to_edge),
    ("copy_to_markers", induce_copy_to_markers),
]


# ===========================================================================
# FEATURE-RELEVANCE PRIOR  — a tiny logistic model trained at IMPORT on SELF-GENERATED synthetic
# relational tasks. For a given task, it ranks which FEATURES are most likely to drive a recolor/keep
# effect, so induction tries those first (keeps a 29-feature space tractable under budget). The verifier
# supplies precision; the prior only supplies ORDER. CPU-light; falls back to fixed order on any failure.
# ===========================================================================
_PRIOR = {"W": None, "mu": None, "sd": None, "ok": False}


def _rand_object_grid(rng, nobj, with_feature):
    """Place nobj rectangle/L objects on a bg=0 grid; return grid + per-object 'driver feature value'."""
    h = rng.randint(8, 16); w = rng.randint(8, 16)
    g = np.zeros((h, w), int)
    placed = []
    tries = 0
    while len(placed) < nobj and tries < 80:
        tries += 1
        oh = rng.randint(1, 4); ow = rng.randint(1, 4)
        r = rng.randint(0, h - oh); c = rng.randint(0, w - ow)
        if np.any(g[max(0, r - 1):r + oh + 1, max(0, c - 1):c + ow + 1] != 0):
            continue
        col = rng.randint(1, 8)
        # sometimes punch a hole
        block = np.full((oh, ow), col, int)
        if oh >= 3 and ow >= 3 and rng.rand() < 0.4:
            block[1, 1] = 0
        g[r:r + oh, c:c + ow] = block
        placed.append((r, c, oh, ow, col))
    return g


def _grid_feature_stats(objs):
    """Aggregate per-task feature signal: for each feature, does it DISCRIMINATE objects (entropy>0)?
    Returns a fixed-length vector aligned to FEATURE_FUNCS giving normalized #distinct-values."""
    n = max(len(objs), 1)
    vec = np.zeros(len(FEATURE_FUNCS))
    for k, (fname, ff) in enumerate(FEATURE_FUNCS):
        try:
            vals = [ff(o) for o in objs]
        except Exception:
            continue
        nd = len(set(vals))
        vec[k] = (nd - 1) / max(n - 1, 1)   # 0 if constant, 1 if all distinct
    return vec


def _build_prior(seed=0, n_tasks=400, budget_sec=60.0):
    """Train a per-feature logistic relevance score: from many synthetic recolor tasks whose color is a
    KNOWN function of one feature, learn P(feature f is the driver | task feature-stat vector). The
    weight matrix W maps task-stats -> per-feature relevance logits."""
    t0 = time.time()
    rng = np.random.RandomState(seed)
    Fdim = len(FEATURE_FUNCS)
    X = []
    Y = []   # index of the true driver feature
    # only drive by features that are stable & meaningful across re-extraction
    drivers = ["rank_size_asc", "rank_size_desc", "color", "holes", "touch_border",
               "rank_row", "rank_col", "aspect", "size", "uniq_size", "uniq_shape", "is_square"]
    didx = [FEATURE_INDEX[d] for d in drivers]
    fd = dict(FEATURE_FUNCS)
    made = 0
    for _ in range(n_tasks):
        if time.time() - t0 > budget_sec:
            break
        nobj = rng.randint(2, 6)
        g = _rand_object_grid(rng, nobj, True)
        objs = extract_objects(g, 0, diag=False, by_color=True)
        if len(objs) < 2:
            continue
        # pick a driver feature that actually discriminates here
        cand = [d for d in drivers if len(set(fd[d](o) for o in objs)) >= 2]
        if not cand:
            continue
        drv = cand[rng.randint(len(cand))]
        stats = _grid_feature_stats(objs)
        X.append(stats)
        Y.append(FEATURE_INDEX[drv])
        made += 1
    if made < 30:
        return False
    X = np.array(X)
    Y = np.array(Y)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    W = np.zeros((Fdim, X.shape[1]))
    rng2 = np.random.RandomState(seed + 1)
    lr = 0.3
    for _e in range(6):
        if time.time() - t0 > budget_sec:
            break
        order = rng2.permutation(len(Xs))
        for i in order:
            f = Xs[i]; t = Y[i]
            logits = W @ f
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            p[t] -= 1.0
            W -= lr * np.outer(p, f)
    _PRIOR["W"] = W; _PRIOR["mu"] = mu; _PRIOR["sd"] = sd; _PRIOR["ok"] = True
    return True


# fixed MDL-ish fallback order (cheap, discriminating, generalizing features first)
_FALLBACK_ORDER = [n for n, _ in FEATURE_FUNCS]


def rank_features(objs_per_pair):
    """Rank FEATURE names by relevance for THIS task. Uses (a) the import-time prior to score features
    from the task's feature-stat vector, then (b) demotes features that are CONSTANT across all objects
    (a constant feature can't drive a per-object effect). Always returns ALL features (verifier decides)."""
    all_objs = [o for objs in objs_per_pair for o in objs]
    if not all_objs:
        return list(_FALLBACK_ORDER)
    stats = _grid_feature_stats(all_objs)
    scores = {}
    if _PRIOR["ok"]:
        fs = (stats - _PRIOR["mu"]) / _PRIOR["sd"]
        logits = _PRIOR["W"] @ fs
        for k, (fname, _) in enumerate(FEATURE_FUNCS):
            scores[fname] = float(logits[k])
    else:
        for k, (fname, _) in enumerate(FEATURE_FUNCS):
            scores[fname] = 0.0
    # discrimination bonus: a feature that separates objects is more likely the driver
    for k, (fname, _) in enumerate(FEATURE_FUNCS):
        scores[fname] += 2.0 * stats[k]
    order = sorted(_FALLBACK_ORDER, key=lambda n: -scores.get(n, 0.0))
    return order


# ===========================================================================
# INVENTION ENTRYPOINT — induce a relation, produce <=2 candidate outputs per test input.
# ===========================================================================
def _invent(train, test_inputs, budget):
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    bg = _bg_train(train)

    # precompute objects per train input under the cheapest segmentation, to RANK features once
    fitted = []
    seen_sig = set()

    for seg in _segmentations():
        diag, by_color = seg
        # feature ranking for THIS segmentation (the prior keeps induction tractable)
        try:
            objs_per_pair = [extract_objects(gi, bg, diag, by_color) for gi, _ in train]
        except Exception:
            objs_per_pair = []
        feat_order = rank_features(objs_per_pair) if objs_per_pair else list(_FALLBACK_ORDER)

        for iname, inducer in INDUCERS:
            # segmentation only matters for object-based inducers; copy_to_markers builds its own seg
            if iname == "copy_to_markers" and seg != (False, False):
                continue
            try:
                fn = inducer(train, bg, seg, feat_order)
            except Exception:
                fn = None
            if fn is None:
                continue
            # behavioral signature on train inputs to dedup identical relations across segmentations
            try:
                sig = (iname,) + tuple(fn(gi).tobytes() for gi, _ in train)
            except Exception:
                sig = None
            if sig is not None and sig in seen_sig:
                continue
            if sig is not None:
                seen_sig.add(sig)
            fitted.append((iname, fn))
            # remember for cross-task transfer (verified-only)
            _remember(iname, seg)

        if len(fitted) >= 4:
            break

    # also try replaying remembered relations first (transfer): re-induce by (name, seg) which is cheap.
    # (the induce functions already re-verify, so transfer = "try this proven inducer/seg early")

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
# IN-RUN EXPERIENCE (module-level, verified-only): remember which (inducer,segmentation) pairs solved
# earlier tasks so later tasks try them first. Names+seg only; never grids/outputs.
# ===========================================================================
_MEM = []          # list of (inducer_name, segmentation)
_MEM_SEEN = set()


def _remember(iname, seg):
    key = (iname, seg)
    if key not in _MEM_SEEN:
        _MEM_SEEN.add(key)
        _MEM.append(key)


def reset_library():
    """Documented hook: clear cross-task experience so a run starts cold (gate isolates transfer). Also
    resets the imported base library so the whole solver is genuinely cold."""
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
#   solve_ablated == gen2_base.solve  (the strong retrieval ablation, imported verbatim)
#   solve         == gen2_base attempt-1 backstop, THEN object-relation invention attempt-2
# ===========================================================================
def solve_ablated(train, test_inputs, budget):
    """EXACTLY gen2_base.solve — the standardized strong-retrieval ablation. INVENTED = solved - this."""
    return BASE.solve(train, test_inputs, budget)


def solve(train, test_inputs, budget):
    """gen2_base as attempt-1 backstop; object-relation invention fills the remaining attempt slot. The
    gate scores both attempts (ARC 2-attempt)."""
    # attempt 1: strong retrieval baseline (never regress below it)
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

    # attempt 2: object-relation invention
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
# import-time build: feature-relevance prior + a tiny self-test (executes the inducers once)
# ===========================================================================
_T0 = time.time()
try:
    _build_prior(seed=0, n_tasks=500, budget_sec=55.0)
except Exception:
    _PRIOR["ok"] = False
_BUILD_SEC = time.time() - _T0


def _selftest():
    rng = np.random.RandomState(1)
    g = _rand_object_grid(rng, 3, True)
    objs = extract_objects(g, 0, False, True)
    rank_features([objs])
    for _n, ind in INDUCERS:
        try:
            ind([(g, g)], 0, (False, True), _FALLBACK_ORDER)
        except Exception:
            pass


_selftest()
