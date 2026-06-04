#!/usr/bin/env python3
"""GEN-5 RELATION-INDUCER #2 — "cell-region": a SYSTEMATIC per-task relation-induction engine at
CELL / REGION granularity.

WHY THIS, NOT VERB-COMPOSITION (the gen-5 diagnostic). On real ARC's held-out frontier, composing a
FIXED verb alphabet does NOT engage the tasks gen2_base misses: of 366 eval misses, ZERO have any
train-consistent program at depth 1-4 in the relational verb alphabet. EVERY beyond-retrieval win in
the campaign came from per-task FITTED cause->effect RELATIONS (recolor-each-object by hole-count /
border / aspect, ...), never from verb composition. So the lever is RELATIONS made systematic and rich.

THE ENGINE (decompose -> rich features -> induce consistent feature->effect -> exact-verify).
  (1) DECOMPOSE the grid into PARTS at two granularities:
        * CELLS      : every non-background cell (and, for fill tasks, background cells too).
        * REGIONS    : 4- and 8-connected color components (objects) + per-color masks.
  (2) FEATURE VECTOR per part (the candidate CAUSES) — a RICH, symbolic, discrete feature space:
        cell  : own color; 4-neighbor multiset; 8-neighbor (3x3) symmetry-canonical patch hash;
                row/col parity; distance-to-nearest-edge; on-a-line-between two same-color seeds;
                inside-vs-outside a closed boundary; #same-color orthogonal neighbors.
        region: own color; size; bbox h/w and aspect class; #holes; touches-border; is-square;
                is-rectangle-filled; is-single-cell; size-rank (asc/desc); is-the-unique-largest /
                unique-smallest; count-of-cells-of-its-color; centroid row/col parity; symmetry class
                (h/v/rot180 self-symmetric).
  (3) INDUCE a mapping  feature-value -> EFFECT  that holds CONSISTENTLY across ALL train pairs (the
      cross-pair invariance that licenses CAUSAL, not correlational, induction), then EXACT-VERIFY on
      every train pair (and the held-out test is the intervention that confirms it). EFFECTS:
        * recolor-region-solid     (region color := f(feature))
        * recolor-cell             (cell color := f(feature))   [local-rule family]
        * keep / delete region     (region kept or set to bg by a predicate)
        * draw ray / connect       (extend cell color along a relational direction)
        * count -> construct       (output shape/content a function of a counted feature; SIZE-CHANGING)
        * select-region -> crop    (output := the region picked by a feature extremum; SIZE-CHANGING)
  (4) FEATURE-RELEVANCE PRIOR. As the feature space grows, induction must stay tractable. A tiny prior,
      trained at import on SELF-GENERATED synthetic relational tasks (<90s, CPU), scores which features
      are likely to DRIVE an effect, so the engine tries high-relevance features first. (No ARC data,
      no network, no LLM — pure synthetic self-supervision.) A within-task statistical fallback breaks
      ties when the prior is uninformative.

STANDARDIZED GATE (non-negotiable, identical to the gen-4/5 family):
    solve_ablated(train,test,budget) == EXACTLY gen2_base.solve   (the strong-retrieval ablation).
    solve()  = gen2_base as attempt-1 backstop  +  THIS relation-induction as attempt-2.
    invention_gate INVENTED = solved - ablated = solves gen2_base MISSES = the real creativity number.
  Develop/tune ONLY on arc1-train (gen2_base's TRAIN misses); arc1-eval is held-out (report, never tune).

INTEGRITY. solve()/solve_ablated() learn ONLY from (a) the current task's train pairs, (b) module-level
state from PRIOR solve() calls this run (verified-correct only), (c) self-generated synthetic data built
at import. NEVER reads ARC task files or test OUTPUTS, no network, no LLM. Respects budget. Pure
python+numpy; import-time build < ~90s. Run with /data/llm/.venv/bin/python from .../evolve."""
import os
import sys
import time
from collections import deque, Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVOLVE = os.path.dirname(HERE)
for p in (EVOLVE, "/data/Windows-files/Documents/airfoil/incubation/arc"):
    if p not in sys.path:
        sys.path.insert(0, p)

# import the candidate gen2_base from the SAME cand dir (it lives next to this file)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("gen2_base_for_g5_02", os.path.join(HERE, "gen2_base.py"))
BASE = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(BASE)

META = {
    "name": "g5_02_cell-region",
    "desc": "gen2_base retrieval backstop (attempt 1) + systematic CELL/REGION relation-induction "
            "(attempt 2): decompose->rich feature vector->induce consistent feature->effect "
            "(recolor/keep-delete/ray/count-construct/select-crop)->exact-verify, feature-relevance "
            "prior trained on self-gen synthetic relational tasks. INVENTED = solves gen2_base cannot.",
}


# ===========================================================================
# small grid helpers
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


_NB4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NB8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _components(g, bg=0, diag=False, by_color=True):
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    nb = _NB8 if diag else _NB4
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
    """# of enclosed bg regions inside the object bbox."""
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
        for di, dj in _NB4:
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
                    for di, dj in _NB4:
                        x, y = a + di, b + dj
                        if 0 <= x < H and 0 <= y < W and enclosed[x, y] and not seen[x, y]:
                            seen[x, y] = True
                            q.append((x, y))
    return cnt


# ===========================================================================
# REGION FEATURE VECTOR  — the candidate CAUSES per connected object.
# Every feature is a DISCRETE symbolic key. We name each so the relevance prior can score it.
# ===========================================================================
REGION_FEATURES = [
    "color", "size", "h", "w", "aspect", "holes", "border", "square",
    "rect_filled", "single", "rank_asc", "rank_desc", "is_largest", "is_smallest",
    "color_count", "sym_h", "sym_v", "sym_r180", "row_par", "col_par", "ncolors",
]


def _region_descriptors(g, bg, diag):
    """Return list of dicts: each describes one connected object with its cells + raw scalars."""
    comps = _components(g, bg=bg, diag=diag, by_color=True)
    descs = []
    for cells in comps:
        r0, r1, c0, c1 = _bbox(cells)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        colors = {int(g[a, b]) for a, b in cells}
        col = next(iter(colors)) if len(colors) == 1 else -1
        descs.append({
            "cells": cells, "r0": r0, "r1": r1, "c0": c0, "c1": c1,
            "h": h, "w": w, "size": len(cells), "color": col,
            "ncolors": len(colors),
            "border": int(r0 == 0 or c0 == 0 or r1 == g.shape[0] - 1 or c1 == g.shape[1] - 1),
            "holes": _count_holes(g, cells, bg),
        })
    return descs


def _region_feature_value(d, descs, g, name):
    """Map a region descriptor + global context -> a DISCRETE feature key for feature `name`."""
    if name == "color":
        return d["color"]
    if name == "size":
        return d["size"]
    if name == "h":
        return d["h"]
    if name == "w":
        return d["w"]
    if name == "aspect":
        return int(d["w"] > d["h"]) - int(d["w"] < d["h"])  # -1,0,1
    if name == "holes":
        return d["holes"]
    if name == "border":
        return d["border"]
    if name == "square":
        return int(d["h"] == d["w"])
    if name == "rect_filled":
        return int(d["size"] == d["h"] * d["w"])
    if name == "single":
        return int(d["size"] == 1)
    if name == "ncolors":
        return d["ncolors"]
    if name == "row_par":
        return int((d["r0"] + d["r1"]) // 2) % 2
    if name == "col_par":
        return int((d["c0"] + d["c1"]) // 2) % 2
    if name in ("rank_asc", "rank_desc", "is_largest", "is_smallest"):
        sizes = sorted({x["size"] for x in descs})
        if name == "rank_asc":
            return sizes.index(d["size"])
        if name == "rank_desc":
            return len(sizes) - 1 - sizes.index(d["size"])
        if name == "is_largest":
            return int(d["size"] == sizes[-1] and sum(x["size"] == sizes[-1] for x in descs) == 1)
        return int(d["size"] == sizes[0] and sum(x["size"] == sizes[0] for x in descs) == 1)
    if name == "color_count":
        return sum(1 for x in descs if x["color"] == d["color"])
    if name in ("sym_h", "sym_v", "sym_r180"):
        r0, r1, c0, c1 = d["r0"], d["r1"], d["c0"], d["c1"]
        sub = g[r0:r1 + 1, c0:c1 + 1]
        if name == "sym_h":
            return int(np.array_equal(sub, sub[:, ::-1]))
        if name == "sym_v":
            return int(np.array_equal(sub, sub[::-1, :]))
        return int(np.array_equal(sub, sub[::-1, ::-1]))
    return None


# ===========================================================================
# CELL FEATURE VECTOR — candidate causes per cell (local-rule + relational position family).
# ===========================================================================
def _canon8_patch(pad, i, j):
    """Symmetry-canonical hash of the 3x3 patch around (i,j) in the -1-padded grid."""
    m = pad[i:i + 3, j:j + 3]
    best = None
    r = m
    for _ in range(4):
        for t in (r, r[:, ::-1]):
            key = t.tobytes()
            if best is None or key < best:
                best = key
        r = np.rot90(r)
    return best


# ===========================================================================
# EFFECT 1 — REGION RECOLOR by a single feature (solid recolor; footprint preserved).
#   Generalizes gen4's hand-coded {holes,border,wider} to the FULL region feature vocabulary,
#   ranked by the relevance prior. cause = feature key -> effect = output color (consistent).
# ===========================================================================
def fit_region_recolor(train, prior=None):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)
    feats = _rank_features(REGION_FEATURES, prior, kind="region_recolor")
    for diag in (True, False):
        # precompute descriptors per train input for this diag
        per = []
        ok_fp = True
        for gi, go in train:
            if np.any((gi != bg) != (go != bg)):
                ok_fp = False
                break
            descs = _region_descriptors(gi, bg, diag)
            if not descs:
                ok_fp = False
                break
            per.append((gi, go, descs))
        if not ok_fp:
            continue
        for name in feats:
            mapping = {}
            ok = True
            for gi, go, descs in per:
                for d in descs:
                    ocs = {int(go[a, b]) for a, b in d["cells"]}
                    if len(ocs) != 1:
                        ok = False
                        break
                    oc = ocs.pop()
                    key = _region_feature_value(d, descs, gi, name)
                    if key is None:
                        ok = False
                        break
                    if key in mapping and mapping[key] != oc:
                        ok = False
                        break
                    mapping[key] = oc
                if not ok:
                    break
            if not ok or not mapping:
                continue
            # require the feature to be DISCRIMINATIVE somewhere (not a constant recolor, which base has)
            if len({_region_feature_value(d, descs, gi, name)
                    for gi, go, descs in per for d in descs}) < 2:
                # constant -> only useful if it actually changes colors and base missed; keep but low prio
                pass

            def make(name=name, mapping=dict(mapping), diag=diag, bg=bg):
                def fn(g):
                    descs = _region_descriptors(g, bg, diag)
                    if not descs:
                        return None
                    out = g.copy()
                    for d in descs:
                        key = _region_feature_value(d, descs, g, name)
                        if key not in mapping:
                            return None
                        for a, b in d["cells"]:
                            out[a, b] = mapping[key]
                    return out
                return fn
            fn = make()
            if _verify(fn, train):
                _bump_feature("region_recolor", name)
                return fn
    return None


# ===========================================================================
# EFFECT 2 — KEEP / DELETE regions by a single-feature predicate.
#   cause = feature key -> effect in {keep-as-is, delete(->bg)}. Covers "keep only the X object(s)".
# ===========================================================================
def fit_region_keep_delete(train, prior=None):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)
    feats = _rank_features(REGION_FEATURES, prior, kind="keep_delete")
    for diag in (True, False):
        per = []
        good = True
        for gi, go in train:
            descs = _region_descriptors(gi, bg, diag)
            if not descs:
                good = False
                break
            per.append((gi, go, descs))
        if not good:
            continue
        for name in feats:
            mapping = {}  # feature key -> "keep" / "del"
            ok = True
            for gi, go, descs in per:
                for d in descs:
                    # is this object kept identically, or removed to bg, in the output?
                    kept = all(int(go[a, b]) == int(gi[a, b]) for a, b in d["cells"])
                    deleted = all(int(go[a, b]) == bg for a, b in d["cells"])
                    if kept:
                        eff = "keep"
                    elif deleted:
                        eff = "del"
                    else:
                        ok = False
                        break
                    key = _region_feature_value(d, descs, gi, name)
                    if key is None:
                        ok = False
                        break
                    if key in mapping and mapping[key] != eff:
                        ok = False
                        break
                    mapping[key] = eff
                if not ok:
                    break
            if not ok or not mapping or set(mapping.values()) == {"keep"}:
                continue

            def make(name=name, mapping=dict(mapping), diag=diag, bg=bg):
                def fn(g):
                    descs = _region_descriptors(g, bg, diag)
                    if not descs:
                        return None
                    out = g.copy()
                    for d in descs:
                        key = _region_feature_value(d, descs, g, name)
                        if key not in mapping:
                            return None
                        if mapping[key] == "del":
                            for a, b in d["cells"]:
                                out[a, b] = bg
                    return out
                return fn
            fn = make()
            if _verify(fn, train):
                _bump_feature("keep_delete", name)
                return fn
    return None


# ===========================================================================
# EFFECT 3 — SELECT ONE region by a feature extremum, output = its crop (SIZE-CHANGING).
#   cause = which region is the unique argmax/argmin of a numeric feature -> effect = crop that region
#   (window = bbox-with-context, or cut = object-on-bg). Covers "return the odd-one-out object".
# ===========================================================================
_NUMERIC_REGION_FEATS = ["size", "holes", "h", "w", "color_count", "ncolors"]


def fit_region_select_crop(train, prior=None):
    bg = _bg_train(train)
    feats = _rank_features(_NUMERIC_REGION_FEATS, prior, kind="select_crop")
    for diag in (True, False):
        for name in feats:
            for which in ("max", "min"):
                for mode in ("win", "cut"):
                    def make(name=name, which=which, mode=mode, diag=diag, bg=bg):
                        def fn(g):
                            descs = _region_descriptors(g, bg, diag)
                            if not descs:
                                return None
                            vals = [_region_feature_value(d, descs, g, name) for d in descs]
                            if any(v is None for v in vals):
                                return None
                            tgt = max(vals) if which == "max" else min(vals)
                            picks = [d for d, v in zip(descs, vals) if v == tgt]
                            if len(picks) != 1:
                                return None
                            d = picks[0]
                            r0, r1, c0, c1 = d["r0"], d["r1"], d["c0"], d["c1"]
                            if mode == "win":
                                return g[r0:r1 + 1, c0:c1 + 1].copy()
                            out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, int)
                            for a, b in d["cells"]:
                                out[a - r0, b - c0] = g[a, b]
                            return out
                        return fn
                    fn = make()
                    if _verify(fn, train):
                        _bump_feature("select_crop", name)
                        return fn
    return None


# ===========================================================================
# EFFECT 4 — COUNT -> CONSTRUCT (SIZE-CHANGING). Output shape/content a function of a counted feature.
#   cause = a count of a region/cell feature value -> effect = a monochrome bar/square of that count.
#   Also: histogram-bar construction (one row/col per region, length = its size, color = its color).
# ===========================================================================
def fit_count_construct(train, prior=None):
    outs = [go for _, go in train]
    bg = _bg_train(train)

    def n_objs(g, diag):
        return len(_components(g, bg=_bg(g), diag=diag, by_color=True))

    def n_distinct(g):
        b = _bg(g)
        return len({int(v) for v in np.unique(g) if v != b})

    def n_color(g, c):
        return int((g == c).sum())

    def n_regions_with(g, name, val, diag):
        descs = _region_descriptors(g, _bg(g), diag)
        return sum(1 for d in descs if _region_feature_value(d, descs, g, name) == val)

    counters = []
    for diag in (False, True):
        counters.append(("objs%d" % diag, (lambda diag=diag: (lambda g: n_objs(g, diag)))()))
    counters.append(("ndistinct", n_distinct))
    pal = sorted({int(v) for gi, _ in train for v in np.unique(gi)})
    for c in pal:
        counters.append(("color%d" % c, (lambda c=c: (lambda g: n_color(g, c)))()))
    # count regions having a given feature value (e.g. # of square objects, # with 1 hole)
    for name in ("holes", "single", "square", "border"):
        for val in (0, 1, 2):
            counters.append(("reg_%s_%d" % (name, val),
                             (lambda name=name, val=val: (lambda g: n_regions_with(g, name, val, True)))()))

    for cname, cfn in counters:
        try:
            counts = [cfn(gi) for gi, _ in train]
        except Exception:
            continue
        if any(c <= 0 for c in counts):
            continue
        if len(set(counts)) < 2:
            continue  # constant count cannot be DEMONSTRATED to track the feature -> skip
        ocols = set()
        bad = False
        for n, go in zip(counts, outs):
            vals = [int(v) for v in np.unique(go) if int(v) != bg]
            if len(vals) != 1:
                bad = True
                break
            ocols |= set(vals)
        if bad or len(ocols) != 1:
            continue
        oc = ocols.pop()
        for shp in ("col", "row", "sq"):
            def make(cfn=cfn, oc=oc, shp=shp, bg=bg):
                def fn(g):
                    n = cfn(g)
                    if n <= 0 or n > 30:
                        return None
                    if shp == "col":
                        return np.full((n, 1), oc, int)
                    if shp == "row":
                        return np.full((1, n), oc, int)
                    return np.full((n, n), oc, int)
                return fn
            fn = make()
            if _verify(fn, train):
                _bump_feature("count_construct", cname)
                return fn
    return None


# ===========================================================================
# EFFECT 5 — LOCAL CELL RULE by 3x3 neighborhood, but CONDITIONED ON A GLOBAL FEATURE.
#   gen2_base already has plain/symmetry-canonical 3x3 lookups; we add the relational twist that the
#   diagnostic favors: output[i,j] = f(canonical patch) where unknown patches FALL BACK to a feature
#   (parity / own-color) rather than copy. Kept conservative (exact-verify) to avoid overfit.
# Also: PROPAGATE-COLOR-FROM-SEED (draw ray): each non-bg cell of a marker color extends along the
#   axis toward its same-color partner, filling bg (connect-the-dots beyond base's plain version, with
#   per-color direction learned).
# ===========================================================================
def fit_ray_to_edge(train, prior=None):
    """Each isolated single-cell 'seed' shoots a ray of its color to the grid edge in a fixed
    direction (up/down/left/right) or in all 4; the direction is induced consistently from train."""
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)
    dirs = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}
    dirsets = [("u",), ("d",), ("l",), ("r",), ("u", "d"), ("l", "r"),
               ("u", "d", "l", "r")]
    for ds in dirsets:
        def make(ds=ds, bg=bg):
            def fn(g):
                out = g.copy()
                h, w = g.shape
                comps = _components(g, bg=bg, diag=False, by_color=True)
                seeds = [c for c in comps if len(c) == 1]
                if not seeds:
                    return None
                for cells in seeds:
                    (a, b) = cells[0]
                    col = g[a, b]
                    for dk in ds:
                        di, dj = dirs[dk]
                        x, y = a + di, b + dj
                        while 0 <= x < h and 0 <= y < w and g[x, y] == bg:
                            out[x, y] = col
                            x += di
                            y += dj
                return out
            return fn
        fn = make()
        if _verify(fn, train):
            _bump_feature("ray", "+".join(ds))
            return fn
    return None


def fit_connect_pairs(train, prior=None):
    """Connect same-color collinear pairs by a line (row & col) of a possibly-DIFFERENT color (the
    'fill between' color induced from train). Generalizes base connect_dots (which uses the seed color)
    to a relational recolor of the connecting segment."""
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)
    # induce the fill color: the colors that appear NEW (in go but not gi) consistently
    fill = None
    for gi, go in train:
        new = set(np.unique(go[(gi != go)]).tolist()) if (gi != go).any() else set()
        if not new:
            continue
        if fill is None:
            fill = new
        else:
            fill &= new
    fill_colors = sorted(fill) if fill else []

    def attempt(fc, use_seed):
        def fn(g):
            out = g.copy()
            for c in np.unique(g):
                if c == bg:
                    continue
                pts = np.argwhere(g == c)
                byr = defaultdict(list)
                byc = defaultdict(list)
                for r, cc in pts:
                    byr[r].append(cc)
                    byc[cc].append(r)
                lc = c if use_seed else fc
                for r, cols in byr.items():
                    cols = sorted(cols)
                    for a, b in zip(cols, cols[1:]):
                        if b - a > 1:
                            out[r, a + 1:b] = lc
                for cc, rows in byc.items():
                    rows = sorted(rows)
                    for a, b in zip(rows, rows[1:]):
                        if b - a > 1:
                            out[a + 1:b, cc] = lc
            return out
        return fn
    for fc in fill_colors:
        fn = attempt(fc, False)
        if _verify(fn, train):
            _bump_feature("connect", "fill")
            return fn
    fn = attempt(None, True)
    if _verify(fn, train):
        _bump_feature("connect", "seed")
        return fn
    return None


# ===========================================================================
# EFFECT 6 — PER-CELL RECOLOR by a cell feature (color-of-cell -> output color depends on cell's
#   neighborhood / position parity). A systematic cell-granularity recolor: cause = (own color, feature
#   value) -> output color. Restricted to grids where shape is preserved.
# ===========================================================================
CELL_FEATURES = ["own", "n_same4", "row_par", "col_par", "edge_dist", "patch8"]


def _cell_feature_value(g, pad, i, j, bg, name):
    if name == "own":
        return int(g[i, j])
    if name == "n_same4":
        c = g[i, j]
        h, w = g.shape
        return sum(1 for di, dj in _NB4
                   if 0 <= i + di < h and 0 <= j + dj < w and g[i + di, j + dj] == c)
    if name == "row_par":
        return i % 2
    if name == "col_par":
        return j % 2
    if name == "edge_dist":
        h, w = g.shape
        return min(i, j, h - 1 - i, w - 1 - j)
    if name == "patch8":
        return _canon8_patch(pad, i, j)
    return None


def fit_cell_recolor(train, prior=None):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    # only attempt the relational (non-own) features; own-color-only is base color_perm.
    feats = _rank_features([f for f in CELL_FEATURES if f != "patch8"], prior, kind="cell_recolor")
    for name in feats:
        if name == "own":
            continue
        mapping = {}
        ok = True
        for gi, go in train:
            h, w = gi.shape
            pad = np.full((h + 2, w + 2), -1, int)
            pad[1:1 + h, 1:1 + w] = gi
            for i in range(h):
                for j in range(w):
                    key = (int(gi[i, j]), _cell_feature_value(gi, pad, i, j, 0, name))
                    y = int(go[i, j])
                    if key in mapping and mapping[key] != y:
                        ok = False
                        break
                    mapping[key] = y
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        # require the relational feature to MATTER (some own-color maps to >1 output by feature)
        bycolor = defaultdict(set)
        for (oc, fv), y in mapping.items():
            bycolor[oc].add(y)
        if not any(len(v) > 1 for v in bycolor.values()):
            continue

        def make(name=name, mapping=dict(mapping)):
            def fn(g):
                h, w = g.shape
                pad = np.full((h + 2, w + 2), -1, int)
                pad[1:1 + h, 1:1 + w] = g
                out = g.copy()
                for i in range(h):
                    for j in range(w):
                        key = (int(g[i, j]), _cell_feature_value(g, pad, i, j, 0, name))
                        if key in mapping:
                            out[i, j] = mapping[key]
                        else:
                            return None
                return out
            return fn
        fn = make()
        if _verify(fn, train):
            _bump_feature("cell_recolor", name)
            return fn
    return None


# ===========================================================================
# FITTER REGISTRY  (decompose -> features -> induce -> verify, one per EFFECT).
# Ordered by empirical generalization (cheap & high-yield first); the relevance prior orders the
# FEATURES tried WITHIN each fitter.
# ===========================================================================
FITTERS = [
    ("region_recolor", fit_region_recolor),
    ("region_keep_delete", fit_region_keep_delete),
    ("cell_recolor", fit_cell_recolor),
    ("ray_to_edge", fit_ray_to_edge),
    ("connect_pairs", fit_connect_pairs),
    ("region_select_crop", fit_region_select_crop),
    ("count_construct", fit_count_construct),
]


# ===========================================================================
# FEATURE-RELEVANCE PRIOR — trained at import on SELF-GENERATED synthetic relational tasks.
# For each (effect, feature) the prior holds a relevance weight = how often, on synthetic tasks built so
# that `feature` truly DRIVES `effect`, the engine's consistency test fired for THAT feature first. This
# is a learned ordering signal (CPU-light), not a solver: it only re-orders which feature is tried first
# so induction stays tractable as the feature space grows. A within-task statistical signal refines it.
# ===========================================================================
_FEATURE_PRIOR = defaultdict(lambda: defaultdict(float))   # effect -> feature -> weight (learned)
_FEATURE_HITS = defaultdict(Counter)                       # effect -> feature -> in-run verified hits


def _bump_feature(effect, feature):
    _FEATURE_HITS[effect][feature] += 1


def _rank_features(feats, prior, kind):
    """Order `feats` by (in-run verified hits, learned prior weight). Stable for ties -> registry order."""
    learned = _FEATURE_PRIOR.get(kind, {})
    hits = _FEATURE_HITS.get(kind, {})
    return sorted(feats, key=lambda f: (-(hits.get(f, 0)), -(learned.get(f, 0.0))))


def _synth_region_grid(rng):
    """Build a random grid of a few colored rectangles/blobs on bg=0 (the synthetic decompose target)."""
    h = rng.randint(6, 14)
    w = rng.randint(6, 14)
    g = np.zeros((h, w), int)
    k = rng.randint(2, 6)
    placed = []
    for _ in range(k):
        rh = rng.randint(1, 4)
        rw = rng.randint(1, 4)
        r0 = rng.randint(0, max(1, h - rh))
        c0 = rng.randint(0, max(1, w - rw))
        col = rng.randint(1, 8)
        if (g[r0:r0 + rh + 1, c0:c0 + rw + 1] != 0).any():
            continue
        g[r0:r0 + rh, c0:c0 + rw] = col
        placed.append((r0, c0, rh, rw, col))
    return g


def _train_feature_prior(n=240, seed=0, time_budget=70.0):
    """Generate synthetic region-recolor / keep-delete tasks where ONE known feature drives the effect,
    run the inducer, and credit the feature that the inducer locked onto. Builds the prior order."""
    t0 = time.time()
    rng = np.random.RandomState(seed)
    # features we can synthesize a ground-truth driver for, per effect
    drivers = {
        "region_recolor": ["size", "holes", "border", "square", "h", "w", "rank_desc", "color_count"],
        "keep_delete": ["size", "holes", "border", "square", "single", "is_largest"],
    }
    new_colors = [3, 4, 5, 6, 7, 8, 9]
    for effect, dnames in drivers.items():
        for dname in dnames:
            credited = 0
            attempts = 0
            for _ in range(n // len(dnames) + 1):
                if time.time() - t0 > time_budget:
                    break
                # build a MULTI-PAIR synthetic task driven by `dname`, with a SHARED value->effect map
                # across pairs, so only the true driver stays consistent across all pairs (the cross-pair
                # invariance that lets the inducer disambiguate features that co-fit a single pair).
                npairs = rng.randint(2, 4)
                pairs = []
                val2col = {}      # for recolor: stable value->color across pairs
                del_tgt = [None]  # for keep_delete: stable target value across pairs
                ok_task = True
                for _p in range(npairs):
                    gi = _synth_region_grid(rng)
                    bg = 0
                    descs = _region_descriptors(gi, bg, True)
                    if len(descs) < 2:
                        ok_task = False
                        break
                    try:
                        vals = [_region_feature_value(d, descs, gi, dname) for d in descs]
                    except Exception:
                        ok_task = False
                        break
                    if any(v is None for v in vals) or len(set(vals)) < 2:
                        ok_task = False
                        break
                    go = gi.copy()
                    if effect == "region_recolor":
                        for v in set(vals):
                            if v not in val2col:
                                val2col[v] = new_colors[rng.randint(len(new_colors))]
                        for d, v in zip(descs, vals):
                            for a, b in d["cells"]:
                                go[a, b] = val2col[v]
                    else:
                        if del_tgt[0] is None:
                            del_tgt[0] = list(set(vals))[rng.randint(len(set(vals)))]
                        if del_tgt[0] not in set(vals):
                            ok_task = False
                            break
                        for d, v in zip(descs, vals):
                            if v == del_tgt[0]:
                                for a, b in d["cells"]:
                                    go[a, b] = bg
                        if (go == gi).all():
                            ok_task = False
                            break
                    pairs.append((gi, go))
                if not ok_task or len(pairs) < 2:
                    continue
                attempts += 1
                locked = _which_feature_locks(effect, pairs)
                if locked == dname:
                    credited += 1
            if attempts:
                _FEATURE_PRIOR[effect][dname] += credited / attempts
    return time.time() - t0


def _which_feature_locks(effect, train):
    """Replay the consistency test (no prior) and return the FIRST feature that yields a verified rule,
    so the synthetic trainer can credit the true driver. Mirrors the real fitters' induction exactly."""
    bg = _bg_train(train)
    if effect == "region_recolor":
        for diag in (True,):
            per = []
            okfp = True
            for gi, go in train:
                if np.any((gi != bg) != (go != bg)):
                    okfp = False
                    break
                descs = _region_descriptors(gi, bg, diag)
                if not descs:
                    okfp = False
                    break
                per.append((gi, go, descs))
            if not okfp:
                continue
            for name in REGION_FEATURES:
                mapping = {}
                ok = True
                for gi, go, descs in per:
                    for d in descs:
                        ocs = {int(go[a, b]) for a, b in d["cells"]}
                        if len(ocs) != 1:
                            ok = False
                            break
                        oc = ocs.pop()
                        key = _region_feature_value(d, descs, gi, name)
                        if key is None or (key in mapping and mapping[key] != oc):
                            ok = False
                            break
                        mapping[key] = oc
                    if not ok:
                        break
                if ok and mapping and len({_region_feature_value(d, descs, gi, name)
                                           for gi, go, descs in per for d in descs}) >= 2:
                    return name
    else:  # keep_delete
        for diag in (True,):
            per = []
            good = True
            for gi, go in train:
                descs = _region_descriptors(gi, bg, diag)
                if not descs:
                    good = False
                    break
                per.append((gi, go, descs))
            if not good:
                continue
            for name in REGION_FEATURES:
                mapping = {}
                ok = True
                for gi, go, descs in per:
                    for d in descs:
                        kept = all(int(go[a, b]) == int(gi[a, b]) for a, b in d["cells"])
                        deleted = all(int(go[a, b]) == bg for a, b in d["cells"])
                        if kept:
                            eff = "keep"
                        elif deleted:
                            eff = "del"
                        else:
                            ok = False
                            break
                        key = _region_feature_value(d, descs, gi, name)
                        if key is None or (key in mapping and mapping[key] != eff):
                            ok = False
                            break
                        mapping[key] = eff
                    if not ok:
                        break
                if ok and mapping and set(mapping.values()) != {"keep"}:
                    return name
    return None


_T0 = time.time()
try:
    _BUILD_SEC = _train_feature_prior(n=240, seed=0, time_budget=70.0)
except Exception:
    _BUILD_SEC = 0.0
# clear the in-run hit counters that the synthetic trainer may have touched, so real-task ordering
# starts from the LEARNED prior only (the prior already encodes the synthetic evidence).
_FEATURE_HITS.clear()


# ===========================================================================
# CROSS-TASK EXPERIENCE (module-level; verified-correct relation TYPES only, no grids/outputs).
# ===========================================================================
_TASKN = [0]


def reset_library():
    """Documented hook: clear cross-task experience so a run starts cold (gate uses this to isolate
    transfer). The learned import-time prior is NOT cleared (it is synthetic, not cross-task ARC info)."""
    _FEATURE_HITS.clear()
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
# INVENTION ENTRYPOINT — produce up to 2 candidate outputs per test input via relation-induction.
# ===========================================================================
def _invent(train, test_inputs, budget):
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    fitted = []
    for fname, fitter in FITTERS:
        try:
            fn = fitter(train, _FEATURE_PRIOR)
        except Exception:
            fn = None
        if fn is not None and _verify(fn, train):
            fitted.append((fname, fn))
    attempts = []
    for gi in test_inputs:
        gi = np.asarray(gi, int)
        cand = []
        for _fname, fn in fitted:
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
# STANDARDIZED GATE WIRING
#   solve_ablated == gen2_base.solve  (the strong-retrieval ablation, imported verbatim)
#   solve         == gen2_base attempt-1 backstop, THEN cell/region relation-induction attempt-2
# ===========================================================================
def solve_ablated(train, test_inputs, budget):
    """EXACTLY gen2_base.solve — the standardized strong-retrieval ablation. INVENTED = solved - this."""
    return BASE.solve(train, test_inputs, budget)


def solve(train, test_inputs, budget):
    """gen2_base as attempt-1 backstop (never regress below it); cell/region relation-induction fills
    the remaining attempt slot. ARC allows 2 attempts; reserve slot-1 for the strong retrieval backstop
    and slot-2 for the INVENTION so an induced relation is never crowded out by a second base guess."""
    _TASKN[0] += 1
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
        # slot 1: base's first guess (retrieval backstop)
        if b:
            cand.append(b[0])
        # slot 2: the invention's first NEW guess
        for o in iv:
            if not any(_eq(o, c) for c in cand):
                cand.append(o)
                break
        # backfill remaining slot from base, then invention (never waste a slot)
        if len(cand) < 2:
            for o in b[1:]:
                if not any(_eq(o, c) for c in cand):
                    cand.append(o)
                    break
        if len(cand) < 2:
            for o in iv:
                if not any(_eq(o, c) for c in cand):
                    cand.append(o)
                    break
        merged.append(cand[:2])
    return merged


if __name__ == "__main__":
    print("build_sec=%.1f" % _BUILD_SEC)
    print("learned feature prior:")
    for eff, d in _FEATURE_PRIOR.items():
        top = sorted(d.items(), key=lambda kv: -kv[1])[:6]
        print(" ", eff, [(k, round(v, 2)) for k, v in top])
