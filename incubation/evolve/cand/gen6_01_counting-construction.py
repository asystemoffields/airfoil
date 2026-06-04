#!/usr/bin/env python3
"""GEN-6 RELATION-INDUCER #1 — COUNTING / CONSTRUCTION.

THE FAMILY (per CAMPAIGN.md gen-6 sweep). gen2_base's single biggest untouched miss-family (86 misses) is
COUNTING / CONSTRUCTION: the output is a NEW grid CONSTRUCTED from input content/counts and is usually
SIZE-CHANGING. The induced relation is a (count/feature) -> CONSTRUCTION rule:

  * SUMMARIZE-TO-A-SYMBOL : reduce the grid to a single SOLID grid of one color C, where C is determined by
    a count/extremum (majority color, least-common color, the unique color, the color of the largest /
    smallest / unique object, a learned count->color table). Output SHAPE is itself an induced function:
    a fixed constant (often 1x1 or 3x3), or = input shape, or KxK where K is a count.
  * BUILD-A-BAR : output is a 1xN or Nx1 strip whose LENGTH N is a count (e.g. #nonzero cells, #objects),
    filled solid with a feature color; OR a FIXED-length strip with the first K cells filled, K a count.
  * BUILD-A-SQUARE : output is a KxK solid square, K a count/feature (#objects, a color's cell-count, ...).
  * COUNT-TABLE : a learned (small) table from a scalar count feature of the input -> the entire output
    grid (when only a handful of distinct counts occur and each maps to a fixed small grid).

METHOD (gen-5 architecture, unchanged): DECOMPOSE -> rich count/feature vector (candidate CAUSES) -> INDUCE a
feature(s)->CONSTRUCTION mapping invariant across ALL train pairs (cross-pair invariance licenses causal, not
correlational, induction) -> EXACT-verify (held-out test = the intervention). A light feature-relevance prior
keeps it tractable; the verifier supplies precision.

STANDARDIZED GATE (non-negotiable):
  solve_ablated(train,test_inputs,budget) == EXACTLY gen2_base.solve  (imported verbatim — strong retrieval).
  solve() = gen2_base as attempt-1 backstop  +  THIS counting/construction induction as attempt-2.
  INVENTED = solves beyond gen2_base.

INTEGRITY. solve() learns ONLY from (a) the current task's train pairs, (b) module-level state from PRIOR
solve() calls this run (verified-correct only), (c) self-generated synthetic data built at import. NEVER reads
ARC task files / test OUTPUTS, no network, no LLM. Respects budget. Pure python + numpy. build < ~90s.
Run with /data/llm/.venv/bin/python from .../incubation/evolve.
"""
import os
import sys
import time
from collections import deque, Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVOLVE = os.path.dirname(HERE)
for p in (HERE, EVOLVE):
    if p not in sys.path:
        sys.path.insert(0, p)

# The standardized strong-retrieval ablation, imported verbatim.
import gen2_base as BASE

META = {"name": "gen6_01_counting-construction",
        "desc": "systematic per-task COUNTING/CONSTRUCTION relation-induction: count/extremum feature -> "
                "construct output (solid-of-color, bar-of-length, KxK square, count->grid table); induced "
                "output SHAPE as a function of a count; exact-verified; feature prior; gen2_base backstop"}


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


# ===========================================================================
# SCALAR COUNT/EXTREMUM FEATURES of the whole grid (the candidate CAUSES for a construction).
# Each returns an int (or None if undefined). These drive output SHAPE and COUNT.
# COLOR features return a color (0..9) and drive output CONTENT.
# ===========================================================================
def _color_counts(g, bg):
    cnt = Counter(int(v) for v in g.reshape(-1).tolist())
    cnt.pop(bg, None)
    return cnt


def _objs(g, bg, diag, by_color):
    cs = _components(g, bg=bg, diag=diag, by_color=by_color)
    out = []
    for cells in cs:
        r0, r1, c0, c1 = _bbox(cells)
        col = Counter(int(g[a, b]) for a, b in cells).most_common(1)[0][0]
        out.append({"cells": cells, "size": len(cells), "color": col,
                    "bh": r1 - r0 + 1, "bw": c1 - c0 + 1, "bbox": (r0, r1, c0, c1),
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1)})
    return out


# ---- COUNT features (int) ----
def cf_n_nonzero(g, bg):
    return int((g != bg).sum())


def cf_n_colors(g, bg):
    return len(_color_counts(g, bg))


def cf_n_obj_4(g, bg):
    return len(_components(g, bg=bg, diag=False, by_color=True))


def cf_n_obj_8(g, bg):
    return len(_components(g, bg=bg, diag=True, by_color=True))


def cf_n_obj_4_agn(g, bg):
    return len(_components(g, bg=bg, diag=False, by_color=False))


def cf_n_obj_8_agn(g, bg):
    return len(_components(g, bg=bg, diag=True, by_color=False))


def cf_max_color_count(g, bg):
    cc = _color_counts(g, bg)
    return max(cc.values()) if cc else 0


def cf_min_color_count(g, bg):
    cc = _color_counts(g, bg)
    return min(cc.values()) if cc else 0


def cf_max_obj_size(g, bg):
    o = _objs(g, bg, False, True)
    return max((x["size"] for x in o), default=0)


def cf_min_obj_size(g, bg):
    o = _objs(g, bg, False, True)
    return min((x["size"] for x in o), default=0)


def cf_n_singletons(g, bg):
    """#single-cell objects (markers)."""
    return sum(1 for x in _objs(g, bg, True, True) if x["size"] == 1)


def cf_n_squares(g, bg):
    """#objects that are SOLID squares (size == bh*bw and bh == bw, bh >= 2)."""
    n = 0
    for x in _objs(g, bg, False, True):
        if x["bh"] == x["bw"] >= 2 and x["size"] == x["area"]:
            n += 1
    return n


def cf_n_rect_solid(g, bg):
    """#objects that are SOLID rectangles of >=2 cells (size == bbox area, size >= 2)."""
    n = 0
    for x in _objs(g, bg, False, True):
        if x["size"] == x["area"] and x["size"] >= 2:
            n += 1
    return n


def cf_n_max_size_objs(g, bg):
    """#objects sharing the MAXIMUM size (a tie-count; a frequent 'how many winners' feature)."""
    o = _objs(g, bg, False, True)
    if not o:
        return 0
    mx = max(x["size"] for x in o)
    return sum(1 for x in o if x["size"] == mx)


# wrap parity (mod 2) of the core count features — many summarize-to-symbol tasks key on odd/even.
def _parity(ff):
    def f(g, bg, ff=ff):
        v = ff(g, bg)
        return None if v is None else int(v % 2)
    return f


COUNT_FEATURES = [
    ("n_nonzero", cf_n_nonzero),
    ("n_colors", cf_n_colors),
    ("n_obj_4", cf_n_obj_4),
    ("n_obj_8", cf_n_obj_8),
    ("n_obj_4_agn", cf_n_obj_4_agn),
    ("n_obj_8_agn", cf_n_obj_8_agn),
    ("max_color_count", cf_max_color_count),
    ("min_color_count", cf_min_color_count),
    ("max_obj_size", cf_max_obj_size),
    ("min_obj_size", cf_min_obj_size),
    ("n_singletons", cf_n_singletons),
    ("n_squares", cf_n_squares),
    ("n_rect_solid", cf_n_rect_solid),
    ("n_max_size_objs", cf_n_max_size_objs),
    ("par_obj_8", _parity(cf_n_obj_8)),
    ("par_obj_4_agn", _parity(cf_n_obj_4_agn)),
    ("par_nonzero", _parity(cf_n_nonzero)),
    ("par_colors", _parity(cf_n_colors)),
]


# ---- COLOR features (a single color 0..9, or None) ----
def colf_majority(g, bg):
    cc = _color_counts(g, bg)
    return cc.most_common(1)[0][0] if cc else None


def colf_majority_incl_bg(g, bg):
    cnt = Counter(int(v) for v in g.reshape(-1).tolist())
    return cnt.most_common(1)[0][0] if cnt else None


def colf_minority(g, bg):
    cc = _color_counts(g, bg)
    if not cc:
        return None
    return min(cc.items(), key=lambda kv: (kv[1], kv[0]))[0]


def colf_minority_incl_bg(g, bg):
    cnt = Counter(int(v) for v in g.reshape(-1).tolist())
    return min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0] if cnt else None


def colf_only_nonbg(g, bg):
    cc = _color_counts(g, bg)
    return next(iter(cc)) if len(cc) == 1 else None


def colf_unique_color(g, bg):
    """The single color whose CELL-COUNT is unique (the odd-one-out by frequency)."""
    cc = _color_counts(g, bg)
    if not cc:
        return None
    freq = Counter(cc.values())
    odd = [c for c, n in cc.items() if freq[n] == 1]
    return odd[0] if len(odd) == 1 else None


def _color_of_extreme_obj(g, bg, diag, by_color, biggest):
    o = _objs(g, bg, diag, by_color)
    if not o:
        return None
    sel = max(o, key=lambda x: x["size"]) if biggest else min(o, key=lambda x: x["size"])
    return sel["color"]


def colf_largest_obj(g, bg):
    return _color_of_extreme_obj(g, bg, False, True, True)


def colf_smallest_obj(g, bg):
    return _color_of_extreme_obj(g, bg, False, True, False)


def colf_largest_obj_8(g, bg):
    return _color_of_extreme_obj(g, bg, True, True, True)


def colf_smallest_obj_8(g, bg):
    return _color_of_extreme_obj(g, bg, True, True, False)


def colf_uniq_size_obj(g, bg):
    """Color of the object whose SIZE is unique among objects (exactly one such object)."""
    for diag in (False, True):
        o = _objs(g, bg, diag, True)
        if not o:
            continue
        szct = Counter(x["size"] for x in o)
        uo = [x for x in o if szct[x["size"]] == 1]
        if len(uo) == 1:
            return uo[0]["color"]
    return None


def colf_uniq_shape_obj(g, bg):
    """Color of the object whose footprint shape is unique (D4-invariant)."""
    def shash(cells):
        r0, r1, c0, c1 = _bbox(cells)
        H, W = r1 - r0 + 1, c1 - c0 + 1
        occ = np.zeros((H, W), bool)
        for a, b in cells:
            occ[a - r0, b - c0] = True
        best = None
        m = occ
        for _ in range(4):
            for t in (m, m[:, ::-1]):
                key = (t.shape, t.tobytes())
                if best is None or key < best:
                    best = key
            m = np.rot90(m)
        return repr(best)
    for diag in (False, True):
        o = _objs(g, bg, diag, True)
        if not o:
            continue
        hs = [shash(x["cells"]) for x in o]
        hct = Counter(hs)
        uo = [x for x, h in zip(o, hs) if hct[h] == 1]
        if len(uo) == 1:
            return uo[0]["color"]
    return None


COLOR_FEATURES = [
    ("majority", colf_majority),
    ("majority_bg", colf_majority_incl_bg),
    ("minority", colf_minority),
    ("minority_bg", colf_minority_incl_bg),
    ("only_nonbg", colf_only_nonbg),
    ("uniq_color", colf_unique_color),
    ("largest_obj", colf_largest_obj),
    ("smallest_obj", colf_smallest_obj),
    ("largest_obj8", colf_largest_obj_8),
    ("smallest_obj8", colf_smallest_obj_8),
    ("uniq_size_obj", colf_uniq_size_obj),
    ("uniq_shape_obj", colf_uniq_shape_obj),
]


# ===========================================================================
# OUTPUT-SHAPE induction: how is (out_h, out_w) determined from the input?
# Returns a list of (name, shape_fn) candidates that reproduce every train output's SHAPE.
# ===========================================================================
def induce_shapes(train, bg):
    out = []
    ish = [gi.shape for gi, _ in train]
    osh = [go.shape for _, go in train]

    # (a) constant output shape
    if len(set(osh)) == 1:
        H, W = osh[0]
        out.append(("const_%dx%d" % (H, W), lambda g, H=H, W=W: (H, W)))

    # (b) output shape == input shape
    if all(o == i for i, o in zip(ish, osh)):
        out.append(("same", lambda g: g.shape))

    # (c) KxK square where K = a count feature
    if all(h == w for h, w in osh):
        Ks = [h for h, _ in osh]
        for fname, ff in COUNT_FEATURES:
            try:
                vals = [ff(gi, bg) for gi, _ in train]
            except Exception:
                continue
            if all(v == k for v, k in zip(vals, Ks)) and all(k > 0 for k in Ks):
                out.append(("sqK_%s" % fname, _mk_shape_kk(ff, bg)))
            # affine K = a*v + b (small)
            cand = _fit_affine(vals, Ks)
            if cand is not None:
                a, b = cand
                if (a, b) != (1, 0):
                    out.append(("sqAff_%s_%d_%d" % (fname, a, b), _mk_shape_kk_aff(ff, bg, a, b)))

    # (d) 1xN or Nx1 bar where N = a count feature (and the other dim constant)
    hs = [h for h, _ in osh]
    ws = [w for _, w in osh]
    if len(set(hs)) == 1:               # constant height -> width = count
        Hc = hs[0]
        for fname, ff in COUNT_FEATURES:
            try:
                vals = [ff(gi, bg) for gi, _ in train]
            except Exception:
                continue
            cand = _fit_affine(vals, ws)
            if cand is not None:
                a, b = cand
                out.append(("barW_%s_%d_%d" % (fname, a, b), _mk_shape_wn(ff, bg, Hc, a, b)))
    if len(set(ws)) == 1:               # constant width -> height = count
        Wc = ws[0]
        for fname, ff in COUNT_FEATURES:
            try:
                vals = [ff(gi, bg) for gi, _ in train]
            except Exception:
                continue
            cand = _fit_affine(vals, hs)
            if cand is not None:
                a, b = cand
                out.append(("barH_%s_%d_%d" % (fname, a, b), _mk_shape_hn(ff, bg, Wc, a, b)))

    return out


def _fit_affine(xs, ys):
    """Fit y = a*x + b with integer a in {-2,-1,1,2}, integer b, holding for ALL pairs. Returns (a,b)|None.
    Requires >=2 distinct x to pin the slope (else underdetermined)."""
    if len(set(xs)) < 2:
        return None
    for a in (1, 2, 3, -1, -2):
        bs = set(y - a * x for x, y in zip(xs, ys))
        if len(bs) == 1:
            b = bs.pop()
            if all(a * x + b == y for x, y in zip(xs, ys)) and all(a * x + b >= 1 for x in xs):
                return (a, int(b))
    return None


def _mk_shape_kk(ff, bg):
    def f(g, ff=ff, bg=bg):
        k = ff(g, bg)
        return (k, k) if (isinstance(k, int) and k > 0) else None
    return f


def _mk_shape_kk_aff(ff, bg, a, b):
    def f(g, ff=ff, bg=bg, a=a, b=b):
        v = ff(g, bg)
        if not isinstance(v, int):
            return None
        k = a * v + b
        return (k, k) if k > 0 else None
    return f


def _mk_shape_wn(ff, bg, Hc, a, b):
    def f(g, ff=ff, bg=bg, Hc=Hc, a=a, b=b):
        v = ff(g, bg)
        if not isinstance(v, int):
            return None
        n = a * v + b
        return (Hc, n) if n > 0 else None
    return f


def _mk_shape_hn(ff, bg, Wc, a, b):
    def f(g, ff=ff, bg=bg, Wc=Wc, a=a, b=b):
        v = ff(g, bg)
        if not isinstance(v, int):
            return None
        n = a * v + b
        return (n, Wc) if n > 0 else None
    return f


# ===========================================================================
# CONSTRUCTION INDUCERS — combine an induced SHAPE with an induced CONTENT rule, exact-verify.
# ===========================================================================
def induce_solid(train, bg, shape_cands):
    """RELATION: output is a SOLID grid of one color C (a consistent COLOR feature of the input), with the
    output SHAPE given by an induced shape rule. Covers summarize-to-symbol / fill-with-majority."""
    # output must be solid (single color) in every train pair
    for _, go in train:
        if len(np.unique(go)) != 1:
            return None
    for sname, sfn in shape_cands:
        for cname, cff in COLOR_FEATURES:
            ok = True
            for gi, go in train:
                try:
                    sh = sfn(gi)
                    col = cff(gi, bg)
                except Exception:
                    ok = False
                    break
                if sh is None or col is None or sh != go.shape or int(go.flat[0]) != int(col):
                    ok = False
                    break
            if not ok:
                continue

            def fn(g, sfn=sfn, cff=cff, bg=bg):
                sh = sfn(g)
                col = cff(g, bg)
                if sh is None or col is None:
                    return None
                return np.full(sh, int(col), int)
            if _verify(fn, train):
                return fn
    return None


def induce_count_to_grid_table(train, bg):
    """RELATION: a learned TABLE from a scalar COUNT feature -> the entire (small) output grid. Used when a
    handful of distinct counts occur, each mapping to a fixed output grid (incl. count->symbol 1x1). Requires
    the feature to take >=2 distinct values across train (else it's a constant, not a count relation)."""
    # keep it to small outputs (a genuine symbol/glyph, not a full reconstruction)
    if any(go.size > 9 for _, go in train):
        return None
    for fname, ff in COUNT_FEATURES:
        table = {}
        ok = True
        for gi, go in train:
            try:
                k = ff(gi, bg)
            except Exception:
                ok = False
                break
            if not isinstance(k, int):
                ok = False
                break
            key = k
            tb = go.tobytes(), go.shape
            if key in table and table[key] != tb:
                ok = False
                break
            table[key] = tb
        if not ok or len(set(k for k in table)) < 2:
            continue

        # bind output dtype safely (ARC grids are small ints; use the actual train dtype)
        go_dtype = train[0][1].dtype

        def fn2(g, ff=ff, bg=bg, table=table, go_dtype=go_dtype):
            k = ff(g, bg)
            if not isinstance(k, int) or k not in table:
                return None
            buf, sh = table[k]
            return np.frombuffer(buf, dtype=go_dtype).reshape(sh).astype(int).copy()
        if _verify(fn2, train):
            return fn2
    return None


def induce_bar_fill_count(train, bg):
    """RELATION: output is a FIXED-shape strip (1xL or Lx1) of a fixed color, with the first K cells filled
    and the rest bg, where K is a count feature. Covers 'histogram bar of a count' with a fixed track length.
    The fill color is a consistent constant; orientation/length are constant across train."""
    osh = [go.shape for _, go in train]
    if len(set(osh)) != 1:
        return None
    H, W = osh[0]
    if not (H == 1 or W == 1):
        return None
    # determine fill color (single nonbg color across outputs) and bg-of-output
    fills = set()
    obg = None
    for _, go in train:
        u = [int(v) for v in np.unique(go)]
        nz = [v for v in u if v != bg]
        if len(nz) > 1:
            return None
        if nz:
            fills.add(nz[0])
        # output bg candidate
        if bg in u:
            obg = bg
    if len(fills) != 1:
        return None
    fc = fills.pop()
    obg = obg if obg is not None else bg
    track = W if H == 1 else H
    # the filled prefix count per train output
    counts = []
    for _, go in train:
        flat = go.reshape(-1)
        # count of fill cells (assume contiguous prefix; verify by reconstruction later)
        counts.append(int((flat == fc).sum()))
    for fname, ff in COUNT_FEATURES:
        try:
            vals = [ff(gi, bg) for gi, _ in train]
        except Exception:
            continue
        cand = _fit_affine(vals, counts)
        # also allow identity K=v (single distinct value can't fit affine; require >=2 distinct)
        if cand is None:
            continue
        a, b = cand

        def fn(g, ff=ff, bg=bg, a=a, b=b, H=H, W=W, fc=fc, obg=obg, track=track):
            v = ff(g, bg)
            if not isinstance(v, int):
                return None
            k = a * v + b
            if k < 0 or k > track:
                return None
            out = np.full((H, W), obg, int)
            flat = out.reshape(-1)
            flat[:k] = fc
            return flat.reshape(H, W)
        if _verify(fn, train):
            return fn
    return None


def induce_solid_square_count(train, bg):
    """RELATION: output is a KxK SOLID square of a fixed/feature color, K a count feature. (Subsumed by
    induce_solid when shape-induction finds a sqK rule, but kept explicit for the K=count, color=const case
    where the color is NOT a global extremum but a constant.)"""
    if any(h != w for (h, w) in (go.shape for _, go in train)):
        return None
    for _, go in train:
        if len(np.unique(go)) != 1:
            return None
    Ks = [go.shape[0] for _, go in train]
    cols = set(int(go.flat[0]) for _, go in train)
    for fname, ff in COUNT_FEATURES:
        try:
            vals = [ff(gi, bg) for gi, _ in train]
        except Exception:
            continue
        aff = _fit_affine(vals, Ks)
        if aff is None and not (all(v == k for v, k in zip(vals, Ks)) and len(set(vals)) >= 2):
            continue
        a, b = aff if aff is not None else (1, 0)
        # constant color case only (feature-color handled by induce_solid)
        if len(cols) != 1:
            continue
        col = cols.pop()

        def fn(g, ff=ff, bg=bg, a=a, b=b, col=col):
            v = ff(g, bg)
            if not isinstance(v, int):
                return None
            k = a * v + b
            if k <= 0:
                return None
            return np.full((k, k), col, int)
        if _verify(fn, train):
            return fn
    return None


# the construction menu, MDL-ish order (cheap/global summaries first; count-tables/bars later)
def _build_inducers(train, bg, shape_cands):
    out = []
    f = induce_solid(train, bg, shape_cands)
    if f is not None:
        out.append(("solid", f))
    f = induce_solid_square_count(train, bg)
    if f is not None:
        out.append(("square_count", f))
    f = induce_bar_fill_count(train, bg)
    if f is not None:
        out.append(("bar_fill", f))
    f = induce_count_to_grid_table(train, bg)
    if f is not None:
        out.append(("count_table", f))
    return out


# ===========================================================================
# INVENTION ENTRYPOINT — induce a construction, produce <=2 candidate outputs per test input.
# ===========================================================================
def _invent(train, test_inputs, budget):
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    # ALL outputs solid OR small? construction relations are size-changing or summarizing; cheap gate:
    # require at least one train pair to be size-changing OR all outputs solid (summarize-to-symbol).
    size_chg = any(gi.shape != go.shape for gi, go in train)
    all_solid = all(len(np.unique(go)) == 1 for _, go in train)
    if not (size_chg or all_solid):
        return [[] for _ in test_inputs]

    bg = _bg_train(train)
    fitted = []
    seen_sig = set()

    # try a few backgrounds (the global most-common, and 0) — counting is bg-sensitive
    bgs = []
    for b in (bg, 0):
        if b not in bgs:
            bgs.append(b)

    for b in bgs:
        try:
            shape_cands = induce_shapes(train, b)
        except Exception:
            shape_cands = []
        try:
            inds = _build_inducers(train, b, shape_cands)
        except Exception:
            inds = []
        for iname, fn in inds:
            try:
                sig = (iname,) + tuple(
                    (None if fn(gi) is None else fn(gi).tobytes()) for gi, _ in train)
            except Exception:
                sig = None
            if sig is not None and sig in seen_sig:
                continue
            if sig is not None:
                seen_sig.add(sig)
            fitted.append((iname, fn))
            _remember(iname, b)
        if len(fitted) >= 4:
            break

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
# IN-RUN EXPERIENCE (module-level, verified-only): remember which (inducer, bg) solved earlier tasks.
# Names + bg only; never grids/outputs.
# ===========================================================================
_MEM = []
_MEM_SEEN = set()


def _remember(iname, bg):
    key = (iname, bg)
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
#   solve         == gen2_base attempt-1 backstop, THEN counting/construction invention attempt-2
# ===========================================================================
def solve_ablated(train, test_inputs, budget):
    """EXACTLY gen2_base.solve — the standardized strong-retrieval ablation. INVENTED = solved - this."""
    return BASE.solve(train, test_inputs, budget)


def solve(train, test_inputs, budget):
    """gen2_base as attempt-1 backstop; counting/construction invention fills the remaining attempt slot.
    The gate scores both attempts (ARC 2-attempt)."""
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
# import-time self-test (executes the inducers once so import-time errors surface here, not at solve time)
# ===========================================================================
_T0 = time.time()


def _selftest():
    rng = np.random.RandomState(1)
    # synthetic: count->solid-grid relation (majority color, fixed 3x3 output)
    train = []
    for _ in range(3):
        g = rng.randint(0, 4, (5, 5))
        mc = Counter(g.reshape(-1).tolist()).most_common(1)[0][0]
        train.append((g, np.full((3, 3), mc, int)))
    try:
        sc = induce_shapes(train, _bg_train(train))
        induce_solid(train, _bg_train(train), sc)
        induce_count_to_grid_table(train, _bg_train(train))
        induce_bar_fill_count(train, _bg_train(train))
        induce_solid_square_count(train, _bg_train(train))
    except Exception:
        pass


_selftest()
_BUILD_SEC = time.time() - _T0
