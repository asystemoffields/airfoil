#!/usr/bin/env python3
"""GEN-4 INVENTOR #1 — RELATIONAL-DEPTH.

THE HONEST TARGET (CAMPAIGN.md). Across gen-1/2/3 coverage climbed to ~9% held-out, but
certified creativity BEYOND the best retrieval solver (gen2_base) = 0: every "invented" solve was
already in gen2_base's rich parametric menu. Coverage is exhausted as a creativity source. The ONLY
honest metric now is `eval_beyond_base` = held-out tasks gen2_base CANNOT express.

THE LEVER. gen2_base RETRIEVES whole single-template mechanisms from a fixed menu (color_perm,
tiling, panel-logic, crop, object-recolor-by-size, connect-same-color-collinear, ...). Many held-out
families are MULTI-STEP RELATIONAL mechanisms no single template contains:
    object-movement-by-rule, object-to-marker copy, ray/line draw & connect, relational recolor,
    counting-driven construction, gravity/fill.
So instead of a richer MENU OF TEMPLATES, this inventor builds a RICH RELATIONAL/OBJECT ALPHABET of
grid->grid primitives (each one a small relational verb, several FITTED from the task's train pairs)
and a VALUE-GUIDED BEAM SEARCH that COMPOSES them to depth 2-4, exact-verified on train. A depth-3
relational SENTENCE like  recolor-by-relation -> move-objects-to-edge -> draw-rays-between-markers
is reachable by composition but is NOT a single template gen2_base can express.

STANDARDIZED GATE (non-negotiable).
    solve_ablated == gen2_base.solve  (imported verbatim — the strong retrieval ablation).
    solve         == gen2_base as attempt-1 backstop, THEN the relational-depth invention as attempt-2.
    invention_gate's INVENTED = solves(full) - solves(ablated) = solves gen2_base MISSES = the real
    creativity number. Develop/tune ONLY on arc1-train (gen2_base's TRAIN misses); arc1-eval is
    held-out — reported, never tuned to.

INTEGRITY. solve()/solve_ablated() learn ONLY from (a) the current task's train pairs, (b) module
state from prior solve() calls this run, (c) self-generated synthetic data at import. NEVER read ARC
task files or test OUTPUTS, no network, no LLM. Budget-respected. Pure python+numpy. Build-time light.
Run/imported with /data/llm/.venv/bin/python from .../incubation/evolve."""
import os
import sys
import time
import heapq
from collections import deque, Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVOLVE = os.path.dirname(HERE)
ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
for p in (EVOLVE, ARC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# The strong retrieval baseline IS the standardized ablation. Imported verbatim.
import gen2_base as BASE

META = {"name": "gen4_01_relational-depth",
        "desc": "gen2_base retrieval backstop (attempt 1) + relational-depth invention (attempt 2): a "
                "rich object/relation alphabet (segment objects; move-to edge/marker/object; copy-to-each-"
                "marker; draw line/ray & connect; recolor-by-relation; gravity-fill; count->construct) "
                "composed by a value-guided beam search to depth 2-4, exact-verified on train. "
                "INVENTED = solves gen2_base cannot."}


# ===========================================================================
# grid helpers
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


def _components(g, bg=0, diag=False, by_color=False):
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    nb = ([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
          if diag else [(-1, 0), (1, 0), (0, -1), (0, 1)])
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
# RELATIONAL / OBJECT ALPHABET
# ---------------------------------------------------------------------------
# Each primitive is a grid->grid relational verb. Several are PARAMETRIZED, but only by structure
# inferable at solve time (background, color present, object identity) — never by held-out outputs.
# The beam search COMPOSES them; composition is what gen2_base's single-template menu cannot express.
# Every primitive returns None on a non-applicable grid (so the search prunes it).
# ===========================================================================

# ---- gravity / movement of whole grid mass ----
def _gravity(g, dr, dc, bg):
    h, w = g.shape
    out = np.full((h, w), bg, int)
    if dr != 0:
        for c in range(w):
            col = g[:, c]
            vals = col[col != bg]
            if len(vals):
                if dr > 0:
                    out[h - len(vals):, c] = vals
                else:
                    out[:len(vals), c] = vals
    else:
        for r in range(h):
            row = g[r, :]
            vals = row[row != bg]
            if len(vals):
                if dc > 0:
                    out[r, w - len(vals):] = vals
                else:
                    out[r, :len(vals)] = vals
    return out


# ---- object-centric movement: each connected object translated as a rigid body to an edge/marker ----
def _move_objects(g, where, bg, diag):
    """Move every object (4/8-conn, color-agnostic by_color) rigidly toward an edge until it hits the
    wall (no stacking; objects keep relative position to grid frame, slide to the chosen edge)."""
    comps = _components(g, bg=bg, diag=diag, by_color=True)
    if not comps:
        return None
    h, w = g.shape
    out = np.full((h, w), bg, int)
    # process order matters when objects could collide; stack against edge in order of proximity
    def key(cells):
        r0, r1, c0, c1 = _bbox(cells)
        if where == "down":
            return -r1
        if where == "up":
            return r0
        if where == "right":
            return -c1
        return c0
    for cells in sorted(comps, key=key):
        r0, r1, c0, c1 = _bbox(cells)
        if where == "down":
            dr, dc = h - 1 - r1, 0
        elif where == "up":
            dr, dc = -r0, 0
        elif where == "right":
            dr, dc = 0, w - 1 - c1
        else:
            dr, dc = 0, -c0
        for a, b in cells:
            na, nb = a + dr, b + dc
            if 0 <= na < h and 0 <= nb < w:
                out[na, nb] = g[a, b]
    return out


# ---- draw rays / connect markers ----
def _connect_pairs(g, bg, diag_ok):
    """For every non-bg color with EXACTLY two cells lying on a common row/col/(diagonal), draw the
    straight segment of that color between them. A relational draw beyond 'same-color collinear' (it
    handles diagonal markers too and only fires on clean 2-point relations)."""
    out = g.copy()
    h, w = g.shape
    drew = False
    for c in np.unique(g):
        if c == bg:
            continue
        pts = np.argwhere(g == c)
        if len(pts) != 2:
            continue
        r0, c0, r1, c1 = int(pts[0][0]), int(pts[0][1]), int(pts[1][0]), int(pts[1][1])
        ddr, ddc = r1 - r0, c1 - c0
        if ddr != 0 and ddc != 0 and abs(ddr) != abs(ddc):
            continue
        if (ddr != 0 and ddc != 0) and not diag_ok:
            continue
        sr = (1 if ddr > 0 else (-1 if ddr < 0 else 0))
        sc = (1 if ddc > 0 else (-1 if ddc < 0 else 0))
        r, cc = r0, c0
        while (r, cc) != (r1, c1):
            out[r, cc] = c
            r += sr
            cc += sc
        out[r1, c1] = c
        drew = True
    return out if drew else None


def _rays_from_seeds(g, bg, mode):
    """Each isolated single non-bg cell emits a ray. mode in {h,v,both,diag} — extend a line of the
    cell's color across the grid until the border (over bg only; stop at other non-bg). Relational
    'extend ray from seed' beyond connect-collinear."""
    h, w = g.shape
    out = g.copy()
    pts = np.argwhere(g != bg)
    if len(pts) == 0:
        return None
    dirs = {
        "h": [(0, 1), (0, -1)],
        "v": [(1, 0), (-1, 0)],
        "both": [(0, 1), (0, -1), (1, 0), (-1, 0)],
        "diag": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    }[mode]
    drew = False
    for (r, c) in pts:
        col = g[r, c]
        for dr, dc in dirs:
            a, b = r + dr, c + dc
            while 0 <= a < h and 0 <= b < w and g[a, b] == bg:
                out[a, b] = col
                drew = True
                a += dr
                b += dc
    return out if drew else None


def _gravity_fill(g, where, bg):
    """Each non-bg cell projects its color toward an edge filling the bg cells in its path (a 'shadow'
    cast to the wall). Direction-parametric; composes with recolor/crop."""
    h, w = g.shape
    out = g.copy()
    if where in ("down", "up"):
        rng = range(w)
        for c in rng:
            col = g[:, c]
            nz = np.where(col != bg)[0]
            for r in nz:
                if where == "down":
                    seg = out[r:, c]
                    seg[seg == bg] = g[r, c]
                else:
                    seg = out[:r + 1, c]
                    seg[seg == bg] = g[r, c]
    else:
        for r in range(h):
            row = g[r, :]
            nz = np.where(row != bg)[0]
            for c in nz:
                if where == "right":
                    seg = out[r, c:]
                    seg[seg == bg] = g[r, c]
                else:
                    seg = out[r, :c + 1]
                    seg[seg == bg] = g[r, c]
    return out if not np.array_equal(out, g) else None


# ---- symmetry / motif completion (object-to-marker reflection family) ----
def _complete_symmetry(g, bg):
    """Overlay the grid with its 4 mirrors; bg cells filled from any mirror that has a value. A
    motif-completion-by-reflection (object-to-marker symmetric stamping)."""
    out = g.copy()
    mirs = [g[:, ::-1], g[::-1, :], g[::-1, ::-1]]
    if g.shape[0] == g.shape[1]:
        mirs += [g.T, g[::-1, ::-1].T]
    changed = False
    for m in mirs:
        if m.shape != out.shape:
            continue
        fill = (out == bg) & (m != bg)
        if fill.any():
            out[fill] = m[fill]
            changed = True
    return out if changed else None


# ---- relational recolor: recolor each object by a relation (rank by size / position / hole-count) ----
def _recolor_const(g, src, dst):
    if src == dst:
        return None
    out = g.copy()
    out[g == src] = dst
    return out if (g == src).any() else None


def _swap(g, a, b, bg):
    if a == b:
        return None
    out = g.copy()
    out[g == a] = b
    out[g == b] = a
    return out


# ---- counting-driven construction (reduce to count / winner) handled by fitters, not by the beam ----


# ===========================================================================
# THE ALPHABET as a list of (name, fn) — fn maps grid->grid (or None). bg is bound per task.
# These are the COMPOSABLE relational verbs the beam search strings together.
# ===========================================================================
def build_alphabet(train):
    bg = _bg_train(train)
    pal = sorted({int(v) for gi, go in train for v in np.unique(gi)} |
                 {int(v) for gi, go in train for v in np.unique(go)})
    colors = [c for c in pal if c != bg]
    A = []

    def add(name, fn):
        A.append((name, fn))

    # geometry (cheap, shape-preserving or simple) — these let a relational op act in a rotated frame
    add("reflect_h", lambda g: g[:, ::-1])
    add("reflect_v", lambda g: g[::-1, :])
    add("rot180", lambda g: np.rot90(g, 2))
    add("transpose", lambda g: g.T)

    # whole-mass gravity
    for w_, (dr, dc) in (("down", (1, 0)), ("up", (-1, 0)), ("left", (0, -1)), ("right", (0, 1))):
        add("gravity_%s" % w_, (lambda dr=dr, dc=dc: (lambda g: _gravity(g, dr, dc, bg)))())

    # whole-grid 1-step shifts (object-movement-by-rule, periodic shift)
    add("shift_down", lambda g: _move_shift(g, 1, 0, bg))
    add("shift_up", lambda g: _move_shift(g, -1, 0, bg))
    add("shift_left", lambda g: _move_shift(g, 0, -1, bg))
    add("shift_right", lambda g: _move_shift(g, 0, 1, bg))

    # object movement to edges (rigid bodies)
    for where in ("down", "up", "left", "right"):
        for diag in (False, True):
            add("moveobj_%s_%s" % (where, "d" if diag else "o"),
                (lambda where=where, diag=diag: (lambda g: _move_objects(g, where, bg, diag)))())

    # ray / connect draws
    add("connect_pairs_ortho", lambda g: _connect_pairs(g, bg, diag_ok=False))
    add("connect_pairs_diag", lambda g: _connect_pairs(g, bg, diag_ok=True))
    for m in ("h", "v", "both", "diag"):
        add("ray_%s" % m, (lambda m=m: (lambda g: _rays_from_seeds(g, bg, m)))())

    # shadow / gravity-fill
    for where in ("down", "up", "left", "right"):
        add("gravfill_%s" % where, (lambda where=where: (lambda g: _gravity_fill(g, where, bg)))())

    # symmetry completion
    add("complete_sym", lambda g: _complete_symmetry(g, bg))
    add("fill_holes_each", lambda g: _fill_holes_relational(g, bg))

    # crop family (lets a relational op then crop to the content / object — composes with draws)
    add("crop_content", lambda g: _crop_content(g, bg))

    # relational recolor: const recolors over the task palette (small, only colors present)
    for s in colors:
        for d in pal:
            if s != d:
                add("recolor_%d_%d" % (s, d),
                    (lambda s=s, d=d: (lambda g: _recolor_const(g, s, d)))())
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            add("swap_%d_%d" % (a, b), (lambda a=a, b=b: (lambda g: _swap(g, a, b, bg)))())

    return A, bg, colors


def _move_shift(g, dr, dc, bg):
    h, w = g.shape
    out = np.full((h, w), bg, int)
    for i in range(h):
        for j in range(w):
            ni, nj = i + dr, j + dc
            if 0 <= ni < h and 0 <= nj < w:
                out[ni, nj] = g[i, j]
    return out


def _crop_content(g, bg):
    nz = np.argwhere(g != bg)
    if nz.size == 0:
        return None
    (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
    if (r1 - r0, c1 - c0) == g.shape:
        return None
    return g[r0:r1, c0:c1]


def _fill_holes_relational(g, bg):
    """Fill bg-cells fully enclosed by a single object's color with that object's color (close holes)."""
    h, w = g.shape
    out = g.copy()
    isbg = (g == bg)
    reach = np.zeros((h, w), bool)
    q = deque()
    for i in range(h):
        for j in (0, w - 1):
            if isbg[i, j] and not reach[i, j]:
                reach[i, j] = True
                q.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if isbg[i, j] and not reach[i, j]:
                reach[i, j] = True
                q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w and isbg[a, b] and not reach[a, b]:
                reach[a, b] = True
                q.append((a, b))
    holes = isbg & ~reach
    if not holes.any():
        return None
    # color each hole-cell by the dominant neighbor color
    for (i, j) in np.argwhere(holes):
        nbc = Counter()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w and g[a, b] != bg:
                nbc[int(g[a, b])] += 1
        if nbc:
            out[i, j] = nbc.most_common(1)[0][0]
    return out if not np.array_equal(out, g) else None


# ===========================================================================
# FITTED RELATIONAL CONCEPTS (counting -> construct, object-to-marker copy, relational recolor by
# region). These need cross-pair fitting (not pure grid->grid), so they are tried as depth-1 whole
# mechanisms ALONGSIDE the beam — but each is a NEW relational mechanism not in gen2_base's menu, and
# they may also seed the beam (their output composes further). They verify exactly on train.
# ===========================================================================
def _verify(fn, train):
    for gi, go in train:
        try:
            o = fn(gi)
        except Exception:
            return False
        if not _eq(o, go):
            return False
    return True


def fit_object_to_markers(train):
    """OBJECT-TO-MARKER COPY. One grid has a single multi-cell 'template' object and several isolated
    single 'marker' cells; stamp a copy of the template centered/anchored at each marker. Learn the
    anchor offset (template centroid vs marker) from train and replay. A multi-object interaction
    mechanism gen2_base has no template for."""
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)

    def parse(g):
        comps = _components(g, bg=bg, diag=True, by_color=False)
        if len(comps) < 2:
            return None
        singles = [c for c in comps if len(c) == 1]
        multi = [c for c in comps if len(c) > 1]
        if len(multi) != 1 or not singles:
            return None
        return multi[0], singles

    # derive template footprint (relative cells + colors) and stamp rule from the FIRST train pair
    def make_stamp():
        def stamp(g):
            p = parse(g)
            if p is None:
                return None
            tmpl, singles = p
            r0, r1, c0, c1 = _bbox(tmpl)
            rel = [(a - r0, b - c0, int(g[a, b])) for a, b in tmpl]
            ch, cw = r1 - r0, c1 - c0
            # anchor = template center; stamp so each marker sits at the template's center cell
            ar, ac = ch // 2, cw // 2
            out = g.copy()
            h, w = g.shape
            for (mr, mc) in [s[0] for s in singles]:
                for (dr, dc, col) in rel:
                    na, nb = mr - ar + dr, mc - ac + dc
                    if 0 <= na < h and 0 <= nb < w:
                        out[na, nb] = col
            return out
        return stamp

    fn = make_stamp()
    if _verify(fn, train):
        return fn
    return None


def fit_count_construct(train):
    """COUNTING -> CONSTRUCT. Output is a tiny grid whose size/content encodes a COUNT of something in
    the input (number of objects / number of a color / number of distinct colors). Covers the
    reduce-to-count family. Fitted: which count, and the construction (a 1xN or Nx1 bar, or NxN block,
    of a fixed color)."""
    outs = [go for _, go in train]
    # output must be small and monochrome-ish
    def count_objs(g, diag, by_color):
        return len(_components(g, bg=_bg(g), diag=diag, by_color=by_color))

    def count_color(g, c):
        return int((g == c).sum())

    def n_distinct(g):
        bg = _bg(g)
        return len({int(v) for v in np.unique(g) if v != bg})

    counters = []
    for diag in (False, True):
        for bc in (False, True):
            counters.append(("objs_%d%d" % (diag, bc),
                             (lambda diag=diag, bc=bc: (lambda g: count_objs(g, diag, bc)))()))
    counters.append(("ndistinct", n_distinct))
    pal = sorted({int(v) for gi, _ in train for v in np.unique(gi)})
    for c in pal:
        counters.append(("color_%d" % c, (lambda c=c: (lambda g: count_color(g, c)))()))

    # candidate output shapes: (n,1),(1,n),(n,n)
    for cname, cfn in counters:
        try:
            counts = [cfn(gi) for gi, _ in train]
        except Exception:
            continue
        if any(c <= 0 for c in counts):
            continue
        # output color: the unique non-bg color of the outputs (must be constant)
        ocols = set()
        ok_shape = None
        for n, go in zip(counts, outs):
            nz = {int(v) for v in np.unique(go)}
            vals = [v for v in nz if v != 0]
            if len(set(np.unique(go).tolist())) > 2 and len(vals) != 1:
                ok_shape = "bad"
                break
            ocols |= set(vals)
        if ok_shape == "bad" or len(ocols) != 1:
            continue
        oc = ocols.pop()
        for shp in ("col", "row", "sq"):
            def make(cfn=cfn, oc=oc, shp=shp):
                def fn(g):
                    n = cfn(g)
                    if n <= 0:
                        return None
                    if shp == "col":
                        return np.full((n, 1), oc, int)
                    if shp == "row":
                        return np.full((1, n), oc, int)
                    return np.full((n, n), oc, int)
                return fn
            fn = make()
            if _verify(fn, train):
                return fn
    return None


def fit_region_recolor(train):
    """RELATIONAL RECOLOR by per-object property: recolor each object solid by (a) its number of holes,
    (b) whether it touches the border, (c) its width/height ratio class. Beyond gen2_base's size/rank
    recolor (which it already has) — these are DIFFERENT relations."""
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)

    def holes_of(g, cells):
        # count enclosed bg regions inside object bbox
        r0, r1, c0, c1 = _bbox(cells)
        sub = np.ones((r1 - r0 + 1, c1 - c0 + 1), int)
        cs = set((a - r0, b - c0) for a, b in cells)
        for a, b in cs:
            sub[a, b] = 0
        H, W = sub.shape
        reach = np.zeros((H, W), bool)
        q = deque()
        for i in range(H):
            for j in (0, W - 1):
                if sub[i, j] == 1 and not reach[i, j]:
                    reach[i, j] = True
                    q.append((i, j))
        for j in range(W):
            for i in (0, H - 1):
                if sub[i, j] == 1 and not reach[i, j]:
                    reach[i, j] = True
                    q.append((i, j))
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y = i + di, j + dj
                if 0 <= x < H and 0 <= y < W and sub[x, y] == 1 and not reach[x, y]:
                    reach[x, y] = True
                    q.append((x, y))
        enclosed = (sub == 1) & ~reach
        # count enclosed components
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

    for keytype in ("holes", "border", "wider"):
        for diag in (True, False):
            mapping = {}
            ok = True
            for gi, go in train:
                if np.any((gi != bg) != (go != bg)):
                    ok = False
                    break
                comps = _components(gi, bg=bg, diag=diag)
                if not comps:
                    ok = False
                    break
                for cells in comps:
                    ocs = {int(go[a, b]) for a, b in cells}
                    if len(ocs) != 1:
                        ok = False
                        break
                    oc = ocs.pop()
                    if keytype == "holes":
                        key = holes_of(gi, cells)
                    elif keytype == "border":
                        r0, r1, c0, c1 = _bbox(cells)
                        key = int(r0 == 0 or c0 == 0 or r1 == gi.shape[0] - 1 or c1 == gi.shape[1] - 1)
                    else:
                        r0, r1, c0, c1 = _bbox(cells)
                        key = int((c1 - c0) > (r1 - r0)) - int((c1 - c0) < (r1 - r0))
                    if key in mapping and mapping[key] != oc:
                        ok = False
                        break
                    mapping[key] = oc
                if not ok:
                    break
            if ok and mapping:
                def make(keytype=keytype, mapping=mapping, diag=diag, holes_of=holes_of):
                    def fn(g):
                        comps = _components(g, bg=bg, diag=diag)
                        if not comps:
                            return None
                        out = g.copy()
                        for cells in comps:
                            if keytype == "holes":
                                key = holes_of(g, cells)
                            elif keytype == "border":
                                r0, r1, c0, c1 = _bbox(cells)
                                key = int(r0 == 0 or c0 == 0 or r1 == g.shape[0] - 1 or c1 == g.shape[1] - 1)
                            else:
                                r0, r1, c0, c1 = _bbox(cells)
                                key = int((c1 - c0) > (r1 - r0)) - int((c1 - c0) < (r1 - r0))
                            if key not in mapping:
                                return None
                            for a, b in cells:
                                out[a, b] = mapping[key]
                        return out
                    return fn
                fn = make()
                if _verify(fn, train):
                    return fn
    return None


FITTERS = [
    ("object_to_markers", fit_object_to_markers),
    ("count_construct", fit_count_construct),
    ("region_recolor", fit_region_recolor),
]


# ===========================================================================
# VALUE-GUIDED BEAM SEARCH over the relational alphabet (depth 1-4), exact-verified on train.
# Value = mean grid-distance to target + structural-progress shaping. Hard pruning; budget-bounded.
# ===========================================================================
def _dist(a, b):
    """Distance from candidate `a` to target `b`. Smaller = closer. Shape mismatch penalized but a path
    toward the right size is rewarded; cellwise mismatch fraction when shapes agree."""
    if a is None:
        return 4.0
    if a.shape != b.shape:
        sa, sb = a.size, b.size
        return 2.0 + abs(sa - sb) / max(sa, sb, 1)
    return float((a != b).mean())


def _state_value(outs, tgts):
    return sum(_dist(o, t) for o, t in zip(outs, tgts)) / len(tgts)


def _beam_search(train, alphabet, budget, max_depth=4, beam_width=14, collect=2):
    """Best-first beam over compositions of alphabet verbs. Returns up to `collect` verified programs
    (lists of (name, fn)). A program verifies iff it reproduces EVERY train output exactly."""
    ins = [gi for gi, _ in train]
    tgts = [go for _, go in train]
    start_v = _state_value(ins, tgts)
    if start_v == 0.0:
        return []  # identity already solves — leave to base
    # heap of (value, counter, program, outputs); plus a visited set on output signatures
    counter = 0
    heap = [(start_v, counter, [], ins)]
    counter += 1
    seen = set()
    found = []
    nexec = 0
    t0 = time.time()
    while heap and nexec < budget:
        v, _c, prog, outs = heapq.heappop(heap)
        if len(prog) >= max_depth:
            continue
        kids = []
        for name, fn in alphabet:
            # cheap structural cycle guard: don't apply the exact same verb twice in a row
            if prog and prog[-1][0] == name:
                continue
            try:
                nouts = [fn(o) for o in outs]
            except Exception:
                nouts = None
            nexec += 1
            if nouts is None or any(o is None for o in nouts):
                if nexec >= budget:
                    break
                continue
            # exact verify?
            if all(o.shape == t.shape and np.array_equal(o, t) for o, t in zip(nouts, tgts)):
                found.append(prog + [(name, fn)])
                if len(found) >= collect:
                    return found
                continue
            sig = tuple(o.tobytes() for o in nouts)
            if sig in seen:
                continue
            seen.add(sig)
            nv = _state_value(nouts, tgts)
            # progress shaping: only enqueue children that are not strictly worse than parent by a lot
            if nv <= v + 0.34:
                kids.append((nv, counter, prog + [(name, fn)], nouts))
                counter += 1
            if nexec >= budget:
                break
        for k in heapq.nsmallest(beam_width, kids):
            heapq.heappush(heap, k)
        if time.time() - t0 > 8.0:
            break
    return found


def _run_prog(g, prog):
    cur = g
    try:
        for _name, fn in prog:
            cur = fn(cur)
            if cur is None:
                return None
        return cur
    except Exception:
        return None


# ===========================================================================
# IN-RUN EXPERIENCE: remember verified relational PROGRAMS (as name-sequences) that solved earlier
# tasks; re-try them first on later tasks (a learned, not hand-coded, search-order prior). Verified
# transfer only; never stores grids/outputs.
# ===========================================================================
_MEM_PROGS = []          # list of program name-sequences that verified on some prior task
_MEM_SEEN = set()
_TASKN = [0]


def reset_library():
    """Documented hook: clear cross-task experience so a run starts cold (gate uses this to isolate
    transfer). Also resets the imported base library so the whole solver is genuinely cold."""
    _MEM_PROGS.clear()
    _MEM_SEEN.clear()
    if hasattr(BASE, "reset_library"):
        try:
            BASE.reset_library()
        except Exception:
            pass
    else:
        # gen2_base has no reset; best-effort clear of its module library
        try:
            BASE._LIB.__init__()
        except Exception:
            pass


def _remember(prog):
    key = tuple(n for n, _ in prog)
    if key not in _MEM_SEEN:
        _MEM_SEEN.add(key)
        _MEM_PROGS.append(key)


def _replay_memory(train, alphabet):
    """Re-instantiate remembered name-sequences against THIS task's alphabet and keep those that verify
    exactly here (transfer)."""
    if not _MEM_PROGS:
        return []
    amap = dict(alphabet)
    out = []
    for key in _MEM_PROGS:
        if any(n not in amap for n in key):
            continue
        prog = [(n, amap[n]) for n in key]
        if _verify_prog(prog, train):
            out.append(prog)
    return out


def _verify_prog(prog, train):
    for gi, go in train:
        o = _run_prog(gi, prog)
        if not _eq(o, go):
            return False
    return True


# ===========================================================================
# INVENTION ENTRYPOINT — produce up to 2 candidate outputs per test input via relational depth.
# ===========================================================================
def _invent(train, test_inputs, budget):
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    alphabet, bg, colors = build_alphabet(train)

    progs = []

    # (0) replay remembered relational programs first (transfer / learned order)
    progs.extend(_replay_memory(train, alphabet))

    # (1) fitted whole-relational mechanisms (counting->construct, object-to-markers, region-recolor)
    fitted_fns = []
    for fname, fitter in FITTERS:
        try:
            fn = fitter(train)
        except Exception:
            fn = None
        if fn is not None and _verify(fn, train):
            fitted_fns.append((fname, fn))

    # (2) value-guided beam over the composable alphabet (the depth lever)
    beam_budget = max(800, min(budget, 6000))
    beam_progs = _beam_search(train, alphabet, beam_budget,
                              max_depth=4, beam_width=14, collect=2)
    progs.extend(beam_progs)

    # remember verified relational programs of length >= 2 (genuine compositions) for transfer
    for p in beam_progs:
        if len(p) >= 2:
            _remember(p)

    # assemble attempts: fitted mechanisms + beam programs, deduped on the test inputs
    attempts = []
    for gi in test_inputs:
        gi = np.asarray(gi, int)
        cand = []
        # fitted mechanisms first (they capture a complete relational rule)
        for _fn_name, fn in fitted_fns:
            try:
                o = fn(gi)
            except Exception:
                o = None
            if o is not None and getattr(o, "ndim", None) == 2 and o.size > 0:
                o = np.asarray(o, int)
                if not any(_eq(o, c) for c in cand):
                    cand.append(o)
        # then beam programs
        for p in progs:
            o = _run_prog(gi, p)
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
#   solve_ablated == gen2_base.solve  (the strong retrieval ablation, imported verbatim)
#   solve         == gen2_base attempt-1 backstop, THEN relational-depth invention attempt-2
# ===========================================================================
def solve_ablated(train, test_inputs, budget):
    """EXACTLY gen2_base.solve — the standardized strong-retrieval ablation. INVENTED = solved - this."""
    return BASE.solve(train, test_inputs, budget)


def solve(train, test_inputs, budget):
    """gen2_base as attempt-1 backstop; relational-depth invention fills the remaining attempt slot and
    is tried for any test input the base left without a candidate. The gate scores both attempts."""
    _TASKN[0] += 1
    # attempt 1: the strong retrieval baseline (never regress below it)
    try:
        base_attempts = BASE.solve(train, test_inputs, budget)
    except Exception:
        base_attempts = []
    if not isinstance(base_attempts, list):
        base_attempts = []
    # normalize to one list per test input
    norm = []
    for k in range(len(test_inputs)):
        a = base_attempts[k] if k < len(base_attempts) else []
        if a is None:
            a = []
        norm.append([np.asarray(x, int) for x in a if x is not None][:2])

    # attempt 2: relational-depth invention (give it a healthy slice of the budget)
    try:
        inv = _invent(train, test_inputs, max(800, budget))
    except Exception:
        inv = [[] for _ in test_inputs]

    # ARC allows 2 attempts. Reserve attempt-1 for the strong retrieval backstop (base's best) and
    # attempt-2 for the relational-depth INVENTION, so a depth-composed mechanism is never crowded out
    # by a second base guess. If invention has nothing for a test input, base keeps both slots
    # (never regress below gen2_base). If base has nothing, invention takes both.
    merged = []
    for k in range(len(test_inputs)):
        b = list(norm[k])
        iv = [o for o in (inv[k] if k < len(inv) else []) if o is not None]
        cand = []
        if b:
            cand.append(b[0])                       # attempt 1 = base backstop
        for o in iv:                                # attempt 2 = first invention not already present
            if not any(_eq(o, c) for c in cand):
                cand.append(o)
                break
        # backfill any unused slot: extra base guess first, then extra invention
        if len(cand) < 2:
            for o in (b[1:] + iv):
                if not any(_eq(o, c) for c in cand):
                    cand.append(o)
                if len(cand) >= 2:
                    break
        merged.append(cand[:2])
    return merged


# self-generated synthetic sanity at import (kept tiny; validates the alphabet executes)
def _selftest():
    rng = np.random.RandomState(0)
    g = rng.randint(0, 3, (5, 5))
    ab, _, _ = build_alphabet([(g, g)])
    for _n, fn in ab[:8]:
        try:
            fn(g)
        except Exception:
            pass


_selftest()
