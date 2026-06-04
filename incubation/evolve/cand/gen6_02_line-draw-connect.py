#!/usr/bin/env python3
"""GEN-6 RELATION-INDUCER #2 — LINE / DRAW / CONNECT / REGION (relational DRAWING).

THE FRONTIER (CAMPAIGN.md gen-5/6). Composing a FIXED verb alphabet does not engage real-ARC's held-out
frontier; EVERY beyond-retrieval win has come from per-task FITTED cause->effect RELATIONS. gen-5 reached
45/400 held-out (11 certified beyond gen2_base) with systematic relation-induction. The two BIGGEST
untouched miss-families are counting/construction (86) and LINE/DRAW-CONNECT (77) — roughly half of
gen2_base's misses. This module sweeps the LINE/DRAW/CONNECT/REGION family.

THE FAMILY = relational DRAWING (all 77 train misses are SAME-SHAPE — draw onto the grid):
  * RAYS: from each seed, emit a ray in an INDUCED direction (constant across train = invariance), of an
    INDUCED color, until the border OR until it hits another non-bg cell.
  * CONNECT: join marker PAIRS (same color, or aligned) with a line of an INDUCED color — orthogonal AND
    diagonal, endpoints kept (gen2_base's connect_dots is orthogonal-only, marker-color-only, no fitted
    fill, joins ALL collinear; here the relation is fitted).
  * OUTLINE / REGION: hollow a solid filled shape to its boundary, or fill a bounded region's interior
    with an INDUCED color (draw the boundary / flood the inside).
  * GRAVITY / MOVE-TO: translate each object in an INDUCED direction until it is adjacent to a target
    (another object / the border), leaving a trail or not (induced).

METHOD (gen-5's, applied to this harder, path-based family):
  decompose -> rich per-part feature vector (candidate CAUSES: seed color/shape/position, pair alignment,
  region solidity/hole, object color/size) -> induce a feature->EFFECT mapping (the DIRECTION / endpoints
  / fill-color) that holds across ALL train pairs (cross-pair invariance licenses CAUSAL induction) ->
  EXACT-verify on train (the held-out test is the intervention). A tiny learned feature-relevance prior
  (trained at import on self-generated synthetic drawing tasks, <90s) ORDERS which inducer to try first so
  induction stays tractable; the verifier supplies precision.

STANDARDIZED GATE (non-negotiable):
    solve_ablated == gen2_base.solve (imported verbatim — the strong retrieval ablation).
    solve         == gen2_base as attempt-1 backstop, THEN this relation-induction as attempt-2.
    invention_gate's INVENTED = solved(full) - solved(ablated) = solves gen2_base MISSES.
Develop/tune ONLY on arc1-train; arc1-eval is held-out — reported, never tuned to.

INTEGRITY. solve()/solve_ablated() learn ONLY from (a) the current task's train pairs, (b) module-level
state from prior solve() calls this run, (c) self-generated synthetic data at import. NEVER read ARC task
files or test OUTPUTS, no network, no LLM. Budget-respected. Pure python+numpy. Build-time light.
Run/imported with /data/llm/.venv/bin/python from .../incubation/evolve."""
import os
import sys
import time
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

META = {"name": "gen6_02_line-draw-connect",
        "desc": "gen2_base retrieval backstop (attempt 1) + LINE/DRAW/CONNECT/REGION relation-induction "
                "(attempt 2): rays from seeds in an induced direction until border/object; connect marker "
                "pairs (orthogonal+diagonal) with induced fill+kept endpoints; outline a solid shape / "
                "fill a bounded region with induced color; gravity-translate objects toward a target. "
                "Direction/endpoints/fill-color induced from train (invariance) and exact-verified. A "
                "learned relevance prior orders inducers. INVENTED = solves gen2_base cannot."}

DIRS8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ===========================================================================
# grid helpers
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


def _components(g, bg=0, diag=False, by_color=False):
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    nb = DIRS8 if diag else DIRS4
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


def _verify(fn, train):
    for gi, go in train:
        try:
            o = fn(gi)
        except Exception:
            return False
        if not _eq(o, go):
            return False
    return True


# ===========================================================================
# RELATION-INDUCER 1 — RAY FROM SEED in an INDUCED direction.
# Decompose into single 'seed' cells (8-conn singletons). The CAUSE = a seed; the relational EFFECT =
# emit a ray (a line of cells) from the seed in a DIRECTION that is INDUCED from train and held CONSTANT
# (invariance), of an INDUCED color (the seed's own color, or a fitted constant, or per-seed-color map),
# extending until the grid BORDER or until it would hit a pre-existing non-bg cell (induced stop rule).
# The direction may be GLOBAL (all seeds shoot the same way) or PER-SEED-COLOR (each color a fixed dir).
# This is a relation gen2_base's fixed menu cannot express (it has no ray primitive).
# ===========================================================================
def _ray_walk(gi, sr, sc, dr, dc, bg):
    """Walk from the seed along (dr,dc), painting over bg only, stopping AT the border OR a pre-existing
    non-bg obstacle (exclusive). Returns the list of bg cells the ray would paint."""
    h, w = gi.shape
    out = []
    a, b = sr + dr, sc + dc
    while 0 <= a < h and 0 <= b < w:
        if gi[a, b] != bg:
            break
        out.append((a, b))
        a += dr
        b += dc
    return out


def fit_ray_from_seed(train):
    """Each seed (a single isolated non-bg cell) emits ray(s). Two INDUCED schemes, both with the
    direction-set / colour held CONSTANT across train (invariance), exact-verified:
      (A) SET-mode: ALL seeds shoot the SAME fixed SET of directions (covers N/S/E/W crosses and the
          4-diagonal X), painting with the seed's own colour, until border/obstacle.
      (B) PER-COLOUR-mode: each seed colour maps to its own fixed set of directions (different colours
          shoot different ways), painting with the seed's own colour.
    The direction-SET is INDUCED from the first informative train pair (which of the 8 dirs, walked to
    border, lands entirely inside the drawn cells) and then verified on ALL pairs."""
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)
    for gi, go in train:
        if np.any((gi != bg) & (go == bg)):      # purely additive (a drawing op)
            return None

    def seeds_of(g):
        comps = _components(g, bg=bg, diag=True, by_color=False)
        return [(c[0], int(g[c[0][0], c[0][1]])) for c in comps if len(c) == 1]

    # Candidate direction sets per seed colour, induced from drawn cells. A dir is admissible for a seed
    # iff its full ray-to-border is a subset of that pair's drawn cells AND the painted colour == seed's.
    percolor_dirs = {}     # colour -> set of admissible dirs (intersection across all its seeds/pairs)
    global_dirs = None
    saw = False
    for gi, go in train:
        seeds = seeds_of(gi)
        if not seeds:
            return None
        drawn = set((int(a), int(b)) for a, b in np.argwhere((go != bg) & (gi == bg)))
        if not drawn:
            continue
        saw = True
        for (sr, sc), col in seeds:
            admis = set()
            for (dr, dc) in DIRS8:
                cells = _ray_walk(gi, sr, sc, dr, dc, bg)
                if not cells:
                    continue
                if all((a, b) in drawn for (a, b) in cells) and \
                        all(int(go[a, b]) == col for (a, b) in cells):
                    admis.add((dr, dc))
            # intersect across seeds of this colour
            if col in percolor_dirs:
                percolor_dirs[col] &= admis
            else:
                percolor_dirs[col] = set(admis)
            global_dirs = set(admis) if global_dirs is None else (global_dirs & admis)
    if not saw:
        return None

    def build(dir_of):
        def fn(g, dir_of=dir_of):
            out = g.copy()
            seeds = seeds_of(g)
            if not seeds:
                return None
            for (sr, sc), col in seeds:
                dirs = dir_of(col)
                if dirs is None:
                    return None
                for (dr, dc) in dirs:
                    for (a, b) in _ray_walk(g, sr, sc, dr, dc, bg):
                        out[a, b] = col
            return out
        return fn

    # (A) SET-mode: every seed shoots the global admissible set.
    if global_dirs:
        fnA = build(lambda col, gd=frozenset(global_dirs): gd)
        if _verify(fnA, train):
            return fnA
    # (B) PER-COLOUR-mode.
    if percolor_dirs and all(percolor_dirs.values()):
        pcd = {c: frozenset(d) for c, d in percolor_dirs.items()}
        fnB = build(lambda col, pcd=pcd: pcd.get(col))
        if _verify(fnB, train):
            return fnB
    return None


# ===========================================================================
# RELATION-INDUCER 2 — CONNECT MARKER PAIRS (orthogonal + DIAGONAL) with INDUCED fill.
# Decompose into single marker cells. CAUSE = a PAIR of markers that are collinear on a row, column, OR
# DIAGONAL. EFFECT = fill the strictly-between segment with an INDUCED color (the markers' color, a
# fitted GLOBAL constant, or a per-pair-color map). Endpoints kept. Beyond gen2_base's connect_dots:
# (a) diagonals, (b) fitted fill color != marker color, (c) optionally only SAME-color pairs.
# ===========================================================================
def fit_connect_pairs(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)
    for gi, go in train:
        if np.any((gi != bg) & (go == bg)):
            return None

    def pairs_of(g, same_color):
        """Yield (color, p0, p1, seg_cells) for marker pairs collinear on row/col/diag with NO non-bg
        cell strictly between (a clean 2-endpoint relation). Endpoints are ISOLATED single cells (8-conn
        singletons) — connecting big objects is left to other inducers — and capped for tractability."""
        h, w = g.shape
        nz = np.argwhere(g != bg)
        pts = []
        for a, b in nz:
            a = int(a); b = int(b)
            iso = True
            for di, dj in DIRS8:
                x, y = a + di, b + dj
                if 0 <= x < h and 0 <= y < w and g[x, y] != bg:
                    iso = False
                    break
            if iso:
                pts.append((a, b))
        out = []
        n = len(pts)
        if n > 60:          # too many markers -> O(n^2) blowup, and not a clean pair relation
            return out
        for i in range(n):
            for j in range(i + 1, n):
                (r0, c0), (r1, c1) = pts[i], pts[j]
                ci = int(g[r0, c0]); cj = int(g[r1, c1])
                if same_color and ci != cj:
                    continue
                dr = r1 - r0
                dc = c1 - c0
                if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
                    continue
                steps = max(abs(dr), abs(dc))
                if steps < 2:
                    continue
                sr = (dr > 0) - (dr < 0)
                sc = (dc > 0) - (dc < 0)
                seg = [(r0 + sr * k, c0 + sc * k) for k in range(1, steps)]
                if any(g[a, b] != bg for a, b in seg):
                    continue
                out.append((ci, (r0, c0), (r1, c1), seg))
        return out

    for same_color in (True, False):
        # induce fill: global const, per-pair-color map, or marker-color
        global_fill = set()
        permap = {}
        ok_global = ok_permap = ok_markcol = True
        saw = False
        for gi, go in train:
            for (c, p0, p1, seg) in pairs_of(gi, same_color):
                if not seg:
                    continue
                vals = {int(go[a, b]) for a, b in seg}
                if len(vals) != 1:
                    ok_global = ok_permap = ok_markcol = False
                    break
                fv = vals.pop()
                saw = True
                global_fill.add(fv)
                if c in permap and permap[c] != fv:
                    ok_permap = False
                permap[c] = fv
                if fv != c:
                    ok_markcol = False
            if not (ok_global or ok_permap or ok_markcol):
                break
        if not saw:
            continue
        cands = []
        if ok_markcol:
            cands.append(("markcol", None, None))
        if ok_global and len(global_fill) == 1:
            cands.append(("global", global_fill.copy().pop(), None))
        if ok_permap and permap:
            cands.append(("permap", None, dict(permap)))
        for tag, gfill, pmap in cands:
            def fn(g, tag=tag, gfill=gfill, pmap=pmap, same_color=same_color):
                out = g.copy()
                for (c, p0, p1, seg) in pairs_of(g, same_color):
                    if tag == "markcol":
                        fv = c
                    elif tag == "global":
                        fv = gfill
                    else:
                        if c not in pmap:
                            return None
                        fv = pmap[c]
                    for (a, b) in seg:
                        out[a, b] = fv
                return out
            if _verify(fn, train):
                return fn
    return None


# ===========================================================================
# RELATION-INDUCER 3 — OUTLINE a solid shape / FILL a bounded region (REGION granularity).
# Decompose into multi-cell objects. Two inverse relations, each induced + verified:
#   (A) HOLLOW: a solid filled object -> keep only its boundary cells (interior -> bg, or -> a fitted
#       color). The object's interior is the bounded REGION; we draw its boundary.
#   (B) FILL: an object that ENCLOSES a bg region -> flood that interior region with an INDUCED color
#       (fill the bounded region). Border/outside untouched.
# These express 'region boundary / interior' relations gen2_base's menu lacks.
# ===========================================================================
def _interior_mask(g, cells, bg):
    """Boolean mask (grid-shaped) of bg cells ENCLOSED by this object (not reachable from outside the
    object's bbox via 4-conn through bg)."""
    h, w = g.shape
    r0, r1, c0, c1 = _bbox(cells)
    cellset = set(cells)
    H, W = r1 - r0 + 1, c1 - c0 + 1
    free = np.zeros((H, W), bool)   # bg cells inside bbox
    for i in range(H):
        for j in range(W):
            if (i + r0, j + c0) not in cellset:
                free[i, j] = True
    reach = np.zeros((H, W), bool)
    q = deque()
    for i in range(H):
        for j in (0, W - 1):
            if free[i, j] and not reach[i, j]:
                reach[i, j] = True; q.append((i, j))
    for j in range(W):
        for i in (0, H - 1):
            if free[i, j] and not reach[i, j]:
                reach[i, j] = True; q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in DIRS4:
            x, y = i + di, j + dj
            if 0 <= x < H and 0 <= y < W and free[x, y] and not reach[x, y]:
                reach[x, y] = True; q.append((x, y))
    mask = np.zeros((h, w), bool)
    for i in range(H):
        for j in range(W):
            if free[i, j] and not reach[i, j]:
                mask[i + r0, j + c0] = True
    return mask


def _boundary_cells(cells):
    """Cells of the object that lie on the outer boundary of its solid shape (4-neighbour touches a
    non-object cell). For a solid rectangle this is the perimeter ring."""
    cellset = set(cells)
    out = []
    for (a, b) in cells:
        if any((a + di, b + dj) not in cellset for di, dj in DIRS4):
            out.append((a, b))
    return out


def fit_region_op(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)

    # ---- (A) HOLLOW: solid object interior -> bg (or fitted color), boundary kept ----
    for inner_target in ("bg", "fit"):
        inner_const = set()
        ok = True
        saw = False
        for gi, go in train:
            comps = _components(gi, bg=bg, diag=False, by_color=True)
            tmp = gi.copy()
            for cells in comps:
                interior = [(a, b) for (a, b) in cells if (a, b) not in set(_boundary_cells(cells))]
                if not interior:
                    continue
                saw = True
                for (a, b) in interior:
                    ov = int(go[a, b])
                    if inner_target == "bg":
                        if ov != bg:
                            ok = False
                    else:
                        inner_const.add(ov)
                    tmp[a, b] = ov
                if not ok:
                    break
            if not ok:
                break
            if not np.array_equal(tmp, go):
                ok = False
                break
        if ok and saw and (inner_target == "bg" or len(inner_const) == 1):
            fillv = bg if inner_target == "bg" else inner_const.copy().pop()

            def fnA(g, fillv=fillv, bg=bg):
                out = g.copy()
                for cells in _components(g, bg=bg, diag=False, by_color=True):
                    bnd = set(_boundary_cells(cells))
                    for (a, b) in cells:
                        if (a, b) not in bnd:
                            out[a, b] = fillv
                return out
            if _verify(fnA, train):
                return fnA

    # ---- (B) FILL bounded interior region with an induced color ----
    fill_const = set()
    okB = True
    sawB = False
    for gi, go in train:
        tmp = gi.copy()
        for cells in _components(gi, bg=bg, diag=False, by_color=True):
            mask = _interior_mask(gi, cells, bg)
            if not mask.any():
                continue
            sawB = True
            vals = {int(go[a, b]) for a, b in zip(*np.where(mask))}
            fill_const |= vals
            for a, b in zip(*np.where(mask)):
                tmp[a, b] = int(go[a, b])
        if not np.array_equal(tmp, go):
            okB = False
            break
    if okB and sawB and len(fill_const) == 1:
        fv = fill_const.copy().pop()

        def fnB(g, fv=fv, bg=bg):
            out = g.copy()
            for cells in _components(g, bg=bg, diag=False, by_color=True):
                mask = _interior_mask(g, cells, bg)
                out[mask] = fv
            return out
        if _verify(fnB, train):
            return fnB
    return None


# ===========================================================================
# RELATION-INDUCER 4 — GRAVITY / MOVE-TO-TARGET.
# Decompose into multi-cell objects. The relational CAUSE = the presence of a TARGET (the border in an
# INDUCED direction, or the single distinguished object). EFFECT = translate each (non-target) object in
# the INDUCED direction until it is blocked (border or first cell of another object), optionally leaving
# the original cells as bg (move) — induced and verified. Direction held CONSTANT across train.
# ===========================================================================
def fit_gravity(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)
    # gravity moves objects but conserves the multiset of non-bg colors (a pure translation/stack).
    for gi, go in train:
        if Counter(gi[gi != bg].tolist()) != Counter(go[go != bg].tolist()):
            return None

    for (dr, dc) in DIRS4:
        def fn(g, dr=dr, dc=dc, bg=bg):
            h, w = g.shape
            comps = _components(g, bg=bg, diag=True, by_color=False)
            # order objects so that those closest to the gravity wall move first (stacking)
            def key(cells):
                if dr == 1:
                    return -max(a for a, _ in cells)
                if dr == -1:
                    return min(a for a, _ in cells)
                if dc == 1:
                    return -max(b for _, b in cells)
                return min(b for _, b in cells)
            comps = sorted(comps, key=key)
            out = np.full((h, w), bg)
            occ = np.zeros((h, w), bool)
            for cells in comps:
                shift = 0
                while True:
                    ns = shift + 1
                    ok = True
                    for (a, b) in cells:
                        na, nb = a + dr * ns, b + dc * ns
                        if not (0 <= na < h and 0 <= nb < w) or occ[na, nb]:
                            ok = False
                            break
                    if not ok:
                        break
                    shift = ns
                for (a, b) in cells:
                    na, nb = a + dr * shift, b + dc * shift
                    out[na, nb] = g[a, b]
                    occ[na, nb] = True
            return out
        if _verify(fn, train):
            return fn
    return None


# ===========================================================================
# THE INDUCER REGISTRY + the learned FEATURE-RELEVANCE PRIOR.
# ===========================================================================
INDUCERS = [
    ("ray_from_seed", fit_ray_from_seed),
    ("connect_pairs", fit_connect_pairs),
    ("region_op", fit_region_op),
    ("gravity", fit_gravity),
]
INDUCER_NAMES = [n for n, _ in INDUCERS]


def _task_features(train):
    bg = _bg_train(train)
    f_same = float(all(gi.shape == go.shape for gi, go in train))
    nobj = nsing = ncol_in = ncol_out = added = removed = solid = 0.0
    n = len(train)
    for gi, go in train:
        comps = _components(gi, bg=bg, diag=True)
        nobj += len(comps)
        nsing += sum(1 for c in comps if len(c) == 1)
        ncol_in += len({int(v) for v in np.unique(gi)})
        ncol_out += len({int(v) for v in np.unique(go)})
        added += float(np.sum((gi == bg) & (go != bg)))
        removed += float(np.sum((gi != bg) & (go == bg)))
        # solid: any object whose bbox is fully filled (>=4 cells)
        for c in comps:
            r0, r1, c0, c1 = _bbox(c)
            if (r1 - r0 + 1) * (c1 - c0 + 1) == len(c) and len(c) >= 4:
                solid += 1.0
    return np.array([
        1.0, f_same,
        nobj / n, nsing / n,
        ncol_in / n, ncol_out / n,
        added / n, removed / n,
        solid / n,
        float(nsing / n >= 2.0), float(removed / n > 0.5),
    ], float)


FEAT_DIM = len(_task_features([(np.zeros((3, 3), int), np.zeros((3, 3), int))]))


# ---- synthetic drawing-task generators (one per inducer kind) for the relevance prior ----
def _gen_ray(rng):
    h = rng.randint(8, 14); w = rng.randint(8, 14)
    g = np.zeros((h, w), int)
    d = DIRS8[rng.randint(0, 8)]
    col = int(rng.randint(1, 9))
    out = g.copy()
    placed = 0
    for _ in range(rng.randint(2, 4)):
        r = rng.randint(1, h - 1); c = rng.randint(1, w - 1)
        if g[r, c] != 0:
            continue
        g[r, c] = col; out[r, c] = col
        a, b = r + d[0], c + d[1]
        while 0 <= a < h and 0 <= b < w and g[a, b] == 0:
            out[a, b] = col
            a += d[0]; b += d[1]
        placed += 1
    if placed == 0:
        return None
    return g, out


def _gen_connect(rng):
    h = rng.randint(7, 13); w = rng.randint(7, 13)
    g = np.zeros((h, w), int)
    out = g.copy()
    col = int(rng.randint(1, 9))
    fill = int(rng.randint(1, 9))
    made = False
    for _ in range(rng.randint(1, 3)):
        kind = rng.randint(0, 3)
        if kind == 0:
            r = rng.randint(0, h); a = rng.randint(0, w - 3); b = rng.randint(a + 2, w)
            g[r, a] = col; g[r, b] = col; out[r, a] = col; out[r, b] = col
            out[r, a + 1:b] = fill; made = True
        elif kind == 1:
            cc = rng.randint(0, w); a = rng.randint(0, h - 3); b = rng.randint(a + 2, h)
            g[a, cc] = col; g[b, cc] = col; out[a, cc] = col; out[b, cc] = col
            out[a + 1:b, cc] = fill; made = True
        else:
            L = rng.randint(2, min(h, w))
            r = rng.randint(0, h - L); c = rng.randint(0, w - L)
            g[r, c] = col; g[r + L - 1, c + L - 1] = col
            out[r, c] = col; out[r + L - 1, c + L - 1] = col
            for k in range(1, L - 1):
                out[r + k, c + k] = fill
            made = True
    if not made:
        return None
    return g, out


def _gen_region(rng):
    h = rng.randint(8, 14); w = rng.randint(8, 14)
    g = np.zeros((h, w), int)
    col = int(rng.randint(1, 9))
    inner = int(rng.randint(1, 9))
    while inner == col:
        inner = int(rng.randint(1, 9))
    made = False
    for _ in range(rng.randint(1, 2)):
        H = rng.randint(3, 6); W = rng.randint(3, 6)
        r = rng.randint(0, h - H); c = rng.randint(0, w - W)
        if np.any(g[r:r + H, c:c + W] != 0):
            continue
        g[r:r + H, c:c + W] = col
        made = True
    if not made:
        return None
    out = g.copy()
    # hollow: interior -> inner color (ring kept)
    for cells in _components(g, bg=0, diag=False, by_color=True):
        bnd = set(_boundary_cells(cells))
        for (a, b) in cells:
            if (a, b) not in bnd:
                out[a, b] = inner
    return g, out


def _gen_gravity(rng):
    h = rng.randint(8, 13); w = rng.randint(8, 13)
    g = np.zeros((h, w), int)
    placed = 0
    for _ in range(rng.randint(2, 4)):
        col = int(rng.randint(1, 9))
        r = rng.randint(0, h - 1); c = rng.randint(0, w)
        if c < w and g[r, c] == 0:
            g[r, c] = col; placed += 1
    if placed < 2:
        return None
    # gravity down
    out = np.zeros((h, w), int)
    for c in range(w):
        col_vals = [g[r, c] for r in range(h) if g[r, c] != 0]
        for k, v in enumerate(col_vals):
            out[h - len(col_vals) + k, c] = v
    return g, out


_GENS = {
    "ray_from_seed": _gen_ray,
    "connect_pairs": _gen_connect,
    "region_op": _gen_region,
    "gravity": _gen_gravity,
}


def _train_relevance_prior(n_per=160, seed=0):
    rng = np.random.RandomState(seed)
    X = []; y = []
    idx = {n: k for k, n in enumerate(INDUCER_NAMES)}
    for name in INDUCER_NAMES:
        gen = _GENS[name]
        made = 0; tries = 0
        while made < n_per and tries < n_per * 8:
            tries += 1
            npairs = rng.randint(2, 4)
            pairs = []
            for _ in range(npairs):
                r = None
                for _t in range(5):
                    r = gen(rng)
                    if r is not None:
                        break
                if r is None:
                    break
                pairs.append((r[0], r[1]))
            if len(pairs) < 2:
                continue
            try:
                f = _task_features(pairs)
            except Exception:
                continue
            X.append(f); y.append(idx[name]); made += 1
    if not X:
        return np.zeros((len(INDUCER_NAMES), FEAT_DIM)), np.zeros(FEAT_DIM), np.ones(FEAT_DIM)
    X = np.array(X); y = np.array(y)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    W = np.zeros((len(INDUCER_NAMES), FEAT_DIM))
    lr = 0.3
    rng2 = np.random.RandomState(1)
    N = len(Xs)
    for _e in range(10):
        for i in rng2.permutation(N):
            f = Xs[i]; t = y[i]
            logits = W @ f
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            p[t] -= 1.0
            W -= lr * np.outer(p, f)
    return W, mu, sd


_T0 = time.time()
try:
    _W, _MU, _SD = _train_relevance_prior()
except Exception:
    _W, _MU, _SD = np.zeros((len(INDUCER_NAMES), FEAT_DIM)), np.zeros(FEAT_DIM), np.ones(FEAT_DIM)
_BUILD_SEC = time.time() - _T0


def _rank_inducers(train):
    try:
        f = _task_features(train)
        fs = (f - _MU) / _SD
        scores = _W @ fs
        order = sorted(range(len(INDUCER_NAMES)), key=lambda k: -scores[k])
        return [INDUCERS[k] for k in order]
    except Exception:
        return list(INDUCERS)


# ===========================================================================
# IN-RUN EXPERIENCE — remember which inducer kind verified, to try it first next time.
# ===========================================================================
_HITS = Counter()


def reset_library():
    _HITS.clear()
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
# INVENTION ENTRYPOINT
# ===========================================================================
def _invent(train, test_inputs, budget):
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    ordered = _rank_inducers(train)
    ordered = sorted(ordered, key=lambda nf: -_HITS.get(nf[0], 0))

    fitted = []
    t0 = time.time()
    for name, fitter in ordered:
        if time.time() - t0 > 8.0:
            break
        try:
            fn = fitter(train)
        except Exception:
            fn = None
        if fn is not None and _verify(fn, train):
            fitted.append((name, fn))
            _HITS[name] += 1

    attempts = []
    for gi in test_inputs:
        gi = np.asarray(gi, int)
        cand = []
        for _name, fn in fitted:
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
# ===========================================================================
def solve_ablated(train, test_inputs, budget):
    """EXACTLY gen2_base.solve — the standardized strong-retrieval ablation. INVENTED = solved - this."""
    return BASE.solve(train, test_inputs, budget)


def solve(train, test_inputs, budget):
    """gen2_base as attempt-1 backstop; relation-induction fills the remaining attempt slot."""
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
            cand.append(b[0])                       # attempt 1 = base backstop
        for o in iv:                                # attempt 2 = first invention not already present
            if not any(_eq(o, c) for c in cand):
                cand.append(o)
                break
        if len(cand) < 2:
            for o in (b[1:] + iv):
                if not any(_eq(o, c) for c in cand):
                    cand.append(o)
                if len(cand) >= 2:
                    break
        merged.append(cand[:2])
    return merged


# self-generated synthetic sanity at import
def _selftest():
    rng = np.random.RandomState(0)
    for name, fitter in INDUCERS:
        gen = _GENS[name]
        pairs = []
        for _ in range(3):
            r = gen(rng)
            if r is not None:
                pairs.append(r)
        if len(pairs) >= 2:
            try:
                fitter(pairs)
            except Exception:
                pass


_selftest()
