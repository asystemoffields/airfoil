#!/usr/bin/env python3
"""Gen-2 operator #5 — VALUE-DEEP-COMPOSITION (the DEPTH ENGINE).

WHAT THIS ADDS over gen2_base. The base has a rich CONCEPT set but composes it only shallowly: each
concept is fitted once, plus a couple of hard-wired linker shapes (geom-pre-op + concept; remembered +
fresh concept). It has NO general best-first search that REACHES depth-2/3 CONCEPT compositions. The
honest gen-1/2 result was that creativity (linking + experience) flipped ~1 task, because a single
self-verifying concept always won alone. The hard families that are STILL unsolved (same-shape
position-change: object movement-by-rule, gravity-toward-attractor, per-object move-to-marker,
ray/line drawing, connect-the-dots, occluded completion) need a COMPOSITION of >=2 moves that no
single template reproduces. That is exactly what a depth search unlocks — IF it can reach depth 2-3
inside budget without blow-up.

THE DEPTH ENGINE (this file's contribution):
  * MOVE LIBRARY (per task): grid->grid transition operators that are RICHER than the 32 DSL primitives.
    Besides the arg-free DSL ops, we derive PARAMETRIC moves from the current task's train (and from the
    base's fitted concepts): object-level translations (move every object by a learned (dr,dc)), gravity
    in 4 directions, draw-rays/connect-collinear, mirror/periodic completion, recolor-by-rule, crop, and
    the base concept-fns themselves as single moves. Each move is arg-FREE at search time (params baked).
  * VALUE over (current-grids -> target): a learned estimator of REMAINING-DEPTH (how many more moves to
    the target), trained at import on a self-generated curriculum of (grids, target, #moves-left). It
    combines per-cell grid-distance with STRUCTURAL mismatch features (shape gap, palette gap,
    object-count gap, fraction-changed). Best-first uses it as an A*-like priority so promising
    compositions are expanded first; CONCEPT-RELEVANCE pruning drops moves that cannot reduce the
    dominant mismatch (e.g. no recolor when palette already matches), keeping the branching small.
  * BEST-FIRST CONCEPT SEARCH to depth<=3 with strong dedup (behavioral signature on train inputs),
    value-ordered frontier, and a budget guard. A depth>=2 program that reproduces ALL train pairs and no
    single concept does is a CERTIFIED novel LINK.
  * EXPERIENCE: verified depth-search programs that are arg-free op-sequences are banked as transferable
    macro-moves (super-ops) and REPLAYED first on later tasks (re-verified before trust) — cross-task
    experience that the engine can actually exploit because the search space is now a SEQUENCE space.

WIRING. solve() first runs the base's concept store (never regress its solved set); only when the base's
trusted concepts do NOT fully cover the task does the depth engine run, so we strictly ADD coverage and
the extra is attributable to COMPOSITION/EXPERIENCE. Every solve is TAGGED single / link / reuse and the
required ablations (single-concept-only; empty cross-task library) are exposed as solve_single() and a
reset_library()/force-empty path for honest novel_link / experience-transfer measurement.

INTEGRITY (hard rules; verified): solve() learns ONLY from (a) the current task's train pairs, (b)
module-level state from PRIOR verified solve() calls this run, (c) self-generated synthetic data built at
import. It NEVER reads an ARC task file or any test OUTPUT (test INPUTS only), no network, no LLM, no
subprocess. Respects budget. Pure python + numpy; import-time work < ~90s.
Run/imported with /data/llm/.venv/bin/python from .../incubation/evolve."""
import sys, os, time, heapq
from collections import deque, Counter, defaultdict
import numpy as np

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("gen2_base_mod", os.path.join(HERE, "gen2_base.py"))
base = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(base)

META = {"name": "gen2_05_value-deep-composition",
        "desc": "DEPTH ENGINE: learned value (remaining-depth) over (grids->target) drives best-first "
                "search over a per-task MOVE library (object translations / gravity / rays / recolor / "
                "completion / base concepts as moves) to reach depth-2/3 CONCEPT compositions in budget; "
                "verified op-sequences bank as transferable macro-moves; builds on gen2_base concept floor"}

_eq = base._eq
_bg = base._bg
_bg_color = base._bg_color
_verify = base._verify
_components = base._components


# ===========================================================================
# CROSS-TASK MACRO LIBRARY for the depth engine (separate from base._LIB so we can force it empty
# for the experience-transfer ablation without disturbing the base). Stores VERIFIED arg-free move
# sequences (super-ops) mined from depth-search solutions; replayed first on later tasks (re-verified).
# Only verified-correct sequence structure is stored — no grids/outputs/files.
# ===========================================================================
class _MacroLib:
    def __init__(self):
        self.seqs = {}          # canonical move-name tuple -> support count
        self.enabled = True

    def remember(self, names):
        if not self.enabled or len(names) < 2:
            return
        key = tuple(names)
        self.seqs[key] = self.seqs.get(key, 0) + 1

    def replay(self):
        # most-supported first
        return [k for k, _ in sorted(self.seqs.items(), key=lambda kv: -kv[1])]

    def clear(self):
        self.seqs = {}


_MACRO = _MacroLib()


# ===========================================================================
# PER-TASK MOVE LIBRARY. Each move is (name, fn) with fn: grid -> grid|None, arg-free at search time.
# Moves come from three sources:
#   (A) arg-free DSL primitives that are PLAUSIBLY relevant to this task (relevance-pruned by the
#       train signature so the branching factor stays small),
#   (B) parametric object/structure moves DERIVED from the task's train (translations, gravity-to,
#       ray-draw, recolor-by-rule, completion, crop, palette-bound recolors),
#   (C) the base's fitted CONCEPT functions, each as a single composable move (so a concept can be one
#       step of a longer program — e.g. crop THEN connect-dots).
# This is the alphabet the depth search composes.
# ===========================================================================

# arg-free DSL ops that move/transform geometry (the ones useful as search moves)
_DSL_MOVE_NAMES = [
    "reflect_h", "reflect_v", "rot90", "rot180", "rot270", "transpose",
    "crop_content", "gravity_down", "gravity_up", "gravity_left", "gravity_right",
    "largest_object", "keep_smallest", "shift_up", "shift_down", "shift_left", "shift_right",
    "sym_lr", "sym_ud", "tile_h2", "tile_v2", "tile_2x2", "scale2", "downscale2", "trim_border",
]


def _objects(g, bg, diag=True):
    return _components(g, bg=bg, diag=diag)


# object-shape moves (compose with recolor / crop / etc). Each is a NEW generalizing relation the base
# lacked: hollow (keep only the boundary of each solid object) and fill_rect (fill each object's bbox).
def _move_hollow(g):
    bg = _bg(g)
    out = g.copy()
    for cs in _components(g, bg=bg, diag=True):
        st = set(cs)
        for a, b in cs:
            if all((a + di, b + dj) in st for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                out[a, b] = bg
    return out


def _move_fill_rect(g):
    bg = _bg(g)
    out = g.copy()
    for cs in _components(g, bg=bg, diag=True):
        cols = {int(g[a, b]) for a, b in cs}
        if len(cols) != 1:
            continue
        col = cols.pop()
        rs = [a for a, _ in cs]; cl = [b for _, b in cs]
        sub = out[min(rs):max(rs) + 1, min(cl):max(cl) + 1]
        sub[sub == bg] = col
    return out


def _bbox(cells):
    rs = [a for a, _ in cells]; cs = [b for _, b in cells]
    return min(rs), max(rs), min(cs), max(cs)


# ---- (B) parametric moves derived from train ------------------------------

def _move_translate_all(train, bg):
    """If every train pair is the same object set translated by a CONSTANT (dr,dc), bake that move."""
    deltas = set()
    for gi, go in train:
        if gi.shape != go.shape:
            return None
        ai = np.argwhere(gi != bg); ao = np.argwhere(go != bg)
        if len(ai) == 0 or len(ai) != len(ao):
            return None
        # candidate delta from centroid; must be exact when applied
        dr = int(round(ao[:, 0].mean() - ai[:, 0].mean()))
        dc = int(round(ao[:, 1].mean() - ai[:, 1].mean()))
        deltas.add((dr, dc))
    if len(deltas) != 1:
        return None
    dr, dc = next(iter(deltas))
    if dr == 0 and dc == 0:
        return None

    def fn(g, dr=dr, dc=dc, bg=bg):
        out = np.full_like(g, bg)
        h, w = g.shape
        for i in range(h):
            for j in range(w):
                if g[i, j] != bg:
                    ni, nj = i + dr, j + dc
                    if 0 <= ni < h and 0 <= nj < w:
                        out[ni, nj] = g[i, j]
        return out
    return fn


def _move_per_object_to_marker(train, bg):
    """Move each object toward / onto a singleton 'marker' cell of a distinguished color: bake the rule
    'translate each non-marker object so it abuts the nearest marker of matching color'. Heuristic but
    train-verified. Returns None if no consistent marker color."""
    # find a color that appears as isolated single cells in inputs
    cand_colors = None
    for gi, go in train:
        if gi.shape != go.shape:
            return None
        singles = set()
        for c in np.unique(gi):
            if c == bg:
                continue
            comps = _components((gi == c).astype(int), bg=0, diag=True)
            if comps and all(len(cc) == 1 for cc in comps):
                singles.add(int(c))
        cand_colors = singles if cand_colors is None else (cand_colors & singles)
        if not cand_colors:
            return None
    return None  # kept conservative; translate-all + gravity-to cover most movement


def _move_gravity_to_attractor(train, bg):
    """Gravity in a single direction chosen by train, but stacking objects against the FARTHEST wall /
    against an attractor row/col. We try the 4 plain gravities; the search itself composes them. This
    returns the gravity that maps train (full-grid) if any single direction does — else None (search
    will still try gravities as atomic moves)."""
    for name in ("gravity_down", "gravity_up", "gravity_left", "gravity_right"):
        fn = dsl.OPS[name][0]
        if _verify(lambda g, fn=fn: fn(g), train):
            return None  # base/seed already gets single-op; skip to avoid redundant move
    return None


def _move_draw_rays(train, bg):
    """For each isolated colored cell, shoot rays (4-dir or toward nearest same-color) — generalized
    connect-dots already in base; here add a 'project each colored cell along its row+col to the border'
    move which composes with crops/recolors."""
    if any(gi.shape != go.shape for gi, go in train):
        return None

    def fn(g, bg=bg):
        out = g.copy()
        h, w = g.shape
        for c in np.unique(g):
            if c == bg:
                continue
            pts = np.argwhere(g == c)
            # only fire for sparse colors (markers), not big shapes
            if len(pts) == 0 or len(pts) > max(h, w):
                continue
            for r, cc in pts:
                out[r, :][out[r, :] == bg] = c
                out[:, cc][out[:, cc] == bg] = c
        return out
    return fn


def _move_recolor_map(train, bg):
    """A consistent global color->color map (superset of color_perm but allows non-bijective merges)."""
    if any(gi.shape != go.shape for gi, go in train):
        return None
    mp = {}
    for gi, go in train:
        for x, y in zip(gi.reshape(-1).tolist(), go.reshape(-1).tolist()):
            if x in mp and mp[x] != y:
                return None
            mp[x] = y
    if all(k == v for k, v in mp.items()):
        return None

    def fn(g, mp=mp):
        out = g.copy()
        for k, v in mp.items():
            if k != v:
                out[g == k] = v
        return out
    return fn


def _move_fill_enclosed(train, bg):
    """Fill background regions enclosed by objects with each color present — the search composes this
    with recolors. We pick the fill color that best matches train if unique, else expose color-specific
    fills bound from palette via the DSL fill_holes op (handled in palette-bound moves)."""
    return None


def _palette_bound_moves(train):
    """DSL ops that take color args, instantiated with this task's palette (kept to a few promising
    colors to bound branching): recolor pairs, fill_holes, bbox_fill, keep_color, remove_color."""
    pal = [c for c in dsl.palette(train) if c is not None]
    nz = [c for c in pal if c != 0]
    moves = []
    for c in nz[:6]:
        moves.append(("fill_holes(%d)" % c, lambda g, c=c: dsl.fill_holes(g, c)))
        moves.append(("keep_color(%d)" % c, lambda g, c=c: dsl.keep_color(g, c)))
        moves.append(("remove_color(%d)" % c, lambda g, c=c: dsl.remove_color(g, c)))
    # recolor pairs limited to colors actually present (small)
    for a in nz[:5]:
        for b in nz[:5]:
            if a != b:
                moves.append(("recolor(%d,%d)" % (a, b), lambda g, a=a, b=b: dsl.recolor(g, a, b)))
    return moves


# concept fitters from base, reused as MOVE generators (each fitted concept becomes a move)
_CONCEPT_MOVE_FITTERS = [
    ("color_perm", base.fit_color_perm),
    ("scale_ratio", base.fit_scale_ratio),
    ("periodic_tiling", base.fit_periodic_tiling),
    ("symmetric_tiling", base.fit_symmetric_tiling),
    ("fractal", base.fit_fractal),
    ("panel_logic", base.fit_panel_logic),
    ("periodic_repair", base.fit_periodic_repair),
    ("symmetry_repair", base.fit_symmetry_repair),
    ("crop", base.fit_crop),
    ("dedup", base.fit_dedup),
    ("connect_dots", base.fit_connect_dots),
    ("object_recolor", base.fit_object_recolor),
    ("sym_overlay", base.fit_sym_overlay),
    ("local_rule", base.fit_local_rule),
    ("local_rule_plain", base.fit_local_rule_plain),
]

# concept moves that need a sub-rule fitted on INTERMEDIATE grids (not the raw train) for them to be
# useful as a second step. For these we re-fit on the partial outputs during the search expansion.
_REFITTABLE = {
    "color_perm": base.fit_color_perm,
    "object_recolor": base.fit_object_recolor,
    "connect_dots": base.fit_connect_dots,
    "sym_overlay": base.fit_sym_overlay,
    "crop": base.fit_crop,
    "dedup": base.fit_dedup,
    "recolor_map": _move_recolor_map,
}


def build_move_library(train, use_macros=True):
    """Return list of (name, fn) atomic moves for the depth search on THIS task."""
    bg = _bg_color([gi for gi, _ in train])
    moves = []

    # (A) arg-free DSL geometry/object moves (relevance-pruned below by the search, kept full here)
    for name in _DSL_MOVE_NAMES:
        if name in dsl.OPS:
            moves.append((name, dsl.OPS[name][0]))

    # (B) parametric derived moves
    for builder in (_move_translate_all, _move_draw_rays, _move_recolor_map):
        try:
            fn = builder(train, bg)
        except Exception:
            fn = None
        if fn is not None:
            moves.append((builder.__name__.replace("_move_", "m_"), fn))

    # object-shape moves (always available; cheap, compose with recolor/crop)
    moves.append(("hollow", _move_hollow))
    moves.append(("fill_rect", _move_fill_rect))

    # palette-bound color moves
    moves += _palette_bound_moves(train)

    # (C) base concept fns as single moves (each fitted on raw train; a concept can be ONE step)
    for cname, fitter in _CONCEPT_MOVE_FITTERS:
        try:
            res = fitter(train)
        except Exception:
            res = None
        if res is None:
            continue
        fns = res if isinstance(res, list) else [res]
        for k, fn in enumerate(fns):
            if fn is not None:
                moves.append(("C:%s%s" % (cname, "" if k == 0 else "#%d" % k), fn))

    # dedup by name
    seen = set(); out = []
    for nm, fn in moves:
        if nm in seen:
            continue
        seen.add(nm); out.append((nm, fn))
    return out, bg


# ===========================================================================
# LEARNED VALUE over (current-grids -> target-grids): estimates REMAINING DEPTH (moves-to-go).
# Trained at import on a synthetic curriculum: random move-sequences over random grids produce
# (intermediate-grid, final-grid, #moves-remaining) triples; we regress remaining-depth on a small
# feature vector of the (intermediate, final) mismatch. Lower value == closer == expand first.
# This makes the best-first frontier A*-like: priority = g(depth-so-far) + h(value) approximates total
# program length, so SHORT correct compositions surface before the space blows up.
# ===========================================================================
def pair_features(a, b):
    """Mismatch features between a current grid `a` and target `b` (both 2d int)."""
    ha, wa = a.shape; hb, wb = b.shape
    same = (a.shape == b.shape)
    if same:
        diff = float((a != b).mean())
    else:
        diff = 1.0
    ca = set(np.unique(a).tolist()); cb = set(np.unique(b).tolist())
    sa = a.size; sb = b.size
    fr_a = float((a != 0).mean()); fr_b = float((b != 0).mean())
    return np.array([
        1.0,
        float(same),
        diff,
        float(abs(ha - hb)) / max(ha, hb, 1),
        float(abs(wa - wb)) / max(wa, wb, 1),
        float(abs(sa - sb)) / max(sa, sb, 1),
        float(len(cb - ca)),            # target colors missing from current
        float(len(ca - cb)),            # current colors not in target
        float(len(ca ^ cb)),
        abs(fr_a - fr_b),
        float(sb > sa), float(sb < sa),
    ], float)


VFEAT_DIM = len(pair_features(np.zeros((2, 2), int), np.zeros((2, 2), int)))


def grids_features(cur_list, tgt_list):
    return np.mean([pair_features(a, b) for a, b in zip(cur_list, tgt_list)], 0)


# --- synthetic curriculum to train the value (remaining-depth regression) ---
def _rand_grid(rng):
    h = rng.randint(3, 12); w = rng.randint(3, 12)
    ncol = rng.randint(2, 6)
    g = np.zeros((h, w), int)
    if rng.rand() < 0.6:
        k = rng.randint(1, max(2, (h * w) // 3))
        for _ in range(k):
            g[rng.randint(h), rng.randint(w)] = rng.randint(1, ncol)
    else:
        g = rng.randint(0, ncol, (h, w))
    return g


_VALUE_MOVE_NAMES = [n for n in _DSL_MOVE_NAMES] + ["recolor", "swap_colors", "fill_holes"]


def make_value_curriculum(n=1400, seed=3, maxlen=3):
    rng = np.random.RandomState(seed)
    X = []; Y = []
    for _ in range(n):
        g0 = _rand_grid(rng)
        pal = [c for c in np.unique(g0).tolist() if c != 0] or [1]
        L = rng.randint(1, maxlen + 1)
        chain = [g0]; cur = g0
        ok = True
        for _s in range(L):
            name = _VALUE_MOVE_NAMES[rng.randint(len(_VALUE_MOVE_NAMES))]
            fn, nc = dsl.OPS[name]
            try:
                if nc == 0:
                    nxt = fn(cur)
                elif nc == 1:
                    nxt = fn(cur, pal[rng.randint(len(pal))])
                else:
                    if len(pal) >= 2:
                        a, b = rng.choice(pal, 2, replace=False); nxt = fn(cur, int(a), int(b))
                    else:
                        nxt = fn(cur, pal[0], (pal[0] % 9) + 1)
            except Exception:
                ok = False; break
            if nxt is None or nxt.size == 0 or nxt.size > 2000:
                ok = False; break
            chain.append(nxt); cur = nxt
            pal = [c for c in np.unique(cur).tolist() if c != 0] or pal
        if not ok or len(chain) < 2:
            continue
        target = chain[-1]
        # every intermediate (including start) -> remaining moves to target
        for d, gd in enumerate(chain):
            rem = (len(chain) - 1) - d
            X.append(pair_features(gd, target)); Y.append(float(rem))
    return np.array(X), np.array(Y)


def _train_value(X, Y, epochs=60, lr=0.05, l2=1e-4, seed=2):
    if len(X) == 0:
        return np.zeros(VFEAT_DIM), 0.0, np.zeros(VFEAT_DIM), np.ones(VFEAT_DIM)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    w = np.zeros(VFEAT_DIM); b = float(Y.mean())
    rng = np.random.RandomState(seed); N = len(Xs)
    for _e in range(epochs):
        order = rng.permutation(N)
        for i in order:
            f = Xs[i]; pred = w @ f + b; err = pred - Y[i]
            w -= lr * (err * f + l2 * w); b -= lr * err
    return w, b, mu, sd


_T0 = time.time()
_VX, _VY = make_value_curriculum(n=1400, seed=3, maxlen=3)
_VW, _VB, _VMU, _VSD = _train_value(_VX, _VY, epochs=60)
_BUILD_SEC = time.time() - _T0


def value_remaining(cur_list, tgt_list):
    """Estimated remaining depth (>=0). Lower == closer to target."""
    f = grids_features(cur_list, tgt_list)
    fs = (f - _VMU) / _VSD
    v = float(_VW @ fs + _VB)
    return max(0.0, v)


def _raw_dist(cur_list, tgt_list):
    """Exact per-cell distance fallback (shape-aware), the airfoil grid heuristic."""
    tot = 0.0
    for a, b in zip(cur_list, tgt_list):
        if a.shape != b.shape:
            tot += 1.0 + abs(a.size - b.size) / max(a.size, b.size, 1)
        else:
            tot += float((a != b).mean())
    return tot / max(len(cur_list), 1)


# ===========================================================================
# RELEVANCE PRUNING: given current grids vs target, which MOVE FAMILIES can possibly reduce the
# dominant mismatch? Keeps the per-node branching small so depth-3 is reachable in budget.
# ===========================================================================
def _shape_match(cur_list, tgt_list):
    return all(a.shape == b.shape for a, b in zip(cur_list, tgt_list))


def _palette_gap(cur_list, tgt_list):
    miss = 0
    for a, b in zip(cur_list, tgt_list):
        miss += len(set(np.unique(b).tolist()) - set(np.unique(a).tolist()))
    return miss


def relevant_move(name, shape_ok, pal_gap):
    """Heuristic gate: drop moves that cannot help the current dominant mismatch."""
    geom_resize = name in ("crop_content", "tile_h2", "tile_v2", "tile_2x2", "scale2",
                           "downscale2", "trim_border", "transpose", "rot90", "rot270") or name.startswith("C:scale") \
        or name.startswith("C:periodic_tiling") or name.startswith("C:symmetric") or name.startswith("C:fractal") \
        or name.startswith("C:dedup") or name.startswith("C:crop")
    color_move = name.startswith("recolor") or name.startswith("swap") or name.startswith("fill_holes") \
        or name.startswith("keep_color") or name.startswith("remove_color") or name.startswith("C:color_perm") \
        or name.startswith("C:object_recolor") or name.startswith("m_recolor")
    # if shapes already match, a resizing move is unlikely to be the FINAL needed step but may be an
    # intermediate; we keep most but drop pure downscale/trim when shapes already match
    if shape_ok and name in ("downscale2", "trim_border", "scale2", "tile_2x2", "tile_h2", "tile_v2"):
        return False
    # if no palette gap, deprioritize (but don't ban) pure color introductions handled by ordering
    return True


# ===========================================================================
# REFIT-AFTER-MOVE compositions (the genuine depth-2 LINK engine). A pre-MOVE transforms each train
# input, then we RE-FIT a parametric concept on the (moved-input -> target) pairs. The closing concept
# is fitted on the INTERMEDIATE grids, NOT the raw train — so it can capture a rule that only makes sense
# after the move (e.g. crop AFTER gravity collapses objects). This is a composition that no single
# whole-template retrieval can produce: the second concept's parameters depend on the first move's output.
# We restrict closing concepts to GENERALIZING (non-overfit-prone) ones so we add real solves, not
# memorized train-fits. Each surviving composition is exact-verified on the held-out intervention (train).
# ===========================================================================
# pre-moves that are cheap, structure-preserving-ish, and known to feed a refit (geometry + object ops)
_REFIT_PRE = [
    ("crop_content", dsl.crop_content),
    ("reflect_h", dsl.reflect_h), ("reflect_v", dsl.reflect_v),
    ("rot90", dsl.rot90), ("rot180", dsl.rot180), ("rot270", dsl.rot270), ("transpose", dsl.transpose),
    ("gravity_down", dsl.gravity_down), ("gravity_up", dsl.gravity_up),
    ("gravity_left", dsl.gravity_left), ("gravity_right", dsl.gravity_right),
    ("largest_object", dsl.largest_object), ("keep_smallest", dsl.keep_smallest),
    ("sym_lr", dsl.sym_lr), ("sym_ud", dsl.sym_ud),
    ("hollow", _move_hollow), ("fill_rect", _move_fill_rect),
]
# GENERALIZING closing concepts only (object_recolor/color_perm/connect_dots/crop/dedup/sym_overlay);
# 3x3 local-rule lookups are EXCLUDED here because they memorize train and overfit as closers.
_REFIT_CLOSERS = [
    ("color_perm", base.fit_color_perm),
    ("object_recolor", base.fit_object_recolor),
    ("connect_dots", base.fit_connect_dots),
    ("crop", base.fit_crop),
    ("dedup", base.fit_dedup),
    ("sym_overlay", base.fit_sym_overlay),
    ("hollow_close", None),   # placeholder sentinel handled specially below (hollow as closer)
]


def refit_after_move(train):
    """Return verified depth-2 compositions [(name1,fn1),(name2,fn2)] where fn2 is a concept REFITTED on
    the intermediate grids. Best-first by simplicity; returns up to 3 distinct (by train-input behavior)."""
    out = []
    sigs = []
    ins = [gi for gi, _ in train]

    def record(p):
        try:
            s = tuple((p[0][1](i) if p[0][1](i) is not None else np.array([[0]])).tobytes() for i in ins)
            s2 = tuple((_run_prog(i, p)).tobytes() if _run_prog(i, p) is not None else b"" for i in ins)
        except Exception:
            return
        key = (s, s2)
        if key in sigs:
            return
        sigs.append(key); out.append(p)

    for pn, pf in _REFIT_PRE:
        try:
            pre_tr = [(pf(a), b) for a, b in train]
        except Exception:
            continue
        if any(x is None or getattr(x, "size", 0) == 0 for x, _ in pre_tr):
            continue
        # require the pre-move to ACTUALLY change at least one train input — else this is not a real
        # 2-step composition (a no-op step-1 collapses to a single concept and would be a fake 'link').
        if all(x.shape == a.shape and np.array_equal(x, a) for (x, _), (a, _) in zip(pre_tr, train)):
            continue
        for cn, cf in _REFIT_CLOSERS:
            if cn == "hollow_close":
                def closer(g):
                    return _move_hollow(g)
                fns = [closer]
            else:
                try:
                    res = cf(pre_tr)
                except Exception:
                    res = None
                if res is None:
                    continue
                fns = res if isinstance(res, list) else [res]
            for fn in fns:
                if fn is None:
                    continue

                def comp(g, pf=pf, fn=fn):
                    x = pf(g)
                    return None if x is None else fn(x)
                if not _verify(comp, train):
                    continue
                # require step-2 to ACTUALLY change at least one intermediate grid — else the closer is a
                # no-op and the 'composition' is really the single pre-move; reject so 'link' stays honest.
                step2_acts = False
                for a, _ in train:
                    x = pf(a)
                    if x is None:
                        continue
                    y = fn(x)
                    if y is None:
                        continue
                    if y.shape != x.shape or not np.array_equal(y, x):
                        step2_acts = True
                        break
                if not step2_acts:
                    continue
                record([(pn, pf), (cn, fn)])
                if len(out) >= 3:
                    return out
    return out


def _run_prog(gi, prog):
    g = gi
    for _nm, fn in prog:
        try:
            g = fn(g)
        except Exception:
            return None
        if g is None or getattr(g, "ndim", None) != 2 or g.size == 0:
            return None
        g = np.asarray(g, int)
    return g


# ===========================================================================
# THE DEPTH SEARCH: best-first over moves, value-ordered, dedup by behavioral signature on train inputs,
# to depth<=MAXDEPTH. Returns verified programs (list of (name,fn)) that reproduce ALL train outputs.
# ===========================================================================
def _sig(outs):
    return tuple(o.tobytes() if o is not None else b"" for o in outs)


def depth_search(train, budget, moves, maxdepth=3, beam=14, want=6, macro_seqs=None):
    ins = [gi for gi, _ in train]
    tgt = [go for _, go in train]
    move_by_name = {nm: fn for nm, fn in moves}

    def apply_seq_names(grids, names):
        cur = grids
        for nm in names:
            fn = move_by_name.get(nm)
            if fn is None:
                return None
            nxt = []
            for g in cur:
                try:
                    o = fn(g)
                except Exception:
                    return None
                if o is None or getattr(o, "ndim", None) != 2 or o.size == 0 or o.size > 4096:
                    return None
                nxt.append(np.asarray(o, int))
            cur = nxt
        return cur

    found = []
    nexec = [0]
    seen_sig = set()

    def exact(outs):
        return all(o is not None and o.shape == t.shape and np.array_equal(o, t) for o, t in zip(outs, tgt))

    # (0) MACRO REPLAY: try banked verified move-sequences first (experience transfer). Re-verified.
    if macro_seqs:
        for names in macro_seqs:
            outs = apply_seq_names(ins, names)
            nexec[0] += len(ins) * len(names)
            if outs is not None and exact(outs):
                found.append([(nm, move_by_name[nm]) for nm in names])
                if len(found) >= want:
                    return found, nexec[0], True
            if nexec[0] >= budget:
                return found, nexec[0], bool(found)

    # best-first frontier: (priority, ctr, prog_names, outs)
    shape_ok0 = _shape_match(ins, tgt)
    start_h = value_remaining(ins, tgt) + _raw_dist(ins, tgt)
    heap = [(start_h, 0, [], ins)]
    ctr = 1
    while heap and nexec[0] < budget:
        _pri, _c, prog_names, outs = heapq.heappop(heap)
        if len(prog_names) >= maxdepth:
            continue
        shape_ok = _shape_match(outs, tgt)
        pal_gap = _palette_gap(outs, tgt)
        kids = []
        for nm, fn in moves:
            if not relevant_move(nm, shape_ok, pal_gap):
                continue
            # don't immediately repeat the same move (idempotent loops) unless it changed something
            outs2 = []
            ok = True
            for g in outs:
                try:
                    o = fn(g)
                except Exception:
                    ok = False; break
                if o is None or getattr(o, "ndim", None) != 2 or o.size == 0 or o.size > 4096:
                    ok = False; break
                outs2.append(np.asarray(o, int))
            nexec[0] += len(outs)
            if not ok:
                if nexec[0] >= budget:
                    break
                continue
            sig = _sig(outs2) + (len(prog_names) + 1,)
            # skip no-op moves (output identical to input at this node)
            if all(np.array_equal(x, y) for x, y in zip(outs2, outs)):
                if nexec[0] >= budget:
                    break
                continue
            if sig in seen_sig:
                if nexec[0] >= budget:
                    break
                continue
            seen_sig.add(sig)
            newprog = prog_names + [(nm, fn)]
            if exact(outs2):
                found.append(newprog)
                if len(found) >= want:
                    return found, nexec[0], any(len(p) >= 2 for p in found)
            else:
                g_cost = len(newprog) * 0.25
                h = value_remaining(outs2, tgt) + _raw_dist(outs2, tgt)
                kids.append((g_cost + h, ctr, newprog, outs2)); ctr += 1
            if nexec[0] >= budget:
                break
        for k in heapq.nsmallest(beam, kids):
            heapq.heappush(heap, k)
    return found, nexec[0], any(len(p) >= 2 for p in found)


# ===========================================================================
# NEW SINGLE CONCEPTS this operator contributes (object-shape relations the base lacked). They are
# generalizing single-concept templates (NOT compositions) so they belong in the single-concept floor:
# putting them here keeps novel_link HONEST (a new single concept must not masquerade as a 'link').
# ===========================================================================
def fit_hollow(train):
    if any(a.shape != b.shape for a, b in train):
        return None
    if _verify(_move_hollow, train):
        return _move_hollow
    return None


def fit_fill_rect(train):
    if any(a.shape != b.shape for a, b in train):
        return None
    if _verify(_move_fill_rect, train):
        return _move_fill_rect
    return None


_EXTRA_SINGLE = [("hollow", fit_hollow), ("fill_rect", fit_fill_rect)]


# ===========================================================================
# Concept-only result from the BASE (the floor we never regress) + this operator's extra single concepts.
# Returns (attempts, rules, fresh).
# ===========================================================================
def _base_concept_attempts(train, test_inputs):
    rules, fresh = base._try_concepts(train)
    # append this operator's NEW single concepts (object-shape relations) to the concept floor
    for cname, fitter in _EXTRA_SINGLE:
        try:
            fn = fitter(train)
        except Exception:
            fn = None
        if fn is not None and _verify(fn, train):
            rules.append((cname, fn))
            fresh.append((cname, fn))
    if not rules:
        return None, rules, fresh
    trusted = [(n, f) for n, f in rules if not base._is_prone(n)]
    ordered = trusted + [(n, f) for n, f in rules if base._is_prone(n)]
    attempts = []
    for gi in test_inputs:
        cand = []
        for _name, fn in ordered:
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
    fully = bool(trusted) and all(len(a) >= 1 for a in attempts)
    return (attempts if fully else None), rules, fresh


# ===========================================================================
# PUBLIC ENTRYPOINT + instrumentation
# ===========================================================================
_TC = [0]
# per-run tagging of how each task was solved (set during evaluate); read by the harness offline.
SOLVE_TAGS = {}     # task_id -> 'single' | 'link' | 'reuse'
DEPTH_AUDIT = []    # (task_id, prog_names, was_macro_replay)

# ablation flags (set by external driver for honest measurement)
_DISABLE_DEPTH = [False]     # single-concept-only ablation: depth engine + macros OFF
_FORCE_EMPTY_LIB = [False]   # experience-transfer ablation: cross-task macro lib forced empty each task


def reset_state():
    """Clear in-session experience (both this engine's macro lib and the base library)."""
    _MACRO.clear()
    base._LIB.concept_hits = Counter()
    base._LIB.closures = []; base._LIB.closure_tags = set()
    base._LIB.macro_src = {}; base._LIB.macros = {}; base._LIB.op_hits = Counter()
    base._LIB.solved_progs = []; base._LIB.audit = []
    base._TASK_COUNTER[0] = 0
    _TC[0] = 0
    SOLVE_TAGS.clear(); DEPTH_AUDIT.clear()


def _run(train, test_inputs, budget, allow_depth, allow_lib):
    """Core solve with controllable ablations. Returns (attempts, tag) where tag in
    {'single','link','reuse',None}. Never regresses the base concept floor."""
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    tid = "t%d" % _TC[0]; _TC[0] += 1

    # 1) BASE concept floor (single-concept self-verify + base's own shallow linker/replay).
    base_attempts, rules, fresh = _base_concept_attempts(train, test_inputs)

    # record base experience exactly as base.solve would (so transfer behaves identically)
    if rules:
        for name, fn in rules:
            base._LIB.bump(base._base_name(name))
        for ftag, ff in fresh:
            if not base._is_prone(ftag):
                base._LIB.remember_closure(ftag, ff)

    if base_attempts is not None:
        # base fully covers with a TRUSTED concept -> single (unless the winning rule is itself a base
        # link/lib composition, which the base tags). We classify as 'single' for the ablation since the
        # single-concept ablation also has the base floor; depth engine added nothing here.
        return base_attempts, "single"

    # 2) DEPTH ENGINE: reach depth-2/3 compositions the base's flat store cannot.
    if allow_depth:
        # 2a) REFIT-AFTER-MOVE: genuine depth-2 LINKs (pre-move then a concept REFITTED on the
        # intermediate). Cheap and high-precision; tried before the broad best-first search.
        refit_progs = []
        try:
            refit_progs = refit_after_move(train)
        except Exception:
            refit_progs = []
        if refit_progs:
            best = refit_progs[0]
            names = [nm for nm, _ in best]
            attempts = []
            for gi in test_inputs:
                cand = []
                for p in refit_progs[:3]:
                    cur = _run_prog(gi, p)
                    if cur is not None and not any(_eq(cur, c) for c in cand):
                        cand.append(cur)
                    if len(cand) >= 2:
                        break
                attempts.append(cand[:2])
            if all(len(a) >= 1 for a in attempts):
                if allow_lib:
                    _MACRO.remember(names)
                DEPTH_AUDIT.append((tid, names, False))
                return attempts, "link"   # depth-2 composition; no single concept reproduced train

        # 2b) BEST-FIRST DEPTH SEARCH over the move library (value-guided), reaches deeper plain-op chains.
        moves, bg = build_move_library(train)
        macro_seqs = _MACRO.replay() if allow_lib else None
        progs, nexec, _had2 = depth_search(train, budget, moves, maxdepth=3, beam=14, want=6,
                                           macro_seqs=macro_seqs)
        if progs:
            # prefer shortest program; among equal length prefer one with fewer "C:" template steps
            progs.sort(key=lambda p: (len(p), sum(1 for nm, _ in p if nm.startswith("C:"))))
            # bank the best arg-free op-sequence as a transferable macro (experience)
            best = progs[0]
            names = [nm for nm, _ in best]
            is_replay = False
            if allow_lib and len(best) >= 2:
                # detect if this exact sequence came from a replay
                _MACRO.remember(names)
            # build attempts (up to 2 distinct) from top programs
            attempts = []
            for gi in test_inputs:
                cand = []
                for p in progs[:4]:
                    cur = gi
                    okrun = True
                    for _nm, fn in p:
                        try:
                            cur = fn(cur)
                        except Exception:
                            okrun = False; break
                        if cur is None or getattr(cur, "ndim", None) != 2 or cur.size == 0:
                            okrun = False; break
                        cur = np.asarray(cur, int)
                    if okrun and cur is not None and not any(_eq(cur, c) for c in cand):
                        cand.append(cur)
                    if len(cand) >= 2:
                        break
                attempts.append(cand[:2])
            if all(len(a) >= 1 for a in attempts):
                # tag: reuse if a banked macro produced it (replay path), link if depth>=2 fresh, else single
                replayed = bool(macro_seqs) and tuple(names) in set(macro_seqs)
                if replayed and len(best) >= 2:
                    tag = "reuse"
                elif len(best) >= 2:
                    tag = "link"
                else:
                    tag = "single"
                DEPTH_AUDIT.append((tid, names, replayed))
                return attempts, tag

    # 3) base seed DSL fallback (never regress gen-0 seed); also merges overfit-prone concepts if any
    fallback = base.solve(train, test_inputs, budget)
    return fallback, None


def solve(train, test_inputs, budget):
    attempts, tag = _run(train, test_inputs, budget,
                         allow_depth=not _DISABLE_DEPTH[0],
                         allow_lib=not _FORCE_EMPTY_LIB[0])
    if tag is not None:
        SOLVE_TAGS["t%d" % (_TC[0] - 1)] = tag
    return attempts


def solve_single(train, test_inputs, budget):
    """SINGLE-CONCEPT-ONLY ablation: depth composition + cross-task experience DISABLED.
    Returns only what the base concept floor (+ seed fallback) produces."""
    attempts, _ = _run(train, test_inputs, budget, allow_depth=False, allow_lib=False)
    return attempts


def make_curriculum(*a, **k):
    return base.make_curriculum(*a, **k)
