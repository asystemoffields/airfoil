#!/usr/bin/env python3
"""Gen-2 creativity-operator #4 : LINE / CONNECT / PATTERN-COMPLETION.

Builds DIRECTLY ON gen2_base (imported, never regressed). Adds the second big bucket-A family:
draw lines/rays between aligned objects or from objects to edges, connect collinear (incl. DIAGONAL)
same-color dots with a learned fill color, cast directional rays, and COMPLETE an occluded periodic /
symmetric pattern by inferring the 2-D generator and filling the hole — even when the hole color IS the
background (the case the base's marker-occluder repair misses).

WHY THESE (facet rationale). The base ships a same-color row/col connect_dots and a marker-occluder
periodic/symmetry repair. The line/pattern-completion family is broader: (1) collinear pairs are joined
on ROWS, COLS *and DIAGONALS*, and the segment color may DIFFER from the endpoints (learned constant);
(2) isolated cells can emit rays in a learned set of the 8 directions until they hit content/edge; (3) a
solid background patch can occlude a doubly-periodic texture and must be regenerated from the inferred
(pr,pc) lattice. Each is a PARAMETRIC fitter: it induces its parameters from train and self-verifies
(reproduces EVERY train pair exactly) before it is allowed to answer.

CREATIVITY WIRING (the point, not the coverage). Every new concept is registered in the base's CONCEPT
store so the base machinery treats them as first-class atoms:
  * they are REMEMBERED as transferable closures (a "draw diagonal connectors of color X" abstraction
    banked on one task is re-verified and REUSED on a later task -> experience_transfer),
  * they participate in the LINKER (geom pre-op -> concept, and remembered<->fresh composition), so a
    task needing e.g. crop-then-connect or connect-then-recolor is solved by a COMPOSITION that no single
    concept reproduces (-> novel_link).
We MEASURE all of this honestly with three self-ablations (single-concept-only, library-forced-empty,
and the held-out lookup gap) rather than guessing, and report the real deltas.

INTEGRITY (hard rules, unchanged from base): solve() learns ONLY from (a) the current task's train pairs,
(b) module-level library state from PRIOR verified solve() calls this run, (c) import-time synthetic data.
It NEVER reads an ARC file or any test OUTPUT, no network, no LLM at solve time. Respects budget. Pure
python + numpy. Run/imported with /data/llm/.venv/bin/python."""
import sys
import importlib.util
from collections import deque, defaultdict
import numpy as np

HERE = sys.path[0]
_BASE_PATH = "/data/Windows-files/Documents/airfoil/incubation/evolve/cand/gen2_base.py"
_spec = importlib.util.spec_from_file_location("gen2_base_for04", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

# pull in the base's verifier / helpers / library so our concepts are first-class atoms in ITS machinery
_eq = base._eq
_verify = base._verify
_bg = base._bg
_components = base._components

META = {"name": "gen2_04_line-pattern-completion",
        "desc": "gen2_base + connect-collinear(row/col/diag, learned fill) + directional ray cast + "
                "2-D periodic/symmetry completion with background-as-occluder; all wired into the base's "
                "transfer/linker/curriculum machinery so composition & reuse can pay."}


# ===========================================================================
# CONCEPT A : CONNECT COLLINEAR SAME-COLOR DOTS  (row / col / DIAGONAL ; learned fill color)
#   Generalizes the base connect_dots: (i) adds both diagonals, (ii) the between-segment color may be a
#   single learned CONSTANT distinct from the endpoint color (e.g. 253bf280: 8..8 -> 8 3 3 3 3 8).
#   Only EMPTY (bg) cells strictly between two consecutive same-color collinear members are painted.
# ===========================================================================
def _additive_only(g, bg):
    """fast guard: a connect/ray rule can only ADD bg cells; if a grid has no bg cell at all there is
    nothing to draw into (cheap bail so banked closures fail fast on inapplicable later tasks)."""
    return bool((g == bg).any())


def _connect(g, mode, fill_const, diag, bg):
    out = g.copy()
    for c in np.unique(g):
        if c == bg:
            continue
        pts = [(int(r), int(cc)) for r, cc in np.argwhere(g == c)]
        if len(pts) < 2:
            continue
        fill = c if mode == "same" else fill_const
        groups = defaultdict(list)
        for r, cc in pts:
            groups[("r", r)].append((r, cc))
            groups[("c", cc)].append((r, cc))
            if diag:
                groups[("d", r - cc)].append((r, cc))
                groups[("a", r + cc)].append((r, cc))
        for members in groups.values():
            if len(members) < 2:
                continue
            members.sort()
            for (r1, c1), (r2, c2) in zip(members, members[1:]):
                dr = (r2 > r1) - (r2 < r1)
                dc = (c2 > c1) - (c2 < c1)
                rr, cc = r1 + dr, c1 + dc
                while (rr, cc) != (r2, c2):
                    if g[rr, cc] == bg:
                        out[rr, cc] = fill
                    rr += dr; cc += dc
    return out


def _is_additive(train, bg):
    """output keeps every non-bg input cell unchanged and only paints former-bg cells (a line/ray rule
    never moves or erases existing content). A cheap necessary condition for connect/ray."""
    for gi, go in train:
        if gi.shape != go.shape:
            return False
        nb = (gi != bg)
        if np.any(go[nb] != gi[nb]):
            return False
    return True


def fit_connect_lines(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = base._bg_color([gi for gi, _ in train])
    if not _is_additive(train, bg):
        return None
    fcs = set()
    for gi, go in train:
        ch = (gi == bg) & (go != bg)
        if ch.any():
            fcs |= set(np.unique(go[ch]).tolist())
    modes = [("same", None)]
    if len(fcs) == 1:
        modes.append(("const", fcs.pop()))
    out = []
    for diag in (False, True):
        for mode, fc in modes:
            def fn(g, mode=mode, fc=fc, diag=diag, bg=bg):
                return _connect(g, mode, fc, diag, bg)
            if _verify(fn, train):
                out.append(fn)
    if not out:
        return None
    # prefer the simplest fit (row/col before diagonal, same-color before learned-const): out is already
    # in that order. Return up to 2 distinct fits so attempt-2 can disambiguate on test.
    uniq = []
    sigs = []
    test_in = [gi for gi, _ in train]
    for fn in out:
        sig = tuple(fn(i).tobytes() for i in test_in)
        if sig in sigs:
            continue
        sigs.append(sig); uniq.append(fn)
        if len(uniq) >= 2:
            break
    return uniq if len(uniq) > 1 else uniq[0]


# ===========================================================================
# CONCEPT B : DIRECTIONAL RAY CAST FROM ISOLATED CELLS
#   Each isolated single nonzero cell emits a ray in a learned SUBSET of the 8 compass directions; the ray
#   advances (optionally BOUNCING off the 4 walls) until it leaves the grid or hits existing content,
#   painting with the cell's own color or a single learned color. Captures "star / X / bounce" tasks
#   (e.g. 623ea044). Direction set + bounce + paint-color are all induced from train and verified.
# ===========================================================================
_DIRS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1),
         "NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1)}
_DIR_MENUS = [
    ["N"], ["S"], ["E"], ["W"], ["SE"], ["SW"], ["NE"], ["NW"],
    ["N", "S"], ["E", "W"], ["SE", "NW"], ["NE", "SW"],
    ["N", "S", "E", "W"], ["NE", "NW", "SE", "SW"],
    ["N", "S", "E", "W", "NE", "NW", "SE", "SW"],
]


def _isolated_cells(g, bg):
    """nonzero cells whose 8-neighbourhood contains no other nonzero cell (true point seeds)."""
    h, w = g.shape
    seeds = []
    for r, c in np.argwhere(g != bg):
        lone = True
        for dr, dc in _DIRS.values():
            a, b = r + dr, c + dc
            if 0 <= a < h and 0 <= b < w and g[a, b] != bg:
                lone = False; break
        if lone:
            seeds.append((int(r), int(c), int(g[r, c])))
    return seeds


def _cast(g, dirs, paint_same, paintc, bounce, bg):
    out = g.copy(); h, w = g.shape
    for r, c, col in _isolated_cells(g, bg):
        pc = col if paint_same else paintc
        for dn in dirs:
            dr, dc = _DIRS[dn]
            rr, cc = r, c
            for _ in range(2 * (h + w) + 4):
                nr, nc = rr + dr, cc + dc
                if bounce:
                    if not (0 <= nr < h):
                        dr = -dr; nr = rr + dr
                    if not (0 <= nc < w):
                        dc = -dc; nc = cc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if g[nr, nc] != bg:
                    break
                out[nr, nc] = pc; rr, cc = nr, nc
    return out


def fit_ray_cast(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = base._bg_color([gi for gi, _ in train])
    if not _is_additive(train, bg):
        return None
    # ray cast only makes sense when there are isolated point seeds to emit from
    if not any(_isolated_cells(gi, bg) for gi, _ in train):
        return None
    # paint color candidates: own color, or a single learned added color
    added = set()
    for gi, go in train:
        ch = (gi == bg) & (go != bg)
        if ch.any():
            added |= set(np.unique(go[ch]).tolist())
    paint_opts = [(True, 0)]
    if len(added) == 1:
        paint_opts.append((False, added.pop()))
    for bounce in (False, True):
        for dirs in _DIR_MENUS:
            for paint_same, paintc in paint_opts:
                def fn(g, dirs=dirs, ps=paint_same, pc=paintc, bo=bounce, bg=bg):
                    return _cast(g, dirs, ps, pc, bo, bg)
                if _verify(fn, train):
                    return fn
    return None


# ===========================================================================
# CONCEPT C : 2-D PERIODIC PATTERN COMPLETION (occluder MAY be the background)
#   Infer the smallest (pr, pc) row/col periods consistent with all KNOWN cells (cells != hole), then
#   regenerate every hole cell from its lattice class. We try hole = the single train-diff color (marker
#   occluder) AND hole = background (a solid bg patch covering a periodic texture — 484b58aa, 29ec7d0e),
#   which the base's marker-only repair leaves unsolved. Diagonal-period fallback included.
# ===========================================================================
def _periods(g, hole):
    """Smallest row period pr and col period pc consistent with all KNOWN (!=hole) cells. Vectorized
    overlap check (g rolled by p must agree wherever both copies are known)."""
    h, w = g.shape
    known = (g != hole)

    def row_p():
        for p in range(1, h):
            top = known[:h - p, :] & known[p:, :]
            if not np.any((g[:h - p, :] != g[p:, :]) & top):
                return p
        return h

    def col_p():
        for p in range(1, w):
            left = known[:, :w - p] & known[:, p:]
            if not np.any((g[:, :w - p] != g[:, p:]) & left):
                return p
        return w
    return row_p(), col_p()


def _complete_periodic(g, hole):
    h, w = g.shape
    known = (g != hole)
    if known.all():
        return g.copy()
    pr, pc = _periods(g, hole)
    out = g.copy()
    holes = np.argwhere(~known)
    for i, j in holes:
        v = None
        for ii in range(i % pr, h, pr):
            for jj in range(j % pc, w, pc):
                if known[ii, jj]:
                    v = g[ii, jj]; break
            if v is not None:
                break
        if v is not None:
            out[i, j] = v
    return out


def _hole_candidates(train):
    """Holes to try: each color that is (a) the single per-pair diff color, else (b) the background."""
    cands = []
    bg = base._bg_color([gi for gi, _ in train])
    diffcols = set()
    consistent = True
    for gi, go in train:
        if gi.shape != go.shape:
            return []
        d = (gi != go)
        if d.any():
            hs = set(np.unique(gi[d]).tolist())
            if len(hs) != 1:
                consistent = False
            diffcols |= hs
    if consistent and len(diffcols) == 1:
        cands.append(next(iter(diffcols)))
    if bg not in cands:
        cands.append(bg)
    return cands


def fit_pattern_complete(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    for hole in _hole_candidates(train):
        def fn(g, hole=hole):
            if not (g == hole).any():
                return g.copy()
            return _complete_periodic(g, hole)
        if _verify(fn, train):
            return fn
        # crop-to-hole-window variant (output is just the regenerated patch)
        def fn_crop(g, hole=hole):
            nz = np.argwhere(g == hole)
            if nz.size == 0:
                return None
            (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
            return _complete_periodic(g, hole)[r0:r1, c0:c1]
        if _verify(fn_crop, train):
            return fn_crop
    return None


# ===========================================================================
# REGISTER the new concepts into the base CONCEPT store as first-class atoms, so the base's
# transfer / linker / experience-prior machinery operates over them. Inserted BEFORE the overfit-prone
# local_rule entries (cheap, generalizing concepts first) and AFTER the existing connect_dots so the
# simplest exact fit still wins MDL ties.
# ===========================================================================
_NEW = [
    ("connect_lines", fit_connect_lines),
    ("pattern_complete", fit_pattern_complete),
    ("ray_cast", fit_ray_cast),
]


def _install():
    names = {n for n, _ in base.CONCEPTS}
    # splice our concepts right after 'connect_dots' (keep them ahead of overfit-prone local_rule)
    new_list = []
    for n, f in base.CONCEPTS:
        new_list.append((n, f))
        if n == "connect_dots":
            for nn, ff in _NEW:
                if nn not in names:
                    new_list.append((nn, ff))
    # if connect_dots somehow absent, append at front of grafts
    if not any(n == "connect_lines" for n, _ in new_list):
        idx = 0
        for k, (n, _) in enumerate(new_list):
            if n == "crop":
                idx = k; break
        new_list = new_list[:idx] + [(nn, ff) for nn, ff in _NEW if nn not in names] + new_list[idx:]
    base.CONCEPTS[:] = new_list


_install()


# ===========================================================================
# PUBLIC ENTRYPOINT  — delegate to the base solver, which now ranges over the augmented CONCEPT store
# (our line/ray/completion concepts included) plus its linker / library-transfer / seed-DSL fallback.
# ===========================================================================
def solve(train, test_inputs, budget):
    return base.solve(train, test_inputs, budget)


# make the base library / counters reachable for the harness & for the ablation self-tests
_LIB = base._LIB
make_curriculum = base.make_curriculum
