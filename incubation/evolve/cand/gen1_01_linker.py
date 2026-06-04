#!/usr/bin/env python3
"""Gen-1 mutation #1 — the CONCEPT-LINKER / breadth proposer.

Thesis served: creativity = a CONCEPT STORE (a large set of parameterized transform
concepts, most learned-from-train at solve time) + a LINKER that, from cheap features of a
task's train pairs, RECOGNIZES which concepts the task evokes, fits their parameters, and
proposes NOVEL COMPOSITIONS (concept links) — incl. concepts used outside their usual role.
Everything is filtered by EXACT verify on every train pair, so wrong proposals cost nothing.

The proven bottleneck is BREADTH (≈93-99% of seed failures are "rule not expressible in the
32-op DSL"), so this candidate EXPANDS what is expressible:
  - GLOBAL COLOR MAP   : learn a per-cell color permutation from train (a multi-recolor the
                         seed DSL needs many steps for; here it's one fitted concept).
  - OBJECT-PROP RECOLOR: recolor each 4-conn component by a property (size rank / size value /
                         color) -> color, table learned from train.
  - MIRROR-TILE        : output = (kh×kw) grid of per-block geometric transforms of input,
                         the per-block transform table learned from train (fractal mirrors).
  - FRACTAL SELF-TILE  : output block (i,j) = input where input[i,j] is fg else background.
  - PANEL SELECT       : split input into equal panels (or by separator lines); pick the panel
                         chosen by a learned rule (unique / most-fg / least-fg / by-position).
  - CROP-TO-OBJECT     : crop to the bbox of {the unique-color / largest / smallest} object.
  - SYMMETRIZE         : overlay input with its mirror/rot copies to repair occluded symmetry.
  - PERIODIC-TILE-FIT  : detect a tiling that, repeated, reproduces a larger output.
  - + the full seed best-first DSL search as the breadth backstop.

LINKER mechanics (no heavy training): cheap features (size-ratio, fg-mask preservation, panel
divisibility, symmetry signatures, integer ratios, object counts) gate which concepts even fire;
each firing concept FITS its parameters from train and SELF-VERIFIES on every train pair. Across
concepts we keep ALL that pass train, ORDER them MDL-style (fitted complexity + a recognition
prior), and — the repurposing/coverage move — also keep a couple of lower-ranked passers so the
2 attempts aren't pure greedy. We always emit ≤2 attempts/test input.

EXPERIENCE channels:
  - PER-TASK INDUCTION  : every concept's params are induced from THIS task's train pairs.
  - IN-SESSION LIBRARY  : module-level state remembers which concept TYPES verified on solved
                          tasks this run; that prior re-orders concept ranking on later tasks
                          (concepts that have paid off get tried first) — experience that grows.

INTEGRITY: solve() reads only (a) the current task's train pairs, (b) module state from prior
solve() calls (verified concept types only — never any grid/output), (c) nothing at import. It
never opens an ARC file and never sees a test OUTPUT. Pure python + numpy. No net, no LLM.
"""
import sys, heapq
from collections import deque, Counter
import numpy as np

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl

META = {"name": "concept_linker_v1",
        "desc": "breadth proposer: large concept store fit-from-train + feature linker + exact verify, seed-search backstop"}

# ---------------------------------------------------------------------------
# IN-SESSION EXPERIENCE LIBRARY (module-level; persists across solve() calls).
# Counts how often each concept TYPE has produced a train-consistent program this run.
# Used only to RE-ORDER which concepts are tried first — never stores any grid.
# ---------------------------------------------------------------------------
_LIB = Counter()          # concept_name -> #times it verified on a prior task
_SOLVE_CALLS = [0]


def _exact(o, t):
    return o is not None and o.shape == t.shape and np.array_equal(o, t)


def _verify(fn, train):
    """fn: grid->grid (or None). True iff it reproduces EVERY train output exactly."""
    for gi, go in train:
        try:
            o = fn(gi)
        except Exception:
            return False
        if not _exact(o, go):
            return False
    return True


def _bg(g):
    v, ct = np.unique(g, return_counts=True)
    return int(v[ct.argmax()])


# ---------- object helpers ----------
def _components(g, bg=0, diag=False):
    h, w = g.shape
    seen = np.zeros_like(g, bool)
    if diag:
        nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nb = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    comps = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != bg and not seen[i, j]:
                comp = []
                q = deque([(i, j)])
                seen[i, j] = True
                while q:
                    a, b = q.popleft()
                    comp.append((a, b))
                    for di, dj in nb:
                        x, y = a + di, b + dj
                        if 0 <= x < h and 0 <= y < w and g[x, y] != bg and not seen[x, y]:
                            seen[x, y] = True
                            q.append((x, y))
                comps.append(comp)
    return comps


# ===========================================================================
# CONCEPT STORE.  Each concept is a builder: train -> (fn or None).
# It FITS params from train and returns a grid->grid fn ONLY if train-consistent.
# The linker calls all that pass the cheap gate, keeps verified ones.
# ===========================================================================

# --- concept: GLOBAL COLOR MAP (per-cell color permutation learned from train) ---
def fit_colormap(train):
    if not all(a.shape == b.shape for a, b in train):
        return None
    mp = {}
    for a, b in train:
        for x, y in zip(a.flat, b.flat):
            x, y = int(x), int(y)
            if x in mp and mp[x] != y:
                return None
            mp[x] = y
    def fn(g, mp=mp):
        out = g.copy()
        for x, y in mp.items():
            out[g == x] = y
        return out
    return fn


# --- concept: OBJECT-PROPERTY RECOLOR (recolor each component by a learned property table) ---
def _fit_objprop(train, prop):
    """prop(comp_cells, grid)->key ; learn key->out_color from train (mask preserved)."""
    if not all(a.shape == b.shape for a, b in train):
        return None
    if not all(np.array_equal(a != 0, b != 0) for a, b in train):
        return None
    table = {}
    for a, b in train:
        for comp in _components(a, bg=0, diag=False):
            k = prop(comp, a)
            outs = {int(b[i, j]) for i, j in comp}
            if len(outs) != 1:
                return None
            oc = outs.pop()
            if k in table and table[k] != oc:
                return None
            table[k] = oc
    def fn(g, table=table, prop=prop):
        out = g.copy()
        for comp in _components(g, bg=0, diag=False):
            k = prop(comp, g)
            if k not in table:
                return None
            for i, j in comp:
                out[i, j] = table[k]
        return out
    return fn


def fit_objprop_size(train):
    return _fit_objprop(train, lambda comp, g: len(comp))


def fit_objprop_sizerank(train):
    # key = rank of this comp's size among all comps in the grid (0 = smallest)
    def builder(train):
        if not all(a.shape == b.shape for a, b in train):
            return None
        if not all(np.array_equal(a != 0, b != 0) for a, b in train):
            return None
        table = {}
        for a, b in train:
            comps = _components(a, bg=0)
            sizes = sorted(set(len(c) for c in comps))
            for comp in comps:
                k = sizes.index(len(comp))
                outs = {int(b[i, j]) for i, j in comp}
                if len(outs) != 1:
                    return None
                oc = outs.pop()
                if k in table and table[k] != oc:
                    return None
                table[k] = oc
        def fn(g, table=table):
            out = g.copy()
            comps = _components(g, bg=0)
            sizes = sorted(set(len(c) for c in comps))
            for comp in comps:
                k = sizes.index(len(comp))
                if k not in table:
                    return None
                for i, j in comp:
                    out[i, j] = table[k]
            return out
        return fn
    return builder(train)


def fit_objprop_color(train):
    return _fit_objprop(train, lambda comp, g: int(g[comp[0][0], comp[0][1]]))


# --- concept: MIRROR-TILE (output = kh×kw grid of geometric transforms of input) ---
_GEOMS = {
    "id": lambda a: a, "fh": lambda a: a[:, ::-1], "fv": lambda a: a[::-1, :],
    "r180": lambda a: a[::-1, ::-1], "r90": lambda a: np.rot90(a, 1),
    "r270": lambda a: np.rot90(a, 3), "T": lambda a: a.T,
}


def fit_mirror_tile(train):
    # need a constant integer (kh,kw) ratio across train
    ratios = set()
    for a, b in train:
        ah, aw = a.shape; bh, bw = b.shape
        if ah == 0 or aw == 0 or bh % ah or bw % aw:
            return None
        ratios.add((bh // ah, bw // aw))
    if len(ratios) != 1:
        return None
    kh, kw = ratios.pop()
    if kh * kw < 2 or kh * kw > 9:
        return None
    # learn, per block (i,j), which geom maps input->that block, consistently across train
    table = {}
    for i in range(kh):
        for j in range(kw):
            cand = None
            for a, b in train:
                ah, aw = a.shape
                blk = b[i * ah:(i + 1) * ah, j * aw:(j + 1) * aw]
                ok = [nm for nm, f in _GEOMS.items()
                      if (g := f(a)).shape == blk.shape and np.array_equal(g, blk)]
                cand = set(ok) if cand is None else (cand & set(ok))
                if not cand:
                    return None
            table[(i, j)] = sorted(cand)[0]
    def fn(g, kh=kh, kw=kw, table=table):
        ah, aw = g.shape
        out = np.zeros((ah * kh, aw * kw), int)
        for i in range(kh):
            for j in range(kw):
                out[i * ah:(i + 1) * ah, j * aw:(j + 1) * aw] = _GEOMS[table[(i, j)]](g)
        return out
    return fn


# --- concept: FRACTAL SELF-TILE (block(i,j) = input if input[i,j] is fg else bg) ---
def fit_fractal(train):
    ratios = set()
    for a, b in train:
        ah, aw = a.shape; bh, bw = b.shape
        if bh != ah * ah or bw != aw * aw:
            return None
    bg = Counter()
    for a, _ in train:
        bg[_bg(a)] += 1
    bgv = bg.most_common(1)[0][0]
    for invert in (False, True):
        def fn(g, bgv=bgv, invert=invert):
            ah, aw = g.shape
            out = np.full((ah * ah, aw * aw), bgv, int)
            for i in range(ah):
                for j in range(aw):
                    on = (g[i, j] != bgv) ^ invert
                    if on:
                        out[i * ah:(i + 1) * ah, j * aw:(j + 1) * aw] = g
            return out
        if _verify(fn, train):
            return fn
    return None


# --- concept: PANEL SELECT (split into equal panels; pick by learned rule) ---
def _split_panels(g, oh, ow):
    H, W = g.shape
    if oh == 0 or ow == 0 or H % oh or W % ow:
        return None
    nr, nc = H // oh, W // ow
    if nr * nc < 2 or nr * nc > 12:
        return None
    panels = []
    for i in range(nr):
        for j in range(nc):
            panels.append(((i, j), g[i * oh:(i + 1) * oh, j * ow:(j + 1) * ow]))
    return panels


def _split_by_lines(g):
    """Split by full rows/cols of a single separator color into a grid of panels."""
    H, W = g.shape
    rowsep = [i for i in range(H) if len(set(g[i, :].tolist())) == 1]
    colsep = [j for j in range(W) if len(set(g[:, j].tolist())) == 1]
    # group separators; panels = segments between them
    def segs(seps, n):
        out = []
        prev = -1
        s = set(seps)
        for k in range(n + 1):
            if k == n or k in s:
                if prev + 1 <= k - 1:
                    out.append((prev + 1, k))
                prev = k
        return out
    rsegs = segs(set(rowsep), H)
    csegs = segs(set(colsep), W)
    if len(rsegs) * len(csegs) < 2 or len(rsegs) * len(csegs) > 12:
        return None
    # uniform panel size?
    rs = set(b - a for a, b in rsegs); cs = set(b - a for a, b in csegs)
    if len(rs) != 1 or len(cs) != 1:
        return None
    panels = []
    for ri, (r0, r1) in enumerate(rsegs):
        for ci, (c0, c1) in enumerate(csegs):
            panels.append(((ri, ci), g[r0:r1, c0:c1]))
    return panels


_PANEL_RULES = {
    "unique": lambda ps: _pick_unique(ps),
    "mostfg": lambda ps: max(ps, key=lambda p: int((p[1] != 0).sum()))[1],
    "leastfg": lambda ps: min(ps, key=lambda p: int((p[1] != 0).sum()))[1],
    "mostcolors": lambda ps: max(ps, key=lambda p: len(set(p[1].flatten().tolist())))[1],
    "first": lambda ps: ps[0][1],
    "last": lambda ps: ps[-1][1],
}


def _pick_unique(ps):
    keys = [p[1].tobytes() for p in ps]
    cnt = Counter(keys)
    uniq = [p for p, k in zip(ps, keys) if cnt[k] == 1]
    return uniq[0][1] if len(uniq) == 1 else None


def fit_panel_select(train):
    # output shape must be constant; panels split either equally or by separator lines
    out_shapes = set(b.shape for _, b in train)
    funcs = []
    if len(out_shapes) == 1:
        oh, ow = next(iter(out_shapes))
        funcs.append(lambda g, oh=oh, ow=ow: _split_panels(g, oh, ow))
    funcs.append(lambda g: _split_by_lines(g))
    for splitter in funcs:
        for rname, rule in _PANEL_RULES.items():
            def fn(g, splitter=splitter, rule=rule):
                ps = splitter(g)
                if not ps:
                    return None
                return rule(ps)
            if _verify(fn, train):
                return fn
    return None


# --- concept: CROP-TO-OBJECT (bbox of a selected object) ---
def _bbox(cells):
    rs = [c[0] for c in cells]; cs = [c[1] for c in cells]
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def fit_crop_object(train):
    selectors = {
        "largest": lambda comps, g: max(comps, key=len),
        "smallest": lambda comps, g: min(comps, key=len),
        "uniqcolor": lambda comps, g: _uniq_color_comp(comps, g),
    }
    for sname, sel in selectors.items():
        for diag in (False, True):
            def fn(g, sel=sel, diag=diag):
                comps = _components(g, bg=_bg(g), diag=diag)
                if not comps:
                    return None
                c = sel(comps, g)
                if c is None:
                    return None
                r0, c0, r1, c1 = _bbox(c)
                return g[r0:r1, c0:c1]
            if _verify(fn, train):
                return fn
    return None


def _uniq_color_comp(comps, g):
    # the object whose color appears on exactly one component
    col_of = [int(g[c[0][0], c[0][1]]) for c in comps]
    cnt = Counter(col_of)
    uniq = [c for c, col in zip(comps, col_of) if cnt[col] == 1]
    return uniq[0] if len(uniq) == 1 else None


# --- concept: SYMMETRIZE (repair occluded symmetry by overlaying mirror/rot copies) ---
def fit_symmetrize(train):
    if not all(a.shape == b.shape for a, b in train):
        return None
    # the 'hole' color = a color present in input, replaced in output. find candidates.
    hole_cands = set()
    for a, b in train:
        diff = a != b
        if diff.any():
            hole_cands |= set(np.unique(a[diff]).tolist())
    if not hole_cands:
        return None
    ops = [lambda g: g[:, ::-1], lambda g: g[::-1, :], lambda g: g[::-1, ::-1]]
    for hole in hole_cands:
        def fn(g, hole=hole, ops=ops):
            out = g.copy()
            mask = out == hole
            if not mask.any():
                return out
            for op in ops:
                t = op(out)
                if t.shape != out.shape:
                    continue
                fillable = mask & (t != hole)
                out[fillable] = t[fillable]
                mask = out == hole
            return out
        if _verify(fn, train):
            return fn
    return None


# --- concept: PERIODIC-TILE-FIT (output = a small tile repeated to output shape) ---
def fit_periodic_tile(train):
    # find smallest tile of input that, np.tile'd, equals output (input already periodic, denoise)
    if not all(b.size >= a.size for a, b in train):
        return None
    for a, b in train[:1]:
        ah, aw = a.shape; bh, bw = b.shape
        if bh % ah or bw % aw:
            return None
    def fn(g, train=train):
        # infer reps from train ratio (constant)
        a0, b0 = train[0]
        rh = b0.shape[0] // a0.shape[0]; rw = b0.shape[1] // a0.shape[1]
        return np.tile(g, (rh, rw))
    if _verify(fn, train):
        return fn
    return None


# --- concept: keep most/least common color object only (denoise) ---
def fit_majority_object(train):
    if not all(a.shape == b.shape for a, b in train):
        return None
    for mode in ("keepmost", "keepleast"):
        def fn(g, mode=mode):
            vals, cts = np.unique(g[g != 0], return_counts=True) if (g != 0).any() else (np.array([]), np.array([]))
            if len(vals) == 0:
                return g
            keep = vals[cts.argmax()] if mode == "keepmost" else vals[cts.argmin()]
            out = np.zeros_like(g)
            out[g == keep] = keep
            return out
        if _verify(fn, train):
            return fn
    return None


# ===========================================================================
# THE LINKER: recognize -> fit -> verify -> rank (MDL + in-session prior).
# Each concept has a base MDL cost (simpler/cheaper concepts first).
# ===========================================================================
_CONCEPTS = [
    # (name, builder, base_mdl_cost)
    ("colormap",        fit_colormap,        2),
    ("periodic_tile",   fit_periodic_tile,   3),
    ("fractal",         fit_fractal,         3),
    ("symmetrize",      fit_symmetrize,      4),
    ("crop_object",     fit_crop_object,     4),
    ("panel_select",    fit_panel_select,    4),
    ("mirror_tile",     fit_mirror_tile,     5),
    ("majority_object", fit_majority_object, 5),
    ("objprop_color",   fit_objprop_color,   5),
    ("objprop_sizerank",fit_objprop_sizerank,6),
    ("objprop_size",    fit_objprop_size,    6),
]


def _features(train):
    """Cheap features the linker uses to GATE which concepts even fire (saves time)."""
    same = all(a.shape == b.shape for a, b in train)
    bigger = all(b.size > a.size for a, b in train)
    smaller = all(b.size < a.size for a, b in train)
    maskpres = same and all(np.array_equal(a != 0, b != 0) for a, b in train)
    const_out = len(set(b.shape for _, b in train)) == 1
    int_ratio = True
    for a, b in train:
        if a.shape[0] == 0 or a.shape[1] == 0 or b.shape[0] % a.shape[0] or b.shape[1] % a.shape[1]:
            int_ratio = False
            break
    return dict(same=same, bigger=bigger, smaller=smaller, maskpres=maskpres,
                const_out=const_out, int_ratio=int_ratio)


# which concepts to even ATTEMPT given features (the recognition step) ------------
def _gate(name, F):
    if name == "colormap":         return F["same"]
    if name == "periodic_tile":    return F["bigger"] and F["int_ratio"]
    if name == "fractal":          return F["bigger"]
    if name == "symmetrize":       return F["same"]
    if name == "crop_object":      return F["smaller"]
    if name == "panel_select":     return F["smaller"] and F["const_out"]
    if name == "mirror_tile":      return F["bigger"] and F["int_ratio"]
    if name == "majority_object":  return F["same"]
    if name.startswith("objprop"): return F["maskpres"]
    return True


def _concept_attempts(train):
    """Run the linker: fit every gated concept, keep verified ones, rank MDL+experience."""
    F = _features(train)
    passers = []  # (rank_cost, name, fn)
    for name, builder, cost in _CONCEPTS:
        if not _gate(name, F):
            continue
        try:
            fn = builder(train)
        except Exception:
            fn = None
        if fn is None:
            continue
        if not _verify(fn, train):
            continue
        # in-session experience: concepts that have paid off before get a rank discount
        prior = _LIB.get(name, 0)
        rank = cost - 0.5 * min(prior, 6)
        passers.append((rank, name, fn))
    passers.sort(key=lambda x: x[0])
    return passers


# ===========================================================================
# SEED DSL SEARCH (breadth backstop) — faithful to seed_solver.
# ===========================================================================
def _instantiate(pal):
    colors = [c for c in pal if c != 0]
    insts = []
    for name, (_fn, nc) in dsl.OPS.items():
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


def _gdist(a, b):
    if a is None:
        return 3.0
    if a.shape != b.shape:
        return 1.0 + abs(a.size - b.size) / max(a.size, b.size, 1)
    return float((a != b).mean())


def _apply_all(outs, name, args):
    res = []
    for g in outs:
        try:
            res.append(dsl.OPS[name][0](g, *args))
        except Exception:
            return None
    return res


def _exact_all(outs, tgt):
    return all(o is not None and o.shape == t.shape and np.array_equal(o, t) for o, t in zip(outs, tgt))


def _search_collect(train, B, W=25, max_len=3, K=4):
    insts = _instantiate(dsl.palette(train))
    ins = [gi for gi, _ in train]; tgt = [go for _, go in train]
    start = sum(_gdist(a, b) for a, b in zip(ins, tgt)) / len(ins)
    heap = [(start, 0, [], ins)]; ctr = 1; nexec = 0; found = []
    while heap and nexec < B:
        _s, _c, prog, outs = heapq.heappop(heap)
        if len(prog) >= max_len:
            continue
        kids = []
        for inst in insts:
            outs2 = _apply_all(outs, inst[0], inst[1]); nexec += 1
            if outs2 is None:
                if nexec >= B:
                    break
                continue
            if _exact_all(outs2, tgt):
                found.append(prog + [inst])
                if len(found) >= K:
                    return found, nexec
            else:
                sc = sum(_gdist(a, b) for a, b in zip(outs2, tgt)) / len(outs2)
                kids.append((sc, ctr, prog + [inst], outs2)); ctr += 1
            if nexec >= B:
                break
        for k in heapq.nsmallest(W, kids):
            heapq.heappush(heap, k)
    return found, nexec


def _plen(p):
    return sum(1 + len(a) for _, a in p)


# ===========================================================================
# solve()
# ===========================================================================
def solve(train, test_inputs, budget):
    _SOLVE_CALLS[0] += 1
    # 1) THE LINKER: fitted-concept proposals, ranked MDL + in-session experience.
    passers = _concept_attempts(train)

    # 2) breadth backstop: seed DSL best-first search (cheap programs the concepts miss).
    #    Give it the remaining budget; concept fitting is ~free, so almost all budget here.
    progs, _ = _search_collect(train, max(int(budget), 200), K=4)
    progs = sorted(progs, key=_plen)
    search_fns = []
    for p in progs:
        search_fns.append((_plen(p) + 6, "search:" + "+".join(n for n, _ in p),
                           (lambda g, p=p: dsl.apply_prog(g, p))))

    # 3) LINK + ORDER all train-consistent proposals (concepts first by MDL, then search).
    ranked = [(r, nm, fn) for (r, nm, fn) in passers] + search_fns
    ranked.sort(key=lambda x: x[0])

    # in-session bookkeeping: a verified concept TYPE is "experience" — remember it.
    # (we only learn the concept NAME, never any grid.) Counts toward future ranking.
    for _r, nm, _fn in passers:
        base = nm.split(":")[0]
        _LIB[base] += 1

    # 4) emit ≤2 attempts per test input, best-first. The "coverage move": the 2 attempts
    #    come from the top-ranked DISTINCT proposals (so attempt-2 explores a different
    #    concept/link, not a near-duplicate of attempt-1).
    attempts = []
    for gi in test_inputs:
        cand = []
        seen = []
        for _r, _nm, fn in ranked:
            try:
                o = fn(gi)
            except Exception:
                o = None
            if o is None or not isinstance(o, np.ndarray) or o.ndim != 2 or o.size == 0:
                continue
            key = o.tobytes()
            if key in seen:
                continue
            seen.append(key)
            cand.append(o)
            if len(cand) >= 2:
                break
        attempts.append(cand)
    return attempts
