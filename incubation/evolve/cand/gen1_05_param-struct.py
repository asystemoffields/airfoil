#!/usr/bin/env python3
"""Gen-1 mutation #5 — PARAMETRIC / STRUCTURAL concepts fitted from train pairs.

Thesis served: creativity = a CONCEPT STORE (here, a family of parametric/structural concept-fitters)
recombined by a LINKER (try fitters; for those that fit ALL train pairs, emit a closed-form rule),
filtered by EXACT verify. Each concept's *parameters* are FITTED from the current task's train pairs:
  - color_perm          : a bijective color relabeling fitted cell-by-cell across train
  - panel_logic         : split by gridline/separator color OR equal halves -> inter-panel AND/OR/XOR/diff/
                          overlay, with a fitted output color map
  - symmetric_tiling    : output = a kRows x kCols mosaic; each block is one of {id, mirror_h, mirror_v,
                          rot180, transpose, ...}; the per-block transform CHOICE is fitted from train
  - fractal_self_tiling : output = HxW blocks; block(i,j) = input where input[i,j] is a 'lit' cell else bg
  - periodic_repair     : detect row/col/diagonal period, fill the 'occluded' color by periodic continuation
  - symmetry_repair     : restore mirror/rotational symmetry by filling occluded cells from their image
  - scale_by_ratio      : nearest-neighbour upscale by the observed integer (rh,rw) ratio

These make a class of bucket-A tasks EXPRESSIBLE that the 32-op length<=3 DSL cannot reach. A LINKER step
also composes a fitted concept AFTER a cheap geometric pre-op (functional repurposing of DSL ops).

Experience channels:
  - PER-TASK INDUCTION: every concept fits its params from the CURRENT task's train pairs only.
  - IN-SESSION LIBRARY (module-level _LIB): when a concept verifies on a task, we remember the
    (concept-name, fitted-spec) so later tasks try previously-successful concept *orderings* first
    (an MDL/experience prior over which concepts to attempt). Only verified rules are stored.

Falls back to the seed best-first DSL search so we never regress below the seed.

INTEGRITY: solve() reads ONLY the train pairs handed in + module state from prior solve() calls +
self-generated synthetic data (none needed here). It never reads ARC task files or test outputs.
No network / no LLM. Run with /data/llm/.venv/bin/python."""
import sys, heapq
import numpy as np

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl

META = {"name": "param_struct_v1",
        "desc": "parametric/structural concept-fitters (color-perm, panel-logic, symmetric+fractal tiling, "
                "periodic/symmetry repair, scale-by-ratio) fitted from train; linker composes w/ geom pre-ops; "
                "in-session concept-order library; seed DSL fallback"}

# ---------------------------------------------------------------------------
# IN-SESSION EXPERIENCE LIBRARY (module-level; persists across solve() calls in a run).
# Counts how often each concept name produced a VERIFIED rule -> an experience prior on attempt order.
# ---------------------------------------------------------------------------
_LIB = {"concept_hits": {}}  # name -> int


def _bump(name):
    _LIB["concept_hits"][name] = _LIB["concept_hits"].get(name, 0) + 1


# ---------------------------------------------------------------------------
# small grid helpers
# ---------------------------------------------------------------------------
def _bg_color(grids):
    """Most-common color across the given grids (the presumed background)."""
    from collections import Counter
    c = Counter()
    for g in grids:
        v, ct = np.unique(g, return_counts=True)
        for vi, ci in zip(v, ct):
            c[int(vi)] += int(ci)
    return c.most_common(1)[0][0] if c else 0


def _eq(a, b):
    return a is not None and a.shape == b.shape and np.array_equal(a, b)


def _verify(fn, train):
    """Return True iff fn reproduces EVERY train output exactly (and runs without error)."""
    for gi, go in train:
        try:
            o = fn(gi)
        except Exception:
            return False
        if not _eq(o, go):
            return False
    return True


# ===========================================================================
# CONCEPT 1 — color permutation (bijection fitted from train cells)
# ===========================================================================
def fit_color_perm(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    mp = {}
    for gi, go in train:
        a = gi.reshape(-1); b = go.reshape(-1)
        for x, y in zip(a.tolist(), b.tolist()):
            if x in mp and mp[x] != y:
                return None
            mp[x] = y
    if all(k == v for k, v in mp.items()):
        return None  # identity is not a useful concept

    def fn(g):
        out = g.copy()
        for k, v in mp.items():
            if k != v:
                out[g == k] = v
        return out
    return fn


# ===========================================================================
# CONCEPT 2 — panel split + inter-panel logic
# ===========================================================================
def _separators(g):
    """Yield (axis, color, sep_indices) for full-span single-color lines that split the grid."""
    out = []
    h, w = g.shape
    for ax in (0, 1):
        n = g.shape[ax]
        for c in np.unique(g):
            line = np.all(g == c, axis=1 - ax)
            idx = np.where(line)[0]
            if 0 < len(idx) < n:
                out.append((ax, int(c), idx))
    return out


def _split_panels(g):
    """Return list of candidate (panels, axis) splittings: by each separator color, and equal halves."""
    cands = []
    h, w = g.shape
    # separator-color splits
    for ax, c, idx in _separators(g):
        segs = []
        prev = 0
        bounds = list(idx) + [g.shape[ax]]
        ok = True
        for i in bounds:
            if i > prev:
                seg = g[prev:i, :] if ax == 0 else g[:, prev:i]
                segs.append(seg)
            prev = i + 1
        shapes = set(s.shape for s in segs)
        if len(segs) >= 2 and len(shapes) == 1:
            cands.append((segs, ax, c))
    # equal halves (no separator)
    if w % 2 == 0:
        cands.append(([g[:, :w // 2], g[:, w // 2:]], 1, None))
    if h % 2 == 0:
        cands.append(([g[:h // 2, :], g[h // 2:, :]], 0, None))
    # equal thirds
    if w % 3 == 0:
        cands.append(([g[:, :w // 3], g[:, w // 3:2 * w // 3], g[:, 2 * w // 3:]], 1, None))
    if h % 3 == 0:
        cands.append(([g[:h // 3, :], g[h // 3:2 * h // 3, :], g[2 * h // 3:, :]], 0, None))
    return cands


def _combine(panels, mode, bg):
    """Combine equal-shape boolean masks of panels under a logical mode; returns boolean mask."""
    masks = [(p != bg) for p in panels]
    if mode == "and":
        m = masks[0].copy()
        for x in masks[1:]:
            m &= x
    elif mode == "or":
        m = masks[0].copy()
        for x in masks[1:]:
            m |= x
    elif mode == "xor":
        m = masks[0].copy()
        for x in masks[1:]:
            m ^= x
    elif mode == "diff":  # in first but not in any other
        m = masks[0].copy()
        for x in masks[1:]:
            m &= ~x
    elif mode == "nand":
        m = masks[0].copy()
        for x in masks[1:]:
            m &= x
        m = ~m
    elif mode == "nor":
        m = masks[0].copy()
        for x in masks[1:]:
            m |= x
        m = ~m
    else:
        return None
    return m


def fit_panel_logic(train):
    # try every (split-strategy index, logic-mode); fit a single output color for the 'true' mask.
    if not train:
        return None
    bg = _bg_color([gi for gi, _ in train])
    # collect split strategies that are CONSISTENT in count/shape across all train inputs & match out shape
    strategies = []
    g0 = train[0][0]
    for si, (panels0, ax0, sep0) in enumerate(_split_panels(g0)):
        pshape = panels0[0].shape
        npan = len(panels0)
        if pshape != train[0][1].shape:
            continue
        # build a key describing the strategy and re-derive on each train input
        ok = True
        per_train = []
        for gi, go in train:
            found = None
            for panels, ax, sep in _split_panels(gi):
                if len(panels) == npan and panels[0].shape == go.shape and ax == ax0 and sep == sep0:
                    found = panels
                    break
            if found is None:
                ok = False
                break
            per_train.append(found)
        if ok:
            strategies.append(per_train)
    for per_train in strategies:
        for mode in ("and", "or", "xor", "diff", "nand", "nor"):
            # fit output color from train true-cells, and a background output color
            on_color = None
            off_color = None
            good = True
            fitted = []
            for (gi, go), panels in zip(train, per_train):
                m = _combine(panels, mode, bg)
                if m is None or m.shape != go.shape:
                    good = False
                    break
                on_vals = set(np.unique(go[m]).tolist()) if m.any() else set()
                off_vals = set(np.unique(go[~m]).tolist()) if (~m).any() else set()
                if len(on_vals) > 1 or len(off_vals) > 1:
                    good = False
                    break
                oc = next(iter(on_vals)) if on_vals else None
                fc = next(iter(off_vals)) if off_vals else None
                if oc is not None:
                    if on_color is None:
                        on_color = oc
                    elif on_color != oc:
                        good = False
                        break
                if fc is not None:
                    if off_color is None:
                        off_color = fc
                    elif off_color != fc:
                        good = False
                        break
            if not good or on_color is None:
                continue
            if off_color is None:
                off_color = 0
            oc_, fc_, mode_, npan_, ax_, sep_ = on_color, off_color, mode, len(per_train[0]), None, None
            # capture strategy key from first train input
            for panels, ax, sep in _split_panels(train[0][0]):
                if len(panels) == npan_ and panels[0].shape == train[0][1].shape:
                    ax_, sep_ = ax, sep
                    break

            def make(oc=oc_, fc=fc_, mode=mode_, npan=npan_, ax=ax_, sep=sep_, bg=bg):
                def fn(g):
                    chosen = None
                    for panels, a, s in _split_panels(g):
                        if len(panels) == npan and a == ax and s == sep:
                            chosen = panels
                            break
                    if chosen is None:
                        return None
                    m = _combine(chosen, mode, bg)
                    if m is None:
                        return None
                    out = np.full(m.shape, fc, int)
                    out[m] = oc
                    return out
                return fn
            fn = make()
            if _verify(fn, train):
                return fn
    return None


# ===========================================================================
# CONCEPT 3 — symmetric tiling (mosaic of fitted per-block transforms)
# ===========================================================================
_TRANSFORMS = {
    "id":   lambda g: g,
    "mh":   lambda g: g[:, ::-1],
    "mv":   lambda g: g[::-1, :],
    "r180": lambda g: np.rot90(g, 2),
    "r90":  lambda g: np.rot90(g, 1),
    "r270": lambda g: np.rot90(g, 3),
    "tp":   lambda g: g.T,
    " tp2": lambda g: g[::-1, ::-1].T,
}


def fit_symmetric_tiling(train):
    gi0, go0 = train[0]
    hi, wi = gi0.shape
    ho, wo = go0.shape
    if hi == 0 or wi == 0 or ho % hi or wo % wi:
        return None
    kr, kc = ho // hi, wo // wi
    if kr * kc < 2 or kr > 4 or kc > 4:
        return None
    # for square blocks rotations/transpose preserve shape; otherwise restrict to shape-preserving transforms
    square = (hi == wi)
    names = list(_TRANSFORMS.keys()) if square else ["id", "mh", "mv", "r180"]
    # consistency: every train pair must share kr,kc
    for gi, go in train:
        if gi.shape[0] == 0 or gi.shape[1] == 0:
            return None
        if go.shape[0] != kr * gi.shape[0] or go.shape[1] != kc * gi.shape[1]:
            return None
    # fit a transform name for each block position independently across all train pairs
    grid_choice = [[None] * kc for _ in range(kr)]
    for bi in range(kr):
        for bj in range(kc):
            chosen = None
            for nm in names:
                ok = True
                for gi, go in train:
                    h, w = gi.shape
                    block = go[bi * h:(bi + 1) * h, bj * w:(bj + 1) * w]
                    t = _TRANSFORMS[nm](gi)
                    if t.shape != block.shape or not np.array_equal(t, block):
                        ok = False
                        break
                if ok:
                    chosen = nm
                    break
            if chosen is None:
                return None
            grid_choice[bi][bj] = chosen

    def fn(g, gc=grid_choice, kr=kr, kc=kc):
        rows = []
        for bi in range(kr):
            row = [_TRANSFORMS[gc[bi][bj]](g) for bj in range(kc)]
            rows.append(np.concatenate(row, axis=1))
        return np.concatenate(rows, axis=0)
    return fn


# ===========================================================================
# CONCEPT 4 — fractal self-tiling (007bbfb7 family)
# ===========================================================================
def fit_fractal(train):
    gi0, go0 = train[0]
    hi, wi = gi0.shape
    ho, wo = go0.shape
    if hi == 0 or wi == 0 or ho != hi * hi or wo != wi * wi:
        return None
    bg = _bg_color([gi for gi, _ in train])
    # two polarities: place block where cell is lit (!=bg), or where cell is bg
    for invert in (False, True):
        def fn(g, bg=bg, invert=invert):
            h, w = g.shape
            out = np.full((h * h, w * w), bg, int)
            for i in range(h):
                for j in range(w):
                    lit = (g[i, j] != bg)
                    if invert:
                        lit = not lit
                    if lit:
                        out[i * h:(i + 1) * h, j * w:(j + 1) * w] = g
            return out
        if _verify(fn, train):
            return fn
    return None


# ===========================================================================
# CONCEPT 5 — periodicity repair (fill occluded color by detected period)
# ===========================================================================
def _detect_occluder(train):
    """The color present in inputs but never (or rarely) in same-position outputs that gets replaced."""
    # heuristic: the color whose cells in input always differ from output (the 'hole' marker)
    cand = None
    for gi, go in train:
        if gi.shape != go.shape:
            return None
        diff = (gi != go)
        if not diff.any():
            continue
        holes = set(np.unique(gi[diff]).tolist())
        if len(holes) != 1:
            return None
        h = next(iter(holes))
        if cand is None:
            cand = h
        elif cand != h:
            return None
    return cand


def _fill_periodic(g, hole):
    """Fill cells equal to `hole` using the smallest consistent (pr,pc) period of the non-hole cells."""
    h, w = g.shape
    known = (g != hole)
    if known.all():
        return g.copy()

    def consistent(pr, pc):
        # all known cells must match the cell `period` away if both known
        for i in range(h):
            for j in range(w):
                if not known[i, j]:
                    continue
                ii, jj = i % pr if pr else i, j % pc if pc else j
                # compare to representative within first tile via stepping
        return True

    # find smallest row period and col period over known cells
    def row_period():
        for p in range(1, h):
            ok = True
            for i in range(h):
                for j in range(w):
                    if known[i, j] and known[(i + p) % h if False else i, j]:
                        pass
            # proper check below
            ok = True
            for i in range(h - p):
                for j in range(w):
                    if known[i, j] and known[i + p, j] and g[i, j] != g[i + p, j]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return p
        return h

    def col_period():
        for p in range(1, w):
            ok = True
            for j in range(w - p):
                for i in range(h):
                    if known[i, j] and known[i, j + p] and g[i, j] != g[i, j + p]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return p
        return w

    pr = row_period()
    pc = col_period()
    out = g.copy()
    for i in range(h):
        for j in range(w):
            if known[i, j]:
                continue
            # search positions sharing the same (i mod pr, j mod pc) class for a known value
            val = None
            for ii in range(i % pr, h, pr):
                for jj in range(j % pc, w, pc):
                    if known[ii, jj]:
                        val = g[ii, jj]
                        break
                if val is not None:
                    break
            if val is None:
                # try diagonal period as a fallback
                for d in range(1, max(h, w)):
                    for (di, dj) in ((d, d), (-d, -d), (d, -d), (-d, d)):
                        ii, jj = i + di, j + dj
                        if 0 <= ii < h and 0 <= jj < w and known[ii, jj]:
                            val = g[ii, jj]
                            break
                    if val is not None:
                        break
            if val is not None:
                out[i, j] = val
    return out


def _fill_diagonal(g, hole):
    """Fill using diagonal/anti-diagonal periodicity (value constant along i-j or i+j modulo period)."""
    h, w = g.shape
    known = (g != hole)
    out = g.copy()
    for axis in ("main", "anti"):
        key = {}
        for i in range(h):
            for j in range(w):
                if known[i, j]:
                    k = (i - j) if axis == "main" else (i + j)
                    key.setdefault(k % max(h, w), set()).add(int(g[i, j]))
        # only use if each diagonal class is single-valued
        if all(len(v) <= 1 for v in key.values()):
            tmp = g.copy()
            for i in range(h):
                for j in range(w):
                    if not known[i, j]:
                        k = (i - j) if axis == "main" else (i + j)
                        vs = key.get(k % max(h, w))
                        if vs:
                            tmp[i, j] = next(iter(vs))
            if (tmp != hole).all():
                out = tmp
                return out
    return out


def fit_periodic_repair(train):
    hole = _detect_occluder(train)
    if hole is None:
        return None
    # try: (a) crop to occluder bbox, (b) full-grid fill
    bg_fill_modes = []

    def fn_full(g, hole=hole):
        o = _fill_periodic(g, hole)
        if (o == hole).any():
            o2 = _fill_diagonal(g, hole)
            if not (o2 == hole).any():
                return o2
        return o
    if _verify(fn_full, train):
        return fn_full

    # output may be ONLY the filled region (crop to occluder bbox)
    def fn_crop(g, hole=hole):
        nz = np.argwhere(g == hole)
        if nz.size == 0:
            return None
        (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
        full = _fill_periodic(g, hole)
        return full[r0:r1, c0:c1]
    if _verify(fn_crop, train):
        return fn_crop
    return None


# ===========================================================================
# CONCEPT 6 — symmetry repair (restore mirror/rotational symmetry; fill occluded cells)
# ===========================================================================
def fit_symmetry_repair(train):
    hole = _detect_occluder(train)
    if hole is None:
        return None
    syms = [
        lambda g: g[:, ::-1],
        lambda g: g[::-1, :],
        lambda g: g[::-1, ::-1],
        lambda g: g.T if g.shape[0] == g.shape[1] else None,
        lambda g: np.rot90(g, 1) if g.shape[0] == g.shape[1] else None,
    ]

    def repair(g, hole=hole):
        h, w = g.shape
        out = g.copy()
        for _ in range(6):
            changed = False
            for s in syms:
                m = s(out)
                if m is None or m.shape != out.shape:
                    continue
                fillable = (out == hole) & (m != hole)
                if fillable.any():
                    out[fillable] = m[fillable]
                    changed = True
            if not changed:
                break
        return out

    def fn_full(g):
        o = repair(g)
        return o
    if _verify(fn_full, train):
        return fn_full

    def fn_crop(g, hole=hole):
        nz = np.argwhere(g == hole)
        if nz.size == 0:
            return None
        (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
        return repair(g)[r0:r1, c0:c1]
    if _verify(fn_crop, train):
        return fn_crop
    return None


# ===========================================================================
# CONCEPT 7 — scale by observed integer ratio (nearest-neighbour upscale)
# ===========================================================================
def fit_scale_ratio(train):
    gi0, go0 = train[0]
    hi, wi = gi0.shape
    ho, wo = go0.shape
    if hi == 0 or wi == 0 or ho % hi or wo % wi:
        return None
    rh, rw = ho // hi, wo // wi
    if rh == 1 and rw == 1:
        return None
    if max(rh, rw) > 10:
        return None

    def fn(g, rh=rh, rw=rw):
        return np.kron(g, np.ones((rh, rw), int))
    return fn


# ===========================================================================
# CONCEPT 8 — periodic TILING of input to a larger output (repeat input as wallpaper)
# ===========================================================================
def fit_periodic_tiling(train):
    gi0, go0 = train[0]
    hi, wi = gi0.shape
    ho, wo = go0.shape
    if hi == 0 or wi == 0 or ho % hi or wo % wi:
        return None
    rh, rw = ho // hi, wo // wi
    if rh * rw < 2:
        return None

    def fn(g, rh=rh, rw=rw):
        return np.tile(g, (rh, rw))
    return fn


# ---------------------------------------------------------------------------
# CONCEPT REGISTRY (the store) — name -> fitter(train) -> fn or None
# ---------------------------------------------------------------------------
CONCEPTS = [
    ("color_perm", fit_color_perm),
    ("scale_ratio", fit_scale_ratio),
    ("periodic_tiling", fit_periodic_tiling),
    ("symmetric_tiling", fit_symmetric_tiling),
    ("fractal", fit_fractal),
    ("panel_logic", fit_panel_logic),
    ("periodic_repair", fit_periodic_repair),
    ("symmetry_repair", fit_symmetry_repair),
]

# cheap geometric pre-ops for the LINKER (functional repurposing of DSL geometry before a fitted concept)
_PRE = [
    ("none", lambda g: g),
    ("crop_content", dsl.crop_content),
    ("reflect_h", dsl.reflect_h),
    ("reflect_v", dsl.reflect_v),
    ("rot90", dsl.rot90),
    ("transpose", dsl.transpose),
]


def _concept_order():
    """Experience prior: concepts that have verified before in this run go first (MDL stays a tiebreak)."""
    hits = _LIB["concept_hits"]
    return sorted(CONCEPTS, key=lambda nc: -hits.get(nc[0], 0))


def _try_concepts(train):
    """Return a list of verified rule-fns (best-first by experience prior then registry order)."""
    rules = []
    # 1) direct fit of each concept on the raw train pairs
    for name, fitter in _concept_order():
        try:
            fn = fitter(train)
        except Exception:
            fn = None
        if fn is not None and _verify(fn, train):
            rules.append((name, fn))
    # 2) LINKER: compose a geometric pre-op then a fitted concept (skip 'none' = already done)
    if not rules:
        for pname, pre in _PRE[1:]:
            try:
                pre_train = [(pre(gi), go) for gi, go in train]
            except Exception:
                continue
            for name, fitter in _concept_order():
                try:
                    fn = fitter(pre_train)
                except Exception:
                    fn = None
                if fn is None:
                    continue

                def composed(g, pre=pre, fn=fn):
                    return fn(pre(g))
                if _verify(composed, train):
                    rules.append((pname + "+" + name, composed))
                    break
            if rules:
                break
    return rules


# ---------------------------------------------------------------------------
# SEED DSL FALLBACK (verbatim behaviour of the gen-0 seed; never regress below it)
# ---------------------------------------------------------------------------
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


def _exact(outs, tgt):
    return all(o is not None and o.shape == t.shape and np.array_equal(o, t) for o, t in zip(outs, tgt))


def _search_collect(train, B, W=25, max_len=3, K=4):
    insts = _instantiate(dsl.palette(train))
    ins = [gi for gi, _ in train]; tgt = [go for _, go in train]
    start = sum(_gdist(a, b) for a, b in zip(ins, tgt)) / len(ins)
    heap = [(start, 0, [], ins)]; ctr = 1; nexec = 0; found = []
    while heap and nexec < B:
        _score, _c, prog, outs = heapq.heappop(heap)
        if len(prog) >= max_len:
            continue
        kids = []
        for inst in insts:
            outs2 = _apply_all(outs, inst[0], inst[1]); nexec += 1
            if outs2 is None:
                if nexec >= B:
                    break
                continue
            if _exact(outs2, tgt):
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


def _seed_attempts(train, test_inputs, budget):
    progs, _ = _search_collect(train, budget, K=4)
    progs = sorted(progs, key=_plen)[:2]
    attempts = []
    for gi in test_inputs:
        cand = []
        for p in progs:
            o = dsl.apply_prog(gi, p)
            if o is not None:
                cand.append(o)
        attempts.append(cand)
    return attempts


# ---------------------------------------------------------------------------
# PUBLIC ENTRYPOINT
# ---------------------------------------------------------------------------
def solve(train, test_inputs, budget):
    # 1) parametric/structural concept store + linker (cheap, exact-verified)
    rules = _try_concepts(train)
    if rules:
        # record experience for verified concept(s)
        for name, _ in rules:
            _bump(name.split("+")[-1])
        attempts = []
        for gi in test_inputs:
            cand = []
            for _name, fn in rules:
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
        # if every test produced at least one candidate, trust the verified rule(s)
        if all(len(a) >= 1 for a in attempts):
            return attempts
        # otherwise fall through and let the seed try to fill gaps

    # 2) seed DSL fallback (never regress below the gen-0 seed)
    return _seed_attempts(train, test_inputs, budget)
